"""
Comprehensive Semantic Search MCP Server
─────────────────────────────────────────
A production-ready MCP (Model Context Protocol) server with:
- Natural language → SQL conversion (handles ANY query type)
- Schema-aware intelligent querying
- Semantic similarity search via pgvector
- Advanced filtering, aggregation, and analysis
- Flexible data retrieval patterns
- Comprehensive error handling

Transport: stdio (reads JSON-RPC from stdin, writes to stdout)
"""

import sys
import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

class SafeStdout:
    """Redirects stray prints to stderr so stdio MCP framing is not corrupted."""
    def __init__(self, original_stdout, stderr):
        self._original = original_stdout
        self._stderr = stderr

    def write(self, data):
        return self._stderr.write(data)

    def flush(self):
        return self._stderr.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


# Redirect stdout to stderr early
_original_stdout = sys.stdout
sys.stdout = sys.stderr

from mcp.server.fastmcp import FastMCP
from embedding_manager import EmbeddingManager

mcp = FastMCP("semantic-search")

_manager = None
_prisma_mcp = None


def _get_manager() -> EmbeddingManager:
    """Lazily initialize the EmbeddingManager."""
    global _manager
    if _manager is None:
        print("Initializing EmbeddingManager and loading embedding model...", file=sys.stderr)
        _manager = EmbeddingManager()
        _manager.initialize_model()
        print("EmbeddingManager ready (Prisma Postgres pgvector).", file=sys.stderr)
    return _manager


def find_joins(tables: List[str], schema_meta: Dict[str, Any]) -> List[str]:
    """Find a list of LEFT JOIN clauses that connect all the target tables using foreign key relationships."""
    if len(tables) <= 1:
        return []
        
    adj = {}
    for tbl_name, tbl_info in schema_meta.items():
        if tbl_name not in adj:
            adj[tbl_name] = []
        for fk in tbl_info.get("foreign_keys", []):
            to_tbl = fk.get("to_table")
            our_col = fk.get("column")
            to_col = fk.get("to_column")
            if to_tbl and our_col and to_col:
                adj[tbl_name].append((to_tbl, our_col, to_col))
                if to_tbl not in adj:
                    adj[to_tbl] = []
                adj[to_tbl].append((tbl_name, to_col, our_col))
                
    start = tables[0]
    joins_needed = []
    visited_edges = set()
    visited_nodes = {start}
    
    for target in tables[1:]:
        queue = []
        for node in visited_nodes:
            queue.append((node, []))
            
        path = None
        bfs_visited = set(visited_nodes)
        while queue:
            curr, curr_path = queue.pop(0)
            if curr == target:
                path = curr_path
                break
            for neighbor, col1, col2 in adj.get(curr, []):
                if neighbor not in bfs_visited:
                    bfs_visited.add(neighbor)
                    queue.append((neighbor, curr_path + [(curr, col1, neighbor, col2)]))
                    
        if path:
            for node1, col1, node2, col2 in path:
                edge_key = tuple(sorted([f"{node1}.{col1}", f"{node2}.{col2}"]))
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    joins_needed.append((node1, col1, node2, col2))
                    visited_nodes.add(node1)
                    visited_nodes.add(node2)
                    
    from_tables = {tables[0]}
    join_clauses = []
    remaining_joins = list(joins_needed)
    
    progress = True
    while remaining_joins and progress:
        progress = False
        for i, (node1, col1, node2, col2) in enumerate(remaining_joins):
            if node1 in from_tables and node2 not in from_tables:
                join_clauses.append(f'LEFT JOIN "{node2}" ON "{node1}"."{col1}" = "{node2}"."{col2}"')
                from_tables.add(node2)
                remaining_joins.pop(i)
                progress = True
                break
            elif node2 in from_tables and node1 not in from_tables:
                join_clauses.append(f'LEFT JOIN "{node1}" ON "{node2}"."{col2}" = "{node1}"."{col1}"')
                from_tables.add(node1)
                remaining_joins.pop(i)
                progress = True
                break
            elif node1 in from_tables and node2 in from_tables:
                remaining_joins.pop(i)
                progress = True
                break
                
    return join_clauses


def get_table_module(table_name: str) -> str:
    """Categorize tables into logical modules for better organization."""
    name = table_name.lower()
    if any(k in name for k in ('lead', 'opportunity', 'contact', 'customer', 'deal', 'funnel', 'account', 'interaction', 'sale')):
        return "CRM"
    if any(k in name for k in ('user', 'shift', 'attendance', 'leave', 'payroll', 'salary', 'department', 'position', 'employee')):
        return "HR & Attendance"
    if any(k in name for k in ('project', 'task', 'subtask', 'todo', 'comment')):
        return "Project Management"
    if any(k in name for k in ('ticket', 'case', 'support', 'kb_', 'knowledge', 'bizdesk')):
        return "BizDesk & Support"
    if any(k in name for k in ('goal', 'okr', 'key_result', 'milestone')):
        return "Goals & OKRs"
    if any(k in name for k in ('inventory', 'product', 'warehouse', 'supplier', 'purchase', 'stock', 'invoice', 'payment', 'quotation', 'billing', 'order')):
        return "ERP & Inventory"
    if any(k in name for k in ('document', 'attachment', 'embedding', 'log', 'audit', 'setting', 'config', 'metadata')):
        return "System & Documents"
    return "General / Other"


_PROJECTS_CACHE = None
_USERS_CACHE = None

class QueryInterpreter:
    """Converts natural language queries to SQL dynamically using database schema metadata."""
    
    AGGREGATE_FUNCTIONS = {
        'count': 'COUNT(*)',
        'total': 'SUM',
        'sum': 'SUM',
        'average': 'AVG',
        'avg': 'AVG',
        'max': 'MAX',
        'maximum': 'MAX',
        'min': 'MIN',
        'minimum': 'MIN',
        'distinct': 'COUNT(DISTINCT',
    }
    
    def __init__(self):
        self.query = ""
        self.table = None
        self.matched_tables = []
        self.columns = []
        self.filters = []
        self.aggregates = []
        self.group_by = None
        self.having = None
        self.order_by = None
        self.limit = 100
        self.joins = []
        
        # Load schema metadata dynamically
        self.manager = _get_manager()
        self.schema_meta = self.manager.fetch_all_tables_meta()
        
        # Build keyword-to-table dictionary
        self.table_aliases = {}
        for tbl_name in self.schema_meta.keys():
            aliases = self._generate_aliases(tbl_name)
            for alias in aliases:
                if alias not in self.table_aliases:
                    self.table_aliases[alias] = tbl_name
                    
    def _generate_aliases(self, table_name: str) -> List[str]:
        aliases = {table_name, table_name.lower()}
        
        # Replace underscores with spaces
        if "_" in table_name:
            aliases.add(table_name.replace("_", " "))
            aliases.add(table_name.replace("_", ""))
            
        # Try simple singular/plural logic
        variants = list(aliases)
        for v in variants:
            if v.endswith("ies"):
                aliases.add(v[:-3] + "y")
            elif v.endswith("s") and not v.endswith("ss"):
                aliases.add(v[:-1])
            else:
                aliases.add(v + "s")
                
        # Handle special common abbreviations
        if table_name == "company_users":
            aliases.update(["employee", "employees", "user", "users", "staff", "member", "members"])
        elif table_name == "companies":
            aliases.update(["company", "organization", "organizations", "firm", "firms", "business", "businesses"])
        elif table_name == "task":
            aliases.update(["task", "tasks", "todo", "todos"])
        elif table_name == "leads":
            aliases.update(["lead", "leads", "prospect", "prospects"])
        elif table_name == "task_updates":
            aliases.update(["task progress updates", "task progress update", "task updates", "task update", "progress updates", "progress update", "update", "updates"])
        elif table_name == "subtask_updates":
            aliases.update(["subtask progress updates", "subtask progress update", "subtask updates", "subtask update"])
        elif table_name == "attendances":
            aliases.update(["attendance", "attendances", "checkin", "checkins", "clockin", "clockins", "clock-in", "clock-ins", "punch", "punches"])
        elif table_name == "leaves":
            aliases.update(["leave", "leaves", "vacation", "vacations", "timeoff", "time off", "time-off", "holiday", "holidays"])
        elif table_name == "shifts":
            aliases.update(["shift", "shifts", "schedule", "schedules", "work shift", "work shifts"])
        elif table_name == "goals":
            aliases.update(["goal", "goals", "okr", "okrs", "target", "targets", "milestone", "milestones"])
        elif table_name == "projects":
            aliases.update(["project", "projects"])
        elif table_name == "subtasks":
            aliases.update(["subtask", "subtasks", "sub-task", "sub-tasks"])
        elif table_name == "comments":
            aliases.update(["comment", "comments", "note", "notes", "feedback"])
        elif table_name == "documents":
            aliases.update(["document", "documents", "file", "files", "attachment", "attachments"])
            
        return sorted(list(aliases), key=len, reverse=True)

    def _detect_tables_by_entities(self, lower_q: str) -> List[str]:
        global _PROJECTS_CACHE, _USERS_CACHE
        matched = []
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        
        # Load project cache if not loaded
        if _PROJECTS_CACHE is None:
            _PROJECTS_CACHE = []
            try:
                res_str = _execute_sql('SELECT name FROM "projects"', project_id, database_id)
                res = json.loads(res_str)
                if res.get("status") == "success":
                    _PROJECTS_CACHE = [row.get("name") for row in res.get("results", []) if row.get("name")]
                    print(f"[_detect_tables_by_entities] Cached {len(_PROJECTS_CACHE)} projects", file=sys.stderr)
            except Exception as e:
                print(f"[_detect_tables_by_entities] Failed to cache projects: {e}", file=sys.stderr)
                
        # Load user cache if not loaded
        if _USERS_CACHE is None:
            _USERS_CACHE = []
            try:
                res_str = _execute_sql('SELECT id, "firstName", "lastName" FROM "company_users"', project_id, database_id)
                res = json.loads(res_str)
                if res.get("status") == "success":
                    _USERS_CACHE = [
                        {
                            "id": row.get("id"),
                            "firstName": row.get("firstName") or "",
                            "lastName": row.get("lastName") or "",
                            "fullName": f"{row.get('firstName') or ''} {row.get('lastName') or ''}".strip()
                        }
                        for row in res.get("results", [])
                    ]
                    print(f"[_detect_tables_by_entities] Cached {len(_USERS_CACHE)} users", file=sys.stderr)
            except Exception as e:
                print(f"[_detect_tables_by_entities] Failed to cache users: {e}", file=sys.stderr)
                
        # Check projects in cache
        for proj_name in _PROJECTS_CACHE:
            if len(proj_name) > 3 and proj_name.lower() in lower_q:
                print(f"[_detect_tables_by_entities] Matched project: {proj_name}", file=sys.stderr)
                matched.append("projects")
                break
                
        # Check users in cache
        for user in _USERS_CACHE:
            full_name = user["fullName"]
            if len(full_name) > 2 and full_name.lower() in lower_q:
                print(f"[_detect_tables_by_entities] Matched employee: {full_name}", file=sys.stderr)
                matched.append("company_users")
                break
            fn = user["firstName"]
            ln = user["lastName"]
            if len(fn) > 2 and re.search(r'\b' + re.escape(fn.lower()) + r'\b', lower_q):
                print(f"[_detect_tables_by_entities] Matched employee firstName: {fn}", file=sys.stderr)
                matched.append("company_users")
                break
            if len(ln) > 2 and re.search(r'\b' + re.escape(ln.lower()) + r'\b', lower_q):
                print(f"[_detect_tables_by_entities] Matched employee lastName: {ln}", file=sys.stderr)
                matched.append("company_users")
                break
                
        return matched

    def _extract_person_filters(self, lower_q: str, tables: List[str]) -> None:
        global _USERS_CACHE
        if _USERS_CACHE is None:
            return
            
        matched_users = []
        for user in _USERS_CACHE:
            full_name = user["fullName"]
            fn = user["firstName"]
            ln = user["lastName"]
            
            has_match = False
            match_type = "any"
            if len(full_name) > 2 and full_name.lower() in lower_q:
                has_match = True
            elif len(fn) > 2 and re.search(r'\b' + re.escape(fn.lower()) + r'\b', lower_q):
                has_match = True
            elif len(ln) > 2 and re.search(r'\b' + re.escape(ln.lower()) + r'\b', lower_q):
                has_match = True
                
            if has_match:
                name_to_check = full_name.lower() if full_name.lower() in lower_q else (fn.lower() if fn.lower() in lower_q else ln.lower())
                assigned_pattern = rf'(?:assigned\s+to|for|tasks\s+of|tasks\s+for)\s+(?:[a-zA-Z]+\s+)?{re.escape(name_to_check)}'
                created_pattern = rf'(?:created\s+by|by|completed\s+by|done\s+by)\s+(?:[a-zA-Z]+\s+)?{re.escape(name_to_check)}'
                
                if re.search(assigned_pattern, lower_q):
                    match_type = "assigned"
                elif re.search(created_pattern, lower_q):
                    match_type = "created"
                elif f"{name_to_check}'s" in lower_q or f"{name_to_check}s" in lower_q:
                    match_type = "owner"
                    
                matched_users.append((user, match_type))
                
        if not matched_users:
            return
            
        for user, match_type in matched_users:
            u_id = user["id"]
            for tbl in tables:
                if tbl == 'company_users':
                    self.filters.append(f'"{tbl}"."id" = \'{u_id}\'')
                    continue
                    
                tbl_info = self.schema_meta.get(tbl, {})
                tbl_cols = tbl_info.get("columns", {})
                potential_cols = []
                
                for fk in tbl_info.get("foreign_keys", []):
                    if fk.get("to_table") == "company_users":
                        potential_cols.append(fk.get("column"))
                        
                common_user_cols = ['userId', 'user_id', 'assignedToId', 'assigned_to_id', 'employeeId', 'employee_id', 'ownerId', 'owner_id', 'createdById', 'created_by_id']
                for c in common_user_cols:
                    if c in tbl_cols and c not in potential_cols:
                        potential_cols.append(c)
                        
                if not potential_cols:
                    continue
                    
                best_col = None
                if match_type == "assigned":
                    for c in ['assignedToId', 'assigned_to_id', 'userId', 'user_id']:
                        if c in potential_cols:
                            best_col = c
                            break
                elif match_type == "created":
                    for c in ['createdById', 'created_by_id', 'ownerId', 'owner_id']:
                        if c in potential_cols:
                            best_col = c
                            break
                elif match_type == "owner":
                    for c in ['ownerId', 'owner_id', 'userId', 'user_id', 'assignedToId', 'assigned_to_id']:
                        if c in potential_cols:
                            best_col = c
                            break
                            
                if not best_col and potential_cols:
                    best_col = potential_cols[0]
                    
                if best_col:
                    self.filters.append(f'"{tbl}"."{best_col}" = \'{u_id}\'')

    def detect_tables(self, lower_q: str, semantic_search_query: str = None) -> List[str]:
        """Detect relevant tables from the query via keywords or semantic similarity."""
        matched = []
        
        # 1. Keyword search (longest aliases first to prevent substring collision)
        sorted_aliases = sorted(self.table_aliases.keys(), key=len, reverse=True)
        temp_q = lower_q
        for alias in sorted_aliases:
            pattern = rf"\b{re.escape(alias)}\b"
            if re.search(pattern, temp_q):
                tbl = self.table_aliases[alias]
                if tbl not in matched:
                    matched.append(tbl)
                # Remove matched alias to avoid matching smaller parts
                temp_q = re.sub(pattern, " ", temp_q)
                
        # 2. Entity-based table detection (matching actual names from database)
        entity_tables = self._detect_tables_by_entities(lower_q)
        for tbl in entity_tables:
            if tbl not in matched:
                matched.append(tbl)
        print(f"[detect_tables] Final matched tables: {matched}", file=sys.stderr)
                
        # 3. Semantic fallback to find additional tables
        if semantic_search_query:
            try:
                results = self.manager.semantic_search(semantic_search_query, top_k=3)
                for doc, score in results:
                    tbl = doc["metadata"].get("table_name")
                    if tbl and tbl not in matched:
                        threshold = 0.50 if matched else 0.45
                        if score > threshold:
                            matched.append(tbl)
            except Exception as e:
                print(f"Semantic table detection failed: {e}", file=sys.stderr)
                
        return matched

    def interpret(self, query: str) -> str:
        """Convert natural language query to SQL."""
        self.query = query.strip()
        lower_q = self.query.lower()
        
        if self.query.upper().startswith('SELECT'):
            return self.query
            
        # Detect tables
        matched_tables = self.detect_tables(lower_q)
        if not matched_tables:
            return None
            
        return self.interpret_with_tables(query, matched_tables)
        
    def interpret_with_tables(self, query: str, matched_tables: List[str]) -> str:
        """Interpret query using an explicit list of matched tables."""
        self.query = query.strip()
        lower_q = self.query.lower()
        
        self.table = matched_tables[0]
        self.matched_tables = matched_tables
        
        # Reset clauses
        self.columns = []
        self.filters = []
        self.joins = []
        self.group_by = None
        self.having = None
        self.order_by = None
        self.limit = 100
        
        # 1. Extract joins
        self.joins = find_joins(matched_tables, self.schema_meta)
        
        # 2. Extract columns
        self._extract_columns_dynamic(lower_q, matched_tables)
        
        # 3. Extract filters
        self._extract_filters_dynamic(lower_q, matched_tables)
        
        # 3.5 Extract person filters
        self._extract_person_filters(lower_q, matched_tables)
        
        # 4. Extract aggregates
        self._extract_aggregates(lower_q, matched_tables)
        
        # 5. Extract sorting
        self._extract_sorting_dynamic(lower_q, matched_tables)
        
        # 6. Extract limit
        self._extract_limit(lower_q)
        
        return self._build_sql()

    def _find_column(self, word: str, tables: List[str]) -> Optional[Tuple[str, str, str]]:
        """Find a column in matched tables. Returns (table_name, col_name, col_type) or None."""
        word_clean = word.lower().replace(" ", "").replace("_", "")
        for tbl in tables:
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            for col_name, col_type in cols.items():
                col_clean = col_name.lower().replace("_", "")
                if word_clean == col_clean:
                    return tbl, col_name, col_type
        return None

    def _extract_columns_dynamic(self, lower_q: str, tables: List[str]) -> None:
        primary_table = tables[0]
        if 'count' in lower_q or 'how many' in lower_q:
            return
            
        found_cols = []
        for tbl in tables:
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            for col_name in cols.keys():
                if col_name.lower() in lower_q:
                    col_ref = f'"{tbl}"."{col_name}"'
                    if col_ref not in found_cols:
                        found_cols.append(col_ref)
                        
        if found_cols:
            self.columns = found_cols
        else:
            self.columns = [f'"{primary_table}".*']
            for tbl in tables[1:]:
                cols = self.schema_meta.get(tbl, {}).get("columns", {})
                for label_col in ('name', 'title', 'email', 'label', 'displayName'):
                    if label_col in cols:
                        self.columns.append(f'"{tbl}"."{label_col}" AS "{tbl}_{label_col}"')
                        break

    def _extract_filters_dynamic(self, lower_q: str, tables: List[str]) -> None:
        # 1. Location filter
        loc_match = re.search(r'(?:in|from|at)\s+([a-zA-Z\s]+)', lower_q)
        if loc_match:
            location = loc_match.group(1).strip()
            if location not in ('today', 'yesterday', 'this', 'last', 'desc', 'asc', 'limit'):
                for tbl in tables:
                    cols = self.schema_meta.get(tbl, {}).get("columns", {})
                    for col in ('city', 'address', 'location'):
                        if col in cols:
                            self.filters.append(f'LOWER("{tbl}"."{col}") LIKE \'%{location.lower()}%\'')
                            
        # 2. Name filter
        name_match = re.search(r'(?:named|called|for|in)\s+(["\']?)([a-zA-Z0-9\s\-_&]+)\1', lower_q)
        if name_match:
            name = name_match.group(2).strip()
            is_city = name.lower() in ('mumbai', 'delhi', 'bangalore', 'pune', 'hyderabad', 'chennai', 'kolkata')
            if name not in ('today', 'yesterday', 'this', 'last', 'desc', 'asc', 'limit') and not is_city:
                for tbl in tables:
                    cols = self.schema_meta.get(tbl, {}).get("columns", {})
                    for col in ('name', 'title', 'label', 'displayName'):
                        if col in cols:
                            self.filters.append(f'LOWER("{tbl}"."{col}") LIKE \'%{name.lower()}%\'')

        # 2.5 Project name filter using cache
        global _PROJECTS_CACHE
        if _PROJECTS_CACHE:
            for proj_name in _PROJECTS_CACHE:
                if len(proj_name) > 3 and proj_name.lower() in lower_q:
                    for tbl in tables:
                        tbl_info = self.schema_meta.get(tbl, {})
                        tbl_cols = tbl_info.get("columns", {})
                        if tbl == 'projects':
                            self.filters.append(f'LOWER("{tbl}"."name") = \'{proj_name.lower()}\'')
                        else:
                            proj_cols = [c for c in tbl_cols.keys() if c.lower() in ('projectid', 'project_id')]
                            if proj_cols:
                                self.filters.append(f'"{tbl}"."{proj_cols[0]}" IN (SELECT id FROM "projects" WHERE LOWER("name") = \'{proj_name.lower()}\')')
                            
        # 3. Status filter (both explicit value and keyword detection)
        status_match = re.search(r'status\s+(?:is\s+)?(["\']?)([a-zA-Z_0-9\-]+)\1', lower_q)
        if status_match:
            status = status_match.group(2).strip().lower()
            if status not in ('and', 'or', 'by', 'in', 'is', 'for', 'a', 'an', 'the', 'of', 'with', 'to', 'from', 'distribution', 'trend'):
                if status in ('not_completed', 'not-completed', 'not completed', 'incomplete', 'unfinished'):
                    status_vals = ("todo", "in_progress", "in-progress", "active", "not_started", "pending")
                elif status in ('completed', 'done', 'closed', 'finished', 'resolved'):
                    status_vals = ("completed", "done", "closed")
                elif status in ('todo', 'not_started', 'not-started', 'pending'):
                    status_vals = ("todo", "not_started", "pending")
                elif status in ('in_progress', 'in-progress', 'doing', 'active', 'ongoing'):
                    status_vals = ("in_progress", "active")
                else:
                    status_vals = (status,)
                
                for tbl in tables:
                    cols = self.schema_meta.get(tbl, {}).get("columns", {})
                    if 'status' in cols:
                        vals_str = ", ".join(f"'{v}'" for v in status_vals)
                        self.filters.append(f'"{tbl}"."status" IN ({vals_str})')
        else:
            status_cols_exist = False
            for tbl in tables:
                cols = self.schema_meta.get(tbl, {}).get("columns", {})
                if 'status' in cols:
                    status_cols_exist = True
                    break
            
            if status_cols_exist:
                if any(k in lower_q for k in ('not completed', 'not_completed', 'not-completed', 'incomplete', 'unfinished')):
                    for tbl in tables:
                        cols = self.schema_meta.get(tbl, {}).get("columns", {})
                        if 'status' in cols:
                            self.filters.append(f'"{tbl}"."status" NOT IN (\'completed\', \'done\', \'closed\')')
                elif any(k in lower_q for k in ('pending', 'not started', 'todo', 'to-do', 'uncompleted')):
                    for tbl in tables:
                        cols = self.schema_meta.get(tbl, {}).get("columns", {})
                        if 'status' in cols:
                            self.filters.append(f'"{tbl}"."status" IN (\'todo\', \'not_started\', \'not-started\', \'pending\')')
                elif any(k in lower_q for k in ('in progress', 'in-progress', 'doing', 'active', 'ongoing')):
                    for tbl in tables:
                        cols = self.schema_meta.get(tbl, {}).get("columns", {})
                        if 'status' in cols:
                            self.filters.append(f'"{tbl}"."status" IN (\'in_progress\', \'in-progress\', \'active\')')
                elif any(k in lower_q for k in ('completed', 'completed tasks', 'done', 'finished', 'closed')):
                    for tbl in tables:
                        cols = self.schema_meta.get(tbl, {}).get("columns", {})
                        if 'status' in cols:
                            self.filters.append(f'"{tbl}"."status" IN (\'completed\', \'done\', \'closed\')')

        # 3.5 Task Type / Project Type filter mapping
        for tbl in tables:
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            if 'taskType' in cols:
                type_match = re.search(r'(?:task\s+)?type\s+(?:is\s+)?(["\']?)([a-zA-Z_0-9\-]+)\1', lower_q)
                if type_match:
                    t_type = type_match.group(2).strip().lower()
                    if t_type in ('bug', 'bugs', 'defect', 'defects', 'issue', 'issues'):
                        self.filters.append(f'LOWER("{tbl}"."taskType") = \'bug\'')
                    elif t_type in ('feature', 'features', 'enhancement', 'enhancements', 'story', 'stories'):
                        self.filters.append(f'LOWER("{tbl}"."taskType") = \'feature\'')
                    elif t_type in ('task', 'tasks', 'todo', 'to-do'):
                        self.filters.append(f'LOWER("{tbl}"."taskType") = \'task\'')
                else:
                    if any(k in lower_q for k in ('bug', 'bugs', 'defect', 'defects', 'issue', 'issues')):
                        self.filters.append(f'LOWER("{tbl}"."taskType") = \'bug\'')
                    elif any(k in lower_q for k in ('feature', 'features', 'enhancement', 'enhancements', 'story', 'stories')):
                        self.filters.append(f'LOWER("{tbl}"."taskType") = \'feature\'')
                    elif re.search(r'\btype\s+tasks?\b', lower_q) or re.search(r'\btasks?\s+type\b', lower_q):
                        self.filters.append(f'LOWER("{tbl}"."taskType") = \'task\'')

            if 'projectType' in cols:
                if any(k in lower_q for k in ('client project', 'client-project', 'external project')):
                    self.filters.append(f'LOWER("{tbl}"."projectType") = \'client\'')
                elif any(k in lower_q for k in ('internal project', 'internal-project')):
                    self.filters.append(f'LOWER("{tbl}"."projectType") = \'internal\'')

        # Overdue / Delayed keyword detection
        if any(k in lower_q for k in ('overdue', 'delayed', 'late')):
            for tbl in tables:
                cols = self.schema_meta.get(tbl, {}).get("columns", {})
                date_cols = [c for c, t in cols.items() if 'timestamp' in t or 'date' in t]
                due_col = None
                for c in ('endDate', 'end_date', 'dueDate', 'due_date', 'latestDueDate', 'deadline'):
                    if c in cols:
                        due_col = c
                        break
                if not due_col and date_cols:
                    due_col = date_cols[0]
                    
                status_col = 'status' if 'status' in cols else None
                if due_col:
                    if status_col:
                        self.filters.append(f'"{tbl}"."{due_col}" < CURRENT_DATE AND "{tbl}"."{status_col}" NOT IN (\'completed\', \'done\', \'closed\')')
                    else:
                        self.filters.append(f'"{tbl}"."{due_col}" < CURRENT_DATE')
                    
        # 4. Amount / numeric comparisons
        for tbl in tables:
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            for col_name, col_type in cols.items():
                if 'int' in col_type or 'numeric' in col_type or 'double' in col_type or 'real' in col_type:
                    col_esc = re.escape(col_name)
                    gt_pattern = rf'(?:{col_esc}\s*)?(?:above|greater than|more than|>)\s*(\d+(?:\.\d+)?)'
                    lt_pattern = rf'(?:{col_esc}\s*)?(?:below|less than|under|<)\s*(\d+(?:\.\d+)?)'
                    eq_pattern = rf'{col_esc}\s*(?:=|\bis\b|equal to)\s*(\d+(?:\.\d+)?)'
                    
                    gt_m = re.search(gt_pattern, lower_q)
                    if gt_m:
                        self.filters.append(f'"{tbl}"."{col_name}" > {gt_m.group(1)}')
                    lt_m = re.search(lt_pattern, lower_q)
                    if lt_m:
                        self.filters.append(f'"{tbl}"."{col_name}" < {lt_m.group(1)}')
                    eq_m = re.search(eq_pattern, lower_q)
                    if eq_m:
                        self.filters.append(f'"{tbl}"."{col_name}" = {eq_m.group(1)}')
                        
        # 5. Date / Timestamp filters
        months_map = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        for tbl in tables:
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            date_cols = [c for c, t in cols.items() if 'timestamp' in t or 'date' in t]
            if not date_cols:
                continue
                
            # Select the single best date column for this table
            best_col = None
            if 'start' in lower_q:
                for c in date_cols:
                    if 'start' in c.lower():
                        best_col = c
                        break
            if not best_col and any(k in lower_q for k in ('end', 'due', 'deadline', 'finish', 'expire')):
                for c in date_cols:
                    if any(k in c.lower() for k in ('end', 'due', 'deadline', 'finish', 'expire')):
                        best_col = c
                        break
            if not best_col and 'create' in lower_q:
                for c in date_cols:
                    if 'create' in c.lower():
                        best_col = c
                        break
            if not best_col and 'update' in lower_q:
                for c in date_cols:
                    if 'update' in c.lower():
                        best_col = c
                        break
            if not best_col and any(k in lower_q for k in ('complete', 'close')):
                for c in date_cols:
                    if any(k in c.lower() for k in ('complete', 'close')):
                        best_col = c
                        break
                        
            # Fallbacks
            if not best_col:
                for c in date_cols:
                    if c.lower() == 'date':
                        best_col = c
                        break
            if not best_col:
                for c in date_cols:
                    if 'create' in c.lower():
                        best_col = c
                        break
            if not best_col:
                best_col = date_cols[0]
                
            # Parse Date logic
            date_filter_added = False
            year_val = None
            year_match = re.search(r'\b(20\d{2})\b', lower_q)
            if year_match:
                year_val = int(year_match.group(1))
            else:
                year_val = datetime.now().year

            def get_month_num(m_str):
                return months_map.get(m_str.lower())

            # 1. Last N days/weeks/months/years
            last_n_match = re.search(r'\b(?:last|past)\s+(\d+)\s+(day|week|month|year)s?\b', lower_q)
            if last_n_match:
                val = int(last_n_match.group(1))
                unit = last_n_match.group(2)
                self.filters.append(f'"{tbl}"."{best_col}" >= CURRENT_DATE - INTERVAL \'{val} {unit}s\'')
                date_filter_added = True

            # 2. Date range: between ... and ...
            if not date_filter_added:
                range_match1 = re.search(r'\bbetween\s+([a-z]+)(?:\s+(\d{4}))?\s+and\s+([a-z]+)(?:\s+(\d{4}))?\b', lower_q)
                range_match2 = re.search(r'\bfrom\s+([a-z]+)(?:\s+(\d{4}))?\s+to\s+([a-z]+)(?:\s+(\d{4}))?\b', lower_q)
                range_match = range_match1 or range_match2
                if range_match:
                    m1 = get_month_num(range_match.group(1))
                    m2 = get_month_num(range_match.group(3))
                    if m1 and m2:
                        y1 = int(range_match.group(2)) if range_match.group(2) else (int(range_match.group(4)) if range_match.group(4) else year_val)
                        y2 = int(range_match.group(4)) if range_match.group(4) else y1
                        self.filters.append(f'"{tbl}"."{best_col}" BETWEEN \'{y1}-{m1:02d}-01\' AND (\'{y2}-{m2:02d}-01\'::date + INTERVAL \'1 month\' - INTERVAL \'1 day\')')
                        date_filter_added = True

            # 3. Year range: from 2024 to 2025
            if not date_filter_added:
                yr_range = re.search(r'\b(?:from|between)\s+(20\d{2})\s+(?:to|and)\s+(20\d{2})\b', lower_q)
                if yr_range:
                    y1 = yr_range.group(1)
                    y2 = yr_range.group(2)
                    self.filters.append(f'EXTRACT(YEAR FROM "{tbl}"."{best_col}") BETWEEN {y1} AND {y2}')
                    date_filter_added = True

            # 4. Quarters
            if not date_filter_added:
                q_match1 = re.search(r'\bq([1-4])\b', lower_q)
                q_match2 = re.search(r'\b(first|second|third|fourth)\s+quarter\b', lower_q)
                q_val = None
                if q_match1:
                    q_val = int(q_match1.group(1))
                elif q_match2:
                    q_map = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
                    q_val = q_map.get(q_match2.group(1))
                if q_val:
                    self.filters.append(f'EXTRACT(QUARTER FROM "{tbl}"."{best_col}") = {q_val}')
                    if year_match:
                        self.filters.append(f'EXTRACT(YEAR FROM "{tbl}"."{best_col}") = {year_val}')
                    date_filter_added = True

            # 5. Week number
            if not date_filter_added:
                w_match = re.search(r'\bweek\s+(\d+)\b', lower_q)
                if w_match:
                    w_val = int(w_match.group(1))
                    self.filters.append(f'EXTRACT(WEEK FROM "{tbl}"."{best_col}") = {w_val}')
                    if year_match:
                        self.filters.append(f'EXTRACT(YEAR FROM "{tbl}"."{best_col}") = {year_val}')
                    date_filter_added = True

            # 6. Day of week
            if not date_filter_added:
                dow_map = {
                    'monday': 1, 'mondays': 1,
                    'tuesday': 2, 'tuesdays': 2,
                    'wednesday': 3, 'wednesdays': 3,
                    'thursday': 4, 'thursdays': 4,
                    'friday': 5, 'fridays': 5,
                    'saturday': 6, 'saturdays': 6,
                    'sunday': 7, 'sundays': 7
                }
                for dow_name, dow_val in dow_map.items():
                    if re.search(r'\b' + re.escape(dow_name) + r'\b', lower_q):
                        self.filters.append(f'EXTRACT(ISODOW FROM "{tbl}"."{best_col}") = {dow_val}')
                        date_filter_added = True
                        break

            # 7. Specific date format: YYYY-MM-DD
            if not date_filter_added:
                iso_match = re.search(r'\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b', lower_q)
                if iso_match:
                    y = int(iso_match.group(1))
                    m = int(iso_match.group(2))
                    d = int(iso_match.group(3))
                    self.filters.append(f'DATE("{tbl}"."{best_col}") = \'{y}-{m:02d}-{d:02d}\'')
                    date_filter_added = True

            # 8. Specific date format: on [Month] [Day], [Year]
            if not date_filter_added:
                month_day_match = re.search(r'\b(?:on\s+)?([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{4}))?\b', lower_q)
                if month_day_match:
                    m_name = month_day_match.group(1)
                    d_val = int(month_day_match.group(2))
                    m_val = get_month_num(m_name)
                    if m_val and 1 <= d_val <= 31:
                        y_val = int(month_day_match.group(3)) if month_day_match.group(3) else year_val
                        self.filters.append(f'DATE("{tbl}"."{best_col}") = \'{y_val}-{m_val:02d}-{d_val:02d}\'')
                        date_filter_added = True

            # 9. Month + Year combo: in March 2025
            if not date_filter_added:
                month_year_match = re.search(r'\b(?:in|during|for)\s+([a-z]+)\s+(20\d{2})\b', lower_q)
                if month_year_match:
                    m_name = month_year_match.group(1)
                    m_val = get_month_num(m_name)
                    y_val = int(month_year_match.group(2))
                    if m_val:
                        self.filters.append(f'EXTRACT(MONTH FROM "{tbl}"."{best_col}") = {m_val} AND EXTRACT(YEAR FROM "{tbl}"."{best_col}") = {y_val}')
                        date_filter_added = True

            # 10. Before/After filters
            if not date_filter_added:
                before_match = re.search(r'\bbefore\s+(?:on\s+)?(?:(20\d{2})[-/](\d{1,2})[-/](\d{1,2})|([a-z]+)\s+(20\d{2})|([a-z]+)\s+(\d{1,2}))\b', lower_q)
                if before_match:
                    if before_match.group(1): # ISO format
                        self.filters.append(f'"{tbl}"."{best_col}" < \'{before_match.group(1)}-{int(before_match.group(2)):02d}-{int(before_match.group(3)):02d}\'')
                        date_filter_added = True
                    elif before_match.group(4): # Month Year
                        m_val = get_month_num(before_match.group(4))
                        y_val = int(before_match.group(5))
                        if m_val:
                            self.filters.append(f'"{tbl}"."{best_col}" < \'{y_val}-{m_val:02d}-01\'')
                            date_filter_added = True
                    elif before_match.group(6): # Month Day
                        m_val = get_month_num(before_match.group(6))
                        d_val = int(before_match.group(7))
                        if m_val:
                            self.filters.append(f'"{tbl}"."{best_col}" < \'{year_val}-{m_val:02d}-{d_val:02d}\'')
                            date_filter_added = True

                after_match = re.search(r'\bafter\s+(?:on\s+)?(?:(20\d{2})[-/](\d{1,2})[-/](\d{1,2})|([a-z]+)\s+(20\d{2})|([a-z]+)\s+(\d{1,2}))\b', lower_q)
                if after_match:
                    if after_match.group(1): # ISO format
                        self.filters.append(f'"{tbl}"."{best_col}" > \'{after_match.group(1)}-{int(after_match.group(2)):02d}-{int(after_match.group(3)):02d}\'')
                        date_filter_added = True
                    elif after_match.group(4): # Month Year
                        m_val = get_month_num(after_match.group(4))
                        y_val = int(after_match.group(5))
                        if m_val:
                            self.filters.append(f'"{tbl}"."{best_col}" >= \'{y_val}-{m_val:02d}-01\'::date + INTERVAL \'1 month\'')
                            date_filter_added = True
                    elif after_match.group(6): # Month Day
                        m_val = get_month_num(after_match.group(6))
                        d_val = int(after_match.group(7))
                        if m_val:
                            self.filters.append(f'"{tbl}"."{best_col}" > \'{year_val}-{m_val:02d}-{d_val:02d}\'')
                            date_filter_added = True

            # Fallbacks
            if not date_filter_added:
                if 'today' in lower_q:
                    self.filters.append(f'DATE("{tbl}"."{best_col}") = CURRENT_DATE')
                elif 'yesterday' in lower_q:
                    self.filters.append(f'DATE("{tbl}"."{best_col}") = CURRENT_DATE - 1')
                elif 'this month' in lower_q:
                    self.filters.append(f'DATE_TRUNC(\'month\', "{tbl}"."{best_col}") = DATE_TRUNC(\'month\', CURRENT_DATE)')
                elif 'last month' in lower_q:
                    self.filters.append(f'DATE_TRUNC(\'month\', "{tbl}"."{best_col}") = DATE_TRUNC(\'month\', CURRENT_DATE - INTERVAL \'1 month\')')
                elif 'this year' in lower_q:
                    self.filters.append(f'DATE_TRUNC(\'year\', "{tbl}"."{best_col}") = DATE_TRUNC(\'year\', CURRENT_DATE)')
                elif 'last year' in lower_q:
                    self.filters.append(f'DATE_TRUNC(\'year\', "{tbl}"."{best_col}") = DATE_TRUNC(\'year\', CURRENT_DATE - INTERVAL \'1 year\')')
                elif 'this week' in lower_q:
                    self.filters.append(f'DATE_TRUNC(\'week\', "{tbl}"."{best_col}") = DATE_TRUNC(\'week\', CURRENT_DATE)')
                elif 'last week' in lower_q:
                    self.filters.append(f'DATE_TRUNC(\'week\', "{tbl}"."{best_col}") = DATE_TRUNC(\'week\', CURRENT_DATE - INTERVAL \'1 week\')')
                else:
                    if year_match:
                        self.filters.append(f'EXTRACT(YEAR FROM "{tbl}"."{best_col}") = {year_val}')
                    for m_name, m_num in months_map.items():
                        if re.search(r'\b' + re.escape(m_name) + r'\b', lower_q):
                            self.filters.append(f'EXTRACT(MONTH FROM "{tbl}"."{best_col}") = {m_num}')
                            break

        # 6. Column-Value pairs (e.g. projectId = "proj_xxx")
        for tbl in tables:
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            for col_name, col_type in cols.items():
                if any(f'"{tbl}"."{col_name}"' in f for f in self.filters):
                    continue
                col_pattern = rf"\b{re.escape(col_name)}\b"
                match = re.search(col_pattern, lower_q)
                if match:
                    start_idx = match.end()
                    sub_str = lower_q[start_idx:].strip()
                    sub_str = re.sub(r'^(?:=|\bis\b|:)\s*', '', sub_str).strip()
                    val_match = re.match(r'^([\'"])(.*?)\1|^([a-zA-Z0-9_\-\.]+)', sub_str)
                    if val_match:
                        val = val_match.group(2) or val_match.group(3)
                        if val not in ('limit', 'group', 'order', 'sort', 'by', 'and', 'or', 'join', 'select', 'today', 'yesterday', 'this', 'last', 'in', 'for', 'to', 'with', 'of', 'at', 'on', 'about', 'is', 'a', 'an', 'the'):
                            if 'int' in col_type or 'numeric' in col_type or 'double' in col_type or 'real' in col_type:
                                try:
                                    float(val)
                                    self.filters.append(f'"{tbl}"."{col_name}" = {val}')
                                except ValueError:
                                    pass
                            else:
                                self.filters.append(f'"{tbl}"."{col_name}" = \'{val}\'')

    def _extract_aggregates(self, lower_q: str, tables: List[str]) -> None:
        group_match = re.search(r'(?:group\s+)?by\s+([a-zA-Z_0-9\s]+)|per\s+([a-zA-Z_0-9\s]+)', lower_q)
        group_col_info = None
        if group_match:
            group_word = group_match.group(1) or group_match.group(2)
            group_col_info = self._find_column(group_word.strip(), tables)
            if group_col_info:
                tbl, col, _ = group_col_info
                self.group_by = f'"{tbl}"."{col}"'
                
        # Time-based grouping check: per/by month, week, day, quarter, year
        time_unit = None
        if re.search(r'\b(?:per|by)\s+month\b', lower_q) or 'monthly' in lower_q:
            time_unit = 'month'
        elif re.search(r'\b(?:per|by)\s+week\b', lower_q) or 'weekly' in lower_q:
            time_unit = 'week'
        elif re.search(r'\b(?:per|by)\s+day\b', lower_q) or 'daily' in lower_q:
            time_unit = 'day'
        elif re.search(r'\b(?:per|by)\s+quarter\b', lower_q) or 'quarterly' in lower_q:
            time_unit = 'quarter'
        elif re.search(r'\b(?:per|by)\s+year\b', lower_q) or 'yearly' in lower_q:
            time_unit = 'year'
            
        time_group_expr = None
        if time_unit:
            tbl = tables[0]
            cols = self.schema_meta.get(tbl, {}).get("columns", {})
            date_cols = [c for c, t in cols.items() if 'timestamp' in t or 'date' in t]
            if date_cols:
                best_col = None
                for c in ('createdAt', 'created_at', 'startDate', 'start_date', 'date'):
                    if c in date_cols:
                        best_col = c
                        break
                if not best_col:
                    best_col = date_cols[0]
                time_group_expr = f"DATE_TRUNC('{time_unit}', \"{tbl}\".\"{best_col}\")"
                self.group_by = time_group_expr
                
        found_agg = False
        for agg_word, sql_func in self.AGGREGATE_FUNCTIONS.items():
            if agg_word in lower_q or (agg_word == 'count' and 'how many' in lower_q):
                if sql_func == 'COUNT(*)':
                    if time_group_expr:
                        self.columns = [f"{time_group_expr} as time_period", 'COUNT(*) as count_result']
                    elif group_col_info:
                        self.columns = [self.group_by, 'COUNT(*) as count_result']
                    else:
                        self.columns = ['COUNT(*) as count_result']
                    found_agg = True
                    break
                else:
                    num_col_info = None
                    for tbl in tables:
                        cols = self.schema_meta.get(tbl, {}).get("columns", {})
                        for col_name, col_type in cols.items():
                            if 'int' in col_type or 'numeric' in col_type or 'double' in col_type or 'real' in col_type:
                                if col_name.lower() in lower_q:
                                    num_col_info = (tbl, col_name)
                                    break
                        if num_col_info:
                            break
                    if not num_col_info:
                        for tbl in tables:
                            cols = self.schema_meta.get(tbl, {}).get("columns", {})
                            for col_name, col_type in cols.items():
                                if 'int' in col_type or 'numeric' in col_type or 'double' in col_type or 'real' in col_type:
                                    num_col_info = (tbl, col_name)
                                    break
                            if num_col_info:
                                break
                                
                    if num_col_info:
                        tbl, col = num_col_info
                        agg_expr = f'{sql_func}("{tbl}"."{col}") as {agg_word}_result'
                        if time_group_expr:
                            self.columns = [f"{time_group_expr} as time_period", agg_expr]
                        elif group_col_info:
                            self.columns = [self.group_by, agg_expr]
                        else:
                            self.columns = [agg_expr]
                        found_agg = True
                        break
                        
        if not found_agg and (time_group_expr or group_col_info):
            if time_group_expr:
                self.columns = [f"{time_group_expr} as time_period", 'COUNT(*) as count_result']
            else:
                self.columns = [self.group_by, 'COUNT(*) as count_result']
                
        # Extract HAVING clause
        having_match = re.search(r'\bhaving\s+(?:more\s+than|greater\s+than|>)\s*(\d+)\b', lower_q)
        if having_match:
            val = having_match.group(1)
            self.having = f"COUNT(*) > {val}"
            
        # Top/Bottom sorting logic for aggregation
        top_match = re.search(r'\btop\s+(\d+)\b', lower_q)
        if top_match:
            self.limit = int(top_match.group(1))
            if self.group_by:
                self.order_by = "count_result DESC"
        else:
            bottom_match = re.search(r'\bbottom\s+(\d+)\b', lower_q)
            if bottom_match:
                self.limit = int(bottom_match.group(1))
                if self.group_by:
                    self.order_by = "count_result ASC"

    def _extract_sorting_dynamic(self, lower_q: str, tables: List[str]) -> None:
        direction = "DESC"
        if 'ascending' in lower_q or 'asc' in lower_q:
            direction = "ASC"
        elif 'descending' in lower_q or 'desc' in lower_q:
            direction = "DESC"
            
        sort_match = re.search(r'sort(?:ed)?\s+by\s+([a-zA-Z_0-9]+)|order\s+by\s+([a-zA-Z_0-9]+)', lower_q)
        if sort_match:
            sort_word = sort_match.group(1) or sort_match.group(2)
            col_info = self._find_column(sort_word.strip(), tables)
            if col_info:
                tbl, col, _ = col_info
                self.order_by = f'"{tbl}"."{col}" {direction}'
        else:
            primary_table = tables[0]
            cols = self.schema_meta.get(primary_table, {}).get("columns", {})
            for col in ('createdAt', 'created_at', 'date'):
                if col in cols:
                    self.order_by = f'"{primary_table}"."{col}" DESC'
                    break

    def _extract_limit(self, lower_q: str) -> None:
        limit_match = re.search(r'(?:limit|top|first)\s+(\d+)', lower_q)
        if limit_match:
            self.limit = int(limit_match.group(1))
        elif 'all' in lower_q:
            self.limit = None

    def _build_sql(self) -> str:
        if not self.table:
            return None
            
        cols = ", ".join(self.columns)
        sql = f'SELECT {cols} FROM "{self.table}"'
        
        for join_clause in self.joins:
            sql += f" {join_clause}"
            
        if self.filters:
            where_clause = " AND ".join([f"({f})" for f in self.filters])
            sql += f" WHERE {where_clause}"
            
        if self.group_by:
            sql += f" GROUP BY {self.group_by}"
            
        if hasattr(self, 'having') and self.having:
            sql += f" HAVING {self.having}"
            
        if self.order_by:
            sql += f" ORDER BY {self.order_by}"
            
        if self.limit is not None:
            sql += f" LIMIT {self.limit}"
            
        return sql


# ─── Tools ───────────────────────────────────────────────────────────────────

@mcp.tool()
def semantic_search(query: str, top_k: int = 5, enable_sql_fallback: bool = True) -> str:
    """
    Universal query tool - handles ANY type of query.
    
    Intelligently routes between:
    1. Direct SQL execution (if query is SQL)
    2. Natural language → SQL conversion (if convertible)
    3. Semantic similarity search (for schema/documentation queries)
    
    Args:
        query: Natural language question or SQL query
        top_k: Number of results to return (for semantic search)
        enable_sql_fallback: Try SQL conversion before semantic search
    
    Returns:
        JSON with results and metadata
    """
    try:
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        
        lower_q = query.lower().strip()
        
        # Route 1: Direct SQL
        if lower_q.startswith('select'):
            return _execute_sql(query, project_id, database_id)
        
        # Route 2: Natural language → SQL
        if enable_sql_fallback:
            interpreter = QueryInterpreter()
            sql = interpreter.interpret(query)
            if not sql:
                matched_tables = interpreter.detect_tables(lower_q, semantic_search_query=query)
                if matched_tables:
                    sql = interpreter.interpret_with_tables(query, matched_tables)
            if sql:
                sql_res = _execute_sql(sql, project_id, database_id)
                parsed_res = json.loads(sql_res)
                if parsed_res.get("status") == "success":
                    return sql_res
        
        # Route 3: Semantic search (schema/documentation)
        manager = _get_manager()
        results = manager.semantic_search(query, top_k=top_k)
        
        return json.dumps({
            "status": "success",
            "query_type": "semantic_search",
            "query": query,
            "results": [
                {
                    "id": r[0]["id"],
                    "content": r[0]["content"],
                    "similarity_score": float(r[1]),
                    "metadata": r[0]["metadata"]
                }
                for r in results
            ],
            "total_results": len(results)
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "query": query
        }, indent=2)


@mcp.tool()
def smart_query(query: str, top_k_tables: int = 3, execute: bool = True) -> str:
    """
    Intelligent query planner. Uses semantic search to find relevant tables,
    automatically detects JOIN paths, constructs a valid PostgreSQL query, and executes it.
    
    Args:
        query: The natural language question (e.g. "list all shifts for Mumbai employees")
        top_k_tables: How many candidate tables to retrieve from vector DB
        execute: If True, executes the generated SQL and returns data; if False, returns the plan.
    
    Returns:
        JSON with the plan, the generated SQL, and query results.
    """
    try:
        interpreter = QueryInterpreter()
        
        if query.strip().lower().startswith("select"):
            if execute:
                project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
                database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
                return _execute_sql(query, project_id, database_id)
            else:
                return json.dumps({"status": "success", "sql": query})
                
        matched_tables = interpreter.detect_tables(query.lower(), semantic_search_query=query)
        
        if not matched_tables:
            return json.dumps({
                "status": "error",
                "message": "Could not identify any relevant tables for this query."
            }, indent=2)
            
        sql = interpreter.interpret_with_tables(query, matched_tables)
        
        if not sql:
            return json.dumps({
                "status": "error",
                "message": f"Could not construct SQL for tables: {matched_tables}"
            }, indent=2)
            
        plan = {
            "query": query,
            "detected_tables": matched_tables,
            "joins_detected": interpreter.joins,
            "filters_applied": interpreter.filters,
            "sql": sql
        }
        
        if not execute:
            return json.dumps({
                "status": "success",
                "plan": plan
            }, indent=2)
            
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        res_str = _execute_sql(sql, project_id, database_id)
        res = json.loads(res_str)
        
        return json.dumps({
            "status": res.get("status", "success"),
            "plan": plan,
            "results": res.get("results", []),
            "row_count": res.get("row_count", 0),
            "columns": res.get("columns", []),
            "error": res.get("error")
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "query": query
        }, indent=2)


@mcp.tool()
def analyze_data(query: str, analysis_type: str = "auto") -> str:
    """
    Perform data analysis - aggregations, comparisons, trends.
    
    Analysis types:
    - "summary": Count, totals, averages
    - "comparison": Compare across categories
    - "trend": Time-based analysis
    - "distribution": Value distribution
    - "outliers": Anomaly detection
    - "auto": Detect automatically
    
    Args:
        query: What data to analyze (e.g., "sales by team", "revenue trends")
        analysis_type: Type of analysis to perform
    
    Returns:
        JSON with analysis results and insights
    """
    try:
        interpreter = QueryInterpreter()
        sql = interpreter.interpret(query)
        
        if not sql:
            matched_tables = interpreter.detect_tables(query.lower(), semantic_search_query=query)
            if matched_tables:
                sql = interpreter.interpret_with_tables(query, matched_tables)
        
        if not sql:
            return json.dumps({
                "status": "error",
                "message": "Could not understand the query for analysis"
            }, indent=2)
        
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        
        result = _execute_sql(sql, project_id, database_id)
        parsed = json.loads(result)
        
        return json.dumps({
            "status": "success",
            "analysis_type": analysis_type,
            "query": query,
            "data": parsed.get("results", []),
            "summary": {
                "total_records": len(parsed.get("results", [])),
                "columns": parsed.get("columns", [])
            }
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


@mcp.tool()
def execute_custom_sql(sql: str, limit: int = 1000) -> str:
    """
    Execute custom SQL query directly.
    
    Args:
        sql: The SQL query to execute
        limit: Maximum rows to return
    
    Returns:
        JSON with query results
    """
    try:
        if not sql.upper().endswith(f"LIMIT {limit}"):
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        
        return _execute_sql(sql, project_id, database_id)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "sql": sql
        }, indent=2)


@mcp.tool()
def search_by_filters(table: str, filters: Dict[str, Any]) -> str:
    """
    Search with advanced filtering options.
    
    Supported filters:
    - exact: {"column": "value"}
    - contains: {"column__contains": "text"}
    - greater_than: {"column__gt": 100}
    - less_than: {"column__lt": 100}
    - between: {"column__between": [1, 100]}
    - in: {"column__in": ["val1", "val2"]}
    
    Args:
        table: Table name
        filters: Dictionary of column:value filters
    
    Returns:
        JSON with matching records
    """
    try:
        # Load schema metadata to validate columns
        manager = _get_manager()
        schema_meta = manager.fetch_all_tables_meta()
        tbl_meta = schema_meta.get(table, {})
        tbl_cols = tbl_meta.get("columns", {})
        
        # Build clean-to-original column mapping
        col_mappings = {}
        for c in tbl_cols.keys():
            col_mappings[c.lower().replace("_", "").replace(" ", "")] = c
            
        sql = f'SELECT * FROM "{table}" WHERE 1=1'
        
        for orig_key, value in filters.items():
            # Extract suffix
            suffix = ""
            key = orig_key
            for s in ('__contains', '__gt', '__lt', '__in', '__between'):
                if orig_key.endswith(s):
                    key = orig_key[:-len(s)]
                    suffix = s
                    break
                    
            clean_key = key.lower().replace("_", "").replace(" ", "")
            target_col = col_mappings.get(clean_key)
            if not target_col:
                # Column doesn't exist in table schema, skip to prevent SQL column errors
                continue
                
            col_ref = f'"{table}"."{target_col}"'
            
            # Map status values
            if target_col.lower() == 'status' and isinstance(value, str):
                cleaned_val = value.lower().replace(' ', '_').replace('-', '_')
                if cleaned_val == 'in_progress':
                    sql += f" AND {col_ref} IN ('in_progress', 'in-progress')"
                elif cleaned_val == 'not_started':
                    sql += f" AND {col_ref} IN ('not_started', 'not-started')"
                else:
                    sql += f" AND {col_ref} = '{cleaned_val}'"
                continue

            if suffix == '__contains':
                sql += f" AND LOWER({col_ref}) LIKE '%{str(value).lower()}%'"
            elif suffix == '__gt':
                sql += f" AND {col_ref} > {value}"
            elif suffix == '__lt':
                sql += f" AND {col_ref} < {value}"
            elif suffix == '__in':
                vals = ','.join([f"'{v}'" for v in value])
                sql += f" AND {col_ref} IN ({vals})"
            else:
                sql += f" AND {col_ref} = '{value}'"
        
        sql += " LIMIT 1000"
        
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        
        return _execute_sql(sql, project_id, database_id)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "table": table
        }, indent=2)


@mcp.tool()
def get_schema_info(detailed: bool = False, include_all: bool = True) -> str:
    """
    Get information about available tables and their schemas, grouped by module.
    
    Args:
        detailed: Include column details and relationships
        include_all: Include empty tables (default True)
    
    Returns:
        JSON with schema information
    """
    try:
        manager = _get_manager()
        docs = manager.fetch_schema_documents()
        
        modules = {}
        for doc in docs:
            tbl_name = doc["metadata"]["table_name"]
            row_count = doc["metadata"].get("row_count", 0)
            
            if not include_all and row_count == 0:
                continue
                
            mod = get_table_module(tbl_name)
            if mod not in modules:
                modules[mod] = []
                
            tbl_info = {
                "table_name": tbl_name,
                "row_count": row_count,
                "column_count": len(doc["metadata"]["columns"]),
            }
            if detailed:
                tbl_info["columns"] = [
                    {"name": col.get("name"), "type": col.get("type"), "nullable": col.get("nullable", False)}
                    for col in doc["metadata"].get("columns", [])
                ]
                tbl_info["relationships"] = doc["metadata"].get("relationships", [])
            else:
                tbl_info["relationships"] = doc["metadata"].get("relationships", [])
                
            modules[mod].append(tbl_info)
            
        sorted_modules = {m: sorted(modules[m], key=lambda x: x["table_name"]) for m in sorted(modules.keys())}
        
        return json.dumps({
            "status": "success",
            "total_tables": sum(len(tables) for tables in sorted_modules.values()),
            "modules": sorted_modules
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


@mcp.tool()
def get_table_preview(table: str, limit: int = 10) -> str:
    """
    Get a preview of table data and structure.
    
    Args:
        table: Table name
        limit: Number of rows to preview
    
    Returns:
        JSON with sample data and structure info
    """
    try:
        sql = f"SELECT * FROM {table} LIMIT {limit}"
        project_id = os.getenv('projectId') or os.getenv('PROJECT_ID')
        database_id = os.getenv('PRISMA_DATABASE_ID') or os.getenv('DATABASE_ID')
        
        return _execute_sql(sql, project_id, database_id)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "table": table
        }, indent=2)


@mcp.tool()
def refresh_embeddings() -> str:
    """
    Re-embed all schema documents after database changes.
    
    Returns:
        JSON with refresh status
    """
    try:
        manager = _get_manager()
        print("Refreshing embeddings...", file=sys.stderr)
        manager.setup_database()
        count = manager.load_and_embed_schema()
        
        return json.dumps({
            "status": "success",
            "documents_embedded": count,
            "message": f"Successfully refreshed {count} schema embeddings"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


@mcp.tool()
def get_embedding_stats() -> str:
    """
    Get statistics about stored embeddings.
    
    Returns:
        JSON with embedding metrics
    """
    try:
        manager = _get_manager()
        count = manager.get_document_count()
        
        return json.dumps({
            "status": "success",
            "total_documents": count,
            "embedding_model": manager.model_name,
            "embedding_dimension": manager.dimension,
            "database": "prisma_postgres",
            "index_type": "IVFFlat (cosine similarity)"
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _execute_sql(sql: str, project_id: str = None, database_id: str = None) -> str:
    """Execute SQL directly via psycopg2 using the same DATABASE_URL as EmbeddingManager."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    try:
        raw_url = os.getenv("DATABASE_URL", "").strip('"')
        if not raw_url:
            raise RuntimeError(
                "DATABASE_URL is not set. Add it to your .env file."
            )

        conn = psycopg2.connect(raw_url)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(sql)
            if cursor.description:
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]
                columns = [desc[0] for desc in cursor.description]
                return json.dumps({
                    "status": "success",
                    "query": sql,
                    "results": results,
                    "columns": columns,
                    "row_count": len(results)
                }, indent=2, default=str)
            else:
                conn.commit()
                return json.dumps({
                    "status": "success",
                    "query": sql,
                    "results": [],
                    "row_count": 0,
                    "message": "Query executed (no rows returned — INSERT/UPDATE/DELETE)"
                }, indent=2)
        finally:
            cursor.close()
            conn.close()

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "query": sql
        }, indent=2)


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Comprehensive Semantic Search MCP Server...", file=sys.stderr)
    print("Available tools: semantic_search, analyze_data, execute_custom_sql, search_by_filters, get_schema_info, get_table_preview, refresh_embeddings, get_embedding_stats, smart_query", file=sys.stderr)
    sys.stdout = SafeStdout(_original_stdout, sys.stderr)
    mcp.run(transport="stdio")