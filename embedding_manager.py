import os
import sys
import json
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

load_dotenv()

class EmbeddingManager:
    """
    Manages vector embeddings using pgvector in Prisma Postgres.
    All storage and similarity search happens server-side via DATABASE_URL.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", connection_string: str = None):
        self.model_name = model_name
        self.model = None
        self.dimension = 768
        raw_url = connection_string or os.getenv("DATABASE_URL", "")
        self.connection_string = raw_url.strip('"')
        if not self.connection_string:
            raise ValueError(
                "DATABASE_URL is not set. "
                "Pass it directly or set it in your .env file (Prisma Postgres direct URL)."
            )

    def _get_connection(self):
        """Create and return a new database connection."""
        return psycopg2.connect(self.connection_string)

    def initialize_model(self) -> SentenceTransformer:
        """Load the BAAI/bge-base-en-v1.5 model locally (no API required)."""
        if self.model is None:
            print(f"Loading local SentenceTransformer model '{self.model_name}'...", file=sys.stderr)
            old_stdout = sys.stdout
            sys.stdout = sys.stderr
            try:
                self.model = SentenceTransformer(self.model_name)
            finally:
                sys.stdout = old_stdout
            print("Model loaded successfully.", file=sys.stderr)
        return self.model

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings locally for a list of texts.
        Returns a normalized numpy array of embeddings.
        """
        self.initialize_model()
        embeddings = self.model.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        return embeddings

    def setup_database(self):
        """Ensure the pgvector extension and the document_embeddings table exist.
        Also creates an IVFFlat index for fast approximate nearest neighbor search.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Drop any existing table to ensure dimension matches the model output
            cursor.execute("DROP TABLE IF EXISTS document_embeddings;")
            cursor.execute(f"""
                CREATE TABLE document_embeddings (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding vector({self.dimension}),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            cursor.execute("""
                SELECT 1 FROM pg_indexes 
                WHERE tablename = 'document_embeddings' 
                AND indexname = 'document_embeddings_embedding_idx';
            """)
            if cursor.fetchone() is None:
                cursor.execute("""
                    CREATE INDEX document_embeddings_embedding_idx
                    ON document_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                print("Created IVFFlat cosine similarity index on embedding column.", file=sys.stderr)
            conn.commit()
            print("Database setup complete (pgvector extension + table + index).", file=sys.stderr)
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def clear_embeddings(self):
        """Delete all rows from the document_embeddings table.
        Useful before re-generating and re-inserting fresh embeddings.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM document_embeddings;")
            conn.commit()
            deleted = cursor.rowcount
            print(f"Cleared {deleted} existing rows from document_embeddings.", file=sys.stderr)
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def store_embeddings(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict[str, Any]] = None) -> int:
        """Store texts with their embeddings and metadata in document_embeddings.
        Returns the number of rows inserted.
        """
        if metadata is None:
            metadata = [{} for _ in range(len(texts))]
        if len(texts) != len(embeddings) or len(texts) != len(metadata):
            raise ValueError("The lengths of texts, embeddings, and metadata must match.")
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            data = []
            for text, emb, meta in zip(texts, embeddings, metadata):
                emb_str = "[" + ",".join(map(str, emb.tolist())) + "]"
                meta_json = json.dumps(meta)
                data.append((text, emb_str, meta_json))
            insert_query = """
                INSERT INTO document_embeddings (content, embedding, metadata) 
                VALUES %s
            """
            execute_values(cursor, insert_query, data)
            conn.commit()
            inserted = len(data)
            print(f"Stored {inserted} document(s) in document_embeddings.", file=sys.stderr)
            return inserted
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Generate embeddings for documents and store them in Prisma Postgres.
        Each document should be a dict with key 'content' and optional 'metadata'.
        Returns the number of documents stored.
        """
        texts = [doc["content"] for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        embeddings = self.generate_embeddings(texts)
        return self.store_embeddings(texts, embeddings, metadata)

    def batch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 32) -> int:
        """Add documents in batches for memory efficiency.
        Returns total number of documents stored.
        """
        total = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} documents)...", file=sys.stderr)
            count = self.add_documents(batch)
            total += count
        return total

    def semantic_search(self, query: str, top_k: int = 5, filters: Dict[str, Any] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Perform semantic search using pgvector's cosine distance operator.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            filters: Optional dict of metadata filters, e.g. {"city": "Mumbai", "amount_gt": 50000}

        Returns:
            List of (document_dict, similarity_score) tuples, highest similarity first.
        """
        query_embedding = self.generate_embeddings([query])[0]
        emb_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SET enable_indexscan TO off;")
            cursor.execute("SET enable_bitmapscan TO off;")
            where_clauses = []
            params = []
            if filters:
                for key, value in filters.items():
                    if key.endswith("_gt"):
                        field = key[:-3]
                        where_clauses.append(f"(metadata->>'%s')::float > %s" % (field, value))
                        params.extend([field, value])
                    elif key.endswith("_lt"):
                        field = key[:-3]
                        where_clauses.append(f"(metadata->>'%s')::float < %s" % (field, value))
                        params.extend([field, value])
                    elif key.endswith("_gte"):
                        field = key[:-4]
                        where_clauses.append(f"(metadata->>'%s')::float >= %s" % (field, value))
                        params.extend([field, value])
                    elif key.endswith("_lte"):
                        field = key[:-4]
                        where_clauses.append(f"(metadata->>'%s')::float <= %s" % (field, value))
                        params.extend([field, value])
                    else:
                        where_clauses.append(f"metadata->>'%s' = %s" % (key, value))
                        params.extend([key, str(value)])
            where_sql = ""
            if where_clauses:
                where_sql = "WHERE " + " AND ".join(where_clauses)
            search_query = f"""
                SELECT 
                    id, content, metadata, created_at,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_embeddings
                {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            cursor.execute(search_query, [emb_str] + params + [emb_str, top_k])
            rows = cursor.fetchall()
            results = []
            for row in rows:
                doc = {
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2],
                    "created_at": str(row[3])
                }
                similarity = float(row[4])
                results.append((doc, similarity))
            return results
        finally:
            cursor.close()
            conn.close()

    def get_document_count(self) -> int:
        """Count total indexed documents."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM document_embeddings;")
            return cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()

    # Schema‑driven methods ---------------------------------------------------
    def load_and_embed_schema(self) -> int:
        """Fetch schema documents, generate embeddings and store them."""
        docs = self.fetch_schema_documents()
        self.clear_embeddings()
        self.write_tools_metadata(docs)
        return self.add_documents(docs)

    def fetch_schema_documents(self) -> List[Dict[str, Any]]:
        """Create document representations of all tables in the schema, including row counts and descriptions."""
        schema = self._fetch_schema_json()
        tables = schema.get("data", {}).get("tables", [])
        documents: List[Dict[str, Any]] = []
        
        # Connect to DB to fetch row counts in a single batch reuse session
        row_counts = {}
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            for tbl in tables:
                name = tbl.get("name")
                try:
                    cur.execute(f'SELECT COUNT(*) FROM "{name}";')
                    row_counts[name] = cur.fetchone()[0]
                except Exception:
                    conn.rollback()
                    row_counts[name] = 0
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to fetch table row counts from database: {e}", file=sys.stderr)
            for tbl in tables:
                row_counts[tbl.get("name")] = 0

        for tbl in tables:
            name = tbl.get("name")
            columns = tbl.get("columns", [])
            row_count = row_counts.get(name, 0)
            
            col_desc = []
            rel_desc = []
            for col in columns:
                nullable = ", nullable" if col.get("nullable") else ""
                col_desc.append(f"{col['name']} ({col['type']}{nullable})")
                if col.get("foreignKey"):
                    fk = col["foreignKey"]
                    rel_desc.append(f"{col['name']} → {fk['table']}.{fk['column']}")
            
            fk_text = ". Foreign keys: " + ", ".join(rel_desc) if rel_desc else ""
            content = f"Table `{name}` has {row_count} row(s). It has columns: " + ", ".join(col_desc) + fk_text + "."
            metadata = {
                "table_name": name,
                "columns": columns,
                "relationships": rel_desc,
                "row_count": row_count
            }
            documents.append({"content": content, "metadata": metadata})
        return documents

    def fetch_all_tables_meta(self) -> Dict[str, Dict[str, Any]]:
        """Return a detailed dictionary of all tables and their column schemas."""
        schema = self._fetch_schema_json()
        tables = schema.get("data", {}).get("tables", [])
        meta = {}
        for tbl in tables:
            name = tbl.get("name")
            columns = tbl.get("columns", [])
            meta[name] = {
                "columns": {c.get("name"): c.get("type") for c in columns},
                "foreign_keys": [
                    {
                        "column": c.get("name"),
                        "to_table": c.get("foreignKey", {}).get("table"),
                        "to_column": c.get("foreignKey", {}).get("column")
                    }
                    for c in columns if c.get("foreignKey")
                ]
            }
        return meta

    def _fetch_schema_json(self) -> Dict[str, Any]:
        """Load the introspected schema JSON.
        Tries a local 'schema_introspection.json' file; falls back to MCP generated output.
        """
        primary_path = os.path.join(os.path.dirname(__file__), "schema_introspection.json")
        if os.path.exists(primary_path):
            with open(primary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        fallback_path = os.path.join(
            os.getenv("HOME", ""), ".gemini", "antigravity-ide", "brain",
            "a2d40ca9-13dd-4283-b0c3-2bc4c7cf89a7", ".system_generated", "steps", "45", "output.txt"
        )
        if os.path.exists(fallback_path):
            with open(fallback_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise FileNotFoundError("Schema introspection JSON not found.")

    def _table_has_data(self, table_name: str) -> bool:
        """Return True if the table contains at least one row."""
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table_name} LIMIT 1);")
            return cur.fetchone()[0]
        except Exception:
            return False
        finally:
            cur.close()
            conn.close()

    def write_tools_metadata(self, documents: List[Dict[str, Any]]) -> None:
        """Write a tools_metadata.json file exposing each table as a searchable tool."""
        path = os.path.join(os.path.dirname(__file__), "tools_metadata.json")
        tools = []
        for doc in documents:
            table = doc["metadata"].get("table_name")
            tools.append({
                "name": f"search_{table}",
                "description": f"Semantic search over the schema of table '{table}'.",
                "metadata": {"table": table}
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"tools": tools}, f, indent=2)

# ─── Command Line / Testing Demonstration ────────────────────────────────────
if __name__ == "__main__":
    import io
    # Fix Windows console encoding for Unicode content from DB
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  Prisma Postgres pgvector Schema Embedding Manager")
    print("=" * 60)

    manager = EmbeddingManager()

    print("\n--- Step 1: Setting up database (pgvector + table + index) ---")
    manager.setup_database()

    print("\n--- Step 2: Embedding table schemas ---")
    try:
        count = manager.load_and_embed_schema()
        print(f"Successfully embedded and stored {count} table schemas.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n--- Step 3: Sample Semantic Search on schema ---")
    sample_query = "foreign key from sales to teams"
    results = manager.semantic_search(sample_query, top_k=3)
    for i, (doc, score) in enumerate(results):
        print(f"  Rank {i+1} [Similarity: {score:.4f}]")
        print(f"  Content: {doc['content']}")
        print(f"  Metadata: {doc['metadata']}\n")

    total = manager.get_document_count()
    print(f"--- Done! Total schema documents in pgvector: {total} ---")