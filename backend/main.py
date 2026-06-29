import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load Agent_Sarthak .env first (OPENAI_API_KEY)
_root_env = Path(__file__).parent.parent / ".env"
if _root_env.exists():
    load_dotenv(_root_env, override=True)

# Load agent_analytics .env (DB credentials)
_analytics_env = Path(r"C:\Users\ \OneDrive\Desktop\agent_analytics\.env")
if _analytics_env.exists():
    load_dotenv(_analytics_env, override=False)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mcp_client import start_all_mcp_servers, stop_all_mcp_servers, get_clients
from agent import (
    build_agent, get_agent, get_tools,
    run_agent_stream, extract_chart_paths, resolve_chart_filenames,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Analytics Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHART_DIR = Path(r"C:\Users\ \OneDrive\Desktop\agent_analytics\chart_outputs")
CHART_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(CHART_DIR)), name="charts")


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=== Analytics Agent starting ===")
    clients = await start_all_mcp_servers()
    if not clients:
        logger.warning("No MCP servers started — check server paths and venvs")
    build_agent(clients)
    logger.info(f"=== Agent ready with {len(get_tools())} tools ===")


@app.on_event("shutdown")
async def shutdown_event():
    await stop_all_mcp_servers()


from typing import Any

def is_retrieval_requested(message: str) -> bool:
    msg_lower = message.lower()
    keywords = ["list", "table", "retrieve", "show records", "show data", "get data", "print", "display data", "database rows", "sql", "query", "details of"]
    return any(kw in msg_lower for kw in keywords)

def format_raw_json_fallback(text: str) -> str:
    """Detect raw JSON data in the text and format it into a clean Markdown table/description."""
    text_stripped = text.strip()
    if (text_stripped.startswith("{") and text_stripped.endswith("}")) or (text_stripped.startswith("[") and text_stripped.endswith("]")):
        try:
            data = json.loads(text_stripped)
            return _format_parsed_data(data)
        except Exception:
            pass

    import re
    # Match code blocks with json or raw
    for m in re.finditer(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL):
        try:
            block = m.group(1).strip()
            data = json.loads(block)
            formatted = _format_parsed_data(data)
            text = text.replace(m.group(0), formatted)
        except Exception:
            pass

    # Match raw JSON-like patterns in the text
    if "{" in text or "[" in text:
        for m in re.finditer(r"(\{.*?\}|\[.*?\])", text, re.DOTALL):
            try:
                candidate = m.group(1).strip()
                data = json.loads(candidate)
                if isinstance(data, (dict, list)) and len(data) > 0:
                    formatted = _format_parsed_data(data)
                    text = text.replace(m.group(0), formatted)
            except Exception:
                pass
    return text

def _format_parsed_data(data: Any) -> str:
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data) and len(data) > 0:
            return _dict_list_to_markdown_table(data)
        return ", ".join(map(str, data))

    if isinstance(data, dict):
        for key in ["data", "rows", "results", "records"]:
            val = data.get(key)
            if isinstance(val, list) and all(isinstance(item, dict) for item in val) and len(val) > 0:
                return _dict_list_to_markdown_table(val)
        
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"- **{k}**: {json.dumps(v)}")
            else:
                lines.append(f"- **{k}**: {v}")
        return "\n".join(lines)
    return str(data)

def _is_id_column(col_name: str) -> bool:
    name_lower = col_name.lower()
    return name_lower == "id" or name_lower.endswith("id") or name_lower.startswith("id_") or "_id_" in name_lower or "assignedby" in name_lower or "assignedto" in name_lower

def _dict_list_to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    original_len = len(rows)
    rows = rows[:15]
    
    all_headers = list(rows[0].keys())
    headers = [h for h in all_headers if not _is_id_column(h)]
    if not headers:
        headers = all_headers
        
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = []
    for r in rows:
        row_cells = []
        for h in headers:
            val = r.get(h, "")
            if isinstance(val, str):
                val = val.replace("\n", " ")
            row_cells.append(str(val))
        row_lines.append("| " + " | ".join(row_cells) + " |")
    table_str = "\n".join([header_line, separator_line] + row_lines)
    if original_len > 15:
        table_str += "\n\n*(Showing top 15 rows)*"
    return table_str


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent_ready": get_agent() is not None,
    }


@app.get("/tools")
async def tools():
    tool_names = [tool.name for tool in get_tools()]
    return {
        "tool_count": len(tool_names),
        "tools": tool_names,
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    SSE endpoint — streams events as JSON lines.

    Event types:
      {"type": "tool_start",   "tool": "semantic_search", "input": "..."}
      {"type": "tool_end",     "output": "..."}
      {"type": "token",        "content": "word"}
      {"type": "chart",        "filename": "chart_xxx.html"}
      {"type": "final_output", "output": "full text"}
      {"type": "error",        "content": "message"}
      {"type": "done"}
    """
    if get_agent() is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")

    # ── Guardrail check: schema queries are prohibited ──
    def is_schema_query(message: str) -> bool:
        msg_lower = message.lower()
        keywords = [
            "schema", "table structure", "database structure", "list tables",
            "show tables", "get tables", "what tables", "available tables", "delete a",
            "foreign key", "primary key", "database schema", "table relations", "delete rows",
            "columns in", "fields in", "data types in", "describe table" , "delete schema","delete columns"
        ] 
        return any(kw in msg_lower for kw in keywords)
 
    if is_schema_query(request.message):
        async def schema_block_generator():
            msg = "I am sorry, but I am not authorized to share database schema, table structure, or metadata details. Please ask about retrieving data or creating visualizations!"
            for word in msg.split(" "):
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            schema_block_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Query-relatedness guard ────────────────────────────────────────────────
    # Detect whether the new query references prior context via pronouns/words.
    # If there are NO back-references, treat this as an entirely fresh query and
    # send an empty history so the agent starts with a clean slate. 
    _BACK_REFS = (
        r'\b(his|her|their|its|those|these|that|same|also|too|again|'  
        r'previously|further|additionally|more|another|above|prior|'  
        r'them|they|he|she|it|which|who|the\s+same|like\s+before|'  
        r'from\s+last\s+query|as\s+before|related|continue|followup|follow.?up)\b'
    )
    _has_back_ref = bool(re.search(_BACK_REFS, request.message.lower()))

    # Build LangChain chat history (cap to last 6 messages = 3 turns)
    from langchain_core.messages import HumanMessage, AIMessage
    chat_history = []
    if _has_back_ref and request.history:
        trimmed = request.history[-6:]   # Keep at most last 3 turns
        for msg in trimmed:
            role = msg.get("role", "human")
            content = msg.get("content", "")
            if role in ("human", "user"):
                chat_history.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                chat_history.append(AIMessage(content=content))
    # If no back-reference → history stays empty → agent sees only current query   

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue()
        accumulated_output = []
        # Buffer — only the LAST final_output text will be emitted to the frontend.
        # This prevents raw intermediate tool outputs (tables, JSON, etc.) from
        # being shown in the chat UI.
        buffered_final_text: str | None = None

        # Start agent in background task
        agent_task = asyncio.create_task(
            run_agent_stream(request.message, chat_history, queue)
        )

        try:
            deadline = 180.0        # total timeout
            elapsed  = 0.0
            heartbeat_interval = 5  # seconds between heartbeats

            while elapsed < deadline:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
                    elapsed = 0.0   # reset on any real event
                except asyncio.TimeoutError:
                    elapsed += heartbeat_interval
                    # Send a keep-alive comment so the browser doesn't drop the connection
                    yield ": heartbeat\n\n"
                    continue

                event_type = event.get("type")

                if event_type == "token":
                    accumulated_output.append(event["content"])
                    yield f"data: {json.dumps(event)}\n\n"

                elif event_type in ("tool_start", "tool_end"):
                    yield f"data: {json.dumps(event)}\n\n"

                elif event_type == "final_output":
                    # Buffer only — do NOT emit yet.
                    # Keep only the latest final_output text (overwrites any previous one).
                    candidate = event.get("output", "")
                    if candidate:
                        buffered_final_text = candidate

                elif event_type == "agent_finish":
                    # Also check for charts in agent_finish output
                    finish_text = event.get("output", "")
                    if finish_text:
                        chart_paths = extract_chart_paths(finish_text)
                        chart_filenames = resolve_chart_filenames(chart_paths)
                        for fname in chart_filenames:
                            yield f"data: {json.dumps({'type': 'chart', 'filename': fname})}\n\n"

                elif event_type == "error":
                    yield f"data: {json.dumps(event)}\n\n"

                elif event_type == "done":
                    # Now process and emit the single buffered final_output
                    full_text = buffered_final_text
                    if not full_text and accumulated_output:
                        full_text = "".join(accumulated_output)

                    if full_text:
                        chart_paths = extract_chart_paths(full_text)
                        chart_filenames = resolve_chart_filenames(chart_paths)

                        # Emit chart events first
                        for fname in chart_filenames:
                            yield f"data: {json.dumps({'type': 'chart', 'filename': fname})}\n\n"

                        # Strip CHART_PATH markers from the text
                        clean = full_text
                        for cp in chart_paths:
                            clean = clean.replace(f"CHART_PATH:{cp}", "").replace(f"CHART_PATH: {cp}", "")

                        if chart_filenames and not is_retrieval_requested(request.message):
                            # Visual-only: strip any remaining markdown tables or JSON blocks
                            import re
                            clean = re.sub(r"```(?:json)?\s*.*?\s*```", "", clean, flags=re.DOTALL)
                            lines = [line for line in clean.split("\n") if not line.strip().startswith("|")]
                            clean = "\n".join(lines)
                        else:
                            clean = format_raw_json_fallback(clean)

                        clean = clean.strip()
                        if clean:
                            yield f"data: {json.dumps({'type': 'final_output', 'output': clean})}\n\n"

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break

        except Exception as e:
            logger.exception("SSE generator error")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        else:
            # while loop exhausted deadline without a 'break' (done event)
            if elapsed >= deadline:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Agent timeout after 3 minutes'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            agent_task.cancel()


    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
