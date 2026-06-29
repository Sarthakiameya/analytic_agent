import asyncio
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from skill_loader import load_system_prompt
from mcp_client import MCPSubprocessClient, build_langchain_tools

logger = logging.getLogger(__name__)

_agent = None          # CompiledStateGraph
_tools_list: list = []


# ── Build agent ───────────────────────────────────────────────────────────────

def build_agent(clients: list[MCPSubprocessClient]):
    """Build and cache the LangGraph ReAct agent."""
    global _agent, _tools_list

    # ── System prompt ─────────────────────────────────────────────────────────
    system_prompt_text = load_system_prompt()
    logger.info(f"System prompt loaded: {len(system_prompt_text)} characters")

    # ── LLM ───────────────────────────────────────────────────────────────────
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    known_models = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"}
    if model_name not in known_models:
        logger.warning(f"Unknown model '{model_name}' → using gpt-4o-mini")
        model_name = "gpt-4o-mini"

    llm = ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        streaming=True,
    )
    logger.info(f"LLM: {model_name}")

    # ── Tools ─────────────────────────────────────────────────────────────────
    tools = build_langchain_tools(clients)
    _tools_list = tools
    if not tools:
        logger.warning("No MCP tools loaded — agent will have no tools!")
    else:
        logger.info(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")

    # ── Create ReAct agent (langgraph 1.x) ───────────────────────────────────
    _agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content=system_prompt_text),
    )

    logger.info("LangGraph ReAct agent ready")
    return _agent


def get_agent():
    return _agent


def get_tools() -> list:
    return _tools_list


# ── Streaming invocation ──────────────────────────────────────────────────────

async def run_agent_stream(message: str, chat_history: list, queue: asyncio.Queue) -> None:
    """
    Stream agent events into an asyncio.Queue using astream_events v2.
    FastAPI SSE endpoint drains the queue and forwards to browser.

    Events pushed to queue:
      {"type": "token",        "content": "<chunk>"}
      {"type": "tool_start",  "tool": "semantic_search"}
      {"type": "tool_end",    "tool": "semantic_search", "output": "..."}
      {"type": "final_output","output": "<full reply>"}
      {"type": "error",       "content": "<error>"}
      {"type": "done"}
    """
    agent = _agent
    if agent is None:
        await queue.put({"type": "error", "content": "Agent not initialized"})
        await queue.put({"type": "done"})
        return

    # Build message list for langgraph (HumanMessage + history)
    from langchain_core.messages import HumanMessage
    messages = list(chat_history) + [HumanMessage(content=message)]

    try:
        async for event in agent.astream_events(
            {"messages": messages},
            version="v2",
        ):
            kind: str = event["event"]

            # ── Token stream from LLM ──────────────────────────────────────
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk is None:
                    continue
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    await queue.put({"type": "token", "content": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text", "")
                            if text:
                                await queue.put({"type": "token", "content": text})

            # ── Tool call starts ───────────────────────────────────────────
            elif kind == "on_tool_start":
                tool_name = event.get("name", "tool")
                inp = event["data"].get("input", "")
                await queue.put({
                    "type": "tool_start",
                    "tool": tool_name,
                    "input": str(inp)[:200],
                })

            # ── Tool call ends ─────────────────────────────────────────────
            elif kind == "on_tool_end":
                tool_name = event.get("name", "tool")
                output = str(event["data"].get("output", ""))[:400]
                await queue.put({"type": "tool_end", "tool": tool_name, "output": output})

            # ── Graph node finished (agent done) ───────────────────────────
            elif kind == "on_chain_end":
                output_data = event["data"].get("output")
                # LangGraph returns {"messages": [...]} at the top level
                if isinstance(output_data, dict) and "messages" in output_data:
                    msgs = output_data["messages"]
                    if msgs:
                        last = msgs[-1]
                        # Only emit final_output for actual AI text responses,
                        # NOT for ToolMessage/intermediate tool outputs.
                        from langchain_core.messages import AIMessage as _AI
                        if isinstance(last, _AI):
                            final_text = getattr(last, "content", "") or ""
                            if final_text and isinstance(final_text, str):
                                await queue.put({"type": "final_output", "output": final_text})

    except Exception as e:
        logger.exception("Agent streaming error")
        await queue.put({"type": "error", "content": str(e)})
    finally:
        await queue.put({"type": "done"})


# ── Chart path extraction ─────────────────────────────────────────────────────

CHART_DIR = Path(r"C:\Users\sarthak\OneDrive\Desktop\agent_analytics\chart_outputs")


def extract_chart_paths(text: str) -> list[str]:
    """Extract .html chart file paths from agent output text."""
    paths = []

    # Pattern 1: explicit CHART_PATH: marker
    for m in re.finditer(r"CHART_PATH:\s*(.+?)(?:\n|$)", text):
        p = m.group(1).strip().strip('"\'')
        if p and p not in paths:
            paths.append(p)

    # Pattern 2: file_path in JSON output
    for m in re.finditer(r'"file_path"\s*:\s*"([^"]+\.html)"', text):
        p = m.group(1).strip()
        if p and p not in paths:
            paths.append(p)

    # Pattern 3: raw Windows .html paths
    for m in re.finditer(r'[A-Za-z][:\\/][^\s"\'<>|?*\n]+\.html', text):
        p = m.group(0).strip()
        if p and p not in paths:
            paths.append(p)

    return paths


def resolve_chart_filenames(chart_paths: list[str]) -> list[str]:
    """Return filenames of charts that actually exist on disk, copying from Plotly dir if needed."""
    filenames = []
    plotly_dir = Path(r"C:\Users\sarthak\OneDrive\Desktop\plotlyserver")
    agent_analytics_dir = Path(r"C:\Users\sarthak\OneDrive\Desktop\agent_analytics")

    for cp in chart_paths:
        p = Path(cp)
        fname = p.name
        
        # 1. Check if it already exists in the target CHART_DIR
        target_path = CHART_DIR / fname
        if target_path.exists():
            if fname not in filenames:
                filenames.append(fname)
            continue
            
        # 2. Check if it exists in the Plotly directory
        source_path = plotly_dir / fname
        if source_path.exists():
            try:
                shutil.copy2(source_path, target_path)
                logger.info(f"Copied chart {fname} from plotly server to {CHART_DIR}")
                if fname not in filenames:
                    filenames.append(fname)
                continue
            except Exception as e:
                logger.error(f"Failed to copy chart {fname}: {e}")

        # 3. Check if the absolute/relative path exists directly
        if p.exists() and p.is_file():
            try:
                shutil.copy2(p, target_path)
                logger.info(f"Copied chart {fname} from {p} to {CHART_DIR}")
                if fname not in filenames:
                    filenames.append(fname)
                continue
            except Exception as e:
                logger.error(f"Failed to copy chart {fname} from {p}: {e}")
                
        # 4. Check if it is in agent_analytics root
        root_source = agent_analytics_dir / fname
        if root_source.exists():
            try:
                shutil.copy2(root_source, target_path)
                logger.info(f"Copied chart {fname} from agent_analytics root to {CHART_DIR}")
                if fname not in filenames:
                    filenames.append(fname)
                continue
            except Exception as e:
                logger.error(f"Failed to copy chart {fname} from root: {e}")

    return filenames
