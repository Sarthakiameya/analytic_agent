import asyncio
import json
import os
import sys
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool
from pydantic import create_model, Field

try:
    from mcp.types import LATEST_PROTOCOL_VERSION
except Exception:  # pragma: no cover
    LATEST_PROTOCOL_VERSION = "2024-11-05"

logger = logging.getLogger(__name__)

# ── Server configurations ──────────────────────────────────────────────────────

AGENT_ANALYTICS_DIR = Path(r"C:\Users\sarthak\OneDrive\Desktop\agent_analytics")
PLOTLY_DIR          = Path(r"C:\Users\sarthak\OneDrive\Desktop\plotlyserver")

# Python executables inside each project's venv
ANALYTICS_PYTHON = str(AGENT_ANALYTICS_DIR / "venv" / "Scripts" / "python.exe")
PLOTLY_PYTHON    = str(PLOTLY_DIR / ".venv" / "Scripts" / "python.exe")

# Fall back to system python if venv doesn't exist
if not Path(ANALYTICS_PYTHON).exists():
    ANALYTICS_PYTHON = sys.executable
if not Path(PLOTLY_PYTHON).exists():
    PLOTLY_PYTHON = sys.executable

MCP_SERVER_CONFIGS = [
    {
        "name": "mcp-server-semantic-search",
        "cmd": [ANALYTICS_PYTHON, "semantic_search_mcp.py"],
        "cwd": str(AGENT_ANALYTICS_DIR),
        "env_extras": {
            "DATABASE_URL": os.getenv("DATABASE_URL", ""),
            "DIRECT_URL": os.getenv("DIRECT_URL", ""),
            "PRISMA_API_KEY": os.getenv("PRISMA_API_KEY", ""),
            "PRISMA_DATABASE_ID": os.getenv("PRISMA_DATABASE_ID", ""),
            "projectId": os.getenv("projectId", ""),
        },
    },
    {
        "name": "mcp-server-plotly",
        "cmd": [PLOTLY_PYTHON, "server.py"],
        "cwd": str(PLOTLY_DIR),
        "env_extras": {},
    },
]

# Prisma remote MCP — handled separately via npx
PRISMA_MCP_CONFIG = {
    "name": "mcp-server-prisma",
    "cmd": ["npx.cmd" if os.name == "nt" else "npx", "-y", "mcp-remote", "https://mcp.prisma.io/mcp"],
    "cwd": str(AGENT_ANALYTICS_DIR),
    "env_extras": {
        "PRISMA_API_KEY": os.getenv("PRISMA_API_KEY", ""),
    },
}


# ── MCP subprocess client ──────────────────────────────────────────────────────

class MCPSubprocessClient:
    """Manages a single MCP server subprocess with JSON-RPC over stdio."""

    def __init__(self, name: str, cmd: list[str], cwd: str, env_extras: dict):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.env_extras = env_extras
        self.process: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._req_id = 0
        self.tools: list[dict] = []
        self._ready = False

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def start(self) -> bool:
        """Start the subprocess and perform MCP initialization."""
        env = {**os.environ, **self.env_extras}

        # Load .env from agent_analytics
        env_file = AGENT_ANALYTICS_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in env:
                        env[k] = v

        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=env,
            )
            if self.process.stdout:
                self.process.stdout._limit = 10 * 1024 * 1024
            logger.info(f"[{self.name}] subprocess started (pid={self.process.pid})")

            await asyncio.sleep(0.2)
            if self.process.returncode is not None:
                stderr_output = ""
                if self.process.stderr is not None:
                    try:
                        stderr_output = (await asyncio.wait_for(self.process.stderr.read(4096), timeout=2.0)).decode(errors="replace")
                    except Exception:
                        stderr_output = ""
                logger.error(f"[{self.name}] exited immediately with code {self.process.returncode}: {stderr_output}")
                return False

            # MCP initialize handshake
            await self._send({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "analytics-agent", "version": "1.0"},
                },
            })
            init_resp = await self._read_response()
            if not init_resp or "error" in init_resp:
                logger.error(f"[{self.name}] initialization failed: {init_resp}")
                return False

            # Send initialized notification
            await self._send({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            })

            # List tools
            await self._send({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            })
            tools_resp = await self._read_response()
            if tools_resp and "result" in tools_resp:
                self.tools = tools_resp["result"].get("tools", [])
                logger.info(f"[{self.name}] {len(self.tools)} tools: {[t['name'] for t in self.tools]}")

            self._ready = True
            return True

        except Exception as e:
            logger.error(f"[{self.name}] failed to start: {e}")
            return False

    async def _send(self, payload: dict) -> None:
        """Write a JSON-RPC message to the subprocess stdin."""
        if not self.process or self.process.stdin is None:
            raise RuntimeError(f"[{self.name}] process not running")
        data = json.dumps(payload) + "\n"
        self.process.stdin.write(data.encode())
        await self.process.stdin.drain()

    async def _read_response(self, timeout: float = 60.0) -> Optional[dict]:
        """Read the next JSON-RPC response from stdout."""
        if not self.process or self.process.stdout is None:
            return None
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=timeout)
            if not line:
                return None
            text = line.decode(errors="replace").strip()
            if not text:
                return await self._read_response(timeout)
            return json.loads(text)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.name}] response timeout")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"[{self.name}] bad JSON: {e} — line: {text[:200]}")
            return await self._read_response(timeout)

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Invoke a tool and return its text result."""
        async with self._lock:
            try:
                req_id = self._next_id()
                await self._send({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                })
                resp = await self._read_response(timeout=120.0)
                if resp is None:
                    return json.dumps({"error": "Didn't got response from MCP servers", "tool": tool_name})
                if "error" in resp:
                    return json.dumps({"error": resp["error"], "tool": tool_name})
                result = resp.get("result", {})
                content = result.get("content", [])
                if isinstance(content, list):
                    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                    return "\n".join(texts)
                return json.dumps(result)
            except Exception as e:
                logger.error(f"[{self.name}] tool call {tool_name} error: {e}")
                return json.dumps({"error": str(e), "tool": tool_name})

    async def stop(self):
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except Exception:
                self.process.kill()


# ── Tool builder ───────────────────────────────────────────────────────────────

def _build_pydantic_model(tool_def: dict):
    """Build a pydantic model from MCP tool inputSchema for StructuredTool."""
    schema = tool_def.get("inputSchema", {})
    properties = schema.get("properties", {})
    required = schema.get("required", [])

    fields = {}
    for prop_name, prop_info in properties.items():
        description = prop_info.get("description", "")
        prop_type = prop_info.get("type", "string")

        # Map JSON Schema types to Python types
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        py_type = type_map.get(prop_type, Any)

        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=description))
        else:
            default = prop_info.get("default", None)
            fields[prop_name] = (Optional[py_type], Field(default=default, description=description))

    if not fields:
        fields["query"] = (Optional[str], Field(default=None, description="Query string"))

    return create_model(f"{tool_def['name']}_Input", **fields)


def build_langchain_tools(clients: list["MCPSubprocessClient"]) -> list[StructuredTool]:
    """Convert all MCP tool definitions into LangChain StructuredTools."""
    langchain_tools = []
    for client in clients:
        if not client._ready:
            logger.warning(f"[{client.name}] not ready — skipping tools")
            continue

        for tool_def in client.tools:
            tool_name = tool_def["name"]
            tool_description = tool_def.get("description", f"Tool: {tool_name}")
            input_model = _build_pydantic_model(tool_def)

            # Capture variables for closure
            _client = client
            _tool_name = tool_name

            def make_sync_fn(c, t):
                def sync_fn(**kwargs) -> str:
                    """Synchronous wrapper that runs the async call_tool in event loop."""
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = None

                    if loop and loop.is_running():
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            future = pool.submit(asyncio.run, c.call_tool(t, args))
                            return future.result(timeout=120)
                    else:
                        new_loop = asyncio.new_event_loop()
                        try:
                            return new_loop.run_until_complete(c.call_tool(t, args))
                        finally:
                            new_loop.close()
                return sync_fn

            def make_async_fn(c, t):
                async def async_fn(**kwargs) -> str:
                    """Asynchronous wrapper that directly calls call_tool."""
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    return await c.call_tool(t, args)
                return async_fn

            lc_tool = StructuredTool(
                name=tool_name,
                description=f"[{client.name}] {tool_description}",
                args_schema=input_model,
                func=make_sync_fn(_client, tool_name),
                coroutine=make_async_fn(_client, tool_name),
            )
            langchain_tools.append(lc_tool)

    return langchain_tools


# ── Global client registry ─────────────────────────────────────────────────────

_clients: list[MCPSubprocessClient] = []


async def start_all_mcp_servers() -> list[MCPSubprocessClient]:
    """Start all MCP servers and return client list."""
    global _clients

    configs = list(MCP_SERVER_CONFIGS)
    # Add Prisma MCP if API key is set
    if os.getenv("PRISMA_API_KEY"):
        configs.append(PRISMA_MCP_CONFIG)

    clients = []
    for cfg in configs:
        client = MCPSubprocessClient(
            name=cfg["name"],
            cmd=cfg["cmd"],
            cwd=cfg["cwd"],
            env_extras=cfg.get("env_extras", {}),
        )
        ok = await client.start()
        if ok:
            clients.append(client)
            logger.info(f"✓ {cfg['name']} ready")
        else:
            logger.warning(f"✗ {cfg['name']} failed to start — skipping")

    _clients = clients
    return clients


async def stop_all_mcp_servers():
    """Gracefully stop all MCP server subprocesses."""
    for client in _clients:
        await client.stop()
    _clients.clear()


def get_clients() -> list[MCPSubprocessClient]:
    return _clients
