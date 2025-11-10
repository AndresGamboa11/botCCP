# app/mcp_server.py
"""
MCP "ligero" expuesto por HTTP:
- POST /mcp/tool  { "name": "consulta_ccp", "input": "texto" }
  → ejecuta herramientas del bot (por ahora: consulta_ccp = RAG)
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from app.rag import answer_with_rag

router = APIRouter(prefix="/mcp", tags=["mcp"])

class MCPRequest(BaseModel):
    name: str
    input: str

@router.post("/tool")
async def invoke_tool(req: MCPRequest):
    if req.name == "consulta_ccp":
        resp = await answer_with_rag(req.input)
        return {"tool": req.name, "ok": True, "result": resp}
    raise HTTPException(status_code=404, detail=f"Herramienta desconocida: {req.name}")
@mcp.get("/mcp/health")
async def mcp_health():
    return {"mcp": "ok"}

@mcp.post("/mcp/echo")
async def mcp_echo(req: Request):
    data = await req.json()
    return {"ok": True, "echo": data}