import os
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 先加载 demo/.env，再 import agent：agent 模块在导入时会读取环境变量（API key 等），
# 必须在 import 之前完成加载，否则从非 demo 目录启动时读不到配置
load_dotenv(Path(__file__).parent / ".env")

import agent  # noqa: E402（导入后执行模块级代码：创建 db_client / tavily_client / llm）

app = FastAPI(title="Agent对话服务")

# 允许跨域，前端页面访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时构建一次 agent 图（带 InMemorySaver checkpointer），后续请求复用
# 多轮对话靠 thread_id 在 checkpointer 中维持历史，无需前端传 history
agent_graph = agent.build_graph()


# 请求体结构
class ChatRequest(BaseModel):
    message: str
    # 保留 history 字段兼容前端，但实际历史由 graph 的 checkpointer 管理
    history: list[dict] = Field(default_factory=list)
    # 多轮对话会话标识；首次为空，后端生成后返回，前端后续请求携带
    thread_id: str | None = None


BASE_DIR = Path(__file__).parent


@app.post("/api/chat")
def chat(req: ChatRequest):
    # 首次请求没有 thread_id，生成一个新会话 id
    tid = req.thread_id or str(uuid.uuid4())

    # 调用 agent 图：把用户消息塞进 messages，重置 final_report / tool_rounds
    # checkpointer 会按 thread_id 加载该会话历史，实现多轮上下文
    result = agent_graph.invoke(
        {"messages": [HumanMessage(content=req.message)],
         "final_report": "", "tool_rounds": 0},
        config={"configurable": {"thread_id": tid}},
    )

    # 优先取最终报告，没有则取最后一条消息内容
    reply = result.get("final_report") or ""
    if not reply:
        last_msg = result["messages"][-1]
        reply = getattr(last_msg, "content", "") or ""

    return {"reply": reply, "thread_id": tid}


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
