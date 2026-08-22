from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.graph import MessagesState,StateGraph,START,END
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from typing import List, Any, Dict
from langchain_core.messages import AIMessage,HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
import os
import sqlite3
import json

from langgraph.graph.message import MessagesState
from openai import api_key

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_API_URL_RESPONSE")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
checkpoint = InMemorySaver()
MAX_TOOL_CALL_ROUNDS = 3


class AgentState(MessagesState):
    final_report: str 
    tool_rounds: int 

class db_search:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def search(self, table:str,columns:str,where:str,params:List[Any],limit:int=100):
        params = params or []
        sql_parts = [f"SELECT {columns} FROM {table}"]
        if where:
            sql_parts.append(f"WHERE {where}")
        sql_parts.append(f"LIMIT {limit}")
        sql = " ".join(sql_parts)
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql, params)
            rows = [dict(r) for r in cursor.fetchall()]
            return {"success": True,"data":rows}
        except Exception as e:
            return {"success": False,"data": []}

class dbquerySchema(BaseModel):
    table: str = Field(description="数据库表名")
    columns: str = Field(description="查询的列，多个列用逗号分隔")
    where: str = Field(description="查询条件")
    params: List[Any] = Field(default_factory=list,description="查询参数")
    limit: int = Field(default=10,description="返回结果数量上限")

# 基于本文件所在目录定位 person.db，避免从非 demo 目录启动时找不到数据库
db_client = db_search(os.path.join(os.path.dirname(__file__), "person.db"))
tavily_client = TavilySearch(max_results=3)

@tool("query_db",args_schema=dbquerySchema)

def db_query_tool(table: str,columns: str,where: str,params: List[Any],limit: int = 50) -> Dict[str, Any]:
    """
    查询本地数据库 person.db（表 person，字段：id/name/age/weight/height）。
    当需要获取某个具体用户（如“张三”）的基本身体数据时使用本工具。
    """
    return db_client.search(
    table= table,
    columns= columns,
    where= where,
    params= params,
    limit= limit,
    )
@tool
def search_tool(query: str) -> str:
    """
    使用Tavily搜索互联网
    """
    return tavily_client.invoke(query)

tools = [db_query_tool,search_tool]

# key 必须与工具的 name 一致：db_query_tool 经 @tool("query_db") 命名为 "query_db"
tools_map = {
    "query_db": db_query_tool,
    "search_tool": search_tool,
}

llm=ChatOpenAI(
    model="qwen-max",
    api_key=api_key,
    base_url=base_url,
    temperature=0,
    extra_body = {"enable_thinking": False}
).bind_tools(tools)

REACT_PROMPT = """
你是一名健康助手。可用工具（已通过 function calling 绑定）：
- query_db：查询本地数据库 person.db（表 person，字段 age/name/weight/height），用于获取指定用户的基本身体数据。
- search_tool：联网搜索互联网资料。

工作方式（重要）：
- 需要工具时，【直接调用对应工具】，不要把工具调用写成文本形式的 "Action: xxx"，那样不会被系统识别执行。
- 当用户问题涉及某个具体个人（如“帮张三推荐食谱”），应先用 query_db 查出该人的 age/weight/height；
  若还需额外营养/医学知识，再用 search_tool 联网搜索。
- 信息足够后不要再调用工具，直接准备生成报告。
- 最多 3 轮工具调用，避免无限循环。
- 禁止编造没有来源的事实。
"""
def react_node(state: AgentState):
    resp = llm.invoke(state["messages"]+[HumanMessage(content=REACT_PROMPT)])
    return {"messages": resp}

def tool_executor_node(state: AgentState):
    last_msg=state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        raise NodeInterrupt(f"预期AIMessage，实际：{type(last_msg).__name__}")
    tool_msg=[]

    for call in last_msg.tool_calls:
        print(f"工具调用：name={call['name']},args={call['args']}")
        if call['name'] in tools_map:
            tool = tools_map[call['name']]
            try:
                result = tool.invoke(call['args'])
            except Exception as e:
                result = f"工具调用失败：{str(e)}"
            tm = ToolMessage(
                content=result,
                tool_call_id=call['id'],
            )
        else:
            tm = ToolMessage(
                content=f"工具调用失败：{call['name']}",
                tool_call_id=call['id'],
            )
    tool_msg.append(tm)
    return {"messages": tool_msg,"tool_rounds": state["tool_rounds"] + 1}

REPORT_PROMPT = """
你是一名营养学家，基于对话和搜索的结果，生成一份健康食谱推荐 
食谱应该包含早中晚 3餐，每餐内容包含以下内容：
1.用户的基本信息
2. 食谱名称
3. 食谱描述
4. 食谱成分
5. 食谱制作步骤
6. 为什么推荐这个食谱
"""

def generate_report_node(state: AgentState):
    resp = llm.invoke(state["messages"] + [HumanMessage(content=REPORT_PROMPT)])
    return {"final_report": resp.content}

def route_after_reasoning(state: AgentState):
    last_msg = state["messages"][-1]
    if state["tool_rounds"] >= MAX_TOOL_CALL_ROUNDS:
        return "generate_report"

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "call_tool"
    return "generate_report"

def build_graph():
    graph = StateGraph(AgentState)
    
    graph.add_node("react", react_node)
    graph.add_node("call_tool", tool_executor_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "react")
    graph.add_conditional_edges(
        source= "react",
        path=route_after_reasoning,
        path_map={
            "call_tool": "call_tool",
            "generate_report": "generate_report"
        }
    )
    graph.add_edge("call_tool", "react")
    graph.add_edge("generate_report", END)
    return graph.compile(checkpointer=checkpoint)

if __name__ == "__main__":
    agent_graph = build_graph()
    config ={"configurable": {"thread_id": "1"}}

    response = agent_graph.invoke(
        input={"messages": [HumanMessage(content="帮张三推荐一份当天健康食谱")],"final_report":"","tool_rounds":0},
        config=config
    )
    print(response["messages"][-1].content)
    response2 = agent_graph.invoke(
    input={"messages": [HumanMessage(content="把晚餐换一下，我更喜欢牛肉类型的食物")],"final_report":""},
    config=config
    )
    print(response2["final_report"])

    for i, msg in enumerate(response2["messages"]):
        print(f"\n[{i}] {type(msg).__name__}")
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"👉 tool_calls: {msg.tool_calls}")
        preview = msg.content[:250].replace("\n", " ")
        print(f"content: {preview} ...")
