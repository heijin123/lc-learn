import os
import asyncio
from typing import List, Dict, Any, Optional
import dashscope
from dotenv import load_dotenv

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# --------------------------
# 数据模型
# --------------------------
class AgentMessage:
    def __init__(self, agent_name: str, thought: str, content: str):
        self.agent_name = agent_name
        self.thought = thought  # Agent内部思考过程
        self.content = content  # 对外输出内容

    def __repr__(self):
        return f"[{self.agent_name}]\n🔍思考：{self.thought}\n📝输出：{self.content}\n"


# --------------------------
# 基础Agent基类
# --------------------------
class BaseAgent:
    agent_name: str
    system_prompt: str

    def __init__(self):
        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    async def run_stream(self, user_input: str) -> AgentMessage:
        """调用Qwen‑max，流式返回，边跑边打印思考&输出片段"""
        self.history.append({"role": "user", "content": user_input})
        response = dashscope.Generation.call(
            model="qwen-max",
            messages=self.history,
            result_format="message",
            stream=True,
            incremental_output=True
        )
        full_content = ""
        print(f"\n========== 🤖 {self.agent_name} 开始执行 ==========")
        for chunk in response:
            if chunk.status_code != 200:
                raise Exception(f"LLM调用失败:{chunk.message}")
            delta = chunk.output.choices[0].message.content
            full_content += delta
            # 流式实时打印片段
            print(delta, end="", flush=True)
        print(f"\n========== 🛑 {self.agent_name} 执行结束 ==========\n")

        self.history.append({"role": "assistant", "content": full_content})
        # 解析：约定输出格式：`<<THOUGHT>>内部思考<<CONTENT>>实际产出`
        thought, real_content = self._parse_thought_content(full_content)
        return AgentMessage(agent_name=self.agent_name, thought=thought, content=real_content)

    def _parse_thought_content(self, raw: str):
        if "<<THOUGHT>>" in raw and "<<CONTENT>>" in raw:
            parts = raw.split("<<THOUGHT>>")
            body = parts[-1]
            t, c = body.split("<<CONTENT>>")
            return t.strip(), c.strip()
        # 如果模型没有遵守格式，降级
        return "无显式思考过程", raw.strip()


# --------------------------
# 1. 规划Agent：任务拆解、输出执行计划
# --------------------------
class PlanAgent(BaseAgent):
    agent_name = "规划Agent"
    system_prompt = """
你是规划Agent。
你的任务：接收用户原始需求，拆解成清晰可执行的步骤方案。
输出严格格式：
<<THOUGHT>>
这里写你的完整思考过程：分析用户需求、识别约束、思考怎么拆分子任务、风险点
<<CONTENT>>
输出结构化执行计划：
1. 任务目标
2. 依赖条件
3. 分步执行清单
4. 预期产出物
"""


# --------------------------
# 2. 编码Agent：根据规划生成代码
# --------------------------
class CodeAgent(BaseAgent):
    agent_name = "编码Agent"
    system_prompt = """
你是编码Agent。
输入是一份任务执行规划。
你需要依据规划，生成完整可运行代码，注意依赖、异常处理、注释。
输出严格格式：
<<THOUGHT>>
你的思考：理解规划目标，选择技术方案，考虑边界，模块划分，异常点
<<CONTENT>>
输出完整代码，附带说明。
"""


# --------------------------
# 3. 审查Agent：代码评审，发现缺陷、给出修改建议
# --------------------------
class ReviewAgent(BaseAgent):
    agent_name = "审查Agent"
    system_prompt = """
你是代码审查Agent。
接收一份代码，做全面审查：逻辑正确性、边界条件、异常处理、安全漏洞、代码规范、性能问题。
输出严格格式：
<<THOUGHT>>
你的思考：逐项扫描代码，识别风险点，判断哪些问题严重，哪些是优化建议
<<CONTENT>>
审查报告：
- 风险等级：高/中/低
- 问题清单
- 修改建议
- 是否需要重新编码：true/false
"""


# --------------------------
# 4. Supervisor 调度主控 Worker模式
# --------------------------
class Supervisor:
    def __init__(self):
        self.plan_agent = PlanAgent()
        self.code_agent = CodeAgent()
        self.review_agent = ReviewAgent()
        self.records: List[AgentMessage] = []

    async def run(self, user_requirement: str):
        print("="*60)
        print(f"📌 Supervisor收到用户需求：{user_requirement}")
        print("="*60)

        # Step1：Supervisor下发任务给规划Agent
        print("\n[Supervisor] → 分发任务给【规划Agent】")
        plan_msg = await self.plan_agent.run_stream(user_requirement)
        self.records.append(plan_msg)

        # Step2：Supervisor把规划结果下发编码Agent
        print("\n[Supervisor] → 将执行计划分发【编码Agent】")
        code_msg = await self.code_agent.run_stream(plan_msg.content)
        self.records.append(code_msg)

        # Step3：Supervisor把代码下发审查Agent
        print("\n[Supervisor] → 将代码分发【审查Agent】进行评审")
        review_msg = await self.review_agent.run_stream(code_msg.content)
        self.records.append(review_msg)

        # 解析审查结果是否需要重跑编码
        need_rework = "true" in review_msg.content.lower() and "是否需要重新编码：true" in review_msg.content
        max_loop = 2
        loop_cnt = 0
        while need_rework and loop_cnt < max_loop:
            loop_cnt +=1
            print(f"\n[Supervisor] 审查发现问题，需要重新编码，第{loop_cnt}次迭代")
            # 将审查意见+原规划一起传给编码Agent
            rework_prompt = f"""
【原始执行计划】
{plan_msg.content}

【审查反馈问题】
{review_msg.content}

请基于上面，修正代码。
"""
            code_msg = await self.code_agent.run_stream(rework_prompt)
            self.records.append(code_msg)

            print("\n[Supervisor] → 重新提交审查Agent")
            review_msg = await self.review_agent.run_stream(code_msg.content)
            self.records.append(review_msg)

            need_rework = "true" in review_msg.content.lower() and "是否需要重新编码：true" in review_msg.content

        print("\n" + "="*60)
        print("✅ Supervisor：任务流程结束，汇总全部Agent记录")
        print("="*60)
        for rec in self.records:
            print(rec)

        return {
            "user_requirement": user_requirement,
            "plan": plan_msg,
            "code": code_msg,
            "review": review_msg,
            "all_records": self.records
        }


# --------------------------
# 入口
# --------------------------
async def main():
    supervisor = Supervisor()
    user_prompt = "写一个异步的任务队列，支持任务提交、状态查询、简单失败重试，使用Python"
    await supervisor.run(user_prompt)


if __name__ == "__main__":
    asyncio.run(main())
