from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict
import os

load_dotenv()

api_key = os.getenv("DASHSCOPE_API_KEY")
base_url = os.getenv("DASHSCOPE_API_URL_RESPONSE")


class LLM_SDK:
    def __init__(self, api_key: str,base_url: str,timeout: int = None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout or 60
        self.model = "qwen3.7-flash"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )
    def think(self, messages:list[Dict[str,str]],temperature:float = 0,stream:bool = True) -> str:
        """ 调用LLM思考
        stream=True  时逐块打印并拼接（适合交互式查看）
        stream=False 时直接返回完整文本（适合需要 json.loads 等结构化解析的场景）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=stream
            )
            # 非流式：response 是 ChatCompletion 对象，不可迭代，需单独处理
            if not stream:
                return response.choices[0].message.content

            # 流式：逐块收集并实时打印
            collected_content  = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None