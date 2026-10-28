from langchain.agents import create_agent
from langchain_core.messages import AIMessageChunk
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    @staticmethod
    def _text_of(chunk: AIMessageChunk) -> str:
        """取出消息块中的文本，兼容字符串与内容块两种格式。"""
        content = chunk.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    def execute_stream(self, query: str):
        """
        逐 token 输出最终回答。

        产出事件：(kind, text)
        - ("answer", 文本)：回答片段，可直接拼接并展示
        - ("reset", "")：当前轮其实是要调用工具，之前乐观输出的思考文字需要撤回
        """
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        current_step = None
        step_calls_tools = False

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk, metadata in self.agent.stream(input_dict, stream_mode="messages", context={"report": False}):
            # 只要模型节点的输出，工具节点直接跳过
            if metadata.get("langgraph_node") != "model" or not isinstance(chunk, AIMessageChunk):
                continue

            # 每一轮模型调用单独判断：这一轮到底是在思考调工具，还是在给出最终回答
            step = metadata.get("langgraph_step")
            if step != current_step:
                current_step = step
                step_calls_tools = False

            if chunk.tool_call_chunks:
                if not step_calls_tools:
                    step_calls_tools = True
                    # 这一轮会调用工具，之前已经输出的文字属于思考过程，需要撤回
                    yield ("reset", "")
                continue

            if step_calls_tools:
                # 工具调用轮的后续文字同样是内部思考，不展示给用户
                continue

            text = self._text_of(chunk)
            if text:
                yield ("answer", text)


if __name__ == '__main__':
    agent = ReactAgent()

    full = ""
    for kind, text in agent.execute_stream("给我生成我的使用报告"):
        if kind == "reset":
            full = ""
        else:
            full += text
        print(text, end="", flush=True)
    print()
