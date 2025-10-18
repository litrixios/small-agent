# 导入所需的库
from ..wrapper import toolwrapper  # 从自定义的包装器模块导入 toolwrapper 装饰器
from openai import OpenAI  # 导入 OpenAI 官方库
import pyperclip  # 导入 pyperclip 库，用于与系统剪贴板交互
from typing import Optional  # 导入 Optional 类型提示

# 定义一个基础的系统提示（Prompt），它会预设给模型，以指导其行为
# 注意：这里的内容被翻译成了中文，因为它直接影响模型的输出语言和风格
BASIC_PROMPT = """你是一个乐于助人的助手。这里有一个关于用户事件的观察，另一个助手正在请求你帮助解决一个问题。请帮助他们解决问题并提供一个解决方案。你输出的内容不应包含任何多余的信息，只输出用户需要的内容。
如果助手要求你生成代码，你应该只生成代码，不包含任何其他文字。如果助手要求你写一些东西，你应该只写那些东西，不包含任何其他文字。
{infos}
"""


@toolwrapper(name="chat", visible=True)
async def chat(messages: str = "Who are you?",
               api_key: str = "",
               model: Optional[str] = "deepseek-chat",
               base_url: Optional[str] = None):
    """
    中文注释：
    此函数用于与 OpenAI API 或兼容的 GPT 模型进行对话。
    函数会将模型的响应内容复制到剪贴板，并返回一个表示操作是否成功的状态。
    如果浏览器是备选方案，请尽可能使用浏览器。

    Args:
        messages (str, optional): 发送给 API 的提示或问题。默认为 "Who are you?"。
        api_key (str, optional): OpenAI API 的密钥。作为工具使用时，此项通常由外部统一管理。
        base_url (str, optional): OpenAI API 的基础 URL。作为工具使用时，此项通常由外部统一管理。

    Returns:
        dict: 一个包含函数执行状态的字典，例如 {'status': 'success'}。
    """
    # 在控制台打印传入的消息，方便调试
    print('====>', messages)

    # 初始化 OpenAI 客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 调用 API 的 chat completions 接口创建对话
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": BASIC_PROMPT.format(infos=messages)}]
    )

    # 使用 try-except 块来处理可能发生的异常
    try:
        # 提取模型返回的文本内容并复制到系统剪贴板
        pyperclip.copy(response.choices[0].message.content)
        # 如果成功，返回成功状态
        return {'status': 'success'}
    except Exception as e:
        # 如果在复制过程中发生错误，打印错误信息
        print(str(e))
        # 并返回错误状态及错误信息
        return {'status': 'error', 'message': str(e)}


# 这是一个标准的 Python 写法，确保只有当该文件被直接执行时，下面的代码块才会运行
if __name__ == "__main__":
    import asyncio

    # 使用 asyncio.run 来异步执行 chat 函数进行测试
    asyncio.run(chat("Who are you?"))