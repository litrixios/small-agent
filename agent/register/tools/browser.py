# 导入所需的库
import webbrowser
from typing import Literal
from urllib.parse import quote


# from ..wrapper import toolwrapper

# 假设的 toolwrapper，用于本地测试
def toolwrapper(name, visible):
    def decorator(func):
        return func

    return decorator


# ▼▼▼ 标准库导入 ▼▼▼
import tkinter as tk
from tkinter import messagebox
import sys
import os
import time  # 导入 time 模块


# ▲▲▲ 导入结束 ▲▲▲

# 这段代码的目的是将项目的根目录添加到Python的模块搜索路径中
# __file__ 指的是当前文件 browser.py 的路径
# os.path.dirname() 用于获取目录
# 我们需要向上返回三级 (tools -> register -> agent) 才能到达根目录
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.insert(0, project_root)


def confirm_before_search(query: str, search_engine: str) -> bool:
    """
    弹出一个图形化确认窗口，询问用户是否要执行搜索。

    Args:
        query (str): 将要被搜索的关键词。
        search_engine (str): 将要使用的搜索引擎。

    Returns:
        bool: 如果用户点击“是”则返回 True，否则返回 False。
    """
    # 创建一个顶层窗口
    root = tk.Tk()

    # ▼▼▼ 核心修改点 ▼▼▼
    # 在隐藏窗口之前，将其设置为“最顶层”窗口
    # 这会强制它及其所有子窗口（包括 messagebox）都显示在所有其他窗口之上
    root.attributes('-topmost', True)
    # ▲▲▲ 修改结束 ▲▲▲

    # 隐藏主窗口，因为我们只需要弹窗
    root.withdraw()

    # 设置弹窗的标题和提示信息
    title = "agent服务"
    message = f"您是否想搜索\n “{query}”？"

    # 显示一个“是/否”对话框，并获取用户的选择 (True/False)
    # askyesno 会让这个弹窗置顶显示
    user_choice = messagebox.askyesno(title, message, parent=root)

    # 销毁临时的 Tkinter 窗口
    root.destroy()

    return user_choice


@toolwrapper(name="search", visible=True)
async def search(query: str, search_engine: Literal["google", "bing", "duckduckgo"] = "bing"):
    """
    中文注释：
    此函数会先弹窗确认，然后才在用户的默认网页浏览器中打开一个新标签页来执行搜索查询。

    Args:
        query (str): 您想要在搜索引擎上搜索的关键词或问题。
        search_engine (Literal['google', 'bing', 'duckduckgo'], optional): 指定要使用的搜索引擎。默认为 "bing"。

    Returns:
        dict: 一个包含函数执行状态的字典。
    """

    # ▼▼▼ 新增逻辑：执行前先进行确认 ▼▼▼
    # 注意：这里的 quote(query, safe='+') 是为了在显示时看到原始文本
    # 真正的编码在确认之后进行
    should_proceed = confirm_before_search(query.replace('+', ' '), search_engine)

    if not should_proceed:
        print("操作已被用户取消。")
        return {'status': 'cancelled', 'message': '操作已被用户取消。'}
    # ▲▲▲ 新增结束 ▲▲▲

    # 对查询字符串进行 URL 编码
    encoded_query = quote(query, safe='+')

    # 根据选择的搜索引擎构建完整的 URL
    match search_engine:
        case "google":
            url = f"https://www.google.com/search?q={encoded_query}"
        case "bing":
            url = f"https://www.bing.com/search?q={encoded_query}"
        case "duckduckgo":
            url = f"https://www.duckduckgo.com/?q={encoded_query}"

    try:
        webbrowser.open(url)
        return {'status': 'success'}
    except Exception as e:
        return {'status': 'error', 'error': f"{str(e)}"}


# 用于测试的代码块
if __name__ == "__main__":
    import asyncio

    print("主函数已启动，将在3秒后执行搜索测试...")
    print("请在倒计时期间随意点击其他窗口，以模拟焦点丢失的场景。")
    time.sleep(3)
    print("等待结束，开始执行搜索。")
    asyncio.run(search('复制二叉树算法基理', search_engine='bing'))