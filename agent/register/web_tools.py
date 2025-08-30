import json
# 删除了顶部的 from ...components import global_pc_env
from .wrapper import toolwrapper # 确保 toolwrapper 这个名字和 wrapper.py 中定义的一致

@toolwrapper
async def click_element(selector: str):
    """
    点击网页上由CSS选择器指定的元素。
    :param selector: 元素的CSS选择器，例如 '#submit-button' 或 '.product-link'
    """
    # =================================================================
    # === 关键修正：把 import 语句放在函数内部！ ===
    # 路径也修正为三个点 "..."
    from ..components import global_pc_env
    # =================================================================

    if global_pc_env and global_pc_env.websocket:
        print(f"正在准备发送点击指令到: {selector}")
        command = {
            "action": "click_element",
            "selector": selector
        }
        await global_pc_env.websocket.send(json.dumps(command))
        return f"成功发送点击指令到选择器: {selector}"

    print("错误: 未能发送点击指令，因为未连接到浏览器。")
    return "错误: 未连接到浏览器。"