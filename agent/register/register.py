import logging
import importlib
import traceback

from copy import deepcopy
from typing import Optional, Callable, Any, Type, Union

from .label import ToolLabels
from .exceptions import ToolNotFound, EnvNotFound, ToolRegisterError

logger = logging.getLogger()

def get_func_name(func: Callable, env = None) -> str:
    if env is None or not hasattr(env, 'env_labels'):
        if hasattr(func, 'tool_labels') and isinstance(func.tool_labels, ToolLabels):
            return func.tool_labels.name
        else:
            return func.__name__
    else:
        if hasattr(func, 'tool_labels') and isinstance(func.tool_labels, ToolLabels):
            return env.env_labels.alias + '_0_' + func.tool_labels.name
        else:
            return env.env_labels.alias + '_0_' + func.__name__

class Tool(Callable):
    tool_labels: ToolLabels

class ToolRegister:
    def __init__(self,
                ):
        # load modules
        self.tools: dict[str, Tool] = {}

        for module_name in ['agent.register.tools',]:
            sub_modules = importlib.import_module(module_name).__all__
            for module in sub_modules:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    self.check_and_register(attr)

        logger.info(
            f'Loaded {len(self.tools)} tools !')
        # print(self.tools)

    def check_and_register(self, attr: Any):
        if hasattr(attr, 'tool_labels') and isinstance(attr.tool_labels, ToolLabels):
            tool_name = get_func_name(attr)
            if tool_name in self.tools:
                logger.warning(
                    f'Tool {tool_name} is replicated! The new one will be replaced!')
                return None

            self.tools[tool_name] = attr
            logger.info(f'Register tool {tool_name}!')
            return attr

        return None


    def dynamic_extension_load(self, extension: str) -> bool:
        '''Load extension dynamically.

        :param string extension: The load path of the extension.
        :return boolean: True if success, False if failed.
        '''
        try:
            module = importlib.import_module(extension)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                self.check_and_register(attr)
        except Exception as e:
            logger.error(
                f'Failed to load extension {extension}! Exception: {e}')
            # logger.error(traceback.format_exc())
            return False

        return True

    def get_tool_dict(self, tool_name: str) -> dict:
        return self[tool_name].tool_labels.dict(name_overwrite=tool_name)

    def get_all_tools(self, include_invisible=False) -> list[str]:
        if include_invisible:
            return [tool_name for tool_name in self.tools]
        else:
            return [tool_name for tool_name in self.tools if self.tools[tool_name].tool_labels.visible]

    def get_all_tools_dict(self, include_invisible=False) -> list[dict]:
        return [self.tools[tool_name].tool_labels.dict(name_overwrite=tool_name) for tool_name in self.get_all_tools(include_invisible)]

    def __getitem__(self, key) -> Tool[..., Any]:
        # two stage index, first find env, then find tool
        if isinstance(key, str):
            if key not in self.tools:
                raise ToolNotFound(tool_name=key)
            return self.tools[key]

        elif isinstance(key, tuple):
            raise NotImplementedError(f'Key {key} is not valid!')

        raise NotImplementedError(f'Key {key} is not valid!')


if __name__ == "__main__":
    import json  # 导入json库以便格式化输出

    print("=" * 50)
    print("Initializing Tool Register and loading tools...")
    print("=" * 50)

    # 1. 创建 ToolRegister 的实例，这会自动触发__init__中的工具发现和加载流程
    tool_registry = ToolRegister()

    print("\n" + "=" * 50)
    print("Fetching information for all registered tools...")
    print("=" * 50)

    # 2. 调用 get_all_tools_dict() 方法获取所有工具的详细信息列表
    #    我们设置 include_invisible=True 来确保所有工具（包括被标记为不可见的）都被打印出来
    all_tools_info = tool_registry.get_all_tools_dict(include_invisible=True)

    if not all_tools_info:
        print("No tools were found or registered.")
    else:
        print(f"Found {len(all_tools_info)} tools in total. Details are as follows:\n")

        # 3. 遍历列表，逐个打印每个工具的详细信息
        for i, tool_info in enumerate(all_tools_info):
            tool_name = tool_info.get('name', 'N/A')
            print(f"=============== Tool #{i + 1}: {tool_name} ===============")

            # 4. 使用 json.dumps 进行格式化输出（pretty-print），使其结构清晰、易于阅读
            #    - indent=4: 使用4个空格进行缩进
            #    - ensure_ascii=False: 确保中文等非ASCII字符能够正常显示，而不是被转义
            formatted_info = json.dumps(tool_info, indent=4, ensure_ascii=False)
            print(formatted_info)
            print("\n")

