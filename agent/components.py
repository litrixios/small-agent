'''
This file provide three distinct components:
1. Environment: The platform which we are interacting with. (PC and Android)
2. Trigger: How we will actually execute the command.
3. Agent: Get the observation from the environment and generate the action.
'''
import io
import os
import json
import base64
import asyncio
import logging
import traceback
import threading
import subprocess
from typing import Iterable, Literal, Optional, Dict, List
import websockets

import colorlog
from PIL import Image
from codelinker import CodeLinker, CodeLinkerConfig, EventProcessor, EventSink
from codelinker.models import SEvent, ChannelTag


from channels import sc
from agentmodule import ActionListener, Executor, ActivityWatchClient
from prompt import SYSTEM_PROMPT
from constant import AgentResponse
from constant import MAX_TRANSFER_SIZE, TIMEOUT, BUFFER_SIZE

from register import ToolRegister
toolreg = ToolRegister()
img_base64 = None

# Set the logger format.
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s%(levelname)s - %(name)s - %(message)s',
                            log_colors={
                                'DEBUG':    'white',
                                'INFO':     'green',
                                'WARNING':  'yellow',
                                'ERROR':    'red',
                                'CRITICAL': 'red,bg_white',
                            })
# formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# load information.
default_cfg_file = os.path.join(os.path.dirname(__file__), '..', 'private.toml')
if not os.path.exists(default_cfg_file):
    default_cfg_file = os.path.join(os.path.dirname(__file__), 'private.toml')

CL_CFGFILE = os.getenv(key = 'CODELINKER_CFG',
                    default = default_cfg_file)

codelinker_config = CodeLinkerConfig.from_toml(CL_CFGFILE)
codelinker_config.request.default_completions_model = "activeagent"
codelinker_config.request.use_cache = False
codelinker_config.request.save_completions = True

clinker = CodeLinker(config = codelinker_config)
eventSink = EventSink(sinkChannels=sc,logger=logger)

class BasicComponent(EventProcessor):
    def __init__(self,name:str):
        super().__init__(name = name,
                        sink = eventSink)
        self.listen(sc.setup)(self.setup)
        self.cl = clinker

    def gather(self,
            tags: ChannelTag | Iterable[ChannelTag] | None = None,
            return_dumper:Literal['identity','json'] = 'identity') -> str | Iterable[dict]:
        messages = super().gather(tags = tags,return_dumper = 'identity')
        match return_dumper:
            case 'identity':
                return messages
            case 'json':
                for msg in messages:
                    o = msg['content']
                    if isinstance(o,SEvent):
                        msg['content'] = json.dumps({
                            "Time": o.time,
                            "Source": o.source,
                            "Tags": o.tags,
                            "Event": o.content
                        },ensure_ascii=False)
                return messages
            case __:
                raise ValueError(f"return_dumper should be 'identity' or 'json', but got {return_dumper}")

class AndroidEnv(BasicComponent):
    def __init__(self, *,
                server_host:str = '0.0.0.0',
                server_port:int = 9999,
                name = "AndroidEnv",):
        """
        Args:
            server_host (str, optional): the IP of the socket connection. Defaults to '0.0.0.0'.
            server_port (int, optional): the port of the socket connection. Defaults to 9999.
            name (str, optional): The name of the environment. Defaults to "AndroidEnv".
        """
        super().__init__(name)

        self.client_count: int = 0
        self.server_host : str = server_host
        self.server_port : int = server_port

        complete_tools:List[Dict] = toolreg.get_all_tools_dict()
        self.tools:List[Dict] = [t for t in complete_tools if 'android' in t["name"]]

    async def setup(self):
        """
        For the set up of the android:
        1. Establish a socket connection and wait a client to connect.
        2. Listen on several channels.
        """
        async def run_server():
            async with server:
                await server.serve_forever()

        self.logger.info("Initializing Android Environment...")
        self.add(sc.agent.operations, content = json.dumps(self.tools), silent = True)

        logger.info("Android socket waiting for connection.")
        server = await asyncio.start_server(self.handle_client, self.server_host, self.server_port, limit = MAX_TRANSFER_SIZE)
        addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
        logger.info(f'Serving on {addrs}')

        self.server_task = asyncio.create_task(run_server())

        while self.client_count == 0:
            await asyncio.sleep(0.5)

        logger.info("Android socket connected.")
        logger.info("Env setup done.")

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.client_count += 1
        async def read_data():

            while True:
                try:
                    datalen_bytes = await asyncio.wait_for(
                        reader.read(4),
                        timeout=TIMEOUT)
                except asyncio.TimeoutError:
                    logger.info("Timeout waiting for data length. Closing connection.")
                    continue

                if not datalen_bytes:
                    logger.info("Received empty message. Closing connection.")
                    await asyncio.sleep(1)
                    continue

                datalen:int = int.from_bytes(datalen_bytes, byteorder='big')
                logger.info(f"Received data length: {datalen}")
                msg:bytes = b''

                while datalen > 0:
                    try:
                        data = await asyncio.wait_for(reader.read(min(BUFFER_SIZE, datalen)), timeout=TIMEOUT)
                    except asyncio.TimeoutError:
                        print("Timeout waiting for data chunk. Closing connection.")
                        break

                    if not data:
                        break
                    datalen -= len(data)
                    msg += data

                try:
                    msg_str:str = msg.decode('utf-8')
                except UnicodeDecodeError:
                    logger.error("Failed to decode data chunk.")

                if datalen > 0:
                    logger.error('Data integrity error. Closing connection.')

                try:
                    msg_json:Dict = json.loads(msg_str)
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON message.")
                    self.logger.error(msg_str)
                    continue

                logger.info(f"Receive Msg Type: {msg_json['type']}")
                logger.info(f'Received msg Down.')

                match msg_json["type"]:
                        case "act_error":
                            logger.error(msg_json["act_error"])
                        case "act_ret":
                            if "screenshot" in msg_json["act_ret"] and len(msg_json["act_ret"]["screenshot"]) > 0:
                                global img_base64
                                img_base64 = msg_json["act_ret"].pop("screenshot")

                                img_data = base64.b64decode(img_base64)

                                with open("screenshot.jpeg", "wb") as f:
                                    f.write(img_data)
                                img = Image.open(io.BytesIO(img_data))
                                img.save("screenshot.jpeg")
                            logger.debug(msg_json["act_ret"])
                        case __:
                            logger.info(msg_json[msg_json["type"]])

                msg_str:str = json.dumps(msg_json)
                self.add(sc.observation, msg_str)

        async def write_data():
            data_event:SEvent = self.get(sc.android.write)
            data_str:str = data_event.content
            send_msg:bytes = data_str.encode(encoding = 'utf-8')
            writer.write(len(send_msg).to_bytes(4, byteorder = 'big'))
            writer.write(send_msg)
            await writer.drain()
            logger.info('<Write complete>')

        try:
            read_task = asyncio.create_task(read_data())
            # listen to the write event.
            self.listen(sc.android.write)(write_data)
            await asyncio.gather(read_task)

        except Exception as e:
            logger.error(f"Error in main_process: {e}")
            logger.error(traceback.format_exc())

        finally:
            read_task.cancel()
            writer.close()
            await writer.wait_closed()
            self.client_count -= 1

class PCEnv(BasicComponent):
    def __init__(self, *,
                 interval_seconds: int = 15,
                 name: str = 'PCEnv',
                 ws_host: str = '127.0.0.1',
                 ws_port: int = 8765
                 ):
        super().__init__(name)
        self.interval_seconds = interval_seconds
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.websocket = None

        global global_pc_env
        global_pc_env = self

        complete_tools = toolreg.get_all_tools_dict()
        self.tools = [t for t in complete_tools if 'android' not in t["name"]]

    async def setup(self):
        self.logger.info("正在初始化网站智能体环境...")
        self.listen(sc.pc.notify)(self.execute)

        server = await websockets.serve(self.handle_connection, self.ws_host, self.ws_port)
        self.logger.info(f"WebSocket服务器已在 ws://{self.ws_host}:{self.ws_port} 启动，等待浏览器连接...")

        self.add(sc.agent.operations, content=json.dumps(self.tools), silent=True)
        await server.wait_closed()

    async def handle_connection(self, websocket):
        """处理来自JS监视器的连接和消息"""
        self.logger.info(f"浏览器监视器已连接！")

        self.websocket = websocket  # 保存连接
        try:
            async for message in websocket:
                self.logger.debug(f"收到观察数据: {message}")
                self.add(sc.observation, content=message)
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("浏览器监视器连接已断开。")
        finally:
            self.websocket = None

    async def execute(self):
        operation_event: SEvent = self.get(sc.agent.execute)
        operation: str = operation_event.content

        if operation == 'nop':
            self.logger.info("大脑决定不执行任何操作。")
            return

        self.logger.info(f"收到大脑指令，准备执行操作: {operation}")
        parts = operation.split('&')
        tool_name = parts[0]
        kwargs = {}
        if len(parts) > 1:
            for part in parts[1:]:
                key, value = part.split('=', 1)
                kwargs[key] = value

        try:
            tool_func = toolreg[tool_name]
            result = await tool_func(**kwargs)
            self.logger.info(f"操作执行完毕: {result}")
        except Exception as e:
            self.logger.error(f"执行操作 {tool_name} 时出错: {e}")
class DemoAgent(BasicComponent):
    def __init__(self,*,
                env:Literal["PC","Mobile"],
                name:str = "ActiveAgent"):
        """
        Args:
            env (Literal['PC','Mobile']): Whether we are on PC or the Mobile.
            name (str, optional): The name of the agent. Defaults to "ActiveAgent".
        """
        super().__init__(name)
        self.env:str = env

    @property
    def memory(self):
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    async def setup(self):
        logger.info("Initializing Agent...")
        self.listen(sc.observation)(self.propose)
        logger.info("Agent setup done.")

    async def propose(self):

        if self.get_tag_lock(sc.agent.propose).locked():
            logger.error("Another agent is proposing.")
            return

        async with self.get_tag_lock(sc.agent.propose):
            async with self.get_tag_lock(sc.activity):

                ops_event:SEvent = self.get(sc.agent.operations)
                ops:str = ops_event.content

                obs:Dict = self.gather([sc.observation],return_dumper='json')

                history = obs

                # TODO: Can we add user feedback for PC?

                user_content: str = json.dumps({
                    "Instructions": "你是一个主动的网站智能助手。请分析用户最近在网页上的行为历史(history)，并判断是否需要提供帮助。如果需要，请从可用操作(operations)中选择一个最合适的工具来执行任务。如果不需要帮助，请将'Operation'字段设为'nop'。",
                    "operations": ops,
                    "history": history  # 把history也放进prompt，让模型看得更清楚
                })

                # if self.env == "Mobile":
                #     history = history[-1:]

                global img_base64

                if self.env == 'Mobile' and img_base64 is not None:
                    img = [{
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{img_base64}",
                            "detail": "low"
                        }
                    }]
                    img_base64 = None

                else:
                    img = []

                logger.debug('Start Proposing....')

                res: AgentResponse = await self.cl.exec(
                    prompt = user_content,
                    return_type = AgentResponse,
                    messages = self.memory + history,
                    images = img
                )

                self.logger.info(res)
                self.add(sc.agent.propose, content = res.model_dump_json())

                if res.Operation is not None and res.Operation != 'null':
                    self.add(sc.agent.execute, res.Operation)
                else:
                    self.add(sc.agent.execute, "nop")


class Trigger(BasicComponent):
    def __init__(self, *,
                 env: Literal["PC", "Mobile"],
                 name: str = "Trigger",
                 ):
        """
        Args:
            env (Literal['PC','Mobile']): 我们当前工作的环境。
            name (str, optional): 组件的名称。
        """
        super().__init__(name)
        self.env: str = env

    async def setup(self):
        logger.info("Initializing Trigger...")
        # 监听来自“大脑”(Agent)的最终决策指令
        self.listen(sc.agent.execute)(self.execute)
        logger.info("Trigger setup done.")

    async def execute(self):
        # 从事件中获取大脑决定要执行的操作字符串
        operation: str = self.get(sc.agent.execute).content
        self.logger.info(f"神经系统(Trigger)收到指令: {operation}")

        # 根据当前环境，将指令转发到正确的执行频道
        if self.env == 'PC':
            # 对于PC/网站环境，我们把指令转发到 sc.pc.notify 频道
            # PCEnv 正在监听这个频道
            self.add(sc.pc.notify, content=operation)

        elif self.env == 'Mobile':
            self.logger.warning(f"收到了Mobile环境的执行指令，但当前未处理。")
        else:
            self.logger.error(f"收到了未知环境 '{self.env}' 的执行指令，已忽略。")
