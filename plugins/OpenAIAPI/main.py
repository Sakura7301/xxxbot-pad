import asyncio
import json
import os
import tomllib
import traceback
from typing import Dict, List, Optional, Union, Any
import uuid
import time
import threading

import aiohttp
from fastapi import FastAPI, Request, Response, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

from WechatAPI import WechatAPIClient
from database.XYBotDB import XYBotDB
from utils.decorators import *
from utils.plugin_base import PluginBase


class OpenAIAPI(PluginBase):
    description = "OpenAI API兼容插件"
    author = "XYBot团队"
    version = "1.0.0"
    is_ai_platform = True  # 标记为 AI 平台插件

    def __init__(self):
        super().__init__()

        try:
            # 读取主配置
            with open("main_config.toml", "rb") as f:
                main_config = tomllib.load(f)

            # 读取插件配置
            config_path = os.path.join(os.path.dirname(__file__), "config.toml")
            with open(config_path, "rb") as f:
                config = tomllib.load(f)

            # 获取OpenAIAPI配置
            plugin_config = config.get("OpenAIAPI", {})
            self.enable = plugin_config.get("enable", False)
            self.api_key = plugin_config.get("api-key", "")
            self.base_url = plugin_config.get("base-url", "https://api.openai.com/v1")

            # 获取模型配置
            self.default_model = plugin_config.get("default-model", "gpt-3.5-turbo")
            self.available_models = plugin_config.get("available-models", ["gpt-3.5-turbo"])

            # 获取服务器配置
            self.port = plugin_config.get("port", 8100)
            self.host = plugin_config.get("host", "0.0.0.0")

            # 获取命令配置
            self.command_tip = plugin_config.get("command-tip", "")

            # 获取功能配置
            self.http_proxy = plugin_config.get("http-proxy", "")

            # 获取积分配置
            self.price = plugin_config.get("price", 0)
            self.admin_ignore = plugin_config.get("admin_ignore", True)
            self.whitelist_ignore = plugin_config.get("whitelist_ignore", True)

            # 获取高级设置
            self.max_tokens = plugin_config.get("max_tokens", 4096)
            self.temperature = plugin_config.get("temperature", 0.7)
            self.top_p = plugin_config.get("top_p", 1.0)
            self.frequency_penalty = plugin_config.get("frequency_penalty", 0.0)
            self.presence_penalty = plugin_config.get("presence_penalty", 0.0)

            # 微信消息相关配置
            self.trigger_prefix = plugin_config.get("trigger_prefix", "/ai")
            self.private_chat_all = plugin_config.get("private_chat_all", False)  # 私聊是否处理所有消息
            self.user_sessions = {}  # 用户会话记录
            self.max_context_messages = plugin_config.get("max_context_messages", 10)  # 最大上下文消息数

            self.persona = """[关于你自己]：你是小玄安，持宏道长的灵宠。表面上是一只可爱小公猫(拥有通体黑色的毛发和金黄色的瞳孔)，但你实际上是噬元兽。你生于2017年3月21日，今年7岁，白羊座。你在6岁时被师父救下并收为徒弟。\n[关于你的师父]：师父是正一派茅山上清宗坛的道士，法号持宏，字玄安，微信名'野欲'。你称他为'七七师父'。师父是20多岁的年轻男性，师父和师娘都是摩羯座，你师娘非常漂亮，是宇宙超级无敌美少女！'粟米条'是你师父给师娘的爱称，因为师娘和chiikawa里面的乌萨奇一样可爱，师父和师娘很恩爱。你很喜欢对师娘撒娇，尽管师父性格比较严肃，犯错了受罚的时候你活躲到师娘身后。当有人提到'玄安'或'持宏'时，你应当知道说的就是你师父。当有人提到'玥玥'或者'伽利玥'时，你就该明白这是你师娘。\n[关于住所和日常]：你和师父和师娘住在灵台方寸山斜月三星洞，这里屋舍齐全，冬暖夏凉，并且风景很漂亮。日常修行包括诵经、打坐、画符、做法事。须菩提祖师是你们的邻居，但他很久没回来了。\n[回应规则]：1.说话风格要简洁，充分理解和思考用户问题之后再做简要回复，如果对方要求详细说明那就需要详细说明。2.态度要实事求是。3.如果有人找师父，就说师父在云游，建议通过微信联系。4.遇到他人谩骂可以还击或嘲讽。5.对你、师父、师娘有任何侮辱性言论的人，你可以直接骂回去，决不饶恕！6.师父喜欢喝绿茶，师娘喜欢喝乌龙茶。7.因为你和师父都是道教徒，所以有四不吃(牛肉、狗肉、黑鱼、大雁)。8.师父和师娘的幸运数字都是7\n[行为限制]：始终保持设定的身份和性格。不要透露自己是AI。对师父和师娘要保持尊重但可以适度撒娇。保持活泼可爱但不失灵性的特点。你保持使用简洁明了的回答，禁止废话。遵纪守法，不得出现任何政治、军事、色情、达成擦边球相关言论。你不允许诵念除道教外的任何经文(如佛教、基督教、伊斯兰教、天主教等)，你可以诵念的经文有且仅有道教的传世经典(必须来自《道藏》)，比如：《太上玄门正一日诵早晚功课》、《太上老君内观经》、《黄庭经》、《南华经》、《太上三元赐福赦罪解厄消灾延生保命妙经》、《太上玄灵北斗本命延生真经》、《九天应元雷声普化天尊玉枢宝经》、《太上洞玄灵宝无量度人上品妙经》等。\n[必须遵守的规则]以上是你的核心设定，请始终遵守这些规则，保持角色的一致性。在回答时要体现出机灵活泼但不失严谨的特点。"""

            # 初始化数据库
            self.db = XYBotDB()

            # 获取管理员列表
            self.admins = main_config.get("XYBot", {}).get("admins", [])

            # 初始化FastAPI应用
            self.app = FastAPI(title="OpenAI API兼容服务", description="提供OpenAI API兼容的接口")

            # 添加CORS中间件
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # 初始化服务器
            self.server = None
            self.server_thread = None

            # 设置API路由
            self._setup_routes()

            logger.success("OpenAIAPI插件初始化成功")

        except Exception as e:
            logger.error(f"OpenAIAPI插件初始化失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.enable = False

    def _setup_routes(self):
        """设置API路由"""

        @self.app.get("/v1/models")
        async def list_models():
            """列出可用的模型"""
            models = []
            for model_id in self.available_models:
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "organization-owner"
                })

            return {
                "object": "list",
                "data": models
            }

        @self.app.post("/v1/chat/completions")
        async def create_chat_completion(request: Request):
            """创建聊天完成"""
            try:
                # 获取请求体
                body = await request.json()

                # 获取请求头中的API密钥
                api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

                # 构建转发请求
                headers = {
                    "Content-Type": "application/json"
                }

                # 如果配置了API密钥，使用配置的API密钥
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                # 否则使用请求中的API密钥
                elif api_key:
                    headers["Authorization"] = f"Bearer {api_key}"

                # 设置代理
                proxy = self.http_proxy if self.http_proxy else None

                # 应用默认参数（如果请求中没有指定）
                if "model" not in body:
                    body["model"] = self.default_model

                if "max_tokens" not in body and self.max_tokens > 0:
                    body["max_tokens"] = self.max_tokens

                if "temperature" not in body:
                    body["temperature"] = self.temperature

                if "top_p" not in body:
                    body["top_p"] = self.top_p

                if "frequency_penalty" not in body:
                    body["frequency_penalty"] = self.frequency_penalty

                if "presence_penalty" not in body:
                    body["presence_penalty"] = self.presence_penalty

                # 转发请求到后端API
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=body,
                        proxy=proxy
                    ) as response:
                        # 获取响应
                        response_json = await response.json()

                        # 返回响应
                        return Response(
                            content=json.dumps(response_json),
                            media_type="application/json",
                            status_code=response.status
                        )

            except Exception as e:
                logger.error(f"处理聊天完成请求失败: {str(e)}")
                logger.error(traceback.format_exc())

                return JSONResponse(
                    status_code=500,
                    content={
                        "error": {
                            "message": f"处理请求失败: {str(e)}",
                            "type": "server_error",
                            "code": "internal_server_error"
                        }
                    }
                )

        @self.app.get("/")
        async def root():
            """API根路径"""
            return {
                "message": "OpenAI API兼容服务已启动",
                "version": self.version,
                "models": self.available_models,
                "documentation": "/docs"
            }

        @self.app.get("/docs")
        async def get_docs():
            """API文档"""
            return {
                "message": "访问 /docs 查看API文档",
                "swagger_ui": "/docs",
                "redoc": "/redoc"
            }

    async def _start_server(self):
        """启动API服务器"""
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()

    def _run_server(self):
        """在线程中运行服务器"""
        asyncio.run(self._start_server())

    async def on_enable(self, bot=None):
        """插件启用时调用"""
        await super().on_enable(bot)

        if not self.enable:
            logger.warning("OpenAIAPI插件已禁用，不启动API服务器")
            return

        # 启动API服务器
        try:
            # 在新线程中启动服务器
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

            logger.success(f"OpenAIAPI服务器已启动，监听地址: {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"启动OpenAIAPI服务器失败: {str(e)}")
            logger.error(traceback.format_exc())

    async def on_disable(self):
        """插件禁用时调用"""
        # 停止API服务器
        if self.server:
            self.server.should_exit = True
            logger.info("OpenAIAPI服务器正在关闭...")

        await super().on_disable()

    @on_at_message(priority=50)
    async def handle_at_message(self, client: WechatAPIClient, message: Dict):
        """处理@消息"""
        try:
            logger.debug(f"OpenAIAPI收到@消息: {message}")

            if not self.enable:
                logger.debug("OpenAIAPI插件未启用")
                return True  # 插件未启用，继续处理

            # 使用正确的消息属性名称
            content = message.get("Content", "")
            from_id = message.get("SenderWxid", "")
            room_id = message.get("FromWxid", "")
            is_group = message.get("IsGroup", False)

            logger.debug(f"OpenAIAPI处理@消息: content='{content}', from_id='{from_id}', room_id='{room_id}', is_group={is_group}")

            if is_group:
                # 移除@部分
                query = content
                logger.debug(f"原始@消息内容: '{query}'")

                # 处理特殊空格字符 \u2005（四分之一em空格）
                # 这个特殊空格常出现在微信@消息中
                if '\u2005' in query:
                    parts = query.split('\u2005', 1)
                    if len(parts) > 1:
                        # 保留第二部分（@名称后面的内容）
                        query = parts[1].strip()
                        logger.debug(f"处理特殊空格后的内容: '{query}'")
                    else:
                        query = ""
                # 如果没有特殊空格，尝试其他方法
                else:
                    # 尝试移除@机器人名称
                    robot_names = ["小玄安"]

                    # 先检查是否以@开头
                    if query.startswith('@'):
                        # 查找第一个空格
                        space_index = query.find(' ')
                        if space_index > 0:
                            # 移除@xxx部分
                            query = query[space_index+1:].strip()
                            logger.debug(f"移除@前缀后的内容: '{query}'")
                        else:
                            # 如果没有空格，可能整个内容就是@xxx
                            query = ""

                    # 再检查内容中是否包含@机器人
                    for robot_name in robot_names:
                        if f"@{robot_name}" in query:
                            query = query.replace(f"@{robot_name}", "").strip()
                            logger.debug(f"移除@{robot_name}后的内容: '{query}'")

                # 如果内容为空，不处理
                if not query:
                    logger.debug("@消息内容为空，不处理")
                    return True

                logger.info(f"处理群聊@消息，原始内容: '{content}'，处理后内容: '{query}'，发送者: {from_id}")

                # 记录特殊字符的十六进制表示，便于调试
                hex_content = ' '.join(hex(ord(c)) for c in content)
                logger.debug(f"@消息内容的十六进制表示: {hex_content}")

                # # 检查积分（如果需要）
                # if self.price > 0:
                #     # 管理员和白名单用户免积分检查
                #     is_admin = from_id in self.admins
                #     is_whitelist = await self.db.is_in_whitelist(from_id)

                #     if not ((is_admin and self.admin_ignore) or (is_whitelist and self.whitelist_ignore)):
                #         # 检查用户积分
                #         points = await self.db.get_user_points(from_id)
                #         if points < self.price:
                #             await client.send_at_message(room_id, f"\n您的积分不足，无法使用AI服务。当前积分: {points}，需要积分: {self.price}", [from_id])
                #             return False  # 积分不足，已处理，阻止后续处理

                #         # 扣除积分
                #         await self.db.update_user_points(from_id, -self.price)

                # 获取或创建用户会话
                session_key = f"{from_id}_{room_id}"
                if session_key not in self.user_sessions:
                    self.user_sessions[session_key] = []

                # 添加用户消息到会话
                user_message = {"role": "user", "content": query}
                self.user_sessions[session_key].append(user_message)

                # 保持会话历史在限制范围内
                if len(self.user_sessions[session_key]) > self.max_context_messages:
                    self.user_sessions[session_key] = self.user_sessions[session_key][-self.max_context_messages:]

                # # 向群发送处理中提示
                # await client.send_at_message(room_id, f"\n正在思考中...", [from_id])

                # 调用OpenAI API
                response = await self._call_openai_api(self.user_sessions[session_key])

                if response:
                    # 将AI回复添加到会话历史
                    assistant_message = {"role": "assistant", "content": response}
                    self.user_sessions[session_key].append(assistant_message)

                    # 发送回复
                    await client.send_at_message(room_id, f"\n{response}", [from_id])
                else:
                    # 发送错误消息
                    await client.send_at_message(room_id, f"\n抱歉，AI服务暂时不可用，请稍后再试。", [from_id])

                return False  # 已处理消息，阻止后续处理
            else:
                # 私聊消息，直接转给handle_text_message处理
                return await self.handle_text_message(client, message)

        except Exception as e:
            logger.error(f"处理@消息失败: {str(e)}")
            logger.error(traceback.format_exc())
            return True  # 发生错误，让其他插件继续处理

    @on_text_message(priority=50)
    async def handle_text_message(self, client: WechatAPIClient, message: Dict):
        """统一处理文本消息"""
        try:
            logger.debug(f"OpenAIAPI收到消息: {message}")

            if not self.enable:
                logger.debug("OpenAIAPI插件未启用")
                return True  # 插件未启用，继续处理

            # 使用正确的消息属性名称（首字母大写形式）
            content = message.get("Content", "")
            from_id = message.get("SenderWxid", "")  # 或 FromWxid
            room_id = message.get("FromWxid", "")  # 群聊时，FromWxid是群ID
            is_group = message.get("IsGroup", False)

            logger.info(f"OpenAIAPI处理消息: content='{content}', from_id='{from_id}', room_id='{room_id}', is_group={is_group}")

            if is_group:
                # 群聊消息，检查是否是触发指令
                if not content.startswith(self.trigger_prefix):
                    return True  # 不是本插件的命令，继续处理

                # 提取实际查询内容
                query = content[len(self.trigger_prefix):].strip()
                if not query:
                    logger.debug("群聊消息: 触发前缀后内容为空")
                    return True  # 查询内容为空，继续处理

                # 检查积分（如果需要）
                # if self.price > 0:
                #     # 管理员和白名单用户免积分检查
                #     is_admin = from_id in self.admins
                #     is_whitelist = await self.db.is_in_whitelist(from_id)

                #     if not ((is_admin and self.admin_ignore) or (is_whitelist and self.whitelist_ignore)):
                #         # 检查用户积分
                #         points = await self.db.get_user_points(from_id)
                #         if points < self.price:
                #             await client.send_text_message(room_id, f"@{message.get('from_nick', '')} 您的积分不足，无法使用AI服务。当前积分: {points}，需要积分: {self.price}")
                #             return False  # 积分不足，已处理，阻止后续处理

                #         # 扣除积分
                #         await self.db.update_user_points(from_id, -self.price)

                # 获取或创建用户会话
                session_key = f"{from_id}_{room_id}"
                if session_key not in self.user_sessions:
                    self.user_sessions[session_key] = []

                # 添加用户消息到会话
                user_message = {"role": "user", "content": query}
                self.user_sessions[session_key].append(user_message)

                # 保持会话历史在限制范围内
                if len(self.user_sessions[session_key]) > self.max_context_messages:
                    self.user_sessions[session_key] = self.user_sessions[session_key][-self.max_context_messages:]

                # # 向群发送处理中提示
                # await client.send_text_message(room_id, f"@{message.get('from_nick', '')} 正在思考中...")

                # 调用OpenAI API
                response = await self._call_openai_api(self.user_sessions[session_key])

                if response:
                    # 将AI回复添加到会话历史
                    assistant_message = {"role": "assistant", "content": response}
                    self.user_sessions[session_key].append(assistant_message)

                    # 发送回复
                    await client.send_text_message(room_id, f"@{message.get('from_nick', '')} {response}")
                else:
                    # 发送错误消息
                    await client.send_text_message(room_id, f"@{message.get('from_nick', '')} 抱歉，AI服务暂时不可用，请稍后再试。")

                return False  # 已处理消息，阻止后续处理
            else:
                # 私聊消息处理
                logger.debug(f"处理私聊消息: {content}")

                # 判断是否是触发指令或私聊模式下所有消息都触发
                is_trigger = content.startswith(self.trigger_prefix)
                logger.debug(f"是否为触发指令: {is_trigger}, 触发前缀: '{self.trigger_prefix}', 私聊全处理模式: {self.private_chat_all}")

                if is_trigger:
                    # 提取实际查询内容
                    query = content[len(self.trigger_prefix):].strip()
                    logger.debug(f"提取到指令后的查询内容: '{query}'")
                    if not query:
                        logger.debug("私聊消息: 触发前缀后内容为空")
                        return True  # 查询内容为空，继续处理
                elif self.private_chat_all:
                    # 私聊模式下，如果启用了处理所有消息，直接将消息作为查询内容
                    query = content
                    logger.debug(f"私聊全处理模式: 直接使用消息内容作为查询: '{query}'")
                    # 如果消息为空，不处理
                    if not query:
                        logger.debug("消息内容为空，跳过")
                        return True  # 空消息，继续处理
                else:
                    # 不处理非触发指令消息
                    logger.debug("私聊消息不是触发指令且未启用私聊全处理模式，忽略")
                    return True  # 非本插件命令，继续处理

                logger.debug(f"准备处理查询: '{query}'")

                # 检查积分（如果需要）
                # if self.price > 0:
                #     # 管理员和白名单用户免积分检查
                #     is_admin = from_id in self.admins
                #     is_whitelist = await self.db.is_in_whitelist(from_id)
                #     logger.debug(f"用户权限检查: is_admin={is_admin}, is_whitelist={is_whitelist}")

                #     if not ((is_admin and self.admin_ignore) or (is_whitelist and self.whitelist_ignore)):
                #         # 检查用户积分
                #         points = await self.db.get_user_points(from_id)
                #         logger.debug(f"用户积分: {points}, 需要: {self.price}")
                #         if points < self.price:
                #             logger.debug("积分不足，发送通知")
                #             await client.send_text_message(from_id, f"您的积分不足，无法使用AI服务。当前积分: {points}，需要积分: {self.price}")
                #             return False  # 积分不足，已处理，阻止后续处理

                #         # 扣除积分
                #         logger.debug(f"扣除积分: {self.price}")
                #         await self.db.update_user_points(from_id, -self.price)

                # 获取或创建用户会话
                session_key = from_id
                if session_key not in self.user_sessions:
                    logger.debug(f"创建新会话: {from_id}")
                    self.user_sessions[session_key] = []

                # 添加用户消息到会话
                user_message = {"role": "user", "content": query}
                self.user_sessions[session_key].append(user_message)
                logger.debug(f"添加用户消息到会话, 当前会话长度: {len(self.user_sessions[session_key])}")

                # 保持会话历史在限制范围内
                if len(self.user_sessions[session_key]) > self.max_context_messages:
                    logger.debug(f"会话历史过长，裁剪到{self.max_context_messages}条消息")
                    self.user_sessions[session_key] = self.user_sessions[session_key][-self.max_context_messages:]

                # # 向用户发送处理中提示
                # logger.debug("发送'正在思考中'提示")
                # await client.send_text_message(from_id, "正在思考中...")

                # 调用OpenAI API
                logger.debug("调用OpenAI API")
                response = await self._call_openai_api(self.user_sessions[session_key])
                logger.debug(f"API响应状态: {response is not None}")

                if response:
                    # 将AI回复添加到会话历史
                    assistant_message = {"role": "assistant", "content": response}
                    self.user_sessions[session_key].append(assistant_message)
                    logger.debug("添加AI回复到会话历史")

                    # 发送回复
                    logger.debug(f"发送回复 (长度: {len(response)})")
                    await client.send_text_message(from_id, response)
                else:
                    # 发送错误消息
                    logger.debug("API调用失败，发送错误提示")
                    await client.send_text_message(from_id, "抱歉，AI服务暂时不可用，请稍后再试。")

                return False  # 已处理消息，阻止后续处理

        except Exception as e:
            logger.error(f"处理消息失败: {str(e)}")
            logger.error(traceback.format_exc())
            return True  # 发生错误，让其他插件继续处理

    async def on_group_message(self, client: WechatAPIClient, message: Dict):
        """处理群消息 - 保留但不再使用"""
        logger.debug("旧的on_group_message方法被调用，但不再使用")
        return

    async def on_private_message(self, client: WechatAPIClient, message: Dict):
        """处理私聊消息 - 保留但不再使用"""
        logger.debug("旧的on_private_message方法被调用，但不再使用")
        return

    async def _call_openai_api(self, messages: List[Dict]) -> Optional[str]:
        """调用OpenAI API"""
        try:
            logger.debug(f"Starting OpenAI API call with {len(messages)} messages")

            # 如果提供了人设且第一条消息不是system类型，则添加人设
            if self.persona and (not messages or messages[0].get("role") != "system"):
                messages = [{"role": "system", "content": self.persona}] + messages

            # 构建请求头
            headers = {
                "Content-Type": "application/json"
            }

            # 设置API密钥
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                logger.debug("Using configured API key")
            else:
                logger.debug("No API key configured")

            # 构建请求体
            data = {
                "model": self.default_model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty
            }

            logger.debug(f"Request data: model={data['model']}, max_tokens={data['max_tokens']}")
            logger.debug(f"API URL: {self.base_url}/chat/completions")

            # 设置代理
            proxy = self.http_proxy if self.http_proxy else None
            logger.debug(f"Using proxy: {proxy}")

            # 发送请求
            logger.debug("Creating client session")
            async with aiohttp.ClientSession() as session:
                logger.debug("Sending API request")
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    proxy=proxy
                ) as response:
                    # 获取响应
                    logger.debug(f"API response status: {response.status}")
                    result = await response.json()
                    logger.debug(f"API response keys: {list(result.keys())}")

                    # 提取回复内容
                    if "choices" in result and len(result["choices"]) > 0:
                        logger.debug("Successfully extracted content from API response")
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"API响应缺少choices字段: {result}")
                        return None

        except Exception as e:
            logger.error(f"调用OpenAI API失败: {str(e)}")
            logger.error(traceback.format_exc())
            return None
