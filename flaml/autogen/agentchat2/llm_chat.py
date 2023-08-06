from __future__ import annotations
import json
import logging
import pprint
from typing import Any, Callable, Dict, List, Optional
from flaml.autogen import oai
from flaml.autogen.agentchat2.address import Address
from flaml.autogen.agentchat2.agent import SingleStateAgent
from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message
from flaml.autogen.agentchat2.stream import MessageStream

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)


class LLMChatContext(Context):
    def __init__(
        self,
        name: str,
        address: str,
        message_stream: MessageStream,
        system_message: Dict,
        llm_config: Dict[str, Any],
        functions: Optional[Dict[str, Callable]] = None,
        chat_history: Optional[List[Dict]] = None,
    ) -> None:
        self.name = name
        self.address = address
        self.message_stream = message_stream
        self.system_message = system_message
        self.llm_config = llm_config
        self.functions = functions or dict()
        self.chat_history = chat_history or list()


class LLMChatMessage(Message):
    def __init__(
        self,
        sender_name: str,
        sender_address: Address,
        receiver_name: str,
        receiver_address: Address,
        message: Dict,
    ) -> None:
        self.sender_name = sender_name
        self.sender_address = sender_address
        self.receiver_name = receiver_name
        self.receiver_address = receiver_address
        self.message = message

    def __repr__(self) -> str:
        return pprint.pformat(self.__dict__)


def llm_chat_action(
    messages: List[LLMChatMessage],
    context: LLMChatContext,
) -> LLMChatContext:
    # Group incoming messages by sender address.
    messages_by_sender_address = dict()
    for message in messages:
        messages_by_sender_address.setdefault(message.sender_address, []).append(message)
    # Process messages from each sender address.
    new_messages = []
    for sender_address, messages_at_address in messages_by_sender_address.items():
        private_messages = []
        while True:
            response = oai.ChatCompletion.create(
                messages=[
                    context.system_message,
                    *context.chat_history,
                    *[msg.message for msg in messages_at_address],
                    *private_messages,
                ],
                **context.llm_config,
            )["choices"][0]["message"]
            logger.debug(f"[Private] {context.name}: \n{pprint.pformat(response)}")
            private_messages.append(response)
            if "function_call" in response:
                function_name = response["function_call"]["name"]
                function = context.functions[function_name]
                try:
                    function_args = json.loads(response["function_call"]["arguments"])
                    logger.debug(f"[Private] {context.name}: executing {function_name}({function_args})...")
                    function_return_val = function(**function_args)
                    logger.debug(f"[Private] {context.name}: {function_name} returned {function_return_val}")
                    try:
                        reply_text = json.dumps(function_return_val)
                    except TypeError:
                        reply_text = str(function_return_val)
                except Exception as e:
                    reply_text = str(e)
                private_messages.append(
                    {
                        "role": "function",
                        "content": reply_text,
                        "name": function_name,
                    }
                )
                logger.debug(f"[Private] {context.name}: \n{pprint.pformat(private_messages[-1])}")
                continue
            break
        reply_text = private_messages[-1]["content"]
        context.message_stream.send(
            message.sender_address,
            LLMChatMessage(
                context.name,
                context.address,
                message.sender_name,
                message.sender_address,
                {"role": "user", "content": reply_text, "name": context.name},
            ),
            lambda address, agent: True,
        )
        new_messages.extend(private_messages)
    return LLMChatContext(
        name=context.name,
        address=context.address,
        message_stream=context.message_stream,
        system_message=context.system_message,
        llm_config=context.llm_config,
        functions=context.functions,
        chat_history=[
            *context.chat_history,
            *[msg.message for msg in messages],
            *new_messages,
        ],
    )


def llm_chat_save_messages(
    messages: List[LLMChatMessage],
    context: LLMChatContext,
) -> LLMChatContext:
    return LLMChatContext(
        name=context.name,
        address=context.address,
        message_stream=context.message_stream,
        system_message=context.system_message,
        llm_config=context.llm_config,
        functions=context.functions,
        chat_history=[
            *context.chat_history,
            *[msg.message for msg in messages],
        ],
    )


class LLMChatAgent(SingleStateAgent):
    def __init__(
        self,
        name: str,
        address: str,
        message_stream: MessageStream,
        system_message: Dict,
        llm_config: Dict[str, Any],
        functions: Optional[Dict[str, Callable]] = None,
        trigger: Optional[Callable[[List[LLMChatMessage], LLMChatContext], bool]] = lambda messages, context: True,
        action: Optional[Callable[[List[LLMChatMessage], LLMChatContext], LLMChatContext]] = llm_chat_action,
        default_action: Optional[
            Callable[[List[LLMChatMessage], LLMChatContext], LLMChatContext]
        ] = llm_chat_save_messages,
    ) -> None:
        super().__init__(
            initial_contexts=[
                LLMChatContext(
                    name=name,
                    address=address,
                    message_stream=message_stream,
                    system_message=system_message,
                    llm_config=llm_config,
                    functions=functions,
                )
            ]
        )
        if trigger is not None and action is not None:
            self.register_action(trigger, action)
        if default_action is not None:
            self.register_default_action(default_action)
        message_stream.add_subscriber(
            address,
            self,
            (lambda message: isinstance(message, LLMChatMessage) and message.sender_name != name),
        )
