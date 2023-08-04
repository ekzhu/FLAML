from __future__ import annotations
from typing import Any, Dict, List, Optional
from flaml.autogen import oai
from flaml.autogen.agentchat2.agent import SingleStateAgent
from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message
from flaml.autogen.agentchat2.stream import ListMessageStream, MessageStream


class LLMGroupChatContext(Context):
    def __init__(
        self,
        name: str,
        message_stream: MessageStream,
        system_message: Dict,
        llm_config: Dict[str, Any],
        chat_history: Optional[List[Dict]] = None,
    ) -> None:
        self.name = name
        self.message_stream = message_stream
        self.system_message = system_message
        self.llm_config = llm_config
        self.chat_history = chat_history or []


class LLMGroupChatMessage(Message):
    def __init__(self, sender_name: str, message: Dict) -> None:
        self.sender_name = sender_name
        self.message = message

    def __repr__(self) -> str:
        return f"LLMGroupChatMessage(sender_name={self.sender_name}, message={self.message})"


class LLMGroupChatMessageStream(ListMessageStream):
    def add_subscriber(self, agent: LLMGroupChatAgent) -> None:
        return super().add_subscriber(agent.name, agent)


class LLMGroupChatAgent(SingleStateAgent):
    def __init__(
        self,
        name: str,
        message_stream: LLMGroupChatMessageStream,
        system_message: Dict,
        llm_config: Dict[str, Any],
    ) -> None:
        super().__init__(
            initial_contexts=[
                LLMGroupChatContext(
                    name,
                    message_stream,
                    system_message,
                    llm_config,
                )
            ]
        )
        self.name = name
        message_stream.add_subscriber(self)


def llm_group_chat_action(message: LLMGroupChatMessage, context: LLMGroupChatContext) -> LLMGroupChatContext:
    response = oai.ChatCompletion.create(
        messages=[context.system_message, *context.chat_history, message.message], **context.llm_config
    )
    reply_text = response.choices[0]["message"]["content"]
    reply = {"role": "user", "content": reply_text, "name": context.name}
    context.message_stream.broadcast(
        LLMGroupChatMessage(sender_name=context.name, message=reply),
    )
    reply["role"] = "assistant"
    return LLMGroupChatContext(
        name=context.name,
        message_stream=context.message_stream,
        system_message=context.system_message,
        llm_config=context.llm_config,
        chat_history=[*context.chat_history, message.message, reply],
    )
