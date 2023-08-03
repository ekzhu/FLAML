from __future__ import annotations
from typing import Any, Dict, List, Optional
from flaml.autogen import oai
from flaml.autogen.agentchat2.agent import SingleStateAgent
from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message
from flaml.autogen.agentchat2.stream import ListMessageStream, MessageStream


class LLMChatContext(Context):
    def __init__(
        self,
        name: str,
        message_stream: MessageStream,
        partner_name: str,
        system_message: Dict,
        llm_config: Dict[str, Any],
        chat_history: Optional[List[Dict]] = None,
    ) -> None:
        self.name = name
        self.message_stream = message_stream
        self.partner_name = partner_name
        self.system_message = system_message
        self.llm_config = llm_config
        self.chat_history = chat_history or []


class LLMChatMessage(Message):
    def __init__(self, sender_name: str, receiver_name: str, message: Dict) -> None:
        self.sender_name = sender_name
        self.receiver_name = receiver_name
        self.message = message

    def __repr__(self) -> str:
        return f"LLMChatMessage(sender_name={self.sender_name}, receiver_name={self.receiver_name}, message={self.message})"


class LLMChatMessageStream(ListMessageStream):
    def add_subscriber(self, agent: LLMChatAgent) -> None:
        return super().add_subscriber(agent.name, agent)

    def send(self, message: LLMChatMessage) -> None:
        return super().send(message.receiver_name, message)


class LLMChatAgent(SingleStateAgent):
    def __init__(
        self,
        name: str,
        message_stream: LLMChatMessageStream,
        partner_name: str,
        system_message: Dict,
        llm_config: Dict[str, Any],
    ) -> None:
        super().__init__(
            initial_context=LLMChatContext(
                name,
                message_stream,
                partner_name,
                system_message,
                llm_config,
            )
        )
        self.name = name


def llm_chat_trigger(message: LLMChatMessage, context: LLMChatContext) -> bool:
    return message.sender_name == context.partner_name and len(context.chat_history) < 10


def llm_chat_action(message: LLMChatMessage, context: LLMChatContext) -> LLMChatContext:
    response = oai.ChatCompletion.create(
        messages=[context.system_message, *context.chat_history, message.message], **context.llm_config
    )
    reply_text = response.choices[0]["message"]["content"]
    reply = {"role": "user", "content": reply_text}
    context.message_stream.send(
        LLMChatMessage(sender_name=context.name, receiver_name=context.partner_name, message=reply),
    )
    return LLMChatContext(
        name=context.name,
        message_stream=context.message_stream,
        partner_name=context.partner_name,
        system_message=context.system_message,
        llm_config=context.llm_config,
        chat_history=[*context.chat_history, message.message, reply],
    )
