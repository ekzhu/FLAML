from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple
import logging

from flaml.autogen.agentchat2.address import Address
from flaml.autogen.agentchat2.agent import Agent
from flaml.autogen.agentchat2.message import Message

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


class MessageStream(ABC):
    @abstractmethod
    def add_subscriber(
        self,
        address: Address,
        agent: Agent,
        subscription_policy: Callable[[Message], bool],
    ) -> None:
        pass

    @abstractmethod
    def send(
        self,
        address: Address,
        message: Message,
        delivery_policy: Callable[[Address, Agent], bool],
    ) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass


class ListMessageStream(MessageStream):
    def __init__(self) -> None:
        self._subscribers: Dict[Address, List[Tuple[Agent, Callable[[Message], bool]]]] = dict()
        self._buffers: Dict[Address, Dict[Agent, List[Message]]] = dict()

    def add_subscriber(
        self,
        address: Address,
        agent: Agent,
        subscription_policy: Callable[[Message], bool],
    ) -> None:
        logger.debug(f"Add subscriber {agent} to address {address}")
        self._subscribers.setdefault(address, []).append((agent, subscription_policy))

    def send(
        self,
        address: Address,
        message: Message,
        delivery_policy: Callable[[Address, Agent], bool],
    ) -> None:
        if address not in self._subscribers:
            logger.warning(f"{message} sent to unknown address {address} is discarded.")
            return
        if len(self._subscribers[address]) == 0:
            logger.warning(f"{message} sent to address {address} with no subscribers is discarded.")
            return
        logger.debug(f"Send\n{message}\nto address {address}")
        for reciever, subscription_policy in self._subscribers[address]:
            if delivery_policy(address, reciever) and subscription_policy(message):
                self._buffers.setdefault(address, {}).setdefault(reciever, []).append(message)

    def run(self) -> None:
        while any(len(buffer) > 0 for address_buffers in self._buffers.values() for buffer in address_buffers.values()):
            for address, address_buffers in list(self._buffers.items()):
                for reciever, buffer in list(address_buffers.items()):
                    reciever.handle(buffer)
                    buffer.clear()
