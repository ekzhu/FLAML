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
    def add_subscriber(self, address: Address, agent: Agent) -> None:
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
    def broadcast(
        self,
        message: Message,
        delivery_policy: Callable[[Address, Agent], bool],
    ) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass


class ListMessageStream(MessageStream):
    def __init__(self) -> None:
        self._subscribers: Dict[Any, List[Agent]] = dict()
        self._messages: List[Tuple[Address, Message, Callable[[Address, Agent], bool]]] = []

    def add_subscriber(self, address: Address, agent: Agent) -> None:
        logger.debug(f"Adding subscriber {agent} to address {address}")
        self._subscribers.setdefault(address, []).append(agent)

    def send(
        self,
        address: Address,
        message: Message,
        delivery_policy: Callable[[Address, Agent], bool],
    ) -> None:
        logger.debug(f"Send message {message} to address {address}")
        self._messages.append((address, message, delivery_policy))

    def broadcast(
        self,
        message: Message,
        delivery_policy: Callable[[Address, Agent], bool],
    ) -> None:
        logger.debug(f"Broadcast message {message}")
        for address in self._subscribers:
            self._messages.append((address, message, delivery_policy))

    def run(self) -> None:
        while len(self._messages) > 0:
            address, message, delivery_policy = self._messages.pop(0)
            if address not in self._subscribers:
                raise ValueError(f"Message {message} sent to unknown address {address}")
            if len(self._subscribers[address]) == 0:
                raise ValueError(f"Message {message} sent to empty address {address}")
            for reciever in self._subscribers[address]:
                if delivery_policy(address, reciever):
                    reciever.handle(message)
