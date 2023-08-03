from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from flaml.autogen.agentchat2.address import Address
from flaml.autogen.agentchat2.nfa import NFA
from flaml.autogen.agentchat2.message import Message


class MessageStream(ABC):
    @abstractmethod
    def add_subscriber(self, address: Address, agent: NFA) -> None:
        pass

    @abstractmethod
    def send(self, address: Address, message: Message) -> None:
        pass

    @abstractmethod
    def broadcast(self, message: Message) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass


class ListMessageStream(MessageStream):
    def __init__(self) -> None:
        self._subscribers: Dict[Any, List[NFA]] = dict()
        self._messages: List[Tuple[Address, Message]] = []

    def add_subscriber(self, address: Address, agent: NFA) -> None:
        self._subscribers.setdefault(address, []).append(agent)

    def send(self, address: Address, message: Message) -> None:
        self._messages.append((address, message))

    def broadcast(self, message: Message) -> None:
        for address in self._subscribers:
            self._messages.append((address, message))

    def run(self) -> None:
        while len(self._messages) > 0:
            address, message = self._messages.pop(0)
            if address not in self._subscribers:
                raise ValueError(f"Message {message} sent to unknown address {address}")
            if len(self._subscribers[address]) == 0:
                raise ValueError(f"Message {message} sent to empty address {address}")
            for reciever in self._subscribers[address]:
                reciever.handle(message)
