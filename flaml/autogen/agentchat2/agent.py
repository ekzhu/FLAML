from abc import ABC, abstractmethod
from enum import Enum, EnumType
from typing import Callable, Dict, List, Tuple, Union

from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message
from flaml.autogen.agentchat2.nfa import NFA


class Agent(ABC):
    """An agent is an abstract message event handler."""

    @abstractmethod
    def handle(self, message: Message) -> None:
        pass


class SingleStateAgent(Agent):
    """A single-state agent is a single-state NFA. It can be used to model
    infinite loop behaviors. For example, a LLM-powered chat bot that always
    responds to user input with LLM-generated text.

    Args:
        initial_contexts (List[Context]): The initial contexts of the agent.

    Raises:
        TypeError: If initial_contexts is not List[Context] type.
    """

    states = Enum("State", ["WAITING_FOR_INPUT"])

    def __init__(self, initial_contexts: List[Context]) -> None:
        self._nfa = NFA({self.states.WAITING_FOR_INPUT: initial_contexts})

    def register(
        self,
        trigger: Callable[[Message, Context], bool],
        action: Callable[[Message, Context], Context],
    ) -> None:
        """Register an agent action triggered by a message.

        Args:
            trigger (Callable[[Message, Context], bool]): A function that
                returns True if the action should be triggered.
            action (Callable[[Message, Context], Context]): A function that
                returns the updated context after the action is performed.

        Raises:
            TypeError: If trigger is not Callable type.
            TypeError: If action is not Callable type.
        """
        self._nfa.register(
            self.states.WAITING_FOR_INPUT,
            self.states.WAITING_FOR_INPUT,
            trigger,
            action,
        )

    def handle(self, message: Message) -> None:
        return self._nfa.process(message)

    def get_contexts(self) -> List[Context]:
        """Get all contexts of the agent.

        Returns:
            List[Context]: A list of contexts.
        """
        return self._nfa.get_contexts(self.states.WAITING_FOR_INPUT)


class MultiStateAgent(NFA, Agent):
    """A multi-state agent is an NFA."""

    def handle(self, message: Message) -> None:
        return self.process(message)
