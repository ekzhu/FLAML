from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List

from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message
from flaml.autogen.agentchat2.automaton import Automaton


class Agent(ABC):
    """An agent is an abstract message handler."""

    @abstractmethod
    def handle(self, messages: List[Message]) -> None:
        pass


class SingleStateAgent(Agent):
    """A single-state agent is a single-state automaton. It can be used to model
    infinite loop behaviors. For example, a LLM-powered chat bot that always
    responds to user input with LLM-generated text.

    Args:
        initial_contexts (List[Context]): The initial contexts of the agent.

    Raises:
        TypeError: If initial_contexts is not List[Context] type.
    """

    states = Enum("State", ["WAITING_FOR_INPUT"])

    def __init__(self, initial_contexts: List[Context]) -> None:
        self._automaton = Automaton({self.states.WAITING_FOR_INPUT: initial_contexts})

    def register_action(
        self,
        trigger: Callable[[List[Message], Context], bool],
        action: Callable[[List[Message], Context], Context],
    ) -> None:
        """Register an agent action triggered by a list of messages.

        Args:
            trigger (Callable[[List[Message], Context], bool]): A function that
                returns True if the action should be triggered.
            action (Callable[[List[Message], Context], Context]): A function that
                returns the updated context after the action is performed.

        Raises:
            TypeError: If trigger is not Callable type.
            TypeError: If action is not Callable type.
        """
        self._automaton.register_action(
            self.states.WAITING_FOR_INPUT,
            self.states.WAITING_FOR_INPUT,
            trigger,
            action,
        )

    def register_default_action(
        self,
        action: Callable[[List[Message], Context], Context],
    ) -> None:
        self._automaton.register_default_action(
            self.states.WAITING_FOR_INPUT,
            self.states.WAITING_FOR_INPUT,
            action,
        )

    def handle(self, messages: List[Message]) -> None:
        return self._automaton.process(messages)

    def get_contexts(self) -> List[Context]:
        """Get all contexts of the agent.

        Returns:
            List[Context]: A list of contexts.
        """
        return self._automaton.get_contexts(self.states.WAITING_FOR_INPUT)


class MultiStateAgent(Automaton, Agent):
    """A multi-state agent is an Automaton."""

    def handle(self, messages: List[Message]) -> None:
        return self.process(messages)
