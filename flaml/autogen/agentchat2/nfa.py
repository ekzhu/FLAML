from __future__ import annotations
from enum import Enum
from typing import Callable, Dict, List, Tuple, Union

from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message


class _Transition:
    def __init__(
        self,
        source_state: Enum,
        target_state: Enum,
        trigger_function: Callable[[Message, Context], bool],
        action_function: Callable[[Message, Context], Context],
    ) -> None:
        self.source_state = source_state
        self.target_state = target_state
        self.trigger_function = trigger_function
        self.action_function = action_function


class NFA:
    """A NFA is a non-deterministic finite automaton that can be used to model
    complex agent behaviors involving multiple states and isolated contexts.

    Args:
        start_context (Dict[Enum, List[Context]]): A dictionary mapping from
            initial states to a list of initial contexts.

    Raises:
        TypeError: If start_context is not Dict type.
        TypeError: If start_context's key is not Enum type.
        TypeError: If start_context's value is not List type.
        TypeError: If start_context's value's element is not Context type.
    """

    def __init__(self, start_context: Dict[Enum, List[Context]]) -> None:
        self._actions: Dict[Enum, List[_Transition]] = dict()
        # Check types.
        for state, contexts in start_context.items():
            if not isinstance(state, Enum):
                raise TypeError(f"State must be Enum type, but {type(state)}")
            if not isinstance(contexts, list):
                raise TypeError(f"Contexts must be List type, but {type(contexts)}")
            for context in contexts:
                if not isinstance(context, Context):
                    raise TypeError(f"Context must be Context type, but {type(context)}")
        self._contexts: Dict[Enum, List[Context]] = start_context

    def register(
        self,
        source_state: Union[Enum, Tuple[Enum]],
        target_state: Union[Enum, Tuple[Enum]],
        trigger: Callable[[Message, Context], bool],
        action: Callable[[Message, Context], Context],
    ) -> None:
        """Register a state transition.

        Args:
            source_state (Union[Enum, Tuple[Enum]]): Source state.
            target_state (Union[Enum, Tuple[Enum]]): Target state.
            trigger (Callable[[Message, Context], bool]): Trigger function that
                returns a boolean value indicating whether the state transition
                should be taken.
            action (Callable[[Message, Context], Context]): Action function
                executed during state transition. It returns a new context
                object that will be associated with the target state.

        Raises:
            TypeError: If source_state or target_state is not Enum or Tuple type.
            TypeError: If trigger or action is not Callable type.
        """
        # Check types.
        if not isinstance(source_state, Enum) and not isinstance(source_state, tuple):
            raise TypeError(f"State must be Enum or Tuple type, but {type(source_state)}")
        if not isinstance(target_state, Enum) and not isinstance(target_state, tuple):
            raise TypeError(f"State must be Enum or Tuple type, but {type(target_state)}")
        if not isinstance(trigger, Callable):
            raise TypeError(f"Trigger must be Callable type, but {type(trigger)}")
        if not isinstance(action, Callable):
            raise TypeError(f"Action must be Callable type, but {type(action)}")

        if isinstance(source_state, Enum):
            source_state = (source_state,)
        if isinstance(target_state, Enum):
            target_state = (target_state,)

        for src in source_state:
            for tgt in target_state:
                if src not in self._actions:
                    self._actions[src] = []
                if tgt not in self._actions:
                    self._actions[tgt] = []
                self._actions[src].append(
                    _Transition(
                        src,
                        tgt,
                        trigger,
                        action,
                    )
                )

    def handle(self, message: Message) -> None:
        """Process a incoming message and make state transitions if necessary.

        Args:
            message (Message): A message to be processed.

        Raises:
            TypeError: If message is not Message type.
        """
        # Check types.
        if not isinstance(message, Message):
            raise TypeError(f"Message must be Message type, but {type(message)}")
        new_contexts = {}
        for state, contexts in self._contexts.items():
            for context in contexts:
                for action in self._actions[state]:
                    if action.trigger_function(message, context):
                        new_context = action.action_function(message, context)
                        if action.target_state not in new_contexts:
                            new_contexts[action.target_state] = []
                        new_contexts[action.target_state].append(new_context)
        # Old context is discard.
        self._contexts = new_contexts
