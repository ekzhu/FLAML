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
        trigger_function: Callable[[List[Message], Context], bool],
        action_function: Callable[[List[Message], Context], Context],
    ) -> None:
        self.source_state = source_state
        self.target_state = target_state
        self.trigger_function = trigger_function
        self.action_function = action_function


class Automaton:
    """An automaton is a finate state machine that can be used to model complex
    agent behaviors involving multiple states and isolated contexts.

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
        self._default_actions: Dict[Enum, List[_Transition]] = dict()
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

    def register_action(
        self,
        source_state: Union[Enum, Tuple[Enum]],
        target_state: Union[Enum, Tuple[Enum]],
        trigger: Callable[[List[Message], Context], bool],
        action: Callable[[List[Message], Context], Context],
    ) -> None:
        """Register a state transition defined by a trigger and an action.

        Args:
            source_state (Union[Enum, Tuple[Enum]]): Source state.
            target_state (Union[Enum, Tuple[Enum]]): Target state.
            trigger (Callable[[List[Message], Context], bool]): Trigger function that
                returns a boolean value indicating whether the state transition
                should be taken.
            action (Callable[[List[Message], Context], Context]): Action function
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
                self._actions.setdefault(src, [])
                self._actions.setdefault(tgt, [])
                self._actions[src].append(
                    _Transition(
                        src,
                        tgt,
                        trigger,
                        action,
                    )
                )

    def register_default_action(
        self,
        source_state: Union[Enum, Tuple[Enum]],
        target_state: Union[Enum, Tuple[Enum]],
        action: Callable[[List[Message], Context], Context],
    ):
        # Check types.
        if not isinstance(source_state, Enum) and not isinstance(source_state, tuple):
            raise TypeError(f"State must be Enum or Tuple type, but {type(source_state)}")
        if not isinstance(target_state, Enum) and not isinstance(target_state, tuple):
            raise TypeError(f"State must be Enum or Tuple type, but {type(target_state)}")
        if not isinstance(action, Callable):
            raise TypeError(f"Action must be Callable type, but {type(action)}")

        if isinstance(source_state, Enum):
            source_state = (source_state,)
        if isinstance(target_state, Enum):
            target_state = (target_state,)

        for src in source_state:
            for tgt in target_state:
                self._default_actions.setdefault(src, [])
                self._default_actions.setdefault(tgt, [])
                self._default_actions[src].append(
                    _Transition(
                        src,
                        tgt,
                        lambda messages, context: True,
                        action,
                    )
                )

    def process(self, messages: List[Message]) -> None:
        """Process incoming messages and make state transitions if necessary.

        Args:
            messages (List[Message]): A list of new messages to be processed.

        Raises:
            TypeError: If messages is not List type.
            TypeError: If messages's element is not Message type.
        """
        # Check types.
        if not isinstance(messages, list):
            raise TypeError(f"Messages must be List type, but {type(messages)}")
        for message in messages:
            if not isinstance(message, Message):
                raise TypeError(f"Message must be Message type, but {type(message)}")
        new_contexts = {}
        for state, contexts in self._contexts.items():
            if state not in self._actions and state not in self._default_actions:
                # No action is registered for the state.
                continue
            for context in contexts:
                handled = False
                for action in self._actions[state]:
                    if action.trigger_function(messages, context):
                        new_context = action.action_function(messages, context)
                        new_contexts.setdefault(action.target_state, []).append(new_context)
                        handled = True
                if not handled and state in self._default_actions:
                    # Execute default actions if this context is not handled by any action.
                    for action in self._default_actions[state]:
                        new_context = action.action_function(messages, context)
                        new_contexts.setdefault(action.target_state, []).append(new_context)
        # Old context is discard if not handled.
        self._contexts = new_contexts

    def get_contexts(self, state: Enum) -> List[Context]:
        """Get contexts associated with the given state.

        Args:
            state (Enum): A state.

        Raises:
            TypeError: If state is not Enum type.

        Returns:
            List[Context]: A list of contexts associated with the given state.
        """
        # Check types.
        if not isinstance(state, Enum):
            raise TypeError(f"State must be Enum type, but {type(state)}")
        if state not in self._contexts:
            return []
        return self._contexts[state]
