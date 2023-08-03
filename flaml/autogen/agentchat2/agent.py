from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

from flaml.autogen.agentchat2.context import Context
from flaml.autogen.agentchat2.message import Message


class _Action:
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


class Agent:
    def __init__(self, start_context: Dict[Enum, List[Context]]) -> None:
        self._actions: Dict[Enum, List[_Action]] = dict()
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
                    _Action(
                        src,
                        tgt,
                        trigger,
                        action,
                    )
                )

    def handle(self, message: Message) -> None:
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
