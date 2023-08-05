import sys
from typing import Callable, Dict, List, Optional, Union
from flaml.autogen.agentchat2.agent import SingleStateAgent
from flaml.autogen.agentchat.responsive_agent import ResponsiveAgent as LegacyResponsiveAgent
from flaml.autogen.agentchat.assistant_agent import AssistantAgent as LegacyAssistantAgent
from flaml.autogen.agentchat.groupchat import GroupChatManager as LegacyGroupChatManager
from flaml.autogen.agentchat.agent import Agent as LegacyAgent
from flaml.autogen.agentchat2.context import Context


class ResponsiveAgentContext(Context, LegacyResponsiveAgent):
    pass


class GroupChatManagerContext(Context, LegacyGroupChatManager):
    pass


class ResponsiveAgent(SingleStateAgent, LegacyAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[str] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,
        llm_config: Optional[Union[Dict, bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
    ) -> None:
        super().__init__(
            initial_contexts=[
                ResponsiveAgentContext(
                    name,
                    system_message,
                    is_termination_msg,
                    max_consecutive_auto_reply,
                    human_input_mode,
                    function_map,
                    code_execution_config,
                    llm_config,
                    default_auto_reply,
                )
            ]
        )
        LegacyAgent.__init__(self, name)

    def _get_context(self) -> ResponsiveAgentContext:
        return super()._get_context()[0]

    def register_auto_reply(self, class_type, reply_func: Callable, position: int = 0):
        self._get_context().register_auto_reply(class_type, reply_func, position)

    @property
    def system_message(self):
        return self._get_context().system_message

    def update_system_message(self, system_message: str):
        return self._get_context().update_system_message(system_message)

    def update_max_consecutive_auto_reply(
        self,
        value: int,
        sender: Optional[LegacyAgent] = None,
    ):
        return self._get_context().update_max_consecutive_auto_reply(value, sender)

    def max_consecutive_auto_reply(self, sender: Optional[LegacyAgent] = None) -> int:
        return self._get_context().max_consecutive_auto_reply(sender)

    @property
    def chat_messages(self) -> Dict[str, List[Dict]]:
        return self._get_context().chat_messages

    def last_message(self, agent: Optional[LegacyAgent] = None) -> Dict:
        return self._get_context().last_message(agent)

    @property
    def use_docker(self) -> Union[bool, str, None]:
        return self._get_context().use_docker

    def send(
        self,
        message: Union[Dict, str],
        recipient: LegacyAgent,
        request_reply: Optional[bool] = None,
    ) -> bool:
        return self._get_context().send(message, recipient, request_reply)

    def receive(
        self,
        message: Union[Dict, str],
        sender: LegacyAgent,
        request_reply: Optional[bool] = None,
    ):
        return self._get_context().receive(message, sender, request_reply)

    def initiate_chat(
        self,
        recipient: LegacyResponsiveAgent,
        clear_history: Optional[bool] = True,
        **context,
    ):
        return self._get_context().initiate_chat(recipient, clear_history, **context)

    def reset(self):
        self._get_context().reset()

    def stop_reply_at_receive(self, sender: Optional[LegacyAgent] = None):
        return self._get_context().stop_reply_at_receive(sender)

    def reset_consecutive_auto_reply_counter(self, sender: Optional[LegacyAgent] = None):
        return self._get_context().reset_consecutive_auto_reply_counter(sender)

    def clear_history(self, agent: Optional[LegacyAgent] = None):
        return self._get_context().clear_history(agent)

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[LegacyAgent] = None,
        exclude: Optional[List[Callable]] = None,
    ) -> Union[str, Dict, None]:
        return self._get_context().generate_reply(messages, sender, exclude)

    def get_human_input(self) -> str:
        return self._get_context().get_human_input()

    def run_code(self, code, **kwargs):
        return self._get_context().run_code(code, **kwargs)

    def execute_code_blocks(self, code_blocks):
        return self._get_context().execute_code_blocks(code_blocks)

    def execute_function(self, func_call):
        return self._get_context().execute_function(func_call)

    def generate_init_message(self, **context) -> Union[str, Dict]:
        return self._get_context().generate_init_message(**context)

    def register_function(self, function_map: Dict[str, Callable]):
        return self._get_context().register_function(function_map)


class AssistantAgent(ResponsiveAgent):
    def __init__(
        self,
        name: str,
        system_message: Optional[str] = LegacyAssistantAgent.DEFAULT_SYSTEM_MESSAGE,
        llm_config: Optional[Union[Dict, bool]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, bool]] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs,
        )


class UserProxyAgent(ResponsiveAgent):
    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Optional[Union[Dict, bool]] = None,
        default_auto_reply: Optional[Union[str, Dict, None]] = "",
        llm_config: Optional[Union[Dict, bool]] = False,
        system_message: Optional[str] = "",
    ) -> None:
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            function_map,
            code_execution_config,
            llm_config,
            default_auto_reply,
        )


class GroupChatManager(ResponsiveAgent):
    def __init__(
        self,
        max_round: Optional[int] = 10,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager.",
        **kwargs,
    ):
        SingleStateAgent.__init__(
            self,
            initial_contexts=[
                GroupChatManagerContext(
                    max_round=max_round,
                    name=name,
                    max_consecutive_auto_reply=max_consecutive_auto_reply,
                    human_input_mode=human_input_mode,
                    system_message=system_message,
                    # seed=seed,
                    **kwargs,
                )
            ],
        )

    def agent_by_name(self, name: str) -> LegacyAgent:
        return self._get_context().agent_by_name(name)

    def reset(self):
        return self._get_context().reset()
