
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-4')
math_problem1 = "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."
math_problem2 = "There are 190 people on the beach.  110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?" # fill the problem here

# # -----------------------------------------------
# # symbolic math
# from langchain.chains.llm_symbolic_math.base import LLMSymbolicMathChain
# llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)
# print(llm_symbolic_math.run(math_problem1))

# # -----------------------------------------------
# # PAL
# from langchain.chains import PALChain
# pal_chain = PALChain.from_math_prompt(llm, verbose=True)
# print(pal_chain.run(math_problem1))


# # -----------------------------------------------
# # math chain
# from langchain import LLMMathChain

# llm_math = LLMMathChain.from_llm(llm, verbose=True)
# print(llm_math.run(math_problem1))


# # -----------------------------------------------
# ReAct chain + Python REPL as tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool

tools = [PythonREPLTool()]
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
print(agent.run(math_problem1))


