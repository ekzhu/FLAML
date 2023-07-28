# Evaluation on Math problem solving

MathChat simulates conversations between a LLM assistant and a user proxy agent, which can easily be implemented with the *AgentChat* framework.
We evaluate MathChat with several other potential math problem-solving tools, including *Auto-GPT*, *ChatGPT+Plugin*, *LangChain ReAct+Python*.

Setup:
For all applications, we use GPT-4 as the base model, and "sympy" package is pre-installed.
- Auto-GPT: We use the out-of-box Auto-GPT. At the beginning, we set the purpose of the GPT to "solve math problems", resulting in a  "MathSolverGPT" with auto-generated goals.
- ChatGPT+Plugin: the Wolfram Alpha plugin is used.
- LangChain ReAct+Python: We use the ReAct+Python chain in LangChain. The ReAct agent is customized to solve math problems.

Qualitative Evaluation:
We random select 2 level-5 problems from the MathChat dataset and test on them for 3 times. The first problem asks to simply a square root fraction, and the second problem ask to solve an inclusion-exlcusion problem. 

The table shows an overview of the results:

**Problem 1**
| Application      | Correct Count | Tool Used | Failed Reason                                                                 |
|------------------|---------------|-----------|-------------------------------------------------------------------------------|
| MathChat         | 3/3           | Python    | N\A                                                                          |
| Auto-GPT         | 0/3           | Python    | The LLM gives code without the print function so the result is not printed    |
| Langchain ReAct  | 0/3           | Python    | LangChain gives 3 different wrong answers                                     |
| ChatGPT+Plugin   | 0/3           | Wolfram   | The return from Wolfram contains 2 simplied result, including the correct answer, but GPT-4 always chooses the wrong answer |

**Problem 2**
| Application      | Correct Count | Tool Used | Failed Reason                                                                 |
|------------------|---------------|-----------|-------------------------------------------------------------------------------|
| MathChat         | 3/3           | Python    | N\A                                                                          |
| Auto-GPT         | 0/3           | Python    | The LLM gives code without the print function so the result is not printed    |
| Langchain ReAct  | 2/3           | Python    | Got one parsing exception during the test                                     |
| ChatGPT+Plugin   | 2/3           | Wolfram   | GPT-4 got stuck by keep giving the wrong querying and has to be stopped.       |

We analyse the results in terms of correctness, verbosity and experience.
- Correctness: *MathChat* can always solves the problem the two problems correctly. *ChatGPT+Plugin* and *Langchain ReAct* can solve the second problem almost certainly, but fails on the first problem. *Auto-GPT* fails on both problems due to a code execution issue. The LLM always give the code without the print function, so the result is not printed. For the first problem, we tried to run the code snippets manually from one trial and found that the result is correct.
- Verbosity/Length: *ChatGPT+Plugin* is the least verbose since it write queries to Wolfram Alpha instead of Python code. *MathChat* is less verbose than *Langchain ReAct* since it has less errors to be corrected. *Auto-GPT* is the most verbose with predefined steps like THOUGHTS, REASONING, PLAN. Verbosity rank: *ChatGPT+Plugin* < *MathChat* < *Langchain ReAct* < *Auto-GPT*
- Experience/Readabilty: We analyse the experience of using these applications in terms of generated content without considering the interface design. The verbosity of *Auto-GPT* makes it harder to read, and the printing issue result in a loop that need to be manually stopped. Other applications are easier to read. *ChatGPT+Plugin* also has the potential to stuck in a loop that needs to be manually stopped. *Langchain ReAct* could finish with error, which is undesirable.


## Observations
*Auto-GPT*
- It is easy to use AutoGPT as an automatic tool. Given a intention, it will build its only name, role, goals and start the process.
- AutoGPT pays attention to security: ask for authorisation before execution, don't allow install python packages.
- AutoGPT should cost more with predefined steps: THOUGHTS, REASONING, PLAN, CRITICISM, SPEAK, NEXT ACTION
- AutoGPT seems like an on-the-shelf tool that is hard to modify. It do allow adding plugins.
- execute_code is a built-in functionality
- AutoGPT can run autogpt without human authorisation: `./run.sh --continuous` (from the document)

*CAMEL*
- The purpose of this project is different from our paper. Quote from website: "CAMEL-AI.org is an open-source community dedicated to the study of autonomous and communicative agents"
- Findings from a simple test with the Agent app on the website: 1. It is not solving the problem at all. 2. The conversation is automatic between two agents.
- Addtional test on [colab notebook](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing): try toggling around some parameters (such as changing model to GPT-4), it is still not solving the problem

*ChatGPT+Plugin*
- GPT-4 + Wolfram Alpha is competitive in solving math problems. It can solve the problem correctly by directly querying wolfram.
- GPT-4 + Wolfram Alpha can sometimes get stuck sending the same error query, which can only be stopped manually.

*LangChain*
- Langchain has several math related chains: Math chain, Symbolic Math chain, PALChain. The ReAct agent is related to problem solving.
- A customized ReAct+Python is simlar to MathChat. The main difference is that ReAct+Python is viewed as one agent, but MathChat simluates interactions between two agents.

## Summary
*Auto-GPT*, *ChatGPT+Plugin* and *LangChain ReAct+Python* can be useful in solving math problems. *Auto-GPT* and *LangChain ReAct+Python* uses Python as tool while *ChatGPT+Plugin* uses Wolfram Alpha. 


*Auto-GPT* is packed into an on-the-shelf tool that is easy to use, which discourages customization. It takes some security measures to prevent malicious code execution. It is alsoo harder t customize *ChatGPT+Plugin* since it is an official release. *LangChain* as an framework allow flexibility to customize chains. However, through reading the documents, it seems much hard to create a complete new chain comparing to adapt from langchain examples.