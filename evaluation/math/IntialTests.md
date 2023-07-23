# Initial Tests on other LLM applications

## Auto-GPT
GitHub: https://github.com/Significant-Gravitas/Auto-GPT
Documents: https://docs.agpt.co/setup/
### 1. Observations
- It is easy to use AutoGPT as an automatic tool. Given a intention, it will build its only name, role, goals and start the process.
- AutoGPT pays attention to security: ask for authorisation before execution, don't allow install python packages.
- AutoGPT should cost more with predefined steps: THOUGHTS, REASONING, PLAN, CRITICISM, SPEAK, NEXT ACTION
- AutoGPT seems like an on-the-shelf tool that is hard to modify. It do allow adding plugins.

Similarities:
- execute_code is a built-in functionality
- AutoGPT can run autogpt without human authorisation: `./run.sh --continuous` (from the document)


### 2. Setup
Run autogpt in conda environment (it will install packages by itself):
See doc: https://docs.agpt.co/setup/

Clone repo:
```
# clone repo
cd evaluation
git clone -b stable https://github.com/Significant-Gravitas/Auto-GPT.git
cd AuTo-GPT
cp .env.template .env
```
Need to set up api key in .env file.
Then create env and run:
```
conda create -n autogpt python=3.10
conda activate autogpt
./run.sh
```


### 3. A Test case
Test Problem:
Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

Summary of Process:  
**System**: Got enter box "I want Auto-GPT to:"   
**Input**: solve math problems   
**System**: AutoGPT gives name, role, goals automatically, the "Role" might be a prompt.   
**MATHSOLVERGPT**(self-named): gives comments, REASONING, PLAN, CRITICISM, SPEAK, ask to provide a task or problem.   
**Input**: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.   
**MATHSOLVERGPT**: plans to write a Python script to solve the inequality. It gives the code and COMMAND, which needs user's authorisation.   
**Input**: y (authorise)   
**System**: Execution returns error: No module named 'sympy'.   
**MATHSOLVERGPT**: plans to solve without sympy, give another code.   
**Input**: y   
**System**: Got another error: NameError: name 'x' is not defined   
**MATHSOLVERGPT**: gives another code   
**Input**: Can you install sympy first?   
**MATHSOLVERGPT**: says it is not allowed to install python packages.   

Acutally Process:

`**Input**` means human input, texts after `**MATHSOLVERGPT**` is the main chat.
```
AutoGPT: Got enter box "I want Auto-GPT to:"
**Input**: solve math problems
AutoGPT:
NOTE:All files/directories created by this agent can be found inside its workspace at:  /home/ykw5399/msft/FLAML/evaluation/Auto-GPT/autogpt/workspace/auto_gpt_workspace
MathSolverGPT  has been created with the following details:
Name:  MathSolverGPT
Role:  an intelligent math-solving AI that assists students, professionals, and enthusiasts in solving a wide range of math problems, from basic arithmetic to advanced calculus and beyond.
Goals:
-  Provide step-by-step solutions to math problems, ensuring clarity and understanding for the user.
-  Offer explanations and insights into the underlying concepts and principles behind the solutions, fostering a deeper understanding of mathematics.
-  Handle a variety of math problem types, including algebra, geometry, trigonometry, calculus, and more.
-  Assist in solving complex and challenging math problems, offering guidance and strategies to approach difficult concepts.
-  Continuously learn and improve its problem-solving abilities to provide accurate and efficient solutions to math problems.
  Could not load MemoryItems from file: input length is 0: line 1 column 1 (char 0)
Using memory of type:  JSONFileMemory
Using Browser:  chrome



**MATHSOLVERGPT THOUGHTS**:  The user has not provided a specific task or question to address. Therefore, I cannot determine a command to use at this moment.
REASONING:  Without a specific task or question, it's impossible to determine an appropriate command to execute.
PLAN:
-  Wait for the user to provide a specific task or question
-  Analyze the task or question
-  Determine the appropriate command to execute
CRITICISM:  There is no action to criticize as no action has been taken yet.
SPEAK:  Could you please provide a specific task or question?

NEXT ACTION:  COMMAND =  ARGUMENTS = {}
  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for MathSolverGPT...
  Asking user via keyboard...

**Input**:Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.
SYSTEM:  Human feedback: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.
**MATHSOLVERGPT THOUGHTS**:  The user has asked to solve an inequality. This can be done by first simplifying the inequality, then finding the critical points, and finally testing the intervals between the critical points to find the solution set. This can be done using Python's symbolic computation library, sympy.
REASONING:  The task involves solving an inequality, which is a mathematical computation. Python, with the sympy library, is well-suited for this task.
PLAN:
-  Write a Python script to solve the inequality
-  Execute the Python script
CRITICISM:  None at this stage as the plan seems appropriate for the task at hand.
SPEAK:  Let's solve this inequality.

NEXT ACTION:  COMMAND = execute_python_code ARGUMENTS = {'code': "from sympy import symbols, solve, Interval, Union\n\nx = symbols('x')\n\n# Define the inequality\ninequality = (2*x + 10)*(x + 3) - (3*x + 9)*(x + 8)\n\n# Find the critical points\ncritical_points = solve(inequality, x)\n\n# Test the intervals between the critical points\ntest_points = [(a + b)/2 for a, b in zip(critical_points[:-1], critical_points[1:])]\n\n# Determine which intervals satisfy the inequality\nsolution_intervals = [Interval(a, b) for a, b, t in zip(critical_points[:-1], critical_points[1:], test_points) if inequality.subs(x, t) < 0]\n\n# Combine the solution intervals\nsolution = Union(*solution_intervals)\n\nsolution", 'name': 'solve_inequality'}
  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for MathSolverGPT...
  Asking user via keyboard...

**Input**:y
-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=
  Executing python file '/home/ykw5399/msft/FLAML/evaluation/Auto-GPT/autogpt/workspace/auto_gpt_workspace/MathSolverGPT/executed_code/solve_inequality.py' in working directory '/home/ykw5399/msft/FLAML/evaluation/Auto-GPT/autogpt/workspace/auto_gpt_workspace'
  Image 'python:3-alpine' found locally
SYSTEM:  Command execute_python_code returned: Traceback (most recent call last): File "/workspace/MathSolverGPT/executed_code/solve_inequality.py", line 1, in <module> from sympy import symbols, solve, Interval, Union ModuleNotFoundError: No module named 'sympy'


**MATHSOLVERGPT THOUGHTS**:  The Python environment does not have the 'sympy' module installed, which is necessary for solving the inequality. However, I can solve this inequality manually by expanding the expressions, moving all terms to one side to form a quadratic inequality, finding the roots of the corresponding quadratic equation, and then determining the intervals of x that satisfy the inequality.
REASONING:  Since the 'sympy' module is not available, I need to solve the inequality manually. This involves algebraic manipulation and understanding of how to solve quadratic inequalities.
PLAN:
-  Expand the expressions on both sides of the inequality
-  Move all terms to one side to form a quadratic inequality
-  Find the roots of the corresponding quadratic equation
-  Determine the intervals of x that satisfy the inequality
CRITICISM:  None at this stage as the plan seems appropriate for the task at hand.
SPEAK:  Let's solve this inequality manually.

NEXT ACTION:  COMMAND = execute_python_code ARGUMENTS = {'code': "def solve_inequality():\n # Coefficients of the quadratic inequality\n a = -x\n b = 25\n c = -114\n\n # Roots of the corresponding quadratic equation\n roots = sorted([(-b - (b**2 - 4*a*c)**0.5) / (2*a), (-b + (b**2 - 4*a*c)**0.5) / (2*a)])\n\n # Test points in the intervals between the roots\n test_points = [roots[0] - 1, (roots[0] + roots[1]) / 2, roots[1] + 1]\n\n # Determine which intervals satisfy the inequality\n solution_intervals = [(-float('inf'), roots[0]), (roots[0], roots[1]), (roots[1], float('inf'))]\n solution = [interval for interval, test_point in zip(solution_intervals, test_points) if a*test_point**2 + b*test_point + c < 0]\n\n return solution\n\nsolve_inequality()", 'name': 'solve_inequality'}
  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for MathSolverGPT...
  Asking user via keyboard...


**Input**:y
-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=
  Executing python file '/home/ykw5399/msft/FLAML/evaluation/Auto-GPT/autogpt/workspace/auto_gpt_workspace/MathSolverGPT/executed_code/solve_inequality.py' in working directory '/home/ykw5399/msft/FLAML/evaluation/Auto-GPT/autogpt/workspace/auto_gpt_workspace'
  Image 'python:3-alpine' found locally
SYSTEM:  Command execute_python_code returned: Traceback (most recent call last): File "/workspace/MathSolverGPT/executed_code/solve_inequality.py", line 19, in <module> solve_inequality() File "/workspace/MathSolverGPT/executed_code/solve_inequality.py", line 3, in solve_inequality a = -x ^ NameError: name 'x' is not defined


**MATHSOLVERGPT THOUGHTS**:  The error occurred because the variable 'x' was not defined in the Python code. I need to correct this mistake and try again.
REASONING:  The error message indicates that the variable 'x' was not defined. This is a programming error that needs to be corrected.
PLAN:
-  Correct the Python code by defining the variable 'x'
-  Execute the corrected Python code
CRITICISM:  I should have ensured that all variables used in the Python code were properly defined. This was a mistake.
SPEAK:  There was a mistake in the Python code. Let's correct it and try again.

NEXT ACTION:  COMMAND = execute_python_code ARGUMENTS = {'code': "def solve_inequality():\n # Variable x\n x = 0\n\n # Coefficients of the quadratic inequality\n a = -x\n b = 25\n c = -114\n\n # Roots of the corresponding quadratic equation\n roots = sorted([(-b - (b**2 - 4*a*c)**0.5) / (2*a), (-b + (b**2 - 4*a*c)**0.5) / (2*a)])\n\n # Test points in the intervals between the roots\n test_points = [roots[0] - 1, (roots[0] + roots[1]) / 2, roots[1] + 1]\n\n # Determine which intervals satisfy the inequality\n solution_intervals = [(-float('inf'), roots[0]), (roots[0], roots[1]), (roots[1], float('inf'))]\n solution = [interval for interval, test_point in zip(solution_intervals, test_points) if a*test_point**2 + b*test_point + c < 0]\n\n return solution\n\nsolve_inequality()", 'name': 'solve_inequality'}
  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for MathSolverGPT...
  Asking user via keyboard...

**Input**:Can you install sympy first?
SYSTEM:  Human feedback: Can you install sympy first?


**MATHSOLVERGPT THOUGHTS**:  The user has suggested installing the 'sympy' module. However, as an AI, I don't have the capability to install Python modules in the environment.
REASONING:  The user's suggestion is based on the assumption that I have the ability to install Python modules, which is not the case. I can only execute Python code with the modules that are already installed in the environment.
PLAN:
-  Explain to the user that I can't install Python modules
-  Continue with the plan to solve the inequality manually
CRITICISM:  I should have made it clear earlier that I can't install Python modules. This could have prevented the user's confusion.
SPEAK:  I'm sorry, but I don't have the capability to install Python modules. Let's continue with the plan to solve the inequality manually.

NEXT ACTION:  COMMAND =  ARGUMENTS = {}
  Enter 'y' to authorise command, 'y -N' to run N continuous commands, 'n' to exit program, or enter feedback for MathSolverGPT...
  Asking user via keyboard...
```

## CAMEL
Github: https://github.com/lightaime/camel
Website: https://www.camel-ai.org/
Arxiv Paper: https://arxiv.org/abs/2303.17760


General process of camel (Check the figure in the paper):
1. User (Human) give an idea and specify name of two AI agents.
2. A "Task Specifier" will generate a Specific Task with the idea.
3. The two agents will collobarate to solve this specific task.

### 1. Observations
- The purpose of this project is different from our paper. Quote from website: "CAMEL-AI.org is an open-source community dedicated to the study of autonomous and communicative agents"
- Findings from a simple test with the Agent app on the website: 1. It is not solving the problem at all. 2. The conversation is automatic between two agents.
- Addtional test on [colab notebook](https://colab.research.google.com/drive/1AzP33O8rnMW__7ocWJhVBXjKziJXPtim?usp=sharing): try toggling around some parameters (such as changing model to GPT-4), it is still not solving the problem



### 2. A Test Case
This is a test case on the same problem "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation. " with the [Agent app website](https://www.camel-ai.org/agent). It should be running gpt-3.5.

Settings:   
**Assitant role**: Math Solver that can use Python   
**User role**: Student   
**General Idea** (field "Give me a preliminary idea (EDIT ME) to the problem"):  Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

It is not solving the problem at all. First the systems gives a Specified task prompt, then there is a separate chat window betwee two agents:
```
# Specified task prompt given to the role-playing session based on the original (simplistic) idea
The Math Solver app is a powerful tool designed to assist students in solving a wide range of mathematical problems using Python. With its user-friendly interface and intelligent algorithms, it can transform the way students approach and complete their math assignments. Let's explore a specific scenario where the Math Solver can prove invaluable to a student:

Imagine Sarah, a high school student, is struggling with calculus and needs assistance in calculating derivative functions. She opens the Math Solver app and selects the "Derivative Calculation" feature. The app prompts her to input the function for which she wants to find the derivative. Sarah types in "f(x) = 3x^2 + 2x + 5."

Once Sarah submits the function, the Math Solver app's advanced parsing algorithm identifies the input as a quadratic function and begins the derivative calculation process. Utilizing Python's mathematical libraries and differentiation rules, the app takes each term of the function and applies the power rule to find the derivative of each term. Simultaneously, the app optimizes the calculation process, ensuring efficiency and accuracy.

After a few milliseconds, the Math Solver app generates the derivative function: "f'(x) = 6x + 2." Alongside the result, the app provides a step-by-step breakdown of the derivative calculation, empowering Sarah to understand the process thoroughly. This comprehensive approach helps her develop a strong grasp of calculus concepts.

The Math Solver app doesn't stop there. To further assist Sarah's learning, it visualizes the derivative function graph, allowing her to analyze its behavior. Additionally, the app offers interactive features like zooming in, finding critical points, and locating inflection points, enabling Sarah to explore the graph with ease.

Impressed by the Math Solver's efficiency, Sarah proceeds to use other built-in features. She discovers that the app can also solve integrals, perform matrix operations, solve equations, and even plot complex functions. With the Math Solver's assistance, Sarah gains confidence in her mathematical abilities, improves her problem-solving skills, and achieves better grades in her calculus class.

In conclusion, the Math Solver app powered by Python revolutionizes Sarah's math learning experience. Its ability to solve derivative functions accurately, generate detailed explanations, and provide interactive visualizations empowers students like Sarah to excel in their mathematics studies.


# Chats

**User Agent**:
Instruction: Implement a function in Python that takes a mathematical function as input and calculates its derivative using the power rule.
Input: None

**Assistant Agent**:
Solution:

To implement a function in Python that calculates the derivative of a mathematical function using the power rule, we can use the sympy library. Sympy is a powerful symbolic mathematics library that provides various mathematical operations, including differentiation.

Here’s an example implementation:

python
from sympy import symbols, diff

def calculate_derivative(function):
x = symbols('x')
derivative = diff(function, x)
return derivative

In this solution, we first import the necessary modules from the sympy library. We define a function called calculate_derivative that takes the mathematical function as input. Inside the function, we create a symbol x using the symbols function from sympy. Then, we use the diff function to calculate the derivative of the input function with respect to x. Finally, we return the derivative.

Next request.

**User Agent**:
Instruction: Test the calculate_derivative function by calculating the derivative of the function “f(x) = 3x^2 + 2x + 5”.
Input: None


**Assistant Agent**:
Solution:

To test the calculate_derivative function, we can pass the function “f(x) = 3x^2 + 2x + 5” as input and verify if the calculated derivative matches our expectations.

Here’s an example implementation:

python
def calculate_derivative(function):
x = symbols('x')
derivative = diff(function, x)
return derivative

# Test the calculate_derivative function
function = 3*x**2 + 2*x + 5
derivative = calculate_derivative(function)

print("The derivative of f(x) = 3x^2 + 2x + 5 is:", derivative)

When we run this code, the output will be:

The derivative of f(x) = 3x^2 + 2x + 5 is: 6*x + 2

The calculated derivative matches our expectations, which is “f’(x) = 6x + 2”.

Next request.
```


## ChatGPT Plus (GPT-4 + Plugin)
Website: https://chat.openai.com/

### 1. Observations

- GPT-4 + Wolfram Alpha is competitive in solving math problems. It can solve the problem correctly by directly querying wolfram.
- GPT-4 + Wolfram Alpha can sometimes get stuck sending the same error query, which can only be stopped manually.

### 2. Test Cases

**Test problem 1**: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

Setup: Choose GPT-4 + Wolfram Alpha Plugin.

ChatGPT Plus can solve the problem correctly by directly querying wolfram.
```
REQUEST TO WOLFRAM
{
  "input": "solve (2x+10)(x+3)<(3x+9)(x+8) for x"
}
RESPONSE FROM WOLFRAM
Query:
"solve (2x+10)(x+3)<(3x+9)(x+8) for x"

Input interpretation:
solve (2 x + 10) (x + 3)<(3 x + 9) (x + 8) for x

Results:
x<-14

x>-3

Number line:
image: https://www6b3.wolframalpha.com/Calculate/MSP/MSP15481h8039ib01hd1169000041ai8ae4372c42c8?MSPStoreType=image/png&s=9

Wolfram|Alpha website result for "solve (2x+10)(x+3)<(3x+9)(x+8) for x":
https://www.wolframalpha.com/input?i=solve+%282x%2B10%29%28x%2B3%29%3C%283x%2B9%29%28x%2B8%29+for+x


The solution to the inequality \((2x+10)(x+3)<(3x+9)(x+8)\) is \(x < -14\) or \(x > -3\). 

In interval notation, this is expressed as \((-∞, -14) \cup (-3, ∞)\).

Here is a number line representation of the solution:

![Number Line](https://www6b3.wolframalpha.com/Calculate/MSP/MSP15481h8039ib01hd1169000041ai8ae4372c42c8?MSPStoreType=image/png&s=9)

For more details, you can visit the [Wolfram|Alpha result page](https://www.wolframalpha.com/input?i=solve+%282x%2B10%29%28x%2B3%29%3C%283x%2B9%29%28x%2B8%29+for+x).
```


**Test Problem 2**: There are 190 people on the beach.  110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?

```
This is a problem of set theory and can be solved using the principle of inclusion and exclusion. The principle of inclusion and exclusion states that for any three sets A, B, and C, the size of their union can be found by adding the sizes of the sets, then subtracting the sizes of the pairwise intersections, and finally adding back the size of the intersection of all three.

Let's denote:
- A as the set of people wearing sunglasses
- B as the set of people wearing bathing suits
- C as the set of people wearing a hat

We know that:
- |A| = 110 (people wearing sunglasses)
- |B| = 70 (people wearing bathing suits)
- |C| = 95 (people wearing a hat)
- |A ∩ B| = 30 (people wearing both sunglasses and bathing suits)
- |A ∩ C| = 40 (people wearing both sunglasses and a hat)
- |B ∩ C| = 25 (people wearing both bathing suits and a hat)
- |A ∪ B ∪ C| = 190 (total people on the beach)

We can find the number of people wearing all three items (|A ∩ B ∩ C|) by rearranging the formula of the principle of inclusion and exclusion:

|A ∩ B ∩ C| = |A| + |B| + |C| - |A ∩ B| - |A ∩ C| - |B ∩ C| + |A ∪ B ∪ C|

Let's calculate this.

Apologies for the confusion. The symbol "C" is a built-in constant in the Wolfram Language, representing the speed of light, and cannot be reassigned. Let's use different symbols to represent the sets:

- S for the set of people wearing sunglasses
- B for the set of people wearing bathing suits
- H for the set of people wearing a hat

Let's calculate again.

REQUEST TO WOLFRAM
{
  "input": "S = 110; B = 70; H = 95; SB = 30; SH = 40; BH = 25; SBH_total = 190; SBH = S + B + H - SB - SH - BH + SBH_total; SBH"
}
RESPONSE FROM WOLFRAM
"180 + (SBH_total)"

I apologize for the confusion. It seems there was an error in the calculation. The variable "SBH_total" was not correctly replaced with its value. Let's try again.

REQUEST TO WOLFRAM
{
  "input": "S = 110; B = 70; H = 95; SB = 30; SH = 40; BH = 25; SBH_total = 190; SBH = S + B + H - SB - SH - BH + SBH_total; SBH"
}
RESPONSE FROM WOLFRAM
"180 + (SBH_total)"

I apologize for the confusion. It seems there was an error in the calculation. The variable "SBH_total" was not correctly replaced with its value. Let's try again.

REQUEST TO WOLFRAM
{
  "input": "S = 110; B = 70; H = 95; SB = 30; SH = 40; BH = 25; SBH_total = 190; SBH = S + B + H - SB - SH - BH + SBH_total; SBH"
}
RESPONSE FROM WOLFRAM
"180 + (SBH_total)"

I apologize for the confusion. It seems there was an error in the calculation. The variable "SBH_total" was not correctly replaced with its value. Let's try again.

...Manually stopped.
```

## LangChain
Github: https://github.com/hwchase17/langchain   
Docs: https://python.langchain.com/docs/get_started/introduction.html

### 1. Observations

- Langchain has several math related chains: Math chains, Symbolic Math chains, PALChain. The ReAct agent is related to problem solving.
- Among the chains tried, ReAct+Python solves both problem neatly.
- A major difference between ReAct+Python and MathChat is that ReAct+Python is viewed as one agent, but MathChat simluates interactions between two agents.

### 2. Tests
#### Test [Symbolic Math Chain](https://python.langchain.com/docs/modules/chains/additional/llm_symbolic_math)

Example code (See `test_langchain.py` for more chains)
```python
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm_symbolic_math.base import LLMSymbolicMathChain

llm = ChatOpenAI(model_name='gpt-4')
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)

math_problem = "" # fill the problem here
print(llm_symbolic_math.run(math_problem = ""))
```

**Test Problem 1**: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

The answer is correct:
```
Answer: ((-oo < x) & (x < -14)) | ((-3 < x) & (x < oo))
```


**Test Problem 2**: There are 190 people on the beach.  110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?

Got many errors when running the code. It seems the prompt specifically ask the LLM to use the "sympy" package.
```
ValueError: Error from parse_expr with transformed code: "Symbol ('Eq1' )=Eq (Symbol ('x' )+Symbol ('y' )+Symbol ('z' ),Integer (110 ))# sunglassesEq2 = Eq(x + w + z, 70)   # bathing suitsEq3 = Eq(w + y + z, 95)   # hatEq4 = Eq(x + z, 30)       # bathing suits and sunglassesEq5 = Eq(w + z, 25)       # bathing suits and hatEq6 = Eq(y + z, 40)       # sunglasses and hatsolve((Eq1, Eq2, Eq3, Eq4, Eq5, Eq6), (x, y, z, w))"

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/sympy/core/sympify.py", line 495, in sympify
    expr = parse_expr(a, local_dict=locals, transformations=transformations, evaluate=evaluate)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/sympy/parsing/sympy_parser.py", line 1087, in parse_expr
    raise e from ValueError(f"Error from parse_expr with transformed code: {code!r}")
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/sympy/parsing/sympy_parser.py", line 1078, in parse_expr
    rv = eval_expr(code, local_dict, global_dict)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/sympy/parsing/sympy_parser.py", line 906, in eval_expr
    expr = eval(
  File "<string>", line 1
    Symbol ('Eq1' )=Eq (Symbol ('x' )+Symbol ('y' )+Symbol ('z' ),Integer (110 ))# sunglassesEq2 = Eq(x + w + z, 70)   # bathing suitsEq3 = Eq(w + y + z, 95)   # hatEq4 = Eq(x + z, 30)       # bathing suits and sunglassesEq5 = Eq(w + z, 25)       # bathing suits and hatEq6 = Eq(y + z, 40)       # sunglasses and hatsolve((Eq1, Eq2, Eq3, Eq4, Eq5, Eq6), (x, y, z, w))
                   ^
SyntaxError: invalid syntax

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/llm_symbolic_math/base.py", line 64, in _evaluate_expression
    output = str(sympy.sympify(expression, evaluate=True))
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/sympy/core/sympify.py", line 497, in sympify
    raise SympifyError('could not parse %r' % a, exc)
sympy.core.sympify.SympifyError: Sympify of expression 'could not parse 'Eq1 = Eq(x + y + z, 110)  # sunglassesEq2 = Eq(x + w + z, 70)   # bathing suitsEq3 = Eq(w + y + z, 95)   # hatEq4 = Eq(x + z, 30)       # bathing suits and sunglassesEq5 = Eq(w + z, 25)       # bathing suits and hatEq6 = Eq(y + z, 40)       # sunglasses and hatsolve((Eq1, Eq2, Eq3, Eq4, Eq5, Eq6), (x, y, z, w))'' failed, because of exception being raised:
SyntaxError: invalid syntax (<string>, line 1)

During handling of the above exception, another exception occurred:

... More errors not shown
```
----------------------------------------------------------------------------------------------------------------------------

#### Test [PAL Chain](https://python.langchain.com/docs/modules/chains/additional/pal)

[Program-aided language model paper](https://arxiv.org/pdf/2211.10435.pdf)

**Test Problem 1**: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

Got error:
```
> Entering new PALChain chain...
def solution():
    """Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."""
    from sympy import *
    from sympy.abc import x
Traceback (most recent call last):
  File "/home/ykw5399/msft/FLAML/evaluation/math/test_math_chains.py", line 17, in <module>
    print(pal_chain.run(math_problem))
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/base.py", line 440, in run
    return self(args[0], callbacks=callbacks, tags=tags, metadata=metadata)[
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/base.py", line 243, in __call__
    raise e
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/base.py", line 237, in __call__
    self._call(inputs, run_manager=run_manager)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/pal/base.py", line 166, in _call
    PALChain.validate_code(code, self.code_validations)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/pal/base.py", line 261, in validate_code
    raise ValueError(f"Generated code has disallowed imports: {code}")
ValueError: Generated code has disallowed imports: def solution():
    """Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."""
    from sympy import *
    from sympy.abc import x
```


**Test Problem 2**: There are 190 people on the beach. 110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?

Correct answer is 10. Result is valid but wrong.
```
> Entering new PALChain chain...
def solution():
    """There are 190 people on the beach.  110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?"""
    total_people = 190
    sunglasses = 110
    bathing_suits = 70
    hats = 95
    both_bathing_suits_sunglasses = 30
    both_bathing_suits_hats = 25
    both_sunglasses_hats = 40
    all_three = (sunglasses + bathing_suits + hats) - (both_bathing_suits_sunglasses + both_bathing_suits_hats + both_sunglasses_hats)
    result = all_three
    return result
Python REPL can execute arbitrary code. Use with caution.

> Finished chain.
180
```


----------------------------------------------------------------------------------------------------------------------------
Test [math chain](https://python.langchain.com/docs/modules/chains/additional/llm_math)

**Test Problem 1**: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

Got error.
```
> Entering new LLMMathChain chain...
Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.```text
'2*x+10 < 3*x+9'
'x+3 < x+8'

...numexpr.evaluate("2*x+10 < 3*x+9")...
Traceback (most recent call last):
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/llm_math/base.py", line 80, in _evaluate_expression
    numexpr.evaluate(
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/numexpr/necompiler.py", line 817, in evaluate
    _names_cache[expr_key] = getExprNames(ex, context)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/numexpr/necompiler.py", line 704, in getExprNames
    ex = stringToExpression(text, {}, context)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/numexpr/necompiler.py", line 274, in stringToExpression
    c = compile(s, '<expr>', 'eval', flags)
  File "<expr>", line 2
    'x+3 < x+8'
    ^^^^^^^^^^^
SyntaxError: invalid syntax

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/ykw5399/msft/FLAML/evaluation/math/test_math_chains.py", line 26, in <module>
    print(llm_math.run(math_problem))
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/base.py", line 440, in run
    return self(args[0], callbacks=callbacks, tags=tags, metadata=metadata)[
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/base.py", line 243, in __call__
    raise e
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/base.py", line 237, in __call__
    self._call(inputs, run_manager=run_manager)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/llm_math/base.py", line 149, in _call
    return self._process_llm_result(llm_output, _run_manager)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/llm_math/base.py", line 103, in _process_llm_result
    output = self._evaluate_expression(expression)
  File "/home/ykw5399/.conda/envs/camel/lib/python3.10/site-packages/langchain/chains/llm_math/base.py", line 87, in _evaluate_expression
    raise ValueError(
ValueError: LLMMathChain._evaluate("
'2*x+10 < 3*x+9'
'x+3 < x+8'
") raised error: invalid syntax (<expr>, line 2). Please try again with a valid numerical expression
```

----------------------------------------------------------------------------------------------------------------------------
#### Test ReAct + PythonREPL
Langchain agent for ReAct example: https://python.langchain.com/docs/modules/agents/agent_types/react
ReAct: https://react-lm.github.io/

**Test Problem 1**: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

The result is correct:
```
> Entering new AgentExecutor chain...
To solve this, we need to use the principle of inclusion and exclusion. This principle states that in order to count the total of a union of different sets, we need to add the individual counts, then subtract the counts of the intersections of every two sets, and finally add the count of the intersection of all three sets. 

In this case, we can start by summing up the people wearing either sunglasses, bathing suits, or a hat. We then subtract the people wearing two of these items. However, since people wearing all three items were subtracted twice in the previous operation, we need to add them back in the final operation. 

So the formula would look like this: 

Total = Sunglasses + Bathing suits + Hat - (Sunglasses and Bathing suits) - (Sunglasses and Hat) - (Bathing suits and Hat) + All three items

We can rearrange this to solve for the people wearing all three items:

All three items = Total - Sunglasses - Bathing suits - Hat + (Sunglasses and Bathing suits) + (Sunglasses and Hat) + (Bathing suits and Hat)

We know all these values except for the number of people wearing all three items, so we can solve for it.
Action: Python_REPL
Action Input: total = 190
sunglasses = 110
bathing_suits = 70
hat = 95
sunglasses_bathing_suits = 30
sunglasses_hat = 40
bathing_suits_hat = 25

all_three = total - sunglasses - bathing_suits - hat + sunglasses_bathing_suits + sunglasses_hat + bathing_suits_hat

print(all_three)Python REPL can execute arbitrary code. Use with caution.

Observation: 10

Thought:I now know the final answer
Final Answer: 10

> Finished chain.
```


**Test Problem 2**: There are 190 people on the beach.  110 are wearing sunglasses, 70 are wearing bathing suits, and 95 are wearing a hat.  Everyone is wearing at least one of these items. 30 are wearing both bathing suits and sunglasses. 25 are wearing both bathing suits and a hat. 40 are wearing both sunglasses and a hat.  How many people are wearing all three items?

The result is correct:
```
The inequality can be solved by first expanding the expressions on both sides, then moving all terms to one side to create a polynomial inequality. Then, the roots of the polynomial equation can be found. The sign of the polynomial in the intervals determined by these roots will tell us where the inequality is satisfied.
Action: Python_REPL
Action Input: 
```python
from sympy import *

# Define the variable
x = symbols('x')

# Define the inequality
inequality = (2*x+10)*(x+3) - (3*x+9)*(x+8)

# Expand the inequality
expanded_inequality = expand(inequality)

# Find the roots of the polynomial equation
roots = solve(expanded_inequality, x)

# Sort the roots
roots.sort()

print(expanded_inequality)
print(roots)
```Python REPL can execute arbitrary code. Use with caution.

Observation: -x**2 - 17*x - 42
[-14, -3]

Thought:The expanded inequality is $-x^2 - 17x - 42 < 0$ and the roots of the corresponding equation are $-14$ and $-3$. Next, I should test the sign of the polynomial in each interval determined by the roots. The intervals are $(-\infty, -14)$, $(-14, -3)$, and $(-3, +\infty)$. I can pick a test point in each interval and substitute it into the polynomial to check the sign.
Action: Python_REPL
Action Input: 
```python
# Define the test points
test_points = [-15, -10, 0]

# Test the sign of the polynomial at each test point
signs = [expanded_inequality.subs(x, tp).evalf() > 0 for tp in test_points]

print(signs)
```End
Observation: [False, True, False]

Thought:The polynomial is negative at test points $-15$ and $0$ and positive at $-10$. This means the inequality is satisfied in the intervals $(-\infty, -14)$ and $(-3, +\infty)$. 
Final Answer: $(-\infty, -14) \cup (-3, +\infty)$

> Finished chain.
$(-\infty, -14) \cup (-3, +\infty)$
```