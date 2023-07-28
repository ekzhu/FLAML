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
