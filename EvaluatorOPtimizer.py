from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import csv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import json
import re
import pandas as pd
from typing import List, Dict
import os
# Load environment variables
load_dotenv()
chatHistory =[]
global_chatHistory: List[Dict] = []

# Define the generator prompt template
generator_prompt = PromptTemplate(
    input_variables=["function_name", "description", "arguments"],  # Fixed typo
    template="""Write a high-quality Python function named {function_name} that {description}. 
    The function should have the following arguments: {arguments}. 

    *Instructions*
    - Focus on function only; do not generate anything else.
    - Function should be optimized and should cover all edge cases.
    - Function arguments are a dataframe and the name of a column in that dataframe. 
    - Use this information to compute the best results.
    """
)

# Initialize Gemini AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def generator(function_name, description, arguments):
    """
    Generates a Python function based on the given details.

    Args:
        function_name (str): Name of the function.
        description (str): Short description of what the function should do.
        arguments (str): List of expected arguments.

    Returns:
        str: Generated Python function as a string.
    """
    # Format the prompt
    global_chatHistory.append({"function_name": function_name})
    global_chatHistory.append({"arguments": arguments})
    prompt_text = generator_prompt.format_prompt(function_name=function_name, description=description, arguments=arguments).to_string()
    # Generate the function code using Gemini AI
    global_chatHistory.append({"generative_prompt": prompt_text})
    chatHistory.append(HumanMessage(content=prompt_text))
    result = llm.invoke(prompt_text)
    responce = result.content
    global_chatHistory.append({"generative_response": responce})
    chatHistory.append(AIMessage(content=responce))
    return responce  # Extract only the generated function

def evaluate_function(code: str) -> str:
    evaluation_prompt = f"""Review the following Python function:\n```python\n{code}\n```\n
    Provide a detailed evaluation and score of evaluatoion of the function based on the following criteria:
    ### Evaluation Criteria:
    Evaluate the function based on the following key characteristics:
    1. **Correctness**: Does the function correctly implement the intended logic? Verify its output against expected results.
    2. **Efficiency**: Is the function optimized for performance? Consider time and space complexity.
    3. **Edge Case Handling**: Does the function handle missing data, extreme values, and potential errors?
    4. **Readability & Maintainability**: Is the code well-structured, using meaningful variable names and comments?
    5. **Scalability**: Can the function handle large datasets efficiently without performance degradation?
    6. **Security & Robustness**: Are there any security vulnerabilities, such as injections, improper input handling, or unexpected behaviors?
    7. **Generalizability**: Can this function be applied to different datasets and use cases with minimal modification?
    8. **Data Handling & Type Safety**: Does the function properly validate data types and conversions?
    9. **Error Handling**: Are there appropriate try-except blocks to catch errors and avoid crashes?
    10. **Output Consistency**: Are the returned values consistent in format, structure, and type? 

    ### Domain-Specific Considerations:
    Assess whether the function properly handles the following data characteristics:
        - **Numerical Data**:  
        - handle numaric columns 
        - some of the functions are can be used for numaric columns 
        - evaluate whcich function is for numaric columns and which is not

        - **Categorical & Text Data**:  
        - Handle categorical columns
        - some of the functions are can be used for categorical columns
        - evaluate whcich function is for categorical columns and which is not


        - **Date & Time Data**:  
        - Handle date and time columns
        - Date, Time , Timezone , Datetime are complex columns to detect and specify diffrence bettween Date,Time,Timezone,Datetime column
        - evaluate input for these columns and detect proper and accurete result
    

        - **Geographical Data**:  
        - these function are complex to use and detect the geographical data
        - detecting city , state , country , postal code , latitude and longitude are complex columns to detect and specify diffrence bettween them
        - handle input and use diffent and efficiaent method to detect these columns

        #Final Score:
        Provide a score from **1 to 10** based on the overall performance above Evaluation Criteria, where:
        - **1-3**: Poor implementation, lacks necessary functionality and error handling.  
        - **4-6**: Functional but has optimization, security, or maintainability issues.  
        - **7-8**: Well-implemented with minor improvements needed.  
        - **9-10**: Highly optimized, efficient, and robust. 
        **important please always provide final score at the end of the evaluation in "score":score_value object format' format** 
"""
    global_chatHistory.append({"evaluation_prompt": evaluation_prompt})
    chatHistory.append(HumanMessage(content=evaluation_prompt))
    result=llm.invoke(chatHistory)
    eveluator_responce=result.content
    global_chatHistory.append({"evaluation_response": eveluator_responce})
    chatHistory.append(AIMessage(content=eveluator_responce))
    return eveluator_responce


def optimize_function(code: str, feedback: str) -> str:
    optimize_prompt = f"""
**Task:** Improve the following function based on the provided feedback.  

#### **Feedback:**  
{feedback}  

#### **Function:**  
```python  
{code}  
```
#### **Instructions:**  
- **Return only the optimized function**—do not generate anything else.  
- Optimize for **performance, readability, maintainability, and scalability**.  
- Ensure **efficient error handling** and **edge case coverage**.  
- The function should accept **a Pandas DataFrame and a column name**—optimize it accordingly.  
- **Enhance compatibility with large datasets** by avoiding memory-intensive operations.  
- Maintain a **consistent return type** and predictable output behavior.  
- Use **best practices for numerical computations, data validation, and type safety**.  
- Prefer **vectorized operations over loops** for better efficiency.  
- If applicable, utilize **optimized Pandas/NumPy functions** for improved performance.  

#### **Key Enhancements in This Version:**  
✅ **Clearer structure**—separates feedback, function, and instructions for better readability.  
✅ **Stronger emphasis on key optimizations**—efficiency, scalability, and best practices.  
✅ **Explicit dataset handling guidance**—ensures seamless performance on large data.  
✅ **More direct and action-driven language**—ensures precision in function improvement.  
                        """
    global_chatHistory.append({"optimize_prompt": optimize_prompt})
    chatHistory.append(HumanMessage(content=optimize_prompt))
    result=llm.invoke(chatHistory)
    optimize__responce=result.content
    global_chatHistory.append({"optimize_response": optimize__responce})
    chatHistory.append(AIMessage(content=optimize__responce))
    return optimize__responce

def extract_score(feedback):
    try:
        feedback_json = json.loads(feedback)  # Parse JSON string into a dictionary
        score = int(feedback_json.get("score", 0))  # Get score, default to 0 if not found
        return score
    except (json.JSONDecodeError, ValueError, TypeError):
        return None  # Handle invalid JSON gracefully





def Workflow(description):
    max_iterations = 5  # Set a max iteration limit
    current_iteration = 0
    threshold_score = 9  # Define a minimum acceptable score

    # Generate initial function
    function_code = generator("find_mean", {description}, "dataframe: df, column_name")
    # print("Function Code Generator",function_code)

    while current_iteration < max_iterations:
        global_chatHistory.append({"current_iteration": current_iteration})
        # Evaluater Function
        feedback = evaluate_function(function_code)
        # print(f'feedback {current_iteration}:{feedback}')

        # Extract score from feedback (Assuming score is provided in feedback)
        match = re.search(r'"score":\s*(\d+)', feedback) 
        score = int(match.group(1)) if match else None  # Extracts numerical score from feedback
        print(f'score:{score}')

        global_chatHistory.append({"score": score})
        
        if score >= threshold_score:
            print(f"✅ Stopping: Function reached acceptable quality with score {score}/10")
            break  # Stop if function is good enough

        # Optimize the function
        function_code = optimize_function(function_code, feedback)
        # print(f'otimized function',function_code)
        current_iteration += 1

    print("\n🎯 Final Optimized Code:\n", function_code)
    return function_code  # Return final function


code=Workflow("write a function to calculate the mean of numaric column")




def time_space_complexity_evaluator(code,df,column_name) -> str:
    """
    these function evaluate the time and space complexity of the function
    on the basis of the following criteria:
    1. **Time Complexity**: Evaluate the function based on the time taken to execute the function.
    2. **Space Complexity**: Evaluate the function based on the memory used by the function.
    3. **Performance**: Evaluate the function based on the performance of the code.
    4. **Scalability**: Evaluate the function based on the scalability of the code.
    5. **Efficiency**: Evaluate the function based on the efficiency of the code.
    6. **Optimization**: Evaluate the function based on the optimization of the code.
    """
    chatHistory =[]
    llm_new = ChatGoogleGenerativeAI(model="gemini-pro")
    time_space_complexity_evaluator_prompt = f"""
**Task:** Analyze the time and space complexity of the given function when executed on a provided DataFrame.

### **Function to Evaluate:**
```python
{code}
### **Execution Details:**
- The function should be executed using the provided DataFrame {df} and column {column_name}.
- Evaluate the function’s performance based on execution time and memory consumption.
- Evaluation Criteria:


1.**Time Complexity:**
- Measure the execution time required for the function to complete.
- Identify whether it scales as O(1), O(log n), O(n), O(n log n), O(n²), etc.
- Check if it uses loops, recursion, or built-in optimized functions.

2.**Space Complexity:**
- Measure the memory used during execution.
- Determine if the function utilizes additional data structures (lists, dictionaries, sets, etc.).
- Identify if memory consumption is O(1), O(n), O(n²), etc.

expected output:
- time complexity: time complexity of the function
- space complexity: space complexity of the function
- time taken: time taken to execute the function(in seconds)
- space used: memory used during execution(in bytes)

**Instructions**
- execute the given function using given arguments only
- evaluate the time and space complexity of the function
- provide the time and space complexity of the function
- do not change the function code
- do not use any other function
- do not use any other library
- do not use any other code
- do not use any other arguments
- do not use any other dataframe
- do not use any other column name
- do not return function in output 
- output should be stricly in given in expected format
- do not provide unnecessary information rather than expected output 
    """
    result = llm_new.invoke(time_space_complexity_evaluator_prompt)
    return result.content

df = pd.read_csv('results.csv')



time_space_evaluation=time_space_complexity_evaluator(code,df,"Hindi")
print("=====================================")
global_chatHistory.append({"time_space_complexity_evaluation": time_space_evaluation})
print(time_space_evaluation)
print("=====================================")

# print(global_chatHistory)

def save_chat_history(chat_history: List[Dict], filename: str = "global_chat_history.json") -> None:
    """
    Save the global chat history to a JSON file as separate objects for each iteration.
    Each iteration will be merged into a single object.

    Args:
        chat_history (List[Dict]): List of dictionaries containing different fields 
                                like generative_prompt, generative_response, etc.
        filename (str): Name of the output JSON file.
    """
    try:
        # Check if the file exists
        if os.path.exists(filename):
            # Read the existing data from the file
            with open(filename, 'r') as f:
                existing_data = json.load(f)
                merged_iterations = existing_data.get("chat_history", [])
        else:
            # If the file doesn't exist, start fresh
            merged_iterations = []
        
        current_iteration_data = {}

        # Merge all key-value pairs into a single object
        for item in chat_history:
            current_iteration_data.update(item)
        
        # Append the merged object as a new iteration
        merged_iterations.append(current_iteration_data)

        # Save the updated data back to the file
        with open(filename, 'w') as f:
            json.dump({"chat_history": merged_iterations}, f, indent=2)

        print(f"✅ Chat history saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving chat history: {str(e)}")


save_chat_history(global_chatHistory)