# import asyncio
# import csv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langgraph.graph import StateGraph  

# load_dotenv()

# # Initialize Gemini AI model
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# async def generate_python_function(state):
#     """Calls LLM to generate a Python function (without explanations)."""
#     query = state["query"]
#     prompt = f"Write a Python function that {query}. Only return the function code, no explanations."

#     response = await llm.ainvoke(prompt)  
#     state["generated_function"] = response.content
#     return state

# # Create a graph workflow
# workflow = StateGraph(dict)  
# workflow.add_node("generate_python_function", generate_python_function)
# workflow.set_entry_point("generate_python_function")
# app = workflow.compile()  

# # Define metadata and corresponding queries
# metadata_queries = {
#     "datatype": "make function that takes dataframe and column and return datatype of each column",
#     "float_precision": "create a function that takes a dataframe and column, checks if it is numerical, and calculates its float precision if numeric",
#     "is_integer": "write a function that takes a dataframe and column and returns whether the column is integer or not",
#     "is_float": "create a function that takes a dataframe and column and returns whether the column is float or not",
#     "is_text": "make a function that takes a dataframe and column and returns whether it is text or not",
#     "eq_median_count": "write a function that takes a dataframe and column and returns the eq_median_count (number of values equal to median)",
#     "gt_median_count":"write a function that takes a dataframe and column and returns the eq_median_count (number of values greater than median)",
#     # "lt_median_count": "write a function that takes a dataframe and column and returns the eq_median_count (number of values less than median)",
#     "mode_value": "write a function that takes a dataframe and column and returns the mode_value (list of modes)",
#     "mean_value": "write a function that takes a dataframe and column and returns the mean_value (mean of column)",
#     "standard_deviation": "write a function that takes a dataframe and column and returns the standard_deviation (std dev of column)",
#     "min_value": "write a function that takes a dataframe and column and returns the min_value (minimum value)",
#     "max_value": "write a function that takes a dataframe and column and returns the max_value (maximum value)"
# }

# async def process_queries():
#     """Executes metadata queries asynchronously."""
#     tasks = {meta: app.ainvoke({"query": query}) for meta, query in metadata_queries.items()}  
#     results = await asyncio.gather(*tasks.values())  
#     return {meta: results[i]["generated_function"] for i, meta in enumerate(tasks.keys())}

# # Run the workflow and save results
# functions = asyncio.run(process_queries())

# # Save to CSV with metadata instead of queries
# csv_filename = "generated_functions.csv"
# with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.writer(file, delimiter=",")
#     writer.writerow(["Metadata", "Function"])
    
#     for meta, func_code in functions.items():
#         # print(f"{meta}: {func_code}")
#         writer.writerow([meta, func_code.replace("\n", "\\n")])  

# print(f"✅ Functions saved successfully in {csv_filename}")
#     # "lt_median_count": "write a function that takes a dataframe and column and returns the eq_median_count (number of values less than median)",
#     # "eq_mean_count": "write a function that takes a dataframe and column and returns the eq_mean_count (number of values equal to mean)",
#     # "gt_mean_count": "write a function that takes a dataframe and column and returns the gt_mean_count (number of values greater than mean)",
#     # "lt_mean_count": "write a function that takes a dataframe and column and returns the lt_mean_count (number of values less than mean)",
#     # "eq_zero_count": "write a function that takes a dataframe and column and returns the eq_zero_count (number of values equal to zero)",
#     # "lt_zero_count": "write a function that takes a dataframe and column and returns the lt_zero_count (number of values less than zero)",