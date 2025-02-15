INTERPRET_TASK_PROMPT = """
You will be provided some text data. This text will contain a task description. You have to interpret and categorize it into one of the following categories:

1. Install uv library and run script
2. Format using prettier
3. Count wednesdays
4. Sort array of contacts
5. Write 10 most recent logs
6. Find markdown files and extract H1 tags
7. Extract sender email address
8. Extract credit card number
9. Find similar comments
10. Find total sales of "Gold" ticket type in the db table

. Not a valid task
Return just the category without the index number and any other text.
For example, if the task description is "Run this script that can be found in this url : <url>", you should return "Install uv library and run script".
"""