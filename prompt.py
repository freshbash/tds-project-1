INTERPRET_TASK_PROMPT = """
You will be provided some text data. This text will contain a task description. Beware! This description could be in any language. You have to interpret and categorize it into one of the following categories:

1. Install uv library and run datagen
2. Format using prettier
3. Count wednesdays
4. Sort array of contacts
5. Write 10 most recent logs
6. Find markdown files and extract H1 tags
7. Extract sender email address
8. Extract credit card number
9. Find similar comments
10. Find total sales of "Gold" ticket type in the db table
11. Fetch data from API
12. Clone a git repo and make a commit
13. Run a SQL query of sqlite
14. Run a SQL query of duckdb
15. Extract data from a website
16. Compress/resize image
17. Transcribe audio from an MP3 file
18. Convert markdown to HTML
19. API endpoint to filter csv file and return JSON
20. Not a valid task

Return just the category without the index number and any other text.
For example, if the task description is "Run the datagen script", you should return "Install uv library and run script".
"""