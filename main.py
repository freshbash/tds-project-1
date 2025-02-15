from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import subprocess
import shutil
import urllib.request
import json
from datetime import datetime
import httpx
import base64
import sqlite3
from prompt import INTERPRET_TASK_PROMPT
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class TaskRequest(BaseModel):
    task: str

LLM_API_URL_COMPLETIONS = os.getenv("LLM_API_URL_COMPLETIONS")
LLM_API_URL_EMBEDDINGS = os.getenv("LLM_API_URL_EMBEDDINGS")
API_KEY = os.getenv("API_KEY")

def execute_query(query, params=()):
    """Executes a SQL query and returns the result."""
    db_path = "/data/ticket-sales.db"
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="Database not found")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query, params)
    result = cursor.fetchall()
    conn.close()
    return result

def api_call_to_llm(system: str, content: str, task="completions") -> str:
    """API calls to GPT-4o-mini"""

    endpoint = LLM_API_URL_COMPLETIONS if task == "completions" else LLM_API_URL_EMBEDDINGS

    # Make the API request
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    if task == "completions":
        endpoint = LLM_API_URL_COMPLETIONS
        payload = {
            "model": "gpt-4o-mini",
            "messages":[
                {"role": "system", "content": system},
                {"role": "user", "content": content}
            ],
            "temperature": 0.0
        }
    else:
        endpoint = LLM_API_URL_EMBEDDINGS
        payload = {
            "input": content,
            "model": "gpt-4o-mini"  # Use a suitable embedding model
        }

    response = httpx.post(endpoint, json=payload, headers=headers).json()

    if task == "completions":
        return response["choices"][0]["message"]["content"].strip()
    else:
        return response['data'][0]['embedding']

def interpret_task(task: str) -> str:
    """Interprets a task description and categorizes it."""
    return api_call_to_llm(INTERPRET_TASK_PROMPT, task, task="completions")

def execute_task(task: str) -> str:
    """Executes a given task and returns the result."""

    std_task = interpret_task(task)

    try:
        if std_task == "Install uv library and run script":
            user_email = task.split()[-1]  # Extract email argument
            if not shutil.which("uv"):
                subprocess.run(["pip", "install", "uv"], check=True)
            script_url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
            script_path = "datagen.py"
            urllib.request.urlretrieve(script_url, script_path)
            subprocess.run(["python", script_path, user_email], check=True)
            return "datagen.py executed successfully"
        
        elif std_task == "Format using prettier":
            if not shutil.which("prettier"):
                subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True)
            subprocess.run(["prettier", "--write", "/data/format.md"], check=True)
            return "File formatted successfully"
        
        elif std_task == "Count wednesdays":
            date_file = "/data/dates.txt"
            output_file = "/data/dates-wednesdays.txt"
            if not os.path.exists(date_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(date_file, "r", encoding="utf-8") as file:
                dates = file.readlines()
            
            wednesday_count = sum(1 for date in dates if datetime.strptime(date.strip(), "%Y-%m-%d").weekday() == 2)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(str(wednesday_count))
            
            return f"Counted {wednesday_count} Wednesdays and wrote to {output_file}"
        
        elif std_task == "Sort array of contacts":
            contacts_file = "/data/contacts.json"
            sorted_file = "/data/contacts-sorted.json"
            if not os.path.exists(contacts_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(contacts_file, "r", encoding="utf-8") as file:
                contacts = json.load(file)
            
            sorted_contacts = sorted(contacts, key=lambda c: (c["last_name"], c["first_name"]))
            
            with open(sorted_file, "w", encoding="utf-8") as file:
                json.dump(sorted_contacts, file, indent=4)
            
            return f"Sorted contacts and wrote to {sorted_file}"
        
        elif std_task == "Write 10 most recent logs":
            logs_dir = "/data/logs"
            output_file = "/data/logs-recent.txt"
            
            if not os.path.exists(logs_dir):
                raise HTTPException(status_code=404, detail="Logs directory not found")
            
            log_files = sorted(
                (os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")),
                key=os.path.getmtime,
                reverse=True
            )[:10]
            
            first_lines = []
            for log_file in log_files:
                with open(log_file, "r", encoding="utf-8") as file:
                    first_line = file.readline().strip()
                    first_lines.append(first_line)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write("\n".join(first_lines))
            
            return f"Extracted first lines from recent logs and wrote to {output_file}"

        elif std_task == "Find markdown files and extract H1 tags":
            docs_dir = "/data/docs"
            index_file = "/data/docs/index.json"
            
            if not os.path.exists(docs_dir):
                raise HTTPException(status_code=404, detail="Docs directory not found")
            
            index = {}
            for root, _, files in os.walk(docs_dir):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("# "):
                                    title = line.strip("# ").strip()
                                    index[os.path.relpath(file_path, docs_dir)] = title
                                    break
            
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=4)
            
            return f"Created Markdown index at {index_file}"
        
        elif std_task == "Extract sender email address from email content":
            email_file = "/data/email.txt"
            output_file = "/data/email-sender.txt"
            
            if not os.path.exists(email_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(email_file, "r", encoding="utf-8") as file:
                email_content = file.read()
            
            system_message = "Extract the sender's email address from the given email content. Return just the email address."

            sender_email = api_call_to_llm(system_message, email_content)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(sender_email)
            
            return f"Extracted sender's email and wrote to {output_file}"

        elif std_task == "Extract credit card number from the given image":
            image_path = "/data/credit-card.png"
            output_file = "/data/credit-card.txt"
            
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail="Image not found")
            
            system_message = "Extract the credit card number from the given image. Return just the credit card number without spaces."

            with open(image_path, "rb") as file:
                binary_data = file.read()
                image_b64 = base64.b64encode(binary_data).decode()
            
            image_uri = f"data:image/png;base64,{image_b64}"

            credit_card_number = api_call_to_llm(system_message, image_uri)
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(credit_card_number)
            
            return f"Extracted credit card number and wrote to {output_file}"

        elif std_task == "Find similar comments":
            comments_file = "/data/comments.txt"
            output_file = "/data/comments-similar.txt"
            
            if not os.path.exists(comments_file):
                raise HTTPException(status_code=404, detail="File not found")
            
            with open(comments_file, "r", encoding="utf-8") as file:
                comments = [line.strip() for line in file.readlines() if line.strip()]
            
            comment_embeddings = dict()

            for comment in comments:
                embedding = api_call_to_llm("Find similar comments", comment, task="embeddings")
                comment_embeddings[comment] = embedding

            # Extract comments and embeddings
            comments = list(comment_embeddings.keys())
            embeddings = np.array(list(comment_embeddings.values()))

            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(embeddings)

            # Find pairs of most similar comments
            similar_pairs = []
            for i in range(len(comments)):
                for j in range(i + 1, len(comments)):
                    similarity = similarity_matrix[i, j]
                    similar_pairs.append((comments[i], comments[j], similarity))

            # Sort pairs by similarity score (descending order)
            similar_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)

            most_similar_pair = list(similar_pairs[0][:2])
            
            with open(output_file, "w", encoding="utf-8") as file:
                file.write("\n".join(most_similar_pair))
            
            return f"Found most similar comments and wrote to {output_file}"

        elif std_task == "Find total sales of 'Gold' ticket type in the db table":
            query = "SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'"
            result = execute_query(query)
            total_sales = result[0][0] if result[0][0] is not None else 0
            with open("/data/ticket-sales-gold.txt", "w") as f:
                f.write(str(total_sales))
            return {"message": "Total sales for Gold tickets computed successfully"}

        else:
            raise ValueError("Unknown task")
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run")
def run_task(task: str = Query(..., description="Task description in plain English")):
    """Executes a plain-English task and returns the output."""
    result = execute_task(task)
    return {"status": "success", "output": result}

@app.get("/read")
def read_file(path: str = Query(..., description="Path to the file")):
    """Returns the content of the specified file."""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    
    return {"status": "success", "content": content}
