import csv
import os
import requests
import json
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import sys

# Load environment variables once at the top
load_dotenv()

# Constants from table_extraction
nvai_url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('API_KEY')}",
    "Accept": "application/json"
}
tools = [
    "markdown_bbox",
    "markdown_no_bbox",
    "detection_only",
]

# Functions from table_extraction.py
def upload_asset(input, description):
    authorize = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers={
            "Content-Type": "application/json",
            **headers,
        },
        json={"contentType": "image/jpeg", "description": description},
        timeout=30,
    )
    authorize.raise_for_status()
    
    response = requests.put(
        authorize.json()["uploadUrl"],
        data=input,
        headers={
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg",
        },
        timeout=300,
    )
    response.raise_for_status()
    
    return str(authorize.json()["assetId"])

def generate_content(task_id, asset_id):
    if task_id < 0 or task_id >= len(tools):
        raise ValueError(f"task_id should be within [0, {len(tools)-1}]")
    tool = [{
        "type": "function",
        "function": {"name": tools[task_id]},
    }]
    content = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;asset_id,{asset_id}"}
    }]
    return content, tool

def process_image(image_path, task_id):
    with open(image_path, "rb") as image_file:
        asset_id = upload_asset(image_file, "Test Image")
    
    content, tool = generate_content(task_id, asset_id)
    
    inputs = {
        "tools": tool,
        "model": "nvidia/nemoretriever-parse",
        "messages": [{"role": "user", "content": content}]
    }
    
    post_headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_id,
        "NVCF-FUNCTION-ASSET-IDS": asset_id,
        **headers
    }
    
    response = requests.post(nvai_url, headers=post_headers, json=inputs)
    
    try:
        response_json = response.json()
        text_output = []
        for entry in response_json["choices"][0]["message"]["tool_calls"]:
            arguments = json.loads(entry["function"]["arguments"])
            for item in arguments:
                if isinstance(item, list):
                    for sub_item in item:
                        text_output.append(sub_item.get("text", ""))
                else:
                    text_output.append(item.get("text", ""))
        return "\n".join(text_output)
    except ValueError:
        return "Response is not in JSON format"

# Functions from table_refind.py
def initialize_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=api_key,
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

def get_ai_response(llm, user_message: str, nlp_query: str) -> str:
    messages = [
        (
            "system",
            """You are tasked with processing table data extracted from an image. Your job is to:
            1. Generate a PostgreSQL schema as a CREATE TABLE statement, inferring appropriate data types (e.g., VARCHAR, INTEGER) for each column. Do not include any escape sequences like /n or /t. The statement must be directly executable in PostgreSQL and end with a semicolon. Use Caption from the Retrived content as the Table name or undersatnding.
            2. Create a PostgreSQL INSERT INTO statement to add the extracted table data into the table defined by the schema. Do not include any escape sequences like /n or /t. The statement must be directly executable in PostgreSQL, with values properly formatted (e.g., strings in single quotes), and end with a semicolon. Make sure to convert all extraction into json, donot leave any feilds or rows for the extracted text.
            3. Answer the provided NLP query based on the table data.
            If the table contains Hindi text, autocorrect any misspelled Hindi words.
            Return your response as a JSON string with the following structure:
            {
                "schema": "<PostgreSQL CREATE TABLE statement>",
                "insert_query": "<PostgreSQL INSERT INTO statement>",
                "answer": "<answer to the NLP query>"
            }
            Ensure the output is a pure, valid JSON string. Do not wrap it in ```json or any other code block markers, and avoid any extra formatting or unnecessary quotes beyond what is required by JSON syntax."""
        ),
        ("human", f"Table data:\n{user_message}\n\nNLP Query: {nlp_query}")
    ]
    try:
        ai_msg = llm.invoke(messages)
        # Post-process to remove ```json markers and any extra whitespace
        response = ai_msg.content.strip()
        if response.startswith("```json"):
            response = response.replace("```json", "", 1).strip()
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0].strip()
        return response if response else "Failed to generate a response."
    except Exception as e:
        return f"An error occurred: {e}"

# Functions from main.py
def extract_table_from_response(response: str):
    """Extracts tabular data by finding the first and last numeric row."""
    lines = response.strip().split("\n")
    table_lines = []

    # Find the first and last valid table rows
    for line in lines:
        if re.search(r'\d', line):  # Check if line contains a number (assumes tables have numbers)
            table_lines.append(line.strip())

    return table_lines

def save_to_json(response: str, filename: str = "ai_response.json"):
    """Saves the AI response (schema, insert_query, answer) as a JSON file."""
    try:
        # Parse the response as JSON
        result = json.loads(response)
        if not isinstance(result, dict) or not all(key in result for key in ["schema", "insert_query", "answer"]):
            raise ValueError("Response must be a dictionary with 'schema', 'insert_query', and 'answer' keys")

        # Write the full result to JSON
        with open(filename, mode='w', encoding='utf-8') as file:
            json.dump(result, file, indent=4, ensure_ascii=False)

        print(f"✅ Response successfully saved to {filename} in JSON format.")
    
    except json.JSONDecodeError:
        print(f"❌ Error: Response is not valid JSON: {response}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        sys.exit(1)

def main():
    """Main function to process the image, extract text, get AI response, and save to JSON."""
    # Hardcoded values
    image_path = r"C:\Users\kuhan\OneDrive\Documents\6sem\DOCX\data\1.jpg"
    task_id = 0
    nlp_query = "What is the percentage in total for refreshments?"  # Example NLP query

    try:
        extracted_text = process_image(image_path, task_id)
    except Exception as e:
        print(f"❌ Error extracting text from image: {e}")
        sys.exit(1)

    llm = initialize_model()

    try:
        refined_response = get_ai_response(llm, extracted_text, nlp_query)
        print("AI Response:\n", refined_response)
        save_to_json(refined_response)
    except Exception as e:
        print(f"❌ Error processing AI response: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()