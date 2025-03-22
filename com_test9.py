import psycopg2
import json
import os
import requests
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
DB_NAME = "IITGN"
DB_USER = "postgres"
DB_PASSWORD = "Suresh@11"
DB_HOST = "localhost"
DB_PORT = "5432"

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
    asset_id = upload_asset(image_path, "Test Image")
    
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
            1. Generate a PostgreSQL schema as a CREATE TABLE statement, inferring appropriate data types (e.g., VARCHAR, INTEGER) for each column. The statement must be a single line with no escape sequences like /n or /t, directly executable in PostgreSQL, and end with a semicolon. Use the caption from the retrieved content as the table name if available; otherwise, infer a suitable name (e.g., 'extracted_table') and ensure it is unique and valid.
            2. Create a PostgreSQL INSERT INTO statement to add the extracted table data into the table defined by the schema. The statement must be a single line with no escape sequences like /n or /t, directly executable in PostgreSQL, with values properly formatted (e.g., strings in single quotes, numbers unquoted), and end with a semicolon. Include all extracted data without omitting any fields or rows. If no data is present, return a minimal valid INSERT statement (e.g., 'INSERT INTO table_name (column) VALUES (NULL);').
            3. Answer the provided NLP query based on the table data. If the query cannot be answered due to missing or invalid data, return a clear explanation (e.g., 'Cannot compute percentage due to missing numeric data').
            If the table contains Hindi text, autocorrect any misspelled Hindi words using standard Hindi spelling conventions.
            If the table contains numeric values with units (e.g., '10 kg', '5.6 m', 'Rs.10'), extract the numeric value only.
            Return your response as a single-line JSON string with the following structure:
            {"schema": "<PostgreSQL CREATE TABLE statement>", "insert_query": "<PostgreSQL INSERT INTO statement>", "answer": "<answer to the NLP query>"}
            The output must be a pure, valid, single-line JSON string with no leading 'json' prefix, no code block markers like json or , and no escape sequences like /n or /t anywhere in the string. If the table data is empty, malformed, or cannot be processed, return a valid JSON response (e.g., {"schema": "CREATE TABLE extracted_table (id INTEGER);", "insert_query": "INSERT INTO extracted_table (id) VALUES (NULL);", "answer": "No valid table data found"})."""
        ),
        ("human", f"Table data:\n{user_message}\n\nNLP Query: {nlp_query}")
    ]
    try:
        ai_msg = llm.invoke(messages)
        # Post-process to ensure clean, single-line JSON
        response = ai_msg.content.strip()
        # Remove any unwanted prefixes or markers
        if response.startswith("json"):
            response = response.replace("json", "", 1).strip()
        if response.startswith("json"):
            response = response.replace("json", "", 1).strip()
        if response.endswith("```"):
            response = response[:-3].strip()
        # Remove any newlines or tabs and ensure single-line
        response = response.replace("\n", "").replace("\t", "")
        # Verify no escape sequences remain
        if "\\n" in response or "\\t" in response:
            raise ValueError("Response contains invalid escape sequences")
        return response if response else "Failed to generate a response."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def main(image_path):
    """Main function to process the image, extract text, get AI response, and interact with PostgreSQL."""
    # Hardcoded values
    image_path = "Mospi.jpg"
    task_id = 0
    nlp_query = "What is the percentage in total for refreshments?"  # Example NLP query

    # Establish connection to PostgreSQL
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        sys.exit(1)

    try:
        # Process the image
        extracted_text = process_image(image_path, task_id)
        if not extracted_text:
            raise ValueError("No text extracted from image")

        # Initialize LLM and get response
        llm = initialize_model()
        refined_response = get_ai_response(llm, extracted_text, nlp_query)
        print("AI Response:\n", refined_response)

        # Parse the JSON response
        try:
            response_dict = json.loads(refined_response)
            if not all(key in response_dict for key in ["schema", "insert_query", "answer"]):
                raise ValueError("Response missing required keys")
        except json.JSONDecodeError as e:
            print(f"❌ Error: Response is not valid JSON: {refined_response}")
            sys.exit(1)

        # Execute schema creation
        try:
            cursor.execute(response_dict["schema"])
            conn.commit()
            print("✅ Table Created Successfully.")
        except psycopg2.Error as e:
            print(f"❌ Error creating table: {e}")
            conn.rollback()
            sys.exit(1)

        # Execute insert query
        try:
            cursor.execute(response_dict["insert_query"])
            conn.commit()
            print("✅ Data Inserted Successfully.")
        except psycopg2.Error as e:
            print(f"❌ Error inserting data: {e}")
            conn.rollback()
            sys.exit(1)

        # Print NLP answer
        print(f"NLP Answer: {response_dict['answer']}")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)

    finally:
        # Close database connection
        cursor.close()
        conn.close()
        print("✅ Database connection closed.")

if __name__ == "__main__":
    main()