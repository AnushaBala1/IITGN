import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import json
import os
import requests
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import psycopg2

# Load environment variables
load_dotenv()

# Specify Tesseract executable path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set API Keys (ensure these are set in your .env file)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants for NVIDIA API and DB
NVAI_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('API_KEY')}",
    "Accept": "application/json"
}
TOOLS = ["markdown_bbox", "markdown_no_bbox", "detection_only"]
DB_NAME = "IITGN"
DB_USER = "postgres"
DB_PASSWORD = "Suresh@11"
DB_HOST = "localhost"
DB_PORT = "5432"

global sql_response 
# PDF to Image Conversion

# NVIDIA API Functions
def upload_asset(input_data, description):
    authorize = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers={"Content-Type": "application/json", **HEADERS},
        json={"contentType": "image/jpeg", "description": description},
        timeout=30,
    )
    authorize.raise_for_status()
    
    response = requests.put(
        authorize.json()["uploadUrl"],
        data=input_data,
        headers={
            "x-amz-meta-nvcf-asset-description": description,
            "content-type": "image/jpeg",
        },
        timeout=300,
    )
    response.raise_for_status()
    
    return str(authorize.json()["assetId"])

def generate_content(task_id, asset_id):
    if task_id < 0 or task_id >= len(TOOLS):
        raise ValueError(f"task_id should be within [0, {len(TOOLS)-1}]")
    tool = [{"type": "function", "function": {"name": TOOLS[task_id]}}]
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;asset_id,{asset_id}"}}]
    return content, tool

def process_image(image_data, task_id=0):
    asset_id = upload_asset(image_data, "Uploaded Image")
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
        **HEADERS
    }
    
    response = requests.post(NVAI_URL, headers=post_headers, json=inputs)
    
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

# AI Response Generation
def initialize_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        api_key=GOOGLE_API_KEY,
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

def get_ai_response(llm, user_message: str) -> str:
    messages = [
        (
            "system",
            """You are tasked with processing table data extracted from an image. Your job is to:
            1. Parse the table data into a structured JSON object (e.g., a list of dictionaries where each dictionary represents a row with column names as keys).
            2. Return the result as a single-line JSON string with the structure: {"table_data": "<parsed table data as JSON>"}.
            3. Ensure the output is pure JSON, with no markdown (e.g., ```json), no triple quotes, no extra text, and no escape sequences like \n or \t.
            4. If the table contains Hindi text, autocorrect misspelled Hindi words using standard Hindi spelling conventions.
            5. If the table contains numeric values with units (e.g., '10 kg', '5.6 m', 'Rs.10'), extract the numeric value only.
            Example output: {"table_data": [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}]}"""
        ),
        ("human", f"Table data:\n{user_message}")
    ]
    try:
        ai_msg = llm.invoke(messages)
        response = ai_msg.content.strip()
        # Remove common unwanted prefixes and suffixes
        unwanted_prefixes = ["```json", "'''json", "```", "'''"]
        unwanted_suffixes = ["```", "'''"]
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]
        for suffix in unwanted_suffixes:
            if response.endswith(suffix):
                response = response[:-len(suffix)]
        # Remove any remaining whitespace or escape sequences
        response = response.replace("\n", "").replace("\t", "").strip()
        # Validate it's non-empty and starts with { (basic JSON check)
        if not response or not response.startswith("{"):
            return '{"table_data": []}'  # Default empty table if invalid
        return response
    except Exception as e:
        return f"LLM error: {str(e)}"

def generate_sql_queries(llm, table_data_json: str) -> str:
    messages = [
        (
            "system",
            """You are tasked with generating PostgreSQL queries from structured JSON table data. Your job is to:
            1. Generate a PostgreSQL schema as a CREATE TABLE statement, inferring appropriate data types (e.g., VARCHAR, INTEGER) for each column based on the JSON data, in a single line ending with a semicolon.
            2. Create a PostgreSQL INSERT INTO statement to add the table data into the schema, in a single line with values properly formatted (e.g., strings in single quotes, numbers unquoted), ending with a semicolon.
            3. Return the result as a single-line JSON string with the structure: {"schema": "<CREATE TABLE statement>", "insert_query": "<INSERT INTO statement>"}.
            4. Ensure the output is pure JSON, with no markdown (e.g., ```json), no triple quotes, no extra text, and no escape sequences like \n or \t.
            Example output: {"schema": "CREATE TABLE users(name VARCHAR, age INTEGER);", "insert_query": "INSERT INTO users(name, age) VALUES('John', 25), ('Jane', 30);"}"""
        ),
        ("human", f"Table data in JSON:\n{table_data_json}")
    ]
    try:
        ai_msg = llm.invoke(messages)
        response = ai_msg.content.strip()
        # Remove common unwanted prefixes and suffixes
        unwanted_prefixes = ["```json", "'''json", "```", "'''"]
        unwanted_suffixes = ["```", "'''"]
        for prefix in unwanted_prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):]
        for suffix in unwanted_suffixes:
            if response.endswith(suffix):
                response = response[:-len(suffix)]
        # Remove any remaining whitespace or escape sequences
        response = response.replace("\n", "").replace("\t", "").strip()
        # Validate it's non-empty and starts with { (basic JSON check)
        if not response or not response.startswith("{"):
            return '{"schema": "CREATE TABLE empty_table(id INTEGER);", "insert_query": "INSERT INTO empty_table(id) VALUES(0);"}'
        return response
    except Exception as e:
        return f"LLM error: {str(e)}"

# Database Loading Functions
def load_to_db(schema, insert_query):
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute(schema)
        conn.commit()
        cursor.execute(insert_query)
        conn.commit()
        return {"message": "Data loaded to database successfully"}
    except psycopg2.Error as e:
        conn.rollback()
        return {"error": f"Database error: {str(e)}"}
    finally:
        cursor.close()
        conn.close()

def load_to_db_from_sql_response(sql_response):
    """Load data into the database directly from the sql_response string."""
    try:
        sql_dict = json.loads(sql_response)
        if not all(key in sql_dict for key in ["schema", "insert_query"]):
            return {"error": "Invalid sql_response: missing 'schema' or 'insert_query'"}, 400
        schema = sql_dict["schema"]
        insert_query = sql_dict["insert_query"]
        result = load_to_db(schema, insert_query)
        return result, 200 if "message" in result else 500
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in sql_response"}, 400
    except Exception as e:
        return {"error": f"Error loading data to database: {str(e)}"}, 500
    
# Main Processing Function with Automatic Loading
def process_image_query(image_file, temp_path="temp.pdf", auto_load=True):
    try:
        page_image = Image.open(image_file).convert('RGB')
        img_byte_arr = io.BytesIO()
        page_image.save(img_byte_arr, format='JPEG')
        image_data = img_byte_arr.getvalue()

        extracted_text = process_image(image_data)
        if not extracted_text:
            return {"error": "No text extracted from page image"}, 500

        llm = initialize_model()
        table_data_json = get_ai_response(llm, extracted_text)
        sql_response = generate_sql_queries(llm, table_data_json)
        print(sql_response,table_data_json)
        sql_dict = json.loads(sql_response)

        # Automatically load to database if auto_load is True
        #load_result = None
        #if auto_load:
        #    load_result, load_status = load_to_db_from_sql_response(sql_response)
        #    if load_status != 200:
        #        return load_result, load_status

        result = {
            "extracted_text": extracted_text,
            "table_data_json": table_data_json,
            "sql_response": sql_dict,
            "message": "PDF query processed successfully"
        }
        #if auto_load:
        #    result["load_result"] = load_result

        return result, 200

    except Exception as e:
        return {"error": f"Error processing PDF query: {str(e)}"}, 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

