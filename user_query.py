import os
import requests
import json
import re
from dotenv import load_dotenv
import openai
import sys
import faiss
import numpy as np

# Load environment variables
load_dotenv()

# Constants for NVIDIA API
nvai_url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.getenv('API_KEY')}",
    "Accept": "application/json"
}
tools = ["markdown_bbox", "markdown_no_bbox", "detection_only"]

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Image Processing Functions (Unchanged) ---


def upload_asset(input, description):
    authorize = requests.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/assets",
        headers={"Content-Type": "application/json", **headers},
        json={"contentType": "image/jpeg", "description": description},
        timeout=30,
    )
    authorize.raise_for_status()

    response = requests.put(
        authorize.json()["uploadUrl"],
        data=input,
        headers={"x-amz-meta-nvcf-asset-description": description,
                 "content-type": "image/jpeg"},
        timeout=300,
    )
    response.raise_for_status()

    return str(authorize.json()["assetId"])


def generate_content(task_id, asset_id):
    if task_id < 0 or task_id >= len(tools):
        raise ValueError(f"task_id should be within [0, {len(tools)-1}]")
    tool = [{"type": "function", "function": {"name": tools[task_id]}}]
    content = [{"type": "image_url", "image_url": {
        "url": f"data:image/jpeg;asset_id,{asset_id}"}}]
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

# --- LLM Initialization ---


def initialize_model():
    # No need for a separate initialization since we'll use openai directly
    return "gpt-4o"  # Just return the model name; OpenAI API handles the rest

# --- Table JSON Generation ---


def generate_table_json(model, extracted_text: str) -> str:
    messages = [
        {"role": "system", "content": """You are tasked with processing table data extracted from an image. The input is markdown or plain text representing a table. Your task is to:
            1. Identify the column headers from the first row of the table. These are the keys for the JSON output.
            2. Parse each subsequent row as data, mapping values to the headers from the first row.
            3. Ignore any row numbers or list markers (e.g., '1.', '2.', etc.) that may appear before the actual data in each row; these are not part of the table content.
            4. Handle irregularities such as extra spaces, separators (e.g., '|', '-', '+'), or misaligned text by focusing only on the meaningful data under each header.
            5. If the table contains Hindi text, autocorrect misspelled Hindi words.
            6. Convert the table into a JSON list of dictionaries where:
               - Keys are the exact column headers from the first row (as strings, preserving original text including spaces and special characters).
               - Values are the cell values (as strings), aligned with the corresponding headers.
            7. Ensure all rows have values matching the number of headers; do not include null unless explicitly present in the data as an empty field.
            8. Return a valid JSON string without extra text or formatting.

            Output format:
            [{"header1": "value1", "header2": "value2"}, {"header1": "value3", "header2": "value4"}]

            Do not include json markers, explanations, or any text outside the JSON string."""},
        {"role": "user", "content": f"Extracted table data:\n{extracted_text}"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.4,
            max_tokens=4096  # Adjust based on GPT-4o's limits
        )
        result = response.choices[0].message["content"].strip()
        json.loads(result)  # Validate JSON
        return result
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response: {result}, Error: {e}")
    except Exception as e:
        raise Exception(f"Error generating table JSON: {e}")

# --- Helper Functions ---


def get_embedding(text: str):
    response = openai.Embedding.create(
        input=text, model="text-embedding-3-small")
    return response["data"][0]["embedding"]

# --- Vector Database Upload Function ---


def upload_table_to_vector_db(table_data: list) -> tuple:
    """Upload the table data into a FAISS vector database after embedding."""
    try:
        row_strings = [
            ", ".join([f"{k}: {v}" for k, v in row.items()]) for row in table_data]
        embeddings = [get_embedding(row_str) for row_str in row_strings]
        embeddings = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(len(embeddings[0]))
        index.add(embeddings)
        return index, table_data
    except Exception as e:
        raise Exception(f"Error uploading table data to vector database: {e}")

# --- Query Processing Function ---


def process_user_query(model, query: str, index: faiss.IndexFlatL2, row_data: list) -> str:
    messages = [
        {"role": "system", "content": """You are provided with a user query and table data as a list of dictionaries. Your task is to:
        1. Answer the query based on the provided data.
        2. Generate a Python script using matplotlib or seaborn to plot the data:
           - If the query specifies a graph type (e.g., 'bar chart', 'pie chart'), use it.
           - Otherwise, choose an appropriate plot type (e.g., bar for comparisons, pie for proportions).
           - Include data for the queried item(s) and a subset of other items for context.
           - Use pandas to create a DataFrame.
           - Convert string values to appropriate types (e.g., float for numbers) before plotting.
           - Add labels, title, and formatting for clarity.

        Output format (raw JSON, no formatting markers like ```json):
        {"answer": "<answer to the query>", "plot_script": "<complete Python script>"}

        Return only valid JSON with no explanations, markdown, or code fences."""},
        {"role": "user",
            "content": f"Query: {query}\nData: {json.dumps(row_data, ensure_ascii=False)}"}
    ]

    try:
        query_embedding = get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        k = min(5, len(row_data))  # Retrieve top 5 similar rows
        D, I = index.search(query_embedding, k)
        retrieved_rows = [row_data[i] for i in I[0]]

        formatted_prompt = [
            messages[0],
            {"role": "user",
                "content": f"Query: {query}\nData: {json.dumps(retrieved_rows, ensure_ascii=False)}"}
        ]

        response = openai.ChatCompletion.create(
            model=model,
            messages=formatted_prompt,
            temperature=0.4,
            max_tokens=4096  # Adjust based on GPT-4o's limits
        )
        result = response.choices[0].message["content"].strip()
        json.loads(result)  # Validate JSON
        return result
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON response: {result}, Error: {e}")
    except Exception as e:
        raise Exception(f"Error processing query '{query}': {e}")

# --- Main Function ---


def main():
    image_path = r"E:\Programming\IITGN-Backend\1.jpg"
    task_id = 0
    queries = [
        "Show me a pie chart of the percentage of total for refreshments?"
    ]

    model = initialize_model()

    # Extract text from image
    try:
        extracted_text = process_image(image_path, task_id)
        print(f"Extracted Text:\n{extracted_text}")
    except Exception as e:
        print(f"❌ Error extracting text from image: {e}")
        sys.exit(1)

    # Generate table JSON
    try:
        table_json_response = generate_table_json(model, extracted_text)
        table_data = json.loads(table_json_response)
        print(f"Generated Table JSON:\n{table_json_response}")
    except Exception as e:
        print(f"❌ Error generating table JSON: {e}")
        sys.exit(1)

    # Upload to vector database
    try:
        index, row_data = upload_table_to_vector_db(table_data)
        print("Data successfully uploaded to vector database.")
    except Exception as e:
        print(f"❌ Error uploading table data to vector database: {e}")
        sys.exit(1)

    # Process user queries
    for query in queries:
        try:
            response = process_user_query(model, query, index, row_data)
            result = json.loads(response)
            print(f"\nQuery: {query}")
            print(f"JSON Response: {response}")
            print(f"Answer: {result['answer']}")
            print(f"Plot Script:\n{result['plot_script']}")
        except Exception as e:
            print(f"❌ Error processing query '{query}': {e}")


if __name__ == "__main__":
    main()
