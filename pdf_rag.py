import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import re

# Specify Tesseract executable path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust if installed elsewhere

# Set API Keys
import os
os.environ["GOOGLE_API_KEY"] = ""  # Replace with your Google API key
#os.environ["OPENAI_API_KEY"] = ""  # Replace with your OpenAI API key

def pdf_to_images(pdf_path):
    """Convert PDF pages to PIL images."""
    doc = fitz.open(pdf_path)
    images = {}
    for i, page in enumerate(doc, 1):
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI for better OCR
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images[i] = img
    doc.close()
    return images

def is_table_like(text):
    """Heuristic to detect table-like content."""
    # Look for patterns like numbers, percentages, or structured delimiters
    table_patterns = [
        r"\d+\.\d+%?",  # Matches numbers like 5.4% or 45.0
        r"\d+\s+[-\d]",  # Matches numbers with spaces or ranges (e.g., 10 - 20)
        r"(\n\s*){2,}",  # Multiple newlines with spaces (common in table rows)
        r"\t",           # Tab-separated content
    ]
    return any(re.search(pattern, text) for pattern in table_patterns)

def extract_text_from_images(image_dict):
    """Extract text from images using Tesseract OCR and filter for table-like content."""
    extracted_data = {}
    for page_no, image in image_dict.items():
        print(f"\nProcessing Page {page_no}...")
        text = pytesseract.image_to_string(image, lang="eng").strip()  # Extract text
        if is_table_like(text):  # Only store if it resembles a table
            extracted_data[page_no] = text
            print(f"Table-like content found on Page {page_no}:\n{text}\n{'-'*50}")
    return extracted_data

def create_vector_db(page_data):
    """Create FAISS vector database from table-like text with page numbers."""
    documents = [
        Document(page_content=text, metadata={"page": page_no})
        for page_no, text in page_data.items() if text  # Already filtered for tables
    ]
    if not documents:
        raise Exception("No table-like text extracted to create vector DB")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

def initialize_retrieval_chain(vector_db):
    """Initialize LangChain Q&A chain with Gemini."""
    retriever = vector_db.as_retriever(search_kwargs={"k": 1})  # Return top 1 match
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def get_page_number(qa_chain, query):
    """Get the page number for the matched table data based on the query."""
    response = qa_chain.invoke({"query": query})
    source_doc = response["source_documents"][0]  # Get the top matched document
    page_number = source_doc.metadata["page"]
    return page_number

# Main execution
def pdf_rag(pdf_path):
    pdf_path = "nss_rep_22.pdf"  # Your PDF path
    print("Converting PDF to images and extracting text with OCR...")

    # Convert PDF to images and extract table-like text
    image_dict = pdf_to_images(pdf_path)
    extracted_data = extract_text_from_images(image_dict)

    # Create FAISS vector database with table data
    print("\nCreating FAISS Vector Database with OpenAI Embeddings (Tables Only)...")
    vector_db = create_vector_db(extracted_data)

    # Initialize retrieval chain
    print("\nInitializing LangChain Q&A with Gemini...")
    qa_chain = initialize_retrieval_chain(vector_db)

    # Query loop
    while True:
        user_query = input("\nEnter your query about tables (or 'exit' to stop): ")
        if user_query.lower() == "exit":
            break
        try:
            page_number = get_page_number(qa_chain, user_query)
            print(f"Page Number with relevant table: {page_number}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")