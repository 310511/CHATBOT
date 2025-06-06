import streamlit as st
st.set_page_config("Chat PDF")

import os
import sys
import tempfile
from dotenv import load_dotenv
load_dotenv()  # Load environment variables

import dns.resolver
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# PDF and text processing
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

try:
    import pytesseract
    # Try to find tesseract in common locations
    tesseract_paths = [
        '/opt/homebrew/bin/tesseract',  # Homebrew on Apple Silicon
        '/usr/local/bin/tesseract',     # Homebrew on Intel
        '/usr/bin/tesseract'            # System default
    ]
    
    tesseract_found = False
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            break
    
    if not tesseract_found:
        st.error("Tesseract not found. Please install it using: brew install tesseract")
        st.stop()
except ImportError as e:
    st.error(f"Failed to import pytesseract: {str(e)}")
    st.stop()

# Load API keys
gemini_api_key = os.getenv("GOOGLE_API_KEY")
gorq_api_key = os.getenv("GORQ_API_KEY")

try:
    online_ocr_api = os.getenv("ONLINE_OCR_API")
except KeyError:
    online_ocr_api = None

if not gemini_api_key or not gorq_api_key or not online_ocr_api:
    missing_keys = []
    if not gemini_api_key:
        missing_keys.append("GOOGLE_API_KEY")
    if not gorq_api_key:
        missing_keys.append("GORQ_API_KEY")
    if not online_ocr_api:
        missing_keys.append("ONLINE_OCR_API")
    st.error(f"Missing API keys in Streamlit secrets: {', '.join(missing_keys)}")
    st.stop()

genai.configure(api_key=gemini_api_key)

# Session state initialization
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Gemini"

def process_with_online_ocr(pdf_content, api_key):
    """Process PDF content using the configured OCR API."""
    if not api_key:
        st.error("OCR API key is not configured")
        return None
        
    # Replace this URL with your actual OCR API endpoint
    api_url = "https://api.ocr.space/parse/image"  # Example using OCR.space API
    
    try:
        # Set up retry mechanism for better reliability
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # For OCR.space API
        payload = {
            'apikey': api_key,
            'language': 'eng',
            'isOverlayRequired': 'false',
            'OCREngine': '2'
        }
        
        files = {
            'file': ('document.pdf', pdf_content, 'application/pdf')
        }
        
        response = session.post(
            api_url,
            files=files,
            data=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        if result.get('IsErroredOnProcessing'):
            st.error(f"OCR API Error: {result.get('ErrorMessage')}")
            return None
            
        # Extract text from all pages
        extracted_text = ""
        for parsed_result in result.get('ParsedResults', []):
            extracted_text += parsed_result.get('ParsedText', '') + "\n"
            
        return extracted_text.strip()
        
    except requests.exceptions.RequestException as e:
        st.error(f"OCR API request failed: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error during OCR processing: {str(e)}")
        return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf.getvalue())
            tmp_file_path = tmp_file.name

        try:
            pdf_reader = PdfReader(tmp_file_path)
            page_text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

            if not page_text.strip():
                st.info("No text extracted, attempting OCR...")
                with open(tmp_file_path, 'rb') as f:
                    pdf_bytes = f.read()
                ocr_text = process_with_online_ocr(pdf_bytes, online_ocr_api)
                text += ocr_text or ""
            else:
                text += page_text

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            os.unlink(tmp_file_path)

    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    st.session_state.vector_store = InMemoryVectorStore.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversational_chain():
    prompt_template = """
    Answer the question using the provided context.
    If the answer is not in the context, respond with: "answer is not available in the context"

    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=gemini_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def resolve_domain(domain):
    try:
        answers = dns.resolver.resolve(domain, 'A')
        return [str(rdata) for rdata in answers]
    except Exception as e:
        st.error(f"DNS resolution failed: {str(e)}")
        return None

def process_with_gorq(question, context):
    """Handles interaction with Groq API."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {gorq_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    data = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "Answer based on context. Say 'answer is not available in the context' if needed."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1,
        "stream": False,
        "n": 1
    }

    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        response = session.post(url, headers=headers, json=data, timeout=30, verify=True)
        response.raise_for_status()
        response_data = response.json()

        choices = response_data.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"]["content"].strip()
        elif choices and "text" in choices[0]:
            return choices[0]["text"].strip()
        return "Unexpected response format from Groq API."

    except requests.exceptions.RequestException as e:
        st.error(f"Groq API request error: {str(e)}")
        return "Groq API request failed."
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return "An unexpected error occurred."

def process_user_input(user_question):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDF files first.")
        return

    docs = st.session_state.vector_store.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])

    if st.session_state.selected_model == "Gemini":
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    else:
        return process_with_gorq(user_question, context)

def main():
    st.header("Chat with PDF using AI\U0001F481")

    with st.sidebar:
        st.title("Menu:")
        st.write("\U0001F4DA Drag and drop files here (Max 10, 200MB each)")

        st.session_state.selected_model = st.radio(
            "Select AI Model:",
            ["Gemini", "Gorq"],
            index=0 if st.session_state.selected_model == "Gemini" else 1
        )

        pdf_docs = st.file_uploader("Select your PDF files", accept_multiple_files=True)
        if pdf_docs:
            st.write(f"\U0001F4C4 Files selected: {len(pdf_docs)}/10")

        if pdf_docs and len(pdf_docs) > 10:
            st.error("Maximum 10 PDF files allowed.")
        elif st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No extractable text found in uploaded PDFs.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete.")
                    st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(f"Ask a question about your PDF (using {st.session_state.selected_model})"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            response = process_user_input(prompt)
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
