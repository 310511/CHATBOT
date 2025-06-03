import streamlit as st
st.set_page_config("Chat PDF")

import os
import sys
import tempfile
from dotenv import load_dotenv
import socket
import dns.resolver
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Basic imports
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Try importing pdf2image
try:
    import pdf2image
    from pdf2image import convert_from_path
except ImportError as e:
    st.error(f"Failed to import pdf2image: {str(e)}")
    st.error("Please make sure pdf2image is installed correctly.")
    st.stop()

# Try importing pytesseract
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
except ImportError as e:
    st.error(f"Failed to import pytesseract: {str(e)}")
    st.error("Please make sure pytesseract is installed correctly.")
    st.stop()

# Load API keys from Streamlit secrets
load_dotenv() # Commented out
gemini_api_key = st.secrets["GOOGLE_API_KEY"]
gorq_api_key = st.secrets["GORQ_API_KEY"]

if not gemini_api_key or not gorq_api_key:
    st.error("Please set both GOOGLE_API_KEY and GORQ_API_KEY in Streamlit secrets (`.streamlit/secrets.toml`).")
    st.stop()

genai.configure(api_key=gemini_api_key)

# Initialize session state for chat history and vector store
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Gemini"

def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        return text
    
    for pdf in pdf_docs:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # First try normal text extraction
            pdf_reader = PdfReader(tmp_file_path)
            page_text = ""
            for page in pdf_reader.pages:
                page_text += page.extract_text()
            
            # If no text was extracted, the PDF might be scanned
            if not page_text.strip():
                # Add /usr/bin to PATH for pdf2image on Streamlit Cloud
                # This is a common location for poppler-utils binaries
                original_path = os.environ.get("PATH", "")
                os.environ["PATH"] = "/usr/bin:" + original_path
                
                # Debugging: Check for pdfinfo in PATH and /usr/bin
                pdfinfo_in_path = False
                pdfinfo_in_usr_bin = False
                pdfinfo_path = ""

                try:
                    # Check if pdfinfo is in the PATH and executable
                    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
                        pdfinfo_candidate = os.path.join(path_dir, "pdfinfo")
                        if os.path.exists(pdfinfo_candidate) and os.access(pdfinfo_candidate, os.X_OK):
                            pdfinfo_in_path = True
                            pdfinfo_path = pdfinfo_candidate
                            break
                    st.write(f"Debug: pdfinfo found in PATH: {pdfinfo_in_path}")
                    if pdfinfo_in_path:
                        st.write(f"Debug: pdfinfo path: {pdfinfo_path}")
                except Exception as e:
                    st.write(f"Debug: Error checking for pdfinfo in PATH: {str(e)}")

                try:
                    # Check specifically for pdfinfo in /usr/bin
                    usr_bin_pdfinfo = "/usr/bin/pdfinfo"
                    if os.path.exists(usr_bin_pdfinfo) and os.access(usr_bin_pdfinfo, os.X_OK):
                        pdfinfo_in_usr_bin = True
                    st.write(f"Debug: pdfinfo found and executable in /usr/bin: {pdfinfo_in_usr_bin}")
                except Exception as e:
                    st.write(f"Debug: Error checking for pdfinfo in /usr/bin: {str(e)}")

                try:
                    # Convert PDF to images
                    images = convert_from_path(tmp_file_path)
                finally:
                    # Restore original PATH
                    os.environ["PATH"] = original_path
                    
                # Perform OCR on each page
                for image in images:
                    page_text += pytesseract.image_to_string(image)
            
            text += page_text

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
        finally:
            # Clean up the temporary file
            os.unlink(tmp_file_path)
            
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    vector_store = InMemoryVectorStore.from_texts(texts=text_chunks, embedding=embeddings)
    st.session_state.vector_store = vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             temperature=0.3,
                             google_api_key=gemini_api_key)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def resolve_domain(domain):
    try:
        # Try to resolve the domain
        answers = dns.resolver.resolve(domain, 'A')
        return [str(rdata) for rdata in answers]
    except Exception as e:
        st.error(f"DNS resolution failed for {domain}: {str(e)}")
        return None

def process_with_gorq(question, context):
    # Debug information - Commented out
    # st.write("Debug Info:")
    # st.write(f"API Key length: {len(gorq_api_key) if gorq_api_key else 0}")
    
    # Use the correct Groq API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {gorq_api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Updated request format with Groq model
    data = {
        "model": "gemma2-9b-it",  # Using gemma2-9b-it model from the provided list
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context. If the answer is not in the context, say 'answer is not available in the context'."
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 1,
        "stream": False,
        "n": 1
    }
    
    try:
        # Debug request - Commented out
        # st.write(f"\nTrying endpoint: {url}")
        # st.write("Request headers:", headers)
        # st.write("Request data:", data)
        
        # Make the API request with retries
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Try to resolve the domain first - Commented out
        # try:
        #     import socket
        #     ip_address = socket.gethostbyname('api.groq.com')
        #     st.write(f"Resolved api.groq.com to IP: {ip_address}")
            
        #     # Try to establish a test connection - Commented out
        #     test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #     test_socket.settimeout(5)
        #     test_socket.connect((ip_address, 443))
        #     test_socket.close()
        #     st.write("Successfully established test connection to api.groq.com")
        # except Exception as e:
        #     st.write(f"Connection test error: {str(e)}")
        
        # Make the actual API request
        # st.write("\nMaking API request...") # Commented out
        response = session.post(
            url, 
            headers=headers, 
            json=data, 
            timeout=30,
            verify=True
        )
        
        # Debug response - Commented out
        # st.write("\nResponse details:")
        # st.write("Status code:", response.status_code)
        # st.write("Response headers:", dict(response.headers))
        # st.write("Raw response content:", response.text)
        
        # Check for specific HTTP status codes
        if response.status_code == 401:
            st.error("Invalid Groq API key. Please check your credentials.")
            # st.write("Full response:", response.text) # Commented out
            return "Authentication failed. Please check your Groq API key."
        elif response.status_code == 404:
            st.error("Groq API endpoint not found. Please check the API documentation.")
            # st.write("Full response:", response.text) # Commented out
            return "API endpoint not found. Please try again later."
        elif response.status_code != 200:
            st.error(f"Groq API returned status code {response.status_code}")
            # st.write("Response content:", response.text) # Commented out
            return f"API error: {response.status_code}. Please try again later."
            
        response.raise_for_status()
        
        try:
            response_data = response.json()
            # st.write("\nParsed response data:", response_data) # Commented out
            
            # Extract the response text
            if "choices" in response_data and len(response_data["choices"]) > 0:
                if "message" in response_data["choices"][0]:
                    return response_data["choices"][0]["message"]["content"].strip()
                elif "text" in response_data["choices"][0]:
                    return response_data["choices"][0]["text"].strip()
            # st.write("Unexpected response format:", response_data) # Commented out
            return "Unexpected response format from Groq API."
        except Exception as e:
            st.error(f"Error parsing response: {str(e)}")
            # st.write("Raw response:", response.text) # Commented out
            return "Error parsing API response. Please try again."
        
    except requests.exceptions.SSLError as e:
        st.error(f"SSL verification failed: {str(e)}")
        # st.write("SSL Error details:", str(e)) # Commented out
        return "Secure connection failed. Please check your network settings."
    except requests.exceptions.ConnectionError as e:
        st.error(f"Connection error: {str(e)}")
        # Try to get more information about the connection error - Commented out
        # try:
        #     import socket
        #     st.write(f"Socket error details: {socket.gethostbyname('api.groq.com')}")
        # except Exception as socket_error:
        #     st.write(f"Socket error: {str(socket_error)}")
        return "Connection failed. Please check your internet connection and try again."
    except requests.exceptions.Timeout as e:
        st.error(f"Timeout error: {str(e)}")
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {str(e)}")
        # st.write("Request error details:", str(e)) # Commented out
        return "I apologize, but I'm having trouble connecting to the Groq service. Please try again later or switch to Gemini."
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        # st.write("Error details:", str(e)) # Commented out
        return "An unexpected error occurred. Please try again or switch to Gemini."

def process_user_input(user_question):
    if st.session_state.vector_store is None:
        st.warning("Please upload and process PDF files first.")
        return
    
    docs = st.session_state.vector_store.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    
    if st.session_state.selected_model == "Gemini":
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True)
        return response["output_text"]
    else:  # Gorq
        return process_with_gorq(user_question, context)

def main():
    st.header("Chat with PDF using AI💁")

    # Model selection in sidebar
    with st.sidebar:
        st.title("Menu:")
        st.write("📚 Drag and drop files here\nLimit 200MB per file\n(Maximum 10 PDFs allowed)")
        
        # Model selection
        st.session_state.selected_model = st.radio(
            "Select AI Model:",
            ["Gemini", "Gorq"],
            index=0 if st.session_state.selected_model == "Gemini" else 1
        )
        
        pdf_docs = st.file_uploader("Select your PDF files", accept_multiple_files=True)
        
        # Display file count
        if pdf_docs:
            st.write(f"📄 Files selected: {len(pdf_docs)}/10")
            
        if pdf_docs and len(pdf_docs) > 10:
            st.error("Maximum 10 PDF files allowed. Please reduce the number of files.")
        elif st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No extractable text found in the uploaded PDFs.")
                        return
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                    # Clear chat history when new documents are processed
                    st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input(f"Ask a question about your PDF (using {st.session_state.selected_model})"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            response = process_user_input(prompt)
            st.write(response)
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()