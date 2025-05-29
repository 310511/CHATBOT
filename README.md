# Chat with PDF using Gemini and Streamlit

This project is a Streamlit web application that allows you to chat with your PDF documents using Google's Gemini models for generating responses and Langchain for handling the document processing and vector storage.

## Features

* Upload multiple PDF files.
* Extract text from PDFs.
* Create vector embeddings of the text content using Google Generative AI embeddings.
* Store embeddings in an in-memory vector store for quick similarity search.
* Answer questions based on the content of the uploaded PDFs using a conversational chain with Google's Gemini model.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/310511/CHATBOT.git
    cd CHATBOT
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

4.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set up your Google API Key:**
    - Create a `.env` file in the root directory of the project.
    - Add your Google API key in the following format:
      ```
      GOOGLE_API_KEY=YOUR_API_KEY
      ```
      Replace `YOUR_API_KEY` with your actual Google Generative AI API key. Alternatively, if deploying on Streamlit Cloud, you can use Streamlit secrets.

## Running the App

1.  Make sure your virtual environment is activated (`source .venv/bin/activate`).
2.  Run the Streamlit application:
    ```bash
    streamlit run chatbot/app.py
    ```
3.  The app will open in your web browser.

## Deployment on Streamlit Cloud

This application is configured to be easily deployed on Streamlit Cloud. Ensure you have:

1.  Pushed your latest code to your GitHub repository.
2.  Added your `GOOGLE_API_KEY` as a secret in your Streamlit Cloud app settings.
3.  Configured the main file path to `chatbot/app.py`.

Streamlit Cloud will automatically install the dependencies from `requirements.txt`. 