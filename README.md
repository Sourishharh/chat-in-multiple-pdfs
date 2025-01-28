
# Chat with Multiple PDFs

A brief description of what this project does and who it's for


## Overview

"Chat with Multiple PDFs" is a Streamlit-based application that allows users to interactively query and retrieve information from multiple PDF files. This tool uses state-of-the-art AI models for natural language understanding, making it an ideal companion for researchers, students, and professionals who need quick answers from large documents.
## Features

I. Upload and Process PDFs: Upload multiple PDFs for text extraction and indexing.

II. Semantic Search: Create a semantic index of the extracted text using Google Generative AI Embeddings.

III. Conversational AI: Ask natural language questions, and the chatbot will provide accurate answers based on the PDF content.

IV. Memory: Retains conversation history for a more personalized and contextual experience.

V. Error Handling: Provides feedback for issues such as invalid files or empty queries.
## Tech Stack

Frontend: Streamlit for an interactive UI.

Backend:

PyPDF2 for PDF text extraction.

LangChain for chain management and processing.

FAISS for semantic indexing.

Google Generative AI for embeddings and conversational AI.

Environment Management: python-dotenv for handling API keys.
## Installation

1. Clone the Repository:

    git clone https://github.com/Sourishharh/chat-in-multiple-pdfs.git

2. Set Up a Virtual Environment:

    python -m venv venv

    source venv/bin/activate # this Code for Mac & Linux

    venv\Scripts\activate # this Code for windows

3. Install Dependencies: 

    pip install -r requirements.txt

4. Set Up API Keys:
 
    Create a .env file in the project root.

    Add your Google Generative AI API Key:

    GOOGLE_API_KEY=your_google_api_key_here
    

## Usage

1. Run the Application:

    streamlit run app.py

2. Upload PDFs:
    
    i. Use the sidebar to upload multiple PDF files.

    ii.Click "Submit & Process" to extract and index the content.

3. Ask Questions:
    
    i. Enter your question in the input box and get answers based on the PDF content.

    ii.View conversation history for reference.


## File Structure

    chat-with-pdfs/
    ├── app.py                 
    ├── requirements.txt       
    ├── .env                   
    ├── README.md              
    └── ... (other files and folders)

    
## LICENSE
    This project is licensed under the MIT License. See the LICENSE file for details.
## Acknowledgments

    1.Streamlit
    2.LangChain
    3.Google Generative AI
    4.FAISS