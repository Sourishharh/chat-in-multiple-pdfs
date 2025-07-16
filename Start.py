import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    st.error("Missing GOOGLE_API_KEY. Please check your .env file.")

MODEL_NAME = "gemini-1.5-pro"  

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle None cases
    return text.strip()

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to create embeddings and vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    st.session_state.vector_store = vector_store

# Function to set up conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question in English as accurately as possible from the provided context.
    If the context is not sufficient, explain what additional information is needed.
    Avoid providing incorrect answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input
def user_input(user_question):
    if not st.session_state.vector_store:
        st.warning("⚠️ Please upload and process PDFs first.")
        return

    vector_store = st.session_state.vector_store
    docs = vector_store.similarity_search(user_question, k=3)

    if not docs:
        st.write(" No relevant documents found.")
        return

    try:
        response = st.session_state.chat_chain.invoke({
            "input_documents": docs,
            "question": user_question
        })

        answer = response.get("output_text", response.get("answer", "Unexpected response format."))

        # Store chat history
        st.session_state.chat_history.append((user_question, answer))
        st.write("**Reply:**", answer)

    except Exception as e:
        st.error(f" Error generating response: {e}")

# Main function
def main():
    st.set_page_config(page_title="📚 PDF Chatbot with Memory", layout="wide")
    st.header("💬 Chat with Multiple PDFs")

    # User input section
    user_question = st.text_input("📝 Ask a question from the uploaded PDFs:")
    if user_question:
        user_input(user_question)

        # Display conversation history
        if st.session_state.chat_history:
            st.subheader("Conversation History")
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.write(f"**Q{i+1}:** {q}")
                st.write(f"**A{i+1}:** {a}")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("📁 Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        # Process PDFs
        if st.button("📂 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.chat_chain = get_conversational_chain()
                        st.success(" PDF processing complete! You can now ask questions.")
                    else:
                        st.error(" No extractable text found in uploaded PDFs.")
            else:
                st.error("⚠️ Please upload at least one PDF.")

if __name__ == "__main__":
    main()