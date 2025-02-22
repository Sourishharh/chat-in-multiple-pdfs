import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs in parallel
def get_pdf_text(pdf_docs):
    text = ""
    
    def extract_text(pdf):
        pdf_reader = PdfReader(pdf)
        return " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(extract_text, pdf_docs)

    return " ".join(results)

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    return text_splitter.split_text(text)

# Cache embeddings & vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if vector store is cached
    if os.path.exists("vector_store.pkl"):
        with open("vector_store.pkl", "rb") as f:
            vector_store = pickle.load(f)
    else:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        with open("vector_store.pkl", "wb") as f:
            pickle.dump(vector_store, f)

    st.session_state.vector_store = vector_store

# Set up conversational chain with optimized prompt
def get_conversational_chain():
    prompt_template = """
    Answer the question in English as accurately as possible from the provided context. 
    If the context is not sufficient, explain what additional information is needed. 
    Avoid providing incorrect answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=200)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user query efficiently
def user_input(user_question):
    if "chat_chain" not in st.session_state or st.session_state.chat_chain is None:
        st.warning("Chat chain is not initialized. Please process PDF files first.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    vector_store = st.session_state.vector_store
    if vector_store:
        docs = vector_store.similarity_search(user_question, k=2)  # Reduced k for speed
        if not docs:
            st.write("No relevant documents found.")
            return

        try:
            response = st.session_state.chat_chain.invoke({
                "input_documents": docs,
                "question": user_question
            })

            answer = response.get("output_text", "No answer found.")
            st.session_state.chat_history.append((user_question, answer))
            st.write("Reply:", answer)

        except Exception as e:
            st.error(f"Error during response generation: {e}")
    else:
        st.warning("No vector store found. Please process the PDFs first.")

# Main Streamlit app
def main():
    st.set_page_config(page_title="PDF Chatbot with Memory", layout="wide")
    st.header("Chat with Multiple PDFs 📚")

    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input section
    user_question = st.text_input("Ask a question from the uploaded PDF files:")
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
        st.title("Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.chat_chain = get_conversational_chain()
                        st.success("PDF processing complete! You can now ask questions.")
                    else:
                        st.error("No text could be extracted from the uploaded PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
