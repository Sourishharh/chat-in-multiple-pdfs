import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables 
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create embeddings and vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    st.session_state.vector_store = vector_store
    st.write("Semantic index created with chunks :")
    for chunk in text_chunks:
        st.write(chunk[:500])  # Display the first 500 characters of each chunk

# Set up the conversational chain & LLM
def get_conversational_chain():
    prompt_template = """
    Answer the question in English as accurately as possible from the provided context. 
    If the context is not sufficient, explain what additional information is needed. 
    Avoid providing incorrect answers.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Process user query
def user_input(user_question):
    if "chat_chain" not in st.session_state or st.session_state.chat_chain is None:
        st.warning("Chat chain is not initialized. Please process PDF files first.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    vector_store = st.session_state.vector_store
    if vector_store:
        docs = vector_store.similarity_search(user_question, k=3)
        if not docs:
            st.write("No documents retrieved for the query.")
            return

        st.write("Retrieved documents:")
        for doc in docs:
            st.write(doc.page_content[:500])  # Display the first 500 characters of each document

        try:
            response = st.session_state.chat_chain.invoke({
                "input_documents": docs,
                "question": user_question,
                "chat_history": st.session_state.chat_history
            })

            st.write("Raw response:", response)  # Debugging the structure of the response

            # Extract the answer based on the observed structure of the response
            if isinstance(response, dict) and "output_text" in response:
                answer = response["output_text"]
            elif isinstance(response, dict) and "answer" in response:
                answer = response["answer"]
            else:
                answer = "Unexpected response format. Please check the response structure."

            st.session_state.chat_history.append((user_question, answer))
            st.write("Reply: ", answer)

        except Exception as e:
            st.error(f"Error during response generation: {e}")
    else:
        st.warning("No vector store found. Please process the PDFs first.")

# Main function
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
