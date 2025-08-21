import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# IMPORTANT: Set your own Google Gemini API key here or as an environment variable
# Example (Linux/Mac): export GOOGLE_API_KEY="your_api_key"
# Example (Windows PowerShell): setx GOOGLE_API_KEY "your_api_key"
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_GEMINI_API_KEY"

# Streamlit UI
st.set_page_config(page_title="NPTEL Prep Buddy", page_icon="📘")
st.title("NPTEL Prep Buddy – Generative AI Study Assistant")
st.markdown("Upload your NPTEL PDFs and use your AI assistant for Q&A, summaries, and quiz preparation.")

# File uploader
uploaded_files = st.file_uploader("Upload NPTEL PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings & ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Gemini Pro Chat model
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever
    )

    # Session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Tabs for Q&A, Summarization, Quiz
    tab1, tab2, tab3 = st.tabs(["Q&A", "Summarization", "Quiz Generator"])

    # ===== Q&A TAB =====
    with tab1:
        st.subheader("Ask your NPTEL Prep Buddy")
        query = st.text_input("Type your question here...")

        if query:
            result = qa({"question": query, "chat_history": st.session_state.chat_history})
            st.session_state.chat_history.append((query, result["answer"]))

            for q, a in st.session_state.chat_history:
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Prep Buddy:** {a}")

    # ===== SUMMARIZATION TAB =====
    with tab2:
        st.subheader("Get a Summary of Uploaded PDFs")
        if st.button("Generate Summary"):
            full_text = " ".join([doc.page_content for doc in documents])
            summary_prompt = PromptTemplate(
                input_variables=["text"],
                template="Summarize the following NPTEL content into concise bullet points for quick revision:\n\n{text}"
            )
            summary_chain = summary_prompt | llm
            summary = summary_chain.invoke({"text": full_text})
            st.markdown("### Summary")
            st.write(summary.content)

    # ===== QUIZ GENERATOR TAB =====
    with tab3:
        st.subheader("Generate Quiz Questions")
        if st.button("Generate Quiz"):
            quiz_text = " ".join([doc.page_content for doc in documents])
            quiz_prompt = PromptTemplate(
                input_variables=["text"],
                template="From the following NPTEL content, generate 5 MCQ-style quiz questions with 4 options each and indicate the correct answer:\n\n{text}"
            )
            quiz_chain = quiz_prompt | llm
            quiz = quiz_chain.invoke({"text": quiz_text})
            st.markdown("### Quiz Questions")
            st.write(quiz.content)

