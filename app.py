import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser


# --- HELPER FUNCTIONS ---

@st.cache_resource
def load_and_process_pdfs(uploaded_files):
    """
    Loads PDF files, splits them into chunks, and creates a retriever.
    The @st.cache_resource decorator ensures this heavy lifting is done only once.
    """
    documents = []
    for uploaded_file in uploaded_files:
        # Use a temporary file to handle the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())

    # 1. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # 2. Create embeddings and vector store
    # Ensure the GOOGLE_API_KEY is set in the environment for the embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(texts, embeddings)

    # 3. Create the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever, documents


# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="NPTEL Prep Buddy", page_icon="📘", layout="wide")

st.title("📘 NPTEL Prep Buddy – Your AI Study Assistant")
st.markdown("""
Welcome to your personal NPTEL study assistant! 
1.  Enter your Google Gemini API key in the sidebar.
2.  Upload your NPTEL course PDFs.
3.  Interact with your documents using the Q&A, Summarization, and Quiz tabs.
""")

# --- SIDEBAR FOR API KEY AND FILE UPLOAD ---

with st.sidebar:
    st.header("Setup")
    google_api_key = st.text_input("Enter Google Gemini API Key", type="password")
    
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key
        st.success("API Key set successfully!")

    uploaded_files = st.file_uploader(
        "Upload NPTEL PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="You can upload multiple PDF files from your course."
    )

# --- MAIN APP LOGIC ---

if not google_api_key:
    st.info("Please enter your Google Gemini API Key to proceed.")
    st.stop()

if uploaded_files:
    try:
        with st.spinner("Processing your PDFs... This may take a moment."):
            retriever, documents = load_and_process_pdfs(uploaded_files)

        # Initialize session state for chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, convert_system_message_to_human=True)
        
        # Main UI with tabs
        tab1, tab2, tab3 = st.tabs(["💬 Conversational Q&A", "📝 Summarization", "❓ Quiz Generator"])

        # --- Q&A TAB ---
        with tab1:
            st.subheader("Ask Questions About Your Documents")
            
            # Display chat history
            for sender, message in st.session_state.chat_history:
                with st.chat_message("user" if sender == "You" else "assistant"):
                    st.markdown(message)
            
            # Chat input
            user_query = st.chat_input("Type your question here...")

            if user_query:
                with st.chat_message("user"):
                    st.markdown(user_query)
                
                with st.spinner("Prep Buddy is thinking..."):
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=retriever,
                        return_source_documents=True
                    )
                    
                    # We pass the history as a list of tuples
                    history_tuples = [(q, a) for sender, q, a in st.session_state.get('full_history', [])]
                    result = qa_chain({"question": user_query, "chat_history": history_tuples})
                    answer = result["answer"]
                    
                    # Store and display the new message
                    st.session_state.chat_history.append(("You", user_query))
                    st.session_state.chat_history.append(("Prep Buddy", answer))

                    # For the actual chain history, store query and answer
                    if 'full_history' not in st.session_state:
                        st.session_state.full_history = []
                    st.session_state.full_history.append(("user", user_query, answer))
                    
                    with st.chat_message("assistant"):
                        st.markdown(answer)

        # --- SUMMARIZATION TAB ---
        with tab2:
            st.subheader("Get a Quick Summary")
            st.markdown("Generate concise, bullet-point summaries of your uploaded content for quick revision.")

            if st.button("Generate Summary", key="summarize_btn"):
                with st.spinner("Creating a summary of all documents..."):
                    # Concatenate content from all documents
                    full_text = "\n\n".join([doc.page_content for doc in documents])
                    
                    # Create the prompt template
                    summary_template = """
                    You are an expert academic summarizer. Based on the provided NPTEL course content, create a concise, easy-to-read summary.
                    Focus on the key concepts, definitions, and important topics.
                    Present the summary in well-structured bullet points.

                    Content to summarize:
                    {text}
                    """
                    summary_prompt = PromptTemplate.from_template(summary_template)
                    
                    # Create and run the summarization chain
                    summary_chain = summary_prompt | llm | StrOutputParser()
                    summary = summary_chain.invoke({"text": full_text})
                    
                    st.markdown("### Here's your summary:")
                    st.markdown(summary)

        # --- QUIZ GENERATOR TAB ---
        with tab3:
            st.subheader("Test Your Knowledge")
            st.markdown("Generate multiple-choice questions (MCQs) based on the PDF content to prepare for exams.")
            
            if st.button("Generate 5 Quiz Questions", key="quiz_btn"):
                with st.spinner("Generating your quiz... Good luck!"):
                    # Concatenate content from all documents
                    quiz_text = "\n\n".join([doc.page_content for doc in documents])
                    
                    # Create the prompt template for the quiz
                    quiz_template = """
                    You are an expert NPTEL quiz creator. Based on the provided course material, generate 5 multiple-choice questions (MCQs) to test a student's understanding.
                    For each question:
                    1.  Provide 4 distinct options (A, B, C, D).
                    2.  Clearly indicate the correct answer after the options.

                    Course Material:
                    {text}
                    """
                    quiz_prompt = PromptTemplate.from_template(quiz_template)
                    
                    # Create and run the quiz generation chain
                    quiz_chain = quiz_prompt | llm | StrOutputParser()
                    quiz_content = quiz_chain.invoke({"text": quiz_text})
                    
                    st.markdown("### Your Practice Quiz:")
                    st.markdown(quiz_content)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure your Google Gemini API key is correct and has been enabled for use.")

else:
    st.info("Upload your PDF files in the sidebar to get started.")

