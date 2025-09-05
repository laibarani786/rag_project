import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()

# Streamlit page setup
st.set_page_config(
    page_title="📄 RAG Q&A with PDF & Chat History",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📄 RAG Q&A with PDF uploads and chat history")

# Sidebar
st.sidebar.header("👨🏻‍🔧 Configuration")
st.sidebar.write(
    "- Enter your GROQ API Key \n"
    "- Upload PDFs on the main page \n"
    "- Ask questions and see chat history"
)

# API Key & embeddings
api_key = st.sidebar.text_input("Groq API Key", type="password")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

if not api_key:
    st.warning(" 🔑 Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Groq LLM
llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

# PDF uploader
uploaded_files = st.file_uploader(
    "📓 🗒 Choose PDF file(s)",
    type="pdf",
    accept_multiple_files=True,
)

all_docs = []
if uploaded_files:
    with st.spinner(" 🔄 Loading and splitting PDFs "):
        for pdf in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.getvalue())
                pdf_path = tmp.name

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(all_docs)

    # ---- In-memory FAISS vector store ----
    @st.cache_resource(show_spinner=False)
    def get_vectorstore(_splits, embeddings):
        return FAISS.from_documents(_splits, embeddings)

    vectorstore = get_vectorstore(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Contextualizer prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, decide what to retrieve."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # History-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    # QA chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant. Use the retrieved context to answer. "
                    "If you don't know, say so. Keep it under three sentences. \n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Chat history session
    if "chathistory" not in st.session_state:
        st.session_state.chathistory = {}

    def get_history(session_id: str):
        if session_id not in st.session_state.chathistory:
            st.session_state.chathistory[session_id] = ChatMessageHistory()
        return st.session_state.chathistory[session_id]

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Session input
    session_id = st.text_input("🆔 Session ID", value="default_session")
    user_question = st.chat_input(" ✍️ Your question here....")

    if user_question:
        history = get_history(session_id)
        result = conversational_rag.invoke(
            {"input": user_question},
            config={"configurable": {"session_id": session_id}},
        )
        answer = result["answer"]

        st.chat_message("user").write(user_question)
        st.chat_message("assistant").write(answer)

        with st.expander(" 📖 Full chat history"):
            for msg in history.messages:
                role = getattr(msg, "role", msg.type)
                content = msg.content
                st.write(f"**{role.title()}:** {content}")

else:
    st.info("ℹ️ Upload one or more PDFs above to begin.")


