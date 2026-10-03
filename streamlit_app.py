# streamlit_app.py
# Streamlit version of your projects.ipynb
# Run with:  streamlit run streamlit_app.py

import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ---- Same settings as your notebook ----
INDEX_NAME = "hybrid-search-rag"
EMBED_MODEL = "all-MiniLM-L6-v2"
BM25_FILE = "bm25_values.json"
LLM_MODEL = "gemini-3.8-flash"      # same model name you used in the notebook

st.set_page_config(page_title="Hybrid Search RAG", page_icon="🔎")
st.title("🔎 Hybrid Search RAG")
st.caption("Pinecone (dense + BM25) • HuggingFace embeddings • Gemini")


# ---- Load heavy objects only once (cached) ----
@st.cache_resource
def load_components():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    bm25_encoder = BM25Encoder().load(BM25_FILE)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return index, embeddings, bm25_encoder, llm


index, embeddings, bm25_encoder, llm = load_components()


def get_text(content):
    """Return plain text from a Gemini response (string or list of blocks)."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# ---- Sidebar settings ----
st.sidebar.header("Settings")
alpha = st.sidebar.slider("Alpha (0 = keyword, 1 = semantic)", 0.0, 1.0, 0.5, 0.1)
top_k = st.sidebar.slider("Chunks to retrieve (top_k)", 1, 8, 3)
show_sources = st.sidebar.checkbox("Show sources", value=True)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

# Retriever is cheap to create, so rebuild it when sliders change
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,
    index=index,
    alpha=alpha,
    top_k=top_k,
)


def ask(question):
    docs = retriever.invoke(question)
    context = "\n".join(d.page_content for d in docs)
    prompt = (
        f"Answer using only this context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    answer = get_text(llm.invoke(prompt).content)
    return answer, docs


# ---- Chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat input ----
question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            answer, docs = ask(question)
        st.markdown(answer)

        if show_sources:
            with st.expander("Sources"):
                for d in docs:
                    score = d.metadata.get("score", 0)
                    st.write(f"**Score {score:.2f}**: {d.page_content}")

    st.session_state.messages.append({"role": "assistant", "content": answer})