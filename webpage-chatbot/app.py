import os
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model=genai.GenerativeModel('gemini-1.5-pro')

def get_gemini_response(context_text, question):
    prompt = (
        "You are a helpful AI assistant. Use the webpage context below to answer the question.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}"
    )
    return model.generate_content(prompt).text

def get_page_content(url):
    base_url = url
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    clean_text = soup.get_text(separator=' ', strip=True)
    return clean_text

def store_memory(content):
    splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    length_function=len
                )
    chunks = splitter.create_documents([content])
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 4}
                )
    return retriever

st.set_page_config(page_title="🌐 WebPage Chatbot", page_icon="🧠")

url_input = st.text_input("🔗 Enter Webpage URL")

if st.button("📥 Submit Link"):
    with st.spinner("Fetching and processing content..."):
        page_content = get_page_content(url_input)
        if "Error" in page_content:
            st.error(page_content)
        else:
            retriever = store_memory(page_content)
            st.success("Webpage content loaded successfully!")
            st.session_state["retriever"] = retriever
            st.session_state["content_loaded"] = True

# Show question input only after content is loaded
if st.session_state.get("content_loaded"):

    st.markdown("---")

    # Initialize state
    if "question" not in st.session_state:
        st.session_state["question"] = ""
    if "generating" not in st.session_state:
        st.session_state["generating"] = False

    # Define a function to handle answer generation
    def generate_answer():
        st.session_state["generating"] = True
        with st.spinner("Thinking..."):
            retrieved_docs = st.session_state["retriever"].invoke(st.session_state["question"])
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            response = get_gemini_response(context_text, st.session_state["question"])
            st.session_state["answer"] = response
        st.session_state["question"] = ""  # Clear input after generation
        st.session_state["generating"] = False

    # Input + Button
    st.text_input("❓ Ask a question about the webpage:", key="question")
    st.button("🤖 Get Answer", on_click=generate_answer, disabled=st.session_state["generating"])

    # Display response
    if "answer" in st.session_state and st.session_state["answer"]:
        st.subheader("🧠 Gemini's Answer:")
        st.write(st.session_state["answer"])

