from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

if 'retriever_ready' not in st.session_state:
    st.session_state.retriever_ready = False
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
llm=ChatGroq(model='llama3-70b-8192')
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    """Format retrieved documents into a single context string."""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def format_history(chat_history):
    return "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history
    )


prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on the provided context.

Context: {context}

Question: {question}

Answer: Based on the provided context, """,
    input_variables=['context', 'question']
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
language_mapping={
    "English":"en",
    "Hindi":"hi"
}

def main():
    st.subheader("Youtube Video Chatbot And Summariser")
    if not st.session_state.retriever_ready:
        input_url = st.text_input("Enter Video URL")
        select_language = st.selectbox("Choose Language", options=["English", "Hindi"])
        button = st.button("Fetch Video")

        if button:
            try:
                query = urlparse(input_url).query
                params = parse_qs(query)
                video_id = params.get("v", [None])[0]
                language = language_mapping[select_language]
                api_fetcher=YouTubeTranscriptApi()

                transcript = api_fetcher.fetch(video_id=video_id, languages=[language])
                full_text = "\n".join(snippet.text for snippet in transcript.snippets)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                chunks = splitter.create_documents([full_text])

                vector_store = FAISS.from_documents(chunks, embeddings)
                st.session_state.retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                st.session_state.retriever_ready = True
                st.success("Transcript processed. Ask your questions!")

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.retriever_ready:
        retriever = st.session_state.retriever
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        parser = StrOutputParser()
        main_chain = parallel_chain | prompt | llm | parser

        question = st.text_input("Ask Any Question")
        submit=st.button("Submit")
        if submit:
            response = main_chain.invoke(question)
            st.write(response)


if __name__ == "__main__":
    main()