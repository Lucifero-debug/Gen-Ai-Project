from flask import Flask, request, render_template,jsonify
from transformers import pipeline
from transformers import AutoProcessor, AutoModelForImageTextToText
import requests
from dotenv import load_dotenv
from PIL import Image
import tempfile
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.output_parsers import StrOutputParser
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace,HuggingFaceEndpoint,HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import warnings
warnings.filterwarnings("ignore")



app = Flask(__name__)
load_dotenv()
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
from huggingface_hub import login

pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
caption_model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

import re
from langchain_core.messages import AIMessage

def clean_response(response) -> str:
    # Convert AIMessage to plain text if needed
    if isinstance(response, AIMessage):
        text = response.content
    else:
        text = str(response)

    # Now extract only the part after <|assistant|>
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>", 1)[-1]

    text = text.strip()
    text = re.sub(r"</s>$", "", text).strip()

    return text



parser=StrOutputParser()

import time

def fetch_gutenberg_text(book_id, retries=3, delay=3):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.encoding = "utf-8"
            if response.status_code == 200:
                text = response.text
                start = text.find("*** START")
                end = text.find("*** END")
                if start != -1 and end != -1:
                    text = text[start:end]
                return text
            else:
                raise Exception(f"Status code: {response.status_code}")
        except Exception as e:
            print(f"Attempt {attempt+1} failed for book {book_id}: {e}")
            time.sleep(delay)
    raise Exception(f"Failed to fetch book {book_id} after {retries} attempts")


def split_into_chunks(text, chunk_size=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

book_ids = {
    "Grimm's Fairy Tales": 2591,
    "The Jungle Book": 236,
    "Peter Pan":16,
    "Aesop’s Fable":11339,
    "Alice’s Adventures in Wonderland": 11,
    "Through the Looking-Glass": 12,
    "Treasure Island": 120,
    "The Adventures of Sherlock Holmes": 1661,
    "Dracula": 345,
    "Frankenstein": 84,
    "The Secret Garden": 113,
}

all_docs = []

def load_background_knowledge():
    all_docs = []
    for title, book_id in book_ids.items():
        try:
            print(f"Fetching {title}...")
            text = fetch_gutenberg_text(book_id)
            chunks = split_into_chunks(text)
            for doc in chunks:
                doc.metadata = {"source": title}
            all_docs.extend(chunks)
            print(f"✓ Loaded {len(chunks)} chunks from {title}")
        except Exception as e:
            print(f"✗ Error with {title}: {e}")
    return all_docs




embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=700,
        repetition_penalty=1.2,   # discourages looping
        do_sample=True 
    )
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=['knowledge', 'topic'],
    template="""
You are a highly creative and imaginative story writer.

Use the background knowledge **as inspiration**, not restriction. Combine it with your own creativity to write a detailed story of at least **Tens paragraphs** based on the given topic.

Your story must be complete, with a clear beginning, middle, and end — and it must fit within **700 tokens**. Do not exceed the length limit.
You are free to invent details, characters, and settings. The story should be immersive, like the reader is living it.

Background Knowledge (optional inspiration):
{knowledge}

Image Topic:
{topic}

Write the story below:
    """
)


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/submit", methods=['POST', 'GET'])
def generate():
    cleaned_output = None
    image_path=None
    if request.method=='POST':
        try:
            image=request.files['image']
            temp_file_path = None
            try:        
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                            image.save(temp_file.name)
                            raw_image = Image.open(temp_file.name).convert('RGB')
                            inputs = processor(raw_image, return_tensors="pt")
                            out = caption_model.generate(**inputs)
                            topic=processor.decode(out[0], skip_special_tokens=True)
                            vector_store = FAISS.from_documents(all_docs, embeddings)
                            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                            parallel_chain = RunnableParallel({
                                'knowledge': retriever | RunnableLambda(format_docs),
                                'topic': RunnablePassthrough()
                                })
                            main_chain = parallel_chain | prompt | model 
                            result=main_chain.invoke(topic)
                            cleaned_output = clean_response(result.content)
                            print(cleaned_output)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass 
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return render_template('index.html', result=f"Error: {str(e)}")
    return render_template('index.html',story=cleaned_output,image_url='/' + image_path if image_path else None)

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        all_docs = load_background_knowledge()
    else:
        all_docs = [] 
    app.run(debug=True)