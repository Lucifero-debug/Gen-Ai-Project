from flask import Flask, request, render_template,jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os,tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=700,
        repetition_penalty=1.2,   # discourages looping
        do_sample=True 
    )
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
 You are a career expert. Analyze the following resume based on given resume text and suggest areas of strength weaknesss and job roles to pursue:

Resume:
{full_text}
    """,
    input_variables = ['full_text']
)

parser = StrOutputParser()

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


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/submit", methods=['POST', 'GET'])
def analyse():
    if request.method=="POST":
        try:
            pdf=request.files['pdf']
            print(pdf)
            if not pdf or pdf.filename == '':
                return render_template('index.html', response="Error: No file selected")
            temp_file_path = None
            try:
                # Save PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    pdf.save(temp_file.name)
                    temp_file_path = temp_file.name
                
                # Load PDF with PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load()
                if not documents:
                    return render_template('index.html', response="Error: Could not extract text from PDF")
                full_text = "\n\n".join([doc.page_content for doc in documents])
                main_chain =  prompt | model | parser
                result=main_chain.invoke(full_text)
                response=clean_response(result)
                return render_template('index.html',response=response)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return render_template('index.html', result=f"Error: {str(e)}")
 

if __name__ == "__main__":
    app.run(debug=True)