from flask import Flask, request, render_template,jsonify
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)
load_dotenv()

# Check for required environment variable
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Use a better model for Q&A tasks
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

def format_docs(retrieved_docs):
    """Format retrieved documents into a single context string."""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

# Improved prompt template
prompt = PromptTemplate(
    template="""You are a helpful assistant that answers questions based on the provided context.

Context: {context}

Question: {question}

Answer: Based on the provided context, """,
    input_variables=['context', 'question']
)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/submit", methods=['POST', 'GET'])
def summary():
    if request.method == 'POST':
        try:
            pdf = request.files['pdf']
            query = request.form['query']
            # Validation
            if not pdf or pdf.filename == '':
                return render_template('index.html', result="Error: No file selected")
            
            if not query:
                return render_template('index.html', result="Error: No query provided")
            
            if not pdf.filename.lower().endswith('.pdf'):
                return render_template('index.html', result="Error: Please upload a PDF file")
            
            # Process PDF
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
                    return render_template('index.html', result="Error: Could not extract text from PDF")
                
                # Prepare text for processing
                full_text = "\n\n".join([doc.page_content for doc in documents])
                
                if not full_text.strip():
                    return render_template('index.html', result="Error: PDF appears to be empty or contains no readable text")
                
                # Create metadata
                full_metadata = documents[0].metadata.copy() if documents else {}
                
                # Split text into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = splitter.create_documents([full_text], metadatas=[full_metadata])
                # Create vector store
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 4}
                )
                # Create the processing chain
                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                
                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | model | parser
                
                # Get answer
                answer = main_chain.invoke(query)
                print(answer)
                
                # Clean up the answer
                if isinstance(answer, str):
                    answer = answer.strip()
                    if not answer:
                        answer = "I couldn't find relevant information in the provided PDF to answer your question."
                
                return jsonify({"query": query, "answer": answer})

                
            finally:
                # Clean up temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except OSError:
                        pass  # File might already be deleted
                        
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return render_template('index.html', result=f"Error: {str(e)}")
    
    # Handle GET request
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)