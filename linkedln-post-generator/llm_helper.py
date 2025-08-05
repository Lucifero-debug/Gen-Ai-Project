from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
llm=ChatGroq(model='llama3-70b-8192')


if __name__=="__main__":
    response=llm.invoke("what is multiverse")
    print(response.content)