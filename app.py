from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import pinecone
from langchain_pinecone import Pinecone  # Import from the new package

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Pinecone API credentials
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone with the new class-based approach
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

index_name = "medical-chatbot"

# Loading the existing Pinecone index directly
index = pc.Index(index_name)

# Assuming the `text_key` is the key used to store text in Pinecone, you should replace 'your_text_key' with the actual text key used in your Pinecone index.
docsearch = Pinecone(index, embeddings, text_key="your_text_key")  # Pass the text_key argument

# Define the prompt template for the LLM
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Chain type kwargs
chain_type_kwargs = {"prompt": PROMPT}

# Initialize the LLM
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

# Initialize the RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Index route
@app.route("/")
def index():
    return render_template('chat.html')


# Chat route to handle POST requests
@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        msg = request.form["msg"]  # Get message from the form (if using a form-based request)
        # Or if you're sending data as JSON, use:
        # msg = request.json.get("msg")  

        input = msg
        print(input)
        result = qa({"query": input})
        print("Response : ", result["result"])
        return str(result["result"])

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
