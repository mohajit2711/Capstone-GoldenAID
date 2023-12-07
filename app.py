import os
import re
import atexit  
import requests
from werkzeug.utils import secure_filename
# from awsS3 import upload_file, downloadfile
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma,FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from flask import Flask, render_template, request, jsonify
from config.loader import config_data
from werkzeug.exceptions import HTTPException, BadRequest, Unauthorized, Forbidden, NotFound, InternalServerError
import sys
app = Flask(__name__)

load_dotenv()
# set config variables
app.config['UPLOAD_DIR'] = config_data["UPLOAD_DIR"]
app.config['ALLOWED_EXT'] = config_data["ALLOWED_EXT"]
app.config['DOC_SET'] = config_data["DOC_SET"]
app.config['LOCAL_DIRECTORY'] = config_data["LOCAL_DIR"]


pattern = re.compile(r'\bI don\'t know\b', re.IGNORECASE)
# global chat_history
count = 0
# Endpoint to handle the response with "I don't know" in the statement
@app.route('/handle_idk_response', methods=['POST'])
def handle_idk_response():
    try:
        data = request.get_json()
        if 'answer' in data and pattern.search(data['answer']):
            return "I don't know. My knowledge is limited to the criteria set here", 200
        # If the answer doesn't match, allow the code to continue
        # ...
        # Additional processing for non-matching answers here
        return 'Response not matched', 400
    except Exception as e:
        return f'Error handling response: {str(e)}', 500

def setup_bot(filename = 'uploads\GoldenAidKB.pdf'):
    # load pdf
    global vector_db
    pdf_loader = PyPDFLoader(filename)
    pdf_data = pdf_loader.load()

    # split data
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_data = text_splitter.split_documents(pdf_data)

    # load var
    collection_name = app.config['DOC_SET']
    local_directory = config_data["LOCAL_DIR"]
    persist_directory = os.path.join(os.getcwd(), local_directory)

    # create embeddings and vector db
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_documents(split_data,
                                    embeddings,
                                    collection_name=collection_name,
                                    persist_directory=persist_directory
                                    )
    vector_db.persist()

    # Q&A 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_bot = ConversationalRetrievalChain.from_llm(
                OpenAI(temperature=0, model_name="gpt-3.5-turbo"), 
                vector_db.as_retriever())
    return qa_bot
qa_bot = setup_bot()
chat_history = []

upload_folder = "uploads/"
if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

app.config['UPLOAD_FOLDER'] = upload_folder

# @app.route('/upload_pdf', methods=['POST'])
# def upload_pdf():
#     # pdf = request.files
#     # print(pdf)
#     global qa_bot, chat_history
#     file_format = app.config['ALLOWED_EXT']
#     if file_format not in request.files:
#         return 'No file part'
#     file = request.files[file_format]
#     if file.filename == '':
#         return 'No selected file'
#     i=0    
#     def allowed_file(filename, allowed_ext):
#         return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext
    
#     if file and allowed_file(file.filename, {'pdf'}):
#         filename = os.path.join(app.config['UPLOAD_DIR'], file.filename)
#         #saving file to system which is uploaded
#         file.save(filename)
#         qa_bot = setup_bot(filename)
#         chat_history = []  # reset chat history if a new document is uploaded
#         return 'PDF uploaded and processed!'
#     else:
#         return 'Invalid file type. Only PDFs are allowed.'


@app.route('/example', methods=['GET', 'POST'])
def example():
    # Simulate different scenarios by raising exceptions
    if request.method == 'POST':
        data = request.get_json()
        if 'key' not in data:
            raise BadRequest()
        return 'Resource created', 201
    elif request.method == 'GET':
        data = request.get_json()
        if 'authorized' not in data:
            raise Unauthorized()
        elif 'allowed' not in data:
            raise Forbidden()
        # Check if the requested resource exists, raise NotFound if not found
        raise NotFound()
    else:
        # Simulate an unexpected server error
        raise InternalServerError()

# Cleanup ChromaDB data
# def cleanup_chroma_data():
#     try:
#         if 'vector_db' in globals():
#             vector_db.delete_all_documents()
#     except Exception as e:
#         print(f"Error cleaning up Chroma data: {str(e)}")

# atexit.register(cleanup_chroma_data)
@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/get_response', methods=['GET','POST'])    
def get_response():
    global chat_history
    global count
    message = request.json['message']
    message1 = "Please answer politely and strictly in english only."
    message3 = message +" "+ message1
    response = qa_bot({"question": message3, "chat_history": chat_history})
    print("Hi")
    print(message)
    chat_history.append({"user": message, "bot": response["answer"]})
    ans = {
        "answer": response["answer"]
        }
    ans1 = request.args.get('ans')
    print(ans1)
    return jsonify(ans)

if __name__ == '__main__':
    # Create the upload directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_DIR']):
        os.makedirs(app.config['UPLOAD_DIR'])
    app.run(host='0.0.0.0', port=8080, debug=True)


