import os 
# load document
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from flask import Flask,render_template,url_for,redirect,request,jsonify  # Get the data of the  request internal data question to ask the langchain
app = Flask(__name__) # Get the langchain to working under the flask request function 
#read_key = open("key.txt",'r')
#key_api = read_key.read() 
os.environ["OPENAI_API_KEY"] = "your API key openai"

loader = PyPDFLoader("DRV8320.PDF")
documents = loader.load()
#print(documents)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)

embeddings = OpenAIEmbeddings()
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":2})
# create a chain to answer questions 
qa = RetrievalQA.from_chain_type(
llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
query = input_question # Get the input question from the input post request 
result = qa({"query": query})
print(result)
print(result['query'])
print(result['result'])

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
'''
@app.route("/request_chain",methods=['GET','POST'])
def requestchain_data():
        # select which embeddings we want to use
        res_q = request.get_json(force=True) 
        input_question = res_q.get("question") # Get the question input to operate the langchain operation  
        embeddings = OpenAIEmbeddings()
        # create the vectorestore to use as the index
        db = Chroma.from_documents(texts, embeddings)
        # expose this index in a retriever interface
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k":2})
        # create a chain to answer questions 
        qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
        query = input_question # Get the input question from the input post request 
        result = qa({"query": query}) 
        print(result)
        print(result['query'])
        print(result['result'])
        result_chainreq = {"Answer":result['result']} # Get the result output data from the request to output into the post request 
        return jsonify(result_chainreq) # Get the result chain reg 
if __name__ == "__main__":

            app.run(debug=True,threaded=True,host="0.0.0.0",port=5589) 
'''
