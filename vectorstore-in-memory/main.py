import os
from dotenv import  load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import  RetrievalQA
from langchain.llms import OpenAI

load_dotenv()
env_variables = dict(os.environ)

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "attention_is_all_you_need.pdf")
    loader = PyPDFLoader(file_path=path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    llm = OpenAI()
    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_documents(docs, embeddings)

    # vectorstore.save_local("faiss_index_react")

    vectorstore = FAISS.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = vectorstore.as_retriever())

    res = qa.run("Explain the difference between a encoder and a decoder")

    print(res)