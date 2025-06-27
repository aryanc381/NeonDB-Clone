from IPython.display import Markdown, display

def RAG(file_path):
    from langchain_community.document_loaders import UnstructuredPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings.base import Embeddings
    from sklearn.feature_extraction.text import TfidfVectorizer
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    import warnings
    import requests
    from pydantic import BaseModel, Field
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from LLM.llm import SarvamChat  

    warnings.filterwarnings('ignore')

    # Set environment variables
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["LANGSMITH_API_KEY"] = "LANGSMITH_API_KEY"
    os.environ["LANGSMITH_TRACING"] = "true"

    # Load PDF
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    print("✅ PDF has been loaded")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    print(f"✅ The PDF is split into {len(chunks)} chunks.")
    texts = [doc.page_content for doc in chunks]

    # TF-IDF embedding
    vectorizer = TfidfVectorizer()
    _ = vectorizer.fit(texts)

    class TFIDFEmbeddings(Embeddings):
        def embed_documents(self, docs):
            return vectorizer.transform(docs).toarray()

        def embed_query(self, query):
            return vectorizer.transform([query]).toarray()[0]

    embedding_model = TFIDFEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    retriever = vectorstore.as_retriever()
    print("✅ Lightweight vector store created.")

    # Multi-query prompt
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2 different versions of the given user questions to retrieve relevant documents
        from the vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
        Provide these new alternative questions on a new line.
        Original question: {question}"""
    )

    print("✅ LLM has been initialized")
    llm = SarvamChat(api_key="a43b869d-bdd4-4257-9f93-2753ccc9736d")

    retriever = MultiQueryRetriever.from_llm(
        vectorstore.as_retriever(),
        llm=llm,
        prompt=QUERY_PROMPT
    )

    print("✅ RAG Infrastructure has been initialized")
    template = """Answer the questions based ONLY on the following context: 
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# RAG init
rag = RAG(r"C:\Users\conta\Desktop\DEV\Projects\Portfolio\research_paper_final.pdf")

# Q/A
def chat(question):
    return rag.invoke(question)

print()
print()
print(chat("What is the PDF about?"))
print()
print(chat("who are the authors of the paper?"))
print()
print(chat("tell me the accuracies of all models"))
print()
print()
print(chat("Summarize the entire paper in depth"))
