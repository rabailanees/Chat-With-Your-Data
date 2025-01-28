from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.docstore import InMemoryDocstore
from langchain.memory import ConversationBufferMemory

# Initialize FastAPI
app = FastAPI()

# Define the request model
class QueryRequest(BaseModel):
    query: str

# Load the FAISS index and metadata
faiss_index_file = "faiss_index.index"
embeddings_file = "embeddings.pkl"

try:
    # Load FAISS index
    faiss_index = faiss.read_index(faiss_index_file)

    # Load metadata
    with open(embeddings_file, "rb") as f:
        texts = pickle.load(f)

    # Reinitialize FAISS vector store
    docstore = InMemoryDocstore({str(i): texts[i] for i in range(len(texts))})
    index_to_docstore_id = {i: str(i) for i in range(len(texts))}

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS(
        index=faiss_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embeddings.embed_query,
    )
except Exception as e:
    print(f"Error initializing FAISS: {e}")
    raise

# Initialize the LLM
groq_api_key = "YOUR_GROQ_API_KEY"
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.3,
    api_key=groq_api_key,
)

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    memory=memory,
    retriever=vectordb.as_retriever(),
    return_source_documents=False,
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
)

# Function to process query and return result only
def process_qa_retrieval_chain(chain, query):
    response = chain.invoke({'query': query})
    return response["result"]

# Define the API endpoint
@app.post("/query")
def query_chatbot(request: QueryRequest):
    try:
        result = process_qa_retrieval_chain(qa_chain, request.query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")