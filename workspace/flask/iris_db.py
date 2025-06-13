

import os
import warnings
import ast
import sys
# from dotenv import load_dotenv
# from langchain.globals import set_verbose, set_debug
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI

import numpy as np
from sentence_transformers import SentenceTransformer

# from llama_index.core.query_engine import RetrieverQueryEngine

import iris  # Make sure you have this package installed in your container
# from llama_index.core.schema import Document
# from llama_index.core import VectorStoreIndex,PromptTemplate
# from llama_index.llms.openai import OpenAI
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.response_synthesizers import get_response_synthesizer
# from llama_index.core.settings import Settings
import diskcache
import hashlib
import logging
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)


# IRIS Database Credentials
hostname = "iris"  # Use the service name from docker-compose
port = 1972
namespace = "IRISAPP"
username = "_SYSTEM"
password = "SYS"

try:
    conn = iris.connect(f"{hostname}:{port}/{namespace}", username, password, sharedmemory=False)
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")

def create_fresh_irispy():
    conn = iris.connect(f"{hostname}:{port}/{namespace}", username, password, sharedmemory=False)
    return iris.createIRIS(conn)

def safe_iris_query(method, *args):
    for attempt in range(2):  # Max 1 or 2 tries
        try:
            irispy = create_fresh_irispy()
            return irispy.classMethodValue("GraphKB.Query", method, *args)
        except RuntimeError as e:
            msg = str(e)
            if ("Message out of order" in msg or "Connection closed" in msg) and attempt == 0:
                print("🔁 IRIS connection broken or idle. Retrying...")
                continue
            raise

def setup_environment():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ OPENAI_API_KEY is missing in your .env file or environment.")
    os.environ["OPENAI_API_KEY"] = key

irispy = iris.createIRIS(conn)

gpt4omini = "gpt-4o-mini"

model = gpt4omini

# docsfile = '/app/CSV/papers100.csv'
# relationsfile = '/app/CSV/relations100.csv'
# entitiesfile = '/app/CSV/entities100.csv'

docsfile = '/home/irisowner/dev/CSV/papers300.csv'
relationsfile = '/home/irisowner/dev/CSV/relations300.csv'
entitiesfile = '/home/irisowner/dev/CSV/entities300.csv'

# Load data
# irispy.classMethodValue("GraphKB.Documents","LoadData",docsfile)
# irispy.classMethodValue("GraphKB.Entity","LoadData",entitiesfile)
# irispy.classMethodValue("GraphKB.Relations","LoadData",relationsfile)


entitiesembeddingsfile = '/home/irisowner/dev/CSV/entities_embeddings300.csv'
papersembeddingsfile = '/home/irisowner/dev/CSV/papers_embeddings300.csv'

# irispy.classMethodValue("GraphKB.DocumentsEmbeddings","LoadData",papersembeddingsfile)
# irispy.classMethodValue("GraphKB.EntityEmbeddings","LoadData",entitiesembeddingsfile)

#use only when new data need to be added
# irispy = iris.createIRIS(conn)

def load_graph_data():
    print("🔁 Loading GraphKB data...")

    irispy.classMethodValue("GraphKB.Documents", "LoadData", docsfile)
    irispy.classMethodValue("GraphKB.Entity", "LoadData", entitiesfile)
    irispy.classMethodValue("GraphKB.Relations", "LoadData", relationsfile)
    irispy.classMethodValue("GraphKB.DocumentsEmbeddings", "LoadData", papersembeddingsfile)
    irispy.classMethodValue("GraphKB.EntityEmbeddings", "LoadData", entitiesembeddingsfile)


# entitiesembeddingsfile = '/home/irisowner/dev/CSV/entities_embeddings.csv'
# papersembeddingsfile = '/home/irisowner/dev/workspace/CSV/papers_embeddings.csv'
# docsfile = '/home/irisowner/dev/CSV/papers100.csv'
# relationsfile = '/home/irisowner/dev/CSV/relations100.csv'
# entitiesfile = '/home/irisowner/dev/CSV/entities100.csv'


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_embeddings(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode([text])[0]
    rounded = np.round(embeddings, 7).tolist()
    return str(rounded)

def ask_query_no_rag(query, cutoff=True):
    # Directly prompt the LLM with no documents
    prompt_text = """You are an expert assistant.
    Answer the following question concisely and clearly.
    """ + (("Use three sentences maximum and keep the answer concise.") if cutoff else "") + """
    Question: {question}
    Answer:"""

    prompt = prompt_text.format(question=query)

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = send_to_llm(model, messages)
    return [line.strip() for line in response.choices[0].message.content.split("\n") if line.strip()]


def ask_query_graphrag(query, graphitems=50,vectoritems=0, method='local'):

    user_query_entity = get_embeddings(query)
    user_query_embeddings = get_embeddings(query)
    with HiddenPrints():
    #   docs = [irispy.classMethodValue("GraphKB.Query","Search",user_query_entity,user_query_embeddings,graphitems,vectoritems)]
        docs = [safe_iris_query("Search", user_query_entity, user_query_embeddings, graphitems, vectoritems)]

    response = llm_answer_graphrag(docs, query,True)
    return response

def ask_query_rag(query, graphitems=0,vectoritems=50, method='local'):

    user_query_entity = get_embeddings(query)
    user_query_embeddings = get_embeddings(query)
    with HiddenPrints():
    #   docs = [irispy.classMethodValue("GraphKB.Query","Search",user_query_entity,user_query_embeddings,graphitems,vectoritems)]
        docs = [safe_iris_query("Search", user_query_entity, user_query_embeddings, graphitems, vectoritems)]

    response = llm_answer_rag(docs, query,False)
    return response


def ask_query_ragvsgraphrag(query, graphitems=0,vectoritems=50, method='local'):

    user_query_entity = get_embeddings(query)
    user_query_embeddings = get_embeddings(query)
    with HiddenPrints():
    #   docs = [irispy.classMethodValue("GraphKB.Query","Search",user_query_entity,user_query_embeddings,graphitems,vectoritems)]
        docs = [safe_iris_query("Search", user_query_entity, user_query_embeddings, graphitems, vectoritems)]

    response = llm_answer_ragvsgraghrag(docs, query,False)
    return response




def send_to_llm(model, messages):
    client = OpenAI()
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return completion


cache = diskcache.Cache('./llm_cache')


def llm_answer_graphrag(batch, query, cutoff=True):
 
        # Step 2: Create a unique cache key based on query and batch
    hash_input = f"{query}|{batch}|{cutoff}".encode("utf-8")
    cache_key = hashlib.sha256(hash_input).hexdigest()

    # Step 3: Return from cache if available
    if cache_key in cache:
        return cache[cache_key]
    

    prompt_text = """You are an expert assistant for graph-based academic search. 
    You are given a graph context of academic papers including authors, abstracts, and related information.
    Use the following pieces of retrieved context from a graph database to answer the question.
    """ + (("Use three sentences maximum and keep the answer concise,do not use any numbered list or bullet points.") if cutoff else " ") + """
    Question: {question}  
    Graph Context: {graph_context}
    Answer:
    """


    prompt = prompt_text.format(**{"question": query, "graph_context": batch})
 
    messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    completion = send_to_llm(model, messages)
    response = completion.choices[0].message.content
 
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]


    cache[cache_key] = answer_lines

    return answer_lines





def llm_answer_rag(batch, query, cutoff=True):
 
        # Step 2: Create a unique cache key based on query and batch
    hash_input = f"{query}|{batch}|{cutoff}".encode("utf-8")
    cache_key = hashlib.sha256(hash_input).hexdigest()

    # Step 3: Return from cache if available
    if cache_key in cache:
        return cache[cache_key]
    

    prompt_text = """You are an assistant for academic paper search.
    Use the following pieces of retrieved context to answer the question.  
    """ + (("Use three sentences maximum and keep the answer concise.") if cutoff else " ") + """
    Question: {question}  
    Graph Context: {context}
    Answer:
    """


    prompt = prompt_text.format(**{"question": query, "context": batch})
 
    messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    completion = send_to_llm(model, messages)
    response = completion.choices[0].message.content
 
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]


    cache[cache_key] = answer_lines

    return answer_lines


def llm_answer_ragvsgraghrag(batch, query, cutoff=True):
 
    hash_input = f"{query}|{batch}|{cutoff}".encode("utf-8")
    cache_key = hashlib.sha256(hash_input).hexdigest()

    # Step 3: Return from cache if available
    if cache_key in cache:
        return cache[cache_key]
    

    prompt_text = """You are an assistant for academic paper search.
    Say I don't know if you don't know.  
    """ + (("Use three sentences maximum and keep the answer concise.") if cutoff else " ") + """
    Question: {question}  
    Graph Context: {context}
    Answer:
    """


    prompt = prompt_text.format(**{"question": query, "context": batch})
 
    messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    
    completion = send_to_llm(model, messages)
    response = completion.choices[0].message.content
 
    answer_lines = [line.strip() for line in response.split('\n') if line.strip()]


    cache[cache_key] = answer_lines

    return answer_lines

    
def ask_query_graphrag_with_docs(query, graphitems=100, vectoritems=0):
    user_query_entity = get_embeddings(query)
    user_query_embeddings = get_embeddings(query)

    with HiddenPrints():
        docs = [safe_iris_query("Search", user_query_entity, user_query_embeddings, graphitems, vectoritems)]

    logger.info("📄 Retrieved %d document(s)", len(docs))

    return docs[0].split("\n\r\n")