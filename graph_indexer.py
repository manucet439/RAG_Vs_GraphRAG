from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Better for structured text
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from config import *

class GraphIndexer:
    def __init__(self):
        """Initialize Neo4j graph connection and components"""
        self.kg = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )
        self.llm_transformer = LLMGraphTransformer(llm=chat)
        self.vector_index = None
        
    def load_and_split_documents(self, file_path: str):
        """Load documents from file and split into chunks"""
        print(f"Loading documents from {file_path}")
        
        # Load documents
        raw_documents = TextLoader(file_path, encoding="latin-1").load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE * 4,  # Characters instead of tokens
            chunk_overlap=CHUNK_OVERLAP * 2,
            separators=["\n________________________________________\n", "\n\n", "\n", ". ", " "]
        )
        documents = text_splitter.split_documents(raw_documents)
        
        print(f"Loaded {len(documents)} chunks from synthetic data.")
        return documents
    
    def create_graph_index(self, documents):
        """Transform documents to graph and store in Neo4j"""
        print("Creating graph index...")
        
        # Transform documents to graph
        graph_documents = self.llm_transformer.convert_to_graph_documents(documents)
        
        # Store to Neo4j as graph
        res = self.kg.add_graph_documents(
            graph_documents,
            include_source=True,
            baseEntityLabel=True,
        )
        
        # Create vector index for hybrid search
        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        
        # Create full-text index for entities
        self.kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        
        print("Graph index created successfully!")
        return res
    
    def build_index(self, file_path: str = SYNTHETIC_DATA_PATH):
        """Complete indexing pipeline"""
        documents = self.load_and_split_documents(file_path)
        return self.create_graph_index(documents)
    
    def get_graph_stats(self):
        """Get basic statistics about the created graph"""
        node_count = self.kg.query("MATCH (n) RETURN count(n) as node_count")[0]["node_count"]
        relationship_count = self.kg.query("MATCH ()-[r]->() RETURN count(r) as rel_count")[0]["rel_count"]
        
        print(f"Graph Statistics:")
        print(f"- Nodes: {node_count}")
        print(f"- Relationships: {relationship_count}")
        
        return {"nodes": node_count, "relationships": relationship_count}