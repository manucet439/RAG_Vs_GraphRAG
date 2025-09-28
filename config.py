from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Neo4j Configuration
AURA_INSTANCENAME = os.environ.get("AURA_INSTANCENAME")
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# Initialize LLM
chat = ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-4o-mini")

# Data Configuration
SYNTHETIC_DATA_PATH = "synthetic_data.txt"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 24