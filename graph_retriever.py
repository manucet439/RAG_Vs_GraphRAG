from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from pydantic import BaseModel, Field
from typing import List
from config import chat

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, business entities, and job titles/roles that appear in the text",
    )

class GraphRetriever:
    def __init__(self, kg, vector_index):
        """Initialize with Neo4j graph and vector index"""
        self.kg = kg
        self.vector_index = vector_index
        
        # Setup entity extraction chain
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting organization, person entities, and job titles/roles from the text. Include specific names, company names, and roles like CEO, CFO, CTO, etc.",
            ),
            (
                "human",
                "Use the given format to extract information from the following input: {question}",
            ),
        ])
        self.entity_chain = prompt | chat.with_structured_output(Entities)
    
    def generate_full_text_query(self, input_text: str) -> str:
        """Generate full-text search query for Neo4j"""
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input_text).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()
    
    def find_role_mentions_in_context(self, question: str) -> str:
        """Dynamically find role-to-person mappings from document context"""
        # Common role patterns to look for
        role_patterns = [
            'CEO', 'CFO', 'CTO', 'CIO', 'COO',
            'Chief Executive Officer', 'Chief Financial Officer', 
            'Chief Technology Officer', 'Chief Innovation Officer', 
            'Chief Operating Officer', 'President', 'Director',
            'Head of', 'founder', 'founded'
        ]
        
        role_context = ""
        
        # Search for documents that mention roles
        for pattern in role_patterns:
            if pattern.lower() in question.lower():
                # Find documents that mention this role
                docs = self.vector_index.similarity_search(f"{pattern} person name who", k=3)
                for doc in docs:
                    # Look for sentences that connect roles to people
                    sentences = doc.page_content.split('.')
                    for sentence in sentences:
                        if pattern.lower() in sentence.lower():
                            # Check if sentence contains both role and a person name
                            words = sentence.split()
                            # Look for capitalized words that might be names
                            potential_names = [w for w in words if w[0].isupper() and len(w) > 2 and w not in ['The', 'In', 'As', 'By']]
                            if len(potential_names) >= 2:  # Likely contains names
                                role_context += f"Role context: {sentence.strip()}\n"
        
        return role_context
    
    def structured_retriever(self, question: str) -> str:
        """Retrieve structured data using entity extraction and graph traversal"""
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        
        for entity in entities.names:
            print(f"Getting Entity: {entity}")
            
            # Enhanced query to find more relationship types
            response = self.kg.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:3})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 100
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            
            entity_relationships = "\n".join([el["output"] for el in response])
            result += entity_relationships + "\n"
        
        # Add dynamic role context discovery
        role_context = self.find_role_mentions_in_context(question)
        if role_context:
            result += "\nRole-to-person context from documents:\n" + role_context
            
        return result
    
    def retrieve(self, question: str) -> str:
        """Main retrieval method combining structured and unstructured data"""
        print(f"Graph Search query: {question}")
        
        # Get structured data (graph relationships)
        structured_data = self.structured_retriever(question)
        
        # Get unstructured data (vector similarity) - prioritize role-related content
        base_docs = self.vector_index.similarity_search(question, k=4)
        
        # If question seems role-related, get additional role-focused documents
        role_keywords = ['approved', 'founded', 'ceo', 'cfo', 'cto', 'cio', 'chief', 'head', 'director']
        if any(keyword in question.lower() for keyword in role_keywords):
            role_docs = self.vector_index.similarity_search(f"role position {question}", k=2)
            combined_docs = base_docs + role_docs
            # Remove duplicates
            seen_content = set()
            unique_docs = []
            for doc in combined_docs:
                if doc.page_content not in seen_content:
                    unique_docs.append(doc)
                    seen_content.add(doc.page_content)
            unstructured_data = [doc.page_content for doc in unique_docs[:6]]
        else:
            unstructured_data = [doc.page_content for doc in base_docs]
        
        # Combine both types of data with emphasis on role resolution
        final_data = f"""Structured data (Graph relationships):
{structured_data}

Unstructured data (Document chunks):
{"#Document ". join(unstructured_data)}

Instructions for role resolution:
- When you see a role mentioned (like CFO, CTO, etc.), look through ALL the context to find who holds that role
- Connect actions performed by roles to the specific people who hold those roles
- If someone "approved" something and they're described by a role, identify the person's name
"""
        
        print(f"\nGraph Retrieval Result:")
        newline_char = '\n'
        print(f"Structured relationships found: {len(structured_data.split(newline_char)) if structured_data else 0}")
        print(f"Document chunks found: {len(unstructured_data)}")
        
        return final_data
    
    def get_entity_relationships(self, entity_name: str) -> List[dict]:
        """Get all relationships for a specific entity"""
        query = """
        CALL db.index.fulltext.queryNodes('entity', $entity_query, {limit:5})
        YIELD node, score
        CALL {
          WITH node
          MATCH (node)-[r]->(neighbor)
          RETURN node.id as source, type(r) as relationship, neighbor.id as target, 'outgoing' as direction
          UNION ALL
          WITH node
          MATCH (node)<-[r]-(neighbor) 
          RETURN neighbor.id as source, type(r) as relationship, node.id as target, 'incoming' as direction
        }
        RETURN source, relationship, target, direction
        LIMIT 20
        """
        
        result = self.kg.query(query, {
            "entity_query": self.generate_full_text_query(entity_name)
        })
        
        return result