from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Tuple, List
from config import chat

class RAGChains:
    def __init__(self):
        """Initialize RAG chains for both Graph and FAISS"""
        self.setup_chains()
    
    def setup_chains(self):
        """Setup the common RAG chain components"""
        # Template for rephrasing questions with chat history
        condense_template = """Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question, in its original language.
        
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        
        self.condense_question_prompt = PromptTemplate.from_template(condense_template)
        
        # Template for answering questions based on context
        answer_template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        Use natural language and be concise.
        Answer:"""
        
        self.answer_prompt = ChatPromptTemplate.from_template(answer_template)
        
    def _format_chat_history(self, chat_history: List[Tuple[str, str]]) -> List:
        """Format chat history for the chain"""
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer
    
    def create_graph_chain(self, graph_retriever):
        """Create RAG chain for Graph RAG"""
        # Search query processing with optional chat history
        search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | self.condense_question_prompt
                | chat
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )
        
        # Complete chain
        chain = (
            RunnableParallel({
                "context": search_query | graph_retriever.retrieve,
                "question": RunnablePassthrough(),
            })
            | self.answer_prompt
            | chat
            | StrOutputParser()
        )
        
        return chain
    
    def create_faiss_chain(self, faiss_retriever):
        """Create RAG chain for FAISS RAG"""
        # Search query processing with optional chat history
        search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: self._format_chat_history(x["chat_history"])
                )
                | self.condense_question_prompt
                | chat
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )
        
        # Complete chain
        chain = (
            RunnableParallel({
                "context": search_query | faiss_retriever.retrieve,
                "question": RunnablePassthrough(),
            })
            | self.answer_prompt
            | chat
            | StrOutputParser()
        )
        
        return chain
    
    def query_graph_rag(self, chain, question: str, chat_history: List[Tuple[str, str]] = None):
        """Query using Graph RAG chain"""
        print(f"\n{'='*50}")
        print("GRAPH RAG QUERY")
        print(f"{'='*50}")
        
        query_input = {"question": question}
        if chat_history:
            query_input["chat_history"] = chat_history
        
        result = chain.invoke(query_input)
        
        print(f"Question: {question}")
        print(f"Answer: {result}")
        print(f"{'='*50}\n")
        
        return result
    
    def query_faiss_rag(self, chain, question: str, chat_history: List[Tuple[str, str]] = None):
        """Query using FAISS RAG chain"""
        print(f"\n{'='*50}")
        print("FAISS RAG QUERY")
        print(f"{'='*50}")
        
        query_input = {"question": question}
        if chat_history:
            query_input["chat_history"] = chat_history
        
        result = chain.invoke(query_input)
        
        print(f"Question: {question}")
        print(f"Answer: {result}")
        print(f"{'='*50}\n")
        
        return result
    
    def compare_rag_methods(self, graph_chain, faiss_chain, question: str, chat_history: List[Tuple[str, str]] = None):
        """Compare both RAG methods side by side"""
        print(f"\n{'='*60}")
        print("RAG COMPARISON")
        print(f"{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        # Query FAISS RAG
        faiss_result = self.query_faiss_rag(faiss_chain, question, chat_history)
        
        # Query Graph RAG
        graph_result = self.query_graph_rag(graph_chain, question, chat_history)
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"FAISS RAG Answer:\n{faiss_result}")
        print(f"\nGraph RAG Answer:\n{graph_result}")
        print(f"{'='*60}\n")
        
        return {
            "question": question,
            "faiss_result": faiss_result,
            "graph_result": graph_result
        }