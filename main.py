#!/usr/bin/env python3
"""
Graph RAG vs Traditional RAG Comparison Script

This script demonstrates the differences between traditional RAG (using FAISS) 
and Graph RAG (using Neo4j) for question answering.

Usage:
    python main.py --mode [faiss|graph|compare|interactive]
"""

import argparse
import time
import os
from config import SYNTHETIC_DATA_PATH, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE
from langchain_neo4j import Neo4jGraph
from faiss_indexer import FAISSIndexer  # Still needed for load_index()
from graph_retriever import GraphRetriever
from faiss_retriever import FAISSRetriever
from rag_chains import RAGChains
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings

class RAGComparison:
    def __init__(self):
        """Initialize the RAG comparison system"""
        self.graph_indexer = None
        self.faiss_indexer = None
        self.graph_retriever = None
        self.faiss_retriever = None
        self.rag_chains = RAGChains()
        
        # Test questions designed to show differences between approaches
        self.test_questions = [
            "Name Who approved the acquisition of SolarOptima?",
            "What is the relationship between Sophia Martinez and Aurora Dynamics?",
            "Tell me about the partnership between Aurora Dynamics and HelioSoft Technologies.",
            "Who founded SolarOptima and what was their previous company name?",
            "What role did Priya Nair play in the acquisition?",
            "How are Amelia Green, NorthBridge Capital, and Aurelia Corp connected?"
        ]
    
    def setup_graph_rag(self):
        """Setup Graph RAG system (assumes index is already built)"""
        print("Setting up Graph RAG system...")
        
        # Check if graph index exists
        if not os.path.exists(".graph_index_built"):
            print("❌ Graph index not found!")
            print("🔧 Please run: python build_indices.py")
            raise FileNotFoundError("Graph index not built. Run build_indices.py first.")
        
        # Create Neo4j connection directly (no need for GraphIndexer)
        kg = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )
        
        # Connect to existing vector index
        vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        
        self.graph_retriever = GraphRetriever(kg, vector_index)
        
        print("✅ Graph RAG setup complete!")
    
    def setup_faiss_rag(self):
        """Setup FAISS RAG system (assumes index is already built)"""
        print("Setting up FAISS RAG system...")
        
        # Check if FAISS index exists
        if not os.path.exists("faiss_index"):
            print("❌ FAISS index not found!")
            print("🔧 Please run: python build_indices.py")
            raise FileNotFoundError("FAISS index not built. Run build_indices.py first.")
        
        self.faiss_indexer = FAISSIndexer()
        # Load existing index instead of rebuilding
        self.faiss_indexer.load_index("faiss_index")
        
        self.faiss_retriever = FAISSRetriever(self.faiss_indexer.vector_store)
        
        print("✅ FAISS RAG setup complete!")
    
    def run_faiss_only(self):
        """Run queries using only FAISS RAG"""
        print("\n" + "="*60)
        print("RUNNING FAISS RAG ONLY")
        print("="*60)
        
        self.setup_faiss_rag()
        faiss_chain = self.rag_chains.create_faiss_chain(self.faiss_retriever)
        
        results = []
        for question in self.test_questions:
            result = self.rag_chains.query_faiss_rag(faiss_chain, question)
            results.append({"question": question, "answer": result})
            time.sleep(1)  # Brief pause between queries
        
        return results
    
    def run_graph_only(self):
        """Run queries using only Graph RAG"""
        print("\n" + "="*60)
        print("RUNNING GRAPH RAG ONLY")
        print("="*60)
        
        self.setup_graph_rag()
        graph_chain = self.rag_chains.create_graph_chain(self.graph_retriever)
        
        results = []
        for question in self.test_questions:
            result = self.rag_chains.query_graph_rag(graph_chain, question)
            results.append({"question": question, "answer": result})
            time.sleep(1)  # Brief pause between queries
        
        return results
    
    def run_comparison(self):
        """Run side-by-side comparison"""
        print("\n" + "="*60)
        print("RUNNING SIDE-BY-SIDE COMPARISON")
        print("="*60)
        
        # Setup both systems
        self.setup_faiss_rag()
        self.setup_graph_rag()
        
        # Create chains
        faiss_chain = self.rag_chains.create_faiss_chain(self.faiss_retriever)
        graph_chain = self.rag_chains.create_graph_chain(self.graph_retriever)
        
        # Run comparisons
        comparison_results = []
        for question in self.test_questions:
            result = self.rag_chains.compare_rag_methods(
                graph_chain, faiss_chain, question
            )
            comparison_results.append(result)
            time.sleep(2)  # Brief pause between comparisons
        
        return comparison_results
    
    def run_interactive_mode(self):
        """Interactive mode for custom queries"""
        print("\n" + "="*60)
        print("INTERACTIVE RAG COMPARISON MODE")
        print("="*60)
        print("Setup both systems...")
        
        # Setup both systems
        self.setup_faiss_rag()
        self.setup_graph_rag()
        
        # Create chains
        faiss_chain = self.rag_chains.create_faiss_chain(self.faiss_retriever)
        graph_chain = self.rag_chains.create_graph_chain(self.graph_retriever)
        
        print("Both systems ready! Enter your questions (type 'quit' to exit):")
        
        while True:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Run comparison for the custom question
            self.rag_chains.compare_rag_methods(graph_chain, faiss_chain, question)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare Graph RAG vs Traditional RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --mode faiss          # Run FAISS RAG only
    python main.py --mode graph          # Run Graph RAG only  
    python main.py --mode compare        # Run side-by-side comparison
    python main.py --mode interactive    # Interactive mode for custom queries
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['faiss', 'graph', 'compare', 'interactive'],
        default='compare',
        help='Mode to run the comparison (default: compare)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to test (works with faiss, graph, or compare modes)'
    )
    
    args = parser.parse_args()
    
    # Initialize comparison system
    rag_comparison = RAGComparison()
    
    try:
        if args.mode == 'faiss':
            if args.question:
                # Single question with FAISS
                rag_comparison.setup_faiss_rag()
                faiss_chain = rag_comparison.rag_chains.create_faiss_chain(rag_comparison.faiss_retriever)
                rag_comparison.rag_chains.query_faiss_rag(faiss_chain, args.question)
            else:
                # All test questions with FAISS
                results = rag_comparison.run_faiss_only()
                
        elif args.mode == 'graph':
            if args.question:
                # Single question with Graph RAG
                rag_comparison.setup_graph_rag()
                graph_chain = rag_comparison.rag_chains.create_graph_chain(rag_comparison.graph_retriever)
                rag_comparison.rag_chains.query_graph_rag(graph_chain, args.question)
            else:
                # All test questions with Graph RAG
                results = rag_comparison.run_graph_only()
                
        elif args.mode == 'compare':
            if args.question:
                # Single question comparison
                rag_comparison.setup_faiss_rag()
                rag_comparison.setup_graph_rag()
                faiss_chain = rag_comparison.rag_chains.create_faiss_chain(rag_comparison.faiss_retriever)
                graph_chain = rag_comparison.rag_chains.create_graph_chain(rag_comparison.graph_retriever)
                rag_comparison.rag_chains.compare_rag_methods(graph_chain, faiss_chain, args.question)
            else:
                # Full comparison with all test questions
                results = rag_comparison.run_comparison()
                
        elif args.mode == 'interactive':
            rag_comparison.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()