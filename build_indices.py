#!/usr/bin/env python3
"""
One-time Index Builder Script

This script builds and saves both FAISS and Graph indices.
Run this once, then use the demo/main scripts multiple times without rebuilding.

Usage:
    python build_indices.py [--rebuild]
"""

import os
import argparse
from config import SYNTHETIC_DATA_PATH
from graph_indexer import GraphIndexer
from faiss_indexer import FAISSIndexer

# Index storage paths
FAISS_INDEX_PATH = "faiss_index"
GRAPH_INDEX_MARKER = ".graph_index_built"

def check_existing_indices():
    """Check if indices already exist"""
    faiss_exists = os.path.exists(FAISS_INDEX_PATH)
    graph_exists = os.path.exists(GRAPH_INDEX_MARKER)
    
    return faiss_exists, graph_exists

def build_faiss_index(force_rebuild=False):
    """Build FAISS index if it doesn't exist or force rebuild"""
    faiss_exists, _ = check_existing_indices()
    
    if faiss_exists and not force_rebuild:
        print("✅ FAISS index already exists, skipping...")
        return True
    
    print("🚀 Building FAISS index...")
    try:
        faiss_indexer = FAISSIndexer()
        faiss_indexer.build_index(SYNTHETIC_DATA_PATH)
        faiss_indexer.save_index(FAISS_INDEX_PATH)
        faiss_indexer.get_index_stats()
        print("✅ FAISS index built and saved successfully!")
        return True
    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")
        return False

def build_graph_index(force_rebuild=False):
    """Build Graph index if it doesn't exist or force rebuild"""
    _, graph_exists = check_existing_indices()
    
    if graph_exists and not force_rebuild:
        print("✅ Graph index already exists, skipping...")
        return True
    
    print("🚀 Building Graph index...")
    try:
        graph_indexer = GraphIndexer()
        
        # Clear existing graph data if rebuilding
        if force_rebuild:
            graph_indexer.kg.query("MATCH (n) DETACH DELETE n")
            print("🧹 Cleared existing graph data")
        
        graph_indexer.build_index(SYNTHETIC_DATA_PATH)
        graph_indexer.get_graph_stats()
        
        # Create marker file to indicate graph is built
        with open(GRAPH_INDEX_MARKER, 'w') as f:
            f.write("Graph index built successfully")
        
        print("✅ Graph index built successfully!")
        return True
    except Exception as e:
        print(f"❌ Error building Graph index: {e}")
        return False

def main():
    """Main index building function"""
    parser = argparse.ArgumentParser(
        description="Build indices for Graph RAG vs Traditional RAG comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python build_indices.py              # Build indices if they don't exist
    python build_indices.py --rebuild    # Force rebuild all indices
        """
    )
    
    parser.add_argument(
        '--rebuild', 
        action='store_true',
        help='Force rebuild indices even if they exist'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("INDEX BUILDER - Graph RAG vs Traditional RAG")
    print("="*60)
    
    if args.rebuild:
        print("🔄 Rebuilding all indices...")
    else:
        faiss_exists, graph_exists = check_existing_indices()
        if faiss_exists and graph_exists:
            print("✅ All indices already exist!")
            print("💡 Use --rebuild flag to force rebuild")
            print("🚀 You can now run: python demo.py")
            return
    
    # Build indices
    success_count = 0
    
    if build_faiss_index(args.rebuild):
        success_count += 1
    
    if build_graph_index(args.rebuild):
        success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("INDEX BUILDING SUMMARY")
    print("="*60)
    
    if success_count == 2:
        print("🎉 All indices built successfully!")
        print("🚀 You can now run:")
        print("   - python demo.py")
        print("   - python main.py --mode compare")
    else:
        print("⚠️  Some indices failed to build. Check the errors above.")
        print("💡 Make sure your .env file is configured correctly")
    
    print("="*60)

if __name__ == "__main__":
    main()