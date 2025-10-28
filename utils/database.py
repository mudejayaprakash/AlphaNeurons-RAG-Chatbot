"""
database.py
Vector database interactions and management for medical policy RAG system.
Provides a clean interface for ChromaDB operations with error handling.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import traceback
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

class VectorDBConfig:
    """Configuration for vector database."""
    
    def __init__(
        self,
        persist_dir: str,
        embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        collection_name: str = "policy_collection"
    ):
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
    def validate(self) -> Tuple[bool, str]:
        """Validate configuration."""
        if not self.persist_dir:
            return False, "Persist directory not specified"
        
        persist_path = Path(self.persist_dir)
        if not persist_path.exists():
            return False, f"Persist directory does not exist: {self.persist_dir}"
        
        return True, ""


# ============================================================
# VECTOR DATABASE WRAPPER
# ============================================================

class VectorDatabase:
    """
    Wrapper class for vector database operations.
    Provides a clean interface with error handling and logging.
    """
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize vector database connection.
        
        Args:
            config: VectorDBConfig instance with database settings
        """
        self.config = config
        self.db: Optional[Chroma] = None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize database connection."""
        try:
            # Validate configuration
            is_valid, error_msg = self.config.validate()
            if not is_valid:
                raise ValueError(f"Invalid configuration: {error_msg}")
            
            # Load embedding model
            print(f"Loading embedding model: {self.config.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            
            # Load existing vector database
            print(f"Loading vector database from: {self.config.persist_dir}")
            self.db = Chroma(
                persist_directory=str(self.config.persist_dir),
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name
            )
            
            # Verify database loaded
            count = self.get_document_count()
            print(f"✓ Vector database loaded successfully! Total documents: {count}")
            
        except Exception as e:
            print(f"✗ Failed to initialize vector database: {e}")
            traceback.print_exc()
            raise
    
    def get_document_count(self) -> int:
        """
        Get total number of documents in database.
        
        Returns:
            Number of documents
        """
        try:
            if self.db and hasattr(self.db, '_collection'):
                return self.db._collection.count()
            return 0
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0
    
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform similarity search on vector database.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Document objects
        """
        try:
            if not self.db:
                print("Error: Database not initialized")
                return []
            
            if not query or not query.strip():
                print("Error: Empty query provided")
                return []
            
            # Perform search
            results = self.db.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            print(f"Similarity search returned {len(results)} results for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {e}")
            traceback.print_exc()
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            if not self.db:
                print("Error: Database not initialized")
                return []
            
            if not query or not query.strip():
                print("Error: Empty query provided")
                return []
            
            # Perform search with scores
            results = self.db.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            print(f"Similarity search with scores returned {len(results)} results")
            return results
            
        except Exception as e:
            print(f"Error during similarity search with scores: {e}")
            traceback.print_exc()
            return []
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance (MMR) search for diverse results.
        MMR balances relevance with diversity to avoid redundant results.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of results to fetch before MMR reranking
            lambda_mult: Balance between relevance (1) and diversity (0)
            filter: Optional metadata filter
            
        Returns:
            List of Document objects
        """
        try:
            if not self.db:
                print("Error: Database not initialized")
                return []
            
            if not query or not query.strip():
                print("Error: Empty query provided")
                return []
            
            # Perform MMR search
            results = self.db.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter
            )
            
            print(f"MMR search returned {len(results)} diverse results")
            return results
            
        except Exception as e:
            print(f"Error during MMR search: {e}")
            traceback.print_exc()
            return []
    
    def get_documents_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 100
    ) -> List[Document]:
        """
        Retrieve documents by metadata filter.
        
        Args:
            metadata_filter: Dictionary of metadata key-value pairs to match
            limit: Maximum number of documents to return
            
        Returns:
            List of matching Document objects
        """
        try:
            if not self.db:
                print("Error: Database not initialized")
                return []
            
            # Get documents with metadata filter
            results = self.db.get(
                where=metadata_filter,
                limit=limit
            )
            
            # Convert to Document objects
            documents = []
            if results and 'documents' in results and 'metadatas' in results:
                for doc_text, metadata in zip(results['documents'], results['metadatas']):
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))
            
            print(f"Retrieved {len(documents)} documents matching filter: {metadata_filter}")
            return documents
            
        except Exception as e:
            print(f"Error retrieving documents by metadata: {e}")
            traceback.print_exc()
            return []
    
    def get_unique_metadata_values(
        self,
        metadata_key: str
    ) -> List[str]:
        """
        Get all unique values for a specific metadata field.
        Useful for filtering or displaying available options.
        
        Args:
            metadata_key: Metadata field to get unique values for
            
        Returns:
            List of unique values
        """
        try:
            if not self.db or not hasattr(self.db, '_collection'):
                return []
            
            # Get all documents and extract unique metadata values
            results = self.db._collection.get()
            if not results or 'metadatas' not in results:
                return []
            
            unique_values = set()
            for metadata in results['metadatas']:
                if metadata and metadata_key in metadata:
                    unique_values.add(str(metadata[metadata_key]))
            
            return sorted(list(unique_values))
            
        except Exception as e:
            print(f"Error getting unique metadata values: {e}")
            return []
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Check database health and connectivity.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        try:
            if not self.db:
                return False, "Database not initialized"
            
            # Try to get document count
            count = self.get_document_count()
            
            if count == 0:
                return False, "Database is empty - no documents found"
            
            # Try a simple search
            test_results = self.similarity_search("test", k=1)
            
            return True, f"Database healthy with {count} documents"
            
        except Exception as e:
            return False, f"Database health check failed: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {
                "total_documents": self.get_document_count(),
                "embedding_model": self.config.embedding_model,
                "persist_dir": str(self.config.persist_dir),
                "collection_name": self.config.collection_name,
            }
            
            # Get unique document sources if available
            unique_sources = self.get_unique_metadata_values("source")
            stats["unique_sources"] = len(unique_sources)
            stats["sources"] = unique_sources[:10]  # First 10 sources
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {
                "error": str(e)
            }
    
    def close(self) -> None:
        """Close database connection and cleanup."""
        try:
            if self.db:
                # ChromaDB doesn't require explicit closing
                print("Database connection closed")
                self.db = None
                self.embeddings = None
        except Exception as e:
            print(f"Error closing database: {e}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_vector_db(
    persist_dir: str,
    embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    collection_name: str = "policy_collection"
) -> Optional[VectorDatabase]:
    """
    Load vector database with default configuration.
    Convenience function for simple use cases.
    
    Args:
        persist_dir: Directory where vector database is stored
        embedding_model: Name of embedding model to use
        collection_name: Name of collection in database
        
    Returns:
        VectorDatabase instance or None if loading failed
    """
    try:
        config = VectorDBConfig(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        
        db = VectorDatabase(config)
        return db
        
    except Exception as e:
        print(f"Failed to load vector database: {e}")
        traceback.print_exc()
        return None


def create_test_database(persist_dir: str = "./test_db") -> Optional[VectorDatabase]:
    """
    Create a small test database for development/testing.
    
    Args:
        persist_dir: Directory to store test database
        
    Returns:
        VectorDatabase instance with test data
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Sample policy documents
        test_docs = [
            "Coverage for MRI scans requires prior authorization from a physician.",
            "Emergency room visits are covered 24/7 with a copay of $100.",
            "Physical therapy is limited to 30 sessions per year.",
            "Prescription medications require a $20 copay for generic drugs.",
        ]
        
        # Create documents
        documents = [
            Document(
                page_content=text,
                metadata={"source": "test_policy", "page": i}
            )
            for i, text in enumerate(test_docs)
        ]
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create and persist database
        print(f"Creating test database at: {persist_dir}")
        test_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        # Load as VectorDatabase
        config = VectorDBConfig(
            persist_dir=persist_dir,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        return VectorDatabase(config)
        
    except Exception as e:
        print(f"Failed to create test database: {e}")
        traceback.print_exc()
        return None


# ============================================================
# TESTING
# ============================================================

def test_vector_database():
    """Test vector database functionality."""
    print("Testing Vector Database...")
    print("=" * 60)
    
    # Create test database
    db = create_test_database()
    
    if db:
        # Health check
        is_healthy, message = db.health_check()
        print(f"Health Check: {message}")
        
        # Get stats
        stats = db.get_stats()
        print(f"Database Stats: {stats}")
        
        # Test search
        results = db.similarity_search("MRI coverage", k=2)
        print(f"Search Results ({len(results)}):")
        for i, doc in enumerate(results, 1):
            print(f"  {i}. {doc.page_content[:100]}...")
        
        # Cleanup
        db.close()
        print("✓ Tests passed!")
    else:
        print("✗ Failed to create test database")


if __name__ == "__main__":
    test_vector_database()
