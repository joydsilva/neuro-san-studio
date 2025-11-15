# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san-studio SDK Software in commercial settings.
#
# END COPYRIGHT

import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neuro_san.interfaces.coded_tool import CodedTool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TextFileInfoProvider(CodedTool):
    """
    CodedTool implementation which provides information from text files containing
    details about excess and specialty lines, its available programs and additional coverages.
    
    Uses in-memory RAG (Retrieval Augmented Generation) with vector embeddings for 
    semantic search and improved query understanding.
    """

    def __init__(self):
        super().__init__()
        self.vector_store: Optional[VectorStore] = None
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536)
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store with the default excess specialty lines data."""
        try:
            # Default to the excess specialty lines info file
            file_path = "data/excess_specialty_lines_info.txt"
            
            # Convert relative path to absolute path from the project root
            if not os.path.isabs(file_path):
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Go up one level to get to the project root
                project_root = os.path.dirname(script_dir)
                file_path = os.path.join(project_root, file_path)

            if not os.path.exists(file_path):
                logger.warning(f"Default data file not found at: {file_path}")
                return

            # Load and process the document
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Create document chunks for better retrieval
            documents = self._create_documents_from_content(content, file_path)
            
            if documents:
                # Create vector store
                self.vector_store = InMemoryVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
                logger.info(f"Vector store initialized with {len(documents)} document chunks")
            else:
                logger.warning("No documents created from content")

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")

    def _create_documents_from_content(self, content: str, source_path: str) -> List[Document]:
        """Create and split documents from text content for optimal retrieval."""
        try:
            # Create a document from the content
            doc = Document(
                page_content=content,
                metadata={
                    "source": source_path,
                    "type": "excess_specialty_lines_info",
                    "content_type": "insurance_documentation"
                }
            )

            # Split the document into smaller chunks for better embedding and retrieval
            # Using smaller chunks for insurance content to maintain context
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=800,  # Larger chunks for insurance content
                chunk_overlap=100,  # Good overlap to maintain context
                separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]
            )

            doc_chunks = text_splitter.split_documents([doc])
            
            # Add additional metadata to chunks for better retrieval
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk.page_content),
                    "keywords": self._extract_keywords(chunk.page_content)
                })

            logger.info(f"Created {len(doc_chunks)} document chunks")
            return doc_chunks

        except Exception as e:
            logger.error(f"Error creating documents from content: {str(e)}")
            return []

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text content for enhanced metadata."""
        # Insurance-specific keywords
        insurance_keywords = [
            "excess", "specialty lines", "surplus lines", "commercial property", 
            "liability", "coverage", "programs", "contractors", "manufacturing",
            "property damage", "bodily injury", "professional liability",
            "workers compensation", "cyber liability", "general liability",
            "auto liability", "umbrella", "crime", "equipment breakdown",
            "inland marine", "builders risk", "vacant building"
        ]
        
        text_lower = text.lower()
        found_keywords = [keyword for keyword in insurance_keywords if keyword in text_lower]
        return found_keywords

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """
        Retrieves and provides information from text files containing details about
        excess and specialty lines using semantic search with RAG.

        :param args: Dictionary containing:
            "file_path": optional path to the text file to read (defaults to excess_specialty_lines_info.txt)
            "query": optional search query to find specific information about programs, coverages, or specialty lines
            "section": optional section to focus on (e.g., "programs", "coverages", "specialty_lines")
            "max_chars": maximum number of characters to return (default is 4000)
            "k": number of top similar documents to retrieve (default is 4)

        :param sly_data: A dictionary whose keys are defined by the agent
            hierarchy, but whose values are meant to be kept out of the
            chat stream.
        """
        # Extract parameters
        file_path = args.get("file_path")
        query = args.get("query", "")
        section = args.get("section", "")
        max_chars = args.get("max_chars", 4000)
        k = args.get("k", 4)

        # If a custom file path is provided, create a temporary vector store
        if file_path and file_path != "data/excess_specialty_lines_info.txt":
            return await self._query_custom_file(file_path, query, section, max_chars)

        # Use the pre-built vector store for the default file
        if not self.vector_store:
            return "Error: Vector store not initialized. Please check the excess_specialty_lines_info.txt file exists."

        try:
            # Build the search query
            search_query = self._build_search_query(query, section)
            
            if not search_query:
                search_query = "excess specialty lines insurance coverage programs"

            # Perform semantic search using the vector store
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            results = await retriever.ainvoke(search_query)

            if not results:
                return f"No relevant information found for query: '{search_query}'"

            # Combine and format the results
            combined_content = self._format_rag_results(results, search_query, max_chars)
            
            return combined_content

        except Exception as e:
            logger.error(f"Error during RAG query: {str(e)}")
            return f"Error retrieving information: {str(e)}"

    def _build_search_query(self, query: str, section: str) -> str:
        """Build an optimized search query for the vector store."""
        search_parts = []
        
        if section:
            search_parts.append(f"{section}")
            
        if query:
            search_parts.append(query)
        
        # Add context for better semantic matching
        if search_parts:
            search_query = " ".join(search_parts)
            # Add insurance context if not already present
            if not any(term in search_query.lower() for term in ["insurance", "coverage", "liability", "property"]):
                search_query += " insurance coverage"
        else:
            search_query = ""
            
        return search_query

    def _format_rag_results(self, results: List[Document], query: str, max_chars: int) -> str:
        """Format the RAG retrieval results into a readable response."""
        if not results:
            return "No relevant information found."

        # Start with a header
        response = f"Found relevant information for '{query}' in excess and specialty lines documentation:\n\n"
        
        # Combine content from retrieved documents
        combined_content = []
        total_chars = len(response)
        
        for i, doc in enumerate(results):
            content = doc.page_content.strip()
            
            # Add section separator
            section_header = f"--- Section {i+1} ---\n"
            section_content = section_header + content
            
            # Check if adding this section would exceed max_chars
            if total_chars + len(section_content) + 2 > max_chars:  # +2 for \n\n
                remaining_chars = max_chars - total_chars - 2
                if remaining_chars > 100:  # Only add if we have reasonable space
                    truncated_content = content[:remaining_chars-20] + "..."
                    combined_content.append(section_header + truncated_content)
                break
            
            combined_content.append(section_content)
            total_chars += len(section_content) + 2

        response += "\n\n".join(combined_content)
        
        # Add metadata about the search
        if len(results) > 1:
            response += f"\n\n[Retrieved {len(combined_content)} relevant sections from {len(results)} total matches]"
        
        return response

    async def _query_custom_file(self, file_path: str, query: str, section: str, max_chars: int) -> str:
        """Handle queries for custom file paths by creating a temporary vector store."""
        try:
            # Convert relative path to absolute path from the project root
            if not os.path.isabs(file_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(script_dir)
                file_path = os.path.join(project_root, file_path)

            if not os.path.exists(file_path):
                return f"Error: File not found at path: {file_path}"

            # Read the custom file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Create documents and temporary vector store
            documents = self._create_documents_from_content(content, file_path)
            
            if not documents:
                return f"Error: Could not process content from {file_path}"

            # Create temporary vector store
            temp_vector_store = InMemoryVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Perform search on temporary vector store
            search_query = self._build_search_query(query, section)
            if not search_query:
                search_query = "information content"

            retriever = temp_vector_store.as_retriever(search_kwargs={"k": 4})
            results = await retriever.ainvoke(search_query)

            if not results:
                # Fallback to simple content return
                return f"Content from {file_path}:\n\n" + content[:max_chars]

            return self._format_rag_results(results, search_query, max_chars)

        except Exception as e:
            logger.error(f"Error querying custom file {file_path}: {str(e)}")
            return f"Error reading file: {str(e)}"

    def get_description(self) -> str:
        return """
        Retrieve and provide information from text files containing details about excess 
        and specialty lines using advanced RAG (Retrieval Augmented Generation) with semantic search.
        
        By default, uses a pre-built in-memory vector store of the excess_specialty_lines_info.txt 
        file containing comprehensive information about Nationwide Excess & Surplus Insurance offerings.
        
        Key Features:
        - Semantic search using OpenAI embeddings for contextual understanding
        - Pre-chunked document storage for optimal retrieval
        - Intelligent query processing and result formatting
        
        Specializes in extracting information about:
        - Excess and surplus insurance lines
        - Commercial property insurance programs
        - Available insurance programs and coverages
        - Additional coverage options
        - Specialty insurance products
        - Professional liability coverages
        - Property and casualty specialty lines
        
        Parameters:
        - file_path (optional): Path to a specific text file (defaults to excess_specialty_lines_info.txt)
        - query (optional): Search query for semantic matching of programs, coverages, or specialty lines
        - section (optional): Focus on specific section (e.g., "programs", "coverages", "commercial property")
        - max_chars (optional): Maximum number of characters to return (default: 4000)
        - k (optional): Number of most relevant document chunks to retrieve (default: 4)
        """