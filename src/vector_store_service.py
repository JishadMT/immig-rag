import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings



class VectorStoreService:
    """Handles embedding creation and persistent Chroma storage."""

    def __init__(self, persist_dir: str, embed_model: str = "nomic-embed-text"):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(model=embed_model)
        self.vector_store = None

    def load_or_create_db(self, documents=None):
        """Load existing Chroma DB, or create and persist a new one."""
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Found existing Chroma DB at {self.persist_dir}. Loading...")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            if documents is None:
                raise ValueError("No documents provided to create new Chroma DB.")

            print("No existing Chroma DB found. Creating a new one...")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
            self.vector_store.persist()
            print(f"New Chroma DB created at {self.persist_dir}")
        return self.vector_store
