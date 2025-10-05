import os
import argparse
import shutil
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from pdf_loader_service import PDFLoaderService
from vector_store_service import VectorStoreService


class ChatService:
    """Handles the chat flow using RetrievalQA."""

    def __init__(self, pdf_dir: str, db_dir: str, llm_model: str = "llama3", rebuild: bool = False):
        self.pdf_dir = pdf_dir
        self.db_dir = db_dir
        self.llm_model = llm_model
        self.rebuild = rebuild
        self.qa_chain = None

    def setup(self):
        """Initialize or reload vector DB and create RetrievalQA chain."""
        documents = None

        # 🧹 If rebuild flag is set, clear the existing DB first
        if self.rebuild and os.path.exists(self.db_dir):
            print(f"Clearing existing Chroma DB at {self.db_dir}...")
            shutil.rmtree(self.db_dir, ignore_errors=True)

        # Only load PDFs if the DB doesn't already exist
        if not (os.path.exists(self.db_dir) and os.listdir(self.db_dir)):
            print("Loading and processing PDFs...")
            pdf_loader = PDFLoaderService(self.pdf_dir)
            documents = pdf_loader.load_and_split()

        # Create or load vector DB
        vector_service = VectorStoreService(self.db_dir)
        vector_store = vector_service.load_or_create_db(documents)

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Custom prompt
        custom_prompt = PromptTemplate.from_template("""
You are a knowledgeable assistant who answers questions *only* based on the provided context.
If the answer cannot be found in the context, say clearly:
"I’m sorry, but I could not find that information in the documents."

Be clear, concise, and cite key details if possible.

---------------------
Context:
{context}
---------------------
Question:
{question}
---------------------
Answer:
""")

        # LLM + RetrievalQA
        llm = OllamaLLM(model=self.llm_model)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": custom_prompt}
        )

    def chat(self):
        """Interactive chat loop."""
        if not self.qa_chain:
            self.setup()

        print("\nChat with your documents! Type 'exit' to quit.")
        while True:
            query = input("\nYour question: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            result = self.qa_chain.invoke({"query": query})
            print("\nAnswer:\n", result["result"])
            print("\nSources:")
            for doc in result["source_documents"]:
                print(" -", doc.metadata.get("source", "Unknown"))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chat with your PDFs using a RAG pipeline.")
    parser.add_argument("--pdf-dir", type=str, default="./assets/pdfs", help="Path to folder containing PDFs.")
    parser.add_argument("--db-dir", type=str, default="./assets/chroma_db_store", help="Path to persistent Chroma DB directory.")
    parser.add_argument("--model", type=str, default="llama3.2", help="LLM model name for Ollama.")
    parser.add_argument("--rebuild", action="store_true", help="Clear and rebuild the vector DB from PDFs.")
    args = parser.parse_args()

    chat_app = ChatService(
        pdf_dir=args.pdf_dir,
        db_dir=args.db_dir,
        llm_model=args.model,
        rebuild=args.rebuild
    )
    chat_app.chat()
