import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



class PDFLoaderService:
    """Handles reading PDFs and splitting them into chunks."""

    def __init__(self, pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.pdf_dir = pdf_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_and_split(self):
        """Load all PDFs in directory and split into text chunks."""
        pdf_files = [
            os.path.join(self.pdf_dir, f)
            for f in os.listdir(self.pdf_dir)
            if f.endswith(".pdf")
        ]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")

        docs = []
        for file in pdf_files:
            print(f"Loading {file}...")
            loader = PyPDFLoader(file)
            docs.extend(loader.load())

        print("Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(docs)
