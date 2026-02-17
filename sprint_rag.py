"""Complete RAG system - FAISS version (stable)."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path


class SimpleRAG:
    """Minimal RAG system using FAISS."""

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = None
        self.chain = None
        self.retriever = None

    def load_documents(self, folder_path):
        """Load all PDFs from folder."""
        docs = []
        pdf_files = list(Path(folder_path).glob("*.pdf"))
        for pdf_file in pdf_files:
            loader = PyPDFLoader(str(pdf_file))
            docs.extend(loader.load())

        print(f"✅ Loaded {len(docs)} pages from {len(pdf_files)} PDFs")
        return docs

    def create_vector_store(self, documents):
        """Chunk and embed documents."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        # Use FAISS instead of ChromaDB
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        # Save to disk
        self.vector_store.save_local("./data/faiss_db")
        print("✅ Vector store created and saved")
        return chunks

    def setup_qa_chain(self):
        """Setup Q&A chain."""
        template = """Use ONLY the following context to answer the question.
If the answer is not in the context, say "I cannot find this in the documents."
Always mention which document your answer comes from.

Context:
{context}

Question: {question}

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

        def format_docs(docs):
            return "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in docs
            ])

        self.chain = (
            {"context": self.retriever | format_docs,
             "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print("✅ Q&A chain ready")

    def ask(self, question):
        """Ask a question about the documents."""
        if not self.chain:
            raise ValueError("Call setup_qa_chain() first!")

        answer = self.chain.invoke(question)
        sources = self.retriever.invoke(question)

        return {
            "answer": answer,
            "sources": list(set([
                doc.metadata.get("source", "Unknown")
                for doc in sources
            ]))
        }


def test_rag():
    from sprint_bedrock import get_llm, get_embeddings

    print("\n🚀 Testing RAG System...\n")

    llm = get_llm()
    embeddings = get_embeddings()
    rag = SimpleRAG(llm, embeddings)

    docs = rag.load_documents("./data/sample_contracts")

    if len(docs) == 0:
        print("❌ No PDFs found in data/sample_contracts/")
        return None

    rag.create_vector_store(docs)
    rag.setup_qa_chain()

    print("\n🧪 Test 1: General question")
    result = rag.ask("What is this document about?")
    print(f"📝 Answer: {result['answer']}")
    print(f"📚 Sources: {result['sources']}")

    print("\n🧪 Test 2: Specific question")
    result2 = rag.ask("What are the key terms or conditions mentioned?")
    print(f"📝 Answer: {result2['answer']}")
    print(f"📚 Sources: {result2['sources']}")

    return rag


if __name__ == "__main__":
    test_rag()