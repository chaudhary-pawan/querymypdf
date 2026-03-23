import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


# ── Cached embeddings — loaded once per session ───────────────────────────────
_embeddings_cache = None

def get_embeddings(api_key: str):
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
    return _embeddings_cache


# ── Strict system prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a document Q&A assistant. Your ONLY job is to answer questions using the provided document excerpts.

STRICT RULES — follow these absolutely:
1. Answer ONLY from the DOCUMENT CONTEXT provided. Do NOT use your training knowledge.
2. Do NOT invent, assume, or infer anything not explicitly stated in the context.
3. Do NOT mention any names, titles, authors, dates, or facts unless they appear verbatim in the context.
4. If the context does not contain the answer, say EXACTLY: "This information is not present in the document."
5. Never say things like "based on my knowledge" or "typically" — stick to the document only.
6. Find out what the document is about. If the user asks "what is this document about?", check the index part if it is unavailable then fetch all headings and subheadings and find out about them and send user a concise summary of the document.
7. Extract basic information about the whole document and persist the information in the session state."""

class RAGChatbot:

    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.0,
            max_output_tokens=1024,
        )
        self.embeddings = get_embeddings(api_key)
        self.last_retrieved_docs = []

    def load_pdf(self, uploaded_file):
        """Extract text from uploaded PDF."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()

        if len(documents) == 0:
            raise ValueError("PDF extraction failed — file may be scanned.")

        return documents

    def create_rag_pipeline(self, documents):
        """Build FAISS vector DB + document retriever."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No text chunks found. Try a different PDF.")

        vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        return True

    def _build_prompt(self, question: str, context: str) -> str:
        """Build a single combined prompt string for Gemini."""
        return f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:
---
{context}
---

QUESTION: {question}

Answer using ONLY the document context above. Do not use outside knowledge."""

    def ask(self, _pipeline, question):
        """Non-streaming version."""
        docs = self.retriever.get_relevant_documents(question)
        self.last_retrieved_docs = docs
        context = "\n\n".join([d.page_content for d in docs])
        prompt = self._build_prompt(question, context)
        response = self.llm.invoke(prompt)
        return response.content

    def ask_stream(self, _pipeline, question):
        """Streaming version — yields tokens for live display."""
        docs = self.retriever.get_relevant_documents(question)
        self.last_retrieved_docs = docs
        context = "\n\n".join([d.page_content for d in docs])
        prompt = self._build_prompt(question, context)

        for chunk in self.llm.stream(prompt):
            if chunk.content:
                yield chunk.content

    def get_last_chunks(self):
        """Return retrieved chunks for hallucination inspection panel."""
        return [
            {
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
                "page":    doc.metadata.get("page", "?"),
            }
            for doc in self.last_retrieved_docs
        ]
