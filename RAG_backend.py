import tempfile
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# ── Cached embeddings — loaded once per session ───────────────────────────────
_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings_cache


# ── Strict system prompt — applied once at the model role level ───────────────
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

    def __init__(self):
        # ── ChatOllama: uses system/user role separation (much better for instruct models)
        self.llm = ChatOllama(
            model="llama3.2",
            num_predict=512,
            temperature=0.0,    # fully deterministic — zero creativity/hallucination risk
            num_ctx=3000,       # enough for 4 chunks of 600 tokens + system prompt
        )
        self.embeddings = get_embeddings()
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

    def _build_messages(self, question: str, context: str):
        """Build the chat message list with system prompt + user query."""
        user_content = f"""DOCUMENT CONTEXT:
---
{context}
---

QUESTION: {question}

Answer using ONLY the document context above. Do not use outside knowledge."""

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    def ask(self, _pipeline, question):
        """Non-streaming version."""
        docs = self.retriever.get_relevant_documents(question)
        self.last_retrieved_docs = docs
        context = "\n\n".join([d.page_content for d in docs])
        messages = self._build_messages(question, context)
        response = self.llm.invoke(messages)
        return response.content

    def ask_stream(self, _pipeline, question):
        """Streaming version — yields tokens for live display."""
        docs = self.retriever.get_relevant_documents(question)
        self.last_retrieved_docs = docs
        context = "\n\n".join([d.page_content for d in docs])
        messages = self._build_messages(question, context)

        for chunk in self.llm.stream(messages):
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
