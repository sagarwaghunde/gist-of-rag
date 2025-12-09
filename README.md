# Introduction to Vector Databases & RAG

A hands-on project demonstrating **Retrieval Augmented Generation (RAG)** using LangChain, OpenAI, and Pinecone vector database.

---

## 🧠 Core Concepts

### What is RAG (Retrieval Augmented Generation)?

RAG is a technique that enhances LLM responses by combining three key steps:

| Step | Description |
|------|-------------|
| **Retrieval** | Retrieving relevant chunks from a knowledge base |
| **Augmentation** | Augmenting the prompt with retrieved context |
| **Generation** | Sending the augmented prompt to LLM for response generation |

> 💡 **Why RAG?** — "Garbage in, garbage out." The more irrelevant tokens we send to an LLM, the more irrelevant the output. RAG helps us send only the most relevant context.

📄 **Research**: [Needle in the Haystack](https://arxiv.org/pdf/2407.01437)

---

### Embeddings

A classic NLP technique that creates a **vector space from text** where the distance between vectors has semantic meaning.

- A vector is simply a sequence of numbers
- Similar concepts have vectors that are close together
- Used to find semantically related content

📄 **Reference**: [OpenAI Text Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)

---

### Vector Store

A specialized database that:
- Stores embeddings (vectors)
- Performs fast similarity searches
- Returns the closest vectors to a query

**Example**: [Pinecone](https://www.pinecone.io/)
**Example**: [FAISS](https://faiss.ai/index.html) [In Memory]
---

### Text Splitters

Text splitters help break large documents ointo smaller chunks to:
- Meet model token limits
- Improve retrieval accuracy

**Key Parameter**: `chunk_overlap` — Creates overlap between chunks to maintain context continuity.

📄 **Reference**: [LangChain Document Loaders](https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain_classic/document_loaders)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   📄 Load File  →  ✂️ Split into Chunks  →  🔢 Embed to     │
│                                               Vectors        │
│                           ↓                                  │
│                   📦 Store in Pinecone                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      RETRIEVAL PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ❓ User Query  →  🔍 Retrieve Relevant  →  📝 Augment     │
│                        Chunks                  Prompt        │
│                           ↓                                  │
│                   🤖 Generate Response (LLM)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
intro-to-vector-dbs/
├── ingestion.py      # Document loading, splitting, embedding & storing
├── main.py           # Retrieval and query chains
├── mediumblog1.txt   # Sample document for ingestion
└── README.md         # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Pipenv](https://pipenv.pypa.io/)
- OpenAI API Key
- Pinecone API Key & Index

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gist-of-rag
   ```

2. **Install dependencies**
   ```bash
   pipenv install
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_index_name
   ```

4. **Activate the virtual environment**
   ```bash
   pipenv shell
   ```

---

## 📖 Usage

### Step 1: Ingest Documents

Load documents, split them into chunks, create embeddings, and store in Pinecone:

```bash
python ingestion.py
```

**What happens:**
1. Loads `mediumblog1.txt` using `TextLoader`
2. Splits into chunks (1000 chars each) using `CharacterTextSplitter`
3. Creates embeddings using OpenAI's `text-embedding-3-small` model
4. Stores vectors in Pinecone

### Step 2: Query with RAG

Run the main application to query the knowledge base:

```bash
python main.py
```

---

## 🔗 Code Examples

### Basic LLM Chain

Simple chain without retrieval:

```python
query = "What is Pinecone in machine learning?"
chain = PromptTemplate.from_template(template=query) | llm
result = chain.invoke(input={})
```

### Retrieval QA Chain (with Pinecone)
In the retrieval chain, the process begins by embedding the query, which allows the system to find and retrieve similar documents. This ensures that both the query and the relevant documents are sent together to the LLM, enabling a more contextual and informed response.

Using LangChain's pre-built retrieval chain:

```python
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX"), embedding=embeddings)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(
    retriever=vectorstore.as_retriever(), 
    combine_docs_chain=combine_docs_chain
)

result = retrieval_chain.invoke(input={"input": query})
```

📄 **Prompt Reference**: [langchain-ai/retrieval-qa-chat](https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat)

### Custom RAG Chain with LCEL

Using LangChain Expression Language (LCEL) with `RunnablePassthrough`:

```python
template = """Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Helpful Answer:
"""

custom_rag_prompt = PromptTemplate.from_template(template=template)
rag_chain = (
    {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt 
    | llm
)

result = rag_chain.invoke(query)
```

> 💡 **RunnablePassthrough**: A runnable that passes inputs unchanged or with additional keys. Behaves like an identity function but can be configured to add extra keys to the output.
>
> 📄 [RunnablePassthrough Docs](https://v03.api.js.langchain.com/classes/_langchain_core.runnables.RunnablePassthrough.html)

---

## 📚 Key LangChain Components

| Component | Purpose |
|-----------|---------|
| `TextLoader` | Load text files |
| `CharacterTextSplitter` | Split text into chunks |
| `OpenAIEmbeddings` | Generate vector embeddings |
| `PineconeVectorStore` | Store & retrieve vectors |
| `create_stuff_documents_chain` | Combine retrieved docs into prompt |
| `create_retrieval_chain` | End-to-end retrieval + generation |
| `RunnablePassthrough` | Pass data through LCEL chains |

---

## 💡 Notes

### Document Combination Strategies

The default **"stuffing" strategy** combines all retrieved documents directly into the prompt. 

**Alternative**: If you want to summarize each document before sending to the LLM, you can chain another summarization step before the combine-docs-chain.

### Supported Document Loaders

LangChain supports various document types:
Ref : https://docs.langchain.com/oss/python/integrations/document_loaders
- `TextLoader` — Plain text files
- `WebBaseLoader` — Web pages
- `PyPDFLoader` — PDF documents
- `CSVLoader` — CSV files
- `JSONLoader` — JSON files
- `Docx2txtLoader` — Word documents

---

## 📦 Dependencies

- `langchain-pinecone` — Pinecone integration
- `langchain-community` — Document loaders & utilities
- `langchainhub` — Pre-built prompts
- `python-dotenv` — Environment variable management
- `black` — Code formatting

---

## 📄 License

This project is for educational purposes.

---

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Needle in the Haystack Paper](https://arxiv.org/pdf/2407.01437)

