import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="RS InfoHub RAG", page_icon="🇬🇪", layout="centered")

# ========================
# API Key
# ========================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY არ არის დაყენებული Secrets-ში!")
    st.stop()

# ========================
# 1. ფაილებიდან წაკითხვა (TXT + PDF)
# ========================
def load_documents_from_folder(folder_path: str = "docs") -> list[Document]:
    documents = []

    if not os.path.exists(folder_path):
        st.error(f"საქაღალდე '{folder_path}' არ მოიძებნა!")
        st.stop()

    files = [f for f in os.listdir(folder_path) if f.endswith(".txt") or f.endswith(".pdf")]

    if not files:
        st.error("docs/ საქაღალდეში .txt ან .pdf ფაილები არ არის!")
        st.stop()

    for filename in sorted(files):
        filepath = os.path.join(folder_path, filename)

        try:
            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()

            elif filename.endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    content = "\n\n".join(
                        page.extract_text() for page in reader.pages
                        if page.extract_text()
                    ).strip()
                except ImportError:
                    st.warning(f"⚠️ {filename}: pypdf არ არის დაინსტალირებული. გამოიყენე .txt ფაილები.")
                    continue

            if content:
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))

        except Exception as e:
            st.warning(f"⚠️ {filename} ვერ წაიკითხა: {e}")

    return documents

# ========================
# 2. RAG სისტემა
# ========================
@st.cache_resource
def setup_rag(_api_key: str):
    raw_docs = load_documents_from_folder("docs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    llm = ChatGroq(
        api_key=_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )

    return retriever, llm, len(split_docs), len(raw_docs)

# ========================
# 3. პასუხის გენერაცია
# ========================
def get_answer(query: str, retriever, llm) -> tuple[str, list[Document]]:
    retrieved_docs = retriever.invoke(query)

    context = "\n\n---\n\n".join(
        f"[წყარო: {d.metadata['source']}]\n{d.page_content}"
        for d in retrieved_docs
    )

    prompt = PromptTemplate.from_template(
        "შენ ხარ საგადასახადო/საბაჟო ასისტენტი RS InfoHub-ისთვის.\n"
        "უპასუხე კითხვას მხოლოდ ქვემოთ მოცემულ კონტექსტზე დაყრდნობით ქართულ ენაზე.\n"
        "თუ კონტექსტში პასუხი არ არის, თქვი: 'ამ კითხვაზე ინფორმაცია ბაზაში არ მოიპოვება.'\n\n"
        "კონტექსტი:\n{context}\n\n"
        "კითხვა: {question}\n\n"
        "პასუხი (ბოლოში მიუთითე წყარო და: "
        "პასუხი მომზადებულია RS InfoHub-ის მიხედვით - https://infohub.rs.ge/ka):"
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    return answer, retrieved_docs

# ========================
# 4. UI
# ========================
st.title("🇬🇪 RS InfoHub — RAG აგენტი")
st.caption("საგადასახადო და საბაჟო კითხვებზე პასუხი docs/ საქაღალდის დოკუმენტების საფუძველზე")

with st.spinner("დოკუმენტები იტვირთება..."):
    retriever, llm, chunk_count, doc_count = setup_rag(GROQ_API_KEY)

st.success(f"✅ {doc_count} დოკუმენტი ჩაიტვირთა → {chunk_count} chunk-ად დაიყო")

with st.expander("📂 ჩატვირთული დოკუმენტები"):
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        for f in sorted(os.listdir(docs_folder)):
            if f.endswith((".txt", ".pdf")):
                size = os.path.getsize(os.path.join(docs_folder, f))
                icon = "📄" if f.endswith(".txt") else "📕"
                st.markdown(f"- {icon} `{f}` — {size:,} byte")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("დასვი კითხვა ქართულად...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("პასუხი იძებნება..."):
                answer, source_docs = get_answer(user_query, retriever, llm)

            st.markdown(answer)

            with st.expander("🔍 გამოყენებული Chunk-ები"):
                for i, doc in enumerate(source_docs, 1):
                    st.markdown(f"**Chunk {i} — {doc.metadata['source']}**")
                    st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

        except Exception as e:
            st.error(f"შეცდომა: {str(e)}")
            answer = "შეცდომა მოხდა."

    st.session_state.messages.append({"role": "assistant", "content": answer})
