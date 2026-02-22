
import streamlit as st
import os
from langchain_groq import ChatGroq
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
# 1. ფაილებიდან წაკითხვა
# ========================
def load_documents_from_folder(folder_path: str = "docs") -> list[Document]:
    documents = []

    if not os.path.exists(folder_path):
        st.error(f"საქაღალდე '{folder_path}' არ მოიძებნა!")
        st.stop()

    # მხოლოდ .txt და .pdf — requirements.txt და სხვა სერვის ფაილები იგნორირდება
    files = [
        f for f in os.listdir(folder_path)
        if (f.endswith(".txt") or f.endswith(".pdf"))
        and f != "requirements.txt"
        and not f.startswith(".")
    ]

    if not files:
        st.error("docs/ საქაღალდეში დოკუმენტები არ არის!")
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
                        p.extract_text() for p in reader.pages if p.extract_text()
                    ).strip()
                except ImportError:
                    st.warning(f"⚠️ {filename}: pypdf საჭიროა PDF-ისთვის")
                    continue

            if content:
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
        except Exception as e:
            st.warning(f"⚠️ {filename}: {e}")

    return documents

# ========================
# 2. BM25-სტილის keyword retriever ქართულისთვის
# ========================
def keyword_retrieve(query: str, documents: list[Document], top_k: int = 2) -> tuple[list[Document], bool]:
    """
    ქართული ტექსტისთვის keyword-based retrieval.
    აბრუნებს: (დოკუმენტების სია, found: bool)
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    query_words = {w for w in query_words if len(w) > 2}

    scored = []
    for doc in documents:
        content_lower = doc.page_content.lower()
        score = 0

        # სრული სიტყვების შემოწმება
        score += sum(2 for w in query_words if w in content_lower)

        # ასევე ვამოწმებთ ნაწილობრივ დამთხვევას (ქართული ფლექსია)
        # მაგ: "სტუდენტი" მოიძებნება "სტუდენტ"-ში
        for w in query_words:
            if len(w) > 4:
                stem = w[:len(w)-2]  # სიტყვის ფუძე (მარტივი)
                score += sum(1 for _ in [1] if stem in content_lower)

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0] if scored else 0

    # თუ საუკეთესო score ძალიან დაბალია — ინფორმაცია ბაზაში არ არის
    if best_score < 1:
        return [], False

    matched = [(s, d) for s, d in scored if s > 0]
    return [d for _, d in matched[:top_k]], True

# ========================
# 3. Chunking + Retrieval
# ========================
@st.cache_resource
def setup_chunks() -> list[Document]:
    raw_docs = load_documents_from_folder("docs")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(raw_docs), raw_docs

@st.cache_resource
def get_llm(_api_key: str):
    return ChatGroq(
        api_key=_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )

# ========================
# 4. პასუხის გენერაცია
# ========================
def get_answer(query: str, chunks: list[Document], llm) -> tuple[str, list[Document], bool]:
    retrieved, found = keyword_retrieve(query, chunks, top_k=3)

    # თუ საერთოდ არ მოიძებნა შესაბამისი chunk
    if not found:
        return "❌ ამ კითხვაზე ინფორმაცია ბაზაში არ მოიპოვება.", [], False

    context = "\n\n---\n\n".join(
        f"[წყარო: {d.metadata['source']}]\n{d.page_content}"
        for d in retrieved
    )

    prompt = PromptTemplate.from_template(
        "შენ ხარ საგადასახადო/საბაჟო ასისტენტი RS InfoHub-ისთვის.\n"
        "უპასუხე კითხვას მხოლოდ ქვემოთ მოცემულ კონტექსტზე დაყრდნობით ქართულ ენაზე.\n"
        "თუ კონტექსტში კითხვაზე პასუხი არ არის, დაწერე მხოლოდ: NO_INFO\n"
        "სხვა შემთხვევაში პასუხი გასცე და ბოლოში მიუთითე:\n"
        "წყარო: [ფაილის სახელი]\n"
        "პასუხი მომზადებულია RS InfoHub-ის მიხედვით - https://infohub.rs.ge/ka\n\n"
        "კონტექსტი:\n{context}\n\n"
        "კითხვა: {question}\n\nპასუხი:"
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    # თუ LLM-მა თქვა NO_INFO — წყარო არ გამოჩნდეს
    if "NO_INFO" in answer:
        return "❌ ამ კითხვაზე ინფორმაცია ბაზაში არ მოიპოვება.", [], False

    return answer, retrieved, True

# ========================
# 5. UI
# ========================
st.title("🇬🇪 RS InfoHub — RAG აგენტი")
st.caption("საგადასახადო და საბაჟო კითხვებზე პასუხი docs/ საქაღალდის დოკუმენტების საფუძველზე")

with st.spinner("დოკუმენტები იტვირთება..."):
    chunks, raw_docs = setup_chunks()
    llm = get_llm(GROQ_API_KEY)

st.success(f"✅ {len(raw_docs)} დოკუმენტი ჩაიტვირთა → {len(chunks)} chunk-ად დაიყო")

with st.expander("📂 ჩატვირთული დოკუმენტები"):
    docs_folder = "docs"
    if os.path.exists(docs_folder):
        for f in sorted(os.listdir(docs_folder)):
            if (f.endswith(".txt") or f.endswith(".pdf")) and f != "requirements.txt":
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
                answer, source_docs, found = get_answer(user_query, chunks, llm)

            st.markdown(answer)

            # chunk-ები მხოლოდ მაშინ ჩანს როცა ინფორმაცია მოიძებნა
            if found and source_docs:
                with st.expander("🔍 გამოყენებული Chunk-ები"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i} — {doc.metadata['source']}**")
                        st.caption(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

        except Exception as e:
            st.error(f"შეცდომა: {str(e)}")
            answer = "შეცდომა მოხდა."

    st.session_state.messages.append({"role": "assistant", "content": answer})
