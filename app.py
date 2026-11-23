import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv(Path(__file__).resolve().parent / ".env")

from kg_pipeline.graph_store import get_driver  # noqa: E402

st.set_page_config(page_title="TN Schemes Knowledge Graph Chat", page_icon="🌾")
st.title("🌾 TN Government Schemes — Knowledge Graph Chat")


@st.cache_resource
def _driver():
    return get_driver()


@st.cache_data(ttl=300)
def _load_graph():
    driver = _driver()
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    with driver.session(database=database) as session:
        nodes = session.run(
            "MATCH (n) RETURN n.name AS name, labels(n)[0] AS type, n.source AS source"
        ).data()
        rels = session.run(
            "MATCH (a)-[r]->(b) RETURN a.name AS source, type(r) AS type, b.name AS target"
        ).data()
    return nodes, rels


def _retrieve(question: str, nodes, rels, top_k: int = 15):
    q = question.lower()
    scored = []
    for n in nodes:
        name = n["name"] or ""
        words = [w for w in name.lower().split() if len(w) > 2]
        score = sum(1 for w in words if w in q)
        if name.lower() in q:
            score += 3
        if score > 0:
            scored.append((score, n))
    scored.sort(key=lambda x: -x[0])
    matched = [n for _, n in scored[:top_k]]
    matched_names = {n["name"] for n in matched}

    matched_rels = [
        r for r in rels if r["source"] in matched_names or r["target"] in matched_names
    ]
    sources = {n["source"] for n in matched if n.get("source")}
    return matched, matched_rels, sources


def _read_sources(sources: set, max_chars_each: int = 3000) -> str:
    texts = []
    for path in sources:
        try:
            content = Path(path).read_text(encoding="utf-8")
        except OSError:
            continue
        texts.append(f"=== {Path(path).name} ===\n{content[:max_chars_each]}")
    return "\n\n".join(texts)


def _build_context(question: str, nodes, rels) -> str:
    matched_nodes, matched_rels, sources = _retrieve(question, nodes, rels)
    facts = "\n".join(
        f"- {r['source']} -[{r['type']}]-> {r['target']}" for r in matched_rels
    )
    doc_text = _read_sources(sources)
    parts = []
    if facts:
        parts.append(f"Graph facts:\n{facts}")
    if doc_text:
        parts.append(f"Source documents:\n{doc_text}")
    return "\n\n".join(parts) if parts else "No matching entities found in the graph."


@st.cache_resource
def _llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


SYSTEM_PROMPT = (
    "You answer questions about Tamil Nadu government schemes using only the "
    "provided graph facts and source documents. If the context does not contain "
    "the answer, say you don't have that information. Cite the scheme name(s) you "
    "used."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about a scheme, subsidy, or department...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    nodes, rels = _load_graph()
    context = _build_context(question, nodes, rels)

    response = _llm().invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
    answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("Retrieved context"):
            st.text(context)
