# Scheme Knowledge Graph — TN Government Schemes

An LLM-powered pipeline that turns Tamil Nadu government scheme documents (agriculture and MSME) into a **Neo4j knowledge graph**, plus a **Streamlit chat app** that answers questions about schemes, subsidies and departments by retrieving from that graph.

![Pipeline demo](docs/hybrid_rag_pipeline.gif)

## How it works

1. **Load** – `src/kg_pipeline/document_loader.py` reads the scheme `.txt` files in `reference/`.
2. **Split** – `text_splitter.py` breaks documents into chunks.
3. **Extract entities** – `entity_extraction.py` uses an OpenAI chat model (via LangChain) to pull out typed entities (schemes, departments, beneficiaries, benefits…).
4. **Map relationships** – `relationship_mapping.py` asks the LLM for relationships between those entities.
5. **Store** – `graph_store.py` writes nodes and relationships to Neo4j.
6. **Chat** – `app.py` loads the graph, retrieves the nodes/relationships relevant to a question, adds the source scheme text, and has the LLM answer with that context.

## Stack

Python · LangChain · OpenAI · Neo4j (Aura) · Streamlit

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirement/requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
```

## Run

```bash
python scripts/run_full_pipeline.py   # build the knowledge graph in Neo4j
streamlit run app.py                  # chat with it
```

`scripts/` also contains small test scripts for each stage (loader, entity extraction, relationship mapping, graph store, Neo4j connection).
