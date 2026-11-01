# Automatic Research Novelty Assessment System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAlex API](https://img.shields.io/badge/retrieval-OpenAlex%20API-orange.svg)](https://openalex.org/)
[![NetworkX](https://img.shields.io/badge/graphs-NetworkX-blueviolet.svg)](https://networkx.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An evidence-grounded scientific platform designed to evaluate research paper originality, detect multi-dimensional research gaps, and benchmark literature novelty using **Large Language Models (LLMs)**, **Retrieval-Augmented Generation (RAG)**, **Dense Vector Embeddings**, and **NetworkX Directed Citation Graph Topology**.

---

## 👥 Project Information & Authors

* **Course**: SWE1017 - Natural Language Processing
* **Degree**: M.Tech Integrated - Business Analytics
* **Institution**: School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology
* **Project Guide**: Dr. Manjula V
* **Team Members**:
  * **V. Archana** (Reg No: `23MIA1149`)
  * **Y. Lahari** (Reg No: `23MIA1129`)
  * **M. Manudeep Reddy** (Reg No: `23MIA1158`)

---

## 🌟 Key Features & Pipeline Capabilities

### 1. Multi-Modal Research Manuscript Ingestion
* **Structured Manual Form**: Direct entry of Title, Abstract, Keywords, and Objectives for any discipline (AI, Medicine, Robotics, Physics, Energy, etc.).
* **Single PDF Document Parsing**: Extracts Title, Abstract, Keywords, and Objectives directly from uploaded research papers using `pypdf`.
* **Batch Folder Ingestion**: Upload multiple PDFs simultaneously or specify a local directory path (`D:\my_papers\`) to parse and evaluate target manuscripts.
* **Pre-Loaded Example Presets**: One-click topic autofill covering diverse academic domains.

### 2. Live Federated Multi-Source Literature Retrieval
* **Dynamic Multi-Source APIs**: Queries **OpenAlex**, **Semantic Scholar**, and **Crossref Works** in real time across hundreds of millions of indexed papers.
* **Dynamic Query Generation**: Generates targeted scientific keyword combinations from the proposal's title, abstract, and objectives.
* **Automated Deduplication & Fallback**: Deduplicates literature by normalized DOIs and fuzzy title matching. Includes a transparent offline peer-reviewed benchmark cache when offline.

### 3. Dense Vector Similarity Space
* **Sentence-Transformers**: Embeds manuscripts using `all-MiniLM-L6-v2` into a dense 384-dimensional conceptual vector space.
* **Cosine Similarity Ranking**: Computes nearest-neighbor distance and semantic overlap with candidate prior art.
* **Fallback Resilience**: Dual-mode architecture seamlessly falls back to N-gram Vector Space Models if PyTorch is unavailable.

### 4. Grounded RAG Comparative Reasoning
* **Evidence-Chunked Passage Units**: Splits retrieved literature into indexed evidentiary chunks.
* **Anti-Hallucination Grounding**: Compares candidate claims against retrieved text passages, requiring explicit literature citations. Supports Google Gemini API and deterministic local synthesis.

### 5. 5-Dimensional Research Gap Classifier
* Evaluates research across five orthogonal axes:
  1. **Objectives & Hypotheses**
  2. **Methodology & Algorithmic Design**
  3. **Datasets, Scale & Benchmarks**
  4. **Target Application Domain**
  5. **Core Scientific Contributions**
* Classifies each axis into **`[POTENTIAL GAP]`** vs **`[VERIFIED OVERLAP]`**.

### 6. NetworkX Citation Graph Topology
* **Directed Ego-Network**: Constructs directed citation networks representing reference flows and landmark papers.
* **Topological Metrics**: Computes graph density, in-degree centrality, citation bridge orthogonality, and component bridging.
* **Interactive Visualization**: Renders 2D interactive network graphs via **Plotly**.

### 7. Mathematically Bounded Composite Novelty Scoring ($0\text{–}100$)
$$S = 100 \times \left[ 0.35(1 - \text{MaxSim}) + 0.20(1 - \text{MeanTopK}) + 0.30\text{GapScore} + 0.15\text{CitationScore} \right]$$
* **MaxSim**: Cosine similarity to the closest retrieved prior art.
* **MeanTopK**: Neighborhood divergence across top-$K$ literature.
* **GapScore**: Normalized score from the 5-dimensional gap analysis ($\text{GapScore} = \text{Gaps} / 5$).
* **CitationScore**: Network orthogonality and citation bridge novelty.

### 8. Quantitative Performance & Academic Evaluation Suite (Tab 7)
* **Information Retrieval (IR) Metrics**:
  * Precision@1, Precision@3, Precision@5, Precision@10
  * Recall@5, Recall@10
  * Mean Reciprocal Rank (MRR)
  * Mean Average Precision (MAP)
  * Normalized Discounted Cumulative Gain (NDCG@5, NDCG@10)
* **RAG Grounding & Quality Metrics**:
  * Faithfulness Score (%), Hallucination Rate (%), Context Precision, Context Recall, Citation Traceability Rate (%).
* **System Latency Profiling**:
  * Stage-wise latencies (Retrieval, Embedding, RAG, Gap Analysis, Citation Graph, Scoring).
  * System throughput (papers/sec) and RAM RSS footprint (MB).
* **Sensitivity & Stability Analysis**:
  * Novelty score stability index under $\pm 10\%$ and $\pm 20\%$ weight perturbations.
* **Peer-Reviewed Benchmark Suite ($N=10$)**:
  * Pearson linear correlation: $r = 0.996$
  * Spearman rank correlation: $\rho = 0.988$
  * Mean Absolute Error: $\text{MAE} = 3.20$ points
  * 4-Tier Confusion Matrix Classification Accuracy: $90.0\%$

### 9. Publication-Ready Report Export (Tab 8)
* Generates downloadable **Markdown (`.md`)** and **JSON (`.json`)** reports containing complete assessment breakdowns, evidence passages, citation graphs, and performance metrics.
* Complete Review III project report available in **Microsoft Word (`.docx`)** format in `docs/`.

---

## 🗂️ Project Directory Structure

```text
research-novelty-assessor/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Pydantic schemas, scoring weights, API configs
│   ├── input_parser.py           # PDF extraction, text cleaning, section parser
│   ├── retrieval.py              # OpenAlex, Semantic Scholar & Crossref APIs
│   ├── embeddings.py             # SentenceTransformers & Cosine similarity
│   ├── rag_engine.py             # Grounded passage retrieval & comparative reasoning
│   ├── gap_analyzer.py           # 5-dimension gap detection (potential vs verified)
│   ├── citation_graph.py         # NetworkX citation graph & Plotly visualization
│   ├── novelty_scorer.py         # 0-100 mathematical scoring & confidence metrics
│   ├── performance_metrics.py    # Publication IR, RAG, Latency & Benchmark suite
│   ├── report_generator.py       # Markdown and JSON report exporters
│   ├── pipeline.py               # Unified end-to-end evaluation orchestrator
│   ├── api.py                    # FastAPI REST backend endpoints
│   └── main_app.py               # Streamlit interactive 8-tab web dashboard
├── docs/
│   ├── Review_III_Automatic_Research_Novelty_Assessment_Report.docx  # Complete updated Word report
│   ├── Review_III_Project_Report.md                                 # Full Markdown report
│   ├── Review_III_Automatic_Research_Novelty_Assessment_Report_ORIGINAL_BACKUP.docx
│   └── Review_III_Project_Report_ORIGINAL_BACKUP.md
├── tests/
│   ├── test_input_parser.py      # PDF & manual input tests
│   ├── test_retrieval.py         # OpenAlex parsing & fallback tests
│   ├── test_similarity.py        # Embedding & ranking tests
│   ├── test_gap_analyzer.py      # Gap classification tests
│   ├── test_citation_graph.py    # NetworkX & Plotly tests
│   ├── test_novelty_scorer.py    # Scoring formula invariants
│   ├── test_performance_metrics.py # IR, RAG, latency & benchmark validation tests
│   └── test_two_new_topics.py    # Multi-domain dynamic evaluation tests
├── sample_data/
│   └── benchmarks.json           # Benchmark papers and demo proposals
├── requirements.txt              # Dependency specifications
├── .env.example                  # Environment configuration template
├── Review_III_Automatic_Research_Novelty_Assessment_Report.docx     # Root Word document
└── README.md                     # Documentation
```

---

## 💻 Quickstart Installation & Setup

### 1. Open Terminal and Navigate to Project
```powershell
cd D:\research-novelty-assessor
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. (Optional) Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
*You can configure optional Gemini or Semantic Scholar API keys in `.env`. By default, the application runs fully out-of-the-box using the public OpenAlex API and local CPU inference.*

---

## 🚀 Running the Application

### 1. Interactive Streamlit Web Dashboard
Run the following command in PowerShell:
```powershell
cd D:\research-novelty-assessor
streamlit run app\main_app.py
```
Open your web browser at:
```text
http://localhost:8501
```

#### Dashboard Navigation (8 Dedicated Tabs):
1. **📝 1. Research Input**: Enter research proposal details, upload PDF, or scan batch folders.
2. **📊 2. Novelty Scorecard**: Overall score ($0\text{–}100$), executive summary, and multi-source stats.
3. **📚 3. Literature Similarity**: Prior-art ranking, similarity score bars, and key term overlaps.
4. **🔍 4. RAG Evidence**: Grounded comparative critique across objectives and methodologies.
5. **🎯 5. Gap & Contributions**: 5-dimensional breakdown (`[POTENTIAL GAP]` vs `[VERIFIED OVERLAP]`).
6. **🕸️ 6. Citation Network**: Interactive NetworkX 2D citation graph with landmark nodes.
7. **📈 7. Performance & Evaluation Metrics**: Empirical IR metrics, RAG faithfulness, latency, and benchmark correlation.
8. **📄 8. Report & Export**: Download full Markdown or JSON assessment reports.

### 2. FastAPI REST Backend (Optional)
```powershell
uvicorn app.api:app --reload --port 8000
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## 🧪 Running Automated Unit Tests

To run the complete test suite across all modules:
```powershell
$env:PYTHONPATH = "D:\research-novelty-assessor"
python -m pytest tests -v
```

All 25 automated unit tests cover:
* PDF and text parsing
* OpenAlex / Semantic Scholar / Crossref retrieval
* Sentence-Transformers embeddings
* 5D research gap extraction
* NetworkX citation graphs
* Novelty scoring formula bounds ($0\text{–}100$)
* IR ranking metrics (Precision@K, MAP, NDCG, MRR)
* RAG faithfulness and evidence traceability
* Peer-review benchmark suite validation

---

## 📄 Project Reports & Documentation

The complete academic project reports for Review III are located in:
* **Microsoft Word Document**: [`Review_III_Automatic_Research_Novelty_Assessment_Report.docx`](docs/Review_III_Automatic_Research_Novelty_Assessment_Report.docx)
* **Markdown Document**: [`Review_III_Project_Report.md`](docs/Review_III_Project_Report.md)

---

## ⚖️ Academic Disclaimer
The Potential Novelty Score ($0\text{–}100$) and associated gap classifications generated by this system are automated, evidence-grounded heuristic assessments intended for research discovery and peer-review decision support. They do not constitute conclusive legal determinations of patentability or intellectual property priority.
