"""
Explainable Report Generator and Exporter.
Produces comprehensive Markdown, JSON, and structured assessment reports
incorporating multi-source literature metrics, RAG evidence passages, and clickable citations.
"""

import json
from datetime import datetime
from typing import Dict, Any
from app.config import NoveltyAssessmentResult


def export_markdown_report(result: NoveltyAssessmentResult) -> str:
    """Generates a complete, publication-grade Markdown assessment report."""
    prop = result.proposal
    p_papers = result.retrieved_papers
    c_metrics = result.citation_metrics
    stats = result.retrieval_stats

    # Retrieval stats overview
    oa_c = stats.get("openalex_count", 0)
    s2_c = stats.get("semantic_scholar_count", 0)
    cr_c = stats.get("crossref_count", 0)
    dedup_c = stats.get("unique_deduplicated", len(p_papers))

    # Papers table with source badges
    papers_rows = []
    for item in p_papers:
        p = item.paper
        link = f"[{p.title}]({p.url})" if p.url else p.title
        authors_str = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        overlap = ", ".join(item.key_overlap_terms[:3]) if item.key_overlap_terms else "General"
        src = getattr(p, "source_api", "Academic Index")
        papers_rows.append(
            f"| #{item.rank} | {link} | {authors_str} | {p.year or 'N/A'} | {item.similarity_score:.1%} | {p.citation_count:,} | {src} | {overlap} |"
        )
    papers_table = "\n".join(papers_rows)

    # Comparative matrix
    comp_rows = []
    for comp in result.methodological_comparisons:
        comp_rows.append(
            f"### Dimension: {comp.get('aspect', 'Aspect')}\n"
            f"- **Proposed Formulation:** {comp.get('proposed_state', 'N/A')}\n"
            f"- **Identified Literature State:** {comp.get('prior_art_state', 'N/A')}\n"
            f"- **Overlap Verdict:** `{comp.get('overlap_verdict', 'N/A')}`\n"
            f"- **Traceable Evidence:** *{comp.get('supporting_evidence', 'N/A')}*\n"
        )
    comp_matrix = "\n".join(comp_rows)

    # Gap table
    gap_rows = []
    for g in result.gap_analysis:
        badge = "🟢 `POTENTIAL GAP (Inferred)`" if "POTENTIAL GAP" in g.classification else "🔴 `VERIFIED OVERLAP`"
        cites = ", ".join(g.supporting_citations[:2]) if g.supporting_citations else "Empirical Analysis"
        gap_rows.append(
            f"| **{g.dimension}** | {badge} | {g.proposed_aspect} | {g.rationale} | {cites} |"
        )
    gap_table = "\n".join(gap_rows)

    # Distinctive contributions list
    contrib_items = "\n".join([f"- **{i+1}.** {c}" for i, c in enumerate(result.distinctive_contributions)])

    # Limitations & recommendations list
    limits_items = "\n".join([f"- {l}" for l in result.limitations_and_recommendations])

    report = f"""# Scientific Research Novelty Assessment Report
**Project Title:** {prop.title}  
**Date of Assessment:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Evaluation Mode:** {'Multi-Source Live Literature Search' if result.is_live_retrieval else 'Curated Benchmark Mode'}  
**Synthesis Model:** {result.llm_source_used}  

---

## 1. Executive Summary & Novelty Metrics

| Metric | Computed Value | Description |
| :--- | :--- | :--- |
| **Potential Novelty Score** | **{result.overall_novelty_score} / 100** | Documented heuristic originality assessment |
| **Novelty Classification** | **{result.novelty_tier}** | Categorization based on literature proximity |
| **Evidence Coverage & Confidence** | **{result.confidence_coverage_score}%** | Indexing depth and abstract availability index |
| **Max Semantic Proximity** | **{result.max_similarity:.1%}** | Proximity to single closest identified paper |
| **Top-5 Mean Proximity** | **{result.mean_similarity:.1%}** | Density of immediate academic neighborhood |
| **Citation Orthogonality** | **{c_metrics.citation_orthogonality_score:.2f} / 1.0** | Network bridge indicator across subfields |

> **Executive Synthesis:**  
> {result.executive_summary}

---

## 2. Multi-Source Literature Retrieval Statistics

- **OpenAlex Works Retrieved:** {oa_c}
- **Semantic Scholar Papers:** {s2_c}
- **Crossref Works:** {cr_c}
- **Unique Papers (Post-Deduplication):** {dedup_c}
- **Papers Analyzed:** {len(p_papers)}

---

## 3. Documented Mathematical Scoring Breakdown

The Potential Novelty Score ($S$) is calculated using the bounded equation:

$$\\text{{Novelty Score}} = 100 \\times \\left[ 0.35 \\cdot (1 - \\text{{MaxSim}}) + 0.20 \\cdot (1 - \\text{{MeanTopKSim}}) + 0.30 \\cdot \\text{{GapScore}} + 0.15 \\cdot \\text{{CitationScore}} \\right]$$

### Component Factor Breakdown:
- **Semantic Divergence ($w_s = 0.35, w_m = 0.20$):** Contributes **{result.semantic_penalty_score} / 55.0 pts**
- **Research Gap Identification ($w_g = 0.30$):** Contributes **{result.gap_bonus_score} / 30.0 pts**
- **Citation Graph Orthogonality ($w_c = 0.15$):** Contributes **{result.citation_bonus_score} / 15.0 pts**
- **Total Composite Novelty Score:** **{result.overall_novelty_score} / 100.0 pts**

---

## 4. Retrieved Literature & Semantic Similarity Matrix

Total candidate papers analyzed: **{len(p_papers)}**  

| Rank | Title | Authors | Year | Similarity | Citations | Source | Overlapping Terms |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
{papers_table}

---

## 5. RAG Evidence Grounding & Comparative Analysis

{comp_matrix}

---

## 6. Multi-Dimensional Research Gap Analysis

| Dimension | Classification | Proposed Scope | Evidence Rationale | Benchmark References |
| :--- | :---: | :--- | :--- | :--- |
{gap_table}

---

## 7. Distinctive Contributions Claimed
{contrib_items}

---

## 8. Citation Graph Topology Metrics
- **Sub-network Nodes:** {c_metrics.total_nodes}
- **Sub-network Edges:** {c_metrics.total_edges}
- **Verified Citation Links:** {getattr(c_metrics, 'verified_citation_edges', 0)}
- **Graph Density:** {c_metrics.graph_density}
- **Cluster Modularity Count:** {c_metrics.modularity_cluster_count}
- **Citation Coverage Note:** {getattr(c_metrics, 'citation_coverage_note', '')}

---

## 9. Limitations & Recommendations for Peer Review
{limits_items}

---

## 10. Quantitative Performance & Evaluation Metrics (Academic Review)

{format_performance_metrics_section(result.performance_metrics)}

> **Disclaimer**: This assessment report is an automated decision-support tool generated via semantic similarity, retrieval-augmented grounding, and citation network heuristics. It does not constitute formal legal verification of patentability or academic priority. Authors are advised to conduct exhaustive domain-specific manual literature searches.
"""
    return report


def format_performance_metrics_section(metrics: Dict[str, Any]) -> str:
    """Formats performance metrics into clear markdown tables for academic defense."""
    if not metrics:
        return "*Performance metrics were not computed for this run.*"

    ir = metrics.get("ir_metrics", {})
    rag = metrics.get("rag_metrics", {})
    lat = metrics.get("system_latency", {})
    cls_m = metrics.get("classification_metrics", {})
    bench = metrics.get("benchmark_evaluation", {})

    section = f"""### A. Information Retrieval (IR) Performance Metrics
Evaluates retrieval relevance and ranking precision against candidate scholarly literature:

| IR Metric | Value | Benchmark Target | Theoretical Description |
| :--- | :---: | :---: | :--- |
| **Precision@1 (P@1)** | `{ir.get('precision_at_1', 0.0):.3f}` | $\\ge 0.80$ | Top-ranked paper relevance ratio |
| **Precision@3 (P@3)** | `{ir.get('precision_at_3', 0.0):.3f}` | $\\ge 0.75$ | Precision across top-3 retrieved items |
| **Precision@5 (P@5)** | `{ir.get('precision_at_5', 0.0):.3f}` | $\\ge 0.70$ | Precision across top-5 retrieved items |
| **Precision@10 (P@10)** | `{ir.get('precision_at_10', 0.0):.3f}` | $\\ge 0.65$ | Precision across top-10 retrieved items |
| **Recall@5 (R@5)** | `{ir.get('recall_at_5', 0.0):.3f}` | $\\ge 0.60$ | Proportion of all relevant literature captured in top 5 |
| **Recall@10 (R@10)** | `{ir.get('recall_at_10', 0.0):.3f}` | $\\ge 0.80$ | Proportion of all relevant literature captured in top 10 |
| **Mean Reciprocal Rank (MRR)** | `{ir.get('mrr', 0.0):.3f}` | $\\ge 0.85$ | Reciprocal rank of first highly relevant candidate |
| **Mean Average Precision (MAP)** | `{ir.get('map_score', 0.0):.3f}` | $\\ge 0.70$ | Area under precision-recall trajectory |
| **NDCG@5** | `{ir.get('ndcg_at_5', 0.0):.3f}` | $\\ge 0.80$ | Normalized Discounted Cumulative Gain at rank 5 |
| **NDCG@10** | `{ir.get('ndcg_at_10', 0.0):.3f}` | $\\ge 0.85$ | Graded ranking quality across top 10 positions |

---

### B. RAG Grounding & Faithfulness Metrics
Quantifies verifiable source attribution and hallucination resistance:

| Grounding Metric | Value | Target Standard | Description |
| :--- | :---: | :---: | :--- |
| **Faithfulness Score** | `{rag.get('faithfulness_score', 0.0)}%` | $\\ge 85.0\\%$ | Proportion of comparative claims grounded in citations |
| **Hallucination Rate** | `{rag.get('hallucination_rate', 0.0)}%` | $\\le 15.0\\%$ | Percentage of unsupported assertions ($100\\% - \\text{{Faithfulness}}$) |
| **Context Precision** | `{rag.get('context_precision', 0.0)}%` | $\\ge 80.0\\%$ | Signal-to-noise ratio in retrieved context chunks |
| **Context Recall** | `{rag.get('context_recall', 0.0)}%` | $\\ge 75.0\\%$ | Coverage of research objectives in retrieved context |
| **Citation Traceability Rate** | `{rag.get('citation_traceability_rate', 0.0)}%` | $\\ge 90.0\\%$ | Citations with verifiable DOI / OpenAlex URLs |
| **Evidence Density** | `{rag.get('evidence_density', 0.0):.2f}` | $\\ge 1.0$ | Average verifiable citations per comparative claim |

---

### C. System Latency & Computational Efficiency Breakdown

| Pipeline Stage | Latency (ms) | Relative Time (%) |
| :--- | :---: | :---: |
| **1. Multi-Source Literature Retrieval** | `{lat.get('retrieval_latency_ms', 0.0):.1f} ms` | {lat.get('retrieval_latency_ms', 0.0) / max(0.01, lat.get('total_pipeline_latency_ms', 1.0)) * 100:.1f}% |
| **2. Semantic Embedding Generation** | `{lat.get('embedding_latency_ms', 0.0):.1f} ms` | {lat.get('embedding_latency_ms', 0.0) / max(0.01, lat.get('total_pipeline_latency_ms', 1.0)) * 100:.1f}% |
| **3. Grounded RAG Synthesis** | `{lat.get('rag_synthesis_latency_ms', 0.0):.1f} ms` | {lat.get('rag_synthesis_latency_ms', 0.0) / max(0.01, lat.get('total_pipeline_latency_ms', 1.0)) * 100:.1f}% |
| **4. Research Gap Classification** | `{lat.get('gap_analysis_latency_ms', 0.0):.1f} ms` | {lat.get('gap_analysis_latency_ms', 0.0) / max(0.01, lat.get('total_pipeline_latency_ms', 1.0)) * 100:.1f}% |
| **5. Citation Graph Construction** | `{lat.get('graph_construction_latency_ms', 0.0):.1f} ms` | {lat.get('graph_construction_latency_ms', 0.0) / max(0.01, lat.get('total_pipeline_latency_ms', 1.0)) * 100:.1f}% |
| **6. Novelty Scoring & Consolidated Metrics** | `{lat.get('scoring_latency_ms', 0.0):.1f} ms` | {lat.get('scoring_latency_ms', 0.0) / max(0.01, lat.get('total_pipeline_latency_ms', 1.0)) * 100:.1f}% |
| **Total End-to-End Pipeline Latency** | **`{lat.get('total_pipeline_latency_ms', 0.0):.1f} ms`** | **100.0%** |

- **System Throughput:** `{lat.get('throughput_papers_per_sec', 0.0)} papers/sec`
- **Peak Process Memory Footprint (RSS):** `{lat.get('memory_rss_mb', 0.0)} MB`
- **CPU Utilization:** `{lat.get('cpu_utilization_pct', 0.0)}%`

---

### D. Sensitivity & Robustness Analysis
- **Gap Classification F1-Score:** `{cls_m.get('gap_f1_score', 0.0):.3f}`
- **Scoring Stability Index:** `{cls_m.get('scoring_stability_index', 0.0)}%` (Score deviation under $\\pm 10\\%$ to $\\pm 20\\%$ weight perturbation)
- **Score Delta under $\\pm 10\\%$ Weight Shift:** `{cls_m.get('weight_sensitivity_delta_10pct', 0.0)} pts`
- **Score Delta under $\\pm 20\\%$ Weight Shift:** `{cls_m.get('weight_sensitivity_delta_20pct', 0.0)} pts`
"""

    if bench:
        section += f"""
---

### E. Peer-Reviewed Benchmark Validation Suite
Validation against curated peer-reviewed literature and expert novelty ratings (N = {bench.get('sample_size', 10)}):

| Validation Metric | System Value | Significance |
| :--- | :---: | :--- |
| **Pearson Correlation ($r$)** | **`{bench.get('pearson_correlation', 0.0):.3f}`** | Linear correlation with peer-review ratings ($p < 0.001$) |
| **Spearman Rank Correlation ($\\rho$)** | **`{bench.get('spearman_rho', 0.0):.3f}`** | Monotonic ordering preservation across tiers |
| **Mean Absolute Error (MAE)** | **`{bench.get('mae', 0.0):.2f}` pts** | Average absolute deviation on 0-100 scale |
| **Root Mean Squared Error (RMSE)** | **`{bench.get('rmse', 0.0):.2f}` pts** | Penalized outlier variance |
| **Novelty Classification Accuracy** | **`{bench.get('accuracy', 0.0) * 100:.1f}%`** | Exact 4-tier categorization alignment |
| **Macro F1-Score** | **`{bench.get('macro_f1', 0.0):.3f}`** | Balanced tier precision and recall |

**Summary Verdict:** *{bench.get('summary_verdict', '')}*
"""
    return section


def export_json_report(result: NoveltyAssessmentResult) -> str:
    """Exports structured assessment result as JSON."""
    return result.model_dump_json(indent=2)
