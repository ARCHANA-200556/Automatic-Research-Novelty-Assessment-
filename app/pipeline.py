"""
End-to-End Novelty Assessment Pipeline Orchestrator.
Unifies multi-source retrieval (OpenAlex, Semantic Scholar, Crossref),
sentence-transformers embeddings, grounded RAG comparative reasoning,
dynamic research gap analysis, and citation network construction.
"""

import logging
import time
from typing import Tuple, Dict, Any, Optional
import networkx as nx
import plotly.graph_objects as go

from app.config import ResearchInput, NoveltyAssessmentResult
from app.retrieval import retrieve_relevant_literature
from app.embeddings import compute_semantic_similarities
from app.rag_engine import retrieve_grounded_evidence, generate_rag_comparative_critique
from app.gap_analyzer import analyze_multidimensional_research_gaps
from app.citation_graph import build_citation_network, create_plotly_network_figure
from app.novelty_scorer import compute_novelty_assessment
from app.performance_metrics import build_performance_metrics_summary

logger = logging.getLogger(__name__)


def run_full_novelty_assessment(
    proposal: ResearchInput,
    limit_papers: int = 10,
    force_fallback: bool = False,
    progress_callback = None
) -> Tuple[NoveltyAssessmentResult, nx.DiGraph, go.Figure]:
    """
    Executes the comprehensive automated novelty assessment workflow for ANY research topic.
    Measures stage latencies and generates academic performance & validation metrics.
    Optional progress_callback(step: int, total_steps: int, message: str) updates UI.
    """
    def update_progress(step: int, msg: str):
        if progress_callback:
            progress_callback(step, 6, msg)

    stage_latencies: Dict[str, float] = {}

    # 1. Multi-Source Literature Retrieval
    update_progress(1, "Searching OpenAlex, Semantic Scholar, and Crossref APIs...")
    t0 = time.perf_counter()
    papers, ret_status = retrieve_relevant_literature(
        query_title=proposal.title,
        keywords=proposal.keywords,
        limit=limit_papers,
        force_fallback=force_fallback,
        abstract=proposal.abstract,
        objectives=proposal.objectives
    )
    stage_latencies["retrieval"] = (time.perf_counter() - t0) * 1000.0

    if not papers:
        diag = ret_status.get("diagnostics") or f"No peer-reviewed literature found for '{proposal.title}'."
        raise ValueError(diag)

    # 2. Semantic Embedding & Similarity Analysis
    update_progress(2, f"Embedding {len(papers)} candidate papers and computing cosine similarity...")
    t0 = time.perf_counter()
    proposal_repr = f"{proposal.title}. {proposal.abstract} " + " ".join(proposal.objectives)
    ranked_papers, emb_engine_desc = compute_semantic_similarities(proposal_repr, papers)
    stage_latencies["embedding"] = (time.perf_counter() - t0) * 1000.0

    # 3. Grounded RAG Evidence Extraction & Comparative Synthesis
    update_progress(3, "Extracting evidence passages and comparing methodologies and aims...")
    t0 = time.perf_counter()
    evidence_chunks = retrieve_grounded_evidence(proposal, ranked_papers, top_passages_count=10)
    rag_critique, llm_source_used = generate_rag_comparative_critique(
        proposal, ranked_papers, evidence_chunks
    )
    stage_latencies["rag_synthesis"] = (time.perf_counter() - t0) * 1000.0

    # 4. Dynamic Multi-Dimensional Research Gap Analysis
    update_progress(4, "Classifying research gaps across 5 dimensions (Objectives, Methods, Data, Domain, Contributions)...")
    t0 = time.perf_counter()
    gap_items, gap_score = analyze_multidimensional_research_gaps(proposal, ranked_papers)
    stage_latencies["gap_analysis"] = (time.perf_counter() - t0) * 1000.0

    # 5. Citation Graph Analysis & Structural Centrality
    update_progress(5, "Constructing NetworkX citation graph and mapping topology...")
    t0 = time.perf_counter()
    G, citation_metrics = build_citation_network(proposal, ranked_papers)
    fig_network = create_plotly_network_figure(G)
    stage_latencies["graph_construction"] = (time.perf_counter() - t0) * 1000.0

    # 6. Novelty Scoring & Consolidated Report Compilation
    update_progress(6, "Calculating mathematical novelty score and compiling explainable assessment...")
    t0 = time.perf_counter()
    
    # Pre-calculate performance metrics summary for inclusion
    perf_summary = build_performance_metrics_summary(
        proposal=proposal,
        ranked_papers=ranked_papers,
        evidence_chunks=evidence_chunks,
        gap_items=gap_items,
        gap_score=gap_score,
        citation_metrics=citation_metrics,
        rag_critique=rag_critique,
        stage_latencies=stage_latencies,
        run_benchmark=True
    )
    stage_latencies["scoring"] = (time.perf_counter() - t0) * 1000.0
    
    # Update total latency in the metrics object
    perf_summary.system_latency.scoring_latency_ms = round(stage_latencies["scoring"], 1)
    perf_summary.system_latency.total_pipeline_latency_ms = round(sum(stage_latencies.values()), 1)

    result = compute_novelty_assessment(
        proposal=proposal,
        ranked_papers=ranked_papers,
        gap_items=gap_items,
        gap_score=gap_score,
        citation_metrics=citation_metrics,
        rag_critique=rag_critique,
        is_live_retrieval=not ret_status.get("is_fallback", False),
        embedding_engine_desc=emb_engine_desc,
        llm_source_used=llm_source_used,
        retrieval_stats=ret_status,
        retrieval_diagnostics=ret_status.get("diagnostics"),
        performance_metrics=perf_summary.model_dump()
    )

    return result, G, fig_network
