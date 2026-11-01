"""
Unit tests for the Performance Metrics and Academic Evaluation Suite.
Verifies IR calculations, RAG faithfulness metrics, stage latencies,
sensitivity analysis, and benchmark validation correlations.
"""

import pytest
from app.config import (
    ResearchInput,
    PaperMetadata,
    SimilarityItem,
    GapDimensionItem,
    CitationGraphMetrics,
    PassageChunk
)
from app.performance_metrics import (
    compute_ir_metrics,
    compute_rag_quality_metrics,
    compute_classification_and_sensitivity_metrics,
    run_benchmark_evaluation_suite,
    build_performance_metrics_summary,
    IRMetrics,
    RAGQualityMetrics,
    SystemLatencyMetrics
)
from app.report_generator import export_markdown_report, export_json_report
from app.novelty_scorer import compute_novelty_assessment


@pytest.fixture
def sample_proposal():
    return ResearchInput(
        title="Decentralized Multi-Agent Swarm Exploration Using Deep Q-Networks",
        abstract="We investigate autonomous drone coordination in GPS-denied environments using policy gradient reinforcement learning.",
        keywords=["Drone swarms", "Multi-agent RL", "Collision avoidance"],
        objectives=[
            "Formulate decentralized policy gradient algorithm for inter-agent collision avoidance.",
            "Benchmark against Voronoi-based coordination protocols in obstacle fields."
        ]
    )


@pytest.fixture
def sample_ranked_papers():
    papers = [
        SimilarityItem(
            paper=PaperMetadata(
                id="W1",
                title="Deep Reinforcement Learning for Drone Swarm Navigation",
                authors=["Zhang, L.", "Wang, K."],
                year=2023,
                citation_count=45,
                doi="10.1109/TRO.2023.12345",
                url="https://doi.org/10.1109/TRO.2023.12345",
                abstract="We study multi-agent deep Q-learning for quadrotor swarms navigating dense forests.",
                source_api="OpenAlex"
            ),
            similarity_score=0.78,
            rank=1,
            key_overlap_terms=["reinforcement learning", "quadrotor swarms"]
        ),
        SimilarityItem(
            paper=PaperMetadata(
                id="W2",
                title="Decentralized Swarm Coordination in GPS-Denied Tunnels",
                authors=["Miller, R."],
                year=2022,
                citation_count=30,
                doi="10.1145/34567.89012",
                url="https://doi.org/10.1145/34567.89012",
                abstract="Investigates decentralized routing for aerial robots in subterranean environments.",
                source_api="Semantic Scholar"
            ),
            similarity_score=0.62,
            rank=2,
            key_overlap_terms=["decentralized routing", "gps-denied"]
        ),
        SimilarityItem(
            paper=PaperMetadata(
                id="W3",
                title="Voronoi Partitioning for Autonomous Mobile Robots",
                authors=["Tanaka, H."],
                year=2021,
                citation_count=18,
                doi="10.1016/j.robot.2021.05",
                url="https://doi.org/10.1016/j.robot.2021.05",
                abstract="Presents geometric Voronoi partitioning algorithms for spatial coverage.",
                source_api="Crossref"
            ),
            similarity_score=0.45,
            rank=3,
            key_overlap_terms=["voronoi partitioning", "spatial coverage"]
        ),
        SimilarityItem(
            paper=PaperMetadata(
                id="W4",
                title="Solar Photovoltaic Cell Efficiency Limits",
                authors=["Green, M."],
                year=2020,
                citation_count=120,
                abstract="Fundamental thermodynamical limits of silicon and perovskite solar cells.",
                source_api="OpenAlex"
            ),
            similarity_score=0.12,
            rank=4,
            key_overlap_terms=[]
        )
    ]
    return papers


def test_ir_metrics_computation(sample_ranked_papers):
    ir = compute_ir_metrics(sample_ranked_papers, relevance_threshold=0.35)
    
    assert isinstance(ir, IRMetrics)
    assert ir.total_evaluated == 4
    # Papers 1, 2, 3 have sim >= 0.35 (relevant). Paper 4 has 0.12 (not relevant)
    assert ir.precision_at_1 == 1.0  # W1 is relevant
    assert ir.precision_at_3 == 1.0  # W1, W2, W3 are all relevant
    assert ir.precision_at_5 == 0.75 # 3/4 = 0.75
    assert ir.mrr == 1.0  # First relevant item is at rank 1 (1/1 = 1.0)
    assert ir.recall_at_5 == 1.0  # All 3 relevant items are in top 5
    assert ir.ndcg_at_5 > 0.80  # Graded ranking is high quality


def test_rag_quality_metrics(sample_proposal, sample_ranked_papers):
    evidence_chunks = [
        PassageChunk(
            paper_id="W1",
            paper_title="Deep Reinforcement Learning for Drone Swarm Navigation",
            text="We study multi-agent deep Q-learning for quadrotor swarms navigating dense forests.",
            relevance_score=0.78,
            section="Abstract"
        ),
        PassageChunk(
            paper_id="W2",
            paper_title="Decentralized Swarm Coordination in GPS-Denied Tunnels",
            text="Investigates decentralized routing for aerial robots in subterranean environments.",
            relevance_score=0.62,
            section="Methodology"
        )
    ]

    comparisons = [
        {
            "aspect": "Multi-agent coordination protocol",
            "proposed_state": "Decentralized policy gradients",
            "prior_art_state": "Centralized Q-learning (Zhang et al., 2023, DOI: 10.1109/TRO.2023.12345)",
            "overlap_verdict": "Partial Conceptual Overlap",
            "supporting_evidence": "Referenced in Zhang et al., 2023 (10.1109/TRO.2023.12345)"
        }
    ]
    distinctive_contributions = [
        "Novel policy gradient loss penalizing inter-agent proximity."
    ]

    rag = compute_rag_quality_metrics(
        proposal=sample_proposal,
        ranked_papers=sample_ranked_papers,
        evidence_chunks=evidence_chunks,
        comparisons=comparisons,
        distinctive_contributions=distinctive_contributions
    )

    assert isinstance(rag, RAGQualityMetrics)
    assert rag.faithfulness_score >= 80.0
    assert rag.hallucination_rate <= 20.0
    assert rag.context_precision >= 80.0
    assert rag.citation_traceability_rate >= 80.0


def test_sensitivity_and_classification_metrics():
    gap_items = [
        GapDimensionItem(
            dimension="Methodological Approach",
            classification="POTENTIAL GAP (Inferred)",
            proposed_aspect="Decentralized policy gradients",
            prior_art_state="Centralized Q-learning",
            confidence=0.85,
            rationale="Existing works rely on centralized critics.",
            supporting_citations=["Zhang et al., 2023"]
        ),
        GapDimensionItem(
            dimension="Target Domain",
            classification="VERIFIED OVERLAP",
            proposed_aspect="GPS-denied environments",
            prior_art_state="Subterranean tunnels and forests",
            confidence=0.90,
            rationale="Subterranean environments previously studied.",
            supporting_citations=["Miller, 2022"]
        )
    ]

    cls_m = compute_classification_and_sensitivity_metrics(
        gap_items=gap_items,
        max_sim=0.78,
        mean_sim=0.62,
        gap_score=0.60,
        citation_ortho=0.45
    )

    assert cls_m.potential_gap_count == 1
    assert cls_m.verified_overlap_count == 1
    assert cls_m.scoring_stability_index >= 70.0
    assert cls_m.weight_sensitivity_delta_10pct >= 0.0
    assert cls_m.weight_sensitivity_delta_20pct >= 0.0


def test_benchmark_evaluation_suite():
    bench = run_benchmark_evaluation_suite()
    
    assert bench.sample_size == 10
    assert bench.pearson_correlation >= 0.90
    assert bench.spearman_rho >= 0.90
    assert bench.mae <= 8.0
    assert bench.accuracy >= 0.80
    assert len(bench.confusion_matrix) == 4


def test_performance_metrics_in_pipeline_and_report(sample_proposal, sample_ranked_papers):
    latencies = {
        "retrieval": 150.0,
        "embedding": 85.0,
        "rag_synthesis": 220.0,
        "gap_analysis": 110.0,
        "graph_construction": 45.0,
        "scoring": 15.0
    }

    gap_items = [
        GapDimensionItem(
            dimension="Methodology",
            classification="POTENTIAL GAP (Inferred)",
            proposed_aspect="Decentralized policy gradients",
            prior_art_state="Centralized Q-learning",
            confidence=0.85,
            rationale="Existing works rely on centralized critics.",
            supporting_citations=["Zhang et al., 2023"]
        )
    ]

    citation_metrics = CitationGraphMetrics(
        total_nodes=5,
        total_edges=4,
        graph_density=0.20,
        most_central_paper="Zhang et al., 2023",
        citation_orthogonality_score=0.75,
        citation_coverage_note="Verified real citation links."
    )

    rag_critique = {
        "comparisons": [
            {
                "aspect": "Coordination Algorithm",
                "proposed_state": "Decentralized PG",
                "prior_art_state": "Zhang et al., 2023 (DOI: 10.1109/TRO.2023.12345)",
                "overlap_verdict": "Potential Extension",
                "supporting_evidence": "Referenced in Zhang et al., 2023 (10.1109/TRO.2023.12345)"
            }
        ],
        "distinctive_contributions": ["Decentralized collision avoidance formulation."],
        "literature_limitations": ["Limited physical quadrotor flight experiments."]
    }

    evidence_chunks = [
        PassageChunk(
            paper_id="W1",
            paper_title="Deep Reinforcement Learning for Drone Swarm Navigation",
            text="We study multi-agent deep Q-learning for quadrotor swarms navigating dense forests.",
            relevance_score=0.78,
            section="Abstract"
        )
    ]

    summary = build_performance_metrics_summary(
        proposal=sample_proposal,
        ranked_papers=sample_ranked_papers,
        evidence_chunks=evidence_chunks,
        gap_items=gap_items,
        gap_score=0.65,
        citation_metrics=citation_metrics,
        rag_critique=rag_critique,
        stage_latencies=latencies,
        run_benchmark=True
    )

    assert summary.ir_metrics.precision_at_3 == 1.0
    assert summary.system_latency.total_pipeline_latency_ms == sum(latencies.values())
    assert summary.benchmark_evaluation is not None

    # Compute novelty assessment with performance metrics
    result = compute_novelty_assessment(
        proposal=sample_proposal,
        ranked_papers=sample_ranked_papers,
        gap_items=gap_items,
        gap_score=0.65,
        citation_metrics=citation_metrics,
        rag_critique=rag_critique,
        is_live_retrieval=True,
        embedding_engine_desc="all-MiniLM-L6-v2",
        llm_source_used="Grounded RAG",
        performance_metrics=summary.model_dump()
    )

    assert "ir_metrics" in result.performance_metrics
    assert "rag_metrics" in result.performance_metrics

    # Test report export includes performance metrics
    md_report = export_markdown_report(result)
    assert "## 10. Quantitative Performance & Evaluation Metrics" in md_report
    assert "Precision@5 (P@5)" in md_report
    assert "Faithfulness Score" in md_report
    assert "Peer-Reviewed Benchmark Validation Suite" in md_report

    json_report = export_json_report(result)
    assert "performance_metrics" in json_report


def test_ir_metrics_empty_relevant_set(sample_ranked_papers):
    # If user threshold is set above any paper's similarity
    ir = compute_ir_metrics(sample_ranked_papers, relevance_threshold=0.99)
    assert ir.precision_at_1 == 0.0
    assert ir.precision_at_5 == 0.0
    assert ir.recall_at_5 == 0.0  # Mathematically 0, not 1.0
    assert ir.mrr == 0.0
    assert ir.map_score == 0.0
    assert ir.ndcg_at_5 == 0.0  # Mathematically 0, not 1.0
