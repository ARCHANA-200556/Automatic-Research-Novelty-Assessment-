"""
Multi-Source Academic Literature Retrieval and Deduplication Engine.
Dynamically searches OpenAlex, Semantic Scholar, and Crossref APIs for any research topic.
Deduplicates across sources, normalizes metadata, and tracks source retrieval metrics.
"""

import re
import json
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict, Any, Set
from app.config import (
    PaperMetadata,
    OPENALEX_API_BASE,
    SEMANTIC_SCHOLAR_API_BASE,
    CROSSREF_API_BASE,
    SEMANTIC_SCHOLAR_API_KEY,
    BASE_DIR
)
from app.input_parser import generate_academic_search_queries, clean_text

logger = logging.getLogger(__name__)

# In-memory query cache to avoid repeated HTTP calls and optimize pipeline latency
_API_CACHE: Dict[str, Tuple[List[PaperMetadata], Optional[str]]] = {}


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstructs original abstract text from OpenAlex's inverted index representation.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    try:
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
    except Exception:
        return ""


def clean_crossref_abstract(raw_abstract: Optional[str]) -> str:
    """Strips XML/HTML tags (like <jats:p>) from Crossref abstract fields."""
    if not raw_abstract:
        return ""
    # Remove XML/HTML tags
    text = re.sub(r'<[^>]+>', ' ', raw_abstract)
    return clean_text(text)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    """Normalizes a DOI string for consistent deduplication."""
    if not doi:
        return None
    d = doi.strip().lower()
    # Remove url prefixes
    d = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', d)
    return d.strip() or None


def normalize_title_for_dedup(title: str) -> str:
    """Strips punctuation and spaces to create a canonical title signature."""
    if not title:
        return ""
    clean = re.sub(r'[^a-zA-Z0-9]', '', title.lower())
    return clean[:80]


# --- 1. OPENALEX API ---
def search_openalex(
    query: str,
    limit: int = 15,
    timeout: int = 6
) -> Tuple[List[PaperMetadata], Optional[str]]:
    """
    Queries the OpenAlex API for works matching the given search query.
    Fetches up to `limit` items across pages with in-memory caching.
    """
    clean_q = " ".join(query.strip().split())
    cache_key = f"oa_{clean_q.lower()}_{limit}"
    if cache_key in _API_CACHE:
        cached_p, cached_e = _API_CACHE[cache_key]
        return [p.model_copy() for p in cached_p], cached_e

    url = f"{OPENALEX_API_BASE}/works"
    papers: List[PaperMetadata] = []
    page = 1
    per_page = min(limit, 25)

    try:
        while len(papers) < limit and page <= 2:
            params = {
                "search": clean_q,
                "per-page": per_page,
                "page": page,
                "mailto": "research.novelty.assessor@gmail.com"
            }
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code != 200:
                err_msg = f"OpenAlex returned HTTP status {resp.status_code}"
                _API_CACHE[cache_key] = (papers, err_msg)
                return papers, err_msg

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for item in results:
                title = item.get("display_name") or item.get("title") or "Untitled"
                
                # Authors
                authors = []
                for authorship in item.get("authorships", []):
                    author_name = authorship.get("author", {}).get("display_name")
                    if author_name:
                        authors.append(author_name)

                # Abstract
                abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))

                # Venue
                venue = None
                primary_loc = item.get("primary_location") or {}
                source = primary_loc.get("source") or {}
                if source:
                    venue = source.get("display_name")

                # Referenced works (used for citation graph)
                refs = [ref.split("/")[-1] for ref in item.get("referenced_works", [])]

                paper_id = item.get("id", "").split("/")[-1]
                raw_doi = item.get("doi")
                norm_d = normalize_doi(raw_doi)
                doi_link = f"https://doi.org/{norm_d}" if norm_d else None
                url_link = doi_link or item.get("id") or ""

                papers.append(PaperMetadata(
                    id=paper_id if paper_id else f"OA_{len(papers)}",
                    title=title,
                    authors=authors[:6],
                    year=item.get("publication_year"),
                    venue=venue,
                    abstract=abstract,
                    doi=doi_link,
                    url=url_link,
                    citation_count=item.get("cited_by_count", 0),
                    referenced_works=refs[:30],
                    source_api="OpenAlex",
                    is_fallback=False
                ))

            if len(results) < per_page:
                break
            page += 1

        _API_CACHE[cache_key] = (papers, None)
        return papers, None

    except requests.exceptions.RequestException as e:
        err_msg = f"OpenAlex network error: {str(e)}"
        _API_CACHE[cache_key] = (papers, err_msg)
        return papers, err_msg
    except Exception as e:
        err_msg = f"OpenAlex parsing error: {str(e)}"
        _API_CACHE[cache_key] = (papers, err_msg)
        return papers, err_msg


# --- 2. SEMANTIC SCHOLAR API ---
def search_semantic_scholar(
    query: str,
    limit: int = 15,
    timeout: int = 6
) -> Tuple[List[PaperMetadata], Optional[str]]:
    """
    Queries the Semantic Scholar Academic Graph API with in-memory caching.
    """
    clean_q = " ".join(query.strip().split())
    cache_key = f"s2_{clean_q.lower()}_{limit}"
    if cache_key in _API_CACHE:
        cached_p, cached_e = _API_CACHE[cache_key]
        return [p.model_copy() for p in cached_p], cached_e

    url = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
    params = {
        "query": clean_q,
        "limit": min(limit, 20),
        "fields": "paperId,title,abstract,authors,year,venue,citationCount,references,externalIds,url"
    }
    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            err_msg = f"Semantic Scholar returned status {resp.status_code}"
            _API_CACHE[cache_key] = ([], err_msg)
            return [], err_msg

        data = resp.json()
        results = data.get("data", [])
        papers = []

        for item in results:
            paper_id = item.get("paperId", "")
            title = item.get("title", "Untitled")
            abstract = item.get("abstract") or ""
            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            refs = [r.get("paperId", "") for r in item.get("references", []) if r.get("paperId")]
            ext_ids = item.get("externalIds") or {}
            raw_doi = ext_ids.get("DOI")
            norm_d = normalize_doi(raw_doi)
            doi_link = f"https://doi.org/{norm_d}" if norm_d else None

            papers.append(PaperMetadata(
                id=paper_id if paper_id else f"S2_{len(papers)}",
                title=title,
                authors=authors[:6],
                year=item.get("year"),
                venue=item.get("venue"),
                abstract=abstract,
                doi=doi_link,
                url=doi_link or item.get("url", ""),
                citation_count=item.get("citationCount", 0),
                referenced_works=refs[:30],
                source_api="Semantic Scholar",
                is_fallback=False
            ))

        _API_CACHE[cache_key] = (papers, None)
        return papers, None

    except requests.exceptions.RequestException as e:
        err_msg = f"Semantic Scholar network error: {str(e)}"
        _API_CACHE[cache_key] = ([], err_msg)
        return [], err_msg
    except Exception as e:
        err_msg = f"Semantic Scholar parsing error: {str(e)}"
        _API_CACHE[cache_key] = ([], err_msg)
        return [], err_msg


# --- 3. CROSSREF API ---
def search_crossref(
    query: str,
    limit: int = 15,
    timeout: int = 6
) -> Tuple[List[PaperMetadata], Optional[str]]:
    """
    Queries the Crossref REST API (multi-disciplinary coverage) with caching.
    """
    clean_q = " ".join(query.strip().split())
    cache_key = f"cr_{clean_q.lower()}_{limit}"
    if cache_key in _API_CACHE:
        cached_p, cached_e = _API_CACHE[cache_key]
        return [p.model_copy() for p in cached_p], cached_e

    url = f"{CROSSREF_API_BASE}/works"
    params = {
        "query": clean_q,
        "rows": min(limit, 20),
        "mailto": "research.novelty.assessor@gmail.com"
    }

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code != 200:
            err_msg = f"Crossref returned status {resp.status_code}"
            _API_CACHE[cache_key] = ([], err_msg)
            return [], err_msg

        data = resp.json()
        items = data.get("message", {}).get("items", [])
        papers = []

        for item in items:
            raw_title = item.get("title", ["Untitled"])
            title = raw_title[0] if isinstance(raw_title, list) and raw_title else str(raw_title)

            # Authors
            authors = []
            for a in item.get("author", []):
                given = a.get("given", "")
                family = a.get("family", "")
                full = f"{given} {family}".strip()
                if full:
                    authors.append(full)

            # Year
            issued = item.get("issued", {}).get("date-parts", [[]])
            year = issued[0][0] if (issued and issued[0]) else None

            # Abstract
            abstract = clean_crossref_abstract(item.get("abstract", ""))

            # Venue
            container = item.get("container-title", [])
            venue = container[0] if (isinstance(container, list) and container) else None

            raw_doi = item.get("DOI")
            norm_d = normalize_doi(raw_doi)
            doi_link = f"https://doi.org/{norm_d}" if norm_d else None
            url_link = item.get("URL") or doi_link or ""
            cites = item.get("is-referenced-by-count", 0)

            # Reference DOIs if available
            refs = []
            for ref in item.get("reference", []):
                ref_doi = ref.get("DOI")
                if ref_doi:
                    refs.append(normalize_doi(ref_doi))

            papers.append(PaperMetadata(
                id=norm_d if norm_d else f"CR_{len(papers)}",
                title=title,
                authors=authors[:6],
                year=year,
                venue=venue,
                abstract=abstract,
                doi=doi_link,
                url=url_link,
                citation_count=cites,
                referenced_works=[r for r in refs if r][:20],
                source_api="Crossref",
                is_fallback=False
            ))

        _API_CACHE[cache_key] = (papers, None)
        return papers, None

    except requests.exceptions.RequestException as e:
        err_msg = f"Crossref network error: {str(e)}"
        _API_CACHE[cache_key] = ([], err_msg)
        return [], err_msg
    except Exception as e:
        err_msg = f"Crossref parsing error: {str(e)}"
        _API_CACHE[cache_key] = ([], err_msg)
        return [], err_msg


# --- 4. DEDUPLICATION & MERGE ---
def deduplicate_and_merge_papers(
    paper_lists: List[List[PaperMetadata]],
    limit: int = 15
) -> List[PaperMetadata]:
    """
    Merges papers from multiple sources, deduplicating by normalized DOI
    or normalized title signature. Preserves papers with the most complete abstracts.
    """
    seen_dois: Set[str] = set()
    seen_titles: Set[str] = set()
    merged: List[PaperMetadata] = []

    # Flatten all papers in order
    all_papers: List[PaperMetadata] = []
    for plist in paper_lists:
        all_papers.extend(plist)

    for p in all_papers:
        # Check title validity
        if not p.title or len(p.title.strip()) < 5:
            continue

        norm_d = normalize_doi(p.doi)
        norm_t = normalize_title_for_dedup(p.title)

        is_duplicate = False
        if norm_d and norm_d in seen_dois:
            is_duplicate = True
        elif norm_t and norm_t in seen_titles:
            is_duplicate = True

        if is_duplicate:
            # If current duplicate has a substantially longer abstract than what we stored, update it
            for existing in merged:
                if (norm_d and normalize_doi(existing.doi) == norm_d) or (norm_t and normalize_title_for_dedup(existing.title) == norm_t):
                    if len(p.abstract) > len(existing.abstract):
                        existing.abstract = p.abstract
                    if len(p.referenced_works) > len(existing.referenced_works):
                        existing.referenced_works = p.referenced_works
                    if p.citation_count > existing.citation_count:
                        existing.citation_count = p.citation_count
                    break
            continue

        if norm_d:
            seen_dois.add(norm_d)
        if norm_t:
            seen_titles.add(norm_t)

        merged.append(p)

    # Sort to prioritize papers that contain abstracts and higher citation counts
    merged.sort(key=lambda x: (1 if len(x.abstract) > 40 else 0, x.citation_count), reverse=True)
    return merged[:limit]


# --- 5. BENCHMARK FALLBACK (EXPLICIT DEMO ONLY) ---
def load_fallback_papers() -> List[PaperMetadata]:
    """Loads curated benchmark literature for explicit offline demo mode."""
    benchmark_file = BASE_DIR / "sample_data" / "benchmarks.json"
    if not benchmark_file.exists():
        return []

    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            papers_data = data.get("benchmarks", [])
            papers = []
            for p in papers_data:
                papers.append(PaperMetadata(
                    id=p.get("id"),
                    title=p.get("title"),
                    authors=p.get("authors", []),
                    year=p.get("year"),
                    venue=p.get("venue"),
                    abstract=p.get("abstract", ""),
                    doi=p.get("doi"),
                    url=p.get("doi"),
                    citation_count=p.get("citation_count", 0),
                    referenced_works=p.get("referenced_works", []),
                    source_api="Offline Benchmark",
                    is_fallback=True
                ))
            return papers
    except Exception as e:
        logger.error(f"Error loading offline benchmark papers: {e}")
        return []


# --- 6. UNIFIED DYNAMIC RETRIEVAL ENTRYPOINT ---
def retrieve_relevant_literature(
    query_title: str,
    keywords: List[str] = None,
    limit: int = 10,
    force_fallback: bool = False,
    abstract: str = "",
    objectives: List[str] = None
) -> Tuple[List[PaperMetadata], Dict[str, Any]]:
    """
    Dynamically retrieves relevant peer-reviewed literature for ANY research topic.
    Queries OpenAlex, Semantic Scholar, and Crossref.
    Deduplicates results and returns detailed multi-source diagnostics.
    """
    stats = {
        "openalex_count": 0,
        "semantic_scholar_count": 0,
        "crossref_count": 0,
        "unique_deduplicated": 0,
        "analyzed_count": 0,
        "is_fallback": False,
        "diagnostics": None,
        "queries_attempted": []
    }

    # Explicit offline benchmark mode
    if force_fallback:
        benchmark_papers = load_fallback_papers()[:limit]
        stats["is_fallback"] = True
        stats["analyzed_count"] = len(benchmark_papers)
        stats["diagnostics"] = "User explicitly requested Offline Benchmark Demo mode."
        return benchmark_papers, stats

    # 1. Generate adaptive queries for the user's specific topic
    from app.config import ResearchInput
    temp_proposal = ResearchInput(
        title=query_title,
        abstract=abstract or "",
        keywords=keywords or [],
        objectives=objectives or []
    )
    search_queries = generate_academic_search_queries(temp_proposal)
    stats["queries_attempted"] = search_queries

    openalex_results: List[PaperMetadata] = []
    s2_results: List[PaperMetadata] = []
    crossref_results: List[PaperMetadata] = []
    api_errors: List[str] = []

    # 2. Parallel Querying across OpenAlex, Semantic Scholar, and Crossref
    def _fetch_source(source_name, queries, fetch_fn, target_limit):
        res = []
        err = None
        for q in queries:
            papers, e = fetch_fn(q, limit=target_limit)
            if e and not err:
                err = f"{source_name}: {e}"
            if papers:
                res.extend(papers)
                if len(res) >= target_limit:
                    break
        return res, err

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            f_oa = executor.submit(_fetch_source, "OpenAlex", search_queries[:3], search_openalex, limit)
            f_s2 = executor.submit(_fetch_source, "Semantic Scholar", search_queries[:2], search_semantic_scholar, limit)
            f_cr = executor.submit(_fetch_source, "Crossref", search_queries[:2], search_crossref, limit)

            oa_papers, oa_err = f_oa.result()
            s2_papers, s2_err = f_s2.result()
            cr_papers, cr_err = f_cr.result()

            if oa_papers:
                openalex_results.extend(oa_papers)
            if oa_err:
                api_errors.append(oa_err)

            if s2_papers:
                s2_results.extend(s2_papers)
            if s2_err:
                api_errors.append(s2_err)

            if cr_papers:
                crossref_results.extend(cr_papers)
            if cr_err:
                api_errors.append(cr_err)
    except Exception as ex:
        logger.warning(f"Error in parallel retrieval: {ex}")

    stats["openalex_count"] = len(openalex_results)
    stats["semantic_scholar_count"] = len(s2_results)
    stats["crossref_count"] = len(crossref_results)

    # 3. Merge and Deduplicate across all sources
    deduped_papers = deduplicate_and_merge_papers(
        [openalex_results, s2_results, crossref_results],
        limit=limit
    )

    # 4. Automatic Query Relaxation if initial strict queries yielded zero papers
    if not deduped_papers:
        relaxed_queries = []
        if keywords:
            for kw in keywords[:3]:
                if len(kw.strip()) > 2 and kw.strip() not in search_queries:
                    relaxed_queries.append(kw.strip())
        
        # Core scientific noun pair from title
        title_nouns = [
            w for w in re.findall(r'[a-zA-Z]{4,}', query_title.lower())
            if w not in {'using', 'based', 'with', 'from', 'into', 'study', 'novel', 'approach', 'system', 'method', 'paper', 'research'}
        ]
        if len(title_nouns) >= 2:
            relaxed_q = f"{title_nouns[0]} {title_nouns[1]}"
            if relaxed_q not in search_queries and relaxed_q not in relaxed_queries:
                relaxed_queries.append(relaxed_q)

        if relaxed_queries:
            stats["queries_attempted"].extend(relaxed_queries)
            for rq in relaxed_queries:
                r_oa, _ = search_openalex(rq, limit=limit)
                if r_oa:
                    openalex_results.extend(r_oa)
                r_s2, _ = search_semantic_scholar(rq, limit=limit)
                if r_s2:
                    s2_results.extend(r_s2)
                r_cr, _ = search_crossref(rq, limit=limit)
                if r_cr:
                    crossref_results.extend(r_cr)
                if len(openalex_results) + len(s2_results) + len(crossref_results) >= limit:
                    break

            deduped_papers = deduplicate_and_merge_papers(
                [openalex_results, s2_results, crossref_results],
                limit=limit
            )
            stats["openalex_count"] = len(openalex_results)
            stats["semantic_scholar_count"] = len(s2_results)
            stats["crossref_count"] = len(crossref_results)

    # 5. Resilient Domain-Aligned Fallback if live APIs are completely unreachable or returned 0
    if not deduped_papers:
        benchmark_papers = load_fallback_papers()[:limit]
        stats["is_fallback"] = True
        stats["unique_deduplicated"] = len(benchmark_papers)
        stats["analyzed_count"] = len(benchmark_papers)
        stats["diagnostics"] = (
            f"Live academic APIs returned 0 results or were unreachable. "
            f"Activated benchmark reference literature to ensure continuous, grounded assessment."
        )
        logger.warning(stats["diagnostics"])
        return benchmark_papers, stats

    stats["unique_deduplicated"] = len(deduped_papers)
    stats["analyzed_count"] = len(deduped_papers)
    if api_errors:
        stats["diagnostics"] = "; ".join(api_errors[:2])

    return deduped_papers, stats
