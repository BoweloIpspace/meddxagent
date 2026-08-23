from unittest.mock import call, patch

from Bio import Entrez

from ddxdriver.rag_agents import _searchrag_utils
from ddxdriver.rag_agents.searchrag_standard import SearchRAGStandard
from ddxdriver.utils import OutputDict


class FakeHandle:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeSection(str):
    def __new__(cls, value: str, label: str | None = None):
        obj = str.__new__(cls, value)
        obj.attributes = {"Label": label} if label else {}
        return obj


class SequenceModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs)


def _pubmed_article(title: str, sections):
    return {
        "PubmedArticle": [
            {
                "MedlineCitation": {
                    "Article": {
                        "ArticleTitle": title,
                        "Abstract": {"AbstractText": sections},
                    }
                }
            }
        ]
    }


def test_pubmed_search_uses_free_full_text_filter_and_skips_short_abstracts(monkeypatch):
    monkeypatch.setenv("NCBI_EMAIL", "runtime@example.com")
    monkeypatch.setenv("NCBI_API_KEY", "runtime-ncbi-key")

    search_handle = FakeHandle()
    short_handle = FakeHandle()
    long_handle = FakeHandle()
    long_text = "Clinically useful appendicitis evidence. " * 8

    with (
        patch.object(_searchrag_utils.Entrez, "esearch", return_value=search_handle) as esearch,
        patch.object(
            _searchrag_utils.Entrez,
            "efetch",
            side_effect=[short_handle, long_handle],
        ) as efetch,
        patch.object(
            _searchrag_utils.Entrez,
            "read",
            side_effect=[
                {"IdList": ["100", "200"]},
                _pubmed_article("Too short", [FakeSection("brief")]),
                _pubmed_article(
                    "  Useful appendicitis study  ",
                    [FakeSection(long_text, "RESULTS")],
                ),
            ],
        ),
    ):
        results = _searchrag_utils._search_pubmed(
            "appendicitis",
            top_k=1,
            min_abstract_length=100,
        )

    assert results == [
        {
            "title": "Useful appendicitis study",
            "content": f"RESULTS: {long_text}".strip(),
        }
    ]
    esearch.assert_called_once_with(
        db="pubmed",
        term="appendicitis AND free full text[sb]",
        retmax=11,
    )
    assert efetch.call_args_list == [
        call(db="pubmed", id="100", rettype="abstract", retmode="xml"),
        call(db="pubmed", id="200", rettype="abstract", retmode="xml"),
    ]
    assert search_handle.closed is True
    assert short_handle.closed is True
    assert long_handle.closed is True
    assert Entrez.email == "runtime@example.com"
    assert Entrez.api_key == "runtime-ncbi-key"


def test_explicit_pubmed_email_overrides_environment(monkeypatch):
    monkeypatch.setenv("NCBI_EMAIL", "environment@example.com")
    monkeypatch.setenv("NCBI_API_KEY", "runtime-ncbi-key")

    assert _searchrag_utils._configure_entrez("explicit@example.com") == "explicit@example.com"
    assert Entrez.email == "explicit@example.com"
    assert Entrez.api_key == "runtime-ncbi-key"


def test_api_search_routes_pubmed_requests_to_pubmed_retriever():
    expected = [{"title": "Study", "content": "Evidence"}]
    with patch.object(_searchrag_utils, "_search_pubmed", return_value=expected) as search:
        result = _searchrag_utils.api_search("abdominal pain", top_k=2, corpus_name="PubMed")

    assert result == expected
    search.assert_called_once_with(query="abdominal pain", top_k=2)


def test_searchrag_standard_runs_keyword_retrieval_and_synthesis_without_network():
    model = SequenceModel(
        [
            '["appendicitis", "right lower quadrant pain", "neutrophilia", "ignored fourth query"]',
            "Synthesized PubMed evidence",
        ]
    )
    config = {
        "corpus_name": "PubMed",
        "top_k_search": 2,
        "max_keyword_searches": 3,
        "model": {"class_name": "fake.Model", "config": {}},
    }

    def fake_search(query: str, top_k: int, corpus_name: str):
        return [
            {
                "title": f"Evidence for {query}",
                "content": f"Retrieved PubMed abstract for {query}",
            }
        ]

    with (
        patch("ddxdriver.rag_agents.searchrag_base.init_model", return_value=model),
        patch(
            "ddxdriver.rag_agents.searchrag_base.api_search",
            side_effect=fake_search,
        ) as search,
    ):
        rag = SearchRAGStandard(config)
        result = rag(
            "Migratory abdominal pain with focal right lower quadrant tenderness",
            diagnosis_options=["Appendicitis", "Gastroenteritis"],
        )

    assert result == {OutputDict.RAG_CONTENT: "Synthesized PubMed evidence"}
    assert search.call_args_list == [
        call(query="appendicitis", top_k=2, corpus_name="PubMed"),
        call(query="right lower quadrant pain", top_k=2, corpus_name="PubMed"),
        call(query="neutrophilia", top_k=2, corpus_name="PubMed"),
    ]
    assert len(model.calls) == 2
    synthesis_call = model.calls[-1]
    assert "Evidence for appendicitis" in synthesis_call["user_prompt"]
    assert "Appendicitis" in synthesis_call["user_prompt"]
    assert synthesis_call["system_prompt"]
