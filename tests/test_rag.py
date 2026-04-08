"""
Tests for the RAG layer (src/rag/indexer.py and src/rag/chain.py).

What's happening here:
- We test the indexer: row_to_document, create_documents, create_index, query_index
- We test the chain: build_prompt, call_llm, ask
- We use a small fake DataFrame (6 rows) to keep tests fast
- For LLM tests, we mock the API call (don't actually call HuggingFace)

New concept — mocking:
When we test call_llm(), we don't want to actually call the HuggingFace API
(it's slow, needs internet, costs money, and results vary).
Instead, we "mock" the requests.post function — we replace it with a fake
that returns a predefined response. This way we test OUR code, not the API.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.indexer import (
    row_to_document,
    create_documents,
    create_index,
    query_index,
    get_collection,
    COLLECTION_NAME,
)
from src.rag.chain import (
    build_prompt,
    call_llm,
    ask,
    SYSTEM_PROMPT,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_df():
    """
    Creates a small DataFrame mimicking features.csv.

    6 rows: 2 campaigns x 3 dates.
    Includes all columns that row_to_document() expects.
    """
    data = {
        "campaign_name": [
            "facebook_tier1", "facebook_tier1", "facebook_tier1",
            "google_hot", "google_hot", "google_hot",
        ],
        "category": ["social", "social", "social", "search", "search", "search"],
        "c_date": [
            "2024-01-01", "2024-01-08", "2024-01-15",
            "2024-01-01", "2024-01-08", "2024-01-15",
        ],
        "impressions": [1000, 2000, 1500, 3000, 2500, 2000],
        "clicks": [50, 80, 60, 120, 100, 90],
        "leads": [10, 15, 12, 25, 20, 18],
        "orders": [5, 8, 6, 12, 10, 9],
        "mark_spent": [100, 200, 150, 300, 250, 200],
        "revenue": [500, 300, 450, 1200, 1000, 400],
        "ctr": [0.05, 0.04, 0.04, 0.04, 0.04, 0.045],
        "cpc": [2.0, 2.5, 2.5, 2.5, 2.5, 2.22],
        "roas": [5.0, 1.5, 3.0, 4.0, 4.0, 2.0],
        "cpl": [10.0, 13.33, 12.5, 12.0, 12.5, 11.11],
        "conversion_rate": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        "roas_wow": [np.nan, -0.70, 1.0, np.nan, 0.0, -0.50],
        "ctr_wow": [np.nan, -0.20, 0.0, np.nan, 0.0, 0.125],
        "spend_wow": [np.nan, 1.0, -0.25, np.nan, -0.167, -0.20],
        "is_anomaly": [0, 1, 0, 0, 0, 1],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_collection(sample_df, tmp_path, monkeypatch):
    """
    Creates a temporary ChromaDB collection for testing.

    We redirect CHROMA_PATH to tmp_path so we don't
    pollute the real chroma_db/ folder.
    """
    monkeypatch.setattr("src.rag.indexer.CHROMA_PATH", str(tmp_path / "test_chroma"))
    collection = create_index(sample_df)
    return collection


# ============================================================
# TESTS FOR row_to_document()
# ============================================================

class TestRowToDocument:
    """Tests for converting DataFrame rows to text documents."""

    def test_returns_string(self, sample_df):
        """Check that row_to_document returns a string."""
        row = sample_df.iloc[0]
        result = row_to_document(row)
        assert isinstance(result, str), "Should return a string"

    def test_contains_campaign_name(self, sample_df):
        """Check that the document contains the campaign name."""
        row = sample_df.iloc[0]
        result = row_to_document(row)
        assert "facebook_tier1" in result, "Should contain campaign name"

    def test_contains_category(self, sample_df):
        """Check that the document contains the category."""
        row = sample_df.iloc[0]
        result = row_to_document(row)
        assert "social" in result, "Should contain category"

    def test_contains_metrics(self, sample_df):
        """Check that the document contains key metrics."""
        row = sample_df.iloc[0]
        result = row_to_document(row)

        assert "CTR" in result, "Should mention CTR"
        assert "ROAS" in result, "Should mention ROAS"
        assert "CPC" in result, "Should mention CPC"

    def test_anomaly_flagged(self, sample_df):
        """Check that anomaly rows are labeled correctly."""
        anomaly_row = sample_df.iloc[1]
        result = row_to_document(anomaly_row)
        assert "ANOMALY" in result, "Anomaly row should be flagged"

    def test_normal_flagged(self, sample_df):
        """Check that normal rows are labeled correctly."""
        normal_row = sample_df.iloc[0]
        result = row_to_document(normal_row)
        assert "normally" in result, "Normal row should say 'normally'"

    def test_not_empty(self, sample_df):
        """Check that the document is not empty."""
        row = sample_df.iloc[0]
        result = row_to_document(row)
        assert len(result) > 50, "Document should have substantial content"


# ============================================================
# TESTS FOR create_documents()
# ============================================================

class TestCreateDocuments:
    """Tests for batch conversion of DataFrame to documents."""

    def test_returns_correct_count(self, sample_df):
        """Check that we get one document per row."""
        documents, ids, metadatas = create_documents(sample_df)

        assert len(documents) == len(sample_df), \
            f"Expected {len(sample_df)} documents, got {len(documents)}"
        assert len(ids) == len(sample_df), \
            f"Expected {len(sample_df)} IDs, got {len(ids)}"
        assert len(metadatas) == len(sample_df), \
            f"Expected {len(sample_df)} metadatas, got {len(metadatas)}"

    def test_ids_are_unique(self, sample_df):
        """Check that all document IDs are unique."""
        documents, ids, metadatas = create_documents(sample_df)

        assert len(set(ids)) == len(ids), "Document IDs should be unique"

    def test_metadata_has_required_keys(self, sample_df):
        """Check that metadata contains campaign_name, category, is_anomaly."""
        documents, ids, metadatas = create_documents(sample_df)

        for meta in metadatas:
            assert "campaign_name" in meta, "Metadata missing 'campaign_name'"
            assert "category" in meta, "Metadata missing 'category'"
            assert "is_anomaly" in meta, "Metadata missing 'is_anomaly'"

    def test_all_documents_are_strings(self, sample_df):
        """Check that all documents are non-empty strings."""
        documents, ids, metadatas = create_documents(sample_df)

        for doc in documents:
            assert isinstance(doc, str), "Each document should be a string"
            assert len(doc) > 0, "Documents should not be empty"


# ============================================================
# TESTS FOR create_index() and query_index()
# ============================================================

class TestIndex:
    """
    Tests for ChromaDB index creation and querying.

    These tests use the sample_collection fixture which
    creates a real (but temporary) ChromaDB collection.
    """

    def test_collection_has_correct_count(self, sample_collection):
        """Check that all documents were indexed."""
        assert sample_collection.count() == 6, \
            f"Expected 6 documents in collection, got {sample_collection.count()}"

    def test_query_returns_results(self, sample_df, tmp_path, monkeypatch):
        """Check that querying the index returns results."""
        monkeypatch.setattr("src.rag.indexer.CHROMA_PATH", str(tmp_path / "test_chroma"))
        create_index(sample_df)

        results = query_index("worst ROAS campaign", top_k=3)

        assert "documents" in results, "Results should have 'documents'"
        assert "metadatas" in results, "Results should have 'metadatas'"
        assert "distances" in results, "Results should have 'distances'"
        assert "ids" in results, "Results should have 'ids'"

    def test_query_returns_correct_count(self, sample_df, tmp_path, monkeypatch):
        """Check that query returns the requested number of results."""
        monkeypatch.setattr("src.rag.indexer.CHROMA_PATH", str(tmp_path / "test_chroma"))
        create_index(sample_df)

        results = query_index("anomalies", top_k=3)
        assert len(results["documents"]) == 3, \
            f"Expected 3 results, got {len(results['documents'])}"

    def test_query_distances_are_sorted(self, sample_df, tmp_path, monkeypatch):
        """
        Check that results are sorted by distance (most similar first).

        Lower distance = more similar to the query.
        """
        monkeypatch.setattr("src.rag.indexer.CHROMA_PATH", str(tmp_path / "test_chroma"))
        create_index(sample_df)

        results = query_index("social media campaign performance", top_k=3)
        distances = results["distances"]

        assert distances == sorted(distances), \
            "Results should be sorted by distance (ascending)"


# ============================================================
# TESTS FOR build_prompt()
# ============================================================

class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_returns_list_of_messages(self):
        """Check that build_prompt returns a list of message dicts."""
        context = ["Campaign facebook on 2024-01-01: ROAS 5.0"]
        result = build_prompt("What is the best campaign?", context)

        assert isinstance(result, list), "Should return a list"
        assert len(result) >= 2, "Should have at least system + user messages"

    def test_has_system_message(self):
        """Check that the first message is a system message."""
        context = ["Campaign data here"]
        result = build_prompt("test question", context)

        assert result[0]["role"] == "system", "First message should be system"
        assert len(result[0]["content"]) > 0, "System message should not be empty"

    def test_has_user_message_with_question(self):
        """Check that the user message contains the question."""
        context = ["Campaign data here"]
        question = "Which campaign has the best ROAS?"
        result = build_prompt(question, context)

        user_msg = result[1]
        assert user_msg["role"] == "user", "Second message should be user"
        assert question in user_msg["content"], \
            "User message should contain the question"

    def test_has_context_in_user_message(self):
        """Check that the context data appears in the user message."""
        context = ["Campaign facebook_tier1: ROAS 5.0"]
        result = build_prompt("test question", context)

        user_content = result[1]["content"]
        assert "facebook_tier1" in user_content, \
            "User message should contain context data"


# ============================================================
# TESTS FOR call_llm()
# ============================================================

class TestCallLLM:
    """
    Tests for the LLM API call.

    We use unittest.mock.patch to replace requests.post
    with a fake function that returns a predefined response.
    This way we don't actually call the API during tests.

    How mocking works:
    @patch("src.rag.chain.requests.post") tells Python:
    "When chain.py calls requests.post(), use my fake instead."
    The fake (mock_post) is passed as an argument to the test.
    """

    @patch("src.rag.chain.requests.post")
    def test_returns_answer_on_success(self, mock_post):
        """Check that a successful API call returns the answer text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Banner_partner has the worst ROAS at 1.03."
                }
            }]
        }
        mock_post.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": "Which campaign is worst?"},
        ]
        result = call_llm(messages)

        assert "Banner_partner" in result, "Should return the mocked answer"

    @patch("src.rag.chain.requests.post")
    def test_handles_503_loading(self, mock_post):
        """Check that a 503 response returns a loading message."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = call_llm(messages)

        assert "loading" in result.lower(), "Should mention model is loading"

    @patch("src.rag.chain.requests.post")
    def test_handles_timeout(self, mock_post):
        """Check that a timeout returns a friendly error message."""
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()

        messages = [{"role": "user", "content": "test"}]
        result = call_llm(messages)

        assert "timed out" in result.lower() or "timeout" in result.lower(), \
            "Should mention timeout"

    @patch("src.rag.chain.requests.post")
    def test_handles_connection_error(self, mock_post):
        """Check that a connection error returns a friendly message."""
        import requests as req
        mock_post.side_effect = req.exceptions.ConnectionError()

        messages = [{"role": "user", "content": "test"}]
        result = call_llm(messages)

        assert "connection" in result.lower(), "Should mention connection error"

    def test_handles_missing_token(self, monkeypatch):
        """Check that missing HF_TOKEN returns an error message."""
        monkeypatch.setattr("src.rag.chain.HF_TOKEN", None)

        messages = [{"role": "user", "content": "test"}]
        result = call_llm(messages)

        assert "HF_TOKEN" in result, "Should mention missing token"


# ============================================================
# TESTS FOR ask() — full pipeline
# ============================================================

class TestAsk:
    """
    Tests for the full RAG pipeline (ask function).

    We mock both the index query and the LLM call
    to test the pipeline logic without external dependencies.
    """

    @patch("src.rag.chain.call_llm")
    @patch("src.rag.chain.query_index")
    def test_returns_correct_structure(self, mock_query, mock_llm):
        """Check that ask() returns dict with answer, sources, metadatas."""
        mock_query.return_value = {
            "documents": ["Campaign data doc 1", "Campaign data doc 2"],
            "metadatas": [
                {"campaign_name": "facebook_tier1"},
                {"campaign_name": "google_hot"},
            ],
            "distances": [0.5, 0.7],
            "ids": ["row_0", "row_1"],
        }
        mock_llm.return_value = "Facebook has better ROAS than Google."

        result = ask("Compare campaigns")

        assert "answer" in result, "Result should have 'answer'"
        assert "sources" in result, "Result should have 'sources'"
        assert "metadatas" in result, "Result should have 'metadatas'"
        assert "distances" in result, "Result should have 'distances'"

    @patch("src.rag.chain.call_llm")
    @patch("src.rag.chain.query_index")
    def test_answer_comes_from_llm(self, mock_query, mock_llm):
        """Check that the answer field contains the LLM response."""
        mock_query.return_value = {
            "documents": ["doc1"],
            "metadatas": [{"campaign_name": "test"}],
            "distances": [0.5],
            "ids": ["row_0"],
        }

        expected_answer = "The worst campaign is banner_partner with ROAS 1.03."
        mock_llm.return_value = expected_answer

        result = ask("Which is worst?")

        assert result["answer"] == expected_answer, \
            "Answer should match LLM response"

    @patch("src.rag.chain.call_llm")
    @patch("src.rag.chain.query_index")
    def test_sources_match_query_results(self, mock_query, mock_llm):
        """Check that sources in result match what query_index returned."""
        expected_docs = ["Doc A", "Doc B", "Doc C"]
        mock_query.return_value = {
            "documents": expected_docs,
            "metadatas": [{"campaign_name": "a"}, {"campaign_name": "b"}, {"campaign_name": "c"}],
            "distances": [0.1, 0.2, 0.3],
            "ids": ["row_0", "row_1", "row_2"],
        }
        mock_llm.return_value = "Some answer"

        result = ask("test question", top_k=3)

        assert result["sources"] == expected_docs, \
            "Sources should match query_index results"