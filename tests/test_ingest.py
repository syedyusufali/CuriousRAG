# tests/test_ingest.py
import os
import tempfile
import pytest
from scripts import ingest

@pytest.fixture
def sample_text():
    return "This is a test document. It contains multiple sentences. We will use it for chunking."

def test_chunk_text(sample_text):
    chunks = ingest.chunk_text(sample_text, chunk_size=20)
    assert all(len(chunk) <= 20 for chunk in chunks)
    assert len(chunks) > 0

def test_load_txt_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample content.")
    content = ingest.load_txt_file(str(file_path))
    assert content == "Sample content."

def test_load_md_file(tmp_path):
    file_path = tmp_path / "test.md"
    file_path.write_text("# Title\nSome description.")
    result = ingest.load_md_file(str(file_path))
    assert "Title" in result and "description" in result

def test_load_documents_with_txt(tmp_path):
    txt_file = tmp_path / "doc1.txt"
    txt_file.write_text("Simple text doc")
    loaded = ingest.load_documents(str(tmp_path))
    assert len(loaded) == 1 and "Simple text" in loaded[0]
