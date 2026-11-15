"""Tests for text cleaning, chunking, and storing chunks through the pipeline."""

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from researchmate.api.main import create_app
from researchmate.config import Settings
from researchmate.db.models import Chunk
from researchmate.db.repositories import ChunkRepository
from researchmate.ingestion.chunking import chunk_units
from researchmate.ingestion.cleaning import (
    build_vocabulary,
    clean_text,
    dehyphenate,
    normalize_unicode,
)
from researchmate.ingestion.parsing import TextUnit


def count_words(text: str) -> int:
    """Same fake as conftest's ``token_counter`` fixture: one token per word."""
    return len(text.split())

# --- cleaning --------------------------------------------------------------------------------

def test_normalize_unicode() -> None:
    raw = "eﬃcient ﬁne­​word\x00 a b\r\nx²"
    assert normalize_unicode(raw) == "efficient fineword a b\nx²"  # x² kept (no NFKC)


def test_dehyphenate_uses_document_vocabulary() -> None:
    vocabulary = build_vocabulary(["sequence transduction models", "position-wise layers"])
    text = "transduc-\ntion and sequence-\naligned and position-\nwise and sur-\nprisingly"
    assert dehyphenate(text, vocabulary) == (
        "transduction and sequence-aligned and position-wise and sur-prisingly"
    )


def test_dehyphenate_is_case_insensitive_and_ignores_numbers() -> None:
    vocabulary = build_vocabulary(["convolutional networks"])
    assert dehyphenate("Convolu-\ntional", vocabulary) == "Convolutional"
    assert dehyphenate("pages 12-\n15", vocabulary) == "pages 12-\n15"  # digits untouched


def test_clean_text_collapses_spaces_and_keeps_lines() -> None:
    text = "  Title  \n\n\nRetriever |   Recall\nBM25 | 0.61 \n"
    assert clean_text(text, set()) == "Title\nRetriever | Recall\nBM25 | 0.61"


# --- chunking ----------------------------------------------------------------------------------

def sentences(n: int, words_each: int = 10, prefix: str = "S") -> list[str]:
    return [" ".join([f"{prefix}{i}"] + ["w"] * (words_each - 2) + ["end."]) for i in range(n)]


def chunk(text: str, size: int = 50, overlap: int = 10, **location) -> list:
    location = location or {"page_number": 1}
    return chunk_units([TextUnit(text=text, **location)], count_tokens=count_words,
                       chunk_size=size, overlap=overlap)


def test_short_text_is_one_chunk_with_its_location() -> None:
    [only] = chunk("One short sentence. Another one.", page_number=7)
    assert (only.text, only.page_number, only.section_number) == (
        "One short sentence. Another one.", 7, None)
    assert (only.chunk_index, only.token_count) == (0, 5)


def test_chunks_respect_size_and_break_between_sentences() -> None:
    text = " ".join(sentences(12))  # 12 sentences x 10 words
    chunks = chunk(text, size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 50
        assert c.text.endswith("end.")           # never cut inside a sentence
        assert re.match(r"S\d+ ", c.text)        # always starts at a sentence start


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    chunks = chunk(" ".join(sentences(12)), size=50, overlap=10)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        last_sentence = re.findall(r"S\d+ [^S]*end\.", previous.text)[-1]
        assert current.text.startswith(last_sentence)  # 10-token sentence carried over


def test_no_text_is_lost() -> None:
    parts = sentences(30, words_each=7)
    chunks = chunk(" ".join(parts), size=50, overlap=10)
    for sentence in parts:
        assert any(sentence in c.text for c in chunks), sentence


def test_overlap_skipped_when_last_sentence_is_too_long() -> None:
    long_sentences = sentences(3, words_each=30)  # each 30 tokens > overlap 10
    chunks = chunk(" ".join(long_sentences), size=50, overlap=10)
    assert [c.text for c in chunks] == long_sentences


def test_chunks_never_cross_units_and_index_runs_across_document() -> None:
    units = [TextUnit(text=" ".join(sentences(8, prefix="A")), page_number=1),
             TextUnit(text=" ".join(sentences(8, prefix="B")), page_number=2)]
    chunks = chunk_units(units, count_tokens=count_words, chunk_size=50, overlap=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert ("A" in c.text) != ("B" in c.text)              # one unit only
        assert c.page_number == (1 if "A" in c.text else 2)    # location of its unit


def test_docx_section_location_and_table_lines_are_kept() -> None:
    [only] = chunk("Method\nRetriever | Recall\nBM25 | 0.61", section_number=3)
    assert (only.page_number, only.section_number) == (None, 3)
    assert only.text == "Method\nRetriever | Recall\nBM25 | 0.61"  # span sliced, not re-joined


def test_sentence_without_punctuation_is_split_by_words() -> None:
    text = " ".join(f"w{i}" for i in range(120))  # one 120-token "sentence"
    chunks = chunk(text, size=50, overlap=10)
    assert [c.token_count for c in chunks] == [50, 50, 20]
    assert " ".join(c.text for c in chunks) == text


def test_single_huge_word_is_cut_by_characters() -> None:
    # Counter charging one token per character, so one word can exceed the limit.
    # (The real bge tokenizer maps words > 100 chars to a single [UNK] token.)
    def per_char(text: str) -> int:
        return len(text.replace(" ", ""))

    chunks = chunk_units([TextUnit(text="x" * 120, page_number=1)], count_tokens=per_char,
                         chunk_size=50, overlap=10)
    assert [len(c.text) for c in chunks] == [50, 50, 20]


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk("a b c.", size=50, overlap=50)


def test_random_texts_always_satisfy_chunk_invariants() -> None:
    rng = random.Random(1234)
    for _ in range(200):
        words = []
        for _ in range(rng.randint(1, 400)):
            word = "w" * rng.randint(1, 3) if rng.random() > 0.01 else "u" * rng.randint(60, 90)
            words.append(word + rng.choice(["", "", "", ".", ","]))
            if rng.random() < 0.05:
                words.append("\n")
        text = clean_text(" ".join(words).replace(".", ". A"), set())
        size, overlap = rng.choice([(50, 10), (60, 0), (80, 20)])
        chunks = chunk(text, size=size, overlap=overlap)
        assert chunks, text
        for c in chunks:
            assert 0 < c.token_count <= size
            assert c.text == c.text.strip()
        covered = "".join(c.text for c in chunks)
        assert all(token in covered for token in text.split())  # nothing lost


# --- pipeline: chunks stored with the right locations -----------------------------------------

@pytest.fixture
def settings(make_settings, tmp_path: Path) -> Settings:
    return make_settings(data_dir=tmp_path / "data", chunk_size_tokens=50,
                         chunk_overlap_tokens=10)


@pytest.fixture
def client(settings: Settings, token_counter) -> Iterator[TestClient]:
    with TestClient(create_app(settings, token_counter=token_counter),
                    raise_server_exceptions=False) as test_client:
        yield test_client


def _long_pdf(path: Path, pages: int) -> Path:
    pdf = pymupdf.open()
    for p in range(pages):
        page = pdf.new_page()
        for line in range(30):  # 30 short sentences per page
            page.insert_text((50, 40 + line * 20), f"Page{p + 1} sentence {line} ends here.")
    pdf.save(path)
    pdf.close()
    return path


def _stored_chunks(client: TestClient, document_id: str) -> list[Chunk]:
    with client.app.state.session_factory() as session:
        return ChunkRepository(session).list_for_document(document_id)


def test_upload_stores_chunks_with_page_numbers(client: TestClient, tmp_path: Path) -> None:
    path = _long_pdf(tmp_path / "long.pdf", pages=3)
    body = client.post("/documents", files={"file": (path.name, path.read_bytes())}).json()
    assert body["status"] == "ready"
    chunks = _stored_chunks(client, body["id"])
    assert body["chunk_count"] == len(chunks) > 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= 50
        assert {int(n) for n in re.findall(r"Page(\d)", c.text)} == {c.page_number}


def test_chunks_are_discarded_if_storing_fails(client: TestClient, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    original = ChunkRepository.add_chunks

    def add_then_fail(self, document, chunks):
        original(self, document, chunks)  # chunks are flushed to the database...
        raise RuntimeError("disk full")    # ...then something goes wrong

    monkeypatch.setattr(ChunkRepository, "add_chunks", add_then_fail)
    path = _long_pdf(tmp_path / "x.pdf", pages=1)
    body = client.post("/documents", files={"file": (path.name, path.read_bytes())}).json()
    assert (body["status"], body["chunk_count"]) == ("failed", 0)
    with client.app.state.session_factory() as session:
        assert session.scalars(select(Chunk)).all() == []  # no half-stored chunks


def test_tests_cannot_load_the_real_tokenizer(make_settings) -> None:
    from researchmate.ingestion.tokens import HuggingFaceTokenCounter

    with pytest.raises(RuntimeError, match="fake token counter"):
        HuggingFaceTokenCounter(make_settings().embedding_model)("hello")
