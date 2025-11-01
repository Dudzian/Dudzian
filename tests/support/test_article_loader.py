from pathlib import Path

import pytest

from ui.backend.support_center import SupportArticleLoader, SupportSearchIndex


def _write_article(path: Path, *, summary: str, body: str, extra: str = "") -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                f"id: {path.stem}",
                f"title: {path.stem.title()}",
                f"summary: {summary}",
                "tags: faq, wsparcie",
                "category: pomoc",
                extra or "runbooks: docs/operations/runbooks/README.md",
                "---",
                body,
            ]
        ),
        encoding="utf-8",
    )


def test_loader_parses_metadata(tmp_path: Path) -> None:
    articles_dir = tmp_path / "docs" / "support" / "articles"
    articles_dir.mkdir(parents=True)
    _write_article(articles_dir / "faq.md", summary="Szybki start", body="# Tytuł\nTreść artykułu")

    loader = SupportArticleLoader(articles_dir)
    articles = loader.load()

    assert len(articles) == 1
    article = articles[0]
    assert article.article_id == "faq"
    assert article.title == "Faq"
    assert article.summary == "Szybki start"
    assert article.tags == ("faq", "wsparcie")
    assert article.category == "pomoc"
    assert article.runbooks and article.runbooks[0].path == Path("docs/operations/runbooks/README.md")


def test_search_index_scores_matches(tmp_path: Path) -> None:
    articles_dir = tmp_path / "docs" / "support" / "articles"
    articles_dir.mkdir(parents=True)
    _write_article(
        articles_dir / "network.md",
        summary="Problemy z siecią",
        body="# Sieć\nJeśli połączenie z giełdą jest zrywane, sprawdź firewall.",
    )
    _write_article(
        articles_dir / "orders.md",
        summary="Problemy z zleceniami",
        body="# Zlecenia\nUpewnij się, że saldo jest wystarczające.",
    )

    loader = SupportArticleLoader(articles_dir)
    articles = loader.load()
    index = SupportSearchIndex(articles)

    results = index.search("połączenie firewall")
    assert results, "Wyniki wyszukiwania powinny zawierać artykuły"
    assert results[0].article.article_id == "network"
    assert results[0].score > 0

    empty_results = index.search("nieistniejące zapytanie")
    assert empty_results == []


def test_loader_missing_directory_raises(tmp_path: Path) -> None:
    missing_dir = tmp_path / "brak"
    loader = SupportArticleLoader(missing_dir)
    with pytest.raises(FileNotFoundError):
        loader.load()
