"""Kontroler centrum pomocy wraz z loaderem artykułów."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtCore import QObject, Property, Signal, Slot

_DEFAULT_ARTICLE_DIRECTORY = Path("docs/support/articles")
_TOKEN_PATTERN = re.compile(r"[\wąćęłńóśźż]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RunbookLink:
    """Reprezentuje odnośnik do runbooka związanego z artykułem."""

    path: Path

    @property
    def title(self) -> str:
        stem = self.path.stem.replace("_", " ")
        return stem.replace("-", " ").title()

    def to_dict(self) -> dict[str, object]:
        resolved = self.path if self.path.is_absolute() else (Path.cwd() / self.path).resolve()
        return {
            "title": self.title,
            "path": str(resolved),
            "relativePath": str(self.path),
        }


@dataclass(frozen=True, slots=True)
class SupportArticle:
    """Pojedynczy artykuł centrum pomocy."""

    article_id: str
    title: str
    summary: str
    body: str
    tags: tuple[str, ...]
    category: str
    runbooks: tuple[RunbookLink, ...]
    source_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.article_id,
            "title": self.title,
            "summary": self.summary,
            "body": self.body,
            "tags": list(self.tags),
            "category": self.category,
            "runbooks": [link.to_dict() for link in self.runbooks],
            "sourcePath": str(self.source_path),
            "matchScore": 0.0,
        }


@dataclass(frozen=True, slots=True)
class SupportSearchResult:
    """Wynik wyszukiwania artykułów."""

    article: SupportArticle
    score: float


class SupportArticleLoader:
    """Odpowiada za wczytywanie artykułów centrum pomocy z plików Markdown."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self._directory = Path(directory) if directory is not None else _DEFAULT_ARTICLE_DIRECTORY

    @property
    def directory(self) -> Path:
        return self._directory

    def load(self) -> list[SupportArticle]:
        directory = self._directory
        if not directory.exists():
            raise FileNotFoundError(f"Katalog artykułów nie istnieje: {directory}")

        articles: list[SupportArticle] = []
        for path in sorted(directory.glob("*.md")):
            article = self._parse_article(path)
            if article is not None:
                articles.append(article)
        return articles

    # ------------------------------------------------------------------
    def _parse_article(self, path: Path) -> SupportArticle | None:
        raw = path.read_text(encoding="utf-8")
        metadata, body = _split_front_matter(raw)

        article_id = metadata.get("id") or path.stem
        title = metadata.get("title") or _extract_heading(body) or path.stem.replace("_", " ").title()
        summary = metadata.get("summary") or body.splitlines()[0].strip() if body.strip() else ""
        tags = _split_list(metadata.get("tags", ""))
        category = metadata.get("category", "")
        runbook_items = _split_list(metadata.get("runbooks", ""))
        runbooks = tuple(RunbookLink(Path(item)) for item in runbook_items if item)

        return SupportArticle(
            article_id=article_id,
            title=title.strip(),
            summary=summary.strip(),
            body=body.strip(),
            tags=tuple(tag for tag in tags if tag),
            category=category.strip(),
            runbooks=runbooks,
            source_path=path,
        )


class SupportSearchIndex:
    """Proste indeksowanie artykułów na potrzeby wyszukiwarki offline."""

    def __init__(self, articles: Sequence[SupportArticle]) -> None:
        self._articles = list(articles)
        self._index = {article.article_id: self._tokenize_article(article) for article in self._articles}

    def _tokenize_article(self, article: SupportArticle) -> set[str]:
        text = " ".join(
            (
                article.title,
                article.summary,
                " ".join(article.tags),
                article.body,
                article.category,
            )
        )
        return set(_tokenize(text))

    def search(self, query: str) -> list[SupportSearchResult]:
        tokens = set(_tokenize(query))
        if not tokens:
            return [SupportSearchResult(article=a, score=0.0) for a in self._articles]

        results: list[SupportSearchResult] = []
        for article in self._articles:
            article_tokens = self._index.get(article.article_id, set())
            score = sum(1.0 for token in tokens if token in article_tokens)
            if score > 0:
                results.append(SupportSearchResult(article=article, score=score))
        results.sort(key=lambda entry: (-entry.score, entry.article.title.lower()))
        return results


class SupportCenterController(QObject):
    """Kontroler udostępniający artykuły centrum pomocy do warstwy QML."""

    articlesChanged = Signal()
    filteredArticlesChanged = Signal()
    searchQueryChanged = Signal()
    errorMessageChanged = Signal()
    lastUpdatedChanged = Signal()
    selectedArticleChanged = Signal()

    def __init__(
        self,
        *,
        article_directory: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._loader = SupportArticleLoader(article_directory)
        self._articles: list[SupportArticle] = []
        self._article_payload: list[dict[str, object]] = []
        self._filtered_payload: list[dict[str, object]] = []
        self._index = SupportSearchIndex(())
        self._search_query = ""
        self._error_message = ""
        self._last_updated = ""
        self._selected_article_id: str | None = None
        self._selected_article: dict[str, object] | None = None
        self.refreshArticles()

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=articlesChanged)
    def articles(self) -> list[dict[str, object]]:  # type: ignore[override]
        return [dict(entry) for entry in self._article_payload]

    @Property("QVariantList", notify=filteredArticlesChanged)
    def filteredArticles(self) -> list[dict[str, object]]:  # type: ignore[override]
        return [dict(entry) for entry in self._filtered_payload]

    @Property(str, notify=searchQueryChanged)
    def searchQuery(self) -> str:  # type: ignore[override]
        return self._search_query

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property(str, notify=lastUpdatedChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return self._last_updated

    @Property("QVariantMap", notify=selectedArticleChanged)
    def selectedArticle(self) -> dict[str, object]:  # type: ignore[override]
        return dict(self._selected_article or {})

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshArticles(self) -> bool:
        try:
            articles = self._loader.load()
        except FileNotFoundError as exc:
            self._set_error(str(exc))
            self._clear_articles()
            return False
        except Exception as exc:  # pragma: no cover - nieoczekiwane błędy IO
            self._set_error(str(exc))
            self._clear_articles()
            return False

        self._set_error("")
        self._articles = articles
        self._article_payload = [article.to_dict() for article in self._articles]
        self._index = SupportSearchIndex(self._articles)
        self._last_updated = _now_local_iso()
        self.lastUpdatedChanged.emit()
        self._apply_search(self._search_query)
        self.articlesChanged.emit()
        return True

    @Slot(str, result=bool)
    def searchArticles(self, query: str) -> bool:
        self._search_query = query
        self.searchQueryChanged.emit()
        self._apply_search(query)
        return True

    @Slot(str, result=bool)
    def selectArticle(self, article_id: str) -> bool:
        if not article_id:
            self._selected_article_id = None
            self._selected_article = None
            self.selectedArticleChanged.emit()
            return False

        for entry in self._filtered_payload:
            if entry.get("id") == article_id:
                self._selected_article_id = article_id
                self._selected_article = dict(entry)
                self.selectedArticleChanged.emit()
                return True

        return False

    @Slot(str, result=bool)
    def openRunbook(self, path: str) -> bool:
        if not path:
            return False
        resolved = Path(path).expanduser()
        return resolved.exists()

    # ------------------------------------------------------------------
    def _apply_search(self, query: str) -> None:
        results = self._index.search(query)
        if not query.strip():
            payload = [dict(article_dict) for article_dict in self._article_payload]
        else:
            payload = []
            for result in results:
                data = result.article.to_dict()
                data["matchScore"] = result.score
                payload.append(data)
        self._filtered_payload = payload
        self.filteredArticlesChanged.emit()
        self._ensure_selected_article()

    def _clear_articles(self) -> None:
        self._articles = []
        self._article_payload = []
        self._filtered_payload = []
        self._selected_article = None
        self._selected_article_id = None
        self.articlesChanged.emit()
        self.filteredArticlesChanged.emit()
        self.selectedArticleChanged.emit()

    def _ensure_selected_article(self) -> None:
        if self._selected_article_id:
            for entry in self._filtered_payload:
                if entry.get("id") == self._selected_article_id:
                    self._selected_article = dict(entry)
                    self.selectedArticleChanged.emit()
                    return

        if self._filtered_payload:
            self._selected_article = dict(self._filtered_payload[0])
            self._selected_article_id = str(self._selected_article.get("id"))
        else:
            self._selected_article = None
            self._selected_article_id = None
        self.selectedArticleChanged.emit()

    def _set_error(self, message: str) -> None:
        if self._error_message == message:
            return
        self._error_message = message
        self.errorMessageChanged.emit()


def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    lines = raw.splitlines()
    if not lines:
        return {}, ""

    if lines[0].strip() != "---":
        return {}, raw

    metadata: dict[str, str] = {}
    idx = 1
    while idx < len(lines):
        line = lines[idx].strip()
        if line == "---":
            idx += 1
            break
        if not line or line.startswith("#"):
            idx += 1
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
        idx += 1

    body = "\n".join(lines[idx:])
    return metadata, body


def _split_list(raw: str) -> list[str]:
    if not raw:
        return []
    items = [item.strip() for item in re.split(r",|;", raw) if item.strip()]
    return items


def _extract_heading(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ")
    return ""


def _tokenize(text: str) -> Iterable[str]:
    return (match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text))


def _now_local_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


__all__ = [
    "RunbookLink",
    "SupportArticle",
    "SupportArticleLoader",
    "SupportSearchIndex",
    "SupportSearchResult",
    "SupportCenterController",
]
