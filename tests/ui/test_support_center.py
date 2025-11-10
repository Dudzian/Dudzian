import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek QtWidgets
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from ui.backend.support_center import SupportCenterController


@pytest.fixture()
def article_directory(tmp_path: Path) -> Path:
    target = tmp_path / "docs" / "support" / "articles"
    target.mkdir(parents=True)
    (target / "faq.md").write_text(
        "\n".join(
            [
                "---",
                "id: faq",
                "title: Najczęstsze pytania",
                "summary: Jak rozpocząć korzystanie z bota",
                "tags: start, konfiguracja",
                "category: onboarding",
                "runbooks: docs/operations/runbooks/README.md",
                "---",
                "# Najczęstsze pytania\nAby rozpocząć, uruchom kreator licencji i konfiguracji.",
            ]
        ),
        encoding="utf-8",
    )
    (target / "network.md").write_text(
        "\n".join(
            [
                "---",
                "id: network",
                "title: Problemy z siecią",
                "summary: Co zrobić, gdy połączenie jest niestabilne",
                "tags: sieć, diagnostyka",
                "category: wsparcie",
                "runbooks: docs/operations/runbooks/network_diagnostics.md",
                "---",
                "# Diagnostyka sieci\nSprawdź firewall i porty wymagane przez giełdy.",
            ]
        ),
        encoding="utf-8",
    )
    return target


@pytest.mark.timeout(30)
def test_support_center_loads_and_filters(article_directory: Path) -> None:
    controller = SupportCenterController(article_directory=article_directory)
    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("supportController", controller)

    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "support" / "SupportCenter.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować SupportCenter.qml"
    root = engine.rootObjects()[0]

    assert controller.articles, "Kontroler powinien załadować artykuły"
    assert root.property("articleCount") == 2

    controller.searchArticles("sieć")
    app.processEvents()
    assert controller.filteredArticles[0]["id"] == "network"
    assert root.property("articleCount") == 1

    controller.selectArticle("network")
    app.processEvents()
    assert controller.selectedArticle["id"] == "network"
    assert "Diagnostyka" in controller.selectedArticle["body"]

    engine.deleteLater()
    app.quit()
