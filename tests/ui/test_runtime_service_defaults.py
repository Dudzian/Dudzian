from ui.backend.runtime_service import RuntimeService


def test_runtime_service_uses_demo_loader_when_no_journal() -> None:
    service = RuntimeService()

    result = service.loadRecentDecisions(5)

    assert result, "Oczekiwano wpisów demonstracyjnych przy pustej konfiguracji"
    assert service.errorMessage == ""
