"""Audyt zgodności strategii i historii transakcji."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import yaml

from core.monitoring.events import ComplianceViolation, EventPublisher


_DEFAULT_CONFIG_PATH = Path("config/compliance/audit.yml")


class ComplianceAuditError(RuntimeError):
    """Podstawowy wyjątek sygnalizujący błędy konfiguracji audytu."""


@dataclass(slots=True)
class ComplianceFinding:
    """Pojedyncze naruszenie zidentyfikowane podczas audytu."""

    rule_id: str
    severity: str
    message: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ComplianceAuditResult:
    """Wynik audytu zgodności dla przekazanej konfiguracji."""

    generated_at: datetime
    passed: bool
    findings: Sequence[ComplianceFinding]
    context_summary: Mapping[str, object]
    config_path: Path | None = None

    def to_dict(self) -> MutableMapping[str, object]:
        return {
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "passed": self.passed,
            "findings": [finding.to_dict() for finding in self.findings],
            "context_summary": dict(self.context_summary),
            "config_path": str(self.config_path) if self.config_path else None,
        }


class ComplianceAuditor:
    """Analizuje konfigurację strategii, źródła danych i historię transakcji."""

    def __init__(
        self,
        *,
        config_path: Path | None = None,
    ) -> None:
        self._config_path = Path(config_path or _DEFAULT_CONFIG_PATH)
        self._config = self._load_config(self._config_path)
        self._kyc_required_fields = tuple(
            str(field).strip()
            for field in self._config.get("kyc", {}).get("required_fields", ())
            if str(field).strip()
        )
        self._kyc_severity = str(
            self._config.get("kyc", {}).get("severity", "high")
        ).strip() or "high"

        aml_section = self._config.get("aml", {})
        self._aml_blocked_countries = {
            str(item).upper()
            for item in aml_section.get("blocked_countries", ())
            if str(item).strip()
        }
        self._aml_high_risk_jurisdictions = {
            str(item).upper()
            for item in aml_section.get("high_risk_jurisdictions", ())
            if str(item).strip()
        }
        self._aml_suspicious_tags = {
            str(item).lower()
            for item in aml_section.get("suspicious_tags", ())
            if str(item).strip()
        }
        self._aml_forbidden_sources = {
            str(item).lower()
            for item in aml_section.get("forbidden_data_sources", ())
            if str(item).strip()
        }
        self._aml_unverified_volume_limit = self._safe_float(
            aml_section.get("max_unverified_volume_usd")
        )
        self._aml_severity = str(aml_section.get("severity", "critical")).strip() or "critical"

        tx_section = self._config.get("transaction_limits", {})
        self._tx_single_limit = self._safe_float(tx_section.get("max_single_trade_usd"))
        self._tx_daily_limit = self._safe_float(tx_section.get("max_daily_volume_usd"))
        self._tx_lookback_days = max(
            0,
            int(tx_section.get("lookback_days", 1)) if str(tx_section.get("lookback_days", "")).strip() else 1,
        )
        self._tx_severity = str(tx_section.get("severity", "warning")).strip() or "warning"

    def audit(
        self,
        *,
        strategy_config: Mapping[str, object] | None,
        data_sources: Sequence[str] | None,
        transactions: Sequence[Mapping[str, object]] | None,
        kyc_profile: Mapping[str, object] | None,
        as_of: datetime | None = None,
        event_publisher: EventPublisher | None = None,
    ) -> ComplianceAuditResult:
        """Wykonuje audyt zgodności i opcjonalnie publikuje alerty guardrail."""

        timestamp = (as_of or datetime.now(timezone.utc)).astimezone(timezone.utc)
        findings: list[ComplianceFinding] = []

        findings.extend(self._evaluate_kyc(kyc_profile))
        findings.extend(
            self._evaluate_aml(
                strategy_config=strategy_config or {},
                data_sources=data_sources or (),
                transactions=transactions or (),
                kyc_profile=kyc_profile or {},
            )
        )
        findings.extend(
            self._evaluate_transaction_limits(
                transactions or (),
                as_of=timestamp,
            )
        )

        if event_publisher is not None:
            for finding in findings:
                severity = finding.severity.lower()
                if severity in {"warning", "high", "critical", "error"}:
                    event_publisher(
                        ComplianceViolation(
                            rule_id=finding.rule_id,
                            severity=finding.severity,
                            message=finding.message,
                            metadata=dict(finding.metadata),
                        )
                    )

        context_summary = self._build_context_summary(
            strategy_config or {}, data_sources or (), transactions or (), kyc_profile or {}
        )
        return ComplianceAuditResult(
            generated_at=timestamp,
            passed=not findings,
            findings=tuple(findings),
            context_summary=context_summary,
            config_path=self._config_path if self._config_path.exists() else None,
        )

    def _evaluate_kyc(
        self, profile: Mapping[str, object] | None
    ) -> Iterable[ComplianceFinding]:
        if not self._kyc_required_fields:
            return ()
        profile = profile or {}
        missing = [
            field
            for field in self._kyc_required_fields
            if not str(profile.get(field) or "").strip()
        ]
        if missing:
            yield ComplianceFinding(
                rule_id="KYC_MISSING_FIELDS",
                severity=self._kyc_severity,
                message="Brak wymaganych pól KYC",
                metadata={"missing_fields": tuple(missing)},
            )

    def _evaluate_aml(
        self,
        *,
        strategy_config: Mapping[str, object],
        data_sources: Sequence[str],
        transactions: Sequence[Mapping[str, object]],
        kyc_profile: Mapping[str, object],
    ) -> Iterable[ComplianceFinding]:
        findings: list[ComplianceFinding] = []
        country = str(kyc_profile.get("country") or "").upper()
        if country and country in self._aml_blocked_countries:
            findings.append(
                ComplianceFinding(
                    rule_id="AML_BLOCKED_COUNTRY",
                    severity=self._aml_severity,
                    message="Profil KYC powiązany z zablokowanym krajem",
                    metadata={"country": country},
                )
            )
        status = str(kyc_profile.get("status") or "").lower()
        if (
            self._aml_unverified_volume_limit is not None
            and status not in {"verified", "full"}
        ):
            volume = sum(
                self._extract_usd_value(tx) or 0.0 for tx in transactions
            )
            if volume > self._aml_unverified_volume_limit:
                findings.append(
                    ComplianceFinding(
                        rule_id="AML_UNVERIFIED_VOLUME",
                        severity=self._aml_severity,
                        message="Nadmiarowy wolumen dla niezweryfikowanego profilu",
                        metadata={"volume_usd": round(volume, 2)},
                    )
                )

        tags = self._extract_tags(strategy_config)
        suspicious = sorted(tag for tag in tags if tag in self._aml_suspicious_tags)
        if suspicious:
            findings.append(
                ComplianceFinding(
                    rule_id="AML_SUSPICIOUS_TAG",
                    severity=self._aml_severity,
                    message="Strategia zawiera tagi oznaczone jako ryzykowne",
                    metadata={"tags": tuple(suspicious)},
                )
            )

        forbidden_sources = sorted(
            source
            for source in (str(item).lower() for item in data_sources)
            if source in self._aml_forbidden_sources
        )
        if forbidden_sources:
            findings.append(
                ComplianceFinding(
                    rule_id="AML_FORBIDDEN_SOURCE",
                    severity=self._aml_severity,
                    message="Wykryto niedozwolone źródła danych",
                    metadata={"sources": tuple(forbidden_sources)},
                )
            )

        high_risk_hits = self._detect_high_risk_transactions(transactions)
        if high_risk_hits:
            findings.append(
                ComplianceFinding(
                    rule_id="AML_HIGH_RISK_TRANSACTION",
                    severity=self._aml_severity,
                    message="Historia transakcji zawiera kontrahentów z jurysdykcji wysokiego ryzyka",
                    metadata={"transactions": tuple(high_risk_hits)},
                )
            )
        return findings

    def _evaluate_transaction_limits(
        self,
        transactions: Sequence[Mapping[str, object]],
        *,
        as_of: datetime,
    ) -> Iterable[ComplianceFinding]:
        if not transactions:
            return ()
        findings: list[ComplianceFinding] = []
        if self._tx_single_limit is not None:
            exceeding = [
                (tx, self._extract_usd_value(tx))
                for tx in transactions
                if (self._extract_usd_value(tx) or 0.0) > self._tx_single_limit
            ]
            for tx, value in exceeding:
                findings.append(
                    ComplianceFinding(
                        rule_id="TX_SINGLE_LIMIT_EXCEEDED",
                        severity=self._tx_severity,
                        message="Pojedyncza transakcja przekracza ustalony limit",
                        metadata={
                            "transaction_id": str(tx.get("id") or tx.get("external_id") or "unknown"),
                            "value_usd": round(value or 0.0, 2),
                            "limit_usd": self._tx_single_limit,
                        },
                    )
                )

        if self._tx_daily_limit is not None and self._tx_lookback_days > 0:
            cutoff = as_of - timedelta(days=self._tx_lookback_days)
            total = 0.0
            for tx in transactions:
                timestamp = self._parse_timestamp(tx.get("timestamp"))
                if timestamp is None or timestamp < cutoff:
                    continue
                total += self._extract_usd_value(tx) or 0.0
            if total > self._tx_daily_limit:
                findings.append(
                    ComplianceFinding(
                        rule_id="TX_DAILY_LIMIT_EXCEEDED",
                        severity=self._tx_severity,
                        message="Łączny wolumen transakcji przekracza limit w oknie czasowym",
                        metadata={
                            "total_volume_usd": round(total, 2),
                            "limit_usd": self._tx_daily_limit,
                            "lookback_days": self._tx_lookback_days,
                        },
                    )
                )
        return findings

    def _detect_high_risk_transactions(
        self, transactions: Sequence[Mapping[str, object]]
    ) -> tuple[str, ...]:
        if not self._aml_high_risk_jurisdictions:
            return ()
        hits: list[str] = []
        for tx in transactions:
            jurisdiction = str(
                tx.get("counterparty_country")
                or tx.get("jurisdiction")
                or tx.get("country")
                or ""
            ).upper()
            if not jurisdiction:
                continue
            if jurisdiction in self._aml_high_risk_jurisdictions:
                identifier = str(tx.get("id") or tx.get("external_id") or tx.get("hash") or "unknown")
                hits.append(identifier)
        return tuple(hits)

    def _build_context_summary(
        self,
        strategy_config: Mapping[str, object],
        data_sources: Sequence[str],
        transactions: Sequence[Mapping[str, object]],
        kyc_profile: Mapping[str, object],
    ) -> Mapping[str, object]:
        strategy_name = (
            str(
                strategy_config.get("name")
                or strategy_config.get("engine")
                or strategy_config.get("strategy")
                or ""
            ).strip()
            or "unknown"
        )
        exchange = str(strategy_config.get("exchange") or "").strip() or None
        summary: dict[str, object] = {
            "strategy": strategy_name,
            "data_sources": tuple(dict.fromkeys(str(item) for item in data_sources if str(item))),
            "transactions_analyzed": len(transactions),
        }
        if exchange:
            summary["exchange"] = exchange
        status = str(kyc_profile.get("status") or "").strip()
        if status:
            summary["kyc_status"] = status
        return summary

    @staticmethod
    def _extract_usd_value(transaction: Mapping[str, object]) -> float | None:
        for key in ("usd_value", "value_usd", "notional_usd", "notionalUSD"):
            value = transaction.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        amount = transaction.get("amount")
        price = transaction.get("price")
        try:
            if amount is not None and price is not None:
                return float(amount) * float(price)
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _parse_timestamp(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _extract_tags(strategy_config: Mapping[str, object]) -> set[str]:
        tags: set[str] = set()
        raw = strategy_config.get("tags") or strategy_config.get("metadata", {}).get("tags")
        if isinstance(raw, Mapping):
            raw = raw.get("default") or raw.get("values")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, Sequence):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    tags.add(item.strip().lower())
        return tags

    @staticmethod
    def _safe_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    @staticmethod
    def _load_config(path: Path) -> Mapping[str, object]:
        if not path.exists():
            return {}
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - odczyt może się nie udać na egzotycznych FS
            raise ComplianceAuditError(f"Nie można odczytać konfiguracji audytu: {exc}") from exc
        data = yaml.safe_load(raw_text) or {}
        if not isinstance(data, Mapping):
            raise ComplianceAuditError("Plik konfiguracji audytu musi zawierać mapę.")
        return data


__all__ = [
    "ComplianceAuditError",
    "ComplianceAuditResult",
    "ComplianceAuditor",
    "ComplianceFinding",
]
