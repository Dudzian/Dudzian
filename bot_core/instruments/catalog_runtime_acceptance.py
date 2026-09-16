"""Core-owned one-shot source Catalog acceptance.

The public boundary deliberately accepts only a release binding name.  It constructs and
invokes the producer itself; a caller can neither submit a payload nor substitute a factory.
Catalog and producer-membership rows share one SQLite writer transaction, which gives
revocation/supersession and acceptance an unambiguous commit order.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import sqlite3
from types import MappingProxyType
from typing import Any, Mapping
import unicodedata

from bot_core.exchanges.http_client import urlopen
from bot_core.instruments.catalog_projection_oracle import (
    SOURCE_FIELDS,
    SOURCE_FINGERPRINT_FIELDS,
    SOURCE_MEMBER_FIELDS,
    _fingerprint,
    validate_accepted_source_catalog_snapshot,
)
from bot_core.instruments.core_time import PRODUCTION_CORE_CLOCK
from bot_core.instruments.source_producer_membership import (
    SourceProducerMembershipAuthority,
    SQLiteMembershipCarrier,
    _exact,
    _time,
)

_DOMAIN = "cryptohunter.catalog_runtime_acceptance.production.v1"
_METADATA_FP_DOMAIN = "cryptohunter.m0.5.source-product-metadata.v1"
_ID = re.compile(r"[A-Za-z0-9._:-]+\Z")
_DECIMAL = re.compile(r"(?:0|[1-9]\d*)(?:\.\d+)?\Z")
_NORMALIZED_PRODUCT_FIELDS = frozenset({
    "venue_symbol", "market_type", "base_asset", "quote_asset", "product_status",
    "price_precision", "quantity_precision", "tick_size", "step_size",
    "min_quantity", "max_quantity", "min_notional", "contract_metadata",
})
_REQUIRED_CATALOG_TABLES = frozenset({
    "accepted_catalog_snapshots", "catalog_authority_head",
    "source_metadata_versions", "source_metadata_authority_head",
})


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(domain: str, value: Mapping[str, Any], excluded: set[str]) -> str:
    body = {key: value[key] for key in sorted(value) if key not in excluded}
    return hashlib.sha256((domain + "\n" + _canonical(body)).encode("utf-8")).hexdigest()


def _utc_ms(value: object) -> str | None:
    if type(value) is not int or value < 0:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _decimal(value: object, *, positive: bool = False) -> str | None:
    if type(value) not in {str, int} or type(value) is bool:
        return None
    text = str(value)
    if _DECIMAL.fullmatch(text) is None:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if not number.is_finite() or (positive and number <= 0):
        return None
    normalized = format(number, "f").rstrip("0").rstrip(".") if "." in format(number, "f") else format(number, "f")
    return normalized or "0"


@dataclass(frozen=True, slots=True)
class SourceProductMetadataVersion:
    source_metadata_version_id: str
    source_exchange_id: str
    market_type: str
    venue_symbol: str
    metadata_version: int
    previous_source_metadata_version_id: str | None
    normalized_metadata: Mapping[str, Any]
    effective_at_utc: str
    content_fingerprint: str


_METADATA_FIELDS = frozenset(field.name for field in fields(SourceProductMetadataVersion))


def _valid_normalized_product(value: object) -> bool:
    if type(value) is not dict or set(value) != _NORMALIZED_PRODUCT_FIELDS:
        return False
    if (
        type(value.get("venue_symbol")) is not str
        or not value["venue_symbol"]
        or _ID.fullmatch(value["venue_symbol"]) is None
        or value.get("market_type") != "SPOT"
        or not all(type(value.get(name)) is str and value[name] and _ID.fullmatch(value[name])
                   for name in ("base_asset", "quote_asset", "product_status"))
        or type(value.get("price_precision")) is not int
        or value["price_precision"] < 0
        or type(value.get("quantity_precision")) is not int
        or value["quantity_precision"] < 0
        or value.get("contract_metadata") is not None
    ):
        return False
    for name in ("tick_size", "step_size"):
        if _decimal(value.get(name), positive=True) != value.get(name):
            return False
    for name in ("min_quantity", "max_quantity"):
        if _decimal(value.get(name)) != value.get(name):
            return False
    minimum_notional = value.get("min_notional")
    return minimum_notional is None or _decimal(minimum_notional) == minimum_notional


@dataclass(frozen=True, slots=True)
class AcceptedSourceCatalogSnapshot:
    accepted_source_catalog_snapshot_id: str
    source_exchange_id: str
    market_type: str
    source_adapter_family_id: str
    source_adapter_implementation_id: str
    source_adapter_release_id: str
    source_adapter_version: str
    accepted_source_producer_membership_id: str
    source_producer_generation: int
    source_producer_membership_fingerprint: str
    upstream_snapshot_or_retrieval_id: str
    observed_at_utc: str
    effective_at_utc: str
    stale_after_utc: str
    acceptance_status: str
    completeness_status: str
    completeness_evidence: Mapping[str, Any]
    member_source_product_metadata_versions: tuple[Mapping[str, Any], ...]
    previous_snapshot_id: str | None
    content_fingerprint: str

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["member_source_product_metadata_versions"] = list(
            value["member_source_product_metadata_versions"]
        )
        return value


@dataclass(frozen=True, slots=True)
class _Observation:
    retrieval_id: str
    observed_at_utc: str
    products: tuple[Mapping[str, Any], ...]
    completeness_evidence: Mapping[str, Any]


class _BinanceSpotCatalogProducer:
    """Release-owned exact producer; Binance exchangeInfo is one atomic full response."""

    __slots__ = ()
    identity = MappingProxyType({
        "source_exchange_id": "binance", "market_type": "SPOT",
        "source_adapter_family_id": "binance_public_catalog",
        "source_adapter_implementation_id": "impl_ccxt_binance",
        "source_adapter_release_id": "release_2026_09_15", "source_adapter_version": "4.5.1",
    })
    generation = 1
    freshness_seconds = 3600

    def fetch(self) -> object:
        with urlopen("https://api.binance.com/api/v3/exchangeInfo", timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))


_RELEASE_PRODUCERS = MappingProxyType({"core_release_1_46_binance_spot": _BinanceSpotCatalogProducer})


def _normalize_binance(raw: object) -> _Observation | None:
    """Closed-schema normalization; unknown product fields are harmless, unknown facts are not invented."""
    try:
        if type(raw) is not dict or set(raw) - {"timezone", "serverTime", "rateLimits", "exchangeFilters", "symbols", "sors"}:
            return None
        observed = _utc_ms(raw.get("serverTime"))
        symbols = raw.get("symbols")
        if observed is None or type(symbols) is not list:
            return None
        products: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for item in symbols:
            if type(item) is not dict:
                return None
            symbol = item.get("symbol")
            base = item.get("baseAsset")
            quote = item.get("quoteAsset")
            status = item.get("status")
            if not all(type(v) is str and v and _ID.fullmatch(v) for v in (symbol, base, quote, status)) or symbol in seen:
                return None
            seen.add(symbol)
            filters = item.get("filters")
            if type(filters) is not list:
                return None
            by_type: dict[str, dict[str, Any]] = {}
            for entry in filters:
                if type(entry) is not dict or type(entry.get("filterType")) is not str or entry["filterType"] in by_type:
                    return None
                by_type[entry["filterType"]] = entry
            lot = by_type.get("LOT_SIZE")
            price = by_type.get("PRICE_FILTER")
            if lot is None or price is None:
                return None
            tick = _decimal(price.get("tickSize"), positive=True)
            step = _decimal(lot.get("stepSize"), positive=True)
            min_qty = _decimal(lot.get("minQty"))
            max_qty = _decimal(lot.get("maxQty"))
            if None in {tick, step, min_qty, max_qty}:
                return None
            notional_filter = by_type.get("NOTIONAL") or by_type.get("MIN_NOTIONAL")
            min_notional = None if notional_filter is None else _decimal(notional_filter.get("minNotional"))
            if notional_filter is not None and min_notional is None:
                return None
            price_precision = item.get("quotePrecision")
            quantity_precision = item.get("baseAssetPrecision")
            if (
                type(price_precision) is not int
                or price_precision < 0
                or type(quantity_precision) is not int
                or quantity_precision < 0
            ):
                return None
            products.append({
                "venue_symbol": unicodedata.normalize("NFC", symbol), "market_type": "SPOT",
                "base_asset": unicodedata.normalize("NFC", base), "quote_asset": unicodedata.normalize("NFC", quote),
                "product_status": status, "price_precision": price_precision,
                "quantity_precision": quantity_precision, "tick_size": tick, "step_size": step,
                "min_quantity": min_qty, "max_quantity": max_qty, "min_notional": min_notional,
                "contract_metadata": None,
            })
        if not products:
            return None
        products.sort(key=lambda item: item["venue_symbol"])
        content = hashlib.sha256(_canonical(products).encode()).hexdigest()
        retrieval = f"binance-exchangeInfo:{raw['serverTime']}"
        evidence = MappingProxyType({"policy": "BINANCE_EXCHANGE_INFO_ATOMIC_V1", "atomic_response": True, "terminal": True, "product_count": len(products), "normalized_content_sha256": content})
        return _Observation(retrieval, observed, tuple(products), evidence)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


class CatalogRuntimeAcceptanceAuthority:
    """Production authority. ``fetch_catalog_once`` is the sole ingestion entrypoint."""

    __slots__ = ("_membership", "_path")

    def __init__(self, membership_authority: SourceProducerMembershipAuthority) -> None:
        if type(membership_authority) is not SourceProducerMembershipAuthority:
            raise TypeError("exact production membership authority required")
        carrier = membership_authority._carrier
        if type(carrier) is not SQLiteMembershipCarrier:
            raise TypeError("exact production membership carrier required")
        self._membership = membership_authority
        self._path = carrier.path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                tables = {
                    row[0]
                    for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                }
                marker_exists = "catalog_authority_metadata" in tables
                existing_catalog_tables = tables & _REQUIRED_CATALOG_TABLES
                nonempty = False
                for table in existing_catalog_tables:
                    if table == "catalog_authority_head":
                        row = db.execute(
                            "SELECT committed_sequence FROM catalog_authority_head WHERE singleton=1"
                        ).fetchone()
                        nonempty = nonempty or (row is not None and row[0] != 0)
                    elif table == "source_metadata_authority_head":
                        row = db.execute(
                            "SELECT committed_sequence FROM source_metadata_authority_head WHERE singleton=1"
                        ).fetchone()
                        nonempty = nonempty or (row is not None and row[0] != 0)
                    else:
                        nonempty = nonempty or bool(
                            db.execute(f"SELECT EXISTS(SELECT 1 FROM {table})").fetchone()[0]
                        )
                if not marker_exists and nonempty:
                    raise ValueError("CATALOG_AUTHORITY_DOMAIN_MISSING")
                if marker_exists:
                    if not _REQUIRED_CATALOG_TABLES <= tables:
                        raise ValueError("CATALOG_AUTHORITY_STORAGE_MISSING")
                else:
                    db.execute("CREATE TABLE catalog_authority_metadata(singleton INTEGER PRIMARY KEY CHECK(singleton=1), authority_domain TEXT NOT NULL, schema_version INTEGER NOT NULL)")
                    db.execute("CREATE TABLE source_metadata_versions(source_metadata_sequence INTEGER PRIMARY KEY, source_metadata_version_id TEXT UNIQUE NOT NULL, source_exchange_id TEXT NOT NULL, market_type TEXT NOT NULL, venue_symbol TEXT NOT NULL, metadata_version INTEGER NOT NULL, introduced_by_snapshot_id TEXT NOT NULL, canonical_record TEXT NOT NULL, previous_metadata_digest TEXT NOT NULL, metadata_record_digest TEXT UNIQUE NOT NULL, UNIQUE(source_exchange_id,market_type,venue_symbol,metadata_version))")
                    db.execute("CREATE TABLE source_metadata_authority_head(singleton INTEGER PRIMARY KEY CHECK(singleton=1), committed_sequence INTEGER NOT NULL, committed_digest TEXT NOT NULL, last_metadata_version_id TEXT)")
                    db.execute("CREATE TABLE accepted_catalog_snapshots(snapshot_sequence INTEGER PRIMARY KEY, accepted_source_catalog_snapshot_id TEXT UNIQUE NOT NULL, source_exchange_id TEXT NOT NULL, market_type TEXT NOT NULL, upstream_snapshot_or_retrieval_id TEXT NOT NULL, normalized_content_sha256 TEXT NOT NULL, canonical_record TEXT NOT NULL, previous_digest TEXT NOT NULL, record_digest TEXT UNIQUE NOT NULL, UNIQUE(source_exchange_id,market_type,upstream_snapshot_or_retrieval_id))")
                    db.execute("CREATE TABLE catalog_authority_head(singleton INTEGER PRIMARY KEY CHECK(singleton=1), committed_sequence INTEGER NOT NULL, committed_digest TEXT NOT NULL, last_snapshot_id TEXT)")
                    db.execute("INSERT INTO catalog_authority_metadata VALUES(1,?,1)", (_DOMAIN,))
                    db.execute("INSERT INTO source_metadata_authority_head VALUES(1,0,?,NULL)", ("0" * 64,))
                    db.execute("INSERT INTO catalog_authority_head VALUES(1,0,?,NULL)", ("0" * 64,))
                self._validate_catalog_domain(db)
                self._replay(db)
                db.commit()
            except Exception:
                db.rollback()
                raise

    @staticmethod
    def _validate_catalog_domain(db: sqlite3.Connection) -> None:
        tables = {
            row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "catalog_authority_metadata" not in tables:
            raise ValueError("CATALOG_AUTHORITY_DOMAIN_MISSING")
        rows = db.execute(
            "SELECT singleton,authority_domain,schema_version FROM catalog_authority_metadata"
        ).fetchall()
        if rows != [(1, _DOMAIN, 1)]:
            raise ValueError("CATALOG_AUTHORITY_DOMAIN_MISMATCH")

    @staticmethod
    def _snapshot_digest(sequence: int, canonical: str, previous: str) -> str:
        return hashlib.sha256((f"{_DOMAIN}\n{sequence}\n{previous}\n{canonical}").encode()).hexdigest()

    @staticmethod
    def _metadata_digest(
        sequence: int, introduced_by_snapshot_id: str, canonical: str, previous: str
    ) -> str:
        return hashlib.sha256(
            (f"{_DOMAIN}\nsource-metadata\n{sequence}\n{introduced_by_snapshot_id}\n{previous}\n{canonical}").encode()
        ).hexdigest()

    def _replay(self, db: sqlite3.Connection) -> tuple[dict[str, Any], ...]:
        self._validate_catalog_domain(db)
        snapshot_head = db.execute("SELECT committed_sequence,committed_digest,last_snapshot_id FROM catalog_authority_head WHERE singleton=1").fetchone()
        snapshot_rows = db.execute(
            "SELECT snapshot_sequence,accepted_source_catalog_snapshot_id,source_exchange_id,"
            "market_type,upstream_snapshot_or_retrieval_id,normalized_content_sha256,"
            "canonical_record,previous_digest,record_digest "
            "FROM accepted_catalog_snapshots ORDER BY snapshot_sequence"
        ).fetchall()
        if snapshot_head is None or snapshot_head[0] != len(snapshot_rows):
            raise ValueError("CORRUPT_CATALOG_AUTHORITY")
        previous = "0" * 64
        snapshots: list[dict[str, Any]] = []
        snapshots_by_id: dict[str, dict[str, Any]] = {}
        scope_heads: dict[tuple[str, str], str] = {}
        canonical_retrievals: set[tuple[str, str, str]] = set()
        membership_state = self._membership._replay(SQLiteMembershipCarrier._validated_rows(db))
        for expected, row in enumerate(snapshot_rows, 1):
            (
                sequence, row_snapshot_id, row_exchange_id, row_market_type,
                row_retrieval_id, row_content_hash, canonical, row_previous, digest,
            ) = row
            if sequence != expected or row_previous != previous or digest != self._snapshot_digest(sequence, canonical, previous):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            raw = json.loads(canonical)
            if not validate_accepted_source_catalog_snapshot(raw):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            evidence = raw["completeness_evidence"]
            if (
                row_snapshot_id != raw["accepted_source_catalog_snapshot_id"]
                or row_exchange_id != raw["source_exchange_id"]
                or row_market_type != raw["market_type"]
                or row_retrieval_id != raw["upstream_snapshot_or_retrieval_id"]
                or row_content_hash != evidence.get("normalized_content_sha256")
            ):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            key = (raw["source_exchange_id"], raw["market_type"])
            retrieval_key = (*key, raw["upstream_snapshot_or_retrieval_id"])
            if retrieval_key in canonical_retrievals:
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            canonical_retrievals.add(retrieval_key)
            if raw["previous_snapshot_id"] != scope_heads.get(key):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            grant = membership_state.grants.get(raw["accepted_source_producer_membership_id"])
            accepted_at = _time(raw["effective_at_utc"])
            terminal = None if grant is None else membership_state.terminal_by_membership.get(grant.accepted_source_producer_membership_id)
            cutoff = None if terminal is None else max(_time(terminal.effective_at_utc), _time(terminal.authority_admitted_at_utc))
            identity = {name: raw[name] for name in ("source_exchange_id", "market_type", "source_adapter_family_id", "source_adapter_implementation_id", "source_adapter_release_id", "source_adapter_version")}
            if (grant is None or accepted_at is None or grant.producer_generation != raw["source_producer_generation"] or grant.content_fingerprint != raw["source_producer_membership_fingerprint"] or not _exact(grant, identity) or _time(grant.effective_at_utc) > accepted_at or _time(grant.authority_admitted_at_utc) > accepted_at or (cutoff is not None and cutoff <= accepted_at)):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            identifier = raw["accepted_source_catalog_snapshot_id"]
            if identifier in snapshots_by_id:
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            snapshots.append(raw); snapshots_by_id[identifier] = raw; scope_heads[key] = identifier
            previous = digest
        if snapshot_head[1] != previous or snapshot_head[2] != (snapshots[-1]["accepted_source_catalog_snapshot_id"] if snapshots else None):
            raise ValueError("CORRUPT_CATALOG_AUTHORITY")

        metadata_head = db.execute("SELECT committed_sequence,committed_digest,last_metadata_version_id FROM source_metadata_authority_head WHERE singleton=1").fetchone()
        metadata_rows = db.execute("SELECT source_metadata_sequence,source_metadata_version_id,source_exchange_id,market_type,venue_symbol,metadata_version,introduced_by_snapshot_id,canonical_record,previous_metadata_digest,metadata_record_digest FROM source_metadata_versions ORDER BY source_metadata_sequence").fetchall()
        if metadata_head is None or metadata_head[0] != len(metadata_rows):
            raise ValueError("CORRUPT_CATALOG_AUTHORITY")
        previous = "0" * 64
        metadata_by_id: dict[str, dict[str, Any]] = {}
        metadata_heads: dict[tuple[str, str, str], tuple[int, str, datetime]] = {}
        for expected, row in enumerate(metadata_rows, 1):
            sequence, metadata_id, exchange_id, market_type, symbol, version, introduced_id, canonical, row_previous, digest = row
            raw = json.loads(canonical)
            key = (exchange_id, market_type, symbol)
            prior = metadata_heads.get(key)
            effective = _time(raw.get("effective_at_utc"))
            introducing = snapshots_by_id.get(introduced_id)
            expected_predecessor = None if prior is None else prior[1]
            if (sequence != expected or row_previous != previous or digest != self._metadata_digest(sequence, introduced_id, canonical, previous) or metadata_id in metadata_by_id or set(raw) != _METADATA_FIELDS or type(metadata_id) is not str or not metadata_id.startswith("smeta_") or raw.get("source_metadata_version_id") != metadata_id or not all(type(item) is str and bool(item) for item in key) or (raw.get("source_exchange_id"), raw.get("market_type"), raw.get("venue_symbol")) != key or type(raw.get("metadata_version")) is not int or raw["metadata_version"] < 1 or raw["metadata_version"] != version or version != (1 if prior is None else prior[0] + 1) or raw.get("previous_source_metadata_version_id") != expected_predecessor or (expected_predecessor is not None and type(expected_predecessor) is not str) or not _valid_normalized_product(raw.get("normalized_metadata")) or raw["normalized_metadata"]["venue_symbol"] != symbol or raw["normalized_metadata"]["market_type"] != market_type or type(raw.get("content_fingerprint")) is not str or raw.get("content_fingerprint") != _digest(_METADATA_FP_DOMAIN, raw, {"source_metadata_version_id", "content_fingerprint"}) or type(introduced_id) is not str or introducing is None or effective is None or effective > _time(introducing["effective_at_utc"]) or (prior is not None and effective < prior[2])):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            references = [member for member in introducing["member_source_product_metadata_versions"] if member["source_metadata_version_id"] == metadata_id]
            if len(references) != 1 or (references[0]["source_exchange_id"], references[0]["market_type"], references[0]["venue_symbol"]) != key or key[:2] != (introducing["source_exchange_id"], introducing["market_type"]):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
            metadata_by_id[metadata_id] = raw; metadata_heads[key] = (version, metadata_id, effective); previous = digest
        if metadata_head[1] != previous or metadata_head[2] != (metadata_rows[-1][1] if metadata_rows else None):
            raise ValueError("CORRUPT_CATALOG_AUTHORITY")
        for snapshot in snapshots:
            products: list[dict[str, Any]] = []
            for member in snapshot["member_source_product_metadata_versions"]:
                metadata = metadata_by_id.get(member["source_metadata_version_id"])
                key = (member["source_exchange_id"], member["market_type"], member["venue_symbol"])
                if metadata is None or (metadata["source_exchange_id"], metadata["market_type"], metadata["venue_symbol"]) != key or key[:2] != (snapshot["source_exchange_id"], snapshot["market_type"]):
                    raise ValueError("CORRUPT_CATALOG_AUTHORITY")
                products.append(metadata["normalized_metadata"])
            products.sort(key=lambda product: product["venue_symbol"])
            rebuilt_hash = hashlib.sha256(_canonical(products).encode()).hexdigest()
            evidence = snapshot["completeness_evidence"]
            if (
                set(evidence) != {
                    "policy", "atomic_response", "terminal", "product_count",
                    "normalized_content_sha256",
                }
                or evidence["policy"] != "BINANCE_EXCHANGE_INFO_ATOMIC_V1"
                or evidence["atomic_response"] is not True
                or evidence["terminal"] is not True
                or type(evidence["product_count"]) is not int
                or evidence["product_count"] != len(snapshot["member_source_product_metadata_versions"])
                or evidence["normalized_content_sha256"] != rebuilt_hash
            ):
                raise ValueError("CORRUPT_CATALOG_AUTHORITY")
        return tuple(snapshots)

    def fetch_catalog_once(self, release_binding: object) -> AcceptedSourceCatalogSnapshot | None:
        """Construct, invoke, normalize and durably accept; malformed/failing input denies."""
        producer_type = _RELEASE_PRODUCERS.get(release_binding) if type(release_binding) is str else None
        if producer_type is not _BinanceSpotCatalogProducer:
            return None
        try:
            producer = producer_type()
            raw = producer.fetch()
            observation = _normalize_binance(raw)
            if observation is None:
                return None
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                result = self._accept_locked(db, producer, observation)
                if result is None:
                    db.rollback()
                else:
                    db.commit()
                return result
        except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError, sqlite3.Error):
            return None

    def _accept_locked(self, db: sqlite3.Connection, producer: _BinanceSpotCatalogProducer, observation: _Observation) -> AcceptedSourceCatalogSnapshot | None:
        self._validate_catalog_domain(db)
        snapshots = self._replay(db)
        state = self._membership._replay(SQLiteMembershipCarrier._validated_rows(db))
        now = PRODUCTION_CORE_CLOCK.now_utc(); when = _time(now); observed = _time(observation.observed_at_utc)
        if when is None or observed is None or observed > when:
            return None
        candidates = []
        for grant in state.grants.values():
            terminal = state.terminal_by_membership.get(grant.accepted_source_producer_membership_id)
            cutoff = None if terminal is None else max(_time(terminal.effective_at_utc), _time(terminal.authority_admitted_at_utc))
            if _exact(grant, producer.identity) and grant.producer_generation == producer.generation and _time(grant.effective_at_utc) <= when and _time(grant.authority_admitted_at_utc) <= when and (cutoff is None or when < cutoff):
                candidates.append(grant)
        if len(candidates) != 1:
            return None
        membership = candidates[0]
        existing_row = db.execute("SELECT canonical_record,normalized_content_sha256 FROM accepted_catalog_snapshots WHERE source_exchange_id=? AND market_type=? AND upstream_snapshot_or_retrieval_id=?", (membership.source_exchange_id, membership.market_type, observation.retrieval_id)).fetchone()
        content_hash = observation.completeness_evidence["normalized_content_sha256"]
        if existing_row is not None:
            if existing_row[1] != content_hash:
                return None
            existing = json.loads(existing_row[0])
            return AcceptedSourceCatalogSnapshot(**{**existing, "member_source_product_metadata_versions": tuple(existing["member_source_product_metadata_versions"])})
        sequence = len(snapshots) + 1
        snapshot_id = f"ascat_{sequence:020d}"
        members: list[dict[str, Any]] = []
        pending_metadata: list[tuple[str, tuple[str, str, str], int, dict[str, Any]]] = []
        for product in observation.products:
            key = (membership.source_exchange_id, membership.market_type, product["venue_symbol"])
            history = db.execute("SELECT canonical_record FROM source_metadata_versions WHERE source_exchange_id=? AND market_type=? AND venue_symbol=? ORDER BY metadata_version", key).fetchall()
            previous_record = json.loads(history[-1][0]) if history else None
            if previous_record is not None and previous_record["normalized_metadata"] == dict(product):
                metadata_id = previous_record["source_metadata_version_id"]
            else:
                version = len(history) + 1
                material = {"source_exchange_id": key[0], "market_type": key[1], "venue_symbol": key[2], "metadata_version": version, "previous_source_metadata_version_id": None if previous_record is None else previous_record["source_metadata_version_id"], "normalized_metadata": dict(product), "effective_at_utc": now}
                fingerprint = _digest(_METADATA_FP_DOMAIN, material, set())
                metadata_id = f"smeta_{hashlib.sha256((str(key)+str(version)+now).encode()).hexdigest()[:24]}"
                pending_metadata.append((metadata_id, key, version, {"source_metadata_version_id": metadata_id, **material, "content_fingerprint": fingerprint}))
            members.append({"source_exchange_id": key[0], "market_type": key[1], "venue_symbol": key[2], "source_metadata_version_id": metadata_id})
        scope = (membership.source_exchange_id, membership.market_type)
        previous_snapshot = next((item["accepted_source_catalog_snapshot_id"] for item in reversed(snapshots) if (item["source_exchange_id"], item["market_type"]) == scope), None)
        stale = (when + timedelta(seconds=producer.freshness_seconds)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        material = {"source_exchange_id": scope[0], "market_type": scope[1], "source_adapter_family_id": membership.source_adapter_family_id, "source_adapter_implementation_id": membership.source_adapter_implementation_id, "source_adapter_release_id": membership.source_adapter_release_id, "source_adapter_version": membership.source_adapter_version, "accepted_source_producer_membership_id": membership.accepted_source_producer_membership_id, "source_producer_generation": membership.producer_generation, "source_producer_membership_fingerprint": membership.content_fingerprint, "upstream_snapshot_or_retrieval_id": observation.retrieval_id, "observed_at_utc": observation.observed_at_utc, "effective_at_utc": now, "stale_after_utc": stale, "previous_snapshot_id": previous_snapshot, "completeness_status": "COMPLETE", "completeness_evidence": dict(observation.completeness_evidence), "acceptance_status": "VALID", "member_source_product_metadata_versions": members}
        fingerprint = _fingerprint("cryptohunter.m0.5.accepted_source_catalog_snapshot.v2", SOURCE_FINGERPRINT_FIELDS, material)
        record = {"accepted_source_catalog_snapshot_id": snapshot_id, **material, "content_fingerprint": fingerprint}
        if set(record) != SOURCE_FIELDS or not validate_accepted_source_catalog_snapshot(record) or any(set(member) != SOURCE_MEMBER_FIELDS for member in members):
            raise ValueError("INVALID_CANONICAL_SOURCE_SNAPSHOT")
        metadata_head = db.execute("SELECT committed_sequence,committed_digest FROM source_metadata_authority_head WHERE singleton=1").fetchone()
        metadata_sequence, metadata_previous = metadata_head
        for metadata_id, key, version, metadata_record in pending_metadata:
            metadata_sequence += 1; canonical_metadata = _canonical(metadata_record)
            metadata_digest = self._metadata_digest(
                metadata_sequence, snapshot_id, canonical_metadata, metadata_previous
            )
            db.execute("INSERT INTO source_metadata_versions VALUES(?,?,?,?,?,?,?,?,?,?)", (metadata_sequence, metadata_id, *key, version, snapshot_id, canonical_metadata, metadata_previous, metadata_digest))
            metadata_previous = metadata_digest
        if pending_metadata:
            db.execute("UPDATE source_metadata_authority_head SET committed_sequence=?,committed_digest=?,last_metadata_version_id=? WHERE singleton=1", (metadata_sequence, metadata_previous, pending_metadata[-1][0]))
        canonical = _canonical(record)
        head = db.execute("SELECT committed_digest FROM catalog_authority_head WHERE singleton=1").fetchone()[0]
        digest = self._snapshot_digest(sequence, canonical, head)
        db.execute("INSERT INTO accepted_catalog_snapshots VALUES(?,?,?,?,?,?,?,?,?)", (sequence, snapshot_id, *scope, observation.retrieval_id, content_hash, canonical, head, digest))
        db.execute("UPDATE catalog_authority_head SET committed_sequence=?,committed_digest=?,last_snapshot_id=? WHERE singleton=1", (sequence, digest, snapshot_id))
        self._replay(db)
        return AcceptedSourceCatalogSnapshot(**{**record, "member_source_product_metadata_versions": tuple(members)})

    def snapshots(self) -> tuple[AcceptedSourceCatalogSnapshot, ...]:
        with self._connect() as db:
            return tuple(AcceptedSourceCatalogSnapshot(**{**raw, "member_source_product_metadata_versions": tuple(raw["member_source_product_metadata_versions"])}) for raw in self._replay(db))

    def metadata_history(self, source_exchange_id: object, market_type: object, venue_symbol: object) -> tuple[SourceProductMetadataVersion, ...]:
        if not all(type(value) is str for value in (source_exchange_id, market_type, venue_symbol)):
            return ()
        with self._connect() as db:
            self._replay(db)
            rows = db.execute("SELECT canonical_record FROM source_metadata_versions WHERE source_exchange_id=? AND market_type=? AND venue_symbol=? ORDER BY metadata_version", (source_exchange_id, market_type, venue_symbol)).fetchall()
            return tuple(SourceProductMetadataVersion(**json.loads(row[0])) for row in rows)


__all__ = ["AcceptedSourceCatalogSnapshot", "CatalogRuntimeAcceptanceAuthority", "SourceProductMetadataVersion"]
