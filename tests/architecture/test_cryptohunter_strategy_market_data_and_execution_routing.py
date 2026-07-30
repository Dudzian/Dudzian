"""Fail-closed, data-driven validators for the CryptoHunter M0.6 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "docs/architecture/cryptohunter_product_architecture"
DOC = ARCH / "strategy_market_data_and_execution_routing.json"
MD = ARCH / "strategy_market_data_and_execution_routing.md"
README = ARCH / "README.md"
M04 = ARCH / "environment_and_product_capabilities.json"
M05 = ARCH / "exchange_accounts_and_instruments.json"


def load(path):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, f"duplicate JSON key: {key}"
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)


DATA = load(DOC)
SCHEMAS = DATA["record_schema_registry"]
ENUMS = DATA["enum_registry"]
DENIALS = set(DATA["denial_code_registry"])
UUID = "01900000-0000-7000-8000-000000000001"
NOW = "2026-07-28T12:00:00Z"


def did(entity, suffix=""):
    prefix = DATA["canonical_id_policy"]["prefixes"][entity]
    value = UUID[:-1] + (suffix or "1")
    return prefix + value


def decision(
    ok, code=None, transition=None, planned_instance_state=None, planned_definition_state=None
):
    assert code is None or code in DENIALS
    return {
        "allowed": ok,
        "denial_code": code,
        "planned_transition": transition,
        "planned_instance_state": planned_instance_state,
        "planned_definition_state": planned_definition_state,
    }


def canonical_hash(configuration, domain=None):
    contract = DATA["canonical_hash_contract"]
    domain = contract["domain_separator"] if domain is None else domain
    raw = (
        domain
        + "\n"
        + json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def valid_constraint(value, constraint):
    kind = constraint["kind"]
    if kind in {"durable_id", "durable_id_or_null"}:
        if value is None:
            return kind == "durable_id_or_null"
        prefix = DATA["canonical_id_policy"]["prefixes"][constraint["entity_ref"]]
        return (
            isinstance(value, str)
            and value.startswith(prefix)
            and bool(re.fullmatch(DATA["canonical_id_policy"]["uuid_regex"], value[len(prefix) :]))
        )
    if kind == "route_durable_id":
        return any(
            valid_constraint(value, {"kind": "durable_id", "entity_ref": entity})
            for entity in ("MarketDataRoute", "ExecutionRoute")
        )
    if kind == "enum":
        return isinstance(value, str) and value in ENUMS[constraint["registry_ref"]]
    if kind == "unique_enum_array":
        return (
            isinstance(value, list)
            and (value != [] or not constraint["non_empty"])
            and len(value) == len(set(value))
            and all(
                valid_constraint(item, {"kind": "enum", "registry_ref": constraint["registry_ref"]})
                for item in value
            )
        )
    if kind == "unique_durable_id_array":
        return (
            isinstance(value, list)
            and (value != [] or not constraint["non_empty"])
            and len(value) == len(set(value))
            and all(
                valid_constraint(
                    item, {"kind": "durable_id", "entity_ref": constraint["entity_ref"]}
                )
                for item in value
            )
        )
    if kind == "positive_integer":
        return isinstance(value, int) and not isinstance(value, bool) and value > 0
    if kind == "utc_timestamp":
        return timestamp(value) is not None
    if kind == "lowercase_sha256":
        return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))
    if kind == "constant":
        return value == constraint["value"]
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "non_empty_string":
        return isinstance(value, str) and bool(value)
    if kind == "closed_object":
        return exact_object(value, DATA["nested_schema_registry"][constraint["schema_ref"]])
    if kind == "closed_scalar_map":
        return isinstance(value, dict) and all(
            isinstance(key, str)
            and key
            and (isinstance(item, (str, int, bool)) and not isinstance(item, float))
            for key, item in value.items()
        )
    return False


def exact_object(record, schema):
    return (
        isinstance(record, dict)
        and set(record) == set(schema["fields"])
        and all(
            valid_constraint(record[field], schema["types"][field]) for field in schema["fields"]
        )
    )


def exact_record(record, schema_name):
    if not exact_object(record, SCHEMAS[schema_name]):
        return False
    if schema_name == "StrategyDefinition":
        if record["canonical_content_hash"] != canonical_hash(
            record["configuration"], record["hash_domain_separator"]
        ):
            return False
    return True


def resolve_strategy_definition_version(strategy_definition_id, definition_version, context):
    current = context["strategy_definitions_by_id"].get(strategy_definition_id)
    if current is None:
        return None
    if current["definition_version"] == definition_version:
        return current
    if definition_version < current["definition_version"]:
        return context["previous_strategy_definitions_by_version_key"].get(
            f"{strategy_definition_id}@{definition_version}"
        )
    return None


def validate_context(context):
    spec = DATA["validation_context_schema"]
    if not isinstance(context, dict) or set(context) != set(spec["fields"]):
        return decision(False, "TRUSTED_CONTEXT_INVALID")
    for field, constraint in spec["scalar_fields"].items():
        if not valid_constraint(context[field], constraint):
            return decision(False, "TRUSTED_CONTEXT_INVALID")
    for map_name, map_spec in spec["map_fields"].items():
        records = context[map_name]
        if not isinstance(records, dict) or (not records and not map_spec["empty_allowed"]):
            return decision(False, "TRUSTED_CONTEXT_INVALID")
        schema = SCHEMAS[map_spec["schema_ref"]]
        for key, record in records.items():
            if not exact_record(record, map_spec["schema_ref"]):
                return decision(False, "TRUSTED_CONTEXT_INVALID")
            expected = record[schema["id_field"]]
            if map_spec["key_rule"] == "canonical_environment":
                expected = record["environment"]
            elif map_spec["key_rule"].startswith("<strategy"):
                expected = f"{record['strategy_definition_id']}@{record['definition_version']}"
            if key != expected:
                return decision(False, "TRUSTED_CONTEXT_INVALID")
    allowed_by_environment = DATA["allowed_product_capabilities_by_environment"]
    for item in context["product_capabilities_by_environment"].values():
        if (
            not set(item["capability_set"]) <= set(allowed_by_environment[item["environment"]])
            or item["private_execution_allowed"]
            is not DATA["private_execution_allowed_by_environment"][item["environment"]]
        ):
            return decision(False, "TRUSTED_CONTEXT_INVALID")
    for route in context["execution_routes_by_id"].values():
        if not set(route["route_capability_ceiling"]) <= set(
            allowed_by_environment[route["environment"]]
        ):
            return decision(False, "TRUSTED_CONTEXT_INVALID")
    for instance in context["strategy_instances_by_id"].values():
        count = sum(
            item is not None
            for item in (instance["market_data_route_id"], instance["execution_route_id"])
        )
        if (instance["lifecycle_state"] == "DRAFT" and count > 1) or (
            instance["lifecycle_state"] in {"BOUND", "ACTIVE", "INACTIVE"} and count != 2
        ):
            return decision(False, "TRUSTED_CONTEXT_INVALID")
    active = context["active_strategy_instances"]
    expected_active = {
        key
        for key, item in context["strategy_instances_by_id"].items()
        if item["lifecycle_state"] == "ACTIVE"
    }
    if set(active) != expected_active or any(
        item not in context["strategy_instances_by_id"] for item in active
    ):
        return decision(False, "TRUSTED_CONTEXT_INVALID")
    for current in context["strategy_definitions_by_id"].values():
        history = context["previous_strategy_definitions_by_version_key"]
        expected = {
            f"{current['strategy_definition_id']}@{version}"
            for version in range(1, current["definition_version"])
        }
        actual = {key for key in history if key.startswith(current["strategy_definition_id"] + "@")}
        if actual != expected:
            return decision(False, "TRUSTED_CONTEXT_INVALID")
        for key in actual:
            old = history[key]
            if any(
                old[field] != current[field]
                for field in ("strategy_definition_id", "workspace_id", "strategy_type_id")
            ):
                return decision(False, "TRUSTED_CONTEXT_INVALID")
    current_ids = set(context["strategy_definitions_by_id"])
    if any(
        record["strategy_definition_id"] not in current_ids
        for record in context["previous_strategy_definitions_by_version_key"].values()
    ):
        return decision(False, "TRUSTED_CONTEXT_INVALID")
    if not validate_references(context):
        return decision(False, "TRUSTED_CONTEXT_INVALID")
    return decision(True)


def validate_references(context):
    definitions, instances = (
        context["strategy_definitions_by_id"],
        context["strategy_instances_by_id"],
    )
    accounts, universes = context["accounts_by_id"], context["universes_by_id"]
    instruments, catalogs = context["instruments_by_id"], context["catalogs_by_id"]
    market_routes, execution_routes = (
        context["market_data_routes_by_id"],
        context["execution_routes_by_id"],
    )
    snapshots, credentials = (
        context["account_capability_snapshots_by_id"],
        context["credential_profiles_by_id"],
    )
    for item in instances.values():
        definition, account, universe = (
            resolve_strategy_definition_version(
                item["strategy_definition_id"], item["strategy_definition_version"], context
            ),
            accounts.get(item["exchange_account_id"]),
            universes.get(item["trading_universe_id"]),
        )
        if (
            definition is None
            or account is None
            or universe is None
            or definition["workspace_id"] != item["workspace_id"]
            or account["portfolio_id"] != item["portfolio_id"]
            or universe["exchange_account_id"] != item["exchange_account_id"]
        ):
            return False
        if (
            item["market_data_route_id"] is not None
            and item["market_data_route_id"] not in market_routes
        ):
            return False
        if (
            item["execution_route_id"] is not None
            and item["execution_route_id"] not in execution_routes
        ):
            return False
        market = market_routes.get(item["market_data_route_id"])
        execution = execution_routes.get(item["execution_route_id"])
        dimensions = ("exchange_id", "environment", "market_type")
        if market is not None and (
            market["workspace_id"] != item["workspace_id"]
            or any(market[field] != account[field] for field in dimensions)
        ):
            return False
        if execution is not None and (
            execution["workspace_id"] != item["workspace_id"]
            or execution["exchange_account_id"] != item["exchange_account_id"]
            or any(execution[field] != account[field] for field in dimensions)
        ):
            return False
        if execution is not None:
            for instrument_id in universe["instrument_ids"]:
                instrument = instruments.get(instrument_id)
                catalog = (
                    None if instrument is None else catalogs.get(instrument["catalog_snapshot_id"])
                )
                if (
                    instrument is None
                    or catalog is None
                    or execution["adapter_family_id"] != instrument["source_adapter_family_id"]
                    or execution["adapter_family_id"] != catalog["adapter_family_id"]
                ):
                    return False
    for route in market_routes.values():
        for iid in route["instrument_ids"]:
            instrument = instruments.get(iid)
            if (
                instrument is None
                or any(
                    instrument[f] != route[f]
                    for f in ("workspace_id", "exchange_id", "environment", "market_type")
                )
                or instrument["source_adapter_family_id"] != route["adapter_family_id"]
            ):
                return False
    for route in execution_routes.values():
        account = accounts.get(route["exchange_account_id"])
        if account is None or any(
            account[f] != route[f] for f in ("exchange_id", "environment", "market_type")
        ):
            return False
    for account in accounts.values():
        snapshot = snapshots.get(account["account_capability_snapshot_id"])
        if snapshot is None or (
            snapshot["exchange_account_id"] != account["exchange_account_id"]
            or any(
                snapshot[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
        ):
            return False
        credential_id = account["active_credential_profile_id"]
        if account["environment"] == "TESTNET" and credential_id is None:
            return False
        if credential_id is not None:
            credential = credentials.get(credential_id)
            if (
                credential is None
                or credential["exchange_account_id"] != account["exchange_account_id"]
                or credential["exchange_id"] != account["exchange_id"]
                or credential["environment_scope"] != account["environment"]
                or credential["lifecycle_state"] != "ACTIVE"
                or (
                    account["environment"] == "TESTNET"
                    and (
                        credential["credential_purpose"] != "ORDER_ENTRY"
                        or "PLACE_ORDERS" not in credential["permission_snapshot"]
                    )
                )
            ):
                return False
    for universe in universes.values():
        account = accounts.get(universe["exchange_account_id"])
        if (
            account is None
            or any(i not in instruments for i in universe["instrument_ids"])
            or any(c not in catalogs for c in universe["source_catalog_snapshot_ids"])
        ):
            return False
        used_catalogs = {
            instruments[iid]["catalog_snapshot_id"] for iid in universe["instrument_ids"]
        }
        if set(universe["source_catalog_snapshot_ids"]) != used_catalogs:
            return False
        for iid in universe["instrument_ids"]:
            instrument = instruments[iid]
            catalog = catalogs.get(instrument["catalog_snapshot_id"])
            if catalog is None or any(
                item[field] != account[field]
                for item in (instrument, catalog)
                for field in ("exchange_id", "environment", "market_type")
            ):
                return False
    for instrument in instruments.values():
        catalog = catalogs.get(instrument["catalog_snapshot_id"])
        if (
            catalog is None
            or instrument["instrument_id"] not in catalog["instrument_ids"]
            or instrument["source_adapter_family_id"] != catalog["adapter_family_id"]
        ):
            return False
    for catalog in catalogs.values():
        for iid in catalog["instrument_ids"]:
            if (
                iid not in instruments
                or instruments[iid]["catalog_snapshot_id"] != catalog["catalog_snapshot_id"]
            ):
                return False
    now = timestamp(context["validation_time_utc"])
    for snapshot in snapshots.values():
        account = accounts.get(snapshot["exchange_account_id"])
        validated, fresh = (
            timestamp(snapshot["validated_at_utc"]),
            timestamp(snapshot["fresh_until_utc"]),
        )
        if (
            account is None
            or account["account_capability_snapshot_id"]
            != snapshot["account_capability_snapshot_id"]
            or any(
                snapshot[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or snapshot["status"] != "VALID"
            or now is None
            or validated is None
            or fresh is None
            or not (validated <= now < fresh)
            or snapshot["source_content_hash"] != snapshot["attested_source_content_hash"]
        ):
            return False
    for credential in credentials.values():
        account = accounts.get(credential["exchange_account_id"])
        if (
            account is None
            or account["active_credential_profile_id"] != credential["credential_profile_id"]
        ):
            return False
    for ready in context["route_readiness_by_id"].values():
        expected = market_routes if ready["route_kind"] == "MARKET_DATA" else execution_routes
        if ready["route_id"] not in expected:
            return False
    return True


def validate_request(operation, request):
    spec = DATA["operation_registry"].get(operation)
    if spec is None:
        return decision(False, "UNKNOWN_OPERATION")
    if not isinstance(request, dict) or set(request) != set(spec["request_fields"]):
        return decision(False, "REQUEST_SCHEMA_INVALID")
    if not all(
        valid_constraint(request[field], spec["request_types"][field])
        for field in spec["request_fields"]
    ):
        return decision(False, "REQUEST_SCHEMA_INVALID")
    return decision(True)


def route_readiness(route, kind, context):
    ready = context["route_readiness_by_id"].get(route[SCHEMAS[type_name(route)]["id_field"]])
    code = "MARKET_DATA_ROUTE_NOT_READY" if kind == "MARKET_DATA" else "EXECUTION_ROUTE_NOT_READY"
    if ready is None or ready["route_kind"] != kind or ready["readiness_state"] != "READY":
        return decision(False, code)
    now, observed = timestamp(context["validation_time_utc"]), timestamp(ready["observed_at"])
    if now is None or observed is None or observed > now:
        return decision(False, "MARKET_DATA_FRESHNESS_BLOCKED" if kind == "MARKET_DATA" else code)
    if (
        kind == "EXECUTION"
        and (now - observed).total_seconds() * 1000 > DATA["execution_readiness_max_age_ms"]
    ):
        return decision(False, "EXECUTION_ROUTE_NOT_READY")
    if kind == "MARKET_DATA":
        if (now - observed).total_seconds() * 1000 > route["freshness_policy"]["max_age_ms"]:
            return decision(False, "MARKET_DATA_FRESHNESS_BLOCKED")
        allowed = DATA["route_readiness_authority"]["market_sequence_required"][
            route["sequence_policy"]
        ]
        if ready["sequence_state"] not in allowed:
            return decision(False, "MARKET_DATA_SEQUENCE_INVALID")
    return decision(True)


def type_name(route):
    return "MarketDataRoute" if "market_data_route_id" in route else "ExecutionRoute"


def route_boundary(route):
    kind = "MARKET_DATA" if "market_data_route_id" in route else "EXECUTION"
    policy = DATA["endpoint_class_policy"].get(route["endpoint_class"])
    if (
        policy is None
        or policy["environment"] != route["environment"]
        or kind not in policy["allowed_route_kinds"]
    ):
        return decision(False, "ENDPOINT_FALLBACK_FORBIDDEN")
    if kind == "MARKET_DATA" and policy["access_scope"] != route["data_scope"]:
        return decision(False, "ENDPOINT_FALLBACK_FORBIDDEN")
    if (
        kind == "MARKET_DATA"
        and route["data_scope"] == "PUBLIC"
        and set(route["channel_types"]) & {"PRIVATE_BALANCES", "PRIVATE_ORDERS"}
    ):
        return decision(False, "ROUTE_CAPABILITY_BLOCKED")
    return decision(True)


def route_scope(instance, account, universe, route, context, kind):
    not_ready = (
        "MARKET_DATA_ROUTE_NOT_READY" if kind == "MARKET_DATA" else "EXECUTION_ROUTE_NOT_READY"
    )
    if route["route_status"] != "ENABLED":
        return decision(False, not_ready)
    if route["workspace_id"] != instance["workspace_id"]:
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    if kind == "EXECUTION" and route["exchange_account_id"] != instance["exchange_account_id"]:
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    if universe["exchange_account_id"] != account["exchange_account_id"]:
        return decision(False, "TRADING_UNIVERSE_INVALID")
    if any(
        route[field] != account[field] for field in ("exchange_id", "environment", "market_type")
    ):
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    boundary = route_boundary(route)
    if not boundary["allowed"]:
        return boundary
    for instrument_id in universe["instrument_ids"]:
        instrument = context["instruments_by_id"].get(instrument_id)
        if instrument is None:
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        catalog = context["catalogs_by_id"].get(instrument["catalog_snapshot_id"])
        if (
            catalog is None
            or instrument["catalog_snapshot_id"] not in universe["source_catalog_snapshot_ids"]
        ):
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        if any(
            instrument[field] != value
            for field, value in (
                ("workspace_id", instance["workspace_id"]),
                ("exchange_id", account["exchange_id"]),
                ("environment", account["environment"]),
                ("market_type", account["market_type"]),
            )
        ):
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        if (
            instrument["source_adapter_family_id"] != route["adapter_family_id"]
            or catalog["adapter_family_id"] != route["adapter_family_id"]
        ):
            return decision(False, "ROUTE_ADAPTER_MISMATCH")
        if any(
            catalog[field] != account[field]
            for field in ("exchange_id", "environment", "market_type")
        ):
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        if (
            catalog["catalog_snapshot_id"] not in universe["source_catalog_snapshot_ids"]
            or instrument_id not in catalog["instrument_ids"]
        ):
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        if kind == "MARKET_DATA" and instrument_id not in route["instrument_ids"]:
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        if (
            kind == "EXECUTION"
            and instrument["instrument_type"] not in route["supported_instrument_types"]
        ):
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
    return decision(True)


def readiness_validation(request, context):
    instance = context["strategy_instances_by_id"].get(request["strategy_instance_id"])
    if instance is None:
        return decision(False, "STRATEGY_INSTANCE_NOT_FOUND")
    definition = resolve_strategy_definition_version(
        instance["strategy_definition_id"], instance["strategy_definition_version"], context
    )
    if definition is None:
        return decision(False, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    if definition["lifecycle_state"] != "ACTIVE":
        return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
    if definition["workspace_id"] != instance["workspace_id"]:
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    if instance["lifecycle_state"] == "RETIRED":
        return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
    if instance["lifecycle_state"] not in DATA["readiness_lifecycle_policy"]["strategy_instance"]:
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    if (instance["market_data_route_id"], instance["execution_route_id"]) != (
        request["market_data_route_id"],
        request["execution_route_id"],
    ):
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    market = context["market_data_routes_by_id"].get(request["market_data_route_id"])
    execution = context["execution_routes_by_id"].get(request["execution_route_id"])
    if market is None:
        return decision(False, "MARKET_DATA_ROUTE_NOT_FOUND")
    if execution is None:
        return decision(False, "EXECUTION_ROUTE_NOT_FOUND")
    if market["route_status"] != "ENABLED":
        return decision(False, "MARKET_DATA_ROUTE_NOT_READY")
    if execution["route_status"] != "ENABLED":
        return decision(False, "EXECUTION_ROUTE_NOT_READY")
    if instance["lifecycle_state"] == "RETIRED":
        return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
    account = context["accounts_by_id"].get(instance["exchange_account_id"])
    if account is None or account["lifecycle_state"] != "ACTIVE":
        return decision(False, "ACCOUNT_READINESS_BLOCKED")
    if (
        market["workspace_id"] != instance["workspace_id"]
        or execution["workspace_id"] != instance["workspace_id"]
        or execution["exchange_account_id"] != instance["exchange_account_id"]
    ):
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    if any(
        route[field] != account[field]
        for route in (market, execution)
        for field in ("exchange_id", "environment", "market_type")
    ):
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    for route in (market, execution):
        result = route_boundary(route)
        if not result["allowed"]:
            return result
    if any(
        market[field] != execution[field]
        for field in (
            "workspace_id",
            "exchange_id",
            "environment",
            "market_type",
            "adapter_family_id",
        )
    ):
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    universe = context["universes_by_id"].get(instance["trading_universe_id"])
    if universe is None or universe["lifecycle_state"] != "ACTIVE":
        return decision(False, "TRADING_UNIVERSE_INVALID")
    for route, kind in ((market, "MARKET_DATA"), (execution, "EXECUTION")):
        result = route_scope(instance, account, universe, route, context, kind)
        if not result["allowed"]:
            return result
        result = route_readiness(route, kind, context)
        if not result["allowed"]:
            return result
    return decision(
        True,
        transition=DATA["operation_registry"]["VALIDATE_ROUTE_READINESS"]["success_transition"],
    )


def deep_activation(request, context):
    instance = context["strategy_instances_by_id"].get(request["strategy_instance_id"])
    if instance is None:
        return decision(False, "STRATEGY_INSTANCE_NOT_FOUND")
    definition = resolve_strategy_definition_version(
        instance["strategy_definition_id"], instance["strategy_definition_version"], context
    )
    if definition is None:
        return decision(False, "STRATEGY_DEFINITION_NOT_FOUND")
    if definition["lifecycle_state"] != "ACTIVE":
        return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
    if (
        definition["strategy_definition_id"] != instance["strategy_definition_id"]
        or definition["definition_version"] != instance["strategy_definition_version"]
        or definition["workspace_id"] != instance["workspace_id"]
    ):
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    if definition["definition_version"] != request["expected_definition_version"]:
        return decision(False, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    if instance["lifecycle_state"] not in DATA["first_bind_policy"]["activation_allowed_states"]:
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    ready_request = {
        key: request[key]
        for key in DATA["operation_registry"]["VALIDATE_ROUTE_READINESS"]["request_fields"]
    }
    ready_result = readiness_validation(ready_request, context)
    if not ready_result["allowed"]:
        return ready_result
    market = context["market_data_routes_by_id"][request["market_data_route_id"]]
    execution = context["execution_routes_by_id"][request["execution_route_id"]]
    account = context["accounts_by_id"].get(instance["exchange_account_id"])
    if account is None or account["lifecycle_state"] != "ACTIVE":
        return decision(False, "ACCOUNT_READINESS_BLOCKED")
    if execution["environment"] == "TESTNET" and (
        account["connection_state"] != "CONNECTED"
        or account["execution_authorization"] != "ORDER_ENTRY_ALLOWED"
    ):
        return decision(False, "ACCOUNT_READINESS_BLOCKED")
    universe = context["universes_by_id"].get(instance["trading_universe_id"])
    if (
        universe is None
        or universe["lifecycle_state"] != "ACTIVE"
        or universe["exchange_account_id"] != account["exchange_account_id"]
    ):
        return decision(False, "TRADING_UNIVERSE_INVALID")
    snapshot = context["account_capability_snapshots_by_id"].get(
        account["account_capability_snapshot_id"]
    )
    if snapshot is None or snapshot["status"] != "VALID":
        return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    if snapshot["source_content_hash"] != snapshot["attested_source_content_hash"]:
        return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    if snapshot["exchange_account_id"] != account["exchange_account_id"] or any(
        snapshot[field] != account[field] for field in ("exchange_id", "environment", "market_type")
    ):
        return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    if snapshot["adapter_family_id"] != execution["adapter_family_id"]:
        return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    required_dependencies = set(
        DATA["authorization_dependencies_by_environment"][execution["environment"]]
    )
    if required_dependencies != set(execution["authorization_dependencies"]):
        return decision(False, "ROUTE_CAPABILITY_BLOCKED")
    now = timestamp(context["validation_time_utc"])
    validated = timestamp(snapshot["validated_at_utc"])
    fresh_until = timestamp(snapshot["fresh_until_utc"])
    if (
        now is None
        or validated is None
        or fresh_until is None
        or not (validated <= now < fresh_until)
        or not validated < fresh_until
    ):
        return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    required_permissions = set(
        DATA["required_account_permissions_by_operation_and_environment"][
            "ACTIVATE_STRATEGY_INSTANCE"
        ][execution["environment"]]
    )
    if not required_permissions <= set(snapshot["permission_set"]):
        return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    if execution["environment"] == "TESTNET":
        credential = context["credential_profiles_by_id"].get(
            account["active_credential_profile_id"]
        )
        if (
            credential is None
            or credential["lifecycle_state"] != "ACTIVE"
            or credential["credential_purpose"] != "ORDER_ENTRY"
            or credential["exchange_account_id"] != account["exchange_account_id"]
            or credential["exchange_id"] != account["exchange_id"]
            or credential["environment_scope"] != account["environment"]
            or not required_permissions <= set(credential["permission_snapshot"])
        ):
            return decision(False, "ACCOUNT_READINESS_BLOCKED")
    capabilities = context["product_capabilities_by_environment"].get(execution["environment"])
    if capabilities is None or capabilities["trust_state"] != "VALID":
        return decision(False, "PRODUCT_CAPABILITIES_BLOCKED")
    if execution["environment"] == "LIVE" and market["environment"] == "LIVE":
        return decision(False, "LIVE_EXECUTION_FORBIDDEN")
    if (
        account["portfolio_id"] != instance["portfolio_id"]
        or execution["exchange_account_id"] != instance["exchange_account_id"]
    ):
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    for route, kind in ((market, "MARKET_DATA"), (execution, "EXECUTION")):
        result = route_scope(instance, account, universe, route, context, kind)
        if not result["allowed"]:
            return result
    for instrument_id in universe["instrument_ids"]:
        instrument = context["instruments_by_id"].get(instrument_id)
        catalog = (
            None
            if instrument is None
            else context["catalogs_by_id"].get(instrument["catalog_snapshot_id"])
        )
        if (
            instrument is None
            or instrument["trading_status"] != "TRADING"
            or catalog is None
            or catalog["status"] != "VALID"
        ):
            return decision(False, "INSTRUMENT_SCOPE_MISMATCH")
        if instrument["instrument_type"] not in snapshot["supported_instrument_types"]:
            return decision(False, "CAPABILITY_SNAPSHOT_BLOCKED")
    required = set(
        DATA["required_product_capabilities_by_operation_and_environment"][
            "ACTIVATE_STRATEGY_INSTANCE"
        ][execution["environment"]]
    )
    ceiling = set(execution["route_capability_ceiling"])
    product = set(capabilities["capability_set"])
    if not required <= ceiling:
        return decision(False, "ROUTE_CAPABILITY_BLOCKED")
    if not required <= product or (
        execution["environment"] == "TESTNET" and not capabilities["private_execution_allowed"]
    ):
        return decision(False, "PRODUCT_CAPABILITIES_BLOCKED")
    return decision(
        True,
        transition=DATA["operation_registry"]["ACTIVATE_STRATEGY_INSTANCE"]["success_transition"],
    )


def bind_validate(operation, request, context, instance):
    market_bind = operation == "BIND_MARKET_DATA_ROUTE"
    field = "market_data_route_id" if market_bind else "execution_route_id"
    routes = context["market_data_routes_by_id" if market_bind else "execution_routes_by_id"]
    missing = "MARKET_DATA_ROUTE_NOT_FOUND" if market_bind else "EXECUTION_ROUTE_NOT_FOUND"
    if instance["lifecycle_state"] != "DRAFT" or instance[field] is not None:
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    route = routes.get(request[field])
    if route is None:
        return decision(False, missing)
    definition = resolve_strategy_definition_version(
        instance["strategy_definition_id"], instance["strategy_definition_version"], context
    )
    if definition is None:
        return decision(False, "STRATEGY_DEFINITION_NOT_FOUND")
    if definition["lifecycle_state"] != "ACTIVE":
        return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
    if definition["definition_version"] != instance["strategy_definition_version"]:
        return decision(False, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    if definition["workspace_id"] != instance["workspace_id"]:
        return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    account = context["accounts_by_id"].get(instance["exchange_account_id"])
    if account is None or account["lifecycle_state"] == "RETIRED":
        return decision(False, "ACCOUNT_READINESS_BLOCKED")
    universe = context["universes_by_id"].get(instance["trading_universe_id"])
    if universe is None or universe["lifecycle_state"] != "ACTIVE":
        return decision(False, "TRADING_UNIVERSE_INVALID")
    if (
        account["portfolio_id"] != instance["portfolio_id"]
        or universe["exchange_account_id"] != instance["exchange_account_id"]
    ):
        return decision(False, "ROUTE_SCOPE_MISMATCH")
    if not market_bind:
        required = set(DATA["authorization_dependencies_by_environment"][route["environment"]])
        if set(route["authorization_dependencies"]) != required:
            return decision(False, "ROUTE_CAPABILITY_BLOCKED")
    result = route_scope(
        instance, account, universe, route, context, "MARKET_DATA" if market_bind else "EXECUTION"
    )
    if not result["allowed"]:
        return result
    other = "execution_route_id" if market_bind else "market_data_route_id"
    state = "BOUND" if instance[other] is not None else "DRAFT"
    return decision(
        True,
        transition=DATA["operation_registry"][operation]["success_transition"],
        planned_instance_state=state,
    )


def direct_validate(operation, request, context):
    request_result = validate_request(operation, request)
    if not request_result["allowed"]:
        return request_result
    context_result = validate_context(context)
    if not context_result["allowed"]:
        return context_result
    transition = DATA["operation_registry"][operation]["success_transition"]
    if operation == "CREATE_STRATEGY_DEFINITION":
        if request["canonical_content_hash"] != canonical_hash(
            request["configuration"], request["hash_domain_separator"]
        ):
            return decision(False, "STRATEGY_DEFINITION_LINEAGE_INVALID")
        current = context["strategy_definitions_by_id"].get(request["strategy_definition_id"])
        if current is None and request["definition_version"] != 1:
            return decision(False, "STRATEGY_DEFINITION_VERSION_MISMATCH")
        if current is not None and request["definition_version"] == 1:
            return decision(False, "STRATEGY_DEFINITION_ID_COLLISION")
        if (
            current is not None
            and request["definition_version"] != current["definition_version"] + 1
        ):
            return decision(False, "STRATEGY_DEFINITION_VERSION_MISMATCH")
        if current is not None and (
            request["workspace_id"] != current["workspace_id"]
            or request["strategy_type_id"] != current["strategy_type_id"]
            or request["hash_domain_separator"] != current["hash_domain_separator"]
        ):
            return decision(False, "STRATEGY_DEFINITION_LINEAGE_INVALID")
        if current is not None and current["lifecycle_state"] == "RETIRED":
            return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
        return decision(True, transition=transition, planned_definition_state="DRAFT")
    if operation in {"ACTIVATE_STRATEGY_DEFINITION", "RETIRE_STRATEGY_DEFINITION"}:
        definition = context["strategy_definitions_by_id"].get(request["strategy_definition_id"])
        if definition is None:
            return decision(False, "STRATEGY_DEFINITION_NOT_FOUND")
        if operation == "RETIRE_STRATEGY_DEFINITION" and any(
            item["strategy_definition_id"] == definition["strategy_definition_id"]
            and item["strategy_definition_version"] == definition["definition_version"]
            and item["lifecycle_state"] == "ACTIVE"
            for item in context["strategy_instances_by_id"].values()
        ):
            return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
        targets = (
            DATA["strategy_definition_transition_matrix"]
            .get(definition["lifecycle_state"], {})
            .get(operation, [])
        )
        target = "ACTIVE" if operation.startswith("ACTIVATE") else "RETIRED"
        ok = definition["lifecycle_state"] == request["expected_state"] and target in targets
        return decision(
            ok,
            None
            if ok
            else (
                "RETIRED_RESOURCE_FORBIDDEN"
                if definition["lifecycle_state"] == "RETIRED"
                else "STRATEGY_INSTANCE_BINDING_MISMATCH"
            ),
            transition if ok else None,
            planned_definition_state=target if ok else None,
        )
    if operation == "CREATE_STRATEGY_INSTANCE":
        if request["strategy_instance_id"] in context["strategy_instances_by_id"]:
            return decision(False, "STRATEGY_INSTANCE_ID_COLLISION")
        definition = resolve_strategy_definition_version(
            request["strategy_definition_id"], request["strategy_definition_version"], context
        )
        if definition is None:
            return decision(
                False,
                "STRATEGY_DEFINITION_VERSION_MISMATCH"
                if request["strategy_definition_id"] in context["strategy_definitions_by_id"]
                else "STRATEGY_DEFINITION_NOT_FOUND",
            )
        current = context["strategy_definitions_by_id"].get(request["strategy_definition_id"])
        if (
            current is None
            or current["definition_version"] != request["strategy_definition_version"]
        ):
            return decision(False, "STRATEGY_DEFINITION_VERSION_MISMATCH")
        if definition["lifecycle_state"] != "ACTIVE":
            return decision(False, "RETIRED_RESOURCE_FORBIDDEN")
        if definition["workspace_id"] != request["workspace_id"]:
            return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
        account = context["accounts_by_id"].get(request["exchange_account_id"])
        if account is None or account["lifecycle_state"] != "ACTIVE":
            return decision(False, "ACCOUNT_READINESS_BLOCKED")
        if account["portfolio_id"] != request["portfolio_id"]:
            return decision(False, "STRATEGY_INSTANCE_BINDING_MISMATCH")
        universe = context["universes_by_id"].get(request["trading_universe_id"])
        if (
            universe is None
            or universe["lifecycle_state"] != "ACTIVE"
            or universe["exchange_account_id"] != account["exchange_account_id"]
        ):
            return decision(False, "TRADING_UNIVERSE_INVALID")
        return decision(True, transition=transition)
    instance = context["strategy_instances_by_id"].get(request["strategy_instance_id"])
    if instance is None:
        return decision(False, "STRATEGY_INSTANCE_NOT_FOUND")
    if operation.startswith("BIND_"):
        return bind_validate(operation, request, context, instance)
    if operation == "ACTIVATE_STRATEGY_INSTANCE":
        return deep_activation(request, context)
    if operation == "VALIDATE_ROUTE_READINESS":
        return readiness_validation(request, context)
    if operation == "UPDATE_STRATEGY_INSTANCE_STATE":
        matrix = (
            DATA["strategy_instance_transition_matrix"]
            .get(instance["lifecycle_state"], {})
            .get(operation, [])
        )
        ok = (
            instance["lifecycle_state"] == request["expected_state"]
            and request["target_state"] in matrix
        )
        return decision(
            ok, None if ok else "STRATEGY_INSTANCE_BINDING_MISMATCH", transition if ok else None
        )
    if operation == "DEACTIVATE_STRATEGY_INSTANCE":
        ok = (
            request["expected_state"] == "ACTIVE"
            and instance["lifecycle_state"] == request["expected_state"]
        )
        return decision(
            ok, None if ok else "STRATEGY_INSTANCE_BINDING_MISMATCH", transition if ok else None
        )
    if operation == "RETIRE_STRATEGY_INSTANCE":
        targets = (
            DATA["strategy_instance_transition_matrix"]
            .get(instance["lifecycle_state"], {})
            .get(operation, [])
        )
        ok = instance["lifecycle_state"] == request["expected_state"] and "RETIRED" in targets
        return decision(
            ok, None if ok else "STRATEGY_INSTANCE_BINDING_MISMATCH", transition if ok else None
        )
    return decision(False, "CONTRACT_INCONSISTENT")


def dispatcher(operation, request, context):
    try:
        result = direct_validate(operation, request, context)
    except (KeyError, TypeError, ValueError, OverflowError):
        result = decision(False, "TRUSTED_CONTEXT_INVALID")
    if (
        operation in DATA["operation_registry"]
        and not result["allowed"]
        and result["denial_code"] not in DATA["allowed_denials_by_operation"][operation]
    ):
        return decision(False, "CONTRACT_INCONSISTENT")
    return result


def fixture():
    config = {"parameters": {"lookback": 10}}
    definition = {
        "strategy_definition_id": did("StrategyDefinition"),
        "workspace_id": did("Workspace"),
        "strategy_type_id": did("StrategyType"),
        "definition_version": 1,
        "configuration": config,
        "canonical_content_hash": canonical_hash(config),
        "hash_domain_separator": DATA["canonical_hash_contract"]["domain_separator"],
        "lifecycle_state": "ACTIVE",
    }
    account = {
        "exchange_account_id": did("ExchangeAccount"),
        "portfolio_id": did("Portfolio"),
        "exchange_id": did("Exchange"),
        "environment": "TESTNET",
        "market_type": "SPOT",
        "lifecycle_state": "ACTIVE",
        "connection_state": "CONNECTED",
        "execution_authorization": "ORDER_ENTRY_ALLOWED",
        "active_credential_profile_id": did("CredentialProfile"),
        "account_capability_snapshot_id": did("AccountCapabilitySnapshot"),
    }
    instrument = {
        "instrument_id": did("Instrument"),
        "workspace_id": definition["workspace_id"],
        "exchange_id": account["exchange_id"],
        "environment": account["environment"],
        "market_type": account["market_type"],
        "instrument_type": "SPOT_PAIR",
        "trading_status": "TRADING",
        "catalog_snapshot_id": did("InstrumentCatalogSnapshot"),
        "source_adapter_family_id": did("AdapterFamily"),
    }
    universe = {
        "trading_universe_id": did("TradingUniverse"),
        "exchange_account_id": account["exchange_account_id"],
        "version": 1,
        "lifecycle_state": "ACTIVE",
        "instrument_ids": [instrument["instrument_id"]],
        "source_catalog_snapshot_ids": [instrument["catalog_snapshot_id"]],
    }
    market = {
        "market_data_route_id": did("MarketDataRoute"),
        "workspace_id": definition["workspace_id"],
        "exchange_id": account["exchange_id"],
        "environment": account["environment"],
        "market_type": account["market_type"],
        "adapter_family_id": instrument["source_adapter_family_id"],
        "endpoint_class": "TESTNET_PUBLIC_DATA",
        "data_scope": "PUBLIC",
        "instrument_ids": universe["instrument_ids"],
        "channel_types": ["TRADES"],
        "snapshot_stream_semantics": "SNAPSHOT_THEN_STREAM",
        "sequence_policy": "MONOTONIC",
        "freshness_policy": {"max_age_ms": 120000},
        "reconnect_policy": "RESUBSCRIBE_EXACT_SCOPE",
        "route_status": "ENABLED",
    }
    execution = {
        "execution_route_id": did("ExecutionRoute"),
        "workspace_id": definition["workspace_id"],
        "exchange_account_id": account["exchange_account_id"],
        "exchange_id": account["exchange_id"],
        "environment": account["environment"],
        "market_type": account["market_type"],
        "adapter_family_id": instrument["source_adapter_family_id"],
        "endpoint_class": "TESTNET_PRIVATE_DATA",
        "supported_instrument_types": ["SPOT_PAIR"],
        "route_status": "ENABLED",
        "route_capability_ceiling": ["TESTNET_PRIVATE_EXECUTION_AFTER_READINESS"],
        "authorization_dependencies": [
            "ACCOUNT_ACTIVE",
            "ACCOUNT_CONNECTED",
            "EXECUTION_AUTHORIZED",
            "CREDENTIAL_READY",
            "CAPABILITY_SNAPSHOT_VALID",
            "PRODUCT_CAPABILITIES_VALID",
        ],
    }
    instance = {
        "strategy_instance_id": did("StrategyInstance"),
        "workspace_id": definition["workspace_id"],
        "portfolio_id": account["portfolio_id"],
        "strategy_definition_id": definition["strategy_definition_id"],
        "strategy_definition_version": 1,
        "exchange_account_id": account["exchange_account_id"],
        "trading_universe_id": universe["trading_universe_id"],
        "market_data_route_id": market["market_data_route_id"],
        "execution_route_id": execution["execution_route_id"],
        "lifecycle_state": "BOUND",
    }
    snapshot = {
        "account_capability_snapshot_id": account["account_capability_snapshot_id"],
        "source_content_hash": "0" * 64,
        "attested_source_content_hash": "0" * 64,
        "exchange_account_id": account["exchange_account_id"],
        "exchange_id": account["exchange_id"],
        "environment": account["environment"],
        "market_type": account["market_type"],
        "status": "VALID",
        "permission_set": ["PLACE_ORDERS"],
        "supported_instrument_types": ["SPOT_PAIR"],
        "adapter_family_id": execution["adapter_family_id"],
        "validated_at_utc": "2026-07-28T11:59:00Z",
        "fresh_until_utc": "2026-07-28T12:05:00Z",
    }
    credential = {
        "credential_profile_id": account["active_credential_profile_id"],
        "exchange_account_id": account["exchange_account_id"],
        "exchange_id": account["exchange_id"],
        "environment_scope": account["environment"],
        "credential_purpose": "ORDER_ENTRY",
        "permission_snapshot": ["PLACE_ORDERS"],
        "lifecycle_state": "ACTIVE",
    }
    catalog = {
        "catalog_snapshot_id": instrument["catalog_snapshot_id"],
        "exchange_id": account["exchange_id"],
        "environment": account["environment"],
        "market_type": account["market_type"],
        "adapter_family_id": execution["adapter_family_id"],
        "instrument_ids": [instrument["instrument_id"]],
        "status": "VALID",
    }
    capability = {
        "environment": "TESTNET",
        "trust_state": "VALID",
        "capability_set": ["TESTNET_PRIVATE_EXECUTION_AFTER_READINESS"],
        "private_execution_allowed": True,
    }
    readiness = {
        market["market_data_route_id"]: {
            "route_id": market["market_data_route_id"],
            "route_kind": "MARKET_DATA",
            "readiness_state": "READY",
            "observed_at": "2026-07-28T11:59:30Z",
            "metadata_version": 1,
            "sequence_state": "CONTIGUOUS",
        },
        execution["execution_route_id"]: {
            "route_id": execution["execution_route_id"],
            "route_kind": "EXECUTION",
            "readiness_state": "READY",
            "observed_at": "2026-07-28T11:59:30Z",
            "metadata_version": 1,
            "sequence_state": "NOT_APPLICABLE",
        },
    }
    return {
        "strategy_definitions_by_id": {definition["strategy_definition_id"]: definition},
        "previous_strategy_definitions_by_version_key": {},
        "strategy_instances_by_id": {instance["strategy_instance_id"]: instance},
        "market_data_routes_by_id": {market["market_data_route_id"]: market},
        "execution_routes_by_id": {execution["execution_route_id"]: execution},
        "accounts_by_id": {account["exchange_account_id"]: account},
        "universes_by_id": {universe["trading_universe_id"]: universe},
        "instruments_by_id": {instrument["instrument_id"]: instrument},
        "catalogs_by_id": {catalog["catalog_snapshot_id"]: catalog},
        "account_capability_snapshots_by_id": {
            snapshot["account_capability_snapshot_id"]: snapshot
        },
        "credential_profiles_by_id": {credential["credential_profile_id"]: credential},
        "product_capabilities_by_environment": {"TESTNET": capability},
        "route_readiness_by_id": readiness,
        "active_strategy_instances": [],
        "validation_time_utc": NOW,
    }


def activation(context):
    instance = next(iter(context["strategy_instances_by_id"].values()))
    return {
        "strategy_instance_id": instance["strategy_instance_id"],
        "expected_definition_version": instance["strategy_definition_version"],
        "market_data_route_id": instance["market_data_route_id"],
        "execution_route_id": instance["execution_route_id"],
        "intent": "ACTIVATE",
    }


def operation_case(operation):
    context = fixture()
    instance = next(iter(context["strategy_instances_by_id"].values()))
    definition = next(iter(context["strategy_definitions_by_id"].values()))
    requests = {
        "CREATE_STRATEGY_DEFINITION": {
            field: definition[field]
            for field in DATA["operation_registry"]["CREATE_STRATEGY_DEFINITION"]["request_fields"]
        },
        "CREATE_STRATEGY_INSTANCE": {
            field: instance[field]
            for field in DATA["operation_registry"]["CREATE_STRATEGY_INSTANCE"]["request_fields"]
        },
        "UPDATE_STRATEGY_INSTANCE_STATE": {
            "strategy_instance_id": instance["strategy_instance_id"],
            "expected_state": "BOUND",
            "target_state": "BOUND",
        },
        "VALIDATE_ROUTE_READINESS": {
            "strategy_instance_id": instance["strategy_instance_id"],
            "market_data_route_id": instance["market_data_route_id"],
            "execution_route_id": instance["execution_route_id"],
            "intent": "VALIDATE_ONLY",
        },
        "ACTIVATE_STRATEGY_INSTANCE": activation(context),
        "DEACTIVATE_STRATEGY_INSTANCE": {
            "strategy_instance_id": instance["strategy_instance_id"],
            "expected_state": "ACTIVE",
        },
        "RETIRE_STRATEGY_INSTANCE": {
            "strategy_instance_id": instance["strategy_instance_id"],
            "expected_state": "INACTIVE",
        },
        "ACTIVATE_STRATEGY_DEFINITION": {
            "strategy_definition_id": definition["strategy_definition_id"],
            "expected_state": "DRAFT",
        },
        "RETIRE_STRATEGY_DEFINITION": {
            "strategy_definition_id": definition["strategy_definition_id"],
            "expected_state": "ACTIVE",
        },
    }
    if operation == "CREATE_STRATEGY_DEFINITION":
        for field in DATA["validation_context_schema"]["map_fields"]:
            context[field] = {}
    if operation == "ACTIVATE_STRATEGY_DEFINITION":
        definition["lifecycle_state"] = "DRAFT"
    if operation == "CREATE_STRATEGY_INSTANCE":
        requests[operation]["strategy_instance_id"] = did("StrategyInstance", "2")
    if operation.startswith("BIND_"):
        instance["lifecycle_state"] = "DRAFT"
        field = (
            "market_data_route_id"
            if operation.endswith("MARKET_DATA_ROUTE")
            else "execution_route_id"
        )
        route_id = instance[field]
        instance[field] = None
        requests[operation] = {
            "strategy_instance_id": instance["strategy_instance_id"],
            field: route_id,
        }
    if operation == "DEACTIVATE_STRATEGY_INSTANCE":
        instance["lifecycle_state"] = "ACTIVE"
        context["active_strategy_instances"] = [instance["strategy_instance_id"]]
    if operation == "RETIRE_STRATEGY_INSTANCE":
        instance["lifecycle_state"] = "INACTIVE"
    return context, requests[operation]


def test_metadata_status_baselines_and_utf8_sync():
    assert (DATA["schema_version"], DATA["status"]) == (
        "cryptohunter.strategy_market_data_and_execution_routing.v1",
        "closed",
    )
    assert DATA["fix3_baseline_commit"] == "bec38d519d4dabc5f9b72e911389a9363487b91e"
    assert DATA["closure_baseline_commit"] == "e2c7653817538d1ccf9662c09a0425561e331555"
    assert DATA["closure_policy"] == {
        "semantic_audit_complete": True,
        "runtime_implemented": False,
        "m07_started": False,
        "closed_scope": "M0.6 architecture contract only",
    }
    artifacts = DOC.read_text(encoding="utf-8") + MD.read_text(encoding="utf-8")
    assert ("under " + "audit") not in artifacts
    assert "Status: closed" in artifacts
    assert "Status M0.6 — closed" in README.read_text(encoding="utf-8")
    assert load(M04)["status"] == load(M05)["status"] == "closed"


@pytest.mark.parametrize(
    "map_name",
    [
        "accounts_by_id",
        "universes_by_id",
        "instruments_by_id",
        "catalogs_by_id",
        "account_capability_snapshots_by_id",
        "product_capabilities_by_environment",
        "route_readiness_by_id",
    ],
)
def test_unrelated_malformed_record_in_every_trusted_map_invalidates_context(map_name):
    context = fixture()
    record = copy.deepcopy(next(iter(context[map_name].values())))
    record["unexpected"] = True
    context[map_name]["unrelated"] = record
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_canonical_hash_constant_vector_domain_and_lineage():
    assert (
        canonical_hash({"parameters": {"lookback": 10}})
        == "64739d55dab5103ff4a14c75f1dd0b992c32a3e17445179a6637882fd00bf001"
    )
    for field, bad in (
        ("configuration", {"parameters": {"lookback": 11}}),
        ("hash_domain_separator", "wrong-domain"),
    ):
        context = fixture()
        next(iter(context["strategy_definitions_by_id"].values()))[field] = bad
        assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    context = fixture()
    current = next(iter(context["strategy_definitions_by_id"].values()))
    current["definition_version"] = 3
    old = copy.deepcopy(current)
    old["definition_version"] = 1
    old["configuration"] = {"parameters": {"lookback": 5}}
    old["canonical_content_hash"] = canonical_hash(old["configuration"])
    context["previous_strategy_definitions_by_version_key"] = {
        f"{old['strategy_definition_id']}@1": old
    }
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_market_freshness_sequence_and_execution_readiness_are_independent():
    for field, bad, code in (
        ("observed_at", "2026-07-28T11:00:00Z", "MARKET_DATA_FRESHNESS_BLOCKED"),
        ("sequence_state", "GAP_DETECTED", "MARKET_DATA_SEQUENCE_INVALID"),
    ):
        context = fixture()
        ready = next(
            item
            for item in context["route_readiness_by_id"].values()
            if item["route_kind"] == "MARKET_DATA"
        )
        ready[field] = bad
        assert (
            dispatcher("ACTIVATE_STRATEGY_INSTANCE", activation(context), context)["denial_code"]
            == code
        )
    assert (
        "readiness_state" not in SCHEMAS["MarketDataRoute"]["fields"]
        and "readiness_state" not in SCHEMAS["ExecutionRoute"]["fields"]
    )


@pytest.mark.parametrize("operation", list(DATA["operation_registry"]))
def test_all_nine_operations_have_positive_direct_dispatcher_parity(operation):
    context, request = operation_case(operation)
    before = copy.deepcopy(context)
    direct = direct_validate(operation, request, context)
    dispatched = dispatcher(operation, request, context)
    assert direct == dispatched and direct["allowed"], (operation, direct)
    assert context == before


def test_known_operations_have_specific_denials_and_unknown_is_separate():
    assert dispatcher("SUBMIT_ORDER", {}, fixture())["denial_code"] == "UNKNOWN_OPERATION"
    for operation in DATA["operation_registry"]:
        assert "UNKNOWN_OPERATION" not in DATA["allowed_denials_by_operation"][operation]
        context, request = operation_case(operation)
        assert dispatcher(operation, request, context)["denial_code"] != "REQUEST_SCHEMA_INVALID"


def test_generated_malformed_inputs_never_raise_and_always_deny():
    generated = [
        (None, {}, fixture()),
        ("ACTIVATE_STRATEGY_INSTANCE", None, fixture()),
        ("ACTIVATE_STRATEGY_INSTANCE", activation(fixture()), None),
    ]
    for operation, request, context in generated:
        result = dispatcher(operation, request, context)
        assert result["allowed"] is False and result["denial_code"] in DENIALS


def test_projection_field_names_are_from_closed_m05_contracts():
    m05 = load(M05)
    pairs = {
        "ExchangeAccountProjection": "exchange_account_contract",
        "TradingUniverseProjection": "trading_universe_contract",
        "InstrumentProjection": "instrument_contract",
        "InstrumentCatalogProjection": "instrument_catalog_snapshot_contract",
        "AccountCapabilitySnapshotProjection": "account_capability_snapshot_contract",
    }
    for projection, source in pairs.items():
        source_fields = m05[source].get("record_fields", m05[source].get("fields"))
        projected = set(SCHEMAS[projection]["fields"])
        if projection == "AccountCapabilitySnapshotProjection":
            projected -= {
                "attested_source_content_hash",
                "source_content_hash",
                "permission_set",
                "validated_at_utc",
                "fresh_until_utc",
            }
            assert {"content_hash", "observed_permission_set", "stale_after_utc"} <= set(
                source_fields
            )
        assert projected <= set(source_fields)
    assert set(ENUMS["account_lifecycle"]) == set(
        m05["exchange_account_contract"]["lifecycle_states"]
    )
    assert set(ENUMS["universe_lifecycle"]) == set(
        m05["trading_universe_contract"]["lifecycle_states"]
    )
    assert set(ENUMS["snapshot_status"]) == set(
        m05["account_capability_snapshot_contract"]["statuses"]
    )
    assert set(ENUMS["catalog_status"]) == set(
        m05["instrument_catalog_snapshot_contract"]["statuses"]
    )
    assert (
        SCHEMAS["ExchangeAccountProjection"]["id_prefix"].rstrip("_")
        == m05["exchange_account_contract"]["id_prefix"]
    )
    assert fixture()["universes_by_id"][did("TradingUniverse")]["exchange_account_id"] == did(
        "ExchangeAccount"
    )
    m04 = load(M04)
    payload = m04["ProductCapabilities"]["document_schemas"]["capability_payload_schema"]
    environments = payload["field_contracts"]["environment_capabilities"][
        "required_environment_keys"
    ]
    assert set(ENUMS["environment"]) == set(environments)
    paper = m04["ProductCapabilities"]["document_schemas"]["paper_environment_capability_schema"]
    assert "private_execution_allowed" in paper["required_fields"]
    credential = m05["credential_profile_contract"]
    assert set(ENUMS["account_permission"]) == set(credential["permission_registry"])
    assert set(ENUMS["credential_purpose"]) == set(credential["credential_purposes"])
    assert set(ENUMS["execution_authorization"]) == set(
        m05["exchange_account_contract"]["execution_authorizations"]
    )
    assert set(ENUMS["product_capability"]) == set(
        m04["capability_id_registry"]["current_schema_allowed_capability_ids"]
    )
    assert (
        m04["ProductCapabilities"]["current_edition_capability_policy"]["environment_capabilities"][
            "PAPER"
        ]["local_execution_allowed"]
        is True
    )


def test_audit_mapping_security_and_m07_remain_out_of_scope():
    assert (
        set(DATA["success_event_by_operation"])
        == set(DATA["denial_event_by_operation"])
        == set(DATA["operation_registry"])
    )
    assert all(
        event["secret_fields_forbidden"] for event in DATA["audit_event_schema_registry"].values()
    )
    assert not re.search(r"https?://", DOC.read_text(encoding="utf-8"))
    assert DATA["status"] == "closed" and "M0.7" in DATA["out_of_scope"]
    forbidden = {"SUBMIT_ORDER", "CREATE_ORDER", "CANCEL_ORDER", "REPLACE_ORDER", "EXECUTE_TRADE"}
    assert forbidden.isdisjoint(DATA["operation_registry"])


# fmt: off








def test_denial_membership_and_success_transitions():
    for operation in DATA["operation_registry"]:
        context, request = operation_case(operation); success = dispatcher(operation, request, context)
        assert success["allowed"] and success["planned_transition"] == DATA["operation_registry"][operation]["success_transition"]
        broken = dict(request); broken.pop(next(iter(broken))); denied = dispatcher(operation, broken, context)
        assert denied["denial_code"] in DATA["allowed_denials_by_operation"][operation]
    assert DATA["denial_reachability_policy"]["contract_only_codes"] == ["CONTRACT_INCONSISTENT"]



# fmt: on


# fmt: off
def paper_context():
    context = fixture(); cap = next(iter(context["product_capabilities_by_environment"].values())); cap.update(environment="PAPER", capability_set=["PAPER_LOCAL_SIMULATION"], private_execution_allowed=False); context["product_capabilities_by_environment"] = {"PAPER": cap}
    for name in ("market_data_routes_by_id", "execution_routes_by_id", "accounts_by_id", "instruments_by_id", "catalogs_by_id", "account_capability_snapshots_by_id"): next(iter(context[name].values()))["environment"] = "PAPER"
    market = next(iter(context["market_data_routes_by_id"].values())); market["endpoint_class"] = "PAPER_PUBLIC_DATA"; execution = next(iter(context["execution_routes_by_id"].values())); execution["endpoint_class"] = "PAPER_SIMULATION"; execution["route_capability_ceiling"] = ["PAPER_LOCAL_SIMULATION"]; execution["authorization_dependencies"] = ["PRODUCT_CAPABILITIES_VALID", "CAPABILITY_SNAPSHOT_VALID"]; next(iter(context["account_capability_snapshots_by_id"].values()))["permission_set"] = []; next(iter(context["accounts_by_id"].values()))["active_credential_profile_id"] = None; context["credential_profiles_by_id"] = {}
    return context


def test_m04_paper_local_simulation_is_allowed_without_private_exchange_authority():
    assert dispatcher("ACTIVATE_STRATEGY_INSTANCE", activation(paper_context()), paper_context())["allowed"]


def build_reachability_case(case):
    operation, code = case["operation"], case["denial_code"]
    context, request = operation_case(operation)
    if operation == "CREATE_STRATEGY_DEFINITION" and code in {"STRATEGY_DEFINITION_ID_COLLISION", "RETIRED_RESOURCE_FORBIDDEN"}:
        context = fixture()
        if code == "RETIRED_RESOURCE_FORBIDDEN":
            current = next(iter(context["strategy_definitions_by_id"].values())); current["lifecycle_state"] = "RETIRED"; request.update(strategy_definition_id=current["strategy_definition_id"], definition_version=2)
    instance = next(iter(context["strategy_instances_by_id"].values())) if context["strategy_instances_by_id"] else None
    if code == "REQUEST_SCHEMA_INVALID": request.pop(next(iter(request)))
    elif code == "TRUSTED_CONTEXT_INVALID": context["accounts_by_id"] = None
    elif code == "STRATEGY_DEFINITION_NOT_FOUND":
        if operation in {"ACTIVATE_STRATEGY_DEFINITION", "RETIRE_STRATEGY_DEFINITION", "CREATE_STRATEGY_INSTANCE"}: request["strategy_definition_id"] = did("StrategyDefinition", "9")
        else: instance["strategy_definition_id"] = did("StrategyDefinition", "9")
    elif code == "STRATEGY_DEFINITION_LINEAGE_INVALID":
        context = fixture(); request.update(strategy_definition_id=did("StrategyDefinition"), definition_version=2, strategy_type_id=did("StrategyType", "2"))
    elif code == "STRATEGY_DEFINITION_VERSION_MISMATCH":
        if operation == "CREATE_STRATEGY_DEFINITION": request["definition_version"] = 3
        elif operation == "CREATE_STRATEGY_INSTANCE": request["strategy_definition_version"] = 2
        elif operation.startswith("BIND_") or operation == "VALIDATE_ROUTE_READINESS": instance["strategy_definition_version"] = 2
        else: request["expected_definition_version"] = 2
    elif code == "TRUSTED_CONTEXT_INVALID":
        if operation == "CREATE_STRATEGY_DEFINITION": request["canonical_content_hash"] = "f" * 64
        else: next(iter(context["strategy_definitions_by_id"].values()))["canonical_content_hash"] = "f" * 64
    elif code == "STRATEGY_INSTANCE_NOT_FOUND": context["strategy_instances_by_id"] = {}; context["active_strategy_instances"] = []
    elif code == "STRATEGY_INSTANCE_BINDING_MISMATCH":
        if operation.startswith("BIND_"): instance["lifecycle_state"] = "BOUND"; instance["market_data_route_id"] = did("MarketDataRoute"); instance["execution_route_id"] = did("ExecutionRoute")
        elif operation == "CREATE_STRATEGY_INSTANCE": request["workspace_id"] = did("Workspace", "2")
        elif "expected_state" in request: request["expected_state"] = "DRAFT"
        else: request["market_data_route_id"] = did("MarketDataRoute", "2")
    elif code == "MARKET_DATA_ROUTE_NOT_FOUND": request["market_data_route_id"] = did("MarketDataRoute", "9")
    elif code == "MARKET_DATA_ROUTE_NOT_READY": next(iter(context["market_data_routes_by_id"].values()))["route_status"] = "DISABLED"
    elif code == "EXECUTION_ROUTE_NOT_FOUND": request["execution_route_id"] = did("ExecutionRoute", "9")
    elif code == "EXECUTION_ROUTE_NOT_READY": next(iter(context["execution_routes_by_id"].values()))["route_status"] = "DISABLED"
    elif code == "ROUTE_SCOPE_MISMATCH": next(iter(context["execution_routes_by_id" if operation == "BIND_EXECUTION_ROUTE" else "market_data_routes_by_id"].values()))["workspace_id"] = did("Workspace", "2")
    elif code == "ROUTE_ADAPTER_MISMATCH":
        targets = ("market_data_routes_by_id", "execution_routes_by_id") if operation in {"VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"} else (("execution_routes_by_id",) if operation == "BIND_EXECUTION_ROUTE" else ("market_data_routes_by_id",))
        for name in targets: next(iter(context[name].values()))["adapter_family_id"] = did("AdapterFamily", "2")
    elif code == "ROUTE_CAPABILITY_BLOCKED":
        if operation in {"BIND_MARKET_DATA_ROUTE", "VALIDATE_ROUTE_READINESS"}: next(iter(context["market_data_routes_by_id"].values()))["channel_types"] = ["PRIVATE_ORDERS"]
        elif operation == "BIND_EXECUTION_ROUTE": next(iter(context["execution_routes_by_id"].values()))["authorization_dependencies"].pop()
        else: next(iter(context["execution_routes_by_id"].values()))["route_capability_ceiling"] = []
    elif code == "MARKET_DATA_FRESHNESS_BLOCKED": next(item for item in context["route_readiness_by_id"].values() if item["route_kind"] == "MARKET_DATA")["observed_at"] = "2026-07-28T11:00:00Z"
    elif code == "MARKET_DATA_SEQUENCE_INVALID": next(item for item in context["route_readiness_by_id"].values() if item["route_kind"] == "MARKET_DATA")["sequence_state"] = "GAP_DETECTED"
    elif code == "ACCOUNT_READINESS_BLOCKED":
        if "exchange_account_id" in request: request["exchange_account_id"] = did("ExchangeAccount", "9")
        else: next(iter(context["accounts_by_id"].values()))["lifecycle_state"] = "RETIRED" if operation.startswith("BIND_") else "DISABLED"
    elif code == "TRADING_UNIVERSE_INVALID":
        if "trading_universe_id" in request: request["trading_universe_id"] = did("TradingUniverse", "9")
        else: next(iter(context["universes_by_id"].values()))["lifecycle_state"] = "DRAFT"
    elif code == "INSTRUMENT_SCOPE_MISMATCH":
        if operation == "BIND_EXECUTION_ROUTE": next(iter(context["execution_routes_by_id"].values()))["supported_instrument_types"] = ["MARGIN_PAIR"]
        else:
            instrument = copy.deepcopy(next(iter(context["instruments_by_id"].values()))); instrument["instrument_id"] = did("Instrument", "2"); context["instruments_by_id"][instrument["instrument_id"]] = instrument; next(iter(context["catalogs_by_id"].values()))["instrument_ids"] = next(iter(context["catalogs_by_id"].values()))["instrument_ids"] + [instrument["instrument_id"]]; next(iter(context["universes_by_id"].values()))["instrument_ids"] = next(iter(context["universes_by_id"].values()))["instrument_ids"] + [instrument["instrument_id"]]
    elif code == "ENVIRONMENT_MISMATCH": next(iter(context["execution_routes_by_id"].values()))["environment"] = "PAPER"
    elif code == "LIVE_EXECUTION_FORBIDDEN":
        for name in ("market_data_routes_by_id", "execution_routes_by_id", "accounts_by_id", "instruments_by_id", "catalogs_by_id", "account_capability_snapshots_by_id"): next(iter(context[name].values()))["environment"] = "LIVE"
        next(iter(context["market_data_routes_by_id"].values()))["endpoint_class"] = "LIVE_PUBLIC_DATA"; next(iter(context["execution_routes_by_id"].values()))["endpoint_class"] = "LIVE_PRIVATE_DATA"; next(iter(context["execution_routes_by_id"].values()))["route_capability_ceiling"] = ["LIVE_VISIBLE_LOCKED_ONLY"]; next(iter(context["execution_routes_by_id"].values()))["authorization_dependencies"] = ["PRODUCT_CAPABILITIES_VALID"]
        cap = next(iter(context["product_capabilities_by_environment"].values())); cap.update(environment="LIVE", capability_set=["LIVE_VISIBLE_LOCKED_ONLY"], private_execution_allowed=False); context["product_capabilities_by_environment"] = {"LIVE": cap}; next(iter(context["accounts_by_id"].values()))["active_credential_profile_id"] = None; context["credential_profiles_by_id"] = {}
    elif code == "ENDPOINT_FALLBACK_FORBIDDEN": next(iter(context["market_data_routes_by_id" if operation == "BIND_MARKET_DATA_ROUTE" else "execution_routes_by_id"].values()))["endpoint_class"] = "LIVE_PUBLIC_DATA" if operation == "BIND_MARKET_DATA_ROUTE" else "LIVE_PRIVATE_DATA"
    elif code == "RETIRED_RESOURCE_FORBIDDEN":
        if operation == "VALIDATE_ROUTE_READINESS": instance["lifecycle_state"] = "RETIRED"
        else: next(iter(context["strategy_definitions_by_id"].values()))["lifecycle_state"] = "RETIRED"
    elif code == "CAPABILITY_SNAPSHOT_BLOCKED": next(iter(context["account_capability_snapshots_by_id"].values()))["permission_set"] = []
    elif code == "PRODUCT_CAPABILITIES_BLOCKED": context["product_capabilities_by_environment"] = {}
    elif code == "STRATEGY_DEFINITION_ID_COLLISION": request["definition_version"] = 1
    elif code == "STRATEGY_INSTANCE_ID_COLLISION": request["strategy_instance_id"] = instance["strategy_instance_id"]
    if operation in {"ACTIVATE_STRATEGY_DEFINITION", "RETIRE_STRATEGY_DEFINITION"} and code == "STRATEGY_INSTANCE_BINDING_MISMATCH": request["expected_state"] = "ACTIVE" if operation == "ACTIVATE_STRATEGY_DEFINITION" else "DRAFT"
    return operation, request, context


def test_declared_denial_reachability_cases_execute():
    declared = {(item["operation"], item["denial_code"]) for item in DATA["denial_reachability_cases"]}
    expected = {(operation, code) for operation, codes in DATA["allowed_denials_by_operation"].items() for code in codes if code != "CONTRACT_INCONSISTENT"}
    assert declared == expected
    for case in DATA["denial_reachability_cases"]:
        operation, request, context = build_reachability_case(case); result = dispatcher(operation, request, context)
        assert result["denial_code"] == case["denial_code"] and result["denial_code"] in DATA["allowed_denials_by_operation"][operation]
    assert dispatcher("NOT_AN_OPERATION", {}, fixture())["denial_code"] == "UNKNOWN_OPERATION"


def test_fix5_forbidden_intersection_phrases_absent():
    text = DOC.read_text(encoding="utf-8") + MD.read_text(encoding="utf-8"); assert "ProductCapabilities intersect AccountCapabilitySnapshot" not in text; assert "trusted intersection M0.4" not in text

# fmt: on


def test_fix6_intents_definition_lifecycle_and_denial_registries():
    for operation, bad in (
        ("VALIDATE_ROUTE_READINESS", "ACTIVATE"),
        ("ACTIVATE_STRATEGY_INSTANCE", "VALIDATE_ONLY"),
    ):
        context, request = operation_case(operation)
        request["intent"] = bad
        assert dispatcher(operation, request, context)["denial_code"] == "REQUEST_SCHEMA_INVALID"
    for operation, expected in (
        ("CREATE_STRATEGY_DEFINITION", "DRAFT"),
        ("ACTIVATE_STRATEGY_DEFINITION", "ACTIVE"),
        ("RETIRE_STRATEGY_DEFINITION", "RETIRED"),
    ):
        context, request = operation_case(operation)
        assert dispatcher(operation, request, context)["planned_definition_state"] == expected
    for operation, allowed in DATA["allowed_denials_by_operation"].items():
        emitted = set().union(
            *(
                set(DATA["validator_denial_registry"][validator])
                for validator in DATA["operation_validator_call_graph"][operation]
            )
        )
        assert emitted == set(allowed) - {"CONTRACT_INCONSISTENT"}
    used = set().union(*map(set, DATA["allowed_denials_by_operation"].values()))
    used |= {"UNKNOWN_OPERATION", *DATA["denial_reachability_policy"]["contract_only_codes"]}
    assert set(DATA["denial_code_registry"]) == used


def versioned_context():
    context = fixture()
    current = next(iter(context["strategy_definitions_by_id"].values()))
    previous = copy.deepcopy(current)
    current["definition_version"] = 2
    context["previous_strategy_definitions_by_version_key"] = {
        f"{previous['strategy_definition_id']}@1": previous
    }
    next(iter(context["strategy_instances_by_id"].values()))["strategy_definition_version"] = 1
    return context


def test_fix7_exact_definition_version_resolution_without_implicit_upgrade():
    context = versioned_context()
    instance = next(iter(context["strategy_instances_by_id"].values()))
    assert validate_context(context)["allowed"]
    resolved = resolve_strategy_definition_version(instance["strategy_definition_id"], 1, context)
    assert (
        resolved
        is context["previous_strategy_definitions_by_version_key"][
            f"{instance['strategy_definition_id']}@1"
        ]
    )
    before = copy.deepcopy(instance)
    assert dispatcher(
        "VALIDATE_ROUTE_READINESS", operation_case("VALIDATE_ROUTE_READINESS")[1], context
    )["allowed"]
    request = activation(context)
    request["expected_definition_version"] = 1
    assert dispatcher("ACTIVATE_STRATEGY_INSTANCE", request, context)["allowed"]
    assert instance == before
    instance["strategy_definition_version"] = 3
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(
    ("map_name", "field", "value"),
    [
        ("account_capability_snapshots_by_id", "exchange_id", did("Exchange", "2")),
        ("account_capability_snapshots_by_id", "environment", "PAPER"),
        ("credential_profiles_by_id", "exchange_id", did("Exchange", "2")),
        ("credential_profiles_by_id", "environment_scope", "PAPER"),
        ("credential_profiles_by_id", "lifecycle_state", "RETIRED"),
        ("execution_routes_by_id", "exchange_account_id", did("ExchangeAccount", "2")),
        ("universes_by_id", "exchange_account_id", did("ExchangeAccount", "2")),
    ],
)
def test_fix7_exact_cross_record_mutations_are_trusted_context_invalid(map_name, field, value):
    context = fixture()
    next(iter(context[map_name].values()))[field] = value
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_fix7_definition_lifecycle_missing_retired_and_bind_dependency_denials():
    for operation in ("ACTIVATE_STRATEGY_DEFINITION", "RETIRE_STRATEGY_DEFINITION"):
        context, request = operation_case(operation)
        request["strategy_definition_id"] = did("StrategyDefinition", "9")
        assert (
            dispatcher(operation, request, context)["denial_code"]
            == "STRATEGY_DEFINITION_NOT_FOUND"
        )
    context, request = operation_case("CREATE_STRATEGY_DEFINITION")
    current = next(iter(fixture()["strategy_definitions_by_id"].values()))
    context = fixture()
    current = next(iter(context["strategy_definitions_by_id"].values()))
    current["lifecycle_state"] = "RETIRED"
    request.update(strategy_definition_id=current["strategy_definition_id"], definition_version=2)
    assert (
        dispatcher("CREATE_STRATEGY_DEFINITION", request, context)["denial_code"]
        == "RETIRED_RESOURCE_FORBIDDEN"
    )
    context, request = operation_case("BIND_EXECUTION_ROUTE")
    next(iter(context["execution_routes_by_id"].values()))["authorization_dependencies"].pop()
    assert (
        dispatcher("BIND_EXECUTION_ROUTE", request, context)["denial_code"]
        == "ROUTE_CAPABILITY_BLOCKED"
    )


@pytest.mark.parametrize(
    ("operation", "resource", "state", "expected"),
    [
        ("VALIDATE_ROUTE_READINESS", "account", "DISABLED", "ACCOUNT_READINESS_BLOCKED"),
        ("VALIDATE_ROUTE_READINESS", "universe", "DRAFT", "TRADING_UNIVERSE_INVALID"),
        ("ACTIVATE_STRATEGY_INSTANCE", "account", "DISABLED", "ACCOUNT_READINESS_BLOCKED"),
        ("ACTIVATE_STRATEGY_INSTANCE", "universe", "DRAFT", "TRADING_UNIVERSE_INVALID"),
        ("ACTIVATE_STRATEGY_INSTANCE", "instrument", "HALTED", "INSTRUMENT_SCOPE_MISMATCH"),
        ("BIND_MARKET_DATA_ROUTE", "account", "RETIRED", "ACCOUNT_READINESS_BLOCKED"),
        ("BIND_MARKET_DATA_ROUTE", "universe", "DRAFT", "TRADING_UNIVERSE_INVALID"),
    ],
)
def test_fix7_ordinary_domain_errors_have_specific_non_contract_denials(
    operation, resource, state, expected
):
    context, request = operation_case(operation)
    map_name = {
        "account": "accounts_by_id",
        "universe": "universes_by_id",
        "instrument": "instruments_by_id",
    }[resource]
    field = "trading_status" if resource == "instrument" else "lifecycle_state"
    next(iter(context[map_name].values()))[field] = state
    result = dispatcher(operation, request, context)
    assert result["denial_code"] == expected != "CONTRACT_INCONSISTENT"


def test_fix8_reachability_uses_valid_context_except_context_denial():
    for case in DATA["denial_reachability_cases"]:
        operation, request, context = build_reachability_case(case)
        if case["denial_code"] != "TRUSTED_CONTEXT_INVALID":
            assert validate_context(context)["allowed"], case
        assert dispatcher(operation, request, context)["denial_code"] == case["denial_code"]


@pytest.mark.parametrize(
    ("domain", "malformed"),
    [
        ("disabled_account", "catalog"),
        ("draft_universe", "credential"),
        ("stale_snapshot", "route"),
        ("missing_definition", None),
        ("missing_account", None),
    ],
)
def test_malformed_unrelated_context_always_precedes_domain_denial(domain, malformed):
    context = fixture()
    if domain == "disabled_account":
        next(iter(context["accounts_by_id"].values()))["lifecycle_state"] = "DISABLED"
    elif domain == "draft_universe":
        next(iter(context["universes_by_id"].values()))["lifecycle_state"] = "DRAFT"
    elif domain == "stale_snapshot":
        next(iter(context["account_capability_snapshots_by_id"].values()))["status"] = "STALE"
    elif domain == "missing_definition":
        context["strategy_definitions_by_id"] = {}
    else:
        context["accounts_by_id"] = {}
    target = {
        "catalog": "catalogs_by_id",
        "credential": "credential_profiles_by_id",
        "route": "market_data_routes_by_id",
    }.get(malformed)
    if target:
        next(iter(context[target].values()))["unexpected"] = True
    request = operation_case("VALIDATE_ROUTE_READINESS")[1]
    assert (
        dispatcher("VALIDATE_ROUTE_READINESS", request, context)["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize(
    ("operation", "request_field", "missing_id", "expected"),
    [
        (
            "CREATE_STRATEGY_INSTANCE",
            "exchange_account_id",
            did("ExchangeAccount", "9"),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            "ACTIVATE_STRATEGY_DEFINITION",
            "strategy_definition_id",
            did("StrategyDefinition", "9"),
            "STRATEGY_DEFINITION_NOT_FOUND",
        ),
        (
            "RETIRE_STRATEGY_DEFINITION",
            "strategy_definition_id",
            did("StrategyDefinition", "9"),
            "STRATEGY_DEFINITION_NOT_FOUND",
        ),
        (
            "BIND_MARKET_DATA_ROUTE",
            "market_data_route_id",
            did("MarketDataRoute", "9"),
            "MARKET_DATA_ROUTE_NOT_FOUND",
        ),
        (
            "BIND_EXECUTION_ROUTE",
            "execution_route_id",
            did("ExecutionRoute", "9"),
            "EXECUTION_ROUTE_NOT_FOUND",
        ),
    ],
)
def test_fix8_request_only_missing_resource_is_operation_specific(
    operation, request_field, missing_id, expected
):
    context, request = operation_case(operation)
    request[request_field] = missing_id
    assert validate_context(context)["allowed"]
    assert dispatcher(operation, request, context)["denial_code"] == expected


def test_fix8_validator_registry_and_call_graph_are_exact_and_closed():
    registry = DATA["validator_denial_registry"]
    graph = DATA["operation_validator_call_graph"]
    used = {validator for validators in graph.values() for validator in validators}
    assert used == set(registry)
    for operation, validators in graph.items():
        expected = set().union(*(set(registry[name]) for name in validators))
        assert expected == set(DATA["allowed_denials_by_operation"][operation]) - {
            "CONTRACT_INCONSISTENT"
        }
    assert all(any(registry[name] for name in validators) for validators in graph.values())


@pytest.mark.parametrize("case", ["missing", "extra", "wrong_catalog"])
def test_fix8_universe_catalog_membership_is_exact(case):
    context = fixture()
    universe = next(iter(context["universes_by_id"].values()))
    catalog = next(iter(context["catalogs_by_id"].values()))
    if case == "missing":
        universe["source_catalog_snapshot_ids"] = []
    else:
        extra = copy.deepcopy(catalog)
        extra["catalog_snapshot_id"] = did("InstrumentCatalogSnapshot", "2")
        context["catalogs_by_id"][extra["catalog_snapshot_id"]] = extra
        if case == "extra":
            universe["source_catalog_snapshot_ids"].append(extra["catalog_snapshot_id"])
        else:
            extra["instrument_ids"] = catalog["instrument_ids"]
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert validate_context(fixture())["allowed"]


def test_fix8_single_authorization_dependency_registry_drives_bind_and_activation():
    policy = DATA["authorization_dependencies_by_environment"]
    original = list(policy["TESTNET"])
    try:
        policy["TESTNET"] = original[:-1]
        for operation in ("BIND_EXECUTION_ROUTE", "ACTIVATE_STRATEGY_INSTANCE"):
            context, request = operation_case(operation)
            assert (
                dispatcher(operation, request, context)["denial_code"] == "ROUTE_CAPABILITY_BLOCKED"
            )
    finally:
        policy["TESTNET"] = original


def test_fix8_credential_map_contains_only_selected_active_profiles():
    context = fixture()
    credential = copy.deepcopy(next(iter(context["credential_profiles_by_id"].values())))
    credential["credential_profile_id"] = did("CredentialProfile", "2")
    context["credential_profiles_by_id"][credential["credential_profile_id"]] = credential
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    context = fixture()
    next(iter(context["credential_profiles_by_id"].values()))["lifecycle_state"] = "RETIRED"
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(
    "validator",
    [
        lambda context: validate_context(context),
        lambda context: dispatcher(
            "VALIDATE_ROUTE_READINESS",
            operation_case("VALIDATE_ROUTE_READINESS")[1],
            context,
        ),
        lambda context: dispatcher("ACTIVATE_STRATEGY_INSTANCE", activation(context), context),
    ],
)
@pytest.mark.parametrize("resource", ["bound_execution", "catalog"])
def test_bound_adapter_mismatch_is_trusted_context_invalid(resource, validator):
    context = fixture()
    target = (
        next(iter(context["execution_routes_by_id"].values()))
        if resource == "bound_execution"
        else next(iter(context["catalogs_by_id"].values()))
    )
    target["adapter_family_id"] = did("AdapterFamily", "2")
    result = validator(context)
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_unrelated_instrument_catalog_adapter_mismatch_invalidates_context():
    context = fixture()
    instrument = copy.deepcopy(next(iter(context["instruments_by_id"].values())))
    catalog = copy.deepcopy(next(iter(context["catalogs_by_id"].values())))
    instrument["instrument_id"] = did("Instrument", "2")
    catalog["catalog_snapshot_id"] = did("InstrumentCatalogSnapshot", "2")
    instrument["catalog_snapshot_id"] = catalog["catalog_snapshot_id"]
    instrument["source_adapter_family_id"] = did("AdapterFamily", "2")
    catalog["instrument_ids"] = [instrument["instrument_id"]]
    context["instruments_by_id"][instrument["instrument_id"]] = instrument
    context["catalogs_by_id"][catalog["catalog_snapshot_id"]] = catalog
    assert validate_context(context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_unbound_execution_adapter_candidate_is_operation_denial():
    context, request = operation_case("BIND_EXECUTION_ROUTE")
    next(iter(context["execution_routes_by_id"].values()))["adapter_family_id"] = did(
        "AdapterFamily", "2"
    )
    assert validate_context(context)["allowed"]
    result = dispatcher("BIND_EXECUTION_ROUTE", request, context)
    assert result["denial_code"] == "ROUTE_ADAPTER_MISMATCH"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"
