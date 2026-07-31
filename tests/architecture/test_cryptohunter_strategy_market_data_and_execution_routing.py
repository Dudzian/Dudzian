"""Executable, pure M0.6 reference validators (not production runtime)."""

import copy
import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

import pytest

ROOT = Path(__file__).parents[2]
PATH = (
    ROOT
    / "docs/architecture/cryptohunter_product_architecture/strategy_market_data_and_execution_routing.json"
)
CONTRACT = json.loads(PATH.read_text(encoding="utf-8"))
M04_CONTRACT = json.loads(
    (PATH.parent / "environment_and_product_capabilities.json").read_text(encoding="utf-8")
)
M05_CONTRACT = json.loads(
    (PATH.parent / "exchange_accounts_and_instruments.json").read_text(encoding="utf-8")
)
SCHEMAS = CONTRACT["record_schemas"]
REQUESTS = {name: value["request_schema"] for name, value in CONTRACT["operation_registry"].items()}
UUID7 = "018f0f3e-7b5a-7abc-8def-1234567890"
IDS = {
    "ws": f"ws_{UUID7}01",
    "port": f"port_{UUID7}02",
    "sdef": f"sdef_{UUID7}03",
    "sinst": f"sinst_{UUID7}04",
    "xacc": f"xacc_{UUID7}05",
    "univ": f"univ_{UUID7}06",
    "mdr": f"mdr_{UUID7}07",
    "xroute": f"xroute_{UUID7}08",
    "instr": f"instr_{UUID7}09",
    "icat": f"icat_{UUID7}0a",
    "capsnap": f"capsnap_{UUID7}0b",
    "cred": f"cred_{UUID7}0c",
    "req": f"req_{UUID7}0d",
}
ID_PATTERN = re.compile(
    r"^(?P<prefix>[a-z]+)_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def deep_freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(deep_freeze(item) for item in value)
    if type(value) is tuple:
        return tuple(deep_freeze(item) for item in value)
    return value


def decision(allowed, operation, denial=None, transition=None):
    event = (
        CONTRACT["success_event_by_operation"].get(operation)
        if allowed
        else CONTRACT["denial_event_by_operation"].get(operation)
    )
    if denial == "CONTRACT_INCONSISTENT":
        event = CONTRACT["audit_event_contract"]["special_events"]["CONTRACT_INCONSISTENT"]
    elif operation not in REQUESTS:
        event = CONTRACT["audit_event_contract"]["special_events"]["UNKNOWN_OPERATION"]
    return deep_freeze(
        {
            "allowed": allowed,
            "denial_code": denial,
            "operation": operation,
            "planned_transition": deep_freeze(transition),
            "audit_event_type": event,
        }
    )


def deny(operation, code):
    return decision(False, operation, code)


def canonical_configuration(value):
    if type(value) is not dict or set(value) != {"lookback", "enabled"}:
        raise ValueError
    if (
        type(value["lookback"]) is not int
        or value["lookback"] < 1
        or type(value["enabled"]) is not bool
    ):
        raise ValueError
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def definition_hash(configuration):
    separator = CONTRACT["strategy_definition_contract"]["hash"]["domain_separator"].encode()
    return hashlib.sha256(separator + canonical_configuration(configuration)).hexdigest()


def is_id(value, prefix=None):
    match = ID_PATTERN.fullmatch(value) if type(value) is str else None
    return bool(match and (prefix is None or match["prefix"] == prefix))


def parse_time(value):
    if type(value) is not str:
        raise ValueError
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo != UTC:
        raise ValueError
    return parsed


CANONICAL_SOURCES = {
    "environment_and_product_capabilities.json": M04_CONTRACT,
    "exchange_accounts_and_instruments.json": M05_CONTRACT,
}


def resolve_canonical_pointer(reference, expected_type):
    if type(reference) is not dict or not {"contract", "json_pointer"} <= set(reference):
        raise TypeError("canonical reference")
    source = CANONICAL_SOURCES.get(reference["contract"])
    pointer = reference["json_pointer"]
    if source is None or type(pointer) is not str or not pointer.startswith("/"):
        raise KeyError("canonical reference")
    value = source
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?![01])", encoded):
            raise ValueError("invalid JSON Pointer escape")
        part = encoded.replace("~1", "/").replace("~0", "~")
        if type(value) is not dict or part not in value:
            raise KeyError(part)
        value = value[part]
    if type(value) is not expected_type:
        raise TypeError("canonical pointer result")
    return value


def canonical_fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_canonical_registry(reference_name):
    references = {
        **CONTRACT["canonical_array_enum_registry_refs"],
        **CONTRACT["canonical_scalar_enum_registry_refs"],
    }
    reference = references[reference_name]
    value = resolve_canonical_pointer(reference, list)
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise TypeError(reference_name)
    return tuple(value)


def canonical_exchange_entries():
    reference = CONTRACT["canonical_cross_contract_registry_refs"]["m05_exchange_registry_entries"]
    entries = resolve_canonical_pointer(reference, list)
    if canonical_fingerprint(entries) != reference["content_fingerprint_sha256"]:
        raise ValueError("exchange registry fingerprint")
    required = {
        "exchange_id",
        "display_name",
        "adapter_family_id",
        "supported_environments",
        "supported_market_types",
        "supported_instrument_types",
        "capability_discovery_policy",
        "instrument_catalog_discovery_policy",
        "account_identity_discovery_policy",
        "status",
        "aliases",
    }

    def closed_values(pointer):
        values = resolve_canonical_pointer(
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": pointer,
            },
            list,
        )
        if any(type(item) is not str or not item for item in values):
            raise TypeError("closed registry")
        return set(values)

    environments = closed_values("/environment_registry")
    market_types = closed_values("/market_type_registry")
    instrument_types = closed_values("/instrument_type_registry")
    identities = []
    for entry in entries:
        if type(entry) is not dict or set(entry) != required:
            raise TypeError("exchange registry entry")
        identities.append(entry["exchange_id"])
        arrays = (
            (entry["supported_environments"], environments),
            (entry["supported_market_types"], market_types),
            (entry["supported_instrument_types"], instrument_types),
        )
        aliases = entry["aliases"]
        policies = (
            entry["capability_discovery_policy"],
            entry["instrument_catalog_discovery_policy"],
            entry["account_identity_discovery_policy"],
        )
        allowed_policies = {
            value
            for item in M05_CONTRACT["exchange_registry_contract"]["entries"]
            for value in (
                item["capability_discovery_policy"],
                item["instrument_catalog_discovery_policy"],
                item["account_identity_discovery_policy"],
            )
        }
        if (
            type(entry["exchange_id"]) is not str
            or not entry["exchange_id"]
            or type(entry["display_name"]) is not str
            or not entry["display_name"]
            or type(entry["adapter_family_id"]) is not str
            or not entry["adapter_family_id"]
            or entry["status"] != "ENABLED"
            or type(aliases) is not list
            or len(aliases) != len(set(aliases))
            or any(type(alias) is not str or not alias for alias in aliases)
            or any(type(policy) is not str or policy not in allowed_policies for policy in policies)
            or any(
                type(values) is not list
                or not values
                or len(values) != len(set(values))
                or not set(values) <= registry
                for values, registry in arrays
            )
        ):
            raise TypeError("exchange registry entry")
    if len(identities) != len(set(identities)):
        raise TypeError("duplicate exchange registry identity")
    return entries


def resolve_hash_definition(reference_name):
    reference = CONTRACT["canonical_hash_definition_refs"][reference_name]
    value = resolve_canonical_pointer(reference, dict)
    if canonical_fingerprint(value) != reference["content_fingerprint_sha256"]:
        raise ValueError(reference_name)
    required = {
        "algorithm",
        "domain_separator",
        "input_fields",
        "canonicalization",
        "encoding",
        "digest_format",
    }
    if (
        not isinstance(value, dict)
        or set(value) not in {frozenset(required), frozenset(required | {"excluded_fields"})}
        or value["algorithm"] != "SHA-256"
    ):
        raise TypeError(reference_name)
    return value


def canonical_m05_hash(reference_name, record):
    definition = resolve_hash_definition(reference_name)
    canonical = {field: record[field] for field in definition["input_fields"]}
    for field in (
        "instrument_ids",
        "source_catalog_snapshot_ids",
        "observed_permission_set",
        "supported_instrument_types",
    ):
        if isinstance(canonical.get(field), list):
            canonical[field] = sorted(canonical[field])
    raw = (
        definition["domain_separator"]
        + "\n"
        + json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def rehash_m05_projections(context):
    for record in context["universes_by_id"].values():
        record["content_hash"] = canonical_m05_hash("trading_universe", record)
    for record in context["catalogs_by_id"].values():
        record["content_hash"] = canonical_m05_hash("instrument_catalog_snapshot", record)
    for record in context["account_capability_snapshots_by_id"].values():
        record["content_hash"] = canonical_m05_hash("account_capability_snapshot", record)
    return context


def exchange_entry_for_environment(environment):
    return next(
        entry
        for entry in canonical_exchange_entries()
        if entry["status"] == "ENABLED" and environment in entry["supported_environments"]
    )


def record_matches_exchange_registry(record):
    entry = next(
        (
            item
            for item in canonical_exchange_entries()
            if item["exchange_id"] == record["exchange_id"]
        ),
        None,
    )
    if (
        entry is None
        or entry["status"] != "ENABLED"
        or record.get("environment", record.get("environment_scope"))
        not in entry["supported_environments"]
        or (
            "market_type" in record and record["market_type"] not in entry["supported_market_types"]
        )
        or (
            "instrument_type" in record
            and record["instrument_type"] not in entry["supported_instrument_types"]
        )
    ):
        return False
    adapter = record.get("adapter_family_id", record.get("source_adapter_family_id"))
    return adapter is None or adapter == entry["adapter_family_id"]


def validate_typed(value, type_name, field, schema):
    if value is None:
        return field in schema["nullable_fields"]
    if type_name == "string":
        return type(value) is str and bool(value)
    if type_name == "integer":
        return type(value) is int and value >= 1
    if type_name == "boolean":
        return type(value) is bool
    if type_name == "object":
        return type(value) is dict
    if type_name == "sha256":
        return type(value) is str and bool(re.fullmatch(r"[0-9a-f]{64}", value))
    if type_name == "timestamp":
        try:
            parse_time(value)
            return True
        except ValueError:
            return False
    if type_name == "enum":
        reference_name = schema.get("enum_registry_ref", {}).get(field)
        allowed = (
            resolve_canonical_registry(reference_name)
            if reference_name
            else schema["enum_registry"].get(
                field, schema.get("nested_schemas", {}).get(field, {}).get("enum", [])
            )
        )
        return value in allowed
    if type_name == "id":
        prefix = CONTRACT["id_prefix_registry"].get(field)
        return is_id(value, prefix)
    if type_name.startswith("array["):
        if type(value) is not list:
            return False
        policy = schema["array_policy"].get(field, {})
        if not policy.get("empty_allowed", True) and not value:
            return False
        if policy.get("unique") and len(value) != len(set(value)):
            return False
        subtype = type_name[6:-1]
        if subtype == "enum":
            registry = policy.get("item_registry")
            if registry:
                allowed = CONTRACT["array_enum_registries"].get(registry)
            else:
                allowed = resolve_canonical_registry(policy.get("item_registry_ref"))
            return type(allowed) in {list, tuple} and all(item in allowed for item in value)
        return all(
            is_id(item) if subtype == "id" else type(item) is str and bool(item) for item in value
        )
    return False


def validate_nested(value, nested):
    if type(value) is not dict or set(value) != set(nested["exact_fields"]):
        return False
    shell = {
        "nullable_fields": nested.get("nullable_fields", []),
        "enum_registry": nested.get("enum_registry", {}),
        "nested_schemas": {},
        "array_policy": {},
    }
    return all(
        validate_typed(value[field], kind, field, shell)
        for field, kind in nested["field_types"].items()
    )


def validate_record(schema_name, record):
    schema = SCHEMAS[schema_name]
    if type(record) is not dict or set(record) != set(schema["exact_fields"]):
        return False
    if not all(
        validate_typed(record[field], kind, field, schema)
        for field, kind in schema["field_types"].items()
    ):
        return False
    if (
        schema["id_field"]
        and schema["id_prefix"]
        and not is_id(record[schema["id_field"]], schema["id_prefix"])
    ):
        return False
    return all(
        validate_nested(record[field], nested) for field, nested in schema["nested_schemas"].items()
    )


def resolve_strategy_definition_exact_version(context, definition_id, version):
    current = context["strategy_definitions_by_id"].get(definition_id)
    if current is not None and current["definition_version"] == version:
        return current
    return context["previous_strategy_definitions_by_version_key"].get(f"{definition_id}@{version}")


def validate_hash_lineage(current, previous, *, id_field, predecessor_field, hash_reference, scope):
    nodes = {**previous, **current}
    if set(previous) & set(current):
        return False
    for key, record in nodes.items():
        if record[id_field] != key or record["content_hash"] != canonical_m05_hash(
            hash_reference, record
        ):
            return False
        predecessor_id = record[predecessor_field]
        if "version" in record:
            if record["version"] == 1 and predecessor_id is not None:
                return False
            if record["version"] > 1 and predecessor_id is None:
                return False
        visited = {key}
        cursor = record
        while cursor[predecessor_field] is not None:
            predecessor_id = cursor[predecessor_field]
            if predecessor_id in visited or predecessor_id not in previous:
                return False
            visited.add(predecessor_id)
            predecessor = previous[predecessor_id]
            if any(predecessor[field] != cursor[field] for field in scope):
                return False
            if "version" in cursor and predecessor["version"] != cursor["version"] - 1:
                return False
            cursor = predecessor
    return True


def valid_projection_times(record, validation_time):
    try:
        observed = parse_time(record["observed_at_utc"])
        effective = parse_time(record["effective_at_utc"])
        stale_after = parse_time(record["stale_after_utc"])
    except ValueError:
        return False
    return (
        observed <= effective < stale_after
        and observed <= validation_time
        and effective <= validation_time
    )


def validate_instrument_record(context, record, validation_time):
    catalogs = {**context["previous_catalogs_by_id"], **context["catalogs_by_id"]}
    catalog = catalogs.get(record.get("catalog_snapshot_id"))
    decimal = re.compile(M05_CONTRACT["decimal_policy"]["regex"] + r"\Z")
    asset_fields = set(M05_CONTRACT["asset_reference_contract"]["fields"])

    def valid_asset(value):
        return (
            type(value) is dict
            and set(value) == asset_fields
            and all(type(value[field]) is str and value[field] for field in asset_fields)
            and value["asset_namespace"] == record["exchange_id"]
            and value["mapping_status"] in {"EXACT", "EXPLICIT_ALIAS"}
        )

    return bool(
        validate_record("InstrumentProjection", record)
        and record_matches_exchange_registry(record)
        and record["workspace_id"] in context["workspace_ids"]
        and record["venue_symbol"].strip() == record["venue_symbol"]
        and catalog
        and record["instrument_id"] in catalog["instrument_ids"]
        and all(
            record[field] == catalog[field]
            for field in ("exchange_id", "environment", "market_type")
        )
        and record["source_adapter_family_id"] == catalog["adapter_family_id"]
        and valid_projection_times(record, validation_time)
        and all(
            valid_asset(record[field])
            for field in ("base_asset_reference", "quote_asset_reference")
        )
        and all(
            type(record[field]) is str
            and decimal.fullmatch(record[field])
            and Decimal(record[field]) > 0
            for field in ("price_tick", "quantity_step")
        )
        and (
            record["instrument_type"] not in {"SPOT_PAIR", "MARGIN_PAIR"}
            or all(
                record[field] is None
                for field in (
                    "contract_size",
                    "contract_value_currency",
                    "derivative_settlement_type",
                    "expiry_at_utc",
                    "strike_price",
                    "option_side",
                )
            )
        )
    )


def validate_instrument_history_map(context, validation_time):
    for instrument_id, records in context["instrument_history_by_id"].items():
        if not is_id(instrument_id, "instr") or type(records) is not list or not records:
            return False
        if any(
            type(record) is not dict or not validate_record("InstrumentProjection", record)
            for record in records
        ):
            return False
        versions = [record["metadata_version"] for record in records]
        if len(versions) != len(records) or versions != sorted(set(versions)):
            return False
        identities = set()
        for record in records:
            if record.get("instrument_id") != instrument_id or not validate_instrument_record(
                context, record, validation_time
            ):
                return False
            identities.add(
                tuple(
                    record[field]
                    for field in ("exchange_id", "environment", "market_type", "venue_symbol")
                )
            )
        current = context["instruments_by_id"].get(instrument_id)
        if (
            len(identities) != 1
            or current
            and (
                tuple(
                    current[field]
                    for field in ("exchange_id", "environment", "market_type", "venue_symbol")
                )
                not in identities
                or current["metadata_version"] <= max(versions)
            )
        ):
            return False
    return True


def validate_snapshot_record(context, record, validation_time):
    account = context["accounts_by_id"].get(record["exchange_account_id"])
    entry = next(
        (
            item
            for item in canonical_exchange_entries()
            if item["exchange_id"] == record["exchange_id"]
        ),
        None,
    )
    types = set(record["supported_instrument_types"])
    return bool(
        account
        and entry
        and record_matches_exchange_registry(record)
        and all(
            record[field] == account[field]
            for field in ("exchange_id", "environment", "market_type")
        )
        and record["adapter_family_id"] == entry["adapter_family_id"]
        and set(record["observed_permission_set"])
        <= set(resolve_canonical_registry("m05_permissions"))
        and types <= set(resolve_canonical_registry("m05_instrument_types"))
        and types
        <= set(M05_CONTRACT["allowed_market_instrument_type_pairs"].get(record["market_type"], []))
        and types <= set(entry["supported_instrument_types"])
        and valid_projection_times(record, validation_time)
    )


def validate_catalog_record(context, record, validation_time, *, historical):
    if not record_matches_exchange_registry(record) or not valid_projection_times(
        record, validation_time
    ):
        return False
    for instrument_id in record["instrument_ids"]:
        instrument = (
            next(
                (
                    item
                    for item in context["instrument_history_by_id"].get(instrument_id, [])
                    if item["catalog_snapshot_id"] == record["catalog_snapshot_id"]
                ),
                None,
            )
            if historical
            else context["instruments_by_id"].get(instrument_id)
        )
        if (
            not instrument
            or any(
                instrument[field] != record[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or instrument["source_adapter_family_id"] != record["adapter_family_id"]
        ):
            return False
        if instrument["catalog_snapshot_id"] != record["catalog_snapshot_id"]:
            return False
    return True


def validate_universe_record(context, record, *, historical):
    account = context["accounts_by_id"].get(record["exchange_account_id"])
    lifecycle = record["lifecycle_state"]
    if (
        not account
        or (
            lifecycle == "ACTIVE"
            and (record["activated_at_utc"] is None or record["retired_at_utc"] is not None)
        )
        or (lifecycle == "RETIRED" and record["retired_at_utc"] is None)
        or (historical and lifecycle == "DRAFT")
    ):
        return False
    sources = set(record["source_catalog_snapshot_ids"])
    instruments = []
    for instrument_id in record["instrument_ids"]:
        candidates = [context["instruments_by_id"].get(instrument_id)]
        if historical:
            candidates += context["instrument_history_by_id"].get(instrument_id, [])
        instruments.append(
            next(
                (item for item in candidates if item and item["catalog_snapshot_id"] in sources),
                None,
            )
        )
    catalogs = {**context["previous_catalogs_by_id"], **context["catalogs_by_id"]}
    if any(item is None for item in instruments) or any(
        catalog_id not in catalogs for catalog_id in record["source_catalog_snapshot_ids"]
    ):
        return False
    for instrument in instruments:
        if any(
            instrument[field] != account[field]
            for field in ("exchange_id", "environment", "market_type")
        ):
            return False
    return set(record["source_catalog_snapshot_ids"]) == {
        item["catalog_snapshot_id"] for item in instruments
    }


def validate_credential_record(context, record, *, historical):
    account = context["accounts_by_id"].get(record["exchange_account_id"])
    if not account or not record_matches_exchange_registry(record):
        return False
    secure_ref = record["secure_store_reference"]
    if (
        record["exchange_id"] != account["exchange_id"]
        or record["environment_scope"] != account["environment"]
        or (historical and record["lifecycle_state"] != "RETIRED")
        or (historical and record["retired_at_utc"] is None)
        or (
            not historical
            and record["lifecycle_state"] == "ACTIVE"
            and record["retired_at_utc"] is not None
        )
        or type(secure_ref) is not str
        or not re.fullmatch(r"secure-store://[^\s?#=]+", secure_ref)
        or any(
            marker in secure_ref.lower()
            for marker in M05_CONTRACT["credential_profile_contract"][
                "secure_store_reference_grammar"
            ]["forbidden_payload_markers"]
        )
        or record["saas_sync_candidate"] is not False
    ):
        return False
    return True


def validate_all_canonical_projections(context, validation_time):
    if not validate_instrument_history_map(context, validation_time) or any(
        not validate_instrument_record(context, record, validation_time)
        for record in context["instruments_by_id"].values()
    ):
        return False
    for map_name in (
        "account_capability_snapshots_by_id",
        "previous_account_capability_snapshots_by_id",
    ):
        if any(
            not validate_snapshot_record(context, record, validation_time)
            for record in context[map_name].values()
        ):
            return False
    for map_name, historical in (("catalogs_by_id", False), ("previous_catalogs_by_id", True)):
        if any(
            not validate_catalog_record(context, record, validation_time, historical=historical)
            for record in context[map_name].values()
        ):
            return False
    for map_name, historical in (("universes_by_id", False), ("previous_universes_by_id", True)):
        if any(
            not validate_universe_record(context, record, historical=historical)
            for record in context[map_name].values()
        ):
            return False
    for map_name, historical in (
        ("credential_profiles_by_id", False),
        ("previous_credential_profiles_by_id", True),
    ):
        if any(
            not validate_credential_record(context, record, historical=historical)
            for record in context[map_name].values()
        ):
            return False
    return True


def validate_references(context):
    definitions = context["strategy_definitions_by_id"]
    history = context["previous_strategy_definitions_by_version_key"]
    validation_time = parse_time(context["validation_time_utc"])
    if not validate_all_canonical_projections(context, validation_time):
        return False
    for did, current in definitions.items():
        expected = {f"{did}@{version}" for version in range(1, current["definition_version"])}
        actual = {key for key in history if key.startswith(f"{did}@")}
        if actual != expected:
            return False
        versions = [current] + [history[key] for key in expected]
        if any(
            item["strategy_definition_id"] != did
            or item["workspace_id"] not in context["workspace_ids"]
            or item["canonical_content_hash"] != definition_hash(item["configuration"])
            or item["workspace_id"] != current["workspace_id"]
            or item["strategy_type_id"] != current["strategy_type_id"]
            or item["hash_domain_separator"] != current["hash_domain_separator"]
            or item["hash_domain_separator"]
            != CONTRACT["strategy_definition_contract"]["hash"]["domain_separator"]
            for item in versions
        ):
            return False
    if any(key.rsplit("@", 1)[0] not in definitions for key in history):
        return False
    if not validate_hash_lineage(
        context["account_capability_snapshots_by_id"],
        context["previous_account_capability_snapshots_by_id"],
        id_field="account_capability_snapshot_id",
        predecessor_field="previous_snapshot_id",
        hash_reference="account_capability_snapshot",
        scope=("exchange_account_id", "exchange_id", "environment", "market_type"),
    ):
        return False
    if not validate_hash_lineage(
        context["catalogs_by_id"],
        context["previous_catalogs_by_id"],
        id_field="catalog_snapshot_id",
        predecessor_field="previous_snapshot_id",
        hash_reference="instrument_catalog_snapshot",
        scope=("exchange_id", "environment", "market_type", "adapter_family_id"),
    ):
        return False
    if not validate_hash_lineage(
        context["universes_by_id"],
        context["previous_universes_by_id"],
        id_field="trading_universe_id",
        predecessor_field="previous_version_id",
        hash_reference="trading_universe",
        scope=("exchange_account_id",),
    ):
        return False
    exchange_bound_maps = (
        "accounts_by_id",
        "instruments_by_id",
        "catalogs_by_id",
        "account_capability_snapshots_by_id",
        "credential_profiles_by_id",
        "market_data_routes_by_id",
        "execution_routes_by_id",
    )
    if any(
        not record_matches_exchange_registry(record)
        for map_name in exchange_bound_maps
        for record in context[map_name].values()
    ):
        return False
    for instrument in context["instruments_by_id"].values():
        catalog = context["catalogs_by_id"].get(instrument["catalog_snapshot_id"])
        try:
            observed = parse_time(instrument["observed_at_utc"])
            effective = parse_time(instrument["effective_at_utc"])
            stale_after = parse_time(instrument["stale_after_utc"])
        except ValueError:
            return False
        if (
            instrument["workspace_id"] not in context["workspace_ids"]
            or type(instrument["metadata_version"]) is not int
            or instrument["metadata_version"] < 1
            or not observed <= effective < stale_after
            or observed > validation_time
            or effective > validation_time
            or not catalog
            or instrument["instrument_id"] not in catalog["instrument_ids"]
            or any(
                instrument[field] != catalog[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or instrument["source_adapter_family_id"] != catalog["adapter_family_id"]
        ):
            return False
    for cid, catalog in context["catalogs_by_id"].items():
        try:
            observed = parse_time(catalog["observed_at_utc"])
            effective = parse_time(catalog["effective_at_utc"])
            stale_after = parse_time(catalog["stale_after_utc"])
            validation_time = parse_time(context["validation_time_utc"])
        except ValueError:
            return False
        if not observed <= effective <= validation_time or stale_after <= effective:
            return False
        for iid in catalog["instrument_ids"]:
            instrument = context["instruments_by_id"].get(iid)
            if (
                not instrument
                or instrument["catalog_snapshot_id"] != cid
                or instrument["source_adapter_family_id"] != catalog["adapter_family_id"]
                or any(
                    instrument[field] != catalog[field]
                    for field in ("exchange_id", "environment", "market_type")
                )
            ):
                return False
    for universe in context["universes_by_id"].values():
        account = context["accounts_by_id"].get(universe["exchange_account_id"])
        instruments = [context["instruments_by_id"].get(iid) for iid in universe["instrument_ids"]]
        if not account or any(item is None for item in instruments):
            return False
        if set(universe["source_catalog_snapshot_ids"]) != {
            item["catalog_snapshot_id"] for item in instruments
        }:
            return False
        for item in instruments:
            catalog = context["catalogs_by_id"].get(item["catalog_snapshot_id"])
            if not catalog or item["instrument_id"] not in catalog["instrument_ids"]:
                return False
            if (
                any(
                    item[field] != account[field]
                    for field in ("exchange_id", "environment", "market_type")
                )
                or item["workspace_id"] != account["workspace_id"]
            ):
                return False
    for account in context["accounts_by_id"].values():
        if account["workspace_id"] not in context["workspace_ids"]:
            return False
        portfolio = context["portfolios_by_id"].get(account["portfolio_id"])
        snapshot = context["account_capability_snapshots_by_id"].get(
            account["account_capability_snapshot_id"]
        )
        exchange_entry = next(
            entry
            for entry in canonical_exchange_entries()
            if entry["exchange_id"] == account["exchange_id"]
        )
        snapshot_required = (
            exchange_entry["capability_discovery_policy"] == "ADAPTER_SNAPSHOT_REQUIRED"
        )
        product = context["product_capabilities_by_environment"].get(account["environment"])
        if (
            not portfolio
            or portfolio["workspace_id"] != account["workspace_id"]
            or (snapshot_required and not snapshot)
            or not product
            or product["environment"] != account["environment"]
        ):
            return False
        if snapshot and (
            snapshot["exchange_account_id"] != account["exchange_account_id"]
            or any(
                snapshot[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
        ):
            return False
        if snapshot is None:
            continue
        try:
            observed = parse_time(snapshot["observed_at_utc"])
            effective = parse_time(snapshot["effective_at_utc"])
            stale_after = parse_time(snapshot["stale_after_utc"])
        except ValueError:
            return False
        if (
            not observed <= effective < stale_after
            or observed > validation_time
            or effective > validation_time
        ):
            return False
        credential_id = account["active_credential_profile_id"]
        if credential_id is not None:
            credential = context["credential_profiles_by_id"].get(credential_id)
            if (
                not credential
                or credential["lifecycle_state"] != "ACTIVE"
                or credential["exchange_account_id"] != account["exchange_account_id"]
                or credential["exchange_id"] != account["exchange_id"]
                or credential["environment_scope"] != account["environment"]
            ):
                return False
    for snapshot in context["account_capability_snapshots_by_id"].values():
        account = context["accounts_by_id"].get(snapshot["exchange_account_id"])
        instrument_types = set(snapshot["supported_instrument_types"])
        registry_types = set(resolve_canonical_registry("m05_instrument_types"))
        allowed_for_market = set(
            M05_CONTRACT["allowed_market_instrument_type_pairs"].get(snapshot["market_type"], [])
        )
        exchange_entry = next(
            entry
            for entry in canonical_exchange_entries()
            if entry["exchange_id"] == snapshot["exchange_id"]
        )
        if (
            not account
            or any(
                snapshot[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or not instrument_types <= registry_types
            or not instrument_types <= allowed_for_market
            or not instrument_types <= set(exchange_entry["supported_instrument_types"])
        ):
            return False
    if any(
        portfolio["workspace_id"] not in context["workspace_ids"]
        for portfolio in context["portfolios_by_id"].values()
    ):
        return False
    selected = [
        value
        for value in context["accounts_by_id"].values()
        if value["active_credential_profile_id"]
    ]
    if set(context["credential_profiles_by_id"]) != {
        item["active_credential_profile_id"] for item in selected
    }:
        return False
    previous_credentials = context["previous_credential_profiles_by_id"]
    reached_credentials = set()
    for credential in context["credential_profiles_by_id"].values():
        visited = {credential["credential_profile_id"]}
        cursor = credential
        while cursor["rotated_from_credential_profile_id"] is not None:
            predecessor_id = cursor["rotated_from_credential_profile_id"]
            if predecessor_id in visited or predecessor_id not in previous_credentials:
                return False
            visited.add(predecessor_id)
            reached_credentials.add(predecessor_id)
            predecessor = previous_credentials[predecessor_id]
            if (
                predecessor["lifecycle_state"] != "RETIRED"
                or predecessor["exchange_account_id"] != credential["exchange_account_id"]
                or predecessor["exchange_id"] != credential["exchange_id"]
                or predecessor["environment_scope"] != credential["environment_scope"]
                or parse_time(predecessor["created_at_utc"])
                > parse_time(predecessor["retired_at_utc"])
                or parse_time(predecessor["retired_at_utc"]) > parse_time(cursor["created_at_utc"])
            ):
                return False
            cursor = predecessor
    if reached_credentials != set(previous_credentials):
        return False
    identity_tuples = set()
    for account_id, identity in context["external_identity_snapshots_by_account_id"].items():
        account = context["accounts_by_id"].get(account_id)
        entry = next(
            (
                item
                for item in canonical_exchange_entries()
                if item["exchange_id"] == identity["exchange_id"]
            ),
            None,
        )
        adapter_source = identity["adapter_version_source"]
        if (
            not account
            or not entry
            or any(
                identity[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or parse_time(identity["verification_timestamp"]) > validation_time
            or type(adapter_source) is not str
            or not re.fullmatch(rf"{re.escape(entry['adapter_family_id'])}/[^/\s]+", adapter_source)
        ):
            return False
        identity_tuple = (
            identity["exchange_id"],
            identity["environment"],
            identity["market_type"],
            identity["venue_account_identifier"],
            identity["subaccount_identifier"],
            identity["account_type"],
        )
        if identity_tuple in identity_tuples:
            return False
        identity_tuples.add(identity_tuple)
    for route in context["market_data_routes_by_id"].values():
        endpoint = CONTRACT["endpoint_class_registry"].get(route["endpoint_class"])
        instruments = [context["instruments_by_id"].get(iid) for iid in route["instrument_ids"]]
        if (
            route["workspace_id"] not in context["workspace_ids"]
            or endpoint is None
            or endpoint["environment"] != route["environment"]
            or "MARKET_DATA" not in endpoint["allowed_route_kinds"]
            or endpoint["access_scope"] != route["data_scope"]
            or any(item is None for item in instruments)
            or (
                route["data_scope"] == "PUBLIC"
                and any(channel.startswith("PRIVATE_") for channel in route["channel_types"])
            )
        ):
            return False
        for instrument in instruments:
            catalog = context["catalogs_by_id"].get(instrument["catalog_snapshot_id"])
            if (
                not catalog
                or any(
                    route[field] != instrument[field]
                    for field in ("exchange_id", "environment", "market_type")
                )
                or route["adapter_family_id"] != instrument["source_adapter_family_id"]
                or route["adapter_family_id"] != catalog["adapter_family_id"]
                or instrument["workspace_id"] != route["workspace_id"]
            ):
                return False
    for route in context["execution_routes_by_id"].values():
        account = context["accounts_by_id"].get(route["exchange_account_id"])
        endpoint = CONTRACT["endpoint_class_registry"].get(route["endpoint_class"])
        if (
            not account
            or route["workspace_id"] not in context["workspace_ids"]
            or account["workspace_id"] != route["workspace_id"]
            or any(
                route[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or endpoint is None
            or endpoint["environment"] != route["environment"]
            or "EXECUTION" not in endpoint["allowed_route_kinds"]
        ):
            return False
    for instance in context["strategy_instances_by_id"].values():
        account = context["accounts_by_id"].get(instance["exchange_account_id"])
        universe = context["universes_by_id"].get(instance["trading_universe_id"])
        definition = resolve_strategy_definition_exact_version(
            context, instance["strategy_definition_id"], instance["strategy_definition_version"]
        )
        portfolio = context["portfolios_by_id"].get(instance["portfolio_id"])
        if not account or not universe or not definition or not portfolio:
            return False
        if (
            instance["workspace_id"] not in context["workspace_ids"]
            or portfolio["workspace_id"] != instance["workspace_id"]
            or account["workspace_id"] != instance["workspace_id"]
            or universe["exchange_account_id"] != account["exchange_account_id"]
            or definition["workspace_id"] != instance["workspace_id"]
            or definition["lifecycle_state"] == "DRAFT"
        ):
            return False
        instruments = [context["instruments_by_id"][iid] for iid in universe["instrument_ids"]]
        if any(
            instrument["workspace_id"] != instance["workspace_id"]
            or instrument["workspace_id"] != account["workspace_id"]
            for instrument in instruments
        ):
            return False
        market = (
            context["market_data_routes_by_id"].get(instance["market_data_route_id"])
            if instance["market_data_route_id"]
            else None
        )
        execution = (
            context["execution_routes_by_id"].get(instance["execution_route_id"])
            if instance["execution_route_id"]
            else None
        )
        if (
            instance["market_data_route_id"]
            and not market
            or instance["execution_route_id"]
            and not execution
        ):
            return False
        if instance["lifecycle_state"] in {"BOUND", "ACTIVE", "INACTIVE"} and not (
            market and execution
        ):
            return False
        if instance["lifecycle_state"] == "DRAFT" and market and execution:
            return False
        if market and (
            market["workspace_id"] != instance["workspace_id"]
            or any(
                market[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
        ):
            return False
        if execution and (
            execution["workspace_id"] != instance["workspace_id"]
            or execution["exchange_account_id"] != account["exchange_account_id"]
            or any(
                execution[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
        ):
            return False
        if market and any(
            instrument["workspace_id"] != market["workspace_id"] for instrument in instruments
        ):
            return False
        if execution and any(
            instrument["workspace_id"] != execution["workspace_id"] for instrument in instruments
        ):
            return False
        if execution:
            for iid in universe["instrument_ids"]:
                instrument = context["instruments_by_id"][iid]
                catalog = context["catalogs_by_id"][instrument["catalog_snapshot_id"]]
                if (
                    instrument["workspace_id"] != instance["workspace_id"]
                    or instrument["workspace_id"] != account["workspace_id"]
                    or (market and instrument["workspace_id"] != market["workspace_id"])
                    or instrument["workspace_id"] != execution["workspace_id"]
                    or execution["adapter_family_id"] != instrument["source_adapter_family_id"]
                    or execution["adapter_family_id"] != catalog["adapter_family_id"]
                ):
                    return False
    active = {
        item["strategy_instance_id"]
        for item in context["strategy_instances_by_id"].values()
        if item["lifecycle_state"] == "ACTIVE"
    }
    if set(context["active_strategy_instances"]) != active:
        return False
    routes = {
        **{key: "MARKET_DATA" for key in context["market_data_routes_by_id"]},
        **{key: "EXECUTION" for key in context["execution_routes_by_id"]},
    }
    now = parse_time(context["validation_time_utc"])
    for rid, readiness in context["route_readiness_by_id"].items():
        if (
            rid not in routes
            or readiness["route_id"] != rid
            or readiness["route_kind"] != routes[rid]
            or parse_time(readiness["observed_at"]) > now
        ):
            return False
    return True


def validate_context(_request, context, operation):
    spec = CONTRACT["trusted_validation_context"]
    expected = (
        set(spec["map_fields"])
        | set(spec["history_map_fields"])
        | set(spec["scalar_fields"])
        | set(spec["array_fields"])
    )
    if type(context) is not dict or set(context) != expected:
        return deny(operation, "TRUSTED_CONTEXT_INVALID")
    for field, schema_name in spec["map_fields"].items():
        values = context[field]
        if type(values) is not dict:
            return deny(operation, "TRUSTED_CONTEXT_INVALID")
        for key, record in values.items():
            if not validate_record(schema_name, record):
                return deny(operation, "TRUSTED_CONTEXT_INVALID")
            id_field = SCHEMAS[schema_name]["id_field"]
            expected_key = record[id_field] if id_field else None
            if field == "previous_strategy_definitions_by_version_key":
                expected_key = f"{record['strategy_definition_id']}@{record['definition_version']}"
            if field == "product_capabilities_by_environment":
                expected_key = record["environment"]
            if field == "external_identity_snapshots_by_account_id":
                expected_key = key if key in context["accounts_by_id"] else None
            if key != expected_key:
                return deny(operation, "TRUSTED_CONTEXT_INVALID")
    for field in spec["history_map_fields"]:
        if type(context[field]) is not dict:
            return deny(operation, "TRUSTED_CONTEXT_INVALID")
    try:
        parse_time(context["validation_time_utc"])
    except ValueError:
        return deny(operation, "TRUSTED_CONTEXT_INVALID")
    for field, policy in spec["array_fields"].items():
        value = context[field]
        if (
            type(value) is not list
            or len(value) != len(set(value))
            or (not policy["empty_allowed"] and not value)
            or not all(is_id(item, policy["id_prefix"]) for item in value)
        ):
            return deny(operation, "TRUSTED_CONTEXT_INVALID")
    if not validate_references(context):
        return deny(operation, "TRUSTED_CONTEXT_INVALID")
    return None


def validate_request(request, _context, operation):
    if operation not in REQUESTS:
        return deny(operation, "UNKNOWN_OPERATION")
    schema = REQUESTS[operation]
    if type(request) is not dict or set(request) != set(schema["request_fields"]):
        return deny(operation, "REQUEST_SCHEMA_INVALID")
    shell = {
        "nullable_fields": schema["nullable_fields"],
        "enum_registry": {},
        "nested_schemas": schema["nested_schemas"],
        "array_policy": {},
    }
    if not all(
        validate_typed(request[field], kind, field, shell)
        for field, kind in schema["request_types"].items()
    ):
        return deny(operation, "REQUEST_SCHEMA_INVALID")
    if any(request[field] != value for field, value in schema["constants"].items()):
        return deny(operation, "REQUEST_SCHEMA_INVALID")
    if not all(
        validate_nested(request[field], nested)
        for field, nested in schema["nested_schemas"].items()
        if nested.get("exact_fields")
    ):
        return deny(operation, "REQUEST_SCHEMA_INVALID")
    return None


def success(operation, transition):
    return decision(True, operation, transition=transition)


def instance_lookup(operation, request, context):
    value = context["strategy_instances_by_id"].get(request["strategy_instance_id"])
    return value if value else deny(operation, "STRATEGY_INSTANCE_NOT_FOUND")


def validate_definition_create(request, context, operation):
    did = request["strategy_definition_id"]
    if request["workspace_id"] not in context["workspace_ids"]:
        return deny(operation, "WORKSPACE_NOT_FOUND")
    current = context["strategy_definitions_by_id"].get(did)
    if current and request["definition_version"] <= current["definition_version"]:
        return deny(operation, "STRATEGY_DEFINITION_ID_COLLISION")
    expected = 1 if current is None else current["definition_version"] + 1
    if request["definition_version"] != expected:
        return deny(operation, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    if current and (
        request["workspace_id"] != current["workspace_id"]
        or request["strategy_type_id"] != current["strategy_type_id"]
    ):
        return deny(operation, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    return success(
        operation,
        {
            "entity": "StrategyDefinition",
            "strategy_definition_id": request["strategy_definition_id"],
            "workspace_id": request["workspace_id"],
            "strategy_type_id": request["strategy_type_id"],
            "definition_version": expected,
            "configuration": copy.deepcopy(request["configuration"]),
            "hash_domain_separator": CONTRACT["strategy_definition_contract"]["hash"][
                "domain_separator"
            ],
            "canonical_content_hash": definition_hash(request["configuration"]),
            "lifecycle_state": "DRAFT",
        },
    )


def validate_definition_lifecycle(request, context, operation):
    record = resolve_strategy_definition_exact_version(
        context, request["strategy_definition_id"], request["definition_version"]
    )
    current = context["strategy_definitions_by_id"].get(request["strategy_definition_id"])
    if record is None:
        return deny(
            operation,
            "STRATEGY_DEFINITION_VERSION_MISMATCH" if current else "STRATEGY_DEFINITION_NOT_FOUND",
        )
    if current is None or current["definition_version"] != request["definition_version"]:
        return deny(operation, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    target = "ACTIVE" if operation.startswith("ACTIVATE") else "RETIRED"
    allowed = {"ACTIVE": "DRAFT", "RETIRED": "ACTIVE"}
    if record["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    if record["lifecycle_state"] != allowed[target]:
        return deny(operation, "STRATEGY_DEFINITION_STATE_CONFLICT")
    return success(
        operation,
        {
            "entity": "StrategyDefinition",
            "strategy_definition_id": request["strategy_definition_id"],
            "definition_version": request["definition_version"],
            "from": record["lifecycle_state"],
            "to": target,
        },
    )


def validate_instance_create(request, context, operation):
    if request["strategy_instance_id"] in context["strategy_instances_by_id"]:
        return deny(operation, "STRATEGY_INSTANCE_ID_COLLISION")
    current = context["strategy_definitions_by_id"].get(request["strategy_definition_id"])
    if not current:
        return deny(operation, "STRATEGY_DEFINITION_NOT_FOUND")
    if request["strategy_definition_version"] != current["definition_version"]:
        return deny(operation, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    if current["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    if current["lifecycle_state"] != "ACTIVE":
        return deny(operation, "STRATEGY_DEFINITION_STATE_CONFLICT")
    account = context["accounts_by_id"].get(request["exchange_account_id"])
    portfolio = context["portfolios_by_id"].get(request["portfolio_id"])
    if (
        request["workspace_id"] not in context["workspace_ids"]
        or not account
        or not portfolio
        or current["workspace_id"] != request["workspace_id"]
        or account["workspace_id"] != request["workspace_id"]
        or portfolio["workspace_id"] != request["workspace_id"]
        or account["portfolio_id"] != request["portfolio_id"]
        or request["market_data_route_id"] is not None
        or request["execution_route_id"] is not None
    ):
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    universe = context["universes_by_id"].get(request["trading_universe_id"])
    if (
        not universe
        or universe["exchange_account_id"] != request["exchange_account_id"]
        or universe["lifecycle_state"] != "ACTIVE"
    ):
        return deny(operation, "TRADING_UNIVERSE_INVALID")
    return success(
        operation,
        {
            "entity": "StrategyInstance",
            "strategy_instance_id": request["strategy_instance_id"],
            "workspace_id": request["workspace_id"],
            "portfolio_id": request["portfolio_id"],
            "strategy_definition_id": request["strategy_definition_id"],
            "strategy_definition_version": request["strategy_definition_version"],
            "exchange_account_id": request["exchange_account_id"],
            "trading_universe_id": request["trading_universe_id"],
            "market_data_route_id": None,
            "execution_route_id": None,
            "lifecycle_state": "DRAFT",
        },
    )


def endpoint_ok(route, account, kind):
    endpoint = CONTRACT["endpoint_class_registry"][route["endpoint_class"]]
    return (
        endpoint["environment"] == account["environment"]
        and kind in endpoint["allowed_route_kinds"]
    )


def validate_market_bind(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    if item["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    if item["lifecycle_state"] != "DRAFT":
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    if item["market_data_route_id"] is not None:
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    route = context["market_data_routes_by_id"].get(request["market_data_route_id"])
    if not route:
        return deny(operation, "MARKET_DATA_ROUTE_NOT_FOUND")
    account = context["accounts_by_id"][item["exchange_account_id"]]
    if route["data_scope"] == "PUBLIC" and any(
        channel.startswith("PRIVATE_") for channel in route["channel_types"]
    ):
        return deny(operation, "ROUTE_SCOPE_MISMATCH")
    if route["workspace_id"] != item["workspace_id"] or any(
        route[f] != account[f] for f in ("exchange_id", "market_type")
    ):
        return deny(operation, "ROUTE_SCOPE_MISMATCH")
    if route["environment"] != account["environment"] or not endpoint_ok(
        route, account, "MARKET_DATA"
    ):
        return deny(operation, "ENDPOINT_FALLBACK_FORBIDDEN")
    universe = context["universes_by_id"][item["trading_universe_id"]]
    if not set(universe["instrument_ids"]).issubset(route["instrument_ids"]):
        return deny(operation, "INSTRUMENT_SCOPE_MISMATCH")
    return success(
        operation,
        {
            "entity": "StrategyInstance",
            "strategy_instance_id": item["strategy_instance_id"],
            "bind": "market_data_route_id",
            "value": route["market_data_route_id"],
            "resulting_state": "BOUND" if item["execution_route_id"] else "DRAFT",
        },
    )


def validate_execution_bind(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    if item["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    if item["lifecycle_state"] != "DRAFT":
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    if item["execution_route_id"] is not None:
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    route = context["execution_routes_by_id"].get(request["execution_route_id"])
    if not route:
        return deny(operation, "EXECUTION_ROUTE_NOT_FOUND")
    account = context["accounts_by_id"][item["exchange_account_id"]]
    if (
        route["exchange_account_id"] != account["exchange_account_id"]
        or route["workspace_id"] != item["workspace_id"]
        or any(route[f] != account[f] for f in ("exchange_id", "market_type"))
    ):
        return deny(operation, "ROUTE_SCOPE_MISMATCH")
    expected = CONTRACT["authorization_dependencies_by_environment"][account["environment"]]
    if set(route["authorization_dependencies"]) != set(expected):
        return deny(operation, "ROUTE_CAPABILITY_BLOCKED")
    universe = context["universes_by_id"][item["trading_universe_id"]]
    return success(
        operation,
        {
            "entity": "StrategyInstance",
            "strategy_instance_id": item["strategy_instance_id"],
            "bind": "execution_route_id",
            "value": route["execution_route_id"],
            "resulting_state": "BOUND" if item["market_data_route_id"] else "DRAFT",
        },
    )


def readiness_denial(item, context, operation):
    now = parse_time(context["validation_time_utc"])
    for field, missing, stale in [
        ("market_data_route_id", "MARKET_DATA_ROUTE_NOT_READY", "MARKET_DATA_FRESHNESS_BLOCKED"),
        ("execution_route_id", "EXECUTION_ROUTE_NOT_READY", "EXECUTION_ROUTE_NOT_READY"),
    ]:
        rid = item[field]
        ready = context["route_readiness_by_id"].get(rid)
        if not ready or ready["readiness_state"] != "READY":
            return deny(operation, missing)
        route = (
            context["market_data_routes_by_id"][rid]
            if field == "market_data_route_id"
            else context["execution_routes_by_id"][rid]
        )
        max_age = (
            route["freshness_policy"]["max_age_seconds"]
            if field == "market_data_route_id"
            else CONTRACT["execution_route_contract"]["execution_readiness_max_age_seconds"]
        )
        if (now - parse_time(ready["observed_at"])).total_seconds() > max_age:
            return deny(operation, stale)
        if field == "market_data_route_id" and ready["sequence_state"] != "CONTIGUOUS":
            return deny(operation, "MARKET_DATA_SEQUENCE_INVALID")
        if field == "execution_route_id" and ready["sequence_state"] != "NOT_APPLICABLE":
            return deny(operation, "EXECUTION_ROUTE_NOT_READY")
    return None


def bound_route_operability_denial(item, context, operation):
    account = context["accounts_by_id"][item["exchange_account_id"]]
    universe = context["universes_by_id"][item["trading_universe_id"]]
    market = context["market_data_routes_by_id"].get(item["market_data_route_id"])
    execution = context["execution_routes_by_id"].get(item["execution_route_id"])
    if not market or market["route_status"] != "ACTIVE":
        return deny(operation, "MARKET_DATA_ROUTE_NOT_READY")
    if not execution or execution["route_status"] != "ACTIVE":
        return deny(operation, "EXECUTION_ROUTE_NOT_READY")
    if not set(universe["instrument_ids"]).issubset(market["instrument_ids"]):
        return deny(operation, "INSTRUMENT_SCOPE_MISMATCH")
    instruments = [context["instruments_by_id"][iid] for iid in universe["instrument_ids"]]
    if any(
        item["instrument_type"] not in execution["supported_instrument_types"]
        for item in instruments
    ):
        return deny(operation, "INSTRUMENT_SCOPE_MISMATCH")
    expected = set(CONTRACT["authorization_dependencies_by_environment"][account["environment"]])
    if set(execution["authorization_dependencies"]) != expected:
        return deny(operation, "ROUTE_CAPABILITY_BLOCKED")
    return None


def validate_authorization_operability(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    account = context["accounts_by_id"][item["exchange_account_id"]]
    product = context["product_capabilities_by_environment"][account["environment"]]
    required_product_capability = {
        "PAPER": "PAPER_LOCAL_SIMULATION",
        "TESTNET": "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
    }.get(account["environment"])
    if (
        not product["execution_enabled"]
        or required_product_capability not in product["allowed_operations"]
    ):
        return deny(operation, "PRODUCT_CAPABILITIES_BLOCKED")
    if account["environment"] == "PAPER":
        return None
    policy = M05_CONTRACT["current_edition_account_operability_policy"]
    if (
        account["lifecycle_state"] not in policy["operational_lifecycle_states"]
        or account["connection_state"] not in policy["operational_connection_states"]
        or account["execution_authorization"] not in policy["operational_authorization_states"]
    ):
        return deny(operation, "ACCOUNT_READINESS_BLOCKED")
    route = context["execution_routes_by_id"][item["execution_route_id"]]
    snapshot = context["account_capability_snapshots_by_id"][
        account["account_capability_snapshot_id"]
    ]
    max_age = CONTRACT["account_capability_snapshot_policy"]["max_age_seconds_by_environment"][
        "TESTNET"
    ]
    age = (
        parse_time(context["validation_time_utc"]) - parse_time(snapshot["observed_at_utc"])
    ).total_seconds()
    if (
        snapshot["status"] != "VALID"
        or age > max_age
        or parse_time(context["validation_time_utc"]) >= parse_time(snapshot["stale_after_utc"])
    ):
        return deny(operation, "CAPABILITY_SNAPSHOT_BLOCKED")
    credential = context["credential_profiles_by_id"].get(account["active_credential_profile_id"])
    identity = context["external_identity_snapshots_by_account_id"].get(
        account["exchange_account_id"]
    )
    identities = list(context["external_identity_snapshots_by_account_id"].values())
    identity_tuple = (
        (
            identity.get("exchange_id"),
            identity.get("environment"),
            identity.get("market_type"),
            identity.get("venue_account_identifier"),
            identity.get("subaccount_identifier"),
        )
        if identity
        else None
    )
    identity_collision = (
        identity_tuple is not None
        and sum(
            (
                candidate["exchange_id"],
                candidate["environment"],
                candidate["market_type"],
                candidate["venue_account_identifier"],
                candidate["subaccount_identifier"],
            )
            == identity_tuple
            for candidate in identities
        )
        != 1
    )
    if (
        not identity
        or account["external_account_identity_state"] != "VERIFIED"
        or identity["state"] != "VERIFIED"
        or any(
            identity[field] != account[field]
            for field in ("exchange_id", "environment", "market_type")
        )
        or identity_collision
    ):
        return deny(operation, "ACCOUNT_READINESS_BLOCKED")
    required_permissions = {"READ_ACCOUNT", "PLACE_ORDERS"}
    permission_sources = (
        set(snapshot["observed_permission_set"]),
        set(credential["permission_snapshot"]) if credential else set(),
        set(identity["observed_permission_set"]),
    )
    if (
        not credential
        or credential["lifecycle_state"] != "ACTIVE"
        or credential["credential_purpose"] != "ORDER_ENTRY"
        or any(not required_permissions <= permissions for permissions in permission_sources)
        or any("WITHDRAW" in permissions for permissions in permission_sources)
    ):
        return deny(operation, "ACCOUNT_READINESS_BLOCKED")
    if "PLACE_ORDERS" not in route["route_capability_ceiling"]:
        return deny(operation, "ROUTE_CAPABILITY_BLOCKED")
    return None


def validate_strategy_execution_operability(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    account = context["accounts_by_id"][item["exchange_account_id"]]
    universe = context["universes_by_id"][item["trading_universe_id"]]
    if universe["lifecycle_state"] != "ACTIVE":
        return deny(operation, "TRADING_UNIVERSE_INVALID")
    instruments = [context["instruments_by_id"][iid] for iid in universe["instrument_ids"]]
    catalogs = [
        context["catalogs_by_id"][catalog_snapshot_id]
        for catalog_snapshot_id in {instrument["catalog_snapshot_id"] for instrument in instruments}
    ]
    validation_time = parse_time(context["validation_time_utc"])
    if any(
        catalog["status"] != "VALID" or validation_time >= parse_time(catalog["stale_after_utc"])
        for catalog in catalogs
    ):
        return deny(operation, "TRADING_UNIVERSE_INVALID")
    if any(
        instrument["workspace_id"] != item["workspace_id"]
        or instrument["trading_status"] != "TRADING"
        or validation_time >= parse_time(instrument["stale_after_utc"])
        for instrument in instruments
    ):
        return deny(operation, "INSTRUMENT_SCOPE_MISMATCH")
    if account["environment"] == "TESTNET":
        snapshot = context["account_capability_snapshots_by_id"][
            account["account_capability_snapshot_id"]
        ]
        if not {instrument["instrument_type"] for instrument in instruments}.issubset(
            snapshot["supported_instrument_types"]
        ):
            return deny(operation, "CAPABILITY_SNAPSHOT_BLOCKED")
    return None


def validate_readiness_lifecycle(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    if item["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    if item["lifecycle_state"] == "DRAFT":
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    return None


def validate_strategy_instance_lifecycle_preflight(request, context, operation):
    if operation not in {"ACTIVATE_STRATEGY_INSTANCE", "DEACTIVATE_STRATEGY_INSTANCE"}:
        return deny(operation, "CONTRACT_INCONSISTENT")
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    if item["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    allowed_states = (
        {"BOUND", "INACTIVE"} if operation == "ACTIVATE_STRATEGY_INSTANCE" else {"ACTIVE"}
    )
    if item["lifecycle_state"] not in allowed_states:
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    return None


def validate_route_readiness_operation(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    blocked = bound_route_operability_denial(item, context, operation) or readiness_denial(
        item, context, operation
    )
    return blocked or success(
        operation, {"validated": True, "strategy_instance_id": item["strategy_instance_id"]}
    )


def validate_activation(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    definition = resolve_strategy_definition_exact_version(
        context, item["strategy_definition_id"], item["strategy_definition_version"]
    )
    if not definition:
        return deny(operation, "STRATEGY_DEFINITION_NOT_FOUND")
    current = context["strategy_definitions_by_id"].get(item["strategy_definition_id"])
    if item["strategy_definition_version"] > current["definition_version"]:
        return deny(operation, "STRATEGY_DEFINITION_VERSION_MISMATCH")
    if definition["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    blocked = bound_route_operability_denial(item, context, operation) or readiness_denial(
        item, context, operation
    )
    if blocked:
        return blocked
    return success(
        operation,
        {
            "entity": "StrategyInstance",
            "strategy_instance_id": item["strategy_instance_id"],
            "from": item["lifecycle_state"],
            "to": "ACTIVE",
        },
    )


def validate_deactivation(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    return success(
        operation,
        {
            "entity": "StrategyInstance",
            "strategy_instance_id": item["strategy_instance_id"],
            "from": "ACTIVE",
            "to": "INACTIVE",
        },
    )


def validate_retirement(request, context, operation):
    item = instance_lookup(operation, request, context)
    if isinstance(item, MappingProxyType):
        return item
    if item["lifecycle_state"] == "ACTIVE":
        return deny(operation, "STRATEGY_INSTANCE_BINDING_MISMATCH")
    if item["lifecycle_state"] == "RETIRED":
        return deny(operation, "RETIRED_RESOURCE_FORBIDDEN")
    return success(
        operation,
        {
            "entity": "StrategyInstance",
            "strategy_instance_id": item["strategy_instance_id"],
            "from": item["lifecycle_state"],
            "to": "RETIRED",
        },
    )


VALIDATORS = {
    name: value
    for name, value in globals().copy().items()
    if name in CONTRACT["validator_denial_registry"]
}


def planned_outcome_is_valid(operation, request, context, result):
    transition = result["planned_transition"]
    if operation == "CREATE_STRATEGY_DEFINITION":
        current = context["strategy_definitions_by_id"].get(request["strategy_definition_id"])
        return transition == {
            "entity": "StrategyDefinition",
            "strategy_definition_id": request["strategy_definition_id"],
            "workspace_id": request["workspace_id"],
            "strategy_type_id": request["strategy_type_id"],
            "definition_version": 1 if current is None else current["definition_version"] + 1,
            "configuration": request["configuration"],
            "hash_domain_separator": CONTRACT["strategy_definition_contract"]["hash"][
                "domain_separator"
            ],
            "canonical_content_hash": definition_hash(request["configuration"]),
            "lifecycle_state": "DRAFT",
        }
    if operation in {"ACTIVATE_STRATEGY_DEFINITION", "RETIRE_STRATEGY_DEFINITION"}:
        target = "ACTIVE" if operation.startswith("ACTIVATE") else "RETIRED"
        source = "DRAFT" if target == "ACTIVE" else "ACTIVE"
        return transition == {
            "entity": "StrategyDefinition",
            "strategy_definition_id": request["strategy_definition_id"],
            "definition_version": request["definition_version"],
            "from": source,
            "to": target,
        }
    if operation == "CREATE_STRATEGY_INSTANCE":
        expected = {
            "entity": "StrategyInstance",
            "strategy_instance_id": request["strategy_instance_id"],
            "workspace_id": request["workspace_id"],
            "portfolio_id": request["portfolio_id"],
            "strategy_definition_id": request["strategy_definition_id"],
            "strategy_definition_version": request["strategy_definition_version"],
            "exchange_account_id": request["exchange_account_id"],
            "trading_universe_id": request["trading_universe_id"],
            "market_data_route_id": None,
            "execution_route_id": None,
            "lifecycle_state": "DRAFT",
        }
        return transition == expected
    if operation in {"BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"}:
        item = context["strategy_instances_by_id"][request["strategy_instance_id"]]
        market = operation == "BIND_MARKET_DATA_ROUTE"
        binding = "market_data_route_id" if market else "execution_route_id"
        other = "execution_route_id" if market else "market_data_route_id"
        route_request = binding
        return (
            transition
            == {
                "entity": "StrategyInstance",
                "strategy_instance_id": item["strategy_instance_id"],
                "bind": binding,
                "value": request[route_request],
                "resulting_state": "BOUND" if item[other] else "DRAFT",
            }
            and item["lifecycle_state"] == "DRAFT"
            and item[binding] is None
        )
    if operation == "VALIDATE_ROUTE_READINESS":
        return transition == {
            "validated": True,
            "strategy_instance_id": request["strategy_instance_id"],
        }
    if operation in {
        "ACTIVATE_STRATEGY_INSTANCE",
        "DEACTIVATE_STRATEGY_INSTANCE",
        "RETIRE_STRATEGY_INSTANCE",
    }:
        item = context["strategy_instances_by_id"][request["strategy_instance_id"]]
        target = {
            "ACTIVATE_STRATEGY_INSTANCE": "ACTIVE",
            "DEACTIVATE_STRATEGY_INSTANCE": "INACTIVE",
            "RETIRE_STRATEGY_INSTANCE": "RETIRED",
        }[operation]
        return transition == {
            "entity": "StrategyInstance",
            "strategy_instance_id": request["strategy_instance_id"],
            "from": item["lifecycle_state"],
            "to": target,
        }
    return False


def execute_validator_call_graph(operation, request, context):
    try:
        if operation not in REQUESTS:
            return deny(operation, "UNKNOWN_OPERATION")
        for name in CONTRACT["operation_validator_call_graph"][operation]:
            result = VALIDATORS[name](request, context, operation)
            if result is not None:
                if (
                    not result["allowed"]
                    and result["denial_code"]
                    not in CONTRACT["allowed_denials_by_operation"][operation]
                ):
                    return deny(operation, "CONTRACT_INCONSISTENT")
                if result["allowed"] and not planned_outcome_is_valid(
                    operation, request, context, result
                ):
                    return deny(operation, "CONTRACT_INCONSISTENT")
                return result
    except (AssertionError, KeyError, StopIteration, TypeError, IndexError, ValueError):
        return deny(operation, "CONTRACT_INCONSISTENT")
    return deny(operation, "CONTRACT_INCONSISTENT")


def dispatcher(operation, request, context):
    return execute_validator_call_graph(operation, request, context)


def run_direct_call_graph(operation, request, context):
    return execute_validator_call_graph(operation, request, context)


def apply_planned_transition_and_validate(operation, context, result):
    updated = copy.deepcopy(context)
    transition = copy.deepcopy(dict(result["planned_transition"]))
    if operation == "CREATE_STRATEGY_INSTANCE":
        transition.pop("entity")
        updated["strategy_instances_by_id"][transition["strategy_instance_id"]] = transition
    elif operation in {"BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"}:
        instance = updated["strategy_instances_by_id"][transition["strategy_instance_id"]]
        instance[transition["bind"]] = transition["value"]
        instance["lifecycle_state"] = transition["resulting_state"]
    else:
        raise ValueError(operation)
    assert validate_context({}, updated, operation) is None
    return updated


def fixture_context(environment="TESTNET", instance_state="BOUND"):
    if environment == "LIVE":
        raise ValueError("M0.5 has no ENABLED LIVE Exchange Registry entry")
    exchange = exchange_entry_for_environment(environment)
    exchange_id = exchange["exchange_id"]
    adapter_family_id = exchange["adapter_family_id"]
    config = {"lookback": 20, "enabled": True}
    now = "2026-01-01T00:00:00Z"
    h = "a" * 64
    definition = {
        "strategy_definition_id": IDS["sdef"],
        "workspace_id": IDS["ws"],
        "strategy_type_id": "momentum-v1",
        "definition_version": 1,
        "configuration": config,
        "canonical_content_hash": definition_hash(config),
        "hash_domain_separator": CONTRACT["strategy_definition_contract"]["hash"][
            "domain_separator"
        ],
        "lifecycle_state": "ACTIVE",
    }
    account = {
        "exchange_account_id": IDS["xacc"],
        "workspace_id": IDS["ws"],
        "portfolio_id": IDS["port"],
        "exchange_id": exchange_id,
        "environment": environment,
        "market_type": "SPOT",
        "lifecycle_state": "ACTIVE",
        "connection_state": "ONLINE",
        "execution_authorization": "ORDER_ENTRY_ALLOWED",
        "external_account_identity_state": "VERIFIED",
        "account_capability_snapshot_id": IDS["capsnap"],
        "active_credential_profile_id": IDS["cred"] if environment == "TESTNET" else None,
    }
    instrument = {
        "instrument_id": IDS["instr"],
        "workspace_id": IDS["ws"],
        "catalog_snapshot_id": IDS["icat"],
        "exchange_id": exchange_id,
        "environment": environment,
        "market_type": "SPOT",
        "instrument_type": "SPOT_PAIR",
        "venue_symbol": "BTCUSDT",
        "display_symbol": "BTC/USDT",
        "base_asset_reference": {
            "asset_namespace": exchange_id,
            "venue_asset_code": "BTC",
            "canonical_display_code": "BTC",
            "mapping_status": "EXACT",
        },
        "quote_asset_reference": {
            "asset_namespace": exchange_id,
            "venue_asset_code": "USDT",
            "canonical_display_code": "USDT",
            "mapping_status": "EXACT",
        },
        "settlement_asset_reference": None,
        "source_adapter_family_id": adapter_family_id,
        "trading_status": "TRADING",
        "price_tick": "0.01",
        "quantity_step": "0.0001",
        "min_quantity": "0.0001",
        "max_quantity": "100",
        "min_notional": "5",
        "max_notional": None,
        "contract_size": None,
        "contract_value_currency": None,
        "derivative_settlement_type": None,
        "expiry_at_utc": None,
        "strike_price": None,
        "option_side": None,
        "metadata_version": 1,
        "observed_at_utc": "2025-12-31T23:59:00Z",
        "effective_at_utc": "2025-12-31T23:59:30Z",
        "stale_after_utc": "2026-01-01T00:05:00Z",
    }
    market_endpoint = {
        "PAPER": "PAPER_PUBLIC_DATA",
        "TESTNET": "TESTNET_PUBLIC_DATA",
        "LIVE": "LIVE_PUBLIC_DATA",
    }[environment]
    execution_endpoint = {
        "PAPER": "PAPER_SIMULATION",
        "TESTNET": "TESTNET_PRIVATE_DATA",
        "LIVE": "LIVE_PRIVATE_DATA",
    }[environment]
    market = {
        "market_data_route_id": IDS["mdr"],
        "workspace_id": IDS["ws"],
        "exchange_id": exchange_id,
        "environment": environment,
        "market_type": "SPOT",
        "adapter_family_id": adapter_family_id,
        "endpoint_class": market_endpoint,
        "data_scope": "PUBLIC",
        "instrument_ids": [IDS["instr"]],
        "channel_types": ["TRADES", "ORDER_BOOK"],
        "snapshot_stream_semantics": "SNAPSHOT_THEN_STREAM",
        "sequence_policy": "MONOTONIC_NO_GAPS",
        "freshness_policy": {"max_age_seconds": 30},
        "reconnect_policy": "RESNAPSHOT",
        "route_status": "ACTIVE",
    }
    execution = {
        "execution_route_id": IDS["xroute"],
        "workspace_id": IDS["ws"],
        "exchange_account_id": IDS["xacc"],
        "exchange_id": exchange_id,
        "environment": environment,
        "market_type": "SPOT",
        "adapter_family_id": adapter_family_id,
        "endpoint_class": execution_endpoint,
        "supported_instrument_types": ["SPOT_PAIR"],
        "route_status": "ACTIVE",
        "route_capability_ceiling": ["PLACE_ORDERS"],
        "authorization_dependencies": CONTRACT["authorization_dependencies_by_environment"][
            environment
        ],
    }
    instance = {
        "strategy_instance_id": IDS["sinst"],
        "workspace_id": IDS["ws"],
        "portfolio_id": IDS["port"],
        "strategy_definition_id": IDS["sdef"],
        "strategy_definition_version": 1,
        "exchange_account_id": IDS["xacc"],
        "trading_universe_id": IDS["univ"],
        "market_data_route_id": IDS["mdr"],
        "execution_route_id": IDS["xroute"],
        "lifecycle_state": instance_state,
    }
    credential = {
        "credential_profile_id": IDS["cred"],
        "exchange_account_id": IDS["xacc"],
        "exchange_id": exchange_id,
        "environment_scope": "TESTNET",
        "lifecycle_state": "ACTIVE",
        "credential_purpose": "ORDER_ENTRY",
        "secure_store_reference": "secure-store://testnet/order-entry",
        "public_key_identifier": None,
        "saas_sync_candidate": False,
        "permission_snapshot": ["READ_ACCOUNT", "PLACE_ORDERS"],
        "created_at_utc": "2025-12-31T23:58:00Z",
        "rotated_from_credential_profile_id": None,
        "retired_at_utc": None,
    }
    context = {
        "strategy_definitions_by_id": {IDS["sdef"]: definition},
        "previous_strategy_definitions_by_version_key": {},
        "strategy_instances_by_id": {IDS["sinst"]: instance},
        "market_data_routes_by_id": {IDS["mdr"]: market},
        "execution_routes_by_id": {IDS["xroute"]: execution},
        "accounts_by_id": {IDS["xacc"]: account},
        "universes_by_id": {
            IDS["univ"]: {
                "trading_universe_id": IDS["univ"],
                "exchange_account_id": IDS["xacc"],
                "instrument_ids": [IDS["instr"]],
                "source_catalog_snapshot_ids": [IDS["icat"]],
                "lifecycle_state": "ACTIVE",
                "version": 1,
                "created_at_utc": "2025-12-31T23:58:00Z",
                "activated_at_utc": "2025-12-31T23:59:00Z",
                "retired_at_utc": None,
                "previous_version_id": None,
                "content_hash": h,
                "creation_reason": "INITIAL_SELECTION",
            }
        },
        "instruments_by_id": {IDS["instr"]: instrument},
        "instrument_history_by_id": {},
        "catalogs_by_id": {
            IDS["icat"]: {
                "catalog_snapshot_id": IDS["icat"],
                "exchange_id": exchange_id,
                "environment": environment,
                "market_type": "SPOT",
                "adapter_family_id": adapter_family_id,
                "instrument_ids": [IDS["instr"]],
                "observed_at_utc": "2025-12-31T23:59:00Z",
                "effective_at_utc": "2025-12-31T23:59:30Z",
                "status": "VALID",
                "stale_after_utc": "2026-01-01T00:05:00Z",
                "adapter_version": "1.0.0",
                "content_hash": h,
                "previous_snapshot_id": None,
            }
        },
        "account_capability_snapshots_by_id": {
            IDS["capsnap"]: {
                "account_capability_snapshot_id": IDS["capsnap"],
                "exchange_account_id": IDS["xacc"],
                "exchange_id": exchange_id,
                "environment": environment,
                "market_type": "SPOT",
                "status": "VALID",
                "observed_permission_set": ["READ_ACCOUNT", "PLACE_ORDERS"],
                "supported_instrument_types": ["SPOT_PAIR"],
                "version": 1,
                "previous_snapshot_id": None,
                "observed_at_utc": "2025-12-31T23:59:00Z",
                "effective_at_utc": "2025-12-31T23:59:30Z",
                "stale_after_utc": "2026-01-01T00:05:00Z",
                "adapter_family_id": adapter_family_id,
                "adapter_version": "1.0.0",
                "content_hash": h,
            }
        },
        "credential_profiles_by_id": {IDS["cred"]: credential} if environment == "TESTNET" else {},
        "product_capabilities_by_environment": {
            environment: {
                "environment": environment,
                "execution_enabled": environment != "LIVE",
                "allowed_operations": [
                    {
                        "PAPER": "PAPER_LOCAL_SIMULATION",
                        "TESTNET": "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
                        "LIVE": "LIVE_VISIBLE_LOCKED_ONLY",
                    }[environment]
                ],
                "edition": "CURRENT",
            }
        },
        "route_readiness_by_id": {
            IDS["mdr"]: {
                "route_id": IDS["mdr"],
                "route_kind": "MARKET_DATA",
                "readiness_state": "READY",
                "observed_at": now,
                "metadata_version": 1,
                "sequence_state": "CONTIGUOUS",
            },
            IDS["xroute"]: {
                "route_id": IDS["xroute"],
                "route_kind": "EXECUTION",
                "readiness_state": "READY",
                "observed_at": now,
                "metadata_version": 1,
                "sequence_state": "NOT_APPLICABLE",
            },
        },
        "portfolios_by_id": {IDS["port"]: {"portfolio_id": IDS["port"], "workspace_id": IDS["ws"]}},
        "validation_time_utc": now,
        "workspace_ids": [IDS["ws"]],
        "active_strategy_instances": [IDS["sinst"]] if instance_state == "ACTIVE" else [],
        "previous_account_capability_snapshots_by_id": {},
        "previous_catalogs_by_id": {},
        "previous_universes_by_id": {},
        "previous_credential_profiles_by_id": {},
        "external_identity_snapshots_by_account_id": {
            IDS["xacc"]: {
                "state": "VERIFIED",
                "exchange_id": exchange_id,
                "environment": environment,
                "market_type": "SPOT",
                "venue_account_identifier": "venue-account-1",
                "subaccount_identifier": None,
                "account_type": "SPOT",
                "observed_permission_set": ["READ_ACCOUNT", "PLACE_ORDERS"],
                "verification_timestamp": now,
                "adapter_version_source": f"{adapter_family_id}/1.0.0",
            }
        },
    }
    context["universes_by_id"][IDS["univ"]]["content_hash"] = canonical_m05_hash(
        "trading_universe", context["universes_by_id"][IDS["univ"]]
    )
    context["catalogs_by_id"][IDS["icat"]]["content_hash"] = canonical_m05_hash(
        "instrument_catalog_snapshot", context["catalogs_by_id"][IDS["icat"]]
    )
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["content_hash"] = (
        canonical_m05_hash(
            "account_capability_snapshot",
            context["account_capability_snapshots_by_id"][IDS["capsnap"]],
        )
    )
    return context


def request_for(operation):
    common = {"operation": operation, "request_id": IDS["req"], "authority": "CoreHost"}
    extras = {
        "CREATE_STRATEGY_DEFINITION": {
            "intent": "PLAN_CREATE",
            "strategy_definition_id": IDS["sdef"],
            "workspace_id": IDS["ws"],
            "strategy_type_id": "momentum-v1",
            "definition_version": 2,
            "configuration": {"lookback": 30, "enabled": True},
        },
        "ACTIVATE_STRATEGY_DEFINITION": {
            "intent": "PLAN_ACTIVATE",
            "strategy_definition_id": IDS["sdef"],
            "definition_version": 1,
        },
        "RETIRE_STRATEGY_DEFINITION": {
            "intent": "PLAN_RETIRE",
            "strategy_definition_id": IDS["sdef"],
            "definition_version": 1,
        },
        "CREATE_STRATEGY_INSTANCE": {
            "intent": "PLAN_CREATE",
            "strategy_instance_id": f"sinst_{UUID7}0e",
            "workspace_id": IDS["ws"],
            "portfolio_id": IDS["port"],
            "strategy_definition_id": IDS["sdef"],
            "strategy_definition_version": 1,
            "exchange_account_id": IDS["xacc"],
            "trading_universe_id": IDS["univ"],
            "market_data_route_id": None,
            "execution_route_id": None,
            "lifecycle_state": "DRAFT",
        },
        "BIND_MARKET_DATA_ROUTE": {
            "intent": "PLAN_FIRST_BIND",
            "strategy_instance_id": IDS["sinst"],
            "market_data_route_id": IDS["mdr"],
        },
        "BIND_EXECUTION_ROUTE": {
            "intent": "PLAN_FIRST_BIND",
            "strategy_instance_id": IDS["sinst"],
            "execution_route_id": IDS["xroute"],
        },
        "VALIDATE_ROUTE_READINESS": {
            "intent": "VALIDATE_ONLY",
            "strategy_instance_id": IDS["sinst"],
        },
        "ACTIVATE_STRATEGY_INSTANCE": {"intent": "ACTIVATE", "strategy_instance_id": IDS["sinst"]},
        "DEACTIVATE_STRATEGY_INSTANCE": {
            "intent": "PLAN_DEACTIVATE",
            "strategy_instance_id": IDS["sinst"],
        },
        "RETIRE_STRATEGY_INSTANCE": {"intent": "PLAN_RETIRE", "strategy_instance_id": IDS["sinst"]},
    }
    return {**common, **extras[operation]}


def make_reachability_case(operation, denial):
    context = fixture_context(
        instance_state="ACTIVE" if operation == "DEACTIVATE_STRATEGY_INSTANCE" else "BOUND"
    )
    request = request_for(operation)
    if operation in {"BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"}:
        candidate_instance = context["strategy_instances_by_id"][IDS["sinst"]]
        candidate_instance["lifecycle_state"] = "DRAFT"
        candidate_instance[
            "market_data_route_id"
            if operation == "BIND_MARKET_DATA_ROUTE"
            else "execution_route_id"
        ] = None
    if operation == "ACTIVATE_STRATEGY_DEFINITION":
        context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "DRAFT"
        context["strategy_instances_by_id"].clear()
    if denial == "REQUEST_SCHEMA_INVALID":
        request["extra"] = True
    elif denial == "TRUSTED_CONTEXT_INVALID":
        context["catalogs_by_id"][IDS["icat"]]["extra"] = True
    elif denial == "WORKSPACE_NOT_FOUND":
        request["workspace_id"] = f"ws_{UUID7}0f"
    elif denial.endswith("NOT_FOUND"):
        field = {
            "STRATEGY_DEFINITION_NOT_FOUND": "strategy_definition_id",
            "STRATEGY_INSTANCE_NOT_FOUND": "strategy_instance_id",
            "MARKET_DATA_ROUTE_NOT_FOUND": "market_data_route_id",
            "EXECUTION_ROUTE_NOT_FOUND": "execution_route_id",
        }[denial]
        request[field] = f"{CONTRACT['id_prefix_registry'][field]}_{UUID7}0f"
    elif denial == "STRATEGY_DEFINITION_ID_COLLISION":
        request["definition_version"] = 1
    elif denial == "STRATEGY_INSTANCE_ID_COLLISION":
        request["strategy_instance_id"] = IDS["sinst"]
    elif denial == "STRATEGY_DEFINITION_VERSION_MISMATCH":
        request[
            "definition_version"
            if "definition_version" in request
            else "strategy_definition_version"
        ] = 99
    elif denial == "RETIRED_RESOURCE_FORBIDDEN":
        if operation in {"ACTIVATE_STRATEGY_DEFINITION", "RETIRE_STRATEGY_DEFINITION"}:
            context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "RETIRED"
        elif operation in {
            "ACTIVATE_STRATEGY_INSTANCE",
            "DEACTIVATE_STRATEGY_INSTANCE",
            "RETIRE_STRATEGY_INSTANCE",
            "VALIDATE_ROUTE_READINESS",
        } or operation.startswith("BIND_"):
            context["strategy_instances_by_id"][IDS["sinst"]]["lifecycle_state"] = "RETIRED"
            context["active_strategy_instances"] = []
            if operation.startswith("BIND_"):
                context["strategy_instances_by_id"][IDS["sinst"]][
                    "market_data_route_id"
                    if operation == "BIND_MARKET_DATA_ROUTE"
                    else "execution_route_id"
                ] = None
        else:
            context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "RETIRED"
    elif denial == "STRATEGY_INSTANCE_BINDING_MISMATCH":
        if operation == "CREATE_STRATEGY_INSTANCE":
            request["market_data_route_id"] = IDS["mdr"]
        elif operation == "ACTIVATE_STRATEGY_INSTANCE":
            context["strategy_instances_by_id"][IDS["sinst"]]["lifecycle_state"] = "DRAFT"
            context["strategy_instances_by_id"][IDS["sinst"]]["execution_route_id"] = None
        elif operation == "VALIDATE_ROUTE_READINESS":
            context["strategy_instances_by_id"][IDS["sinst"]].update(
                lifecycle_state="DRAFT", market_data_route_id=None, execution_route_id=None
            )
        elif operation == "DEACTIVATE_STRATEGY_INSTANCE":
            context["strategy_instances_by_id"][IDS["sinst"]]["lifecycle_state"] = "INACTIVE"
            context["active_strategy_instances"] = []
        elif operation == "RETIRE_STRATEGY_INSTANCE":
            context["strategy_instances_by_id"][IDS["sinst"]]["lifecycle_state"] = "ACTIVE"
            context["active_strategy_instances"] = [IDS["sinst"]]
        elif operation.startswith("BIND_"):
            instance = context["strategy_instances_by_id"][IDS["sinst"]]
            instance[
                "market_data_route_id"
                if operation == "BIND_MARKET_DATA_ROUTE"
                else "execution_route_id"
            ] = IDS["mdr"] if operation == "BIND_MARKET_DATA_ROUTE" else IDS["xroute"]
            instance["lifecycle_state"] = "BOUND"
    elif denial == "STRATEGY_DEFINITION_STATE_CONFLICT":
        definition = context["strategy_definitions_by_id"][IDS["sdef"]]
        if operation == "ACTIVATE_STRATEGY_DEFINITION":
            definition["lifecycle_state"] = "ACTIVE"
        elif operation == "RETIRE_STRATEGY_DEFINITION":
            definition["lifecycle_state"] = "DRAFT"
            context["strategy_instances_by_id"].clear()
        else:
            definition["lifecycle_state"] = "DRAFT"
            context["strategy_instances_by_id"].clear()
    elif denial == "TRADING_UNIVERSE_INVALID":
        context["universes_by_id"][IDS["univ"]]["lifecycle_state"] = "DRAFT"
    elif denial == "ROUTE_SCOPE_MISMATCH":
        route = (
            context["market_data_routes_by_id"][IDS["mdr"]]
            if operation == "BIND_MARKET_DATA_ROUTE"
            else context["execution_routes_by_id"][IDS["xroute"]]
        )
        if operation == "BIND_MARKET_DATA_ROUTE":
            workspace_id, iid, cid = f"ws_{UUID7}0f", f"instr_{UUID7}0f", f"icat_{UUID7}0f"
            instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
            catalog = copy.deepcopy(context["catalogs_by_id"][IDS["icat"]])
            instrument.update(instrument_id=iid, workspace_id=workspace_id, catalog_snapshot_id=cid)
            catalog.update(catalog_snapshot_id=cid, instrument_ids=[iid])
            context["instruments_by_id"][iid] = instrument
            context["catalogs_by_id"][cid] = catalog
            context["workspace_ids"].append(workspace_id)
            route.update(workspace_id=workspace_id, instrument_ids=[iid])
        else:
            account = copy.deepcopy(context["accounts_by_id"][IDS["xacc"]])
            account_id, account_capability_snapshot_id = f"xacc_{UUID7}0f", f"capsnap_{UUID7}0f"
            account.update(
                exchange_account_id=account_id,
                account_capability_snapshot_id=account_capability_snapshot_id,
                active_credential_profile_id=f"cred_{UUID7}0f",
            )
            snapshot = copy.deepcopy(context["account_capability_snapshots_by_id"][IDS["capsnap"]])
            snapshot.update(
                account_capability_snapshot_id=account_capability_snapshot_id,
                exchange_account_id=account_id,
            )
            credential = copy.deepcopy(context["credential_profiles_by_id"][IDS["cred"]])
            credential.update(
                credential_profile_id=f"cred_{UUID7}0f", exchange_account_id=account_id
            )
            context["accounts_by_id"][account_id] = account
            context["account_capability_snapshots_by_id"][account_capability_snapshot_id] = snapshot
            context["credential_profiles_by_id"][credential["credential_profile_id"]] = credential
            route.update(exchange_account_id=account_id)
    elif denial == "INSTRUMENT_SCOPE_MISMATCH":
        if operation == "BIND_MARKET_DATA_ROUTE":
            iid = f"instr_{UUID7}0f"
            instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
            instrument["instrument_id"] = iid
            context["instruments_by_id"][iid] = instrument
            context["catalogs_by_id"][IDS["icat"]]["instrument_ids"].append(iid)
            context["market_data_routes_by_id"][IDS["mdr"]]["instrument_ids"] = [iid]
        elif operation == "VALIDATE_ROUTE_READINESS":
            iid = f"instr_{UUID7}0f"
            instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
            instrument["instrument_id"] = iid
            context["instruments_by_id"][iid] = instrument
            context["catalogs_by_id"][IDS["icat"]]["instrument_ids"].append(iid)
            context["market_data_routes_by_id"][IDS["mdr"]]["instrument_ids"] = [iid]
        else:
            context["instruments_by_id"][IDS["instr"]]["trading_status"] = "HALTED"
    elif denial == "ROUTE_CAPABILITY_BLOCKED":
        context["execution_routes_by_id"][IDS["xroute"]]["authorization_dependencies"] = [
            "PRODUCT_CAPABILITIES"
        ]
    elif denial in {"MARKET_DATA_ROUTE_NOT_READY", "EXECUTION_ROUTE_NOT_READY"}:
        context["route_readiness_by_id"].pop(
            IDS["mdr"] if denial.startswith("MARKET") else IDS["xroute"]
        )
    elif denial == "MARKET_DATA_FRESHNESS_BLOCKED":
        context["route_readiness_by_id"][IDS["mdr"]]["observed_at"] = "2025-12-31T23:00:00Z"
    elif denial == "MARKET_DATA_SEQUENCE_INVALID":
        context["route_readiness_by_id"][IDS["mdr"]]["sequence_state"] = "GAP"
    elif denial == "ACCOUNT_READINESS_BLOCKED":
        context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    elif denial == "CAPABILITY_SNAPSHOT_BLOCKED":
        context["account_capability_snapshots_by_id"][IDS["capsnap"]]["status"] = "REJECTED"
    elif denial == "PRODUCT_CAPABILITIES_BLOCKED":
        context["product_capabilities_by_environment"]["TESTNET"]["execution_enabled"] = False
    if denial != "TRUSTED_CONTEXT_INVALID":
        rehash_m05_projections(context)
    return request, context


RUNTIME_PAIRS = [
    (operation, denial)
    for operation, values in CONTRACT["allowed_denials_by_operation"].items()
    for denial in values
    if denial != "CONTRACT_INCONSISTENT"
]


def test_machine_schemas_are_exact_and_complete():
    assert set(SCHEMAS) >= {
        "StrategyDefinition",
        "StrategyInstance",
        "MarketDataRoute",
        "ExecutionRoute",
        "RouteReadiness",
        "ExchangeAccountProjection",
        "TradingUniverseProjection",
        "InstrumentProjection",
        "InstrumentCatalogProjection",
        "AccountCapabilitySnapshotProjection",
        "CredentialProfileProjection",
        "ProductCapabilitiesProjection",
    }
    assert all(
        {
            "exact_fields",
            "field_types",
            "id_field",
            "id_prefix",
            "enum_registry",
            "nullable_fields",
            "array_policy",
            "nested_schemas",
            "entity_references",
        }
        <= set(schema)
        for schema in SCHEMAS.values()
    )
    assert all(
        {"request_fields", "request_types", "constants", "nullable_fields", "nested_schemas"}
        <= set(schema)
        for schema in REQUESTS.values()
    )


def test_callable_graph_is_real_complete_and_exact():
    graph_names = {
        name for graph in CONTRACT["operation_validator_call_graph"].values() for name in graph
    }
    assert graph_names == set(VALIDATORS) == set(CONTRACT["validator_denial_registry"])
    for operation, graph in CONTRACT["operation_validator_call_graph"].items():
        assert graph[:2] == ["validate_request", "validate_context"]
        union = {
            code
            for name in graph
            for code in CONTRACT["validator_denial_registry"][name]["denials"]
        }
        assert union == set(CONTRACT["allowed_denials_by_operation"][operation]) - {
            "CONTRACT_INCONSISTENT"
        }
    preflight = "validate_strategy_instance_lifecycle_preflight"
    activation_graph = CONTRACT["operation_validator_call_graph"]["ACTIVATE_STRATEGY_INSTANCE"]
    deactivation_graph = CONTRACT["operation_validator_call_graph"]["DEACTIVATE_STRATEGY_INSTANCE"]
    assert activation_graph.index(preflight) < activation_graph.index(
        "validate_authorization_operability"
    )
    assert deactivation_graph.index(preflight) < deactivation_graph.index("validate_deactivation")


@pytest.mark.parametrize(("operation", "expected"), RUNTIME_PAIRS)
def test_every_declared_runtime_denial_is_reached_by_real_mutation(operation, expected):
    request, context = make_reachability_case(operation, expected)
    if expected not in {"REQUEST_SCHEMA_INVALID", "TRUSTED_CONTEXT_INVALID"}:
        assert validate_context(request, context, operation) is None
    result = dispatcher(operation, request, context)
    assert result["denial_code"] == expected
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(("operation", "expected"), RUNTIME_PAIRS)
def test_direct_validator_dispatcher_parity_for_every_runtime_pair(operation, expected):
    request, context = make_reachability_case(operation, expected)
    if expected not in {"REQUEST_SCHEMA_INVALID", "TRUSTED_CONTEXT_INVALID"}:
        assert validate_context(request, context, operation) is None
    direct = run_direct_call_graph(operation, request, context)
    dispatched = dispatcher(operation, request, context)
    assert {
        key: direct[key]
        for key in ("allowed", "denial_code", "audit_event_type", "planned_transition")
    } == {
        key: dispatched[key]
        for key in ("allowed", "denial_code", "audit_event_type", "planned_transition")
    }


def test_positive_create_activate_versioning_and_instance_paths():
    context = fixture_context()
    assert dispatcher(
        "CREATE_STRATEGY_DEFINITION", request_for("CREATE_STRATEGY_DEFINITION"), context
    )["allowed"]
    context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "DRAFT"
    context["strategy_instances_by_id"].clear()
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_DEFINITION", request_for("ACTIVATE_STRATEGY_DEFINITION"), context
        )["planned_transition"]["to"]
        == "ACTIVE"
    )
    context = fixture_context()
    v1 = context["strategy_definitions_by_id"][IDS["sdef"]]
    v2 = copy.deepcopy(v1)
    v2["definition_version"] = 2
    context["strategy_definitions_by_id"][IDS["sdef"]] = v2
    context["previous_strategy_definitions_by_version_key"][f"{IDS['sdef']}@1"] = v1
    assert resolve_strategy_definition_exact_version(context, IDS["sdef"], 1) is v1
    assert context["strategy_instances_by_id"][IDS["sinst"]]["strategy_definition_version"] == 1
    create = request_for("CREATE_STRATEGY_INSTANCE")
    create["strategy_definition_version"] = 2
    assert dispatcher("CREATE_STRATEGY_INSTANCE", create, context)["allowed"]


def test_positive_first_binds_bound_readiness_paper_testnet_and_transitions():
    context = fixture_context()
    instance = context["strategy_instances_by_id"][IDS["sinst"]]
    instance["lifecycle_state"] = "DRAFT"
    instance["market_data_route_id"] = None
    instance["execution_route_id"] = None
    market = dispatcher("BIND_MARKET_DATA_ROUTE", request_for("BIND_MARKET_DATA_ROUTE"), context)
    assert market["allowed"]
    instance["market_data_route_id"] = IDS["mdr"]
    execution = dispatcher("BIND_EXECUTION_ROUTE", request_for("BIND_EXECUTION_ROUTE"), context)
    assert execution["allowed"] and execution["planned_transition"]["resulting_state"] == "BOUND"
    context = fixture_context()
    assert dispatcher("VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context)[
        "allowed"
    ]
    assert dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )["allowed"]
    paper = fixture_context("PAPER")
    assert dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), paper
    )["allowed"]
    active = fixture_context(instance_state="ACTIVE")
    assert dispatcher(
        "DEACTIVATE_STRATEGY_INSTANCE", request_for("DEACTIVATE_STRATEGY_INSTANCE"), active
    )["allowed"]
    inactive = fixture_context(instance_state="INACTIVE")
    assert dispatcher(
        "RETIRE_STRATEGY_INSTANCE", request_for("RETIRE_STRATEGY_INSTANCE"), inactive
    )["allowed"]


def test_dispatcher_is_immutable_and_does_not_mutate_inputs():
    request = request_for("ACTIVATE_STRATEGY_INSTANCE")
    context = fixture_context()
    before = (copy.deepcopy(request), copy.deepcopy(context))
    result = dispatcher("ACTIVATE_STRATEGY_INSTANCE", request, context)
    assert isinstance(result, MappingProxyType) and isinstance(
        result["planned_transition"], MappingProxyType
    )
    assert (request, context) == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["catalogs_by_id"][IDS["icat"]].update(extra=True),
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(extra=True),
    ],
)
def test_malformed_unrelated_record_has_precedence(mutation):
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    mutation(context)
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["strategy_instances_by_id"][IDS["sinst"]].update(
            strategy_definition_id=f"sdef_{UUID7}0f"
        ),
        lambda c: c["strategy_instances_by_id"][IDS["sinst"]].update(
            exchange_account_id=f"xacc_{UUID7}0f"
        ),
        lambda c: c["strategy_instances_by_id"][IDS["sinst"]].update(
            market_data_route_id=f"mdr_{UUID7}0f"
        ),
    ],
)
def test_persisted_dangling_references_are_context_invalid(mutation):
    context = fixture_context()
    mutation(context)
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


def test_exact_request_counterexamples_are_executed():
    for mutation in [
        lambda r: r.update(extra=True),
        lambda r: r.update(definition_version=True),
        lambda r: r.update(intent="UNKNOWN"),
        lambda r: r.update(strategy_definition_id=f"xroute_{UUID7}0f"),
    ]:
        request = request_for("CREATE_STRATEGY_DEFINITION")
        mutation(request)
        assert (
            dispatcher("CREATE_STRATEGY_DEFINITION", request, fixture_context())["denial_code"]
            == "REQUEST_SCHEMA_INVALID"
        )
    assert dispatcher("SUBMIT_ORDER", {}, fixture_context())["denial_code"] == "UNKNOWN_OPERATION"


def test_hash_lineage_adapter_snapshot_credential_and_readiness_mutations_are_real():
    mutations = []
    mutations.append(
        lambda c: c["strategy_definitions_by_id"][IDS["sdef"]].update(
            canonical_content_hash="0" * 64
        )
    )
    mutations.append(
        lambda c: c["instruments_by_id"][IDS["instr"]].update(source_adapter_family_id="other-v1")
    )
    mutations.append(
        lambda c: c["execution_routes_by_id"][IDS["xroute"]].update(adapter_family_id="other-v1")
    )
    mutations.append(
        lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
            exchange_id="KRAKEN"
        )
    )
    mutations.append(
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(environment_scope="LIVE")
    )
    mutations.append(
        lambda c: c["route_readiness_by_id"][IDS["mdr"]].update(route_kind="EXECUTION")
    )
    mutations.append(
        lambda c: c["route_readiness_by_id"][IDS["mdr"]].update(observed_at="2026-01-01T00:01:00Z")
    )
    for mutation in mutations:
        context = fixture_context()
        mutation(context)
        assert (
            dispatcher(
                "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
            )["denial_code"]
            == "TRUSTED_CONTEXT_INVALID"
        )


def test_activation_credential_permission_and_retirement_mutations():
    retired = fixture_context()
    retired["credential_profiles_by_id"][IDS["cred"]]["lifecycle_state"] = "RETIRED"
    assert validate_context({}, retired, "ACTIVATE_STRATEGY_INSTANCE")["denial_code"] == (
        "TRUSTED_CONTEXT_INVALID"
    )
    for mutation in [
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(
            permission_snapshot=["READ_ACCOUNT"]
        ),
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(
            credential_purpose="ACCOUNT_READ"
        ),
    ]:
        context = fixture_context()
        mutation(context)
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )
        assert result["denial_code"] == "ACCOUNT_READINESS_BLOCKED"


@pytest.mark.parametrize("operation", ["BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"])
@pytest.mark.parametrize("state", ["BOUND", "ACTIVE", "INACTIVE", "RETIRED"])
def test_bind_is_exclusive_to_draft(operation, state):
    context = fixture_context(instance_state=state)
    item = context["strategy_instances_by_id"][IDS["sinst"]]
    if state == "RETIRED":
        item[
            "market_data_route_id"
            if operation == "BIND_MARKET_DATA_ROUTE"
            else "execution_route_id"
        ] = None
    context["active_strategy_instances"] = [IDS["sinst"]] if state == "ACTIVE" else []
    result = dispatcher(operation, request_for(operation), context)
    expected = (
        "RETIRED_RESOURCE_FORBIDDEN" if state == "RETIRED" else "STRATEGY_INSTANCE_BINDING_MISMATCH"
    )
    assert result["denial_code"] == expected


def test_definition_workspace_and_exact_domain_separator_are_enforced():
    request = request_for("CREATE_STRATEGY_DEFINITION")
    request["workspace_id"] = f"ws_{UUID7}0f"
    assert (
        dispatcher("CREATE_STRATEGY_DEFINITION", request, fixture_context())["denial_code"]
        == "WORKSPACE_NOT_FOUND"
    )
    for history_too in (False, True):
        context = fixture_context()
        current = context["strategy_definitions_by_id"][IDS["sdef"]]
        current["hash_domain_separator"] = "foreign-domain"
        if history_too:
            prior = copy.deepcopy(current)
            current["definition_version"] = 2
            prior["definition_version"] = 1
            context["previous_strategy_definitions_by_version_key"][f"{IDS['sdef']}@1"] = prior
        assert (
            validate_context({}, context, "CREATE_STRATEGY_DEFINITION")["denial_code"]
            == "TRUSTED_CONTEXT_INVALID"
        )


@pytest.mark.parametrize("field", ["exchange_id", "environment", "market_type"])
def test_catalog_scope_is_global_and_exact(field):
    context = fixture_context()
    context["catalogs_by_id"][IDS["icat"]][field] = "WRONG"
    assert (
        validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


def test_product_capabilities_projection_is_required_without_exceptions():
    context = fixture_context()
    context["product_capabilities_by_environment"].clear()
    assert (
        validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize("operation", sorted(REQUESTS))
def test_every_operation_is_exception_safe_for_missing_persisted_projection(operation):
    context = fixture_context()
    context["product_capabilities_by_environment"].clear()
    result = dispatcher(operation, request_for(operation), context)
    assert isinstance(result, MappingProxyType)
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("schema_map", "record_id", "field", "value"),
    [
        ("market_data_routes_by_id", "mdr", "channel_types", ["UNKNOWN"]),
        ("execution_routes_by_id", "xroute", "supported_instrument_types", ["UNKNOWN"]),
        ("execution_routes_by_id", "xroute", "route_capability_ceiling", ["UNKNOWN"]),
        ("execution_routes_by_id", "xroute", "authorization_dependencies", ["UNKNOWN"]),
    ],
)
def test_array_enum_registries_reject_unknown_values(schema_map, record_id, field, value):
    context = fixture_context()
    context[schema_map][IDS[record_id]][field] = value
    assert (
        validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize(
    ("scope", "endpoint", "channels", "valid"),
    [
        ("PRIVATE", "TESTNET_PUBLIC_DATA", ["PRIVATE_ORDERS"], False),
        ("PUBLIC", "TESTNET_PRIVATE_DATA", ["TRADES"], False),
        ("PUBLIC", "TESTNET_PUBLIC_DATA", ["PRIVATE_ORDERS"], False),
        ("PRIVATE", "TESTNET_PRIVATE_DATA", ["PRIVATE_ORDERS"], True),
    ],
)
def test_market_route_endpoint_access_is_exact(scope, endpoint, channels, valid):
    context = fixture_context()
    context["market_data_routes_by_id"][IDS["mdr"]].update(
        data_scope=scope, endpoint_class=endpoint, channel_types=channels
    )
    result = validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")
    assert (result is None) is valid


@pytest.mark.parametrize(
    ("max_age", "observed", "denial"),
    [
        (5, "2025-12-31T23:59:50Z", "MARKET_DATA_FRESHNESS_BLOCKED"),
        (60, "2025-12-31T23:59:30Z", None),
        (30, "2025-12-31T23:59:30Z", None),
    ],
)
def test_market_freshness_uses_route_policy_and_inclusive_boundary(max_age, observed, denial):
    context = fixture_context()
    context["market_data_routes_by_id"][IDS["mdr"]]["freshness_policy"]["max_age_seconds"] = max_age
    context["route_readiness_by_id"][IDS["mdr"]]["observed_at"] = observed
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == denial


@pytest.mark.parametrize("sequence", ["GAP", "CONTIGUOUS"])
def test_execution_readiness_requires_not_applicable_sequence(sequence):
    context = fixture_context()
    context["route_readiness_by_id"][IDS["xroute"]]["sequence_state"] = sequence
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == "EXECUTION_ROUTE_NOT_READY"


def test_reverse_integrity_covers_unrelated_records():
    mutations = []

    def missing_catalog(context):
        instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
        instrument.update(instrument_id=f"instr_{UUID7}0f", catalog_snapshot_id=f"icat_{UUID7}0f")
        context["instruments_by_id"][instrument["instrument_id"]] = instrument

    def missing_back_reference(context):
        instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
        instrument["instrument_id"] = f"instr_{UUID7}0f"
        context["instruments_by_id"][instrument["instrument_id"]] = instrument

    def orphan_snapshot(context):
        snapshot = copy.deepcopy(context["account_capability_snapshots_by_id"][IDS["capsnap"]])
        snapshot.update(
            account_capability_snapshot_id=f"capsnap_{UUID7}0f",
            exchange_account_id=f"xacc_{UUID7}0f",
        )
        context["account_capability_snapshots_by_id"][
            snapshot["account_capability_snapshot_id"]
        ] = snapshot

    def orphan_portfolio(context):
        context["portfolios_by_id"][f"port_{UUID7}0f"] = {
            "portfolio_id": f"port_{UUID7}0f",
            "workspace_id": f"ws_{UUID7}0f",
        }

    mutations.extend([missing_catalog, missing_back_reference, orphan_snapshot, orphan_portfolio])
    for mutation in mutations:
        context = fixture_context()
        mutation(context)
        result = validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")
        assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_paper_authorization_ignores_snapshot_and_credentials():
    context = fixture_context("PAPER")
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    snapshot = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    snapshot.update(
        status="REJECTED",
        observed_permission_set=["READ_ACCOUNT"],
    )
    rehash_m05_projections(context)
    assert dispatcher("VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context)[
        "allowed"
    ]
    assert dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )["allowed"]
    context["product_capabilities_by_environment"]["PAPER"]["execution_enabled"] = False
    assert (
        dispatcher("VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context)[
            "denial_code"
        ]
        == "PRODUCT_CAPABILITIES_BLOCKED"
    )


@pytest.mark.parametrize("bindings", [(), ("market",), ("execution",)])
def test_readiness_draft_incomplete_binding_is_domain_denial(bindings):
    context = fixture_context(instance_state="DRAFT")
    item = context["strategy_instances_by_id"][IDS["sinst"]]
    item["market_data_route_id"] = IDS["mdr"] if "market" in bindings else None
    item["execution_route_id"] = IDS["xroute"] if "execution" in bindings else None
    assert validate_context({}, context, "VALIDATE_ROUTE_READINESS") is None
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == "STRATEGY_INSTANCE_BINDING_MISMATCH"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_readiness_retired_is_forbidden_before_authorization_lookup():
    context = fixture_context(instance_state="RETIRED")
    context["strategy_instances_by_id"][IDS["sinst"]]["execution_route_id"] = None
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == "RETIRED_RESOURCE_FORBIDDEN"


@pytest.mark.parametrize("state", ["BOUND", "INACTIVE", "ACTIVE"])
def test_persisted_instance_cannot_reference_draft_definition(state):
    context = fixture_context(instance_state=state)
    context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "DRAFT"
    context["active_strategy_instances"] = [IDS["sinst"]] if state == "ACTIVE" else []
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")["denial_code"] == (
        "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize("state", ["BOUND", "INACTIVE"])
def test_activation_rejects_retired_exact_definition(state):
    context = fixture_context(instance_state=state)
    context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "RETIRED"
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "RETIRED_RESOURCE_FORBIDDEN"


def test_definition_state_conflicts_are_precise():
    active = fixture_context()
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_DEFINITION", request_for("ACTIVATE_STRATEGY_DEFINITION"), active
        )["denial_code"]
        == "STRATEGY_DEFINITION_STATE_CONFLICT"
    )
    draft = fixture_context()
    draft["strategy_instances_by_id"].clear()
    draft["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "DRAFT"
    assert (
        dispatcher("RETIRE_STRATEGY_DEFINITION", request_for("RETIRE_STRATEGY_DEFINITION"), draft)[
            "denial_code"
        ]
        == "STRATEGY_DEFINITION_STATE_CONFLICT"
    )
    assert (
        dispatcher("CREATE_STRATEGY_INSTANCE", request_for("CREATE_STRATEGY_INSTANCE"), draft)[
            "denial_code"
        ]
        == "STRATEGY_DEFINITION_STATE_CONFLICT"
    )


def test_create_plans_are_complete_and_deeply_immutable():
    definition_request = request_for("CREATE_STRATEGY_DEFINITION")
    definition = dispatcher("CREATE_STRATEGY_DEFINITION", definition_request, fixture_context())
    assert set(definition["planned_transition"]) == {
        "entity",
        "strategy_definition_id",
        "workspace_id",
        "strategy_type_id",
        "definition_version",
        "configuration",
        "canonical_content_hash",
        "hash_domain_separator",
        "lifecycle_state",
    }
    assert definition["planned_transition"]["lifecycle_state"] == "DRAFT"
    with pytest.raises(TypeError):
        definition["planned_transition"]["configuration"]["lookback"] = 99
    assert definition_request["configuration"]["lookback"] == 30
    instance = dispatcher(
        "CREATE_STRATEGY_INSTANCE", request_for("CREATE_STRATEGY_INSTANCE"), fixture_context()
    )
    assert set(instance["planned_transition"]) == {
        "entity",
        "strategy_instance_id",
        "workspace_id",
        "portfolio_id",
        "strategy_definition_id",
        "strategy_definition_version",
        "exchange_account_id",
        "trading_universe_id",
        "market_data_route_id",
        "execution_route_id",
        "lifecycle_state",
    }
    assert instance["planned_transition"]["lifecycle_state"] == "DRAFT"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda c: c["accounts_by_id"][IDS["xacc"]].update(lifecycle_state="DISABLED"),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
                status="REJECTED"
            ),
            "CAPABILITY_SNAPSHOT_BLOCKED",
        ),
        (
            lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
                observed_at_utc="2025-12-31T23:00:00Z"
            ),
            "CAPABILITY_SNAPSHOT_BLOCKED",
        ),
        (
            lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
                observed_permission_set=["READ_ACCOUNT"]
            ),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            lambda c: c["product_capabilities_by_environment"]["TESTNET"].update(
                execution_enabled=False
            ),
            "PRODUCT_CAPABILITIES_BLOCKED",
        ),
        (
            lambda c: c["product_capabilities_by_environment"]["TESTNET"].update(
                allowed_operations=["LIVE_VISIBLE_LOCKED_ONLY"]
            ),
            "PRODUCT_CAPABILITIES_BLOCKED",
        ),
        (
            lambda c: (
                c["accounts_by_id"][IDS["xacc"]].update(active_credential_profile_id=None),
                c["credential_profiles_by_id"].clear(),
            ),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(
                credential_purpose="ACCOUNT_READ"
            ),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(
                permission_snapshot=["READ_ACCOUNT"]
            ),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            lambda c: c["execution_routes_by_id"][IDS["xroute"]].update(
                route_capability_ceiling=["READ_MARKET_DATA"]
            ),
            "ROUTE_CAPABILITY_BLOCKED",
        ),
    ],
)
def test_testnet_authorization_operability_is_shared_by_readiness(mutation, expected):
    context = fixture_context()
    mutation(context)
    rehash_m05_projections(context)
    assert validate_context({}, context, "VALIDATE_ROUTE_READINESS") is None
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == expected
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_live_execution_is_a_cross_milestone_registry_blocker():
    assert not any(
        entry["status"] == "ENABLED" and "LIVE" in entry["supported_environments"]
        for entry in canonical_exchange_entries()
    )
    assert CONTRACT["live_exchange_registry_audit"] == {
        "live_enabled_entry_present": False,
        "ordinary_live_execution_denial_reachable": False,
        "runtime": False,
        "reachable": False,
        "cross_milestone_blocked": True,
        "cross_milestone_blocker": (
            "M0.5 closed Exchange Registry has no ENABLED entry supporting LIVE; persisted LIVE "
            "context fails TRUSTED_CONTEXT_INVALID before operation authority and "
            "LIVE_EXECUTION_FORBIDDEN is not an ordinary M0.6 reachability case"
        ),
    }


def test_special_audit_events_cover_machine_contract_faults():
    unknown = dispatcher("NOT_AN_OPERATION", {}, fixture_context())
    assert unknown["audit_event_type"] == "STRATEGY_ROUTING_UNKNOWN_OPERATION_DENIED"
    operation = "DEACTIVATE_STRATEGY_INSTANCE"
    context = fixture_context(instance_state="ACTIVE")
    original_graph = CONTRACT["operation_validator_call_graph"][operation]
    try:
        CONTRACT["operation_validator_call_graph"][operation] = ["missing_validator"]
        broken = dispatcher(operation, request_for(operation), context)
        assert broken["denial_code"] == "CONTRACT_INCONSISTENT"
        assert broken["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        CONTRACT["operation_validator_call_graph"][operation] = original_graph
    original_validator = VALIDATORS["validate_deactivation"]
    try:
        VALIDATORS["validate_deactivation"] = lambda request, context, operation: success(
            operation, {"entity": "wrong"}
        )
        broken = dispatcher(operation, request_for(operation), context)
        assert broken["denial_code"] == "CONTRACT_INCONSISTENT"
        assert broken["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        VALIDATORS["validate_deactivation"] = original_validator


def test_denial_registry_has_no_dead_ordinary_codes():
    ordinary = set(CONTRACT["denial_code_registry"]) - {
        "UNKNOWN_OPERATION",
        "CONTRACT_INCONSISTENT",
    }
    reachable = {
        denial
        for validator in CONTRACT["validator_denial_registry"].values()
        for denial in validator["denials"]
    }
    assert ordinary == reachable


@pytest.mark.parametrize(
    ("map_name", "record_id", "field", "alien"),
    [
        (
            "account_capability_snapshots_by_id",
            "capsnap",
            "observed_permission_set",
            "ALIEN_CAPABILITY",
        ),
        ("credential_profiles_by_id", "cred", "permission_snapshot", "ALIEN_PERMISSION"),
        (
            "product_capabilities_by_environment",
            "TESTNET",
            "allowed_operations",
            "ALIEN_OPERATION",
        ),
    ],
)
def test_projection_arrays_reject_values_outside_canonical_contracts(
    map_name, record_id, field, alien
):
    context = fixture_context()
    key = IDS[record_id] if record_id in IDS else record_id
    context[map_name][key][field].append(alien)
    result = validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_projection_arrays_accept_multiple_canonical_values():
    context = fixture_context()
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["observed_permission_set"].append(
        "READ_ORDERS"
    )
    context["credential_profiles_by_id"][IDS["cred"]]["permission_snapshot"].append("READ_ORDERS")
    context["product_capabilities_by_environment"]["TESTNET"]["allowed_operations"].extend(
        ["PAPER_LOCAL_SIMULATION", "LIVE_VISIBLE_LOCKED_ONLY"]
    )
    rehash_m05_projections(context)
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None


@pytest.mark.parametrize(
    "operation", ["ACTIVATE_STRATEGY_INSTANCE", "DEACTIVATE_STRATEGY_INSTANCE"]
)
def test_retired_instance_is_terminal_before_other_preflights(operation):
    context = fixture_context(instance_state="RETIRED")
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    context["product_capabilities_by_environment"]["TESTNET"]["execution_enabled"] = False
    result = dispatcher(operation, request_for(operation), context)
    assert result["denial_code"] == "RETIRED_RESOURCE_FORBIDDEN"
    assert result["audit_event_type"] == CONTRACT["denial_event_by_operation"][operation]
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_retired_definition_is_distinct_from_retired_instance_activation():
    context = fixture_context(instance_state="BOUND")
    context["strategy_definitions_by_id"][IDS["sdef"]]["lifecycle_state"] = "RETIRED"
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "RETIRED_RESOURCE_FORBIDDEN"


def test_shared_lifecycle_preflight_fails_closed_for_wrong_operation_graph():
    operation = "RETIRE_STRATEGY_INSTANCE"
    graph = CONTRACT["operation_validator_call_graph"][operation]
    original = list(graph)
    try:
        graph.insert(2, "validate_strategy_instance_lifecycle_preflight")
        result = dispatcher(operation, request_for(operation), fixture_context())
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        graph[:] = original


@pytest.mark.parametrize(
    ("operation", "state", "expected_allowed", "expected_denial"),
    [
        ("ACTIVATE_STRATEGY_INSTANCE", "DRAFT", False, "STRATEGY_INSTANCE_BINDING_MISMATCH"),
        ("ACTIVATE_STRATEGY_INSTANCE", "BOUND", True, None),
        ("ACTIVATE_STRATEGY_INSTANCE", "ACTIVE", False, "STRATEGY_INSTANCE_BINDING_MISMATCH"),
        ("ACTIVATE_STRATEGY_INSTANCE", "INACTIVE", True, None),
        ("ACTIVATE_STRATEGY_INSTANCE", "RETIRED", False, "RETIRED_RESOURCE_FORBIDDEN"),
        ("DEACTIVATE_STRATEGY_INSTANCE", "DRAFT", False, "STRATEGY_INSTANCE_BINDING_MISMATCH"),
        ("DEACTIVATE_STRATEGY_INSTANCE", "BOUND", False, "STRATEGY_INSTANCE_BINDING_MISMATCH"),
        ("DEACTIVATE_STRATEGY_INSTANCE", "ACTIVE", True, None),
        ("DEACTIVATE_STRATEGY_INSTANCE", "INACTIVE", False, "STRATEGY_INSTANCE_BINDING_MISMATCH"),
        ("DEACTIVATE_STRATEGY_INSTANCE", "RETIRED", False, "RETIRED_RESOURCE_FORBIDDEN"),
    ],
)
def test_complete_terminal_lifecycle_matrix(operation, state, expected_allowed, expected_denial):
    context = fixture_context(instance_state=state)
    if state == "DRAFT":
        context["strategy_instances_by_id"][IDS["sinst"]].update(
            market_data_route_id=None, execution_route_id=None
        )
    result = dispatcher(operation, request_for(operation), context)
    assert result["allowed"] is expected_allowed
    assert result["denial_code"] == expected_denial
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"
    expected_event = (
        CONTRACT["success_event_by_operation"][operation]
        if expected_allowed
        else CONTRACT["denial_event_by_operation"][operation]
    )
    assert result["audit_event_type"] == expected_event


@pytest.mark.parametrize("state", ["DRAFT", "ACTIVE", "RETIRED"])
def test_activation_terminal_denial_prevents_authorization_execution(state):
    context = fixture_context(instance_state=state)
    if state == "DRAFT":
        context["strategy_instances_by_id"][IDS["sinst"]].update(
            market_data_route_id=None, execution_route_id=None
        )
    original = VALIDATORS["validate_authorization_operability"]
    try:
        VALIDATORS["validate_authorization_operability"] = lambda *_args: (_ for _ in ()).throw(
            AssertionError("authorization must not run after lifecycle denial")
        )
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )
        assert result["denial_code"] in {
            "STRATEGY_INSTANCE_BINDING_MISMATCH",
            "RETIRED_RESOURCE_FORBIDDEN",
        }
    finally:
        VALIDATORS["validate_authorization_operability"] = original


def test_snapshot_structure_is_distinct_from_testnet_operability():
    rejected = fixture_context()
    rejected["account_capability_snapshots_by_id"][IDS["capsnap"]]["status"] = "REJECTED"
    rehash_m05_projections(rejected)
    assert validate_context({}, rejected, "ACTIVATE_STRATEGY_INSTANCE") is None
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), rejected
        )["denial_code"]
        == "CAPABILITY_SNAPSHOT_BLOCKED"
    )

    structural_mutations = [
        lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
            observed_at="2026-01-01T00:00:01Z"
        ),
        lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
            source_payload_hash="0" * 64
        ),
        lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
            exchange_id="KRAKEN"
        ),
    ]
    for mutation in structural_mutations:
        context = fixture_context()
        mutation(context)
        rehash_m05_projections(context)
        assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")["denial_code"] == (
            "TRUSTED_CONTEXT_INVALID"
        )


def test_extra_global_m04_capabilities_do_not_grant_cross_environment_authority():
    context = fixture_context()
    product = context["product_capabilities_by_environment"]["TESTNET"]
    product["allowed_operations"].extend(["PAPER_LOCAL_SIMULATION", "LIVE_VISIBLE_LOCKED_ONLY"])
    assert dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )["allowed"]
    product["allowed_operations"].remove("TESTNET_PRIVATE_EXECUTION_AFTER_READINESS")
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "PRODUCT_CAPABILITIES_BLOCKED"
    )


@pytest.mark.parametrize(
    "reference_name", ["m05_capabilities", "m05_permissions", "m04_product_capabilities"]
)
def test_broken_canonical_registry_pointer_is_contract_inconsistent(reference_name):
    reference = CONTRACT["canonical_array_enum_registry_refs"][reference_name]
    original = reference["json_pointer"]
    try:
        reference["json_pointer"] = "/missing/canonical/registry"
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            fixture_context(),
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        reference["json_pointer"] = original


def test_missing_request_resource_is_domain_denial_not_contract_fault():
    context = fixture_context()
    request = request_for("BIND_EXECUTION_ROUTE")
    context["strategy_instances_by_id"][IDS["sinst"]].update(
        execution_route_id=None, lifecycle_state="DRAFT"
    )
    request["execution_route_id"] = f"xroute_{UUID7}0f"
    assert validate_context(request, context, "BIND_EXECUTION_ROUTE") is None
    assert (
        dispatcher("BIND_EXECUTION_ROUTE", request, context)["denial_code"]
        == "EXECUTION_ROUTE_NOT_FOUND"
    )


def test_lineage_gap_missing_history_and_implicit_upgrade_are_executed():
    for present_version in (1, 2):
        context = fixture_context()
        current = context["strategy_definitions_by_id"][IDS["sdef"]]
        current["definition_version"] = 3
        historical = copy.deepcopy(current)
        historical["definition_version"] = present_version
        context["previous_strategy_definitions_by_version_key"][
            f"{IDS['sdef']}@{present_version}"
        ] = historical
        assert (
            dispatcher(
                "CREATE_STRATEGY_DEFINITION", request_for("CREATE_STRATEGY_DEFINITION"), context
            )["denial_code"]
            == "TRUSTED_CONTEXT_INVALID"
        )

    context = fixture_context()
    v1 = context["strategy_definitions_by_id"][IDS["sdef"]]
    v2 = copy.deepcopy(v1)
    v2["definition_version"] = 2
    context["strategy_definitions_by_id"][IDS["sdef"]] = v2
    context["previous_strategy_definitions_by_version_key"][f"{IDS['sdef']}@1"] = v1
    request = request_for("CREATE_STRATEGY_INSTANCE")
    request["strategy_definition_version"] = 1
    assert (
        dispatcher("CREATE_STRATEGY_INSTANCE", request, context)["denial_code"]
        == "STRATEGY_DEFINITION_VERSION_MISMATCH"
    )


def test_candidate_route_account_public_channel_and_authorization_mutations():
    request, context = make_reachability_case("BIND_EXECUTION_ROUTE", "ROUTE_SCOPE_MISMATCH")
    assert validate_context(request, context, "BIND_EXECUTION_ROUTE") is None
    assert (
        dispatcher("BIND_EXECUTION_ROUTE", request, context)["denial_code"]
        == "ROUTE_SCOPE_MISMATCH"
    )

    context = fixture_context()
    instance = context["strategy_instances_by_id"][IDS["sinst"]]
    instance.update(lifecycle_state="DRAFT", market_data_route_id=None)
    context["market_data_routes_by_id"][IDS["mdr"]]["channel_types"].append("PRIVATE_ORDERS")
    assert (
        dispatcher("BIND_MARKET_DATA_ROUTE", request_for("BIND_MARKET_DATA_ROUTE"), context)[
            "denial_code"
        ]
        == "TRUSTED_CONTEXT_INVALID"
    )


def test_universe_account_snapshot_and_credential_binding_mutations_are_context_invalid():
    mutations = [
        lambda c: c["universes_by_id"][IDS["univ"]].update(exchange_account_id=f"xacc_{UUID7}0f"),
        lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
            environment="LIVE"
        ),
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(exchange_id="KRAKEN"),
    ]
    for mutation in mutations:
        context = fixture_context()
        mutation(context)
        assert (
            dispatcher(
                "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
            )["denial_code"]
            == "TRUSTED_CONTEXT_INVALID"
        )


SCALAR_ENUM_CASES = [
    (
        "ExchangeAccountProjection",
        "lifecycle_state",
        "accounts_by_id",
        IDS["xacc"],
        ("DRAFT", "ACTIVE", "DISABLED", "RETIRED"),
    ),
    (
        "TradingUniverseProjection",
        "lifecycle_state",
        "universes_by_id",
        IDS["univ"],
        ("DRAFT", "ACTIVE", "RETIRED", "REJECTED"),
    ),
    (
        "AccountCapabilitySnapshotProjection",
        "status",
        "account_capability_snapshots_by_id",
        IDS["capsnap"],
        ("VALID", "STALE", "REJECTED"),
    ),
    (
        "CredentialProfileProjection",
        "credential_purpose",
        "credential_profiles_by_id",
        IDS["cred"],
        ("ACCOUNT_READ", "ORDER_ENTRY", "RECONCILIATION"),
    ),
    (
        "InstrumentProjection",
        "trading_status",
        "instruments_by_id",
        IDS["instr"],
        ("TRADING", "HALTED", "SUSPENDED", "DELISTED", "UNKNOWN"),
    ),
    (
        "InstrumentCatalogProjection",
        "status",
        "catalogs_by_id",
        IDS["icat"],
        ("VALID", "PARTIAL", "STALE", "REJECTED"),
    ),
]


@pytest.mark.parametrize(
    ("schema_name", "field", "map_name", "record_id", "values"), SCALAR_ENUM_CASES
)
def test_canonical_m05_scalar_values_are_structurally_valid(
    schema_name, field, map_name, record_id, values
):
    reference_name = SCHEMAS[schema_name]["enum_registry_ref"][field]
    assert resolve_canonical_registry(reference_name) == values
    for value in values:
        context = fixture_context()
        context[map_name][record_id][field] = value
        if schema_name == "TradingUniverseProjection":
            universe = context[map_name][record_id]
            if value == "ACTIVE":
                universe["activated_at_utc"] = "2025-01-01T00:02:00Z"
                universe["retired_at_utc"] = None
            elif value == "RETIRED":
                universe["retired_at_utc"] = "2025-01-01T00:03:00Z"
        rehash_m05_projections(context)
        assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None


@pytest.mark.parametrize(
    ("map_name", "record_id", "field", "value"),
    [
        ("accounts_by_id", IDS["xacc"], "lifecycle_state", "ALIEN"),
        ("universes_by_id", IDS["univ"], "lifecycle_state", "ALIEN"),
        ("account_capability_snapshots_by_id", IDS["capsnap"], "status", "INVALID"),
        ("account_capability_snapshots_by_id", IDS["capsnap"], "status", "ALIEN"),
        ("credential_profiles_by_id", IDS["cred"], "credential_purpose", "MARKET_DATA"),
        ("credential_profiles_by_id", IDS["cred"], "credential_purpose", "ALIEN"),
        ("instruments_by_id", IDS["instr"], "trading_status", "ACTIVE"),
        ("instruments_by_id", IDS["instr"], "trading_status", "RETIRED"),
        ("instruments_by_id", IDS["instr"], "trading_status", "ALIEN"),
        ("catalogs_by_id", IDS["icat"], "status", "ALIEN"),
    ],
)
def test_noncanonical_m05_scalar_values_invalidate_trusted_context(
    map_name, record_id, field, value
):
    context = fixture_context()
    context[map_name][record_id][field] = value
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("state", ["DRAFT", "DISABLED", "RETIRED"])
def test_canonical_inactive_testnet_account_states_block_operability(state):
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = state
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "ACCOUNT_READINESS_BLOCKED"


@pytest.mark.parametrize("state", ["DRAFT", "RETIRED", "REJECTED"])
def test_canonical_inactive_universe_states_are_operation_denials(state):
    context = fixture_context()
    context["universes_by_id"][IDS["univ"]]["lifecycle_state"] = state
    if state == "RETIRED":
        context["universes_by_id"][IDS["univ"]]["retired_at_utc"] = "2025-01-01T00:03:00Z"
    rehash_m05_projections(context)
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRADING_UNIVERSE_INVALID"


@pytest.mark.parametrize("status", ["STALE", "REJECTED"])
def test_canonical_inoperable_snapshot_statuses_block_testnet(status):
    context = fixture_context()
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["status"] = status
    rehash_m05_projections(context)
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "CAPABILITY_SNAPSHOT_BLOCKED"


@pytest.mark.parametrize("credential_purpose", ["ACCOUNT_READ", "RECONCILIATION"])
def test_canonical_non_order_entry_purposes_block_testnet(credential_purpose):
    context = fixture_context()
    context["credential_profiles_by_id"][IDS["cred"]]["credential_purpose"] = credential_purpose
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "ACCOUNT_READINESS_BLOCKED"


@pytest.mark.parametrize(
    "reference_name",
    [
        "m05_account_lifecycle_states",
        "m05_universe_lifecycle_states",
        "m05_snapshot_statuses",
        "m05_credential_purposes",
        "m05_instrument_trading_statuses",
        "m05_catalog_statuses",
    ],
)
def test_broken_scalar_registry_reference_is_contract_inconsistent(reference_name):
    reference = CONTRACT["canonical_scalar_enum_registry_refs"][reference_name]
    original = dict(reference)
    try:
        reference["json_pointer"] = "/missing/canonical/registry"
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            fixture_context(),
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        reference.clear()
        reference.update(original)


def test_unknown_scalar_registry_reference_is_special_contract_denial():
    schema = SCHEMAS["AccountCapabilitySnapshotProjection"]
    original = schema["enum_registry_ref"]["status"]
    try:
        schema["enum_registry_ref"]["status"] = "missing_scalar_registry"
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            fixture_context(),
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        schema["enum_registry_ref"]["status"] = original


@pytest.mark.parametrize(
    ("field", "value"),
    [("contract", "alien_contract.json"), ("json_pointer", "/schema_version")],
)
def test_canonical_registry_resolver_faults_are_special_contract_denials(field, value):
    reference = CONTRACT["canonical_scalar_enum_registry_refs"]["m05_snapshot_statuses"]
    original = dict(reference)
    try:
        reference[field] = value
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            fixture_context(),
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        reference.clear()
        reference.update(original)


def test_markdown_projection_policies_match_all_canonical_m05_registries():
    markdown = PATH.with_suffix(".md").read_text(encoding="utf-8")
    expected = {
        "m05_snapshot_statuses": ("VALID", "STALE", "REJECTED"),
        "m05_instrument_trading_statuses": (
            "TRADING",
            "HALTED",
            "SUSPENDED",
            "DELISTED",
            "UNKNOWN",
        ),
        "m05_catalog_statuses": ("VALID", "PARTIAL", "STALE", "REJECTED"),
    }
    for reference_name, values in expected.items():
        assert resolve_canonical_registry(reference_name) == values
    for pointer in (
        "/account_capability_snapshot_contract/statuses",
        "/instrument_contract/trading_statuses",
        "/instrument_catalog_snapshot_contract/statuses",
    ):
        assert pointer in markdown
    projection_section = markdown.split("## Canonical M0.5 Instrument and Catalog projections", 1)[
        1
    ]
    assert "structurally trusted" in projection_section
    assert "operation-specific operability" in projection_section
    assert "All declared external JSON Pointers" in markdown
    assert "stale or invalid snapshots" not in markdown
    assert "`INVALID`" not in projection_section
    assert "`ACTIVE`" not in projection_section
    assert "`RETIRED`" not in projection_section


def test_all_m05_scalar_projection_bindings_use_canonical_references_only():
    expected = {
        ("ExchangeAccountProjection", "lifecycle_state"): "m05_account_lifecycle_states",
        ("TradingUniverseProjection", "lifecycle_state"): "m05_universe_lifecycle_states",
        ("AccountCapabilitySnapshotProjection", "status"): "m05_snapshot_statuses",
        ("CredentialProfileProjection", "credential_purpose"): "m05_credential_purposes",
        ("InstrumentProjection", "trading_status"): "m05_instrument_trading_statuses",
        ("InstrumentCatalogProjection", "status"): "m05_catalog_statuses",
    }
    expected.update(
        {
            ("ExchangeAccountProjection", "connection_state"): "m05_connection_states",
            (
                "ExchangeAccountProjection",
                "execution_authorization",
            ): "m05_execution_authorizations",
            (
                "ExchangeAccountProjection",
                "external_account_identity_state",
            ): "m05_external_identity_states",
        }
    )
    assert set(CONTRACT["canonical_scalar_enum_registry_refs"]) == set(expected.values())
    for (schema_name, field), reference_name in expected.items():
        schema = SCHEMAS[schema_name]
        assert schema["enum_registry_ref"][field] == reference_name
        assert field not in schema["enum_registry"]
        assert resolve_canonical_registry(reference_name)


@pytest.mark.parametrize("status", ["HALTED", "SUSPENDED", "DELISTED", "UNKNOWN"])
def test_canonical_nontrading_instruments_are_operation_denials(status):
    context = fixture_context()
    context["instruments_by_id"][IDS["instr"]]["trading_status"] = status
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "INSTRUMENT_SCOPE_MISMATCH"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"
    assert (
        result["audit_event_type"]
        == CONTRACT["denial_event_by_operation"]["ACTIVATE_STRATEGY_INSTANCE"]
    )


def test_trading_instrument_and_fresh_valid_catalog_allow_activation():
    context = fixture_context()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    assert dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )["allowed"]


@pytest.mark.parametrize("status", ["PARTIAL", "STALE", "REJECTED"])
def test_canonical_nonoperational_catalog_statuses_are_domain_denials(status):
    context = fixture_context()
    context["catalogs_by_id"][IDS["icat"]]["status"] = status
    rehash_m05_projections(context)
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRADING_UNIVERSE_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_expired_valid_catalog_is_structural_but_not_operational():
    context = fixture_context()
    context["catalogs_by_id"][IDS["icat"]]["stale_after_utc"] = "2025-12-31T23:59:59Z"
    rehash_m05_projections(context)
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRADING_UNIVERSE_INVALID"


@pytest.mark.parametrize("operation", ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workspace_id", f"ws_{UUID7}0f"),
        ("metadata_version", False),
        ("metadata_version", 0),
        ("observed_at_utc", "2026-01-01T00:00:01Z"),
        ("effective_at_utc", "2026-01-01T00:00:01Z"),
        ("observed_at_utc", "2025-12-31T23:59:31Z"),
        ("stale_after_utc", "2025-12-31T23:59:30Z"),
    ],
)
def test_instrument_structural_authority_failures_are_context_invalid(operation, field, value):
    context = fixture_context()
    context["instruments_by_id"][IDS["instr"]][field] = value
    result = dispatcher(operation, request_for(operation), context)
    assert not result["allowed"]
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("operation", ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"])
@pytest.mark.parametrize("status", ["HALTED", "SUSPENDED", "DELISTED", "UNKNOWN"])
def test_instrument_operability_is_shared_by_readiness_and_activation(operation, status):
    context = fixture_context()
    context["instruments_by_id"][IDS["instr"]]["trading_status"] = status
    result = dispatcher(operation, request_for(operation), context)
    assert result["denial_code"] == "INSTRUMENT_SCOPE_MISMATCH"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("operation", ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"])
def test_instrument_freshness_boundary_is_exclusive(operation):
    context = fixture_context()
    context["instruments_by_id"][IDS["instr"]]["stale_after_utc"] = context["validation_time_utc"]
    assert validate_context({}, context, operation) is None
    result = dispatcher(operation, request_for(operation), context)
    assert result["denial_code"] == "INSTRUMENT_SCOPE_MISMATCH"


@pytest.mark.parametrize("operation", ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"])
def test_testnet_snapshot_must_authorize_every_universe_instrument_type(operation):
    context = fixture_context()
    assert dispatcher(operation, request_for(operation), context)["allowed"]
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["supported_instrument_types"] = []
    rehash_m05_projections(context)
    assert validate_context({}, context, operation) is None
    result = dispatcher(operation, request_for(operation), context)
    assert result["denial_code"] == "CAPABILITY_SNAPSHOT_BLOCKED"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("instrument_type", ["ALIEN", "MARGIN_PAIR"])
def test_unknown_or_market_disallowed_snapshot_instrument_type_invalidates_context(
    instrument_type,
):
    context = fixture_context()
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["supported_instrument_types"] = [
        instrument_type
    ]
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_canonical_projection_identities_and_markdown_are_machine_synchronized():
    expected = {
        "InstrumentCatalogProjection": "catalog_snapshot_id",
        "AccountCapabilitySnapshotProjection": "account_capability_snapshot_id",
    }
    for schema_name, identity in expected.items():
        assert SCHEMAS[schema_name]["id_field"] == identity
        assert identity in SCHEMAS[schema_name]["exact_fields"]
        assert "projection_aliases" not in SCHEMAS[schema_name]
    assert (
        "observed_permission_set" in SCHEMAS["AccountCapabilitySnapshotProjection"]["exact_fields"]
    )
    assert "source_catalog_snapshot_ids" in SCHEMAS["TradingUniverseProjection"]["exact_fields"]
    assert SCHEMAS["AccountCapabilitySnapshotProjection"]["array_policy"][
        "supported_instrument_types"
    ] == {"unique": True, "empty_allowed": True, "item_registry_ref": "m05_instrument_types"}
    assert resolve_canonical_registry("m05_instrument_types") == (
        "SPOT_PAIR",
        "MARGIN_PAIR",
        "PERPETUAL_CONTRACT",
        "DELIVERY_FUTURE",
        "OPTION",
    )
    markdown = PATH.with_suffix(".md").read_text(encoding="utf-8")
    for text in (
        "catalog_snapshot_id",
        "account_capability_snapshot_id",
        "observed_permission_set",
        "source_catalog_snapshot_ids",
        "/instrument_type_registry",
        "validate_strategy_execution_operability",
        "validation_time_utc < stale_after_utc",
    ):
        assert text in markdown


def test_cross_contract_projection_audit_is_complete_and_executable():
    audit = CONTRACT["cross_contract_projection_audit"]
    expected = {
        "ExchangeAccountProjection": "exchange_account_id",
        "TradingUniverseProjection": "trading_universe_id",
        "InstrumentProjection": "instrument_id",
        "InstrumentCatalogProjection": "catalog_snapshot_id",
        "AccountCapabilitySnapshotProjection": "account_capability_snapshot_id",
        "CredentialProfileProjection": "credential_profile_id",
    }
    assert set(audit) == set(expected)
    for projection, identity in expected.items():
        entry = audit[projection]
        assert entry["canonical_identity_field"] == identity
        assert identity in SCHEMAS[projection]["exact_fields"]
        assert set(entry["authority_relevant_fields"]) == set(SCHEMAS[projection]["exact_fields"])
        assert entry["canonical_registries"]
        assert entry["structural_invariants"]
        assert entry["operation_specific_operability"]
        assert "intentional_omissions" in entry
        assert entry["validated_context_maps"]
        covered_maps = {
            **CONTRACT["trusted_validation_context"]["map_fields"],
            **CONTRACT["trusted_validation_context"]["history_map_fields"],
        }
        assert all(map_name in covered_maps for map_name in entry["validated_context_maps"])
        assert "validated upstream" not in json.dumps(entry).lower()
        if entry["hash_definition_ref"] is not None:
            assert entry["hash_definition_ref"] in CONTRACT["canonical_hash_definition_refs"]
            assert entry["lineage_source"] in entry["validated_context_maps"]
    snapshot_fields = set(audit["AccountCapabilitySnapshotProjection"]["authority_relevant_fields"])
    assert {
        "version",
        "observed_at_utc",
        "effective_at_utc",
        "stale_after_utc",
        "adapter_family_id",
        "adapter_version",
        "content_hash",
    } <= snapshot_fields
    assert "m05_exchange_registry_entries" in audit["InstrumentProjection"]["canonical_registries"]


@pytest.mark.parametrize(
    "operation", ["CREATE_STRATEGY_INSTANCE", "BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"]
)
def test_every_successful_mutating_plan_produces_a_valid_persisted_graph(operation):
    context = fixture_context(instance_state="DRAFT")
    instance = context["strategy_instances_by_id"][IDS["sinst"]]
    if operation == "CREATE_STRATEGY_INSTANCE":
        context["strategy_instances_by_id"].clear()
    elif operation == "BIND_MARKET_DATA_ROUTE":
        instance["market_data_route_id"] = None
        instance["execution_route_id"] = IDS["xroute"]
    else:
        instance["market_data_route_id"] = IDS["mdr"]
        instance["execution_route_id"] = None
    request = request_for(operation)
    result = dispatcher(operation, request, context)
    assert result["allowed"]
    apply_planned_transition_and_validate(operation, context, result)


@pytest.mark.parametrize(
    "operation", ["CREATE_STRATEGY_INSTANCE", "BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"]
)
def test_cross_workspace_instrument_cannot_be_created_or_bound_into_a_strategy_graph(operation):
    context = fixture_context(instance_state="DRAFT")
    foreign_workspace = f"ws_{UUID7}0f"
    context["workspace_ids"].append(foreign_workspace)
    context["instruments_by_id"][IDS["instr"]]["workspace_id"] = foreign_workspace
    instance = context["strategy_instances_by_id"][IDS["sinst"]]
    if operation == "CREATE_STRATEGY_INSTANCE":
        context["strategy_instances_by_id"].clear()
    elif operation == "BIND_MARKET_DATA_ROUTE":
        instance["market_data_route_id"] = None
    else:
        instance["execution_route_id"] = None
    result = dispatcher(operation, request_for(operation), context)
    assert not result["allowed"]
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_at_utc", "2026-01-01T00:00:01Z"),
        ("effective_at_utc", "2025-12-31T23:58:59Z"),
        ("stale_after_utc", "2025-12-31T23:59:30Z"),
    ],
)
def test_malformed_snapshot_temporal_authority_is_context_invalid(field, value):
    context = fixture_context()
    context["account_capability_snapshots_by_id"][IDS["capsnap"]][field] = value
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize("operation", ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"])
def test_canonically_expired_snapshot_is_structural_but_blocks_testnet_operability(operation):
    context = fixture_context()
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["stale_after_utc"] = context[
        "validation_time_utc"
    ]
    rehash_m05_projections(context)
    assert validate_context({}, context, operation) is None
    result = dispatcher(operation, request_for(operation), context)
    assert result["denial_code"] == "CAPABILITY_SNAPSHOT_BLOCKED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stale_after_utc", "not-a-timestamp"),
        ("stale_after_utc", "2026-01-01T00:05:00"),
        ("observed_at_utc", "2026-01-01T00:00:01Z"),
        ("effective_at_utc", "2025-12-31T23:58:00Z"),
        ("stale_after_utc", "2025-12-31T23:59:00Z"),
    ],
)
def test_malformed_or_inconsistent_catalog_timestamps_invalidate_context(field, value):
    context = fixture_context()
    context["catalogs_by_id"][IDS["icat"]][field] = value
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("status", "STALE"),
        ("observed_permission_set", ["READ_ACCOUNT"]),
        ("supported_instrument_types", []),
        ("observed_at_utc", "2025-12-31T23:58:59Z"),
        ("effective_at_utc", "2025-12-31T23:59:31Z"),
        ("stale_after_utc", "2026-01-01T00:06:00Z"),
        ("adapter_family_id", "wrong-adapter"),
        ("adapter_version", "2.0.0"),
        ("previous_snapshot_id", f"capsnap_{UUID7}0f"),
    ],
)
def test_snapshot_hash_inputs_cannot_change_without_canonical_rehash(field, value):
    context = fixture_context()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    context["account_capability_snapshots_by_id"][IDS["capsnap"]][field] = value
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(
    ("map_name", "field", "value"),
    [
        ("catalogs_by_id", "instrument_ids", []),
        ("catalogs_by_id", "status", "STALE"),
        ("catalogs_by_id", "adapter_version", "2.0.0"),
        ("catalogs_by_id", "observed_at_utc", "2025-12-31T23:58:59Z"),
        ("catalogs_by_id", "previous_snapshot_id", f"icat_{UUID7}0f"),
        ("universes_by_id", "instrument_ids", []),
        ("universes_by_id", "source_catalog_snapshot_ids", []),
        ("universes_by_id", "version", 2),
        ("universes_by_id", "lifecycle_state", "RETIRED"),
        ("universes_by_id", "previous_version_id", f"univ_{UUID7}0f"),
        ("universes_by_id", "creation_reason", "CHANGED"),
    ],
)
def test_catalog_and_universe_hash_inputs_cannot_change_without_rehash(map_name, field, value):
    context = fixture_context()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    record_id = IDS["icat"] if map_name == "catalogs_by_id" else IDS["univ"]
    context[map_name][record_id][field] = value
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_canonical_hash_sorts_unordered_arrays_and_rejects_arbitrary_digest():
    context = fixture_context()
    snapshot = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    original = snapshot["content_hash"]
    snapshot["observed_permission_set"].reverse()
    assert canonical_m05_hash("account_capability_snapshot", snapshot) == original
    snapshot["content_hash"] = "0" * 64
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize("reference_name", list(CONTRACT["canonical_hash_definition_refs"]))
def test_broken_hash_definition_pointer_is_contract_inconsistent(reference_name):
    reference = CONTRACT["canonical_hash_definition_refs"][reference_name]
    original = reference["json_pointer"]
    context = fixture_context()
    try:
        reference["json_pointer"] = "/missing/hash/definition"
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            context,
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        reference["json_pointer"] = original


@pytest.mark.parametrize(
    "lineage_fault", ["missing", "self_cycle", "two_node_cycle", "multi_node_cycle"]
)
def test_snapshot_lineage_is_total_and_cycle_safe(lineage_fault):
    context = fixture_context()
    current = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    first_id, second_id = f"capsnap_{UUID7}0e", f"capsnap_{UUID7}0f"
    first = copy.deepcopy(current)
    first.update(account_capability_snapshot_id=first_id, version=1, previous_snapshot_id=None)
    first["content_hash"] = canonical_m05_hash("account_capability_snapshot", first)
    second = copy.deepcopy(current)
    second.update(
        account_capability_snapshot_id=second_id, version=2, previous_snapshot_id=first_id
    )
    second["content_hash"] = canonical_m05_hash("account_capability_snapshot", second)
    current.update(version=3, previous_snapshot_id=second_id)
    context["previous_account_capability_snapshots_by_id"] = {
        first_id: first,
        second_id: second,
    }
    if lineage_fault == "missing":
        current["previous_snapshot_id"] = f"capsnap_{UUID7}00"
    elif lineage_fault == "self_cycle":
        current["previous_snapshot_id"] = IDS["capsnap"]
    elif lineage_fault == "two_node_cycle":
        second["previous_snapshot_id"] = second_id
    else:
        first["previous_snapshot_id"] = second_id
    for record in [first, second, current]:
        record["content_hash"] = canonical_m05_hash("account_capability_snapshot", record)
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(
    ("source", "permissions"),
    [
        ("credential", ["PLACE_ORDERS"]),
        ("snapshot", ["PLACE_ORDERS"]),
        ("identity", ["PLACE_ORDERS"]),
        ("credential", ["READ_ACCOUNT"]),
        ("snapshot", ["READ_ACCOUNT"]),
        ("identity", ["READ_ACCOUNT"]),
        ("credential", ["READ_ACCOUNT", "PLACE_ORDERS", "WITHDRAW"]),
        ("snapshot", ["READ_ACCOUNT", "PLACE_ORDERS", "WITHDRAW"]),
        ("identity", ["READ_ACCOUNT", "PLACE_ORDERS", "WITHDRAW"]),
    ],
)
def test_effective_permission_intersection_blocks_missing_or_withdraw(source, permissions):
    context = fixture_context()
    if source == "credential":
        context["credential_profiles_by_id"][IDS["cred"]]["permission_snapshot"] = permissions
    elif source == "snapshot":
        context["account_capability_snapshots_by_id"][IDS["capsnap"]]["observed_permission_set"] = (
            permissions
        )
        rehash_m05_projections(context)
    else:
        context["external_identity_snapshots_by_account_id"][IDS["xacc"]][
            "observed_permission_set"
        ] = permissions
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "ACCOUNT_READINESS_BLOCKED"
    assert (
        result["audit_event_type"]
        == CONTRACT["denial_event_by_operation"]["ACTIVATE_STRATEGY_INSTANCE"]
    )
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connection_state", "DISCONNECTED"),
        ("execution_authorization", "READ_ONLY"),
        ("external_account_identity_state", "UNAVAILABLE"),
    ],
)
def test_canonical_nonoperational_account_authority_is_domain_denial(field, value):
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]][field] = value
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "ACCOUNT_READINESS_BLOCKED"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract", "missing-contract.json"),
        ("json_pointer", "exchange_registry_contract/entries"),
        ("json_pointer", "/exchange_registry_contract"),
        ("json_pointer", "/exchange_registry_contract/missing"),
    ],
)
def test_exchange_registry_pointer_corruption_is_exception_safe(field, value):
    reference = CONTRACT["canonical_cross_contract_registry_refs"]["m05_exchange_registry_entries"]
    original = reference[field]
    context = fixture_context()
    try:
        reference[field] = value
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            context,
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        reference[field] = original


def test_duplicate_exchange_registry_identity_is_exception_safe():
    entries = M05_CONTRACT["exchange_registry_contract"]["entries"]
    context = fixture_context()
    entries.append(copy.deepcopy(entries[0]))
    try:
        result = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE",
            request_for("ACTIVATE_STRATEGY_INSTANCE"),
            context,
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"
    finally:
        entries.pop()


def context_with_previous_snapshot():
    context = fixture_context()
    current = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    predecessor_id = f"capsnap_{UUID7}0e"
    predecessor = copy.deepcopy(current)
    predecessor.update(
        account_capability_snapshot_id=predecessor_id,
        version=1,
        previous_snapshot_id=None,
    )
    current.update(version=2, previous_snapshot_id=predecessor_id)
    context["previous_account_capability_snapshots_by_id"][predecessor_id] = predecessor
    rehash_m05_projections(context)
    predecessor["content_hash"] = canonical_m05_hash("account_capability_snapshot", predecessor)
    return context, predecessor


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exchange_account_id", f"xacc_{UUID7}0f"),
        ("exchange_id", "unknown_exchange"),
        ("adapter_family_id", "wrong-adapter"),
        ("observed_at_utc", "2026-01-01T00:00:00Z"),
        ("effective_at_utc", "2025-12-31T23:00:00Z"),
    ],
)
def test_every_previous_snapshot_receives_full_canonical_validation(field, value):
    context, predecessor = context_with_previous_snapshot()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    predecessor[field] = value
    predecessor["content_hash"] = canonical_m05_hash("account_capability_snapshot", predecessor)
    denial = validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")
    assert denial is not None and denial["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def context_with_previous_catalog():
    context = fixture_context()
    current = context["catalogs_by_id"][IDS["icat"]]
    predecessor_id = f"icat_{UUID7}0e"
    predecessor = copy.deepcopy(current)
    predecessor.update(catalog_snapshot_id=predecessor_id, previous_snapshot_id=None)
    current["previous_snapshot_id"] = predecessor_id
    context["previous_catalogs_by_id"][predecessor_id] = predecessor
    historical_instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
    historical_instrument["catalog_snapshot_id"] = predecessor_id
    context["instruments_by_id"][IDS["instr"]]["metadata_version"] = 2
    context["instrument_history_by_id"][IDS["instr"]] = [historical_instrument]
    rehash_m05_projections(context)
    predecessor["content_hash"] = canonical_m05_hash("instrument_catalog_snapshot", predecessor)
    return context, predecessor


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("observed_at_utc", "2026-01-01T00:00:00Z"),
        ("instrument_ids", [f"instr_{UUID7}0f"]),
        ("exchange_id", "unknown_exchange"),
        ("adapter_family_id", "wrong-adapter"),
    ],
)
def test_every_previous_catalog_receives_full_canonical_validation(field, value):
    context, predecessor = context_with_previous_catalog()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    predecessor[field] = value
    predecessor["content_hash"] = canonical_m05_hash("instrument_catalog_snapshot", predecessor)
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def context_with_previous_universe():
    context = fixture_context()
    current = context["universes_by_id"][IDS["univ"]]
    predecessor_id = f"univ_{UUID7}0e"
    predecessor = copy.deepcopy(current)
    predecessor.update(
        trading_universe_id=predecessor_id,
        lifecycle_state="RETIRED",
        version=1,
        previous_version_id=None,
        retired_at_utc="2025-01-01T00:03:00Z",
    )
    current.update(version=2, previous_version_id=predecessor_id)
    context["previous_universes_by_id"][predecessor_id] = predecessor
    rehash_m05_projections(context)
    predecessor["content_hash"] = canonical_m05_hash("trading_universe", predecessor)
    return context, predecessor


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exchange_account_id", f"xacc_{UUID7}0f"),
        ("instrument_ids", [f"instr_{UUID7}0f"]),
        ("source_catalog_snapshot_ids", [f"icat_{UUID7}0f"]),
        ("lifecycle_state", "DRAFT"),
    ],
)
def test_every_previous_universe_receives_full_canonical_validation(field, value):
    context, predecessor = context_with_previous_universe()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    predecessor[field] = value
    predecessor["content_hash"] = canonical_m05_hash("trading_universe", predecessor)
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def context_with_previous_credential():
    context = fixture_context()
    current = context["credential_profiles_by_id"][IDS["cred"]]
    predecessor_id = f"cred_{UUID7}0e"
    predecessor = copy.deepcopy(current)
    predecessor.update(
        credential_profile_id=predecessor_id,
        lifecycle_state="RETIRED",
        created_at_utc="2025-12-31T23:56:00Z",
        retired_at_utc="2025-12-31T23:57:00Z",
        rotated_from_credential_profile_id=None,
    )
    current["rotated_from_credential_profile_id"] = predecessor_id
    context["previous_credential_profiles_by_id"][predecessor_id] = predecessor
    return context, predecessor


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lifecycle_state", "ACTIVE"),
        ("retired_at_utc", None),
        ("created_at_utc", "2026-01-01T00:02:00Z"),
        ("exchange_account_id", f"xacc_{UUID7}0f"),
        ("exchange_id", "unknown_exchange"),
        ("environment_scope", "PAPER"),
    ],
)
def test_every_previous_credential_receives_full_canonical_validation(field, value):
    context, predecessor = context_with_previous_credential()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    predecessor[field] = value
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_active_selected_credential_cannot_have_retirement_timestamp():
    context = fixture_context()
    context["credential_profiles_by_id"][IDS["cred"]]["retired_at_utc"] = "2025-01-01T00:00:30Z"
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_all_external_identity_records_are_globally_validated():
    context = fixture_context()
    second_account_id = f"xacc_{UUID7}0f"
    second = copy.deepcopy(context["accounts_by_id"][IDS["xacc"]])
    second["exchange_account_id"] = second_account_id
    second["account_capability_snapshot_id"] = None
    second["active_credential_profile_id"] = None
    context["accounts_by_id"][second_account_id] = second
    duplicate = copy.deepcopy(context["external_identity_snapshots_by_account_id"][IDS["xacc"]])
    context["external_identity_snapshots_by_account_id"][second_account_id] = duplicate
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_instrument_history_exact_catalog_resolution_and_version_order():
    context, _ = context_with_previous_catalog()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    history = context["instrument_history_by_id"][IDS["instr"]]
    history.append(copy.deepcopy(history[0]))
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize("fault", ["empty", "wrong_key", "identity", "catalog", "adapter"])
def test_instrument_history_fail_closed_mutations(fault):
    context, _ = context_with_previous_catalog()
    history = context["instrument_history_by_id"]
    record = history[IDS["instr"]][0]
    if fault == "empty":
        history[IDS["instr"]] = []
    elif fault == "wrong_key":
        history[f"instr_{UUID7}0f"] = history.pop(IDS["instr"])
    elif fault == "identity":
        record["venue_symbol"] = "ETHUSDT"
    elif fault == "catalog":
        record["catalog_snapshot_id"] = f"icat_{UUID7}0f"
    else:
        record["source_adapter_family_id"] = "wrong-adapter"
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


def test_previous_catalog_cannot_borrow_current_instrument_membership():
    context, _ = context_with_previous_catalog()
    context["instrument_history_by_id"].clear()
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_name", 7),
        ("aliases", "paper"),
        ("aliases", ["paper", "paper"]),
        ("capability_discovery_policy", None),
        ("capability_discovery_policy", "ALIEN"),
        ("supported_environments", []),
        ("supported_market_types", []),
        ("supported_instrument_types", []),
    ],
)
def test_exchange_registry_metadata_and_fingerprint_are_closed(field, value):
    context = fixture_context()
    entry = M05_CONTRACT["exchange_registry_contract"]["entries"][0]
    original = entry[field]
    try:
        entry[field] = value
        direct = run_direct_call_graph(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )
        dispatched = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )
        assert tuple(
            direct[key]
            for key in ("allowed", "denial_code", "audit_event_type", "planned_transition")
        ) == tuple(
            dispatched[key]
            for key in ("allowed", "denial_code", "audit_event_type", "planned_transition")
        )
        assert dispatched["denial_code"] == "CONTRACT_INCONSISTENT"
    finally:
        entry[field] = original


@pytest.mark.parametrize("reference_name", list(CONTRACT["canonical_hash_definition_refs"]))
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain_separator", "BROKEN"),
        ("input_fields", []),
        ("canonicalization", "BROKEN"),
        ("encoding", "UTF-16"),
        ("digest_format", "UPPERCASE_HEX"),
        ("algorithm", "SHA-1"),
    ],
)
def test_hash_definition_fingerprint_covers_exact_metadata(reference_name, field, value):
    context = fixture_context()
    definition = resolve_canonical_pointer(
        CONTRACT["canonical_hash_definition_refs"][reference_name], dict
    )
    original = definition[field]
    try:
        definition[field] = value
        direct = run_direct_call_graph(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )
        dispatched = dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )
        assert direct == dispatched
        assert dispatched["denial_code"] == "CONTRACT_INCONSISTENT"
    finally:
        definition[field] = original


def test_paper_static_build_time_snapshot_is_optional():
    context = fixture_context(environment="PAPER")
    context["accounts_by_id"][IDS["xacc"]]["account_capability_snapshot_id"] = None
    context["account_capability_snapshots_by_id"].clear()
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    assert dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )["allowed"]


def test_testnet_adapter_snapshot_is_structurally_required():
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["account_capability_snapshot_id"] = None
    context["account_capability_snapshots_by_id"].clear()
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("secure_store_reference", "plaintext-secret"),
        ("secure_store_reference", "http://locator"),
        ("public_key_identifier", 7),
        ("saas_sync_candidate", True),
    ],
)
def test_credential_model_a_security_metadata_is_structural(field, value):
    context = fixture_context()
    context["credential_profiles_by_id"][IDS["cred"]][field] = value
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )


@pytest.mark.parametrize(
    "source", ["totally-wrong-source", "unknown-family/1.0", "generic_testnet_adapter_family/"]
)
def test_external_identity_adapter_version_source_is_closed(source):
    context = fixture_context()
    context["external_identity_snapshots_by_account_id"][IDS["xacc"]]["adapter_version_source"] = (
        source
    )
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )
