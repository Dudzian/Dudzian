"""Executable, pure M0.6 reference validators (not production runtime)."""

import copy
import hashlib
import json
import re
from datetime import UTC, datetime
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


def resolve_canonical_registry(reference_name):
    references = {
        **CONTRACT["canonical_array_enum_registry_refs"],
        **CONTRACT["canonical_scalar_enum_registry_refs"],
    }
    reference = references[reference_name]
    sources = {
        "environment_and_product_capabilities.json": M04_CONTRACT,
        "exchange_accounts_and_instruments.json": M05_CONTRACT,
    }
    value = sources[reference["contract"]]
    pointer = reference["json_pointer"]
    if type(pointer) is not str or not pointer.startswith("/"):
        raise KeyError(reference_name)
    for part in pointer[1:].split("/"):
        value = value[part.replace("~1", "/").replace("~0", "~")]
    if (
        type(value) is not list
        or not value
        or any(type(item) is not str or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise TypeError(reference_name)
    return tuple(value)


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


def validate_references(context):
    definitions = context["strategy_definitions_by_id"]
    history = context["previous_strategy_definitions_by_version_key"]
    validation_time = parse_time(context["validation_time_utc"])
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
    for instrument in context["instruments_by_id"].values():
        catalog = context["catalogs_by_id"].get(instrument["catalog_id"])
        if (
            not catalog
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
                or instrument["catalog_id"] != cid
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
        if set(universe["source_catalog_ids"]) != {item["catalog_id"] for item in instruments}:
            return False
        for item in instruments:
            catalog = context["catalogs_by_id"].get(item["catalog_id"])
            if not catalog or item["instrument_id"] not in catalog["instrument_ids"]:
                return False
            if any(
                item[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            ):
                return False
    for account in context["accounts_by_id"].values():
        if account["workspace_id"] not in context["workspace_ids"]:
            return False
        portfolio = context["portfolios_by_id"].get(account["portfolio_id"])
        snapshot = context["account_capability_snapshots_by_id"].get(
            account["capability_snapshot_id"]
        )
        product = context["product_capabilities_by_environment"].get(account["environment"])
        if (
            not portfolio
            or portfolio["workspace_id"] != account["workspace_id"]
            or not snapshot
            or not product
            or product["environment"] != account["environment"]
        ):
            return False
        if snapshot["exchange_account_id"] != account["exchange_account_id"] or any(
            snapshot[field] != account[field]
            for field in ("exchange_id", "environment", "market_type")
        ):
            return False
        if snapshot["source_payload_hash"] != snapshot["attested_payload_hash"]:
            return False
        if parse_time(snapshot["observed_at"]) > validation_time:
            return False
        credential_id = account["active_credential_profile_id"]
        if credential_id is not None:
            credential = context["credential_profiles_by_id"].get(credential_id)
            if (
                not credential
                or credential["lifecycle_state"] != "ACTIVE"
                or credential["exchange_account_id"] != account["exchange_account_id"]
                or any(
                    credential[field] != account[field] for field in ("exchange_id", "environment")
                )
            ):
                return False
    for snapshot in context["account_capability_snapshots_by_id"].values():
        account = context["accounts_by_id"].get(snapshot["exchange_account_id"])
        if (
            not account
            or any(
                snapshot[field] != account[field]
                for field in ("exchange_id", "environment", "market_type")
            )
            or snapshot["source_payload_hash"] != snapshot["attested_payload_hash"]
            or parse_time(snapshot["observed_at"]) > validation_time
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
            catalog = context["catalogs_by_id"].get(instrument["catalog_id"])
            if (
                not catalog
                or any(
                    route[field] != instrument[field]
                    for field in ("exchange_id", "environment", "market_type")
                )
                or route["adapter_family_id"] != instrument["source_adapter_family_id"]
                or route["adapter_family_id"] != catalog["adapter_family_id"]
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
        if execution:
            for iid in universe["instrument_ids"]:
                instrument = context["instruments_by_id"][iid]
                catalog = context["catalogs_by_id"][instrument["catalog_id"]]
                if (
                    execution["adapter_family_id"] != instrument["source_adapter_family_id"]
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
    expected = set(spec["map_fields"]) | set(spec["scalar_fields"]) | set(spec["array_fields"])
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
            if key != expected_key:
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
    try:
        if not validate_references(context):
            return deny(operation, "TRUSTED_CONTEXT_INVALID")
    except (KeyError, TypeError, ValueError):
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
    if route["environment"] != account["environment"] or not endpoint_ok(
        route, account, "EXECUTION"
    ):
        return deny(operation, "ENDPOINT_FALLBACK_FORBIDDEN")
    if (
        route["exchange_account_id"] != account["exchange_account_id"]
        or route["workspace_id"] != item["workspace_id"]
        or any(route[f] != account[f] for f in ("exchange_id", "market_type"))
    ):
        return deny(operation, "ROUTE_SCOPE_MISMATCH")
    if account["environment"] == "LIVE":
        return deny(operation, "LIVE_EXECUTION_FORBIDDEN")
    expected = CONTRACT["authorization_dependencies_by_environment"][account["environment"]]
    if set(route["authorization_dependencies"]) != set(expected):
        return deny(operation, "ROUTE_CAPABILITY_BLOCKED")
    universe = context["universes_by_id"][item["trading_universe_id"]]
    if any(
        context["instruments_by_id"][iid]["source_adapter_family_id"] != route["adapter_family_id"]
        for iid in universe["instrument_ids"]
    ):
        return deny(operation, "ROUTE_ADAPTER_MISMATCH")
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
    if account["environment"] == "LIVE":
        return deny(operation, "LIVE_EXECUTION_FORBIDDEN")
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
    if account["lifecycle_state"] != "ACTIVE":
        return deny(operation, "ACCOUNT_READINESS_BLOCKED")
    route = context["execution_routes_by_id"][item["execution_route_id"]]
    snapshot = context["account_capability_snapshots_by_id"][account["capability_snapshot_id"]]
    max_age = CONTRACT["account_capability_snapshot_policy"]["max_age_seconds_by_environment"][
        "TESTNET"
    ]
    age = (
        parse_time(context["validation_time_utc"]) - parse_time(snapshot["observed_at"])
    ).total_seconds()
    if (
        snapshot["status"] != "VALID"
        or age > max_age
        or "PLACE_ORDERS" not in snapshot["capabilities"]
    ):
        return deny(operation, "CAPABILITY_SNAPSHOT_BLOCKED")
    credential = context["credential_profiles_by_id"].get(account["active_credential_profile_id"])
    if (
        not credential
        or credential["lifecycle_state"] != "ACTIVE"
        or credential["purpose"] != "ORDER_ENTRY"
        or "PLACE_ORDERS" not in credential["permissions"]
    ):
        return deny(operation, "ACCOUNT_READINESS_BLOCKED")
    if "PLACE_ORDERS" not in route["route_capability_ceiling"]:
        return deny(operation, "ROUTE_CAPABILITY_BLOCKED")
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
    account = context["accounts_by_id"][item["exchange_account_id"]]
    universe = context["universes_by_id"][item["trading_universe_id"]]
    if universe["lifecycle_state"] != "ACTIVE":
        return deny(operation, "TRADING_UNIVERSE_INVALID")
    catalogs = {
        context["instruments_by_id"][iid]["catalog_id"] for iid in universe["instrument_ids"]
    }
    validation_time = parse_time(context["validation_time_utc"])
    if any(
        context["catalogs_by_id"][catalog_id]["status"] != "VALID"
        or validation_time >= parse_time(context["catalogs_by_id"][catalog_id]["stale_after_utc"])
        for catalog_id in catalogs
    ):
        return deny(operation, "TRADING_UNIVERSE_INVALID")
    if any(
        context["instruments_by_id"][iid]["trading_status"] != "TRADING"
        for iid in universe["instrument_ids"]
    ):
        return deny(operation, "INSTRUMENT_SCOPE_MISMATCH")
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


def dispatcher(operation, request, context):
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
    except (KeyError, TypeError, IndexError, ValueError):
        return deny(operation, "CONTRACT_INCONSISTENT")
    return deny(operation, "CONTRACT_INCONSISTENT")


def run_direct_call_graph(operation, request, context):
    for name in CONTRACT["operation_validator_call_graph"][operation]:
        result = VALIDATORS[name](request, context, operation)
        if result is not None:
            return result
    return deny(operation, "CONTRACT_INCONSISTENT")


def fixture_context(environment="TESTNET", instance_state="BOUND"):
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
        "exchange_id": "BINANCE",
        "environment": environment,
        "market_type": "SPOT",
        "lifecycle_state": "ACTIVE",
        "capability_snapshot_id": IDS["capsnap"],
        "active_credential_profile_id": IDS["cred"] if environment == "TESTNET" else None,
    }
    instrument = {
        "instrument_id": IDS["instr"],
        "catalog_id": IDS["icat"],
        "exchange_id": "BINANCE",
        "environment": environment,
        "market_type": "SPOT",
        "instrument_type": "SPOT_PAIR",
        "source_adapter_family_id": "binance-v1",
        "trading_status": "TRADING",
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
        "exchange_id": "BINANCE",
        "environment": environment,
        "market_type": "SPOT",
        "adapter_family_id": "binance-v1",
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
        "exchange_id": "BINANCE",
        "environment": environment,
        "market_type": "SPOT",
        "adapter_family_id": "binance-v1",
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
        "exchange_id": "BINANCE",
        "environment": "TESTNET",
        "lifecycle_state": "ACTIVE",
        "purpose": "ORDER_ENTRY",
        "permissions": ["PLACE_ORDERS"],
    }
    return {
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
                "source_catalog_ids": [IDS["icat"]],
                "lifecycle_state": "ACTIVE",
            }
        },
        "instruments_by_id": {IDS["instr"]: instrument},
        "catalogs_by_id": {
            IDS["icat"]: {
                "catalog_id": IDS["icat"],
                "exchange_id": "BINANCE",
                "environment": environment,
                "market_type": "SPOT",
                "adapter_family_id": "binance-v1",
                "instrument_ids": [IDS["instr"]],
                "observed_at_utc": "2025-12-31T23:59:00Z",
                "effective_at_utc": "2025-12-31T23:59:30Z",
                "status": "VALID",
                "stale_after_utc": "2026-01-01T00:05:00Z",
            }
        },
        "account_capability_snapshots_by_id": {
            IDS["capsnap"]: {
                "snapshot_id": IDS["capsnap"],
                "exchange_account_id": IDS["xacc"],
                "exchange_id": "BINANCE",
                "environment": environment,
                "market_type": "SPOT",
                "status": "VALID",
                "capabilities": ["PLACE_ORDERS"],
                "observed_at": now,
                "source_payload_hash": h,
                "attested_payload_hash": h,
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
    }


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
            route["workspace_id"] = f"ws_{UUID7}0f"
            context["workspace_ids"].append(route["workspace_id"])
        else:
            account = copy.deepcopy(context["accounts_by_id"][IDS["xacc"]])
            account_id, snapshot_id = f"xacc_{UUID7}0f", f"capsnap_{UUID7}0f"
            account.update(
                exchange_account_id=account_id,
                capability_snapshot_id=snapshot_id,
                active_credential_profile_id=f"cred_{UUID7}0f",
            )
            snapshot = copy.deepcopy(context["account_capability_snapshots_by_id"][IDS["capsnap"]])
            snapshot.update(snapshot_id=snapshot_id, exchange_account_id=account_id)
            credential = copy.deepcopy(context["credential_profiles_by_id"][IDS["cred"]])
            credential.update(
                credential_profile_id=f"cred_{UUID7}0f", exchange_account_id=account_id
            )
            context["accounts_by_id"][account_id] = account
            context["account_capability_snapshots_by_id"][snapshot_id] = snapshot
            context["credential_profiles_by_id"][credential["credential_profile_id"]] = credential
            route.update(exchange_account_id=account_id)
    elif denial == "ENDPOINT_FALLBACK_FORBIDDEN":
        route = (
            context["market_data_routes_by_id"][IDS["mdr"]]
            if operation == "BIND_MARKET_DATA_ROUTE"
            else context["execution_routes_by_id"][IDS["xroute"]]
        )
        if operation == "BIND_MARKET_DATA_ROUTE":
            iid, cid = f"instr_{UUID7}0f", f"icat_{UUID7}0f"
            instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
            catalog = copy.deepcopy(context["catalogs_by_id"][IDS["icat"]])
            instrument.update(instrument_id=iid, catalog_id=cid, environment="LIVE")
            catalog.update(catalog_id=cid, instrument_ids=[iid], environment="LIVE")
            context["instruments_by_id"][iid] = instrument
            context["catalogs_by_id"][cid] = catalog
            route.update(
                environment="LIVE", endpoint_class="LIVE_PUBLIC_DATA", instrument_ids=[iid]
            )
        else:
            account = copy.deepcopy(context["accounts_by_id"][IDS["xacc"]])
            account_id, snapshot_id = f"xacc_{UUID7}0f", f"capsnap_{UUID7}0f"
            account.update(
                exchange_account_id=account_id,
                capability_snapshot_id=snapshot_id,
                active_credential_profile_id=None,
                environment="PAPER",
            )
            snapshot = copy.deepcopy(context["account_capability_snapshots_by_id"][IDS["capsnap"]])
            snapshot.update(
                snapshot_id=snapshot_id, exchange_account_id=account_id, environment="PAPER"
            )
            context["accounts_by_id"][account_id] = account
            context["account_capability_snapshots_by_id"][snapshot_id] = snapshot
            context["product_capabilities_by_environment"]["PAPER"] = fixture_context("PAPER")[
                "product_capabilities_by_environment"
            ]["PAPER"]
            route.update(
                exchange_account_id=account_id,
                environment="PAPER",
                endpoint_class="PAPER_SIMULATION",
                authorization_dependencies=CONTRACT["authorization_dependencies_by_environment"][
                    "PAPER"
                ],
            )
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
    elif denial == "ROUTE_ADAPTER_MISMATCH":
        context["execution_routes_by_id"][IDS["xroute"]]["adapter_family_id"] = "other-v1"
    elif denial == "ROUTE_CAPABILITY_BLOCKED":
        context["execution_routes_by_id"][IDS["xroute"]]["authorization_dependencies"] = [
            "PRODUCT_CAPABILITIES"
        ]
    elif denial == "LIVE_EXECUTION_FORBIDDEN":
        context = fixture_context("LIVE")
        request = request_for(operation)
        if operation == "BIND_EXECUTION_ROUTE":
            context["strategy_instances_by_id"][IDS["sinst"]].update(
                execution_route_id=None, lifecycle_state="DRAFT"
            )
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
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(environment="LIVE")
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
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(permissions=["READ_ACCOUNT"]),
        lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(purpose="ACCOUNT_READ"),
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
        instrument.update(instrument_id=f"instr_{UUID7}0f", catalog_id=f"icat_{UUID7}0f")
        context["instruments_by_id"][instrument["instrument_id"]] = instrument

    def missing_back_reference(context):
        instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
        instrument["instrument_id"] = f"instr_{UUID7}0f"
        context["instruments_by_id"][instrument["instrument_id"]] = instrument

    def orphan_snapshot(context):
        snapshot = copy.deepcopy(context["account_capability_snapshots_by_id"][IDS["capsnap"]])
        snapshot.update(snapshot_id=f"capsnap_{UUID7}0f", exchange_account_id=f"xacc_{UUID7}0f")
        context["account_capability_snapshots_by_id"][snapshot["snapshot_id"]] = snapshot

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
        status="REJECTED", observed_at="2020-01-01T00:00:00Z", capabilities=["READ_ACCOUNT"]
    )
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
                observed_at="2025-12-31T23:00:00Z"
            ),
            "CAPABILITY_SNAPSHOT_BLOCKED",
        ),
        (
            lambda c: c["account_capability_snapshots_by_id"][IDS["capsnap"]].update(
                capabilities=["READ_ACCOUNT"]
            ),
            "CAPABILITY_SNAPSHOT_BLOCKED",
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
            lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(purpose="ACCOUNT_READ"),
            "ACCOUNT_READINESS_BLOCKED",
        ),
        (
            lambda c: c["credential_profiles_by_id"][IDS["cred"]].update(
                permissions=["READ_ACCOUNT"]
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
    assert validate_context({}, context, "VALIDATE_ROUTE_READINESS") is None
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == expected
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_live_execution_readiness_is_forbidden():
    context = fixture_context("LIVE")
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == "LIVE_EXECUTION_FORBIDDEN"


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
            "capabilities",
            "ALIEN_CAPABILITY",
        ),
        ("credential_profiles_by_id", "cred", "permissions", "ALIEN_PERMISSION"),
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
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["capabilities"].append(
        "READ_ACCOUNT"
    )
    context["credential_profiles_by_id"][IDS["cred"]]["permissions"].append("READ_ORDERS")
    context["product_capabilities_by_environment"]["TESTNET"]["allowed_operations"].extend(
        ["PAPER_LOCAL_SIMULATION", "LIVE_VISIBLE_LOCKED_ONLY"]
    )
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
        "purpose",
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
        assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None


@pytest.mark.parametrize(
    ("map_name", "record_id", "field", "value"),
    [
        ("accounts_by_id", IDS["xacc"], "lifecycle_state", "ALIEN"),
        ("universes_by_id", IDS["univ"], "lifecycle_state", "ALIEN"),
        ("account_capability_snapshots_by_id", IDS["capsnap"], "status", "INVALID"),
        ("account_capability_snapshots_by_id", IDS["capsnap"], "status", "ALIEN"),
        ("credential_profiles_by_id", IDS["cred"], "purpose", "MARKET_DATA"),
        ("credential_profiles_by_id", IDS["cred"], "purpose", "ALIEN"),
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
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRADING_UNIVERSE_INVALID"


@pytest.mark.parametrize("status", ["STALE", "REJECTED"])
def test_canonical_inoperable_snapshot_statuses_block_testnet(status):
    context = fixture_context()
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["status"] = status
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "CAPABILITY_SNAPSHOT_BLOCKED"


@pytest.mark.parametrize("purpose", ["ACCOUNT_READ", "RECONCILIATION"])
def test_canonical_non_order_entry_purposes_block_testnet(purpose):
    context = fixture_context()
    context["credential_profiles_by_id"][IDS["cred"]]["purpose"] = purpose
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
        ("CredentialProfileProjection", "purpose"): "m05_credential_purposes",
        ("InstrumentProjection", "trading_status"): "m05_instrument_trading_statuses",
        ("InstrumentCatalogProjection", "status"): "m05_catalog_statuses",
    }
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
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRADING_UNIVERSE_INVALID"
    assert result["denial_code"] != "CONTRACT_INCONSISTENT"


def test_expired_valid_catalog_is_structural_but_not_operational():
    context = fixture_context()
    context["catalogs_by_id"][IDS["icat"]]["stale_after_utc"] = "2025-12-31T23:59:59Z"
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRADING_UNIVERSE_INVALID"


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
