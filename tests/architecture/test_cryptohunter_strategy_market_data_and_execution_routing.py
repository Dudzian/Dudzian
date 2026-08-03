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
    expected = EXPECTED_EXECUTABLE_OPERATION_PROTOCOL
    operation_is_string = type(operation) is str
    if denial == "CONTRACT_INCONSISTENT":
        event = expected["special_events"]["CONTRACT_INCONSISTENT"]
    elif not operation_is_string or operation not in expected["operation_registry"]:
        event = expected["special_events"]["UNKNOWN_OPERATION"]
    elif allowed:
        event = expected["success_event_by_operation"][operation]
    else:
        event = expected["denial_event_by_operation"][operation]
    return deep_freeze(
        {
            "allowed": allowed,
            "denial_code": denial,
            "operation": operation if operation_is_string else None,
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


IMMUTABLE_CANONICAL_REGISTRY_FINGERPRINTS = deep_freeze(
    {
        (
            "environment_and_product_capabilities.json",
            "/capability_id_registry/current_schema_allowed_capability_ids",
        ): "da387f22de48e101f293b5dd304eedf52e0cb5dbcf8142b0f0c98b972acd55d2",
        (
            "exchange_accounts_and_instruments.json",
            "/account_capability_snapshot_contract/statuses",
        ): "ebe92c083d6949c70946e9367ee6f073b03dee6ab1c4ef937ff594c27d6fca34",
        (
            "exchange_accounts_and_instruments.json",
            "/credential_profile_contract/credential_purposes",
        ): "65b6616dc26bda201d1348cf2bd88c3a02f19408642ad30d5911330c5d4761aa",
        (
            "exchange_accounts_and_instruments.json",
            "/credential_profile_contract/permission_registry",
        ): "23a0295aaa597c702ab17348a8552057760d5ffe4df849eb9995e36fd0f8d847",
        (
            "exchange_accounts_and_instruments.json",
            "/environment_registry",
        ): "c2182111523b676163dda381a902ed4ef238a9e1508484e1da1ce790e83cf0d9",
        (
            "exchange_accounts_and_instruments.json",
            "/exchange_account_contract/connection_states",
        ): "4e7471865d166327ab3ca006ae01b7312d762ce09a1ce9ce85a8a3826f067fc0",
        (
            "exchange_accounts_and_instruments.json",
            "/exchange_account_contract/execution_authorizations",
        ): "06cd91f979f0f90a6ae247565950152397305215299cc3450a54acee307d120b",
        (
            "exchange_accounts_and_instruments.json",
            "/exchange_account_contract/lifecycle_states",
        ): "b986f40965e0e81b1293a709b08dbd90b88818e78e4583e15c1e95fd7762cbb4",
        (
            "exchange_accounts_and_instruments.json",
            "/external_account_identity_contract/states",
        ): "a3af91e769180487348480ebf5e24c8e456aeea3ffed772aebce448007560cd6",
        (
            "exchange_accounts_and_instruments.json",
            "/instrument_catalog_snapshot_contract/statuses",
        ): "620b1ffbc737b7d095f742945301e604522813d4837d36243c817ce76867246e",
        (
            "exchange_accounts_and_instruments.json",
            "/instrument_contract/trading_statuses",
        ): "fffc70c35a4aa8440c884815fd6f51b67e4e76b76ff74ec5e1abc3b35ccab206",
        (
            "exchange_accounts_and_instruments.json",
            "/instrument_type_registry",
        ): "e99ba1e3af1a3b5771add15b7d0ab0659c0b24a1bd855c04c0d5c27f2d2831f8",
        (
            "exchange_accounts_and_instruments.json",
            "/market_type_registry",
        ): "f9f3ddd7226000e3d8998868a218adb2fa44127623df1bb39d486c4128129ac2",
        (
            "exchange_accounts_and_instruments.json",
            "/trading_universe_contract/lifecycle_states",
        ): "f462d4ea15629414b3b8862183985f4759242a6834052eead5117907f5596a16",
    }
)

EXPECTED_EXTERNAL_CANONICAL_ENUM_CONSUMERS = deep_freeze(
    {
        "AccountCapabilitySnapshotProjection.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "AccountCapabilitySnapshotProjection",
        },
        "AccountCapabilitySnapshotProjection.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "AccountCapabilitySnapshotProjection",
        },
        "AccountCapabilitySnapshotProjection.observed_permission_set": {
            "array_policy": {
                "empty_allowed": False,
                "item_registry_ref": "m05_capabilities",
                "unique": True,
            },
            "binding_kind": "item_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "observed_permission_set",
            "field_type": "array[enum]",
            "json_pointer": "/credential_profile_contract/permission_registry",
            "nullable": False,
            "registry_ref": "m05_capabilities",
            "schema": "AccountCapabilitySnapshotProjection",
        },
        "AccountCapabilitySnapshotProjection.status": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "status",
            "field_type": "enum",
            "json_pointer": "/account_capability_snapshot_contract/statuses",
            "nullable": False,
            "registry_ref": "m05_snapshot_statuses",
            "schema": "AccountCapabilitySnapshotProjection",
        },
        "AccountCapabilitySnapshotProjection.supported_instrument_types": {
            "array_policy": {
                "empty_allowed": True,
                "item_registry_ref": "m05_instrument_types",
                "unique": True,
            },
            "binding_kind": "item_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "supported_instrument_types",
            "field_type": "array[enum]",
            "json_pointer": "/instrument_type_registry",
            "nullable": False,
            "registry_ref": "m05_instrument_types",
            "schema": "AccountCapabilitySnapshotProjection",
        },
        "CredentialProfileProjection.credential_purpose": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "credential_purpose",
            "field_type": "enum",
            "json_pointer": "/credential_profile_contract/credential_purposes",
            "nullable": False,
            "registry_ref": "m05_credential_purposes",
            "schema": "CredentialProfileProjection",
        },
        "CredentialProfileProjection.environment_scope": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment_scope",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "CredentialProfileProjection",
        },
        "CredentialProfileProjection.permission_snapshot": {
            "array_policy": {
                "empty_allowed": False,
                "item_registry_ref": "m05_permissions",
                "unique": True,
            },
            "binding_kind": "item_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "permission_snapshot",
            "field_type": "array[enum]",
            "json_pointer": "/credential_profile_contract/permission_registry",
            "nullable": False,
            "registry_ref": "m05_permissions",
            "schema": "CredentialProfileProjection",
        },
        "ExchangeAccountProjection.connection_state": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "connection_state",
            "field_type": "enum",
            "json_pointer": "/exchange_account_contract/connection_states",
            "nullable": False,
            "registry_ref": "m05_connection_states",
            "schema": "ExchangeAccountProjection",
        },
        "ExchangeAccountProjection.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "ExchangeAccountProjection",
        },
        "ExchangeAccountProjection.execution_authorization": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "execution_authorization",
            "field_type": "enum",
            "json_pointer": "/exchange_account_contract/execution_authorizations",
            "nullable": False,
            "registry_ref": "m05_execution_authorizations",
            "schema": "ExchangeAccountProjection",
        },
        "ExchangeAccountProjection.external_account_identity_state": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "external_account_identity_state",
            "field_type": "enum",
            "json_pointer": "/external_account_identity_contract/states",
            "nullable": False,
            "registry_ref": "m05_external_identity_states",
            "schema": "ExchangeAccountProjection",
        },
        "ExchangeAccountProjection.lifecycle_state": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "lifecycle_state",
            "field_type": "enum",
            "json_pointer": "/exchange_account_contract/lifecycle_states",
            "nullable": False,
            "registry_ref": "m05_account_lifecycle_states",
            "schema": "ExchangeAccountProjection",
        },
        "ExchangeAccountProjection.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "ExchangeAccountProjection",
        },
        "ExecutionRoute.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "ExecutionRoute",
        },
        "ExecutionRoute.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "ExecutionRoute",
        },
        "ExecutionRoute.supported_instrument_types": {
            "array_policy": {
                "empty_allowed": False,
                "item_registry_ref": "m05_instrument_types",
                "unique": True,
            },
            "binding_kind": "item_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "supported_instrument_types",
            "field_type": "array[enum]",
            "json_pointer": "/instrument_type_registry",
            "nullable": False,
            "registry_ref": "m05_instrument_types",
            "schema": "ExecutionRoute",
        },
        "InstrumentCatalogProjection.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "InstrumentCatalogProjection",
        },
        "InstrumentCatalogProjection.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "InstrumentCatalogProjection",
        },
        "InstrumentCatalogProjection.status": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "status",
            "field_type": "enum",
            "json_pointer": "/instrument_catalog_snapshot_contract/statuses",
            "nullable": False,
            "registry_ref": "m05_catalog_statuses",
            "schema": "InstrumentCatalogProjection",
        },
        "InstrumentProjection.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "InstrumentProjection",
        },
        "InstrumentProjection.instrument_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "instrument_type",
            "field_type": "enum",
            "json_pointer": "/instrument_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "InstrumentProjection",
        },
        "InstrumentProjection.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "InstrumentProjection",
        },
        "InstrumentProjection.trading_status": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "trading_status",
            "field_type": "enum",
            "json_pointer": "/instrument_contract/trading_statuses",
            "nullable": False,
            "registry_ref": "m05_instrument_trading_statuses",
            "schema": "InstrumentProjection",
        },
        "MarketDataRoute.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "MarketDataRoute",
        },
        "MarketDataRoute.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "MarketDataRoute",
        },
        "ProductCapabilitiesProjection.allowed_operations": {
            "array_policy": {
                "empty_allowed": True,
                "item_registry_ref": "m04_product_capabilities",
                "unique": True,
            },
            "binding_kind": "item_registry_ref",
            "contract": "environment_and_product_capabilities.json",
            "exact_fields_membership": True,
            "field": "allowed_operations",
            "field_type": "array[enum]",
            "json_pointer": "/capability_id_registry/current_schema_allowed_capability_ids",
            "nullable": False,
            "registry_ref": "m04_product_capabilities",
            "schema": "ProductCapabilitiesProjection",
        },
        "ProductCapabilitiesProjection.environment": {
            "array_policy": None,
            "binding_kind": "derived_canonical_projection",
            "contract": "environment_and_product_capabilities.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/execution_environments",
            "nullable": False,
            "projection": "ordered environment_id values",
            "registry_ref": None,
            "schema": "ProductCapabilitiesProjection",
        },
        "TradingUniverseProjection.lifecycle_state": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "lifecycle_state",
            "field_type": "enum",
            "json_pointer": "/trading_universe_contract/lifecycle_states",
            "nullable": False,
            "registry_ref": "m05_universe_lifecycle_states",
            "schema": "TradingUniverseProjection",
        },
        "TrustedExternalIdentityProjection.environment": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "environment",
            "field_type": "enum",
            "json_pointer": "/environment_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "TrustedExternalIdentityProjection",
        },
        "TrustedExternalIdentityProjection.market_type": {
            "array_policy": None,
            "binding_kind": "enum_canonical_pointer_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "market_type",
            "field_type": "enum",
            "json_pointer": "/market_type_registry",
            "nullable": False,
            "registry_ref": None,
            "schema": "TrustedExternalIdentityProjection",
        },
        "TrustedExternalIdentityProjection.observed_permission_set": {
            "array_policy": {
                "empty_allowed": True,
                "item_registry_ref": "m05_capabilities",
                "unique": True,
            },
            "binding_kind": "item_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "observed_permission_set",
            "field_type": "array[enum]",
            "json_pointer": "/credential_profile_contract/permission_registry",
            "nullable": False,
            "registry_ref": "m05_capabilities",
            "schema": "TrustedExternalIdentityProjection",
        },
        "TrustedExternalIdentityProjection.state": {
            "array_policy": None,
            "binding_kind": "enum_registry_ref",
            "contract": "exchange_accounts_and_instruments.json",
            "exact_fields_membership": True,
            "field": "state",
            "field_type": "enum",
            "json_pointer": "/external_account_identity_contract/states",
            "nullable": False,
            "registry_ref": "m05_external_identity_states",
            "schema": "TrustedExternalIdentityProjection",
        },
    }
)

EXPECTED_EXECUTABLE_OPERATION_PROTOCOL = deep_freeze(
    {
        "allowed_denials_by_operation": {
            "ACTIVATE_STRATEGY_DEFINITION": [
                "CONTRACT_INCONSISTENT",
                "REQUEST_SCHEMA_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
                "STRATEGY_DEFINITION_NOT_FOUND",
                "STRATEGY_DEFINITION_VERSION_MISMATCH",
                "TRUSTED_CONTEXT_INVALID",
                "STRATEGY_DEFINITION_STATE_CONFLICT",
            ],
            "ACTIVATE_STRATEGY_INSTANCE": [
                "ACCOUNT_READINESS_BLOCKED",
                "CAPABILITY_SNAPSHOT_BLOCKED",
                "CONTRACT_INCONSISTENT",
                "EXECUTION_ROUTE_NOT_READY",
                "INSTRUMENT_SCOPE_MISMATCH",
                "MARKET_DATA_FRESHNESS_BLOCKED",
                "MARKET_DATA_ROUTE_NOT_READY",
                "MARKET_DATA_SEQUENCE_INVALID",
                "REQUEST_SCHEMA_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
                "ROUTE_CAPABILITY_BLOCKED",
                "STRATEGY_INSTANCE_NOT_FOUND",
                "TRADING_UNIVERSE_INVALID",
                "TRUSTED_CONTEXT_INVALID",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
            ],
            "BIND_EXECUTION_ROUTE": [
                "CONTRACT_INCONSISTENT",
                "EXECUTION_ROUTE_NOT_FOUND",
                "REQUEST_SCHEMA_INVALID",
                "ROUTE_CAPABILITY_BLOCKED",
                "ROUTE_SCOPE_MISMATCH",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
                "STRATEGY_INSTANCE_NOT_FOUND",
                "TRUSTED_CONTEXT_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
            ],
            "BIND_MARKET_DATA_ROUTE": [
                "CONTRACT_INCONSISTENT",
                "INSTRUMENT_SCOPE_MISMATCH",
                "MARKET_DATA_ROUTE_NOT_FOUND",
                "REQUEST_SCHEMA_INVALID",
                "ROUTE_SCOPE_MISMATCH",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
                "STRATEGY_INSTANCE_NOT_FOUND",
                "TRUSTED_CONTEXT_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
            ],
            "CREATE_STRATEGY_DEFINITION": [
                "CONTRACT_INCONSISTENT",
                "REQUEST_SCHEMA_INVALID",
                "STRATEGY_DEFINITION_ID_COLLISION",
                "STRATEGY_DEFINITION_VERSION_MISMATCH",
                "TRUSTED_CONTEXT_INVALID",
                "WORKSPACE_NOT_FOUND",
            ],
            "CREATE_STRATEGY_INSTANCE": [
                "CONTRACT_INCONSISTENT",
                "REQUEST_SCHEMA_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
                "STRATEGY_DEFINITION_NOT_FOUND",
                "STRATEGY_DEFINITION_VERSION_MISMATCH",
                "STRATEGY_INSTANCE_ID_COLLISION",
                "TRADING_UNIVERSE_INVALID",
                "TRUSTED_CONTEXT_INVALID",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
                "STRATEGY_DEFINITION_STATE_CONFLICT",
            ],
            "DEACTIVATE_STRATEGY_INSTANCE": [
                "CONTRACT_INCONSISTENT",
                "REQUEST_SCHEMA_INVALID",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
                "STRATEGY_INSTANCE_NOT_FOUND",
                "TRUSTED_CONTEXT_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
            ],
            "RETIRE_STRATEGY_DEFINITION": [
                "CONTRACT_INCONSISTENT",
                "REQUEST_SCHEMA_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
                "STRATEGY_DEFINITION_NOT_FOUND",
                "STRATEGY_DEFINITION_VERSION_MISMATCH",
                "TRUSTED_CONTEXT_INVALID",
                "STRATEGY_DEFINITION_STATE_CONFLICT",
            ],
            "RETIRE_STRATEGY_INSTANCE": [
                "CONTRACT_INCONSISTENT",
                "REQUEST_SCHEMA_INVALID",
                "RETIRED_RESOURCE_FORBIDDEN",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
                "STRATEGY_INSTANCE_NOT_FOUND",
                "TRUSTED_CONTEXT_INVALID",
            ],
            "VALIDATE_ROUTE_READINESS": [
                "CONTRACT_INCONSISTENT",
                "EXECUTION_ROUTE_NOT_READY",
                "MARKET_DATA_FRESHNESS_BLOCKED",
                "MARKET_DATA_ROUTE_NOT_READY",
                "MARKET_DATA_SEQUENCE_INVALID",
                "REQUEST_SCHEMA_INVALID",
                "STRATEGY_INSTANCE_NOT_FOUND",
                "TRUSTED_CONTEXT_INVALID",
                "ROUTE_CAPABILITY_BLOCKED",
                "INSTRUMENT_SCOPE_MISMATCH",
                "ACCOUNT_READINESS_BLOCKED",
                "CAPABILITY_SNAPSHOT_BLOCKED",
                "STRATEGY_INSTANCE_BINDING_MISMATCH",
                "RETIRED_RESOURCE_FORBIDDEN",
                "TRADING_UNIVERSE_INVALID",
            ],
        },
        "denial_code_registry": [
            "ACCOUNT_READINESS_BLOCKED",
            "CAPABILITY_SNAPSHOT_BLOCKED",
            "CONTRACT_INCONSISTENT",
            "EXECUTION_ROUTE_NOT_FOUND",
            "EXECUTION_ROUTE_NOT_READY",
            "INSTRUMENT_SCOPE_MISMATCH",
            "MARKET_DATA_FRESHNESS_BLOCKED",
            "MARKET_DATA_ROUTE_NOT_FOUND",
            "MARKET_DATA_ROUTE_NOT_READY",
            "MARKET_DATA_SEQUENCE_INVALID",
            "REQUEST_SCHEMA_INVALID",
            "RETIRED_RESOURCE_FORBIDDEN",
            "ROUTE_CAPABILITY_BLOCKED",
            "ROUTE_SCOPE_MISMATCH",
            "STRATEGY_DEFINITION_ID_COLLISION",
            "STRATEGY_DEFINITION_NOT_FOUND",
            "STRATEGY_DEFINITION_STATE_CONFLICT",
            "STRATEGY_DEFINITION_VERSION_MISMATCH",
            "STRATEGY_INSTANCE_BINDING_MISMATCH",
            "STRATEGY_INSTANCE_ID_COLLISION",
            "STRATEGY_INSTANCE_NOT_FOUND",
            "TRADING_UNIVERSE_INVALID",
            "TRUSTED_CONTEXT_INVALID",
            "UNKNOWN_OPERATION",
            "WORKSPACE_NOT_FOUND",
        ],
        "denial_event_by_operation": {
            "ACTIVATE_STRATEGY_DEFINITION": "STRATEGY_ROUTING_ACTIVATE_STRATEGY_DEFINITION_DENIED",
            "ACTIVATE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_ACTIVATE_STRATEGY_INSTANCE_DENIED",
            "BIND_EXECUTION_ROUTE": "STRATEGY_ROUTING_BIND_EXECUTION_ROUTE_DENIED",
            "BIND_MARKET_DATA_ROUTE": "STRATEGY_ROUTING_BIND_MARKET_DATA_ROUTE_DENIED",
            "CREATE_STRATEGY_DEFINITION": "STRATEGY_ROUTING_CREATE_STRATEGY_DEFINITION_DENIED",
            "CREATE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_CREATE_STRATEGY_INSTANCE_DENIED",
            "DEACTIVATE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_DEACTIVATE_STRATEGY_INSTANCE_DENIED",
            "RETIRE_STRATEGY_DEFINITION": "STRATEGY_ROUTING_RETIRE_STRATEGY_DEFINITION_DENIED",
            "RETIRE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_RETIRE_STRATEGY_INSTANCE_DENIED",
            "VALIDATE_ROUTE_READINESS": "STRATEGY_ROUTING_VALIDATE_ROUTE_READINESS_DENIED",
        },
        "forbidden_operations": [
            "SUBMIT_ORDER",
            "CREATE_ORDER",
            "CANCEL_ORDER",
            "REPLACE_ORDER",
            "EXECUTE_TRADE",
        ],
        "operation_registry": {
            "ACTIVATE_STRATEGY_DEFINITION": {
                "allowed_denials_ref": "allowed_denials_by_operation.ACTIVATE_STRATEGY_DEFINITION",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.ACTIVATE_STRATEGY_DEFINITION",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_ACTIVATE",
                        "operation": "ACTIVATE_STRATEGY_DEFINITION",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_definition_id",
                        "definition_version",
                    ],
                    "request_types": {
                        "authority": "string",
                        "definition_version": "integer",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_definition_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.ACTIVATE_STRATEGY_DEFINITION",
                "success_transition": "immutable planned activate_strategy_definition",
            },
            "ACTIVATE_STRATEGY_INSTANCE": {
                "allowed_denials_ref": "allowed_denials_by_operation.ACTIVATE_STRATEGY_INSTANCE",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.ACTIVATE_STRATEGY_INSTANCE",
                "intent": "ACTIVATE",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "ACTIVATE",
                        "operation": "ACTIVATE_STRATEGY_INSTANCE",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_instance_id",
                    ],
                    "request_types": {
                        "authority": "string",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_instance_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.ACTIVATE_STRATEGY_INSTANCE",
                "success_transition": "immutable planned activate_strategy_instance",
            },
            "BIND_EXECUTION_ROUTE": {
                "allowed_denials_ref": "allowed_denials_by_operation.BIND_EXECUTION_ROUTE",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.BIND_EXECUTION_ROUTE",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_FIRST_BIND",
                        "operation": "BIND_EXECUTION_ROUTE",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_instance_id",
                        "execution_route_id",
                    ],
                    "request_types": {
                        "authority": "string",
                        "execution_route_id": "id",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_instance_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.BIND_EXECUTION_ROUTE",
                "success_transition": "immutable planned bind_execution_route",
            },
            "BIND_MARKET_DATA_ROUTE": {
                "allowed_denials_ref": "allowed_denials_by_operation.BIND_MARKET_DATA_ROUTE",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.BIND_MARKET_DATA_ROUTE",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_FIRST_BIND",
                        "operation": "BIND_MARKET_DATA_ROUTE",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_instance_id",
                        "market_data_route_id",
                    ],
                    "request_types": {
                        "authority": "string",
                        "intent": "string",
                        "market_data_route_id": "id",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_instance_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.BIND_MARKET_DATA_ROUTE",
                "success_transition": "immutable planned bind_market_data_route",
            },
            "CREATE_STRATEGY_DEFINITION": {
                "allowed_denials_ref": "allowed_denials_by_operation.CREATE_STRATEGY_DEFINITION",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.CREATE_STRATEGY_DEFINITION",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_CREATE",
                        "operation": "CREATE_STRATEGY_DEFINITION",
                    },
                    "nested_schemas": {
                        "configuration": {
                            "exact_fields": ["lookback", "enabled"],
                            "field_types": {"enabled": "boolean", "lookback": "integer"},
                            "nullable_fields": [],
                        }
                    },
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_definition_id",
                        "workspace_id",
                        "strategy_type_id",
                        "definition_version",
                        "configuration",
                    ],
                    "request_types": {
                        "authority": "string",
                        "configuration": "object",
                        "definition_version": "integer",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_definition_id": "id",
                        "strategy_type_id": "string",
                        "workspace_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.CREATE_STRATEGY_DEFINITION",
                "success_transition": "immutable planned create_strategy_definition",
            },
            "CREATE_STRATEGY_INSTANCE": {
                "allowed_denials_ref": "allowed_denials_by_operation.CREATE_STRATEGY_INSTANCE",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.CREATE_STRATEGY_INSTANCE",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_CREATE",
                        "lifecycle_state": "DRAFT",
                        "operation": "CREATE_STRATEGY_INSTANCE",
                    },
                    "nested_schemas": {},
                    "nullable_fields": ["market_data_route_id", "execution_route_id"],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
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
                    ],
                    "request_types": {
                        "authority": "string",
                        "exchange_account_id": "id",
                        "execution_route_id": "id",
                        "intent": "string",
                        "lifecycle_state": "string",
                        "market_data_route_id": "id",
                        "operation": "string",
                        "portfolio_id": "id",
                        "request_id": "id",
                        "strategy_definition_id": "id",
                        "strategy_definition_version": "integer",
                        "strategy_instance_id": "id",
                        "trading_universe_id": "id",
                        "workspace_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.CREATE_STRATEGY_INSTANCE",
                "success_transition": "immutable planned create_strategy_instance",
            },
            "DEACTIVATE_STRATEGY_INSTANCE": {
                "allowed_denials_ref": "allowed_denials_by_operation.DEACTIVATE_STRATEGY_INSTANCE",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_DEACTIVATE",
                        "operation": "DEACTIVATE_STRATEGY_INSTANCE",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_instance_id",
                    ],
                    "request_types": {
                        "authority": "string",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_instance_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE",
                "success_transition": "immutable planned deactivate_strategy_instance",
            },
            "RETIRE_STRATEGY_DEFINITION": {
                "allowed_denials_ref": "allowed_denials_by_operation.RETIRE_STRATEGY_DEFINITION",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.RETIRE_STRATEGY_DEFINITION",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_RETIRE",
                        "operation": "RETIRE_STRATEGY_DEFINITION",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_definition_id",
                        "definition_version",
                    ],
                    "request_types": {
                        "authority": "string",
                        "definition_version": "integer",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_definition_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.RETIRE_STRATEGY_DEFINITION",
                "success_transition": "immutable planned retire_strategy_definition",
            },
            "RETIRE_STRATEGY_INSTANCE": {
                "allowed_denials_ref": "allowed_denials_by_operation.RETIRE_STRATEGY_INSTANCE",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.RETIRE_STRATEGY_INSTANCE",
                "intent": "MUTATION_PLAN_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "PLAN_RETIRE",
                        "operation": "RETIRE_STRATEGY_INSTANCE",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_instance_id",
                    ],
                    "request_types": {
                        "authority": "string",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_instance_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.RETIRE_STRATEGY_INSTANCE",
                "success_transition": "immutable planned retire_strategy_instance",
            },
            "VALIDATE_ROUTE_READINESS": {
                "allowed_denials_ref": "allowed_denials_by_operation.VALIDATE_ROUTE_READINESS",
                "authority": "CoreHost",
                "default_success": False,
                "denial_event_ref": "denial_event_by_operation.VALIDATE_ROUTE_READINESS",
                "intent": "VALIDATE_ONLY",
                "planned_result_immutable": True,
                "request_schema": {
                    "closed": True,
                    "constants": {
                        "authority": "CoreHost",
                        "intent": "VALIDATE_ONLY",
                        "operation": "VALIDATE_ROUTE_READINESS",
                    },
                    "nested_schemas": {},
                    "nullable_fields": [],
                    "request_fields": [
                        "operation",
                        "request_id",
                        "authority",
                        "intent",
                        "strategy_instance_id",
                    ],
                    "request_types": {
                        "authority": "string",
                        "intent": "string",
                        "operation": "string",
                        "request_id": "id",
                        "strategy_instance_id": "id",
                    },
                },
                "storage_mutation": False,
                "success_event_ref": "success_event_by_operation.VALIDATE_ROUTE_READINESS",
                "success_transition": "immutable planned validate_route_readiness",
            },
        },
        "operation_validator_call_graph": {
            "ACTIVATE_STRATEGY_DEFINITION": [
                "validate_request",
                "validate_context",
                "validate_definition_lifecycle",
            ],
            "ACTIVATE_STRATEGY_INSTANCE": [
                "validate_request",
                "validate_context",
                "validate_strategy_instance_lifecycle_preflight",
                "validate_authorization_operability",
                "validate_strategy_execution_operability",
                "validate_activation",
            ],
            "BIND_EXECUTION_ROUTE": [
                "validate_request",
                "validate_context",
                "validate_execution_bind",
            ],
            "BIND_MARKET_DATA_ROUTE": [
                "validate_request",
                "validate_context",
                "validate_market_bind",
            ],
            "CREATE_STRATEGY_DEFINITION": [
                "validate_request",
                "validate_context",
                "validate_definition_create",
            ],
            "CREATE_STRATEGY_INSTANCE": [
                "validate_request",
                "validate_context",
                "validate_instance_create",
            ],
            "DEACTIVATE_STRATEGY_INSTANCE": [
                "validate_request",
                "validate_context",
                "validate_strategy_instance_lifecycle_preflight",
                "validate_deactivation",
            ],
            "RETIRE_STRATEGY_DEFINITION": [
                "validate_request",
                "validate_context",
                "validate_definition_lifecycle",
            ],
            "RETIRE_STRATEGY_INSTANCE": [
                "validate_request",
                "validate_context",
                "validate_retirement",
            ],
            "VALIDATE_ROUTE_READINESS": [
                "validate_request",
                "validate_context",
                "validate_readiness_lifecycle",
                "validate_authorization_operability",
                "validate_strategy_execution_operability",
                "validate_route_readiness_operation",
            ],
        },
        "special_events": {
            "CONTRACT_INCONSISTENT": "STRATEGY_ROUTING_CONTRACT_INCONSISTENT",
            "UNKNOWN_OPERATION": "STRATEGY_ROUTING_UNKNOWN_OPERATION_DENIED",
        },
        "success_event_by_operation": {
            "ACTIVATE_STRATEGY_DEFINITION": "STRATEGY_ROUTING_ACTIVATE_STRATEGY_DEFINITION_SUCCEEDED",
            "ACTIVATE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_ACTIVATE_STRATEGY_INSTANCE_SUCCEEDED",
            "BIND_EXECUTION_ROUTE": "STRATEGY_ROUTING_BIND_EXECUTION_ROUTE_SUCCEEDED",
            "BIND_MARKET_DATA_ROUTE": "STRATEGY_ROUTING_BIND_MARKET_DATA_ROUTE_SUCCEEDED",
            "CREATE_STRATEGY_DEFINITION": "STRATEGY_ROUTING_CREATE_STRATEGY_DEFINITION_SUCCEEDED",
            "CREATE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_CREATE_STRATEGY_INSTANCE_SUCCEEDED",
            "DEACTIVATE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_DEACTIVATE_STRATEGY_INSTANCE_SUCCEEDED",
            "RETIRE_STRATEGY_DEFINITION": "STRATEGY_ROUTING_RETIRE_STRATEGY_DEFINITION_SUCCEEDED",
            "RETIRE_STRATEGY_INSTANCE": "STRATEGY_ROUTING_RETIRE_STRATEGY_INSTANCE_SUCCEEDED",
            "VALIDATE_ROUTE_READINESS": "STRATEGY_ROUTING_VALIDATE_ROUTE_READINESS_SUCCEEDED",
        },
        "validator_denial_registry": {
            "validate_activation": {
                "callable_required": True,
                "denials": [
                    "RETIRED_RESOURCE_FORBIDDEN",
                    "ACCOUNT_READINESS_BLOCKED",
                    "MARKET_DATA_ROUTE_NOT_READY",
                    "EXECUTION_ROUTE_NOT_READY",
                    "MARKET_DATA_FRESHNESS_BLOCKED",
                    "MARKET_DATA_SEQUENCE_INVALID",
                    "ROUTE_CAPABILITY_BLOCKED",
                ],
            },
            "validate_authorization_operability": {
                "callable_required": True,
                "denials": [
                    "ACCOUNT_READINESS_BLOCKED",
                    "CAPABILITY_SNAPSHOT_BLOCKED",
                    "ROUTE_CAPABILITY_BLOCKED",
                ],
            },
            "validate_context": {"callable_required": True, "denials": ["TRUSTED_CONTEXT_INVALID"]},
            "validate_deactivation": {"callable_required": True, "denials": []},
            "validate_definition_create": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_DEFINITION_ID_COLLISION",
                    "STRATEGY_DEFINITION_VERSION_MISMATCH",
                    "WORKSPACE_NOT_FOUND",
                ],
            },
            "validate_definition_lifecycle": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_DEFINITION_NOT_FOUND",
                    "STRATEGY_DEFINITION_VERSION_MISMATCH",
                    "RETIRED_RESOURCE_FORBIDDEN",
                    "STRATEGY_DEFINITION_STATE_CONFLICT",
                ],
            },
            "validate_execution_bind": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "STRATEGY_INSTANCE_BINDING_MISMATCH",
                    "EXECUTION_ROUTE_NOT_FOUND",
                    "ROUTE_SCOPE_MISMATCH",
                    "ROUTE_CAPABILITY_BLOCKED",
                    "RETIRED_RESOURCE_FORBIDDEN",
                ],
            },
            "validate_instance_create": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_ID_COLLISION",
                    "STRATEGY_DEFINITION_NOT_FOUND",
                    "STRATEGY_DEFINITION_VERSION_MISMATCH",
                    "RETIRED_RESOURCE_FORBIDDEN",
                    "TRADING_UNIVERSE_INVALID",
                    "STRATEGY_INSTANCE_BINDING_MISMATCH",
                    "STRATEGY_DEFINITION_STATE_CONFLICT",
                ],
            },
            "validate_market_bind": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "STRATEGY_INSTANCE_BINDING_MISMATCH",
                    "MARKET_DATA_ROUTE_NOT_FOUND",
                    "ROUTE_SCOPE_MISMATCH",
                    "INSTRUMENT_SCOPE_MISMATCH",
                    "RETIRED_RESOURCE_FORBIDDEN",
                ],
            },
            "validate_readiness_lifecycle": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "STRATEGY_INSTANCE_BINDING_MISMATCH",
                    "RETIRED_RESOURCE_FORBIDDEN",
                ],
            },
            "validate_request": {"callable_required": True, "denials": ["REQUEST_SCHEMA_INVALID"]},
            "validate_retirement": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "STRATEGY_INSTANCE_BINDING_MISMATCH",
                    "RETIRED_RESOURCE_FORBIDDEN",
                ],
            },
            "validate_route_readiness_operation": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "MARKET_DATA_ROUTE_NOT_READY",
                    "EXECUTION_ROUTE_NOT_READY",
                    "MARKET_DATA_FRESHNESS_BLOCKED",
                    "MARKET_DATA_SEQUENCE_INVALID",
                    "ROUTE_CAPABILITY_BLOCKED",
                    "INSTRUMENT_SCOPE_MISMATCH",
                ],
            },
            "validate_strategy_execution_operability": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "TRADING_UNIVERSE_INVALID",
                    "INSTRUMENT_SCOPE_MISMATCH",
                    "CAPABILITY_SNAPSHOT_BLOCKED",
                ],
            },
            "validate_strategy_instance_lifecycle_preflight": {
                "callable_required": True,
                "denials": [
                    "STRATEGY_INSTANCE_NOT_FOUND",
                    "STRATEGY_INSTANCE_BINDING_MISMATCH",
                    "RETIRED_RESOURCE_FORBIDDEN",
                ],
            },
        },
    }
)

EXPECTED_CANONICAL_DEPENDENCIES = deep_freeze(
    {
        "m04_capability_ids": {
            "authority_classification": "product authority",
            "consumers": ["ProductCapabilitiesProjection.allowed_operations"],
            "content_fingerprint_sha256": "da387f22de48e101f293b5dd304eedf52e0cb5dbcf8142b0f0c98b972acd55d2",
            "expected_result_type": "array",
            "json_pointer": "/capability_id_registry/current_schema_allowed_capability_ids",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m04_current_edition_environment_capabilities": {
            "authority_classification": "edition environment authority",
            "consumers": ["resolve_m04_environment_ids.current_edition_environment_capabilities"],
            "content_fingerprint_sha256": "3ef879a0dad726b6f225a2748599a7a4a013e85a2cfba04ee581b348324212ee",
            "expected_result_type": "object",
            "json_pointer": "/current_product_edition/environment_capabilities",
            "projection": "ordered object keys",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m04_current_product_edition": {
            "authority_classification": "edition authority",
            "consumers": ["resolve_m04_current_product_capability_policy.current_product_edition"],
            "content_fingerprint_sha256": "5fa73f356d9b315161031b4fb41661b8294ed14ce42bfae4565e87b34120adc8",
            "expected_result_type": "object",
            "json_pointer": "/current_product_edition",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m04_execution_environments": {
            "authority_classification": "environment authority",
            "consumers": ["resolve_m04_environment_ids.execution_environments"],
            "content_fingerprint_sha256": "61f5dd195aa69c646dc296c2d8c0f2ce97683979e3ef6ddc35b0adaa269a809b",
            "expected_result_type": "array",
            "json_pointer": "/execution_environments",
            "projection": "ordered environment_id values",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m04_product_capabilities_environment_capabilities": {
            "authority_classification": "signed product environment authority",
            "consumers": [
                "resolve_m04_environment_ids.product_capabilities_environment_capabilities"
            ],
            "content_fingerprint_sha256": "c012c2b9793e58b9f52b81aa9eb90e142f33b6f73ab23c0e4d210ed053c5b1dc",
            "expected_result_type": "object",
            "json_pointer": "/ProductCapabilities/current_edition_capability_policy/environment_capabilities",
            "projection": "ordered object keys",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m04_product_capability_policy": {
            "authority_classification": "product capability grant authority",
            "consumers": [
                "resolve_m04_current_product_capability_policy.product_capability_policy",
                "resolve_m04_product_capabilities_projection",
            ],
            "content_fingerprint_sha256": "b18334a5589d077b73e9785c2fb85283cc456e158cd6fdafbf4af3c6c54066da",
            "expected_result_type": "object",
            "json_pointer": "/ProductCapabilities/current_edition_capability_policy",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m04_signed_payload_policy": {
            "authority_classification": "signed edition authority",
            "consumers": ["resolve_m04_current_product_capability_policy.signed_payload_policy"],
            "content_fingerprint_sha256": "7aae83f097f31226715512641da00316aa4efed1e74f34ba84fd25c76c2a7862",
            "expected_result_type": "object",
            "json_pointer": "/current_edition_signed_payload_policy",
            "source_contract": "environment_and_product_capabilities.json",
        },
        "m05_account_operability_policy": {
            "authority_classification": "operation authority",
            "consumers": ["resolve_m05_account_operability_policy"],
            "content_fingerprint_sha256": "83026ad96253eea2105229d6f3aa478964340f4626a6b6e386badf8022fd72b1",
            "expected_result_type": "object",
            "json_pointer": "/current_edition_account_operability_policy",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_account_snapshot_hash": {
            "authority_classification": "lineage authority",
            "consumers": ["resolve_hash_definition.account_capability_snapshot"],
            "content_fingerprint_sha256": "afa575e89ad8672adf68698e842c9e2fccb93b657246760f567f7bc57cfeb70e",
            "expected_result_type": "object",
            "json_pointer": "/account_capability_snapshot_contract/content_hash_definition",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_allowed_pairs": {
            "authority_classification": "structural pairing authority",
            "consumers": ["resolve_instrument_projection_contract.allowed_pairs"],
            "content_fingerprint_sha256": "3904c78c08f7ac68f632db1e605417dac64e21cba9f734e0830bb2864ab77936",
            "expected_result_type": "object",
            "json_pointer": "/allowed_market_instrument_type_pairs",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_asset_reference_contract": {
            "authority_classification": "asset identity authority",
            "consumers": ["resolve_instrument_projection_contract.asset_reference_contract"],
            "content_fingerprint_sha256": "837e1452a60de230d0ca091a7e2c05308ff41d961499f7800d350d7c9b3682ae",
            "expected_result_type": "object",
            "json_pointer": "/asset_reference_contract",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_catalog_hash": {
            "authority_classification": "lineage authority",
            "consumers": ["resolve_hash_definition.instrument_catalog_snapshot"],
            "content_fingerprint_sha256": "42ba26a65d7c9440842a3ba3700b4fc0c1e8606697061a440d46066bf0e40c53",
            "expected_result_type": "object",
            "json_pointer": "/instrument_catalog_snapshot_contract/content_hash_definition",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_decimal_policy": {
            "authority_classification": "numeric authority",
            "consumers": ["resolve_instrument_projection_contract.decimal_policy"],
            "content_fingerprint_sha256": "e526f728e0075a3a27380a87a213eb5e5f074d90cc4b63a99b48485957070b48",
            "expected_result_type": "object",
            "json_pointer": "/decimal_policy",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_derivative_rules": {
            "authority_classification": "derivative authority",
            "consumers": ["resolve_instrument_projection_contract.derivative_rules"],
            "content_fingerprint_sha256": "d4d4f6dd2ec09b5feb87a967fb3d79d4e74243049dc6e0202450ecb55d9868ee",
            "expected_result_type": "object",
            "json_pointer": "/derivative_consistency_rules",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_environment_registry": {
            "authority_classification": "environment authority",
            "consumers": [
                "resolve_m04_environment_ids.m05_environment_registry",
                "validate_canonical_enum_binding_manifest.environment_registry",
            ],
            "content_fingerprint_sha256": "c2182111523b676163dda381a902ed4ef238a9e1508484e1da1ce790e83cf0d9",
            "expected_result_type": "array",
            "json_pointer": "/environment_registry",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_exchange_registry_entries": {
            "authority_classification": "exchange identity authority",
            "consumers": ["canonical_exchange_entries"],
            "content_fingerprint_sha256": "90622ea826a2b95342155ab27933bc79d15fa3b2390a2086a9031e129641de13",
            "expected_result_type": "array",
            "json_pointer": "/exchange_registry_contract/entries",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_instrument_record_fields": {
            "authority_classification": "schema authority",
            "consumers": ["resolve_instrument_projection_contract.record_fields"],
            "content_fingerprint_sha256": "4c10117c28505bc7b546126690f3f3ef6cbc15c47815a47e016bd8c646b71409",
            "expected_result_type": "array",
            "json_pointer": "/instrument_contract/record_fields",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_instrument_trading_statuses": {
            "authority_classification": "instrument status authority",
            "consumers": ["validate_canonical_enum_binding_manifest.instrument_trading_statuses"],
            "content_fingerprint_sha256": "fffc70c35a4aa8440c884815fd6f51b67e4e76b76ff74ec5e1abc3b35ccab206",
            "expected_result_type": "array",
            "json_pointer": "/instrument_contract/trading_statuses",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_instrument_types": {
            "authority_classification": "instrument authority",
            "consumers": ["validate_canonical_enum_binding_manifest.instrument_types"],
            "content_fingerprint_sha256": "e99ba1e3af1a3b5771add15b7d0ab0659c0b24a1bd855c04c0d5c27f2d2831f8",
            "expected_result_type": "array",
            "json_pointer": "/instrument_type_registry",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_market_types": {
            "authority_classification": "market authority",
            "consumers": ["validate_canonical_enum_binding_manifest.market_types"],
            "content_fingerprint_sha256": "f9f3ddd7226000e3d8998868a218adb2fa44127623df1bb39d486c4128129ac2",
            "expected_result_type": "array",
            "json_pointer": "/market_type_registry",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_secure_store_grammar": {
            "authority_classification": "credential locator authority",
            "consumers": ["resolve_m05_secure_store_grammar"],
            "content_fingerprint_sha256": "1f4ae0b510ed548d3a3662c02ede1e43c69c47f371be353bb00a20fbadf0656a",
            "expected_result_type": "object",
            "json_pointer": "/credential_profile_contract/secure_store_reference_grammar",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
        "m05_universe_hash": {
            "authority_classification": "lineage authority",
            "consumers": ["resolve_hash_definition.trading_universe"],
            "content_fingerprint_sha256": "09ee36059a9ebd6be6c6165b8bfcc0f9a0abca185e8a5b6ad2d0cd668cdecb8a",
            "expected_result_type": "object",
            "json_pointer": "/trading_universe_contract/content_hash_definition",
            "source_contract": "exchange_accounts_and_instruments.json",
        },
    }
)

EXPECTED_CANONICAL_DEPENDENCY_CONSUMER_BINDINGS = deep_freeze(
    {
        "ProductCapabilitiesProjection.allowed_operations": "m04_capability_ids",
        "canonical_exchange_entries": "m05_exchange_registry_entries",
        "resolve_hash_definition.account_capability_snapshot": "m05_account_snapshot_hash",
        "resolve_hash_definition.instrument_catalog_snapshot": "m05_catalog_hash",
        "resolve_hash_definition.trading_universe": "m05_universe_hash",
        "resolve_instrument_projection_contract.allowed_pairs": "m05_allowed_pairs",
        "resolve_instrument_projection_contract.asset_reference_contract": "m05_asset_reference_contract",
        "resolve_instrument_projection_contract.decimal_policy": "m05_decimal_policy",
        "resolve_instrument_projection_contract.derivative_rules": "m05_derivative_rules",
        "resolve_instrument_projection_contract.record_fields": "m05_instrument_record_fields",
        "resolve_m04_current_product_capability_policy.current_product_edition": "m04_current_product_edition",
        "resolve_m04_current_product_capability_policy.product_capability_policy": "m04_product_capability_policy",
        "resolve_m04_current_product_capability_policy.signed_payload_policy": "m04_signed_payload_policy",
        "resolve_m04_environment_ids.current_edition_environment_capabilities": "m04_current_edition_environment_capabilities",
        "resolve_m04_environment_ids.execution_environments": "m04_execution_environments",
        "resolve_m04_environment_ids.m05_environment_registry": "m05_environment_registry",
        "resolve_m04_environment_ids.product_capabilities_environment_capabilities": "m04_product_capabilities_environment_capabilities",
        "resolve_m04_product_capabilities_projection": "m04_product_capability_policy",
        "resolve_m05_account_operability_policy": "m05_account_operability_policy",
        "resolve_m05_secure_store_grammar": "m05_secure_store_grammar",
        "validate_canonical_enum_binding_manifest.environment_registry": "m05_environment_registry",
        "validate_canonical_enum_binding_manifest.instrument_trading_statuses": "m05_instrument_trading_statuses",
        "validate_canonical_enum_binding_manifest.instrument_types": "m05_instrument_types",
        "validate_canonical_enum_binding_manifest.market_types": "m05_market_types",
    }
)

EXPECTED_M06_LOCAL_AUTHORITY_POLICIES = deep_freeze(
    {
        "account_capability_snapshot_policy": {
            "authorization_dependency_environments": ["TESTNET"],
            "freshness_boundary": "age_seconds <= max_age_seconds is fresh",
            "max_age_seconds_by_environment": {"TESTNET": 300},
            "supported_instrument_types": {
                "canonical_registry_ref": "m05_instrument_types",
                "empty_allowed": True,
                "m05_structural_semantics": "empty "
                "canonical "
                "set "
                "is "
                "structurally "
                "valid "
                "and "
                "blocks "
                "every "
                "non-empty "
                "Universe "
                "at "
                "TESTNET "
                "operability",
                "missing_authority_denial": "CAPABILITY_SNAPSHOT_BLOCKED",
                "testnet_subset_rule": "set(Universe "
                "Instrument.instrument_type) "
                "<= "
                "set(AccountCapabilitySnapshot.supported_instrument_types)",
                "unique": True,
            },
        },
        "authorization_dependencies_by_environment": {
            "PAPER": ["PRODUCT_CAPABILITIES"],
            "TESTNET": [
                "ACCOUNT_ACTIVE",
                "ACCOUNT_CAPABILITY_SNAPSHOT_VALID",
                "CREDENTIAL_PROFILE_ACTIVE",
                "ORDER_ENTRY_PURPOSE",
                "PLACE_ORDERS_PERMISSION",
                "PRODUCT_CAPABILITIES",
            ],
        },
        "current_edition_execution_pair_policy": {
            "allowed_pairs": {"SPOT": ["SPOT_PAIR"]},
            "bind_checks_declaration_not_current_operability": True,
            "policy_id": "M0_6_CURRENT_EDITION_EXECUTION_PAIRS",
            "structural_projection_is_not_authority": True,
            "unsupported_pair_denial": "INSTRUMENT_SCOPE_MISMATCH",
            "used_by": ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"],
        },
        "endpoint_class_registry": {
            "LIVE_PRIVATE_DATA": {
                "access_scope": "PRIVATE",
                "allowed_route_kinds": ["MARKET_DATA", "EXECUTION"],
                "environment": "LIVE",
            },
            "LIVE_PUBLIC_DATA": {
                "access_scope": "PUBLIC",
                "allowed_route_kinds": ["MARKET_DATA"],
                "environment": "LIVE",
            },
            "PAPER_PUBLIC_DATA": {
                "access_scope": "PUBLIC",
                "allowed_route_kinds": ["MARKET_DATA"],
                "environment": "PAPER",
            },
            "PAPER_SIMULATION": {
                "access_scope": "LOCAL_SIMULATION",
                "allowed_route_kinds": ["EXECUTION"],
                "environment": "PAPER",
            },
            "TESTNET_PRIVATE_DATA": {
                "access_scope": "PRIVATE",
                "allowed_route_kinds": ["MARKET_DATA", "EXECUTION"],
                "environment": "TESTNET",
            },
            "TESTNET_PUBLIC_DATA": {
                "access_scope": "PUBLIC",
                "allowed_route_kinds": ["MARKET_DATA"],
                "environment": "TESTNET",
            },
        },
        "execution_readiness_max_age_seconds": 30,
    }
)


def _validate_dependency_consumer_bindings():
    manifest_bindings = CONTRACT["canonical_dependency_consumer_bindings"]
    if (
        type(manifest_bindings) is not dict
        or deep_freeze(manifest_bindings) != EXPECTED_CANONICAL_DEPENDENCY_CONSUMER_BINDINGS
    ):
        raise TypeError("canonical dependency consumer bindings")
    grouped = {dependency_id: [] for dependency_id in EXPECTED_CANONICAL_DEPENDENCIES}
    for consumer, dependency_id in EXPECTED_CANONICAL_DEPENDENCY_CONSUMER_BINDINGS.items():
        if dependency_id not in grouped:
            raise TypeError(consumer)
        grouped[dependency_id].append(consumer)
    for dependency_id, expected in EXPECTED_CANONICAL_DEPENDENCIES.items():
        if tuple(grouped[dependency_id]) != expected["consumers"]:
            raise TypeError(f"dependency consumers: {dependency_id}")


def resolve_canonical_dependency(dependency_id):
    if dependency_id not in EXPECTED_CANONICAL_DEPENDENCIES:
        raise KeyError("unknown canonical dependency")
    expected = EXPECTED_CANONICAL_DEPENDENCIES[dependency_id]
    manifest = CONTRACT["canonical_dependency_root_manifest"]
    if type(manifest) is not dict or set(manifest) != set(EXPECTED_CANONICAL_DEPENDENCIES):
        raise TypeError("canonical dependency root inventory")
    entry = manifest.get(dependency_id)
    if type(entry) is not dict or deep_freeze(entry) != expected:
        raise TypeError(dependency_id)
    _validate_dependency_consumer_bindings()
    expected_type = {"array": list, "object": dict}[expected["expected_result_type"]]
    actual = resolve_canonical_pointer(
        {"contract": expected["source_contract"], "json_pointer": expected["json_pointer"]},
        expected_type,
    )
    if canonical_fingerprint(actual) != expected["content_fingerprint_sha256"]:
        raise ValueError(dependency_id)
    return deep_freeze(copy.deepcopy(actual))


def _validate_specialized_dependency_attestations():
    exchange_dependency = EXPECTED_CANONICAL_DEPENDENCIES["m05_exchange_registry_entries"]
    exchange_ref = CONTRACT["canonical_cross_contract_registry_refs"].get(
        "m05_exchange_registry_entries"
    )
    if exchange_ref != {
        "contract": exchange_dependency["source_contract"],
        "json_pointer": exchange_dependency["json_pointer"],
        "expected_result_type": exchange_dependency["expected_result_type"],
        "content_fingerprint_sha256": exchange_dependency["content_fingerprint_sha256"],
        "dependency_id": "m05_exchange_registry_entries",
    }:
        raise TypeError("exchange specialized attestation")
    expected_hashes = {
        "account_capability_snapshot": "m05_account_snapshot_hash",
        "instrument_catalog_snapshot": "m05_catalog_hash",
        "trading_universe": "m05_universe_hash",
    }
    references = CONTRACT["canonical_hash_definition_refs"]
    if set(references) != set(expected_hashes):
        raise TypeError("hash specialized attestations")
    for name, dependency_id in expected_hashes.items():
        dependency = EXPECTED_CANONICAL_DEPENDENCIES[dependency_id]
        if references[name] != {
            "contract": dependency["source_contract"],
            "json_pointer": dependency["json_pointer"],
            "expected_result_type": dependency["expected_result_type"],
            "content_fingerprint_sha256": dependency["content_fingerprint_sha256"],
            "dependency_id": dependency_id,
        }:
            raise TypeError(name)


def validate_canonical_dependency_roots():
    for dependency_id in EXPECTED_CANONICAL_DEPENDENCIES:
        resolve_canonical_dependency(dependency_id)
    _validate_specialized_dependency_attestations()
    return CONTRACT["canonical_dependency_root_manifest"]


def resolve_m04_environment_ids():
    validate_canonical_dependency_roots()
    records = resolve_canonical_dependency("m04_execution_environments")
    required = {
        "environment_id",
        "external_side_effect_class",
        "credential_scope",
        "execution_adapter_class",
        "private_exchange_connection_policy",
        "real_funds_possible",
        "current_edition_selectable",
        "current_edition_executable",
        "endpoint_classes",
    }
    if any(not isinstance(item, MappingProxyType) or set(item) != required for item in records):
        raise TypeError("M0.4 execution environment shape")
    environment_ids = [item["environment_id"] for item in records]
    if (
        any(type(value) is not str or not value for value in environment_ids)
        or len(environment_ids) != len(set(environment_ids))
        or any(type(item["endpoint_classes"]) is not tuple for item in records)
    ):
        raise TypeError("M0.4 environment IDs")
    edition_keys = list(
        resolve_canonical_dependency("m04_current_edition_environment_capabilities")
    )
    product_keys = list(
        resolve_canonical_dependency("m04_product_capabilities_environment_capabilities")
    )
    m05_ids = list(resolve_canonical_dependency("m05_environment_registry"))
    if (
        environment_ids != edition_keys
        or environment_ids != product_keys
        or environment_ids != m05_ids
    ):
        raise ValueError("M0.4/M0.5 environment divergence")
    return tuple(environment_ids)


def resolve_m05_account_operability_policy():
    validate_canonical_dependency_roots()
    policy = resolve_canonical_dependency("m05_account_operability_policy")
    required = {
        "policy_id",
        "validating_authority",
        "live_operational_use_denial_code",
        "operational_lifecycle_states",
        "operational_connection_states",
        "operational_authorization_states",
        "live_forbidden_states",
        "live_forbidden_side_effects",
        "legacy_live_record_allowed_shape",
        "rules",
        "maintenance_operations",
        "normal_account_operations_block_live",
        "default_success_fallthrough_forbidden",
    }
    if (
        set(policy) != required
        or policy["policy_id"] != "M0_5_CURRENT_EDITION_ACCOUNT_OPERABILITY"
        or policy["validating_authority"] != "CoreHost"
        or policy["default_success_fallthrough_forbidden"] is not True
        or any(
            type(policy[name]) is not tuple or not policy[name]
            for name in (
                "operational_lifecycle_states",
                "operational_connection_states",
                "operational_authorization_states",
                "live_forbidden_side_effects",
                "rules",
                "maintenance_operations",
                "normal_account_operations_block_live",
            )
        )
        or not isinstance(policy["live_forbidden_states"], MappingProxyType)
        or not isinstance(policy["legacy_live_record_allowed_shape"], MappingProxyType)
    ):
        raise TypeError("M0.5 account operability policy")
    return deep_freeze(policy)


def resolve_m05_secure_store_grammar():
    validate_canonical_dependency_roots()
    grammar = resolve_canonical_dependency("m05_secure_store_grammar")
    if (
        set(grammar) != {"prefix", "locator", "forbidden_characters", "forbidden_payload_markers"}
        or grammar["prefix"] != "secure-store://"
        or grammar["locator"] != "non-empty opaque string"
        or grammar["forbidden_characters"] != ("whitespace", "?", "#", "=")
        or grammar["forbidden_payload_markers"]
        != (
            "api_key",
            "apikey",
            "secret",
            "password",
            "token",
            "private_key",
            "credential_value",
            "plaintext",
        )
    ):
        raise TypeError("M0.5 secure-store grammar")
    return deep_freeze(grammar)


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


def validate_executable_operation_protocol():
    expected = EXPECTED_EXECUTABLE_OPERATION_PROTOCOL
    actual = {
        "operation_registry": CONTRACT["operation_registry"],
        "operation_validator_call_graph": CONTRACT["operation_validator_call_graph"],
        "validator_denial_registry": CONTRACT["validator_denial_registry"],
        "allowed_denials_by_operation": CONTRACT["allowed_denials_by_operation"],
        "denial_code_registry": CONTRACT["denial_code_registry"],
        "success_event_by_operation": CONTRACT["success_event_by_operation"],
        "denial_event_by_operation": CONTRACT["denial_event_by_operation"],
        "special_events": CONTRACT["audit_event_contract"]["special_events"],
        "forbidden_operations": CONTRACT["forbidden_operations"],
    }
    if deep_freeze(actual) != expected:
        raise TypeError("executable operation protocol")
    operations = set(expected["operation_registry"])
    if (
        set(expected["operation_validator_call_graph"]) != operations
        or set(expected["allowed_denials_by_operation"]) != operations
        or set(expected["success_event_by_operation"]) != operations
        or set(expected["denial_event_by_operation"]) != operations
    ):
        raise TypeError("operation protocol coverage")
    expected_validators = set(expected["validator_denial_registry"])
    if (
        set(REQUESTS) != operations
        or set(EXPECTED_VALIDATOR_IMPLEMENTATIONS) != expected_validators
        or any(
            name not in expected_validators
            for graph in expected["operation_validator_call_graph"].values()
            for name in graph
        )
    ):
        raise TypeError("executable protocol implementation bindings")
    for operation, entry in expected["operation_registry"].items():
        schema = entry["request_schema"]
        if deep_freeze(REQUESTS[operation]) != schema:
            raise TypeError(f"request schema binding: {operation}")
        if (
            schema["constants"]["authority"] != "CoreHost"
            or schema["constants"]["operation"] != operation
            or tuple(expected["operation_validator_call_graph"][operation])[:2]
            != ("validate_request", "validate_context")
            or len(expected["operation_validator_call_graph"][operation])
            != len(set(expected["operation_validator_call_graph"][operation]))
        ):
            raise TypeError(operation)
    return expected


def resolve_m06_local_authority_policy(policy_id):
    if policy_id not in EXPECTED_M06_LOCAL_AUTHORITY_POLICIES:
        raise KeyError("unknown M0.6 local authority policy")
    manifest = CONTRACT["m06_local_authority_policy_manifest"]
    if (
        type(manifest) is not dict
        or set(manifest) != set(EXPECTED_M06_LOCAL_AUTHORITY_POLICIES)
        or deep_freeze(manifest) != EXPECTED_M06_LOCAL_AUTHORITY_POLICIES
    ):
        raise TypeError("M0.6 local authority policy manifest")
    raw = (
        CONTRACT["execution_route_contract"]["execution_readiness_max_age_seconds"]
        if policy_id == "execution_readiness_max_age_seconds"
        else CONTRACT[policy_id]
    )
    if deep_freeze(raw) != EXPECTED_M06_LOCAL_AUTHORITY_POLICIES[policy_id]:
        raise TypeError(policy_id)
    return deep_freeze(copy.deepcopy(raw))


def resolve_m04_current_product_capability_policy():
    edition = resolve_canonical_dependency("m04_current_product_edition")
    product = resolve_canonical_dependency("m04_product_capability_policy")
    signed = resolve_canonical_dependency("m04_signed_payload_policy")
    edition_id = "CRYPTOHUNTER_TESTNET_EDITION"
    capability_set = (
        "LIVE_VISIBLE_LOCKED_ONLY",
        "PAPER_LOCAL_SIMULATION",
        "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
    )
    if (
        edition["edition_id"] != edition_id
        or product["edition_id"] != edition_id
        or signed["edition_id"] != edition_id
        or product["capability_set"] != capability_set
        or signed["capability_set"] != capability_set
        or signed["environment_capabilities_ref"]
        != "ProductCapabilities.current_edition_capability_policy.environment_capabilities"
        or product["environment_capabilities"]
        != resolve_canonical_dependency("m04_product_capabilities_environment_capabilities")
        or product["feature_flags"]
        != deep_freeze(
            {"live_activation_dialog": True, "live_execution": False, "testnet_execution": True}
        )
        or signed["feature_flags_ref"]
        != "ProductCapabilities.current_edition_capability_policy.feature_flags"
        or product["source"] != "signed_build_resource_after_validation"
        or signed["source"] != product["source"]
        or product["fail_closed_policy"] != "SAFE_LOCAL_ONLY"
        or signed["fail_closed_policy"] != product["fail_closed_policy"]
        or product["live_allowed_in_current_edition"] is not False
        or product["capability_set_hash_required"] is not True
    ):
        raise TypeError("M0.4 current ProductCapabilities policy")
    return product


def resolve_m04_product_capabilities_projection(environment):
    policy = resolve_m04_current_product_capability_policy()
    projections = {
        "PAPER": {
            "environment": "PAPER",
            "execution_enabled": True,
            "allowed_operations": ["PAPER_LOCAL_SIMULATION"],
            "edition": "CRYPTOHUNTER_TESTNET_EDITION",
        },
        "TESTNET": {
            "environment": "TESTNET",
            "execution_enabled": True,
            "allowed_operations": ["TESTNET_PRIVATE_EXECUTION_AFTER_READINESS"],
            "edition": "CRYPTOHUNTER_TESTNET_EDITION",
        },
        "LIVE": {
            "environment": "LIVE",
            "execution_enabled": False,
            "allowed_operations": ["LIVE_VISIBLE_LOCKED_ONLY"],
            "edition": "CRYPTOHUNTER_TESTNET_EDITION",
        },
    }
    if environment not in projections or environment not in policy["environment_capabilities"]:
        raise KeyError(environment)
    return deep_freeze(projections[environment])


def resolve_current_edition_execution_pair_policy():
    policy = resolve_m06_local_authority_policy("current_edition_execution_pair_policy")
    expected = {
        "policy_id": "M0_6_CURRENT_EDITION_EXECUTION_PAIRS",
        "allowed_pairs": {"SPOT": ["SPOT_PAIR"]},
        "used_by": ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"],
        "unsupported_pair_denial": "INSTRUMENT_SCOPE_MISMATCH",
        "structural_projection_is_not_authority": True,
        "bind_checks_declaration_not_current_operability": True,
    }
    if not isinstance(policy, MappingProxyType) or policy != deep_freeze(expected):
        raise TypeError("current-edition execution pair policy")
    return policy


def validate_canonical_projection_binding_manifest():
    manifest = CONTRACT["canonical_projection_binding_manifest"]
    expected = {
        "environment_registry": ("/environment_registry", list),
        "market_type_registry": ("/market_type_registry", list),
        "instrument_type_registry": ("/instrument_type_registry", list),
        "allowed_market_instrument_type_pairs": (
            "/allowed_market_instrument_type_pairs",
            dict,
        ),
        "trading_statuses": ("/instrument_contract/trading_statuses", list),
        "record_fields": ("/instrument_contract/record_fields", list),
        "asset_reference_contract": ("/asset_reference_contract", dict),
        "decimal_policy": ("/decimal_policy", dict),
        "derivative_consistency_rules": ("/derivative_consistency_rules", dict),
    }
    if type(manifest) is not dict or set(manifest) != set(expected):
        raise TypeError("canonical projection binding manifest")
    expected_consumers = {
        "environment_registry": [
            "InstrumentProjection.environment",
            "CredentialProfileProjection.environment_scope",
            "MarketDataRoute.environment",
            "ExecutionRoute.environment",
            "ExchangeAccountProjection.environment",
            "InstrumentCatalogProjection.environment",
            "AccountCapabilitySnapshotProjection.environment",
            "TrustedExternalIdentityProjection.environment",
        ],
        "market_type_registry": [
            "InstrumentProjection.market_type",
            "ExchangeAccountProjection.market_type",
            "InstrumentCatalogProjection.market_type",
            "AccountCapabilitySnapshotProjection.market_type",
            "TrustedExternalIdentityProjection.market_type",
            "MarketDataRoute.market_type",
            "ExecutionRoute.market_type",
        ],
        "instrument_type_registry": [
            "InstrumentProjection.instrument_type",
            "AccountCapabilitySnapshotProjection.supported_instrument_types",
            "ExecutionRoute.supported_instrument_types",
        ],
        "allowed_market_instrument_type_pairs": [
            "resolve_instrument_projection_contract",
            "validate_instrument_record",
            "validate_snapshot_record",
        ],
        "trading_statuses": ["InstrumentProjection.trading_status"],
        "record_fields": ["InstrumentProjection.exact_fields"],
        "asset_reference_contract": ["validate_instrument_record"],
        "decimal_policy": ["validate_instrument_record.canonical_decimal"],
        "derivative_consistency_rules": ["validate_instrument_record"],
    }
    references = SCHEMAS["InstrumentProjection"]["canonical_projection_refs"]
    for name, (pointer, result_type) in expected.items():
        binding = manifest[name]
        if (
            type(binding) is not dict
            or set(binding)
            != {
                "contract",
                "json_pointer",
                "expected_result_type",
                "content_fingerprint_sha256",
                "consumers",
            }
            or binding["contract"] != "exchange_accounts_and_instruments.json"
            or binding["json_pointer"] != pointer
            or binding["expected_result_type"] != ("array" if result_type is list else "object")
            or binding["consumers"] != expected_consumers[name]
            or references.get(name)
            != {"contract": binding["contract"], "json_pointer": binding["json_pointer"]}
        ):
            raise TypeError(name)
        value = resolve_canonical_pointer(binding, result_type)
        if canonical_fingerprint(value) != binding["content_fingerprint_sha256"]:
            raise ValueError(name)
    return manifest


def collect_external_canonical_enum_consumers():
    """Collect observed external bindings without defining which consumers are required."""
    consumers = {}
    references = {
        **CONTRACT["canonical_scalar_enum_registry_refs"],
        **CONTRACT["canonical_array_enum_registry_refs"],
    }

    def add(key, value):
        if key in consumers:
            raise TypeError(f"dual external canonical enum binding: {key}")
        consumers[key] = value

    for schema_name, schema in SCHEMAS.items():
        for field, reference in schema.get("enum_canonical_pointer_ref", {}).items():
            add(
                f"{schema_name}.{field}",
                (schema_name, field, "enum_canonical_pointer_ref", reference, None),
            )
        for field, reference in schema.get("derived_canonical_projection_ref", {}).items():
            add(
                f"{schema_name}.{field}",
                (schema_name, field, "derived_canonical_projection", reference, None),
            )
        for field, reference_name in schema.get("enum_registry_ref", {}).items():
            add(
                f"{schema_name}.{field}",
                (
                    schema_name,
                    field,
                    "enum_registry_ref",
                    references.get(reference_name),
                    reference_name,
                ),
            )
        for field, policy in schema.get("array_policy", {}).items():
            if "item_registry_ref" in policy:
                reference_name = policy["item_registry_ref"]
                add(
                    f"{schema_name}.{field}",
                    (
                        schema_name,
                        field,
                        "item_registry_ref",
                        references.get(reference_name),
                        reference_name,
                    ),
                )
    return consumers


def validate_canonical_enum_binding_manifest():
    manifest = CONTRACT["canonical_enum_binding_manifest"]
    consumers = collect_external_canonical_enum_consumers()
    expected = EXPECTED_EXTERNAL_CANONICAL_ENUM_CONSUMERS
    if (
        type(manifest) is not dict
        or set(consumers) != set(manifest)
        or set(manifest) != set(expected)
    ):
        raise TypeError("canonical enum binding manifest completeness")
    exact_entry_fields = {
        "schema",
        "field",
        "field_type",
        "binding_kind",
        "nullable",
        "exact_fields_membership",
        "array_policy",
        "contract",
        "json_pointer",
        "registry_ref",
        "expected_result_type",
        "content_fingerprint_sha256",
    }
    for key, immutable in expected.items():
        schema_name, field, kind, reference, registry_ref = consumers[key]
        binding = manifest[key]
        schema = SCHEMAS[schema_name]
        if (
            type(binding) is not dict
            or set(binding)
            != exact_entry_fields
            | ({"projection"} if kind == "derived_canonical_projection" else set())
            or any(binding[name] != value for name, value in immutable.items())
            or immutable["schema"] != schema_name
            or immutable["field"] != field
            or immutable["binding_kind"] != kind
            or immutable["registry_ref"] != registry_ref
            or type(reference) is not dict
            or reference.get("contract") != immutable["contract"]
            or reference.get("json_pointer") != immutable["json_pointer"]
            or binding["expected_result_type"] != "array"
            or schema["exact_fields"].count(field) != 1
            or schema["field_types"].get(field) != immutable["field_type"]
            or (field in schema["nullable_fields"]) != immutable["nullable"]
            or field in schema.get("enum_registry", {})
        ):
            raise TypeError(key)
        mechanisms = sum(
            (
                field in schema.get("enum_canonical_pointer_ref", {}),
                field in schema.get("enum_registry_ref", {}),
                field in schema.get("derived_canonical_projection_ref", {}),
                "item_registry_ref" in schema.get("array_policy", {}).get(field, {}),
            )
        )
        if mechanisms != 1:
            raise TypeError(f"binding cardinality: {key}")
        if kind == "item_registry_ref":
            policy = schema.get("array_policy", {}).get(field)
            if (
                type(policy) is not dict
                or policy != immutable["array_policy"]
                or policy != binding["array_policy"]
                or "item_registry" in policy
            ):
                raise TypeError(key)
        elif binding["array_policy"] is not None:
            raise TypeError(key)
        canonical = resolve_canonical_pointer(binding, list)
        identity = (immutable["contract"], immutable["json_pointer"])
        immutable_fingerprint = IMMUTABLE_CANONICAL_REGISTRY_FINGERPRINTS.get(identity)
        if kind == "derived_canonical_projection":
            immutable_fingerprint = EXPECTED_CANONICAL_DEPENDENCIES["m04_execution_environments"][
                "content_fingerprint_sha256"
            ]
        actual_fingerprint = canonical_fingerprint(canonical)
        if (
            immutable_fingerprint is None
            or actual_fingerprint != immutable_fingerprint
            or binding["content_fingerprint_sha256"] != immutable_fingerprint
        ):
            raise ValueError(key)
    projection_refs = SCHEMAS["InstrumentProjection"]["canonical_projection_refs"]
    for field, projection_name in {
        "environment": "environment_registry",
        "market_type": "market_type_registry",
        "instrument_type": "instrument_type_registry",
        "trading_status": "trading_statuses",
    }.items():
        binding = manifest[f"InstrumentProjection.{field}"]
        if resolve_canonical_pointer(binding, list) != resolve_canonical_pointer(
            projection_refs[projection_name], list
        ):
            raise ValueError("InstrumentProjection scalar/projection mismatch")
    return manifest


def resolve_instrument_projection_contract():
    """Resolve and validate every M0.5 binding declared by InstrumentProjection."""
    validate_canonical_projection_binding_manifest()
    schema = SCHEMAS["InstrumentProjection"]
    references = schema["canonical_projection_refs"]
    expected = {
        "environment_registry": list,
        "market_type_registry": list,
        "instrument_type_registry": list,
        "allowed_market_instrument_type_pairs": dict,
        "trading_statuses": list,
        "record_fields": list,
        "asset_reference_contract": dict,
        "decimal_policy": dict,
        "derivative_consistency_rules": dict,
    }
    if set(references) != set(expected):
        raise TypeError("InstrumentProjection canonical refs")
    resolved = {
        name: resolve_canonical_pointer(references[name], result_type)
        for name, result_type in expected.items()
    }

    for name in (
        "environment_registry",
        "market_type_registry",
        "instrument_type_registry",
        "trading_statuses",
        "record_fields",
    ):
        values = resolved[name]
        if (
            not values
            or any(type(value) is not str or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise TypeError(name)

    markets = resolved["market_type_registry"]
    instrument_types = resolved["instrument_type_registry"]
    pairs = resolved["allowed_market_instrument_type_pairs"]
    if set(pairs) != set(markets) or any(
        type(values) is not list
        or not values
        or any(type(value) is not str or value not in instrument_types for value in values)
        or len(values) != len(set(values))
        for values in pairs.values()
    ):
        raise TypeError("allowed pairs")
    if sorted(value for values in pairs.values() for value in values) != sorted(instrument_types):
        raise TypeError("allowed pairs coverage")
    if pairs != {
        "SPOT": ["SPOT_PAIR"],
        "MARGIN": ["MARGIN_PAIR"],
        "PERPETUAL": ["PERPETUAL_CONTRACT"],
        "DELIVERY_FUTURES": ["DELIVERY_FUTURE"],
        "OPTIONS": ["OPTION"],
    }:
        raise TypeError("allowed pair semantics")

    asset = resolved["asset_reference_contract"]
    if set(asset) != {"value_objects", "fields", "mapping_statuses", "rules"}:
        raise TypeError("asset reference contract")
    for field in ("value_objects", "fields", "mapping_statuses", "rules"):
        values = asset[field]
        if (
            type(values) is not list
            or not values
            or any(type(value) is not str or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise TypeError("asset reference contract")
    if set(asset["fields"]) != {
        "venue_asset_code",
        "canonical_display_code",
        "asset_namespace",
        "mapping_status",
    }:
        raise TypeError("asset reference fields")
    if set(asset["mapping_statuses"]) != {"EXACT", "EXPLICIT_ALIAS", "AMBIGUOUS", "UNKNOWN"}:
        raise TypeError("asset mapping statuses")

    decimal_policy = resolved["decimal_policy"]
    if set(decimal_policy) != {
        "format",
        "regex",
        "zero",
        "trailing_zero_policy",
        "forbidden",
        "rules",
    }:
        raise TypeError("decimal policy")
    if any(
        type(decimal_policy[field]) is not str or not decimal_policy[field]
        for field in ("format", "regex", "zero", "trailing_zero_policy")
    ) or any(
        type(decimal_policy[field]) is not list
        or not decimal_policy[field]
        or any(type(value) is not str or not value for value in decimal_policy[field])
        for field in ("forbidden", "rules")
    ):
        raise TypeError("decimal policy")
    re.compile(decimal_policy["regex"])
    if decimal_policy["regex"] != r"^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$":
        raise TypeError("decimal regex")

    derivatives = resolved["derivative_consistency_rules"]
    if set(derivatives) != set(instrument_types) or any(
        type(rule) is not dict for rule in derivatives.values()
    ):
        raise TypeError("derivative consistency rules")
    exact_derivative_keys = {
        "SPOT_PAIR": {
            "settlement_asset_reference",
            "contract_size",
            "expiry_at_utc",
            "strike_price",
            "option_side",
        },
        "MARGIN_PAIR": {
            "settlement_asset_reference",
            "contract_size",
            "expiry_at_utc",
            "strike_price",
            "option_side",
        },
        "PERPETUAL_CONTRACT": {
            "contract_size",
            "settlement_asset_reference",
            "expiry_at_utc",
            "strike_price",
            "option_side",
            "derivative_settlement_type",
        },
        "DELIVERY_FUTURE": {
            "contract_size",
            "settlement_asset_reference",
            "expiry_at_utc",
            "strike_price",
            "option_side",
        },
        "OPTION": {
            "contract_size",
            "settlement_asset_reference",
            "expiry_at_utc",
            "strike_price",
            "option_side",
        },
    }
    if any(set(derivatives[name]) != keys for name, keys in exact_derivative_keys.items()):
        raise TypeError("derivative consistency rules")
    expected_derivatives = {
        "SPOT_PAIR": {
            "settlement_asset_reference": "null_or_quote",
            "contract_size": None,
            "expiry_at_utc": None,
            "strike_price": None,
            "option_side": None,
        },
        "MARGIN_PAIR": {
            "settlement_asset_reference": "null_or_quote",
            "contract_size": None,
            "expiry_at_utc": None,
            "strike_price": None,
            "option_side": None,
        },
        "PERPETUAL_CONTRACT": {
            "contract_size": "required",
            "settlement_asset_reference": "required",
            "expiry_at_utc": None,
            "strike_price": None,
            "option_side": None,
            "derivative_settlement_type": ["LINEAR", "INVERSE"],
        },
        "DELIVERY_FUTURE": {
            "contract_size": "required",
            "settlement_asset_reference": "required",
            "expiry_at_utc": "required",
            "strike_price": None,
            "option_side": None,
        },
        "OPTION": {
            "contract_size": "required",
            "settlement_asset_reference": "required",
            "expiry_at_utc": "required",
            "strike_price": "required",
            "option_side": ["CALL", "PUT"],
        },
    }
    if derivatives != expected_derivatives:
        raise TypeError("derivative consistency semantics")

    if set(schema["exact_fields"]) != set(resolved["record_fields"]):
        raise TypeError("InstrumentProjection record fields")
    limit_fields = {"min_quantity", "max_quantity", "min_notional", "max_notional"}
    if not limit_fields <= set(schema["nullable_fields"]) or any(
        schema["field_types"].get(field) != "string" for field in limit_fields
    ):
        raise TypeError("InstrumentProjection limit schema")
    return resolved


def canonical_exchange_entries():
    entries = resolve_canonical_dependency("m05_exchange_registry_entries")
    _validate_specialized_dependency_attestations()
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
        if not isinstance(entry, MappingProxyType) or set(entry) != required:
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
            for item in entries
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
            or type(aliases) is not tuple
            or len(aliases) != len(set(aliases))
            or any(type(alias) is not str or not alias for alias in aliases)
            or any(type(policy) is not str or policy not in allowed_policies for policy in policies)
            or any(
                type(values) is not tuple
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
    dependency_ids = {
        "account_capability_snapshot": "m05_account_snapshot_hash",
        "instrument_catalog_snapshot": "m05_catalog_hash",
        "trading_universe": "m05_universe_hash",
    }
    if reference_name not in dependency_ids:
        raise KeyError(reference_name)
    _validate_specialized_dependency_attestations()
    value = resolve_canonical_dependency(dependency_ids[reference_name])
    required = {
        "algorithm",
        "domain_separator",
        "input_fields",
        "canonicalization",
        "encoding",
        "digest_format",
    }
    if (
        not isinstance(value, MappingProxyType)
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
        pointer_reference = schema.get("enum_canonical_pointer_ref", {}).get(field)
        derived_reference = schema.get("derived_canonical_projection_ref", {}).get(field)
        if reference_name:
            allowed = resolve_canonical_registry(reference_name)
        elif pointer_reference:
            allowed = resolve_canonical_pointer(pointer_reference, list)
        elif derived_reference:
            if derived_reference.get("projection") != "ordered environment_id values":
                raise TypeError("derived canonical projection")
            allowed = resolve_m04_environment_ids()
        else:
            allowed = schema["enum_registry"].get(
                field, schema.get("nested_schemas", {}).get(field, {}).get("enum", [])
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
    projection_contract = resolve_instrument_projection_contract()
    catalogs = {**context["previous_catalogs_by_id"], **context["catalogs_by_id"]}
    catalog = catalogs.get(record.get("catalog_snapshot_id"))
    asset_fields = set(projection_contract["asset_reference_contract"]["fields"])
    mapping_statuses = set(projection_contract["asset_reference_contract"]["mapping_statuses"])

    def canonical_decimal(value, *, positive=False, nullable=False):
        if value is None:
            return nullable
        if type(value) is not str or not re.fullmatch(
            projection_contract["decimal_policy"]["regex"], value
        ):
            return False
        parsed = Decimal(value)
        return parsed > 0 if positive else parsed >= 0

    def valid_asset(value):
        return (
            type(value) is dict
            and set(value) == asset_fields
            and all(type(value[field]) is str and value[field] for field in asset_fields)
            and value["asset_namespace"] == record["exchange_id"]
            and value["mapping_status"] in mapping_statuses
            and value["mapping_status"] in {"EXACT", "EXPLICIT_ALIAS"}
        )

    if not validate_record("InstrumentProjection", record):
        return False
    if record["instrument_type"] not in projection_contract[
        "allowed_market_instrument_type_pairs"
    ].get(record["market_type"], []):
        return False
    if not all(
        valid_asset(record[field]) for field in ("base_asset_reference", "quote_asset_reference")
    ) or (
        record["settlement_asset_reference"] is not None
        and not valid_asset(record["settlement_asset_reference"])
    ):
        return False
    if (
        not all(
            canonical_decimal(record[field], positive=True)
            for field in ("price_tick", "quantity_step")
        )
        or not all(
            canonical_decimal(record[field], nullable=True)
            for field in ("min_quantity", "min_notional")
        )
        or not all(
            canonical_decimal(record[field], nullable=True)
            for field in ("max_quantity", "max_notional")
        )
    ):
        return False
    if any(
        record[maximum] is not None
        and (record[minimum] is None or Decimal(record[minimum]) > Decimal(record[maximum]))
        for minimum, maximum in (("min_quantity", "max_quantity"), ("min_notional", "max_notional"))
    ):
        return False

    instrument_type = record["instrument_type"]
    derivative_fields = (
        "contract_size",
        "contract_value_currency",
        "derivative_settlement_type",
        "expiry_at_utc",
        "strike_price",
        "option_side",
    )
    if instrument_type in {"SPOT_PAIR", "MARGIN_PAIR"}:
        if any(record[field] is not None for field in derivative_fields):
            return False
        settlement = record["settlement_asset_reference"]
        if (
            settlement is not None
            and settlement["venue_asset_code"]
            != record["quote_asset_reference"]["venue_asset_code"]
        ):
            return False
    else:
        if (
            not canonical_decimal(record["contract_size"], positive=True)
            or record["settlement_asset_reference"] is None
            or type(record["contract_value_currency"]) is not str
            or not record["contract_value_currency"]
            or record["derivative_settlement_type"] not in {"LINEAR", "INVERSE"}
        ):
            return False
        if instrument_type == "PERPETUAL_CONTRACT":
            if any(
                record[field] is not None
                for field in ("expiry_at_utc", "strike_price", "option_side")
            ):
                return False
        elif instrument_type == "DELIVERY_FUTURE":
            if (
                record["expiry_at_utc"] is None
                or record["strike_price"] is not None
                or record["option_side"] is not None
            ):
                return False
        elif instrument_type == "OPTION" and (
            record["expiry_at_utc"] is None
            or not canonical_decimal(record["strike_price"], positive=True)
            or record["option_side"] not in {"CALL", "PUT"}
        ):
            return False

    return bool(
        record_matches_exchange_registry(record)
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


def validate_global_instrument_identity(context):
    """Enforce both directions of canonical identity across current and complete history."""
    tuple_to_id = {}
    id_to_tuple = {}
    records = list(context["instruments_by_id"].values())
    records.extend(
        record for history in context["instrument_history_by_id"].values() for record in history
    )
    for record in records:
        identity = tuple(
            record[field] for field in ("exchange_id", "environment", "market_type", "venue_symbol")
        )
        instrument_id = record["instrument_id"]
        if tuple_to_id.setdefault(identity, instrument_id) != instrument_id:
            return False
        if id_to_tuple.setdefault(instrument_id, identity) != identity:
            return False
    return True


def validate_snapshot_record(context, record, validation_time):
    projection_contract = resolve_instrument_projection_contract()
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
        <= set(
            projection_contract["allowed_market_instrument_type_pairs"].get(
                record["market_type"], []
            )
        )
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
            for field in ("workspace_id", "exchange_id", "environment", "market_type")
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
            for marker in resolve_m05_secure_store_grammar()["forbidden_payload_markers"]
        )
        or record["saas_sync_candidate"] is not False
    ):
        return False
    return True


def trusted_external_identity_tuple(account, identity):
    """Return M0.5 identity only for ACTIVE accounts with a matching VERIFIED snapshot."""
    if (
        account["lifecycle_state"] != "ACTIVE"
        or account["external_account_identity_state"] != "VERIFIED"
        or identity is None
        or identity.get("state") != "VERIFIED"
        or any(
            identity.get(field) != account.get(field)
            for field in ("exchange_id", "environment", "market_type")
        )
    ):
        return None
    return tuple(
        identity.get(field)
        for field in (
            "exchange_id",
            "environment",
            "market_type",
            "venue_account_identifier",
            "subaccount_identifier",
        )
    )


def trusted_external_identity_collision(context, account_id):
    account = context["accounts_by_id"][account_id]
    identity = context["external_identity_snapshots_by_account_id"].get(account_id)
    trusted_tuple = trusted_external_identity_tuple(account, identity)
    if trusted_tuple is None:
        return False
    matches = 0
    for candidate_id, candidate in context["accounts_by_id"].items():
        candidate_identity = context["external_identity_snapshots_by_account_id"].get(candidate_id)
        matches += trusted_external_identity_tuple(candidate, candidate_identity) == trusted_tuple
    return matches != 1


def validate_all_canonical_projections(context, validation_time):
    if (
        not validate_instrument_history_map(context, validation_time)
        or not validate_global_instrument_identity(context)
        or any(
            not validate_instrument_record(context, record, validation_time)
            for record in context["instruments_by_id"].values()
        )
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
    projection_contract = resolve_instrument_projection_contract()
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
            projection_contract["allowed_market_instrument_type_pairs"].get(
                snapshot["market_type"], []
            )
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
        identity_tuple = trusted_external_identity_tuple(account, identity)
        if identity_tuple is not None:
            if identity_tuple in identity_tuples:
                return False
            identity_tuples.add(identity_tuple)
    for route in context["market_data_routes_by_id"].values():
        endpoint = resolve_m06_local_authority_policy("endpoint_class_registry").get(
            route["endpoint_class"]
        )
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
        endpoint = resolve_m06_local_authority_policy("endpoint_class_registry").get(
            route["endpoint_class"]
        )
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


def validate_context(request, context, operation):
    validate_canonical_dependency_roots()
    resolve_m04_environment_ids()
    resolve_current_edition_execution_pair_policy()
    resolve_m04_current_product_capability_policy()
    for policy_id in EXPECTED_M06_LOCAL_AUTHORITY_POLICIES:
        resolve_m06_local_authority_policy(policy_id)
    validate_canonical_enum_binding_manifest()
    resolve_instrument_projection_contract()
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
                if deep_freeze(record) != resolve_m04_product_capabilities_projection(
                    record["environment"]
                ):
                    return deny(operation, "TRUSTED_CONTEXT_INVALID")
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


def validate_request(request, context, operation):
    if operation not in REQUESTS:
        return deny(operation, "UNKNOWN_OPERATION")
    protocol = validate_executable_operation_protocol()
    schema = protocol["operation_registry"][operation]["request_schema"]
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
    endpoint = resolve_m06_local_authority_policy("endpoint_class_registry")[
        route["endpoint_class"]
    ]
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
    expected = resolve_m06_local_authority_policy("authorization_dependencies_by_environment")[
        account["environment"]
    ]
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
            else resolve_m06_local_authority_policy("execution_readiness_max_age_seconds")
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
    expected = set(
        resolve_m06_local_authority_policy("authorization_dependencies_by_environment")[
            account["environment"]
        ]
    )
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
        raise TypeError("canonical ProductCapabilities invariant")
    if account["environment"] == "PAPER":
        return None
    policy = resolve_m05_account_operability_policy()
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
    max_age = resolve_m06_local_authority_policy("account_capability_snapshot_policy")[
        "max_age_seconds_by_environment"
    ]["TESTNET"]
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
    identity_collision = trusted_external_identity_collision(
        context, account["exchange_account_id"]
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
    policy = resolve_current_edition_execution_pair_policy()
    if any(
        instrument["instrument_type"]
        not in policy["allowed_pairs"].get(instrument["market_type"], [])
        for instrument in instruments
    ):
        return deny(operation, policy["unsupported_pair_denial"])
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


EXPECTED_VALIDATOR_IMPLEMENTATIONS = MappingProxyType(
    {
        "validate_request": validate_request,
        "validate_context": validate_context,
        "validate_definition_create": validate_definition_create,
        "validate_definition_lifecycle": validate_definition_lifecycle,
        "validate_instance_create": validate_instance_create,
        "validate_market_bind": validate_market_bind,
        "validate_execution_bind": validate_execution_bind,
        "validate_route_readiness_operation": validate_route_readiness_operation,
        "validate_activation": validate_activation,
        "validate_deactivation": validate_deactivation,
        "validate_retirement": validate_retirement,
        "validate_authorization_operability": validate_authorization_operability,
        "validate_readiness_lifecycle": validate_readiness_lifecycle,
        "validate_strategy_instance_lifecycle_preflight": (
            validate_strategy_instance_lifecycle_preflight
        ),
        "validate_strategy_execution_operability": validate_strategy_execution_operability,
    }
)


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
    if (
        type(operation) is not str
        or operation not in EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["operation_registry"]
    ):
        return deny(operation, "UNKNOWN_OPERATION")
    try:
        protocol = validate_executable_operation_protocol()
        graph = EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["operation_validator_call_graph"][operation]
        for name in graph:
            result = EXPECTED_VALIDATOR_IMPLEMENTATIONS[name](request, context, operation)
            if result is not None:
                if not result["allowed"]:
                    denial = result["denial_code"]
                    if (
                        denial not in protocol["allowed_denials_by_operation"][operation]
                        or denial not in protocol["validator_denial_registry"][name]["denials"]
                    ):
                        return deny(operation, "CONTRACT_INCONSISTENT")
                if result["allowed"] and not planned_outcome_is_valid(
                    operation, request, context, result
                ):
                    return deny(operation, "CONTRACT_INCONSISTENT")
                return result
    except (
        AssertionError,
        AttributeError,
        KeyError,
        StopIteration,
        TypeError,
        IndexError,
        ValueError,
    ):
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
        "authorization_dependencies": list(
            resolve_m06_local_authority_policy("authorization_dependencies_by_environment")[
                environment
            ]
        ),
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
                "edition": "CRYPTOHUNTER_TESTNET_EDITION",
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


def canonical_pair_context(market_type, instrument_type, *, instance_state="BOUND"):
    """Build a complete PAPER graph for one canonical M0.5 market/instrument pair."""
    context = fixture_context("PAPER", instance_state=instance_state)
    for record in (
        context["accounts_by_id"][IDS["xacc"]],
        context["instruments_by_id"][IDS["instr"]],
        context["catalogs_by_id"][IDS["icat"]],
        context["account_capability_snapshots_by_id"][IDS["capsnap"]],
        context["market_data_routes_by_id"][IDS["mdr"]],
        context["execution_routes_by_id"][IDS["xroute"]],
        context["external_identity_snapshots_by_account_id"][IDS["xacc"]],
    ):
        record["market_type"] = market_type
    instrument = context["instruments_by_id"][IDS["instr"]]
    instrument["instrument_type"] = instrument_type
    snapshot = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    snapshot["supported_instrument_types"] = [instrument_type]
    context["execution_routes_by_id"][IDS["xroute"]]["supported_instrument_types"] = [
        instrument_type
    ]
    if instrument_type not in {"SPOT_PAIR", "MARGIN_PAIR"}:
        instrument.update(
            settlement_asset_reference=copy.deepcopy(instrument["quote_asset_reference"]),
            contract_size="1",
            contract_value_currency="USD",
            derivative_settlement_type="LINEAR",
        )
    if instrument_type in {"DELIVERY_FUTURE", "OPTION"}:
        instrument["expiry_at_utc"] = "2027-01-01T00:00:00Z"
    if instrument_type == "OPTION":
        instrument.update(strike_price="1", option_side="CALL")
    rehash_m05_projections(context)
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
            instrument["venue_symbol"] = "FOREIGN-WORKSPACE-SYMBOL"
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
            instrument["venue_symbol"] = "ROUTE-ONLY-SYMBOL"
            context["instruments_by_id"][iid] = instrument
            context["catalogs_by_id"][IDS["icat"]]["instrument_ids"].append(iid)
            context["market_data_routes_by_id"][IDS["mdr"]]["instrument_ids"] = [iid]
        elif operation == "VALIDATE_ROUTE_READINESS":
            iid = f"instr_{UUID7}0f"
            instrument = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
            instrument["instrument_id"] = iid
            instrument["venue_symbol"] = "READINESS-ONLY-SYMBOL"
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
    elif denial == "TRUSTED_CONTEXT_INVALID":
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
    assert (
        graph_names
        == set(EXPECTED_VALIDATOR_IMPLEMENTATIONS)
        == set(CONTRACT["validator_denial_registry"])
    )
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
        == "TRUSTED_CONTEXT_INVALID"
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
            "TRUSTED_CONTEXT_INVALID",
        ),
        (
            lambda c: c["product_capabilities_by_environment"]["TESTNET"].update(
                allowed_operations=["LIVE_VISIBLE_LOCKED_ONLY"]
            ),
            "TRUSTED_CONTEXT_INVALID",
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
    validation = validate_context({}, context, "VALIDATE_ROUTE_READINESS")
    if expected == "TRUSTED_CONTEXT_INVALID":
        assert validation["denial_code"] == expected
    else:
        assert validation is None
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
    malformed = success(operation, {"entity": "wrong"})
    assert not planned_outcome_is_valid(operation, request_for(operation), context, malformed)


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
    rehash_m05_projections(context)
    assert validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE") is None


@pytest.mark.parametrize(
    "operation", ["ACTIVATE_STRATEGY_INSTANCE", "DEACTIVATE_STRATEGY_INSTANCE"]
)
def test_retired_instance_is_terminal_before_other_preflights(operation):
    context = fixture_context(instance_state="RETIRED")
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
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
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    request = request_for(operation)
    context = fixture_context(instance_state=state)
    if state == "DRAFT":
        context["strategy_instances_by_id"][IDS["sinst"]].update(
            market_data_route_id=None, execution_route_id=None
        )
    observed = []
    result = None
    for name in EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["operation_validator_call_graph"][operation]:
        observed.append(name)
        result = EXPECTED_VALIDATOR_IMPLEMENTATIONS[name](request, context, operation)
        if result is not None:
            break
    assert result["denial_code"] in {
        "STRATEGY_INSTANCE_BINDING_MISMATCH",
        "RETIRED_RESOURCE_FORBIDDEN",
    }
    assert "validate_authorization_operability" not in observed


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
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )
    product["allowed_operations"] = []
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_quantity", "banana"),
        ("max_quantity", "banana"),
        ("min_notional", "banana"),
        ("max_notional", "banana"),
        ("price_tick", "0"),
        ("quantity_step", "0"),
        ("min_quantity", "1e2"),
        ("min_quantity", "1.0"),
        ("min_quantity", 1),
        ("min_quantity", 1.0),
    ],
)
def test_every_instrument_decimal_uses_one_canonical_policy(field, value):
    context = fixture_context("PAPER")
    record = context["instruments_by_id"][IDS["instr"]]
    record[field] = value
    assert not validate_instrument_record(
        context, record, parse_time(context["validation_time_utc"])
    )


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [("min_quantity", "max_quantity"), ("min_notional", "max_notional")],
)
def test_instrument_decimal_bounds_are_ordered_and_null_maximum_is_unbounded(minimum, maximum):
    context = fixture_context("PAPER")
    record = context["instruments_by_id"][IDS["instr"]]
    record[minimum], record[maximum] = "2", "1"
    assert not validate_instrument_record(
        context, record, parse_time(context["validation_time_utc"])
    )
    record[maximum] = None
    assert validate_instrument_record(context, record, parse_time(context["validation_time_utc"]))


@pytest.mark.parametrize("mapping_status", ["UNKNOWN", "AMBIGUOUS"])
def test_settlement_asset_reference_fails_closed_for_non_authoritative_mapping(mapping_status):
    context = fixture_context("PAPER")
    record = context["instruments_by_id"][IDS["instr"]]
    record["settlement_asset_reference"] = copy.deepcopy(record["quote_asset_reference"])
    record["settlement_asset_reference"]["mapping_status"] = mapping_status
    assert not validate_instrument_record(
        context, record, parse_time(context["validation_time_utc"])
    )


def test_credential_environment_scope_resolves_paper_from_canonical_registry():
    schema = SCHEMAS["CredentialProfileProjection"]
    reference = schema["enum_canonical_pointer_ref"]["environment_scope"]
    assert "PAPER" in resolve_canonical_pointer(reference, list)


CANONICAL_INSTRUMENT_PAIRS = [
    ("SPOT", "SPOT_PAIR"),
    ("MARGIN", "MARGIN_PAIR"),
    ("PERPETUAL", "PERPETUAL_CONTRACT"),
    ("DELIVERY_FUTURES", "DELIVERY_FUTURE"),
    ("OPTIONS", "OPTION"),
]


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong_contract",
        "missing_segment",
        "invalid_escape",
        "wrong_result_type",
        "empty_registry",
        "duplicate_registry",
        "non_string_registry",
        "allowed_pairs",
        "asset_reference",
        "decimal_policy",
        "derivative_rules",
        "record_fields_binding",
    ],
)
def test_instrument_projection_pointer_corruption_is_contract_inconsistent(monkeypatch, corruption):
    references = SCHEMAS["InstrumentProjection"]["canonical_projection_refs"]
    operation = "VALIDATE_ROUTE_READINESS"
    request = request_for(operation)
    context = fixture_context()
    if corruption == "wrong_contract":
        monkeypatch.setitem(
            references,
            "environment_registry",
            {"contract": "wrong.json", "json_pointer": "/environment_registry"},
        )
    elif corruption == "missing_segment":
        monkeypatch.setitem(
            references,
            "environment_registry",
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/missing/registry",
            },
        )
    elif corruption == "invalid_escape":
        monkeypatch.setitem(
            references,
            "environment_registry",
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/environment_registry/~2",
            },
        )
    elif corruption == "wrong_result_type":
        monkeypatch.setitem(
            references,
            "environment_registry",
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/decimal_policy",
            },
        )
    elif corruption == "empty_registry":
        monkeypatch.setitem(M05_CONTRACT, "environment_registry", [])
    elif corruption == "duplicate_registry":
        monkeypatch.setitem(M05_CONTRACT, "environment_registry", ["PAPER", "PAPER"])
    elif corruption == "non_string_registry":
        monkeypatch.setitem(M05_CONTRACT, "environment_registry", ["PAPER", 7])
    elif corruption == "allowed_pairs":
        monkeypatch.setitem(M05_CONTRACT, "allowed_market_instrument_type_pairs", {"SPOT": []})
    elif corruption == "asset_reference":
        monkeypatch.setitem(M05_CONTRACT, "asset_reference_contract", {"fields": []})
    elif corruption == "decimal_policy":
        monkeypatch.setitem(M05_CONTRACT, "decimal_policy", {"regex": ".*"})
    elif corruption == "derivative_rules":
        monkeypatch.setitem(M05_CONTRACT, "derivative_consistency_rules", {"SPOT_PAIR": {}})
    else:
        monkeypatch.setitem(
            SCHEMAS["InstrumentProjection"],
            "exact_fields",
            SCHEMAS["InstrumentProjection"]["exact_fields"][:-1],
        )
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("prefix", ["quantity", "notional"])
@pytest.mark.parametrize(
    ("minimum", "maximum", "valid"),
    [(None, None, True), ("1", None, True), (None, "2", False), ("1", "2", True)],
)
def test_limit_nullability_is_validated_through_full_context(prefix, minimum, maximum, valid):
    context = fixture_context("PAPER")
    instrument = context["instruments_by_id"][IDS["instr"]]
    instrument[f"min_{prefix}"] = minimum
    instrument[f"max_{prefix}"] = maximum
    denial = validate_context({}, context, "VALIDATE_ROUTE_READINESS")
    if valid:
        assert denial is None
    else:
        assert denial is not None and denial["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(("market_type", "instrument_type"), CANONICAL_INSTRUMENT_PAIRS)
@pytest.mark.parametrize("operation", ["VALIDATE_ROUTE_READINESS", "ACTIVATE_STRATEGY_INSTANCE"])
def test_complete_paper_graph_structurally_represents_every_canonical_pair(
    market_type, instrument_type, operation
):
    context = canonical_pair_context(market_type, instrument_type)
    request = request_for(operation)
    assert validate_context(request, context, operation) is None
    result = dispatcher(operation, request, context)
    if (market_type, instrument_type) == ("SPOT", "SPOT_PAIR"):
        assert result["allowed"] is True
    else:
        assert result["denial_code"] == "INSTRUMENT_SCOPE_MISMATCH"
        assert result["audit_event_type"] == CONTRACT["denial_event_by_operation"][operation]
        assert result["denial_code"] != "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(("market_type", "instrument_type"), CANONICAL_INSTRUMENT_PAIRS[1:])
def test_bind_checks_declaration_not_current_edition_operability(market_type, instrument_type):
    context = canonical_pair_context(market_type, instrument_type, instance_state="DRAFT")
    instance = context["strategy_instances_by_id"][IDS["sinst"]]
    instance["market_data_route_id"] = None
    request = request_for("BIND_MARKET_DATA_ROUTE")
    assert validate_context(request, context, "BIND_MARKET_DATA_ROUTE") is None
    assert dispatcher("BIND_MARKET_DATA_ROUTE", request, context)["allowed"] is True


def test_historical_universe_rejects_foreign_workspace_instrument_end_to_end():
    context, previous_catalog = context_with_previous_catalog()
    previous_universe_id = f"univ_{UUID7}0e"
    previous_universe = copy.deepcopy(context["universes_by_id"][IDS["univ"]])
    previous_universe.update(
        trading_universe_id=previous_universe_id,
        source_catalog_snapshot_ids=[previous_catalog["catalog_snapshot_id"]],
        lifecycle_state="RETIRED",
        version=1,
        previous_version_id=None,
        retired_at_utc="2025-12-31T23:59:50Z",
    )
    context["universes_by_id"][IDS["univ"]].update(
        version=2, previous_version_id=previous_universe_id
    )
    context["previous_universes_by_id"][previous_universe_id] = previous_universe
    foreign_workspace = f"ws_{UUID7}0f"
    context["workspace_ids"].append(foreign_workspace)
    context["instrument_history_by_id"][IDS["instr"]][0]["workspace_id"] = foreign_workspace
    rehash_m05_projections(context)
    previous_universe["content_hash"] = canonical_m05_hash("trading_universe", previous_universe)
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    request = request_for(operation)
    denial = validate_context(request, context, operation)
    assert denial is not None and denial["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert dispatcher(operation, request, context)["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def second_account_with_identity(context, *, lifecycle_state, account_type):
    account_id = f"xacc_{UUID7}0f"
    account = copy.deepcopy(context["accounts_by_id"][IDS["xacc"]])
    account.update(
        exchange_account_id=account_id,
        lifecycle_state=lifecycle_state,
        account_capability_snapshot_id=None,
        active_credential_profile_id=None,
    )
    identity = copy.deepcopy(context["external_identity_snapshots_by_account_id"][IDS["xacc"]])
    identity["account_type"] = account_type
    context["accounts_by_id"][account_id] = account
    context["external_identity_snapshots_by_account_id"][account_id] = identity
    return account_id


def test_active_verified_identity_collision_ignores_account_type_end_to_end():
    context = fixture_context()
    second_account_with_identity(context, lifecycle_state="ACTIVE", account_type="DIFFERENT")
    result = dispatcher(
        "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_disabled_duplicate_identity_does_not_block_active_account_end_to_end():
    context = fixture_context("PAPER")
    second_account_with_identity(context, lifecycle_state="DISABLED", account_type="DIFFERENT")
    operation = "VALIDATE_ROUTE_READINESS"
    request = request_for(operation)
    assert validate_context(request, context, operation) is None
    result = dispatcher(operation, request, context)
    assert result["allowed"] is True
    assert result["denial_code"] is None


def test_two_current_instrument_ids_with_one_tuple_fail_full_context():
    context = fixture_context("PAPER")
    duplicate = copy.deepcopy(context["instruments_by_id"][IDS["instr"]])
    duplicate_id = f"instr_{UUID7}0f"
    duplicate["instrument_id"] = duplicate_id
    context["instruments_by_id"][duplicate_id] = duplicate
    context["catalogs_by_id"][IDS["icat"]]["instrument_ids"].append(duplicate_id)
    rehash_m05_projections(context)
    denial = validate_context({}, context, "VALIDATE_ROUTE_READINESS")
    assert denial is not None and denial["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_current_and_history_different_ids_with_one_tuple_fail_full_context():
    context, previous_catalog = context_with_previous_catalog()
    historical = context["instrument_history_by_id"].pop(IDS["instr"])[0]
    historical_id = f"instr_{UUID7}0f"
    historical["instrument_id"] = historical_id
    context["instrument_history_by_id"][historical_id] = [historical]
    previous_catalog["instrument_ids"] = [historical_id]
    previous_catalog["content_hash"] = canonical_m05_hash(
        "instrument_catalog_snapshot", previous_catalog
    )
    denial = validate_context({}, context, "VALIDATE_ROUTE_READINESS")
    assert denial is not None and denial["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_one_instrument_id_identity_drift_in_history_fails_full_context():
    context, _ = context_with_previous_catalog()
    context["instrument_history_by_id"][IDS["instr"]][0]["venue_symbol"] = "DRIFTED"
    denial = validate_context({}, context, "VALIDATE_ROUTE_READINESS")
    assert denial is not None and denial["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def test_instrument_projection_cross_contract_audit_is_executable():
    resolved = resolve_instrument_projection_contract()
    projection_manifest = validate_canonical_projection_binding_manifest()
    enum_manifest = validate_canonical_enum_binding_manifest()
    policy = resolve_current_edition_execution_pair_policy()
    audit = CONTRACT["cross_contract_projection_audit"]["InstrumentProjection"]
    assert set(audit["canonical_projection_refs_executed"]) == set(
        SCHEMAS["InstrumentProjection"]["canonical_projection_refs"]
    )
    assert set(resolved["record_fields"]) == set(SCHEMAS["InstrumentProjection"]["exact_fields"])
    assert {"min_quantity", "max_quantity", "min_notional", "max_notional"} <= set(
        SCHEMAS["InstrumentProjection"]["nullable_fields"]
    )
    assert audit["current_edition_authority_policy"] == "current_edition_execution_pair_policy"
    assert audit["pointer_corruption_denial"] == "CONTRACT_INCONSISTENT"
    assert len(audit["full_graph_structural_pairs"]) == 5
    assert set(projection_manifest) == set(
        SCHEMAS["InstrumentProjection"]["canonical_projection_refs"]
    )
    assert set(enum_manifest) == set(collect_external_canonical_enum_consumers())
    assert audit["edition_policy_used_by"] == list(policy["used_by"])
    assert audit["bind_operations_outside_edition_gate"] == [
        "BIND_MARKET_DATA_ROUTE",
        "BIND_EXECUTION_ROUTE",
    ]
    assert audit["failure_classification"] == {
        "machine_contract": "CONTRACT_INCONSISTENT",
        "persisted_record": "TRUSTED_CONTEXT_INVALID",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "add_margin",
        "add_perpetual",
        "remove_spot",
        "empty_pairs",
        "unknown_market",
        "unknown_instrument",
        "wrong_policy_id",
        "wrong_denial",
        "missing_readiness",
        "missing_activation",
        "extra_operation",
        "false_structural_flag",
        "false_bind_flag",
        "extra_field",
        "missing_field",
        "wrong_field_type",
    ],
)
def test_current_edition_policy_corruption_fails_closed_in_both_paths(monkeypatch, mutation):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    context = fixture_context("PAPER")
    request = request_for(operation)
    policy = copy.deepcopy(CONTRACT["current_edition_execution_pair_policy"])
    if mutation == "add_margin":
        policy["allowed_pairs"]["MARGIN"] = ["MARGIN_PAIR"]
    elif mutation == "add_perpetual":
        policy["allowed_pairs"]["PERPETUAL"] = ["PERPETUAL_CONTRACT"]
    elif mutation == "remove_spot":
        policy["allowed_pairs"].pop("SPOT")
    elif mutation == "empty_pairs":
        policy["allowed_pairs"] = {}
    elif mutation == "unknown_market":
        policy["allowed_pairs"]["UNKNOWN"] = ["SPOT_PAIR"]
    elif mutation == "unknown_instrument":
        policy["allowed_pairs"]["SPOT"] = ["UNKNOWN_PAIR"]
    elif mutation == "wrong_policy_id":
        policy["policy_id"] = "WRONG"
    elif mutation == "wrong_denial":
        policy["unsupported_pair_denial"] = "TRUSTED_CONTEXT_INVALID"
    elif mutation == "missing_readiness":
        policy["used_by"].remove("VALIDATE_ROUTE_READINESS")
    elif mutation == "missing_activation":
        policy["used_by"].remove("ACTIVATE_STRATEGY_INSTANCE")
    elif mutation == "extra_operation":
        policy["used_by"].append("BIND_MARKET_DATA_ROUTE")
    elif mutation == "false_structural_flag":
        policy["structural_projection_is_not_authority"] = False
    elif mutation == "false_bind_flag":
        policy["bind_checks_declaration_not_current_operability"] = False
    elif mutation == "extra_field":
        policy["extra"] = True
    elif mutation == "missing_field":
        policy.pop("unsupported_pair_denial")
    else:
        policy["allowed_pairs"] = "SPOT"
    monkeypatch.setitem(CONTRACT, "current_edition_execution_pair_policy", policy)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["allowed"] is False
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


def test_margin_authority_cannot_be_enabled_by_mutating_machine_policy(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    context = canonical_pair_context("MARGIN", "MARGIN_PAIR")
    request = request_for(operation)
    assert validate_context(request, context, operation) is None
    policy = copy.deepcopy(CONTRACT["current_edition_execution_pair_policy"])
    policy["allowed_pairs"]["MARGIN"] = ["MARGIN_PAIR"]
    monkeypatch.setitem(CONTRACT, "current_edition_execution_pair_policy", policy)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["allowed"] is False
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("binding", "pointer"),
    [
        ("environment_registry", "/instrument_contract/trading_statuses"),
        ("trading_statuses", "/environment_registry"),
        ("market_type_registry", "/instrument_type_registry"),
        ("instrument_type_registry", "/environment_registry"),
    ],
)
def test_same_type_projection_pointer_swaps_fail_closed(monkeypatch, binding, pointer):
    context = fixture_context()
    operation = "VALIDATE_ROUTE_READINESS"
    request = request_for(operation)
    references = copy.deepcopy(SCHEMAS["InstrumentProjection"]["canonical_projection_refs"])
    references[binding]["json_pointer"] = pointer
    monkeypatch.setitem(SCHEMAS["InstrumentProjection"], "canonical_projection_refs", references)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "fault",
    [
        "remove_pointer_binding",
        "remove_registry_binding",
        "wrong_valid_contract",
        "wrong_same_type_pointer",
        "content_without_fingerprint",
        "fingerprint_without_content",
    ],
)
def test_canonical_enum_binding_manifest_faults_are_machine_contract_errors(monkeypatch, fault):
    context = fixture_context()
    operation = "VALIDATE_ROUTE_READINESS"
    request = request_for(operation)
    if fault == "remove_pointer_binding":
        bindings = copy.deepcopy(SCHEMAS["MarketDataRoute"]["enum_canonical_pointer_ref"])
        bindings.pop("market_type")
        monkeypatch.setitem(SCHEMAS["MarketDataRoute"], "enum_canonical_pointer_ref", bindings)
    elif fault == "remove_registry_binding":
        bindings = copy.deepcopy(SCHEMAS["InstrumentProjection"]["enum_registry_ref"])
        bindings.pop("trading_status")
        monkeypatch.setitem(SCHEMAS["InstrumentProjection"], "enum_registry_ref", bindings)
    elif fault == "wrong_valid_contract":
        bindings = copy.deepcopy(SCHEMAS["MarketDataRoute"]["enum_canonical_pointer_ref"])
        bindings["market_type"] = {
            "contract": "environment_and_product_capabilities.json",
            "json_pointer": "/environment_registry",
        }
        monkeypatch.setitem(SCHEMAS["MarketDataRoute"], "enum_canonical_pointer_ref", bindings)
    elif fault == "wrong_same_type_pointer":
        bindings = copy.deepcopy(SCHEMAS["InstrumentProjection"]["enum_canonical_pointer_ref"])
        bindings["environment"]["json_pointer"] = "/instrument_contract/trading_statuses"
        monkeypatch.setitem(SCHEMAS["InstrumentProjection"], "enum_canonical_pointer_ref", bindings)
    elif fault == "content_without_fingerprint":
        monkeypatch.setitem(M05_CONTRACT, "market_type_registry", ["SPOT"])
    else:
        manifest = copy.deepcopy(CONTRACT["canonical_enum_binding_manifest"])
        manifest["MarketDataRoute.market_type"]["content_fingerprint_sha256"] = "0" * 64
        monkeypatch.setitem(CONTRACT, "canonical_enum_binding_manifest", manifest)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(("market_type", "instrument_type"), CANONICAL_INSTRUMENT_PAIRS[1:])
@pytest.mark.parametrize("operation", ["BIND_MARKET_DATA_ROUTE", "BIND_EXECUTION_ROUTE"])
def test_both_route_binds_ignore_current_edition_gate_for_valid_declarations(
    market_type, instrument_type, operation
):
    context = canonical_pair_context(market_type, instrument_type, instance_state="DRAFT")
    instance = context["strategy_instances_by_id"][IDS["sinst"]]
    instance["market_data_route_id"] = None
    instance["execution_route_id"] = None
    request = request_for(operation)
    assert validate_context(request, context, operation) is None
    result = dispatcher(operation, request, context)
    assert result["allowed"] is True
    assert result["planned_transition"]["resulting_state"] == "DRAFT"


def test_persisted_noncanonical_enum_remains_trusted_context_invalid():
    context = fixture_context()
    context["instruments_by_id"][IDS["instr"]]["trading_status"] = "NOT_CANONICAL"
    result = dispatcher(
        "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
    )
    assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(
    ("reference_name", "source_reference_name"),
    [
        ("m05_account_lifecycle_states", "m05_universe_lifecycle_states"),
        ("m05_snapshot_statuses", "m05_catalog_statuses"),
        ("m05_catalog_statuses", "m05_snapshot_statuses"),
        ("m05_external_identity_states", "m05_snapshot_statuses"),
        ("m05_permissions", "m04_product_capabilities"),
        ("m05_capabilities", "m04_product_capabilities"),
        ("m04_product_capabilities", "m05_permissions"),
    ],
)
def test_registry_reference_same_type_swaps_are_machine_faults(
    monkeypatch, reference_name, source_reference_name
):
    context = fixture_context()
    operation = "VALIDATE_ROUTE_READINESS"
    request = request_for(operation)
    scalar = CONTRACT["canonical_scalar_enum_registry_refs"]
    arrays = CONTRACT["canonical_array_enum_registry_refs"]
    target = scalar if reference_name in scalar else arrays
    source = scalar.get(source_reference_name, arrays.get(source_reference_name))
    monkeypatch.setitem(target, reference_name, copy.deepcopy(source))
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("schema_name", "field", "replacement"),
    [
        ("MarketDataRoute", "market_type", "string"),
        ("InstrumentProjection", "trading_status", "string"),
        ("ExecutionRoute", "supported_instrument_types", "array[string]"),
        ("AccountCapabilitySnapshotProjection", "status", "string"),
        ("CredentialProfileProjection", "permission_snapshot", "array[string]"),
    ],
)
def test_external_enum_field_type_corruption_is_machine_fault(
    monkeypatch, schema_name, field, replacement
):
    context = fixture_context()
    field_types = copy.deepcopy(SCHEMAS[schema_name]["field_types"])
    field_types[field] = replacement
    monkeypatch.setitem(SCHEMAS[schema_name], "field_types", field_types)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("fault", ["remove", "duplicate", "rename"])
def test_external_enum_exact_fields_corruption_is_machine_fault(monkeypatch, fault):
    context = fixture_context()
    schema = SCHEMAS["MarketDataRoute"]
    fields = copy.deepcopy(schema["exact_fields"])
    index = fields.index("market_type")
    if fault == "remove":
        fields.pop(index)
    elif fault == "duplicate":
        fields.append("market_type")
    else:
        fields[index] = "market_kind"
    monkeypatch.setitem(schema, "exact_fields", fields)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("schema_name", "field"),
    [
        ("MarketDataRoute", "market_type"),
        ("InstrumentProjection", "trading_status"),
        ("ExchangeAccountProjection", "lifecycle_state"),
    ],
)
def test_external_enum_nullability_corruption_is_machine_fault(monkeypatch, schema_name, field):
    context = fixture_context()
    nullable = copy.deepcopy(SCHEMAS[schema_name]["nullable_fields"])
    nullable.append(field)
    monkeypatch.setitem(SCHEMAS[schema_name], "nullable_fields", nullable)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "fault", ["unique", "empty_allowed", "remove_ref", "swap_ref", "local_fallback"]
)
def test_external_enum_array_policy_corruption_is_machine_fault(monkeypatch, fault):
    context = fixture_context()
    schema_name = "CredentialProfileProjection" if fault == "empty_allowed" else "ExecutionRoute"
    field = "permission_snapshot" if fault == "empty_allowed" else "supported_instrument_types"
    policies = copy.deepcopy(SCHEMAS[schema_name]["array_policy"])
    policy = policies[field]
    if fault == "unique":
        policy["unique"] = False
    elif fault == "empty_allowed":
        policy["empty_allowed"] = True
    elif fault == "remove_ref":
        policy.pop("item_registry_ref")
    elif fault == "swap_ref":
        policy["item_registry_ref"] = "m05_permissions"
    else:
        policy["item_registry"] = "instrument_types"
    monkeypatch.setitem(SCHEMAS[schema_name], "array_policy", policies)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "fault",
    [
        "lifecycle",
        "snapshot_status",
        "catalog_status",
        "trading_status",
        "credential_purpose",
        "permission",
        "supported_instrument_type",
    ],
)
def test_noncanonical_persisted_external_enum_is_trusted_context_invalid(fault):
    context = fixture_context()
    if fault == "lifecycle":
        context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "NOT_CANONICAL"
    elif fault == "snapshot_status":
        context["account_capability_snapshots_by_id"][IDS["capsnap"]]["status"] = "NOT_CANONICAL"
    elif fault == "catalog_status":
        context["catalogs_by_id"][IDS["icat"]]["status"] = "NOT_CANONICAL"
    elif fault == "trading_status":
        context["instruments_by_id"][IDS["instr"]]["trading_status"] = "NOT_CANONICAL"
    elif fault == "credential_purpose":
        context["credential_profiles_by_id"][IDS["cred"]]["credential_purpose"] = "NOT_CANONICAL"
    elif fault == "permission":
        context["credential_profiles_by_id"][IDS["cred"]]["permission_snapshot"] = ["NOT_CANONICAL"]
    else:
        context["account_capability_snapshots_by_id"][IDS["capsnap"]][
            "supported_instrument_types"
        ] = ["NOT_CANONICAL"]
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def _assert_contract_fault_in_both_paths(context):
    operation = "VALIDATE_ROUTE_READINESS"
    request = request_for(operation)
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["allowed"] is False
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


def test_expected_observed_and_manifest_external_consumers_are_exactly_equal():
    assert (
        set(EXPECTED_EXTERNAL_CANONICAL_ENUM_CONSUMERS)
        == set(collect_external_canonical_enum_consumers())
        == set(CONTRACT["canonical_enum_binding_manifest"])
    )
    validate_canonical_enum_binding_manifest()


@pytest.mark.parametrize(
    ("schema_name", "field"),
    [
        ("MarketDataRoute", "environment"),
        ("ExecutionRoute", "environment"),
        ("ExchangeAccountProjection", "environment"),
        ("InstrumentProjection", "environment"),
        ("InstrumentCatalogProjection", "environment"),
        ("AccountCapabilitySnapshotProjection", "environment"),
        ("CredentialProfileProjection", "environment_scope"),
        ("TrustedExternalIdentityProjection", "environment"),
        ("ProductCapabilitiesProjection", "environment"),
    ],
)
def test_local_environment_fallback_drift_is_machine_fault(monkeypatch, schema_name, field):
    context = fixture_context()
    local = copy.deepcopy(SCHEMAS[schema_name].get("enum_registry", {}))
    local[field] = ["PAPER", "TESTNET", "LIVE", "SHADOW"]
    monkeypatch.setitem(SCHEMAS[schema_name], "enum_registry", local)
    _assert_contract_fault_in_both_paths(context)


def test_dual_external_binding_is_machine_fault(monkeypatch):
    context = fixture_context()
    refs = copy.deepcopy(SCHEMAS["ExchangeAccountProjection"]["enum_registry_ref"])
    refs["environment"] = "m05_account_lifecycle_states"
    monkeypatch.setitem(SCHEMAS["ExchangeAccountProjection"], "enum_registry_ref", refs)
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize(
    ("schema_name", "field", "replacement_ref"),
    [
        ("ExchangeAccountProjection", "lifecycle_state", "m05_universe_lifecycle_states"),
        ("AccountCapabilitySnapshotProjection", "status", "m05_catalog_statuses"),
        ("CredentialProfileProjection", "permission_snapshot", "m04_product_capabilities"),
    ],
)
def test_coordinated_schema_and_manifest_registry_swap_still_fails_closed(
    monkeypatch, schema_name, field, replacement_ref
):
    context = fixture_context()
    schema = SCHEMAS[schema_name]
    is_array = schema["field_types"][field] == "array[enum]"
    if is_array:
        policies = copy.deepcopy(schema["array_policy"])
        policies[field]["item_registry_ref"] = replacement_ref
        monkeypatch.setitem(schema, "array_policy", policies)
    else:
        refs = copy.deepcopy(schema["enum_registry_ref"])
        refs[field] = replacement_ref
        monkeypatch.setitem(schema, "enum_registry_ref", refs)
    manifest = copy.deepcopy(CONTRACT["canonical_enum_binding_manifest"])
    entry = manifest[f"{schema_name}.{field}"]
    source = {
        **CONTRACT["canonical_scalar_enum_registry_refs"],
        **CONTRACT["canonical_array_enum_registry_refs"],
    }[replacement_ref]
    entry.update(source)
    entry["registry_ref"] = replacement_ref
    entry["content_fingerprint_sha256"] = canonical_fingerprint(
        resolve_canonical_pointer(source, list)
    )
    if is_array:
        entry["array_policy"]["item_registry_ref"] = replacement_ref
    monkeypatch.setitem(CONTRACT, "canonical_enum_binding_manifest", manifest)
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize(
    ("field", "policy_key", "weakened", "persisted"),
    [
        ("permission_snapshot", "unique", False, ["READ_ACCOUNT", "READ_ACCOUNT"]),
        ("permission_snapshot", "empty_allowed", True, []),
    ],
)
def test_coordinated_array_policy_weakening_still_fails_closed(
    monkeypatch, field, policy_key, weakened, persisted
):
    context = fixture_context()
    schema = SCHEMAS["CredentialProfileProjection"]
    policies = copy.deepcopy(schema["array_policy"])
    policies[field][policy_key] = weakened
    monkeypatch.setitem(schema, "array_policy", policies)
    manifest = copy.deepcopy(CONTRACT["canonical_enum_binding_manifest"])
    manifest[f"CredentialProfileProjection.{field}"]["array_policy"][policy_key] = weakened
    monkeypatch.setitem(CONTRACT, "canonical_enum_binding_manifest", manifest)
    context["credential_profiles_by_id"][IDS["cred"]][field] = persisted
    _assert_contract_fault_in_both_paths(context)


def test_canonical_content_and_manifest_fingerprint_cannot_drift_together(monkeypatch):
    context = fixture_context()
    changed = [*M05_CONTRACT["environment_registry"], "SHADOW"]
    monkeypatch.setitem(M05_CONTRACT, "environment_registry", changed)
    manifest = copy.deepcopy(CONTRACT["canonical_enum_binding_manifest"])
    changed_fingerprint = canonical_fingerprint(changed)
    for entry in manifest.values():
        if (
            entry["contract"] == "exchange_accounts_and_instruments.json"
            and entry["json_pointer"] == "/environment_registry"
        ):
            entry["content_fingerprint_sha256"] = changed_fingerprint
    monkeypatch.setitem(CONTRACT, "canonical_enum_binding_manifest", manifest)
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize(
    ("consumer", "path", "value"),
    [
        ("ExchangeAccountProjection.lifecycle_state", ("registry_ref",), "changed"),
        ("ExchangeAccountProjection.lifecycle_state", ("json_pointer",), "/changed"),
        ("CredentialProfileProjection.permission_snapshot", ("array_policy", "unique"), False),
        (
            "CredentialProfileProjection.permission_snapshot",
            ("array_policy", "empty_allowed"),
            True,
        ),
        ("ExchangeAccountProjection.lifecycle_state", ("binding_kind",), "changed"),
    ],
)
def test_expected_consumer_specification_is_deeply_immutable(consumer, path, value):
    target = EXPECTED_EXTERNAL_CANONICAL_ENUM_CONSUMERS[consumer]
    with pytest.raises(TypeError):
        if len(path) == 1:
            target[path[0]] = value
        else:
            target[path[0]][path[1]] = value
    context = fixture_context()
    assert (
        dispatcher("VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context)[
            "denial_code"
        ]
        is None
    )


def test_expected_dependency_consumers_are_deeply_immutable():
    dependency = EXPECTED_CANONICAL_DEPENDENCIES["m05_decimal_policy"]
    with pytest.raises(TypeError):
        dependency["json_pointer"] = "/changed"
    with pytest.raises(TypeError):
        dependency["consumers"][0] = "changed"


@pytest.mark.parametrize(
    "fault",
    ["execution_environment", "edition_keys", "product_keys", "m05_divergence"],
)
def test_m04_environment_authority_drift_is_machine_fault(monkeypatch, fault):
    context = fixture_context()
    if fault == "execution_environment":
        environments = copy.deepcopy(M04_CONTRACT["execution_environments"])
        environments[0]["environment_id"] = "SHADOW"
        monkeypatch.setitem(M04_CONTRACT, "execution_environments", environments)
    elif fault == "edition_keys":
        capabilities = copy.deepcopy(M04_CONTRACT["current_product_edition"])
        capabilities["environment_capabilities"]["SHADOW"] = {}
        monkeypatch.setitem(M04_CONTRACT, "current_product_edition", capabilities)
    elif fault == "product_keys":
        product = copy.deepcopy(M04_CONTRACT["ProductCapabilities"])
        product["current_edition_capability_policy"]["environment_capabilities"]["SHADOW"] = {}
        monkeypatch.setitem(M04_CONTRACT, "ProductCapabilities", product)
    else:
        monkeypatch.setitem(M05_CONTRACT, "environment_registry", ["PAPER", "TESTNET"])
    _assert_contract_fault_in_both_paths(context)


def _update_dependency_attestation(monkeypatch, dependency_id, value):
    manifest = copy.deepcopy(CONTRACT["canonical_dependency_root_manifest"])
    manifest[dependency_id]["content_fingerprint_sha256"] = canonical_fingerprint(value)
    monkeypatch.setitem(CONTRACT, "canonical_dependency_root_manifest", manifest)


@pytest.mark.parametrize(
    ("dependency_id", "pointer", "mutation"),
    [
        ("m05_decimal_policy", "/decimal_policy", lambda value: value["rules"].append("drift")),
        (
            "m05_asset_reference_contract",
            "/asset_reference_contract",
            lambda value: value["rules"].append("drift"),
        ),
        (
            "m05_allowed_pairs",
            "/allowed_market_instrument_type_pairs",
            lambda value: value["SPOT"].append("OPTION"),
        ),
        (
            "m05_derivative_rules",
            "/derivative_consistency_rules",
            lambda value: value["OPTION"]["option_side"].append("drift"),
        ),
        (
            "m05_instrument_record_fields",
            "/instrument_contract/record_fields",
            lambda value: value.append("drift"),
        ),
    ],
)
def test_projection_source_and_attestation_coordinated_drift_fails_closed(
    monkeypatch, dependency_id, pointer, mutation
):
    context = fixture_context()
    source = resolve_canonical_pointer(
        {"contract": "exchange_accounts_and_instruments.json", "json_pointer": pointer},
        list if dependency_id in {"m05_instrument_record_fields"} else dict,
    )
    changed = copy.deepcopy(source)
    mutation(changed)
    parent = M05_CONTRACT
    parts = pointer.strip("/").split("/")
    for part in parts[:-1]:
        parent = parent[part]
    monkeypatch.setitem(parent, parts[-1], changed)
    _update_dependency_attestation(monkeypatch, dependency_id, changed)
    _assert_contract_fault_in_both_paths(context)


def test_account_operability_policy_cannot_be_coordinately_broadened(monkeypatch):
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["connection_state"] = "BLOCKED"
    policy = copy.deepcopy(M05_CONTRACT["current_edition_account_operability_policy"])
    policy["operational_connection_states"].append("BLOCKED")
    monkeypatch.setitem(M05_CONTRACT, "current_edition_account_operability_policy", policy)
    _update_dependency_attestation(monkeypatch, "m05_account_operability_policy", policy)
    _assert_contract_fault_in_both_paths(context)


def test_blocked_account_is_ordinary_readiness_denial_with_valid_contract():
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["connection_state"] = "BLOCKED"
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "ACCOUNT_READINESS_BLOCKED"


def test_secure_store_grammar_weakening_and_attestation_update_fails_closed(monkeypatch):
    context = fixture_context()
    grammar = copy.deepcopy(
        M05_CONTRACT["credential_profile_contract"]["secure_store_reference_grammar"]
    )
    for marker in ("api_key", "secret", "token"):
        grammar["forbidden_payload_markers"].remove(marker)
    monkeypatch.setitem(
        M05_CONTRACT["credential_profile_contract"], "secure_store_reference_grammar", grammar
    )
    _update_dependency_attestation(monkeypatch, "m05_secure_store_grammar", grammar)
    context["credential_profiles_by_id"][IDS["cred"]]["secure_store_reference"] = (
        "secure-store://api_key"
    )
    _assert_contract_fault_in_both_paths(context)


def test_forbidden_secure_store_locator_is_persisted_data_failure():
    context = fixture_context()
    context["credential_profiles_by_id"][IDS["cred"]]["secure_store_reference"] = (
        "secure-store://api_key"
    )
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("exchange_id", "changed_exchange"),
        ("adapter_family_id", "changed_adapter"),
        ("capability_discovery_policy", "changed_policy"),
        ("supported_environments", ["PAPER", "TESTNET"]),
    ],
)
def test_exchange_registry_and_attestation_coordinated_drift_fails_closed(
    monkeypatch, field, value
):
    context = fixture_context("PAPER")
    entries = copy.deepcopy(M05_CONTRACT["exchange_registry_contract"]["entries"])
    entries[0][field] = value
    monkeypatch.setitem(M05_CONTRACT["exchange_registry_contract"], "entries", entries)
    _update_dependency_attestation(monkeypatch, "m05_exchange_registry_entries", entries)
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize(
    ("dependency_id", "reference_name", "field", "value"),
    [
        ("m05_account_snapshot_hash", "account_capability_snapshot", "domain_separator", "drift"),
        ("m05_catalog_hash", "instrument_catalog_snapshot", "input_fields", ["drift"]),
        ("m05_universe_hash", "trading_universe", "canonicalization", "drift"),
        ("m05_catalog_hash", "instrument_catalog_snapshot", "encoding", "drift"),
        ("m05_universe_hash", "trading_universe", "digest_format", "drift"),
    ],
)
def test_hash_definition_and_both_attestations_coordinated_drift_fails_closed(
    monkeypatch, dependency_id, reference_name, field, value
):
    context = fixture_context()
    reference = CONTRACT["canonical_hash_definition_refs"][reference_name]
    definition = copy.deepcopy(resolve_canonical_pointer(reference, dict))
    definition[field] = value
    parts = reference["json_pointer"].strip("/").split("/")
    parent = M05_CONTRACT
    for part in parts[:-1]:
        parent = parent[part]
    monkeypatch.setitem(parent, parts[-1], definition)
    references = copy.deepcopy(CONTRACT["canonical_hash_definition_refs"])
    references[reference_name]["content_fingerprint_sha256"] = canonical_fingerprint(definition)
    monkeypatch.setitem(CONTRACT, "canonical_hash_definition_refs", references)
    _update_dependency_attestation(monkeypatch, dependency_id, definition)
    _assert_contract_fault_in_both_paths(context)


def test_dependency_resolver_uses_immutable_identity_and_returns_deep_frozen_value():
    entries = resolve_canonical_dependency("m05_exchange_registry_entries")
    assert isinstance(entries, tuple)
    assert isinstance(entries[0], MappingProxyType)
    with pytest.raises(TypeError):
        entries[0]["display_name"] = "changed"
    with pytest.raises(KeyError):
        resolve_canonical_dependency("not-inventory")


def test_dependency_root_consumers_equal_manifest_and_executable_registry():
    manifest = CONTRACT["canonical_dependency_root_manifest"]
    grouped = {dependency_id: [] for dependency_id in EXPECTED_CANONICAL_DEPENDENCIES}
    for consumer, dependency_id in EXPECTED_CANONICAL_DEPENDENCY_CONSUMER_BINDINGS.items():
        grouped[dependency_id].append(consumer)
    assert CONTRACT["canonical_dependency_consumer_bindings"] == dict(
        EXPECTED_CANONICAL_DEPENDENCY_CONSUMER_BINDINGS
    )
    for dependency_id, consumers in grouped.items():
        assert tuple(consumers) == EXPECTED_CANONICAL_DEPENDENCIES[dependency_id]["consumers"]
        assert manifest[dependency_id]["consumers"] == consumers


def _redirect_exchange_attestation(monkeypatch, alternate_entries, *, contract=None):
    monkeypatch.setitem(M05_CONTRACT, "alternate_exchange_entries", alternate_entries)
    references = copy.deepcopy(CONTRACT["canonical_cross_contract_registry_refs"])
    reference = references["m05_exchange_registry_entries"]
    reference["contract"] = contract or "exchange_accounts_and_instruments.json"
    reference["json_pointer"] = "/alternate_exchange_entries"
    reference["content_fingerprint_sha256"] = canonical_fingerprint(alternate_entries)
    monkeypatch.setitem(CONTRACT, "canonical_cross_contract_registry_refs", references)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_name", "Alternate paper"),
        ("capability_discovery_policy", "ALTERNATE_DISCOVERY"),
        ("supported_environments", ["PAPER", "TESTNET"]),
    ],
)
def test_alternate_exchange_pointer_cannot_acquire_authority(monkeypatch, field, value):
    context = fixture_context("PAPER")
    alternate = copy.deepcopy(M05_CONTRACT["exchange_registry_contract"]["entries"])
    alternate[0][field] = value
    _redirect_exchange_attestation(monkeypatch, alternate)
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize("fault", ["wrong_contract", "wrong_pointer", "updated_fingerprint"])
def test_exchange_specialized_attestation_cannot_choose_pointer_authority(monkeypatch, fault):
    context = fixture_context("PAPER")
    alternate = copy.deepcopy(M05_CONTRACT["exchange_registry_contract"]["entries"])
    alternate[0]["display_name"] = "Alternate paper"
    if fault == "wrong_contract":
        references = copy.deepcopy(CONTRACT["canonical_cross_contract_registry_refs"])
        references["m05_exchange_registry_entries"]["contract"] = (
            "environment_and_product_capabilities.json"
        )
        monkeypatch.setitem(CONTRACT, "canonical_cross_contract_registry_refs", references)
    elif fault == "wrong_pointer":
        _redirect_exchange_attestation(monkeypatch, alternate)
    else:
        references = copy.deepcopy(CONTRACT["canonical_cross_contract_registry_refs"])
        references["m05_exchange_registry_entries"]["content_fingerprint_sha256"] = (
            canonical_fingerprint(alternate)
        )
        monkeypatch.setitem(CONTRACT, "canonical_cross_contract_registry_refs", references)
    _assert_contract_fault_in_both_paths(context)


def test_alternate_exchange_adapter_family_cannot_redirect_fully_synchronized_graph(monkeypatch):
    context = fixture_context("PAPER")
    alternate = copy.deepcopy(M05_CONTRACT["exchange_registry_contract"]["entries"])
    alternate[0]["adapter_family_id"] = "alternate_adapter_family"
    for route_map in (context["market_data_routes_by_id"], context["execution_routes_by_id"]):
        for route in route_map.values():
            route["adapter_family_id"] = "alternate_adapter_family"
    instrument = context["instruments_by_id"][IDS["instr"]]
    instrument["source_adapter_family_id"] = "alternate_adapter_family"
    context["catalogs_by_id"][IDS["icat"]]["adapter_family_id"] = "alternate_adapter_family"
    context["account_capability_snapshots_by_id"][IDS["capsnap"]]["adapter_family_id"] = (
        "alternate_adapter_family"
    )
    identity = context["external_identity_snapshots_by_account_id"][IDS["xacc"]]
    identity["adapter_version_source"] = "alternate_adapter_family"
    # Rehash the synchronized persisted graph before the specialized ref is redirected.
    rehash_m05_projections(context)
    _redirect_exchange_attestation(monkeypatch, alternate)
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request_for(operation), context)
        assert result["allowed"] is False
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


def _hash_with_explicit_definition(definition, record):
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


@pytest.mark.parametrize(
    ("reference_name", "context_map", "record_id"),
    [
        ("account_capability_snapshot", "account_capability_snapshots_by_id", IDS["capsnap"]),
        ("instrument_catalog_snapshot", "catalogs_by_id", IDS["icat"]),
        ("trading_universe", "universes_by_id", IDS["univ"]),
    ],
)
def test_alternate_hash_pointer_and_rehashed_record_cannot_acquire_authority(
    monkeypatch, reference_name, context_map, record_id
):
    context = fixture_context()
    original_ref = CONTRACT["canonical_hash_definition_refs"][reference_name]
    alternate = copy.deepcopy(resolve_canonical_pointer(original_ref, dict))
    alternate["domain_separator"] += ".alternate"
    alternate_pointer = f"/alternate_{reference_name}_hash_definition"
    monkeypatch.setitem(M05_CONTRACT, alternate_pointer[1:], alternate)
    references = copy.deepcopy(CONTRACT["canonical_hash_definition_refs"])
    references[reference_name]["json_pointer"] = alternate_pointer
    references[reference_name]["content_fingerprint_sha256"] = canonical_fingerprint(alternate)
    monkeypatch.setitem(CONTRACT, "canonical_hash_definition_refs", references)
    context[context_map][record_id]["content_hash"] = _hash_with_explicit_definition(
        alternate, context[context_map][record_id]
    )
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize(
    "fault",
    [
        "contract",
        "pointer",
        "dependency_id",
        "fingerprint",
        "extra_name",
        "missing_name",
        "swap_account_catalog",
        "swap_catalog_universe",
    ],
)
def test_hash_specialized_attestation_drift_is_machine_fault(monkeypatch, fault):
    context = fixture_context()
    references = copy.deepcopy(CONTRACT["canonical_hash_definition_refs"])
    if fault == "contract":
        references["account_capability_snapshot"]["contract"] = (
            "environment_and_product_capabilities.json"
        )
    elif fault == "pointer":
        references["account_capability_snapshot"]["json_pointer"] = "/decimal_policy"
    elif fault == "dependency_id":
        references["account_capability_snapshot"]["dependency_id"] = "m05_catalog_hash"
    elif fault == "fingerprint":
        references["account_capability_snapshot"]["content_fingerprint_sha256"] = "0" * 64
    elif fault == "extra_name":
        references["alternate"] = copy.deepcopy(references["account_capability_snapshot"])
    elif fault == "missing_name":
        references.pop("account_capability_snapshot")
    elif fault == "swap_account_catalog":
        references["account_capability_snapshot"] = copy.deepcopy(
            references["instrument_catalog_snapshot"]
        )
    else:
        references["instrument_catalog_snapshot"] = copy.deepcopy(references["trading_universe"])
    monkeypatch.setitem(CONTRACT, "canonical_hash_definition_refs", references)
    _assert_contract_fault_in_both_paths(context)


def _update_m04_root_attestation(monkeypatch, dependency_id, value):
    manifest = copy.deepcopy(CONTRACT["canonical_dependency_root_manifest"])
    manifest[dependency_id]["content_fingerprint_sha256"] = canonical_fingerprint(value)
    monkeypatch.setitem(CONTRACT, "canonical_dependency_root_manifest", manifest)


@pytest.mark.parametrize(
    "fault",
    [
        "product_capability",
        "signed_capability",
        "current_edition_id",
        "all_edition_ids",
        "paper_execution",
        "testnet_execution",
        "live_execution",
        "feature_flag",
        "source",
        "fail_closed",
    ],
)
def test_m04_product_capability_policy_drift_is_machine_fault(monkeypatch, fault):
    context = fixture_context()
    current = copy.deepcopy(M04_CONTRACT["current_product_edition"])
    product_container = copy.deepcopy(M04_CONTRACT["ProductCapabilities"])
    product = product_container["current_edition_capability_policy"]
    signed = copy.deepcopy(M04_CONTRACT["current_edition_signed_payload_policy"])
    dependency_id = "m04_product_capability_policy"
    changed = product
    if fault == "product_capability":
        product["capability_set"].remove("TESTNET_PRIVATE_EXECUTION_AFTER_READINESS")
    elif fault == "signed_capability":
        signed["capability_set"].remove("TESTNET_PRIVATE_EXECUTION_AFTER_READINESS")
        dependency_id, changed = "m04_signed_payload_policy", signed
    elif fault == "current_edition_id":
        current["edition_id"] = "OTHER"
        dependency_id, changed = "m04_current_product_edition", current
    elif fault == "all_edition_ids":
        current["edition_id"] = product["edition_id"] = signed["edition_id"] = "OTHER"
        monkeypatch.setitem(M04_CONTRACT, "current_product_edition", current)
        monkeypatch.setitem(M04_CONTRACT, "current_edition_signed_payload_policy", signed)
    elif fault == "paper_execution":
        product["environment_capabilities"]["PAPER"]["local_execution_allowed"] = False
    elif fault == "testnet_execution":
        product["environment_capabilities"]["TESTNET"][
            "private_execution_allowed_when_trust_state_valid"
        ] = False
    elif fault == "live_execution":
        product["environment_capabilities"]["LIVE"]["private_execution_allowed"] = True
    elif fault == "feature_flag":
        product["feature_flags"]["live_execution"] = True
    elif fault == "source":
        product["source"] = "OTHER"
    else:
        product["fail_closed_policy"] = "OPEN"
    monkeypatch.setitem(M04_CONTRACT, "current_product_edition", current)
    monkeypatch.setitem(M04_CONTRACT, "ProductCapabilities", product_container)
    monkeypatch.setitem(M04_CONTRACT, "current_edition_signed_payload_policy", signed)
    _update_m04_root_attestation(monkeypatch, dependency_id, changed)
    _assert_contract_fault_in_both_paths(context)


@pytest.mark.parametrize(
    ("environment", "mutation"),
    [
        ("TESTNET", lambda record: record.update(edition="CURRENT")),
        ("TESTNET", lambda record: record.update(edition="OTHER_EDITION")),
        (
            "PAPER",
            lambda record: record.update(
                allowed_operations=["TESTNET_PRIVATE_EXECUTION_AFTER_READINESS"]
            ),
        ),
        ("TESTNET", lambda record: record.update(allowed_operations=["PAPER_LOCAL_SIMULATION"])),
        (
            "TESTNET",
            lambda record: record.update(allowed_operations=["LIVE_VISIBLE_LOCKED_ONLY"]),
        ),
        ("TESTNET", lambda record: record.update(allowed_operations=[])),
        (
            "TESTNET",
            lambda record: record.update(
                allowed_operations=[
                    "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
                    "PAPER_LOCAL_SIMULATION",
                ]
            ),
        ),
        ("PAPER", lambda record: record.update(execution_enabled=False)),
        ("TESTNET", lambda record: record.update(execution_enabled=False)),
        ("LIVE", lambda record: record.update(execution_enabled=True)),
        (
            "TESTNET",
            lambda record: record.update(
                allowed_operations=[
                    "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
                    "TESTNET_PRIVATE_EXECUTION_AFTER_READINESS",
                ]
            ),
        ),
    ],
)
def test_noncanonical_persisted_product_capabilities_is_trusted_context_invalid(
    environment, mutation
):
    context = fixture_context(environment if environment != "LIVE" else "TESTNET")
    if environment == "LIVE":
        context["product_capabilities_by_environment"]["LIVE"] = {
            "environment": "LIVE",
            "execution_enabled": False,
            "allowed_operations": ["LIVE_VISIBLE_LOCKED_ONLY"],
            "edition": "CRYPTOHUNTER_TESTNET_EDITION",
        }
    mutation(context["product_capabilities_by_environment"][environment])
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(
            "VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context
        )
        assert result["denial_code"] == "TRUSTED_CONTEXT_INVALID"


def _coordinated_local_policy_mutation(monkeypatch, policy_id, changed):
    manifest = copy.deepcopy(CONTRACT["m06_local_authority_policy_manifest"])
    manifest[policy_id] = copy.deepcopy(changed)
    monkeypatch.setitem(CONTRACT, "m06_local_authority_policy_manifest", manifest)
    if policy_id == "execution_readiness_max_age_seconds":
        route_contract = copy.deepcopy(CONTRACT["execution_route_contract"])
        route_contract["execution_readiness_max_age_seconds"] = changed
        monkeypatch.setitem(CONTRACT, "execution_route_contract", route_contract)
    else:
        monkeypatch.setitem(CONTRACT, policy_id, copy.deepcopy(changed))


def test_snapshot_freshness_policy_cannot_be_coordinately_weakened(monkeypatch):
    context = fixture_context()
    snapshot = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    snapshot.update(
        observed_at_utc="2025-12-31T23:50:00Z",
        effective_at_utc="2025-12-31T23:50:30Z",
        stale_after_utc="2026-01-01T00:05:00Z",
    )
    rehash_m05_projections(context)
    changed = copy.deepcopy(CONTRACT["account_capability_snapshot_policy"])
    changed["max_age_seconds_by_environment"]["TESTNET"] = 3600
    _coordinated_local_policy_mutation(monkeypatch, "account_capability_snapshot_policy", changed)
    _assert_contract_fault_in_both_paths(context)


def test_old_snapshot_is_ordinary_capability_snapshot_denial():
    context = fixture_context()
    snapshot = context["account_capability_snapshots_by_id"][IDS["capsnap"]]
    snapshot.update(
        observed_at_utc="2025-12-31T23:50:00Z",
        effective_at_utc="2025-12-31T23:50:30Z",
        stale_after_utc="2026-01-01T00:05:00Z",
    )
    rehash_m05_projections(context)
    for execute in (dispatcher, run_direct_call_graph):
        assert (
            execute("VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context)[
                "denial_code"
            ]
            == "CAPABILITY_SNAPSHOT_BLOCKED"
        )


def test_execution_readiness_policy_cannot_be_coordinately_weakened(monkeypatch):
    context = fixture_context()
    context["route_readiness_by_id"][IDS["xroute"]]["observed_at"] = "2025-12-31T23:59:00Z"
    _coordinated_local_policy_mutation(monkeypatch, "execution_readiness_max_age_seconds", 300)
    _assert_contract_fault_in_both_paths(context)


def test_old_execution_readiness_is_ordinary_not_ready_denial():
    context = fixture_context()
    context["route_readiness_by_id"][IDS["xroute"]]["observed_at"] = "2025-12-31T23:59:00Z"
    for execute in (dispatcher, run_direct_call_graph):
        assert (
            execute("VALIDATE_ROUTE_READINESS", request_for("VALIDATE_ROUTE_READINESS"), context)[
                "denial_code"
            ]
            == "EXECUTION_ROUTE_NOT_READY"
        )


def test_endpoint_registry_coordinated_weakening_is_machine_fault(monkeypatch):
    context = fixture_context()
    changed = copy.deepcopy(CONTRACT["endpoint_class_registry"])
    changed["TESTNET_PRIVATE_DATA"].update(
        environment="PAPER", access_scope="PUBLIC", allowed_route_kinds=["EXECUTION"]
    )
    route = context["execution_routes_by_id"][IDS["xroute"]]
    route["environment"] = "PAPER"
    _coordinated_local_policy_mutation(monkeypatch, "endpoint_class_registry", changed)
    _assert_contract_fault_in_both_paths(context)


def test_authorization_dependencies_coordinated_weakening_is_machine_fault(monkeypatch):
    context = fixture_context()
    changed = copy.deepcopy(CONTRACT["authorization_dependencies_by_environment"])
    removed = changed["TESTNET"].pop()
    context["execution_routes_by_id"][IDS["xroute"]]["authorization_dependencies"].remove(removed)
    _coordinated_local_policy_mutation(
        monkeypatch, "authorization_dependencies_by_environment", changed
    )
    _assert_contract_fault_in_both_paths(context)


def _assert_protocol_fault_both_paths(request, context, operation="ACTIVATE_STRATEGY_INSTANCE"):
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, context)
        assert result["allowed"] is False
        assert result["denial_code"] == "CONTRACT_INCONSISTENT"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "fault",
    [
        "missing_authorization",
        "missing_strategy",
        "early_terminal",
        "missing_request",
        "missing_context",
        "duplicate",
        "unknown",
        "swap",
        "missing_terminal",
        "terminal_too_early",
        "after_terminal",
        "wrong_operation_validator",
        "empty",
        "not_array",
    ],
)
def test_operation_graph_mutations_fail_closed_before_execution(monkeypatch, fault):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    request = request_for(operation)
    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    context["instruments_by_id"][IDS["instr"]]["trading_status"] = "HALTED"
    if fault == "missing_context":
        product = context["product_capabilities_by_environment"]["TESTNET"]
        product["edition"] = "OTHER_EDITION"
    if fault == "missing_request":
        request["authority"] = "DesktopShell"
    graphs = copy.deepcopy(CONTRACT["operation_validator_call_graph"])
    graph = graphs[operation]
    if fault == "missing_authorization":
        graph.remove("validate_authorization_operability")
    elif fault == "missing_strategy":
        graph.remove("validate_strategy_execution_operability")
    elif fault == "early_terminal":
        graph.remove("validate_activation")
        graph.insert(3, "validate_activation")
    elif fault == "missing_request":
        graph.remove("validate_request")
    elif fault == "missing_context":
        graph.remove("validate_context")
    elif fault == "duplicate":
        graph.insert(2, graph[2])
    elif fault == "unknown":
        graph.insert(2, "validate_not_real")
    elif fault == "swap":
        graph[2], graph[3] = graph[3], graph[2]
    elif fault == "missing_terminal":
        graph.remove("validate_activation")
    elif fault == "terminal_too_early":
        graph.remove("validate_activation")
        graph.insert(2, "validate_activation")
    elif fault == "after_terminal":
        graph.append("validate_context")
    elif fault == "wrong_operation_validator":
        graph[-1] = "validate_retirement"
    elif fault == "empty":
        graphs[operation] = []
    else:
        graphs[operation] = "validate_request"
    monkeypatch.setitem(CONTRACT, "operation_validator_call_graph", graphs)
    _assert_protocol_fault_both_paths(request, context, operation)


def test_unchanged_protocol_rejects_desktop_authority_as_request_error():
    request = request_for("ACTIVATE_STRATEGY_INSTANCE")
    request["authority"] = "DesktopShell"
    for execute in (dispatcher, run_direct_call_graph):
        assert (
            execute("ACTIVATE_STRATEGY_INSTANCE", request, fixture_context())["denial_code"]
            == "REQUEST_SCHEMA_INVALID"
        )


def test_request_schema_cannot_coordinately_change_authority(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    registry = copy.deepcopy(CONTRACT["operation_registry"])
    registry[operation]["request_schema"]["constants"]["authority"] = "DesktopShell"
    monkeypatch.setitem(CONTRACT, "operation_registry", registry)
    request = request_for(operation)
    request["authority"] = "DesktopShell"
    _assert_protocol_fault_both_paths(request, fixture_context(), operation)


def test_coordinated_graph_and_denial_metadata_drift_still_fails_closed(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    graphs = copy.deepcopy(CONTRACT["operation_validator_call_graph"])
    graphs[operation].remove("validate_authorization_operability")
    validator_denials = copy.deepcopy(CONTRACT["validator_denial_registry"])
    validator_denials.pop("validate_authorization_operability")
    allowed = copy.deepcopy(CONTRACT["allowed_denials_by_operation"])
    allowed[operation] = [
        denial
        for denial in allowed[operation]
        if denial not in {"ACCOUNT_READINESS_BLOCKED", "CAPABILITY_SNAPSHOT_BLOCKED"}
    ]
    monkeypatch.setitem(CONTRACT, "operation_validator_call_graph", graphs)
    monkeypatch.setitem(CONTRACT, "validator_denial_registry", validator_denials)
    monkeypatch.setitem(CONTRACT, "allowed_denials_by_operation", allowed)
    _assert_protocol_fault_both_paths(request_for(operation), fixture_context(), operation)


def test_coordinated_event_mapping_drift_still_fails_closed(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    success = copy.deepcopy(CONTRACT["success_event_by_operation"])
    denial = copy.deepcopy(CONTRACT["denial_event_by_operation"])
    registry = copy.deepcopy(CONTRACT["operation_registry"])
    success[operation] = "ALTERED_SUCCESS"
    denial[operation] = "ALTERED_DENIAL"
    registry[operation]["success_event_ref"] = (
        "success_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE"
    )
    registry[operation]["denial_event_ref"] = (
        "denial_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE"
    )
    monkeypatch.setitem(CONTRACT, "success_event_by_operation", success)
    monkeypatch.setitem(CONTRACT, "denial_event_by_operation", denial)
    monkeypatch.setitem(CONTRACT, "operation_registry", registry)
    _assert_protocol_fault_both_paths(request_for(operation), fixture_context(), operation)


@pytest.mark.parametrize(
    ("reference_field", "incorrect_reference"),
    [
        (
            "success_event_ref",
            "success_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE",
        ),
        (
            "denial_event_ref",
            "denial_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE",
        ),
    ],
)
def test_operation_registry_event_reference_is_exact_bound(
    monkeypatch, reference_field, incorrect_reference
):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    registry = copy.deepcopy(CONTRACT["operation_registry"])
    assert reference_field in registry[operation]
    registry[operation][reference_field] = incorrect_reference
    monkeypatch.setitem(CONTRACT, "operation_registry", registry)
    _assert_protocol_fault_both_paths(request_for(operation), fixture_context(), operation)


def test_mutable_requests_cache_cannot_override_immutable_request_authority(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    requests = copy.deepcopy(REQUESTS)
    requests[operation]["constants"]["authority"] = "DesktopShell"
    monkeypatch.setitem(REQUESTS, operation, requests[operation])
    request = request_for(operation)
    request["authority"] = "DesktopShell"
    _assert_protocol_fault_both_paths(request, fixture_context(), operation)


def test_special_event_drift_still_emits_immutable_contract_audit_event(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    audit = copy.deepcopy(CONTRACT["audit_event_contract"])
    audit["special_events"]["CONTRACT_INCONSISTENT"] = "ALTERED_EVENT"
    monkeypatch.setitem(CONTRACT, "audit_event_contract", audit)
    _assert_protocol_fault_both_paths(request_for(operation), fixture_context(), operation)


def test_trusted_implementation_dispatch_preserves_ordinary_denials():
    request = request_for("ACTIVATE_STRATEGY_INSTANCE")
    request["authority"] = "DesktopShell"
    assert (
        dispatcher("ACTIVATE_STRATEGY_INSTANCE", request, fixture_context())["denial_code"]
        == "REQUEST_SCHEMA_INVALID"
    )

    context = fixture_context()
    context["product_capabilities_by_environment"]["TESTNET"]["edition"] = "OTHER_EDITION"
    validation = validate_context({}, context, "ACTIVATE_STRATEGY_INSTANCE")
    assert validation["denial_code"] == "TRUSTED_CONTEXT_INVALID"
    assert (
        "TRUSTED_CONTEXT_INVALID"
        in EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["validator_denial_registry"]["validate_context"][
            "denials"
        ]
    )
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "TRUSTED_CONTEXT_INVALID"
    )

    context = fixture_context()
    context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "ACCOUNT_READINESS_BLOCKED"
    )

    context = fixture_context()
    context["instruments_by_id"][IDS["instr"]]["trading_status"] = "HALTED"
    rehash_m05_projections(context)
    assert (
        dispatcher(
            "ACTIVATE_STRATEGY_INSTANCE", request_for("ACTIVATE_STRATEGY_INSTANCE"), context
        )["denial_code"]
        == "INSTRUMENT_SCOPE_MISMATCH"
    )


EXPECTED_EXECUTABLE_TRUST_BOUNDARY = deep_freeze(
    {
        "trusted_computing_base": [
            "Python module source loaded from the verified repository/package",
            "module namespace and function objects",
            "Python interpreter and standard library",
            "process memory and closure cells",
            "dispatcher and direct-call entrypoint implementations",
            "validator, resolver and helper implementations",
        ],
        "untrusted_inputs": [
            "M0.6 machine CONTRACT data",
            "M0.4 and M0.5 canonical JSON contracts",
            "request data",
            "persisted trusted-validation context",
            "machine-readable schemas and manifests",
            "mutable request-schema cache",
            "external canonical pointers and their metadata",
        ],
        "out_of_scope": [
            "arbitrary in-process Python code execution",
            "monkeypatching module functions",
            "replacing function __code__",
            "mutating closure cell contents",
            "rebinding dispatcher or executor globals",
            "memory corruption",
            "interpreter compromise",
        ],
        "assumptions": {
            "trusted_code_immutable_by_deployment_assumption": True,
            "in_process_code_mutation_out_of_scope": True,
            "machine_contract_data_untrusted": True,
            "request_and_persisted_context_untrusted": True,
            "runtime_code_integrity_requires_external_trust_anchor": True,
        },
        "deferred_code_integrity_requirement": (
            "Runtime code-integrity enforcement, if required, must be implemented by an "
            "external bootstrap or deployment trust mechanism that validates signed/hash-pinned "
            "artifacts before importing the application module."
        ),
    }
)


def test_executable_trust_boundary_is_exact_and_machine_readable():
    assert deep_freeze(CONTRACT["executable_trust_boundary"]) == EXPECTED_EXECUTABLE_TRUST_BOUNDARY


def test_executable_trust_boundary_rejects_same_process_root_of_trust_claims():
    documentation = (PATH.with_suffix(".md")).read_text(encoding="utf-8")
    serialized_contract = PATH.read_text(encoding="utf-8")
    forbidden_claims = (
        "module-global rebinding cannot replace authority",
        "closure cells are protected",
        "function-object identity proves code integrity",
        "same-process self-attestation is a root of trust",
    )
    for claim in forbidden_claims:
        assert claim not in documentation
        assert claim not in serialized_contract


def test_dispatcher_and_direct_call_are_functionally_equivalent_trusted_entrypoints():
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    request = request_for(operation)
    context = fixture_context()
    assert dispatcher(operation, request, context) == run_direct_call_graph(
        operation, request, context
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing_denial_events",
        "null_denial_events",
        "array_denial_events",
        "string_denial_events",
        "integer_denial_events",
        "missing_success_events",
        "null_success_events",
        "array_success_events",
        "missing_audit_contract",
        "null_audit_contract",
        "array_audit_contract",
        "missing_special_events",
        "null_special_events",
        "null_operation_registry",
        "null_operation_graph",
        "null_validator_denials",
        "null_allowed_denials",
    ],
)
def test_structurally_malformed_machine_event_protocol_fails_closed(monkeypatch, fault):
    if fault == "missing_denial_events":
        monkeypatch.delitem(CONTRACT, "denial_event_by_operation")
    elif fault == "null_denial_events":
        monkeypatch.setitem(CONTRACT, "denial_event_by_operation", None)
    elif fault == "array_denial_events":
        monkeypatch.setitem(CONTRACT, "denial_event_by_operation", [])
    elif fault == "string_denial_events":
        monkeypatch.setitem(CONTRACT, "denial_event_by_operation", "broken")
    elif fault == "integer_denial_events":
        monkeypatch.setitem(CONTRACT, "denial_event_by_operation", 1)
    elif fault == "missing_success_events":
        monkeypatch.delitem(CONTRACT, "success_event_by_operation")
    elif fault == "null_success_events":
        monkeypatch.setitem(CONTRACT, "success_event_by_operation", None)
    elif fault == "array_success_events":
        monkeypatch.setitem(CONTRACT, "success_event_by_operation", [])
    elif fault == "missing_audit_contract":
        monkeypatch.delitem(CONTRACT, "audit_event_contract")
    elif fault == "null_audit_contract":
        monkeypatch.setitem(CONTRACT, "audit_event_contract", None)
    elif fault == "array_audit_contract":
        monkeypatch.setitem(CONTRACT, "audit_event_contract", [])
    elif fault == "missing_special_events":
        audit = copy.deepcopy(CONTRACT["audit_event_contract"])
        audit.pop("special_events")
        monkeypatch.setitem(CONTRACT, "audit_event_contract", audit)
    elif fault == "null_special_events":
        audit = copy.deepcopy(CONTRACT["audit_event_contract"])
        audit["special_events"] = None
        monkeypatch.setitem(CONTRACT, "audit_event_contract", audit)
    elif fault == "null_operation_registry":
        monkeypatch.setitem(CONTRACT, "operation_registry", None)
    elif fault == "null_operation_graph":
        monkeypatch.setitem(CONTRACT, "operation_validator_call_graph", None)
    elif fault == "null_validator_denials":
        monkeypatch.setitem(CONTRACT, "validator_denial_registry", None)
    else:
        monkeypatch.setitem(CONTRACT, "allowed_denials_by_operation", None)
    _assert_protocol_fault_both_paths(request_for("ACTIVATE_STRATEGY_INSTANCE"), fixture_context())


def test_coordinated_all_event_metadata_drift_uses_independent_contract_fault_event(monkeypatch):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    success_events = copy.deepcopy(CONTRACT["success_event_by_operation"])
    denial_events = copy.deepcopy(CONTRACT["denial_event_by_operation"])
    registry = copy.deepcopy(CONTRACT["operation_registry"])
    audit = copy.deepcopy(CONTRACT["audit_event_contract"])
    success_events[operation] = "ALTERED_SUCCESS"
    denial_events[operation] = "ALTERED_DENIAL"
    registry[operation]["success_event_ref"] = (
        "success_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE"
    )
    registry[operation]["denial_event_ref"] = (
        "denial_event_by_operation.DEACTIVATE_STRATEGY_INSTANCE"
    )
    audit["special_events"]["CONTRACT_INCONSISTENT"] = "ALTERED_CONTRACT_FAULT"
    monkeypatch.setitem(CONTRACT, "success_event_by_operation", success_events)
    monkeypatch.setitem(CONTRACT, "denial_event_by_operation", denial_events)
    monkeypatch.setitem(CONTRACT, "operation_registry", registry)
    monkeypatch.setitem(CONTRACT, "audit_event_contract", audit)
    _assert_protocol_fault_both_paths(request_for(operation), fixture_context(), operation)


@pytest.mark.parametrize(
    "operation",
    [None, 0, 1.5, False, [], {}, set(), ("ACTIVATE_STRATEGY_INSTANCE",)],
)
def test_non_string_outer_operation_is_safely_normalized_and_denied(operation):
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, {}, {})
        assert result["allowed"] is False
        assert result["denial_code"] == "UNKNOWN_OPERATION"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_UNKNOWN_OPERATION_DENIED"
        assert result["operation"] is None
        assert isinstance(result, MappingProxyType)


@pytest.mark.parametrize(
    "operation", ["", "UNKNOWN", "SUBMIT_ORDER", "ACTIVATE_STRATEGY_INSTANCE "]
)
def test_unknown_string_outer_operation_is_preserved_and_denied(operation):
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, {}, {})
        assert result["denial_code"] == "UNKNOWN_OPERATION"
        assert result["audit_event_type"] == "STRATEGY_ROUTING_UNKNOWN_OPERATION_DENIED"
        assert result["operation"] == operation
        assert isinstance(result, MappingProxyType)


@pytest.mark.parametrize("request_operation", [[], None, {}, "OTHER"])
def test_malformed_request_operation_is_request_schema_invalid_not_unknown(request_operation):
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    request = request_for(operation)
    request["operation"] = request_operation
    for execute in (dispatcher, run_direct_call_graph):
        result = execute(operation, request, fixture_context())
        assert result["denial_code"] == "REQUEST_SCHEMA_INVALID"
        assert result["operation"] == operation
        assert (
            result["audit_event_type"]
            == EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["denial_event_by_operation"][operation]
        )


def test_ordinary_decisions_preserve_expected_success_and_denial_events():
    operation = "ACTIVATE_STRATEGY_INSTANCE"
    expected_denial_event = EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["denial_event_by_operation"][
        operation
    ]
    expected_success_event = EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["success_event_by_operation"][
        operation
    ]

    desktop_request = request_for(operation)
    desktop_request["authority"] = "DesktopShell"
    disabled_context = fixture_context()
    disabled_context["accounts_by_id"][IDS["xacc"]]["lifecycle_state"] = "DISABLED"
    halted_context = fixture_context()
    halted_context["instruments_by_id"][IDS["instr"]]["trading_status"] = "HALTED"
    rehash_m05_projections(halted_context)

    cases = (
        (desktop_request, fixture_context(), "REQUEST_SCHEMA_INVALID", expected_denial_event),
        (
            request_for(operation),
            disabled_context,
            "ACCOUNT_READINESS_BLOCKED",
            expected_denial_event,
        ),
        (
            request_for(operation),
            halted_context,
            "INSTRUMENT_SCOPE_MISMATCH",
            expected_denial_event,
        ),
    )
    for execute in (dispatcher, run_direct_call_graph):
        for request, context, denial_code, event in cases:
            result = execute(operation, request, context)
            assert result["denial_code"] == denial_code
            assert result["audit_event_type"] == event
        successful = execute(operation, request_for(operation), fixture_context())
        assert successful["allowed"] is True
        assert successful["audit_event_type"] == expected_success_event
        unknown = execute("SUBMIT_ORDER", {}, {})
        assert unknown["denial_code"] == "UNKNOWN_OPERATION"
        assert (
            unknown["audit_event_type"]
            == EXPECTED_EXECUTABLE_OPERATION_PROTOCOL["special_events"]["UNKNOWN_OPERATION"]
        )
        assert isinstance(unknown, MappingProxyType)


def test_fail_closed_decision_boundary_audit_is_exact():
    assert CONTRACT["executable_operation_protocol_audit"]["decision_boundary"] == {
        "all_decision_event_mappings_source": "immutable EXPECTED_EXECUTABLE_OPERATION_PROTOCOL",
        "machine_event_mappings_role": "untrusted attestations only",
        "contract_fault_decision_reads_machine_event_metadata": False,
        "unknown_operation_decision_reads_machine_event_metadata": False,
        "non_string_outer_operation": "normalized to null and denied as UNKNOWN_OPERATION",
        "malformed_machine_event_registry": "CONTRACT_INCONSISTENT, never an exception",
    }
