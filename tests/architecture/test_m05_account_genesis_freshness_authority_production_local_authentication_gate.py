"""Statyczna część ostatniej bramki autoryzacji implementacji FreshnessAuthority."""
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
STEM = "m05_account_genesis_freshness_authority_production_local_authentication_gate"
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"


def load():
    return json.loads((DOCS / f"{STEM}.json").read_text(encoding="utf-8"))


def render(value):
    body = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    return f"# M0.5 — executable PRODUCTION_LOCAL authentication gate\n\n> Deterministyczna projekcja `{STEM}.json`. JSON jest źródłem prawdy.\n\n```json\n{body}```\n"


def test_projection_and_exact_peer_binding():
    value = load()
    assert (DOCS / f"{STEM}.md").read_text(encoding="utf-8") == render(value)
    auth = value["authentication_binding"]
    assert auth["method"] == "peer with an explicit pg_ident map"
    assert auth["identity_map"] == {
        "os_freshness_crypto_verifier": "freshness_crypto_verifier",
        "os_freshness_runtime": "freshness_runtime",
    }
    assert auth["hba_order"][-2:] == ["local <authority_database> all reject", "local all all reject"]
    assert auth["network_listeners"] is False
    assert auth["caller_supplied_identity_allowed"] is False


def test_exact_roles_and_final_flags():
    value = load()
    assert set(value["roles"]) == {
        "freshness_schema_owner", "freshness_function_owner", "freshness_admin",
        "freshness_crypto_verifier", "freshness_runtime", "freshness_reader",
    }
    assert all(not role["inherit"] for role in value["roles"].values())
    assert value["roles"]["freshness_crypto_verifier"]["direct_execute"] == ["prepare_verified_freshness_candidate"]
    assert value["roles"]["freshness_runtime"]["direct_execute"] == ["compare_and_advance", "reviewed reads"]
    assert value["final_flags"] == {
        "FreshnessAuthority_implementation_allowed_after_iteration": True,
        "FreshnessAuthority_implemented": False,
        "production_substrate_implemented": False,
        "PRODUCTION_LOCAL_RUNTIME_AVAILABLE": False,
        "ROOT_PROOF_ISSUER_IMPLEMENTED": False,
    }
