"""Ograniczony transport dowodów Linux; artefakt GitHub nie jest źródłem zaufania."""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile
import zlib

MAX_BUNDLE_BYTES = 256 * 1024
MAX_TRANSPORT_BYTES = 48 * 1024
MAX_ARCHIVE_BYTES = MAX_BUNDLE_BYTES + 64 * 1024
MAX_API_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_PAGES = 10
INGRESS_WORKFLOW = ".github/workflows/linux-evidence-ingress.yml"
POLICY_BRANCH = "stage9-linux-evidence-policy"
# Tożsamość aplikacji GitHub Actions na używanym tu api.github.com.
# Sam dodatni identyfikator innej aplikacji nie uwierzytelnia wymaganych kontroli.
GITHUB_ACTIONS_APP_ID = 15368
REQUIRED_PROTECTION_CHECKS = frozenset(
    {
        "Ruff format (all tracked files)",
        "Quality ratchet (combined coverage gate)",
        "linux-evidence-contract",
    }
)


class HandoffError(ValueError):
    """Transport lub konfiguracja nie spełnia zamkniętego kontraktu."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _bounded_response(url: str, token: str | None, limit: int) -> tuple[bytes, str | None]:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with build_opener(_NoRedirect).open(request, timeout=30) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise HandoffError("Odpowiedź przekracza limit rozmiaru")
            return body, None
    except HTTPError as exc:
        if exc.code == 302:
            return b"", exc.headers.get("Location")
        raise HandoffError("GitHub odmówił odczytu transportu lub konfiguracji") from None
    except (OSError, ValueError) as exc:
        raise HandoffError("Nie można odczytać transportu GitHub") from exc


class GitHubClient:
    def __init__(self, token: str, repository: str):
        if not token or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise HandoffError("Brak tożsamości API lub nieprawidłowe repozytorium")
        self.token = token
        self.prefix = f"https://api.github.com/repos/{repository}"

    def json(self, path: str) -> dict[str, Any]:
        raw, redirect = _bounded_response(self.prefix + path, self.token, MAX_API_BYTES)
        if redirect is not None:
            raise HandoffError("Nieoczekiwane przekierowanie metadanych API")
        value = _json_object(raw)
        return value

    def archive(self, artifact_id: int) -> bytes:
        raw, location = _bounded_response(
            f"{self.prefix}/actions/artifacts/{artifact_id}/zip", self.token, MAX_ARCHIVE_BYTES
        )
        if location is None:
            return raw
        parsed = urlsplit(location)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
            or not (
                host.endswith(".blob.core.windows.net")
                or host.endswith(".actions.githubusercontent.com")
                or host == "objects.githubusercontent.com"
            )
        ):
            raise HandoffError("Niezatwierdzony adres magazynu artefaktów")
        # Signed storage URL is transport only. Never forward the GitHub token.
        raw, redirected_again = _bounded_response(location, None, MAX_ARCHIVE_BYTES)
        if redirected_again is not None:
            raise HandoffError("Zbyt wiele przekierowań artefaktu")
        return raw


def _json_object(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise HandoffError("Powtórzone pole JSON")
            value[key] = item
        return value

    def constant(value: str) -> None:
        raise HandoffError("Niedozwolona stała JSON")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise HandoffError("Nieprawidłowy dokument JSON") from exc
    if not isinstance(value, dict):
        raise HandoffError("Dokument JSON musi być obiektem")
    return value


def ingress(raw: bytes, output: Path) -> None:
    """Zapisuje niezaufane bajty; nie deklaruje wykonania ani wyniku testów."""
    if not raw or len(raw) > MAX_BUNDLE_BYTES:
        raise HandoffError("Brak dowodu lub przekroczony limit 256 KiB")
    _json_object(raw)
    with output.open("xb") as stream:
        stream.write(raw)


def unpack_transport(raw: bytes) -> bytes:
    """Rozpakuj jeden strumień gzip z ograniczeniem pamięci i ścisłym Base64."""
    if not raw or len(raw) > MAX_TRANSPORT_BYTES:
        raise HandoffError("Brak transportu lub przekroczony limit 48 KiB")
    if not raw.startswith(b"gzb64:"):
        return raw
    encoded = raw[len(b"gzb64:") :]
    try:
        compressed = base64.b64decode(encoded, validate=True)
        if base64.b64encode(compressed) != encoded:
            raise HandoffError("Niejednoznaczne kodowanie Base64")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        unpacked = decoder.decompress(compressed, MAX_BUNDLE_BYTES + 1)
        if (
            len(unpacked) > MAX_BUNDLE_BYTES
            or not decoder.eof
            or decoder.unconsumed_tail
            or decoder.unused_data
        ):
            raise HandoffError("Nadmierny, niepełny lub wieloczęściowy strumień gzip")
        return unpacked
    except (binascii.Error, zlib.error, ValueError) as exc:
        raise HandoffError("Nieprawidłowy ograniczony transport gzip+Base64") from exc


def pack_bundle(bundle: Path, output: Path) -> None:
    with bundle.open("rb") as stream:
        raw = stream.read(MAX_BUNDLE_BYTES + 1)
    if not raw or len(raw) > MAX_BUNDLE_BYTES:
        raise HandoffError("Brak dowodu lub przekroczony limit 256 KiB")
    _json_object(raw)
    encoded = b"gzb64:" + base64.b64encode(gzip.compress(raw, mtime=0))
    if len(encoded) > MAX_TRANSPORT_BYTES:
        raise HandoffError("Skompresowany transport przekracza 48 KiB")
    with output.open("xb") as stream:
        stream.write(encoded)


def _enabled(value: object, expected: bool) -> bool:
    return isinstance(value, dict) and value.get("enabled") is expected


def _code_owners(client: GitHubClient, revision: str) -> set[str]:
    """Obsługiwany profil CODEOWNERS: jedna globalna reguła i jawne konta ludzi.

    Celowo nie interpretujemy przybliżeniem wzorców gitignore, wyjątków lub
    członkostwa zespołów. Nieobsługiwany profil pozostaje BLOCKED.
    """
    document = client.json(f"/contents/.github/CODEOWNERS?ref={revision}")
    encoded = document.get("content")
    if (
        document.get("type") != "file"
        or document.get("encoding") != "base64"
        or not isinstance(encoded, str)
        or len(encoded) > 96 * 1024
    ):
        raise HandoffError("Brak niezależnie sprawdzalnego pliku CODEOWNERS")
    try:
        raw = base64.b64decode("".join(encoded.splitlines()), validate=True)
        if len(raw) > 64 * 1024:
            raise ValueError
        rules = [line.split("#", 1)[0].strip() for line in raw.decode("utf-8").splitlines()]
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise HandoffError("Nieprawidłowy plik CODEOWNERS") from exc
    rules = [line for line in rules if line]
    fields = rules[0].split() if len(rules) == 1 else []
    if (
        not 3 <= len(fields) <= 7
        or fields[0] != "*"
        or any(
            re.fullmatch(r"@[A-Za-z0-9][A-Za-z0-9-]{0,38}", owner) is None for owner in fields[1:]
        )
    ):
        raise HandoffError(
            "CODEOWNERS wymaga jednej globalnej reguły z co najmniej dwoma właścicielami"
        )
    owners = {owner[1:].lower() for owner in fields[1:]}
    if len(owners) < 2:
        raise HandoffError("Wymagani odrębni właściciele procesu aktualizacji polityki")
    identities: set[int] = set()
    for owner in sorted(owners):
        permission = client.json(f"/collaborators/{owner}/permission")
        user = permission.get("user")
        if (
            permission.get("permission") not in {"write", "maintain", "admin"}
            or not isinstance(user, dict)
            or user.get("type") != "User"
            or not isinstance(user.get("login"), str)
            or user["login"].lower() != owner
            or type(user.get("id")) is not int
            or user["id"] <= 0
            or user["id"] in identities
        ):
            raise HandoffError("Właściciel CODEOWNERS nie ma potwierdzonego dostępu do zapisu")
        identities.add(user["id"])
    return owners


def _require_branch(
    client: GitHubClient, branch: str, expected_revision: str | None = None
) -> None:
    value = client.json(f"/branches/{branch}")
    commit = value.get("commit")
    if (
        value.get("name") != branch
        or value.get("protected") is not True
        or not isinstance(commit, dict)
        or not isinstance(commit.get("sha"), str)
        or re.fullmatch(r"[0-9a-f]{40}", commit["sha"]) is None
        or (expected_revision is not None and commit["sha"] != expected_revision)
    ):
        raise HandoffError("Gałąź nie jest chroniona albo przypięta polityka utraciła aktualność")
    protection = client.json(f"/branches/{branch}/protection")
    reviews = protection.get("required_pull_request_reviews")
    checks = protection.get("required_status_checks")
    if (
        not _enabled(protection.get("enforce_admins"), True)
        or not _enabled(protection.get("allow_force_pushes"), False)
        or not _enabled(protection.get("allow_deletions"), False)
        or not _enabled(protection.get("required_conversation_resolution"), True)
        or not isinstance(reviews, dict)
        or reviews.get("dismiss_stale_reviews") is not True
        or reviews.get("require_code_owner_reviews") is not True
        or reviews.get("require_last_push_approval") is not True
        or type(reviews.get("required_approving_review_count")) is not int
        or not 2 <= reviews["required_approving_review_count"] <= 6
        or not isinstance(checks, dict)
        or checks.get("strict") is not True
        or not isinstance(checks.get("checks"), list)
    ):
        raise HandoffError(
            "Brak wymaganych przeglądów, kontroli CI lub egzekwowania ochrony administratorów"
        )
    bypass = reviews.get("bypass_pull_request_allowances")
    if bypass is not None and (
        not isinstance(bypass, dict)
        or any(bypass.get(kind) != [] for kind in ("users", "teams", "apps"))
    ):
        raise HandoffError("Obejście wymaganych przeglądów nie jest dozwolone")
    trusted_checks: set[str] = set()
    for check in checks["checks"]:
        if (
            not isinstance(check, dict)
            or not isinstance(check.get("context"), str)
            or type(check.get("app_id")) is not int
            or check["app_id"] <= 0
        ):
            raise HandoffError(
                "Wymagana kontrola CI nie jest związana z konkretną aplikacją GitHub"
            )
        if (
            check["context"] in REQUIRED_PROTECTION_CHECKS
            and check["app_id"] != GITHUB_ACTIONS_APP_ID
        ):
            raise HandoffError("Wymagana kontrola nie pochodzi z aplikacji GitHub Actions")
        trusted_checks.add(check["context"])
    if not REQUIRED_PROTECTION_CHECKS <= trusted_checks:
        raise HandoffError("Brak wymaganych kontroli formatowania, testów dowodowych lub pokrycia")
    _code_owners(client, commit["sha"])


def require_environment(client: GitHubClient, name: str, policy_commit: str | None = None) -> None:
    if name not in {"stage9-linux-challenge", "stage9-linux-evidence"}:
        raise HandoffError("Nieznane środowisko kwalifikacji")
    if not isinstance(policy_commit, str) or re.fullmatch(r"[0-9a-f]{40}", policy_commit) is None:
        raise HandoffError("Brak przypiętej rewizji niezależnej gałęzi polityki")
    # protected_branches=true samo w sobie nie ustanawia ochrony: GitHub
    # dopuszcza wszystkie gałęzie, gdy żadne reguły ochrony nie istnieją.
    _require_branch(client, "main")
    _require_branch(client, POLICY_BRANCH, policy_commit)
    value = client.json(f"/environments/{name}")
    rules = value.get("protection_rules")
    reviewer_rules = (
        [
            rule
            for rule in rules
            if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
        ]
        if isinstance(rules, list)
        else []
    )
    branches = value.get("deployment_branch_policy")
    if (
        value.get("name") != name
        or value.get("can_admins_bypass") is not False
        or len(reviewer_rules) != 1
        or reviewer_rules[0].get("prevent_self_review") is not True
        or not isinstance(reviewer_rules[0].get("reviewers"), list)
        or not reviewer_rules[0]["reviewers"]
        or len(reviewer_rules[0]["reviewers"]) > 6
        or not isinstance(branches, dict)
        or branches.get("protected_branches") is not True
        or branches.get("custom_branch_policies") is not False
    ):
        raise HandoffError("Środowisko wymaga niezależnych reviewerów i chronionych gałęzi")
    for reviewer in reviewer_rules[0]["reviewers"]:
        account = reviewer.get("reviewer") if isinstance(reviewer, dict) else None
        if (
            not isinstance(reviewer, dict)
            or reviewer.get("type") != "User"
            or not isinstance(account, dict)
            or account.get("type") != "User"
            or type(account.get("id")) is not int
            or account["id"] <= 0
            or not isinstance(account.get("login"), str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", account["login"]) is None
        ):
            raise HandoffError("Reviewer środowiska nie ma jednoznacznej tożsamości")


def _identity(value: str) -> str:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise HandoffError("Nieprawidłowy identyfikator przebiegu lub próby")
    return value


def fetch_bundle(
    client: GitHubClient, repository: str, run_id: str, attempt: str, output: Path
) -> None:
    name = f"stage9-linux-bundle-{_identity(run_id)}-{_identity(attempt)}"
    matches: list[dict[str, Any]] = []
    for page in range(1, MAX_ARTIFACT_PAGES + 1):
        value = client.json(f"/actions/artifacts?name={name}&per_page=100&page={page}")
        artifacts = value.get("artifacts")
        if (
            type(value.get("total_count")) is not int
            or not 0 <= value["total_count"] <= MAX_ARTIFACT_PAGES * 100
            or not isinstance(artifacts, list)
            or len(artifacts) > 100
            or any(not isinstance(item, dict) for item in artifacts)
        ):
            raise HandoffError("Nieprawidłowy lub zbyt duży katalog artefaktów")
        matches.extend(item for item in artifacts if item.get("name") == name)
        if len(artifacts) < 100:
            break
    else:
        raise HandoffError("Niepełna paginacja katalogu artefaktów")
    if len(matches) != 1:
        raise HandoffError("Wymagany dokładnie jeden artefakt bieżącego challenge")
    artifact = matches[0]
    source_run = artifact.get("workflow_run")
    if (
        artifact.get("expired") is not False
        or type(artifact.get("id")) is not int
        or artifact["id"] <= 0
        or type(artifact.get("size_in_bytes")) is not int
        or not 0 < artifact["size_in_bytes"] <= MAX_ARCHIVE_BYTES
        or not isinstance(source_run, dict)
        or type(source_run.get("id")) is not int
        or source_run["id"] <= 0
    ):
        raise HandoffError("Artefakt wygasł lub ma nieprawidłową tożsamość/rozmiar")
    run = client.json(f"/actions/runs/{source_run['id']}")
    run_repository = run.get("repository")
    head_repository = run.get("head_repository")
    if (
        run.get("path") != INGRESS_WORKFLOW
        or run.get("event") != "workflow_dispatch"
        or run.get("head_branch") != "main"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or not isinstance(run_repository, dict)
        or run_repository.get("full_name") != repository
        or not isinstance(head_repository, dict)
        or head_repository.get("full_name") != repository
        or not isinstance(run.get("head_sha"), str)
        or re.fullmatch(r"[0-9a-f]{40}", run["head_sha"]) is None
    ):
        raise HandoffError("Artefakt nie pochodzi z zatwierdzonego transportu main")
    archive = client.archive(artifact["id"])
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise HandoffError("Archiwum przekracza limit rozmiaru")
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            members = bundle.infolist()
            if len(members) != 1:
                raise HandoffError("Archiwum musi zawierać wyłącznie bundle.json")
            member = members[0]
            mode = member.external_attr >> 16
            if (
                member.filename != "bundle.json"
                or member.is_dir()
                or stat.S_ISLNK(mode)
                or member.flag_bits & 1
                or not 0 < member.file_size <= MAX_BUNDLE_BYTES
            ):
                raise HandoffError("Niedozwolony element archiwum dowodowego")
            raw = bundle.read(member)
    except (zipfile.BadZipFile, OSError, RuntimeError, NotImplementedError) as exc:
        raise HandoffError("Nieprawidłowe archiwum dowodowe") from exc
    ingress(raw, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bezpieczny transport podpisanych dowodów Linux")
    commands = parser.add_subparsers(dest="command", required=True)
    incoming = commands.add_parser("ingress")
    incoming.add_argument("--output", type=Path, required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--bundle", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--repository", required=True)
    fetch.add_argument("--target-run-id", required=True)
    fetch.add_argument("--target-run-attempt", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    environment = commands.add_parser("require-environment")
    environment.add_argument("--repository", required=True)
    environment.add_argument("--name", required=True)
    environment.add_argument("--policy-commit", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "ingress":
            ingress(
                unpack_transport(os.environ.get("LINUX_EVIDENCE_BUNDLE", "").encode("utf-8")),
                args.output,
            )
        elif args.command == "pack":
            pack_bundle(args.bundle, args.output)
        else:
            client = GitHubClient(os.environ.get("GH_TOKEN", ""), args.repository)
            if args.command == "fetch":
                fetch_bundle(
                    client,
                    args.repository,
                    args.target_run_id,
                    args.target_run_attempt,
                    args.output,
                )
            else:
                require_environment(client, args.name, args.policy_commit)
    except (HandoffError, OSError, UnicodeError) as exc:
        print(f"ODMOWA_TRANSPORTU: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
