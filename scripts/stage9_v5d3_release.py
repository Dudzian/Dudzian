"""Odczytowe etapy przygotowania V5d3, uruchamiane przez zaufany Python hosta.

Nie importuje źródeł kontrolera ani pakietów kandydata. Wszystkie pliki
instalacyjne są propozycją do przeglądu, bez podpisów i niezależnej aprobaty.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
from pathlib import Path
from typing import Any

BASE = "17e03a9cb42a84302401415f96ac90e756926915"
DOC = "docs/architecture/stage9_operator_v5d3_release_preparation.md"
ROOTS = (
    "deployment/canonical_json.py",
    "deployment/stage9_observer_collector_identity_v5d1.py",
    "deployment/stage9_observer_kvm_domain_v5d1.py",
    "deployment/stage9_observer_kvm_image_v5d1.py",
    "deployment/stage9_observer_kvm_preflight_v5d1.py",
    "scripts/stage9_observer_kvm_preflight_v5d1.py",
)
SAFETY = {
    "Stage 9": "IN_PROGRESS",
    "Observer 3A": "BLOCKED",
    "Stage 3B": "NOT AUTHORIZED",
    "Production": "NO-GO",
    "HOST_PREFLIGHT_VERIFIED": "NOT_CLAIMED",
    "EXECUTION_AUTHORIZATION": "NOT_AUTHORIZED",
    "PYC_RECONSTRUCTION": "UNVERIFIED",
}
ENV = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC"}
GIT_ENV = {
    **ENV,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
LIMIT = 8 * 1024 * 1024


class Refusal(ValueError):
    """Stały komunikat odmowy bez ujawniania treści plików."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refusal(message)


def dump(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def direct(path: str) -> Path:
    candidate = Path(os.path.abspath(path))
    for component in (*reversed(candidate.parents), candidate):
        require(not component.is_symlink(), "Dowiązanie w ścieżce roboczej.")
    require(candidate.exists(), "Brak katalogu roboczego.")
    return candidate


def git(repository: Path, *args: str) -> bytes:
    result = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.pager=cat",
            "-c",
            "trace2.normalTarget=0",
            "-c",
            "trace2.eventTarget=0",
            "-c",
            "trace2.perfTarget=0",
            "-C",
            str(repository),
            *args,
        ],
        env=GIT_ENV,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )
    require(result.returncode == 0, "Odmowa odczytu repozytorium.")
    require(len(result.stdout) <= 64 * LIMIT, "Przekroczono limit odczytu repozytorium.")
    return result.stdout


def record(state: dict[str, Any], area: str, status: str, detail: str) -> None:
    state["areas"][area] = {"area": area, "status": status, "detail": detail}


def save_state(attempt: Path, state: dict[str, Any]) -> None:
    # Jedyny wymieniany plik to stan NOWEJ próby, poza pakietem i starymi raportami.
    pending = attempt / "state.pending"
    dump(pending, state)
    pending.replace(attempt / "state.json")


def summary(attempt: Path, state: dict[str, Any], *, interrupted: bool = False) -> None:
    bundle = attempt / "bundle"
    areas = dict(state["areas"])
    for area, status, detail in (
        ("package", "FAIL" if interrupted else "PASS", "Spójność lokalna; bez aprobaty wydania."),
        (
            "installation_environment",
            "BLOCKED",
            "Brak zatwierdzonego pełnego zestawu Python/native/ABI i launchera.",
        ),
        ("host", "NOT_TESTED", "Nie wykonano pomiarów fizycznego hosta ani kwalifikacji."),
        ("policy", "BLOCKED", "Brak niezależnych anchors, podpisu i zatwierdzonych wymagań."),
        ("vm", "BLOCKED", "Brak osobnej zatwierdzonej VM, UUID i dowodów izolacji."),
        (
            "qualification",
            "BLOCKED",
            "Nieprzyznana niezależna tożsamość, izolacja, trace i akceptacja.",
        ),
        (
            "execution_authorization",
            "BLOCKED",
            "NOT_AUTHORIZED; przygotowanie nie autoryzuje wykonania.",
        ),
        (
            "exact_cpython_execution",
            "NOT_TESTED",
            "Nie wykonano badanego interpretera ani bytecode.",
        ),
    ):
        areas[area] = {"area": area, "status": status, "detail": detail}
    if any(state["areas"].get(letter, {}).get("status") != "PASS" for letter in "ABCDEFGHI"):
        areas["package"]["status"] = "FAIL"
    for name in ("SUMMARY.json", "SUMMARY.tsv"):
        # Podsumowanie częściowe jest wyłącznie w nowej próbie, nigdy w --resume.
        (bundle / name).unlink(missing_ok=True)
    ordered = [areas[key] for key in sorted(areas)]
    dump(
        bundle / "SUMMARY.json",
        {"profile": "stage9-v5d3-summary-v1", "areas": ordered, "safety": SAFETY},
    )
    with (bundle / "SUMMARY.tsv").open("x", encoding="utf-8") as stream:
        stream.write("AREA\tSTATUS\tDETAIL\n")
        for item in ordered:
            stream.write("\t".join(item[key] for key in ("area", "status", "detail")) + "\n")
        for key, value in SAFETY.items():
            stream.write(f"{key}\t{value}\tGranica kontraktu; bez zmiany.\n")


def initialize(repository_arg: str, parent_arg: str, resume_arg: str) -> Path:
    require(
        sys.platform == "linux" and getattr(os, "geteuid")() != 0,
        "Przygotowanie wymaga zwykłego nieuprzywilejowanego konta Linux.",
    )
    repository, parent = direct(repository_arg), direct(parent_arg)
    require(repository.is_dir() and parent.is_dir(), "Wymagane katalogi.")
    actual_root = Path(os.fsdecode(git(repository, "rev-parse", "--show-toplevel")).strip())
    require(actual_root == repository, "Wskaż katalog główny repozytorium.")
    require(not parent.is_relative_to(repository), "Wyniki muszą znajdować się poza repozytorium.")
    require(
        not any(
            parent.is_relative_to(Path(prefix))
            for prefix in ("/opt", "/etc", "/usr", "/boot", "/dev", "/proc", "/sys")
        ),
        "Nie wolno zapisywać wyników w chronionym katalogu systemowym.",
    )
    revision = git(repository, "rev-parse", "HEAD").decode().strip()
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "Nieprawidłowa rewizja.")
    previous = ""
    if resume_arg:
        previous = str(direct(resume_arg))
        require(Path(previous).is_dir(), "Wznowienie wymaga katalogu starej próby.")
        require(
            (Path(previous) / "SUMMARY.json").is_file() and (Path(previous) / "logs").is_dir(),
            "--resume wymaga katalogu bundle poprzedniej próby.",
        )
        require(
            not parent.is_relative_to(Path(previous)),
            "Nowa próba nie może zmieniać starego pakietu.",
        )
    attempt = Path(tempfile.mkdtemp(prefix=f"stage9-v5d3-{revision[:12]}-", dir=parent))
    (attempt / "bundle" / "logs").mkdir(parents=True)
    state: dict[str, Any] = {
        "repository": str(repository),
        "revision": revision,
        "previous_bundle": previous,
        "areas": {},
    }
    save_state(attempt, state)
    summary(attempt, state, interrupted=True)
    return attempt


def tree(repository: Path, revision: str) -> dict[str, str]:
    paths = {}
    for entry in git(repository, "ls-tree", "-r", "-z", revision).split(b"\0"):
        if not entry:
            continue
        info, raw_path = entry.split(b"\t", 1)
        mode, kind, _ = info.decode().split()
        path = raw_path.decode("utf-8")
        require(kind == "blob" or mode == "160000", "Nieobsługiwany obiekt Git.")
        paths[path] = mode
    return paths


def source(repository: Path, revision: str, paths: dict[str, str], path: str) -> bytes:
    require(paths.get(path) in {"100644", "100755"}, "Brak źródła lub niedozwolony obiekt Git.")
    require(
        re.fullmatch(r"[a-zA-Z0-9_./-]+", path) is not None and ".." not in path.split("/"),
        "Niebezpieczna ścieżka źródła.",
    )
    data = git(repository, "show", f"{revision}:{path}")
    require(len(data) <= LIMIT, "Źródło przekracza limit.")
    scan_bytes(data)
    return data


def scan_bytes(data: bytes) -> None:
    """Odmowa przed pierwszym zapisem rozpoznawalnego sekretu do pakietu."""
    require(
        re.search(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----", data) is None,
        "Materiał klucza prywatnego w źródle; nie skopiowano danych.",
    )
    # Niezależna kopia reguł treści jest w check_secrets weryfikatora;
    # testy graniczne sprawdzają zgodność bez importowania kodu pakietu.
    require(
        re.search(
            rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|AKIA[A-Z0-9]{16})", data
        )
        is None,
        "Potencjalne dane uwierzytelniające; nie skopiowano danych.",
    )


def local_imports(node: ast.AST, module: str, paths: dict[str, str]) -> set[str]:
    names: list[str] = []
    required: set[str] = set()
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
        required.update(names)
    elif isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
            package = module.split(".")[:-1]
            require(node.level <= len(package), "Nierozwiązany import względny.")
            base = ".".join(package[: len(package) - node.level + 1] + ([base] if base else []))
        names = [base]
        names += [f"{base}.{alias.name}" for alias in node.names if alias.name != "*"]
        required.add(base)
        if base in {"deployment", "scripts"}:
            require(
                all(alias.name != "*" for alias in node.names),
                "Nierozwiązany import gwiazdkowy pakietu.",
            )
            required.update(names)
    result = set()
    for name in names:
        if name.split(".")[0] not in {"deployment", "scripts"}:
            continue
        stem = name.replace(".", "/")
        candidates = (stem + ".py", stem + "/__init__.py")
        found = next((p for p in candidates if p in paths), None)
        if found:
            result.add(found)
            parent = Path(found).parent
            while str(parent) != ".":
                initializer = parent.as_posix() + "/__init__.py"
                if initializer in paths:
                    result.add(initializer)
                parent = parent.parent
        elif name in required:
            raise Refusal("Brak lokalnego modułu importowanego.")
    return result


def dynamic_calls(tree: ast.AST) -> list[ast.Call]:
    """Rozpoznaje mechanizmy po nazwie AST, bez dopasowania podciągu execute.

    Alias nie stanowi dowodu wykonania; pozostaje wskazaniem do przeglądu.
    Nierozstrzygalne przepływy danych wymagają osobnego przeglądu domknięcia.
    """
    mechanisms = {
        "__import__",
        "import_module",
        "exec",
        "eval",
        "compile",
        "run_path",
        "run_module",
        "CDLL",
        "PyDLL",
        "LoadLibrary",
        "spec_from_file_location",
        "spec_from_loader",
        "dlopen",
        "load_dynamic",
        "module_from_spec",
        "exec_module",
        "load_module",
    }
    aliases = set(mechanisms)
    # Proste aliasy importu i przypisania są domykane przed analizą wywołań.
    changed = True
    while changed:
        previous = set(aliases)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in aliases:
                        aliases.add(alias.asname or alias.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                value = node.value
                name = (
                    value.id
                    if isinstance(value, ast.Name)
                    else (value.attr if isinstance(value, ast.Attribute) else "")
                )
                if name in aliases:
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    aliases.update(target.id for target in targets if isinstance(target, ast.Name))
        changed = aliases != previous
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Zachowujemy getattr(..., "exec") i słownikowe wyszukiwanie
            # __builtins__["eval"], które wcześniejszy skan tekstowy widział.
            names = {
                part.id
                if isinstance(part, ast.Name)
                else part.attr
                if isinstance(part, ast.Attribute)
                else part.value
                for part in ast.walk(node.func)
                if isinstance(part, (ast.Name, ast.Attribute))
                or (isinstance(part, ast.Constant) and isinstance(part.value, str))
            }
            if names & aliases:
                result.append(node)
    return result


def collect_sources(repository: Path, revision: str, bundle: Path) -> dict[str, Any]:
    paths = tree(repository, revision)
    initializers = {path.split("/")[0] + "/__init__.py" for path in ROOTS}
    pending, collected, dynamic = set(ROOTS) | (initializers & paths.keys()), {}, []
    while pending:
        path = min(pending)
        pending.remove(path)
        if path in collected:
            continue
        data = source(repository, revision, paths, path)
        syntax = ast.parse(data, filename=path)
        collected[path] = data
        for node in ast.walk(syntax):
            pending.update(
                local_imports(node, path.removesuffix(".py").replace("/", "."), paths)
                - collected.keys()
            )
        for node in dynamic_calls(syntax):
            dynamic.append(
                {"path": path, "line": node.lineno, "expression": ast.unparse(node)[:1000]}
            )
    collected["pyproject.toml"] = source(repository, revision, paths, "pyproject.toml")
    for path, data in sorted(collected.items()):
        destination = bundle / "sources" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(data)
    return {
        "profile": "stage9-v5d3-imports-v1",
        "static_sources": sorted(path for path in collected if path.endswith(".py")),
        "dynamic_imports": sorted(dynamic, key=lambda item: (item["path"], item["line"])),
        "status": "UNVERIFIED",
        "detail": "AST sprawdza lokalne krawędzie. Libvirt, opcjonalny zstandard, startup/site, runtime discovery i native loading wymagają niezależnych dowodów; nie wykonano importów.",
    }


def secret_scan(bundle: Path) -> None:
    for path in bundle.rglob("*"):
        metadata = path.lstat()
        require(not path.is_symlink(), "Dowiązanie w pakiecie.")
        require(
            stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode),
            "Niedozwolony obiekt w pakiecie.",
        )
        if path.is_file():
            require(metadata.st_nlink == 1, "Dowiązanie twarde w pakiecie.")
            require(metadata.st_size <= LIMIT, "Przekroczony limit pliku.")
            require(
                path.suffix.lower() not in {".pem", ".key", ".p12", ".pfx", ".pyc"}
                and path.name not in {".env", "id_rsa", "id_ed25519", "credentials"},
                "Niedozwolona nazwa pliku.",
            )
            data = path.read_bytes()
            scan_bytes(data)


def stage(attempt: Path, letter: str, state: dict[str, Any]) -> None:
    repository, revision, bundle = Path(state["repository"]), state["revision"], attempt / "bundle"
    if letter == "A":
        require(
            git(repository, "rev-parse", "HEAD").decode().strip() == revision,
            "Zmiana rewizji podczas przygotowania.",
        )
        require(
            not git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
            "Repozytorium nie jest czyste.",
        )
        git(repository, "merge-base", "--is-ancestor", BASE, revision)
        require(sys.version_info >= (3, 11), "Niezgodny Python narzędziowy.")
        paths = tree(repository, revision)
        grafts = git(repository, "rev-parse", "--git-path", "info/grafts").decode().strip()
        grafts_path = Path(grafts) if Path(grafts).is_absolute() else repository / grafts
        require(not grafts_path.exists(), "Lokalne grafts nie mogą zmieniać historii źródeł.")
        for path in (
            *ROOTS,
            "pyproject.toml",
            DOC,
            "docs/architecture/stage9_operator_v5d2_deployment_plan.md",
            "docs/architecture/stage9_observer_kvm_preflight_v5d1.md",
            "docs/architecture/stage9_closure_matrix_v5d1.json",
        ):
            source(repository, revision, paths, path)
        tools = {
            name: shutil.which(name) for name in ("bash", "git", "python3", "sha256sum", "readelf")
        }
        require(
            all(tools[key] for key in ("bash", "git", "python3", "sha256sum")),
            "Brak wymaganego narzędzia przygotowania.",
        )
        dump(
            bundle / "RELEASE.json",
            {
                "profile": "stage9-v5d3-release-v1",
                "expected_base": BASE,
                "release_commit": revision,
                "repository_clean": True,
                "tools": tools,
                "previous_bundle": state["previous_bundle"],
                "source_approval": "UNVERIFIED",
                "detail": "SHA identyfikuje rewizję; nie jest niezależną aprobatą.",
            },
        )
        if state["previous_bundle"]:
            previous = Path(state["previous_bundle"])
            if (previous / "COMPLETE.json").is_file():
                result = subprocess.run(
                    [
                        "/bin/bash",
                        str(Path(__file__).with_name("stage9_v5d3_verify_bundle.sh")),
                        str(previous),
                    ],
                    env=ENV,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                for suffix, data in (
                    ("stdout", result.stdout),
                    ("stderr", result.stderr),
                    ("exit", f"{result.returncode}\n".encode("ascii")),
                ):
                    with (bundle / "logs" / f"previous-verification.{suffix}").open("xb") as stream:
                        stream.write(data)
                require(
                    result.returncode == 0,
                    "Poprzedni ukończony pakiet nie przeszedł weryfikacji; zachowano go bez zmian.",
                )
            print("Wznowienie: nowa próba, stare raporty zachowane; bez powtarzania regresji.")
    elif letter == "B":
        require(
            state["areas"].get("A", {}).get("status") == "PASS",
            "Zbieranie źródeł zablokowane przez etap A.",
        )
        dump(bundle / "IMPORTS.json", collect_sources(repository, revision, bundle))
        record(
            state,
            "import_completeness",
            "UNVERIFIED",
            "Dynamiczne importy, native i pliki startup wymagają niezależnego przeglądu.",
        )
    elif letter == "C":
        declaration = tomllib.loads((bundle / "sources/pyproject.toml").read_text())
        project = declaration["project"]
        extra = project["optional-dependencies"]["observer-kvm-preflight"]
        require(
            "cryptography>=50.0.1,<51" in project["dependencies"] and "defusedxml==0.7.1" in extra,
            "Brak wymaganych deklaracji zależności.",
        )
        dump(
            bundle / "DEPENDENCIES.json",
            {
                "profile": "stage9-v5d3-dependencies-v1",
                "requires_python": project["requires-python"],
                "project_dependencies": project["dependencies"],
                "observer_extra": extra,
                "libvirt_binding": "REVIEWED_SYSTEM_BINDING_REQUIRED",
                "approved_package_set": "UNVERIFIED",
                "transitive_closure": "UNVERIFIED",
            },
        )
        dump(
            bundle / "INTERPRETER.json",
            {
                "profile": "stage9-v5d3-interpreter-v1",
                "scope": "PREPARATION_TOOL_ONLY",
                "executable": sys.executable,
                "actual_executable": os.readlink("/proc/self/exe"),
                "version": sys.version,
                "implementation": sys.implementation.name,
                "cache_tag": sys.implementation.cache_tag,
                "soabi": sysconfig.get_config_var("SOABI"),
                "multiarch": sysconfig.get_config_var("MULTIARCH"),
                "sys_path": sys.path,
                "isolated": sys.flags.isolated,
                "no_site": sys.flags.no_site,
                "deployment_interpreter": "UNVERIFIED",
                "startup_files": "UNVERIFIED",
                "detail": "Nie kopiowano ani nie wykonywano interpretera kandydata; pyvenv.cfg, .pth, sitecustomize, usercustomize, loader/provider i ścieżki wymagają przeglądu.",
            },
        )
        record(
            state,
            "dependency_resolution",
            "BLOCKED",
            "Brak zatwierdzonego wheelhouse/lock, pełnych zależności przechodnich i systemowego bindingu ABI; niczego nie instalowano.",
        )
    elif letter == "D":
        dump(
            bundle / "NATIVE.json",
            {
                "profile": "stage9-v5d3-native-v1",
                "status": "UNVERIFIED",
                "required_components": [
                    "libvirt-python",
                    "libvirtmod",
                    "libvirt.so i zależności przechodnie",
                    "Natywne rozszerzenie cryptography oraz dostawcy OpenSSL",
                    "Moduły natywne CFFI, jeśli wymaga ich zatwierdzony zestaw",
                    "Plik wykonywalny CPython, loader, biblioteka standardowa i rozszerzenia",
                    "/usr/bin/qemu-img",
                    "/usr/bin/qemu-system-x86_64",
                    "Linux KVM i demon libvirt",
                ],
                "libraries": [],
                "detail": "Lista wymaganych komponentów, bez zinwentaryzowanych/zaakceptowanych bibliotek. Brak zestawu wejściowego; DT_NEEDED nie zastępuje rzeczywistego closure. Nie użyto ldd ani badanego runtime.",
            },
        )
        (bundle / "review").mkdir()
        data = source(repository, revision, tree(repository, revision), DOC)
        (bundle / "review" / Path(DOC).name).write_bytes(data)
    elif letter == "E":
        secret_scan(bundle)
        require(
            git(repository, "rev-parse", "HEAD").decode().strip() == revision
            and not git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
            "Repozytorium zmieniło się podczas przygotowania.",
        )
        record(
            state,
            "credential_scan",
            "UNVERIFIED",
            "Ograniczony skan nazw i wzorców przeszedł; nie dowodzi braku wszystkich sekretów. Wymagany niezależny skan.",
        )
    elif letter in {"F", "G", "H"}:
        require(
            (bundle / "review" / Path(DOC).name).is_file(),
            "Brak procedury przekazania do przeglądu.",
        )
        print(
            {
                "F": "Projekt chronionej instalacji: dokument w review; nie wykonano instalacji.",
                "G": "Pola polityki i braki: dokument w review; bez podpisu i kluczy.",
                "H": "Specyfikacja osobnej VM: dokument w review; UUID i dowody nadal wymagane; bez operacji VM.",
            }[letter]
        )
    else:
        raise Refusal("Nieznany etap.")
    record(
        state, letter, "PASS", "Ukończono czynności przygotowania etapu; brak aprobaty wdrożenia."
    )
    print(f"{letter}\tPASS\tZachowano materiał przygotowawczy.")


def finalize(attempt: Path, state: dict[str, Any]) -> None:
    bundle = attempt / "bundle"
    blockers = [
        "Niezależna aprobata źródeł, pochodzenia bajtów i pełnego inwentarza instalacji.",
        "Zatwierdzony pełny zestaw zależności (także przechodnich), wersje, sumy i pochodzenie natywnego ABI.",
        "Dokładny interpreter instalacji, startup/import paths, pakiety i rzeczywiście ładowane biblioteki.",
        "Dynamiczny libvirt oraz opcjonalny zstandard; AST nie dowodzi wykonaniowego closure.",
        "Chroniona instalacja i rodzice; odrębnie zatwierdzony launcher bez obejść sys.path.",
        "Niezależne anchors, polityka Ed25519, domain requirements i dane authority/revision/UUID/hashes.",
        "Kwalifikacja fizycznego hosta, niezależny launch witness i loaded execution closure (brak admission path V5d1).",
        "Osobna jednorazowa VM; izolacja, trace, measurement channel, limity, stop/cleanup całej VM.",
        "Pełne runtime inventory, zgoda na konkretny eksperyment, niezależna akceptacja, freshness i antirollback.",
    ]
    dump(
        bundle / "MISSING.json",
        {
            "profile": "stage9-v5d3-missing-v1",
            "items": blockers,
            "execution_authorization": "NOT_AUTHORIZED",
        },
    )
    record(state, "I", "PASS", "Wygenerowano rozdzielone statusy i listę blokad.")
    summary(attempt, state)
    print("Raport przygotowania ukończony; zamknięcie manifestu nastąpi po zamknięciu tego logu.")


def seal(attempt: Path, state: dict[str, Any]) -> None:
    bundle = attempt / "bundle"
    # Blokady zewnętrznych dowodów nie są błędem budowy materiału do review.
    require(
        all(state["areas"].get(letter, {}).get("status") == "PASS" for letter in "ABCDEFGHI"),
        "Etap przygotowania nie został ukończony; nie tworzę COMPLETE.",
    )
    require(
        git(Path(state["repository"]), "rev-parse", "HEAD").decode().strip() == state["revision"]
        and not git(Path(state["repository"]), "status", "--porcelain=v1", "--untracked-files=all"),
        "Końcowa zmiana repozytorium.",
    )
    dump(
        bundle / "COMPLETE.json",
        {"profile": "stage9-v5d3-complete-v1", "release_commit": state["revision"]},
    )
    secret_scan(bundle)
    entries = []
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            metadata = path.stat()
            entries.append(
                {
                    "path": path.relative_to(bundle).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size": metadata.st_size,
                    "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                }
            )
    dump(
        bundle / "MANIFEST.json",
        {"profile": "stage9-v5d3-bundle-v1", "release_commit": state["revision"], "files": entries},
    )
    digest_entries = {entry["path"]: entry["sha256"] for entry in entries}
    digest_entries["MANIFEST.json"] = hashlib.sha256(
        (bundle / "MANIFEST.json").read_bytes()
    ).hexdigest()
    with (bundle / "SHA256SUMS").open("x", encoding="ascii") as stream:
        for digest_path, digest in sorted(digest_entries.items()):
            stream.write(f"{digest}  {digest_path}\n")


def handoff(attempt: Path) -> None:
    """Wiąże oryginalny stdout/stderr/exit z zamknięciem bundle, poza jego manifestem.

    Weryfikacja nie może znajdować się we własnym badanym manifeście.
    Koperta i hashe zapewniają spójność transportu, bez ustanawiania aprobaty.
    """
    names = (
        "bundle/MANIFEST.json",
        "bundle/SHA256SUMS",
        "verification.json",
        "verification.stderr",
        "verification.exit",
    )
    entries = []
    for name in names:
        path = attempt / name
        require(path.is_file() and not path.is_symlink(), "Niekompletny materiał przekazania.")
        metadata = path.stat()
        require(metadata.st_nlink == 1 and metadata.st_size <= LIMIT, "Nieprawidłowy raport.")
        data = path.read_bytes()
        scan_bytes(data)
        entries.append(
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        )
    require(
        (attempt / "verification.exit").read_bytes() == b"0\n", "Weryfikator odmówił przekazania."
    )
    report = json.loads((attempt / "verification.json").read_bytes())
    require(
        type(report) is dict and report.get("local_consistency") == "PASS",
        "Brak pomyślnego raportu weryfikatora.",
    )
    dump(
        attempt / "HANDOFF.json",
        {
            "profile": "stage9-v5d4-handoff-v1",
            "files": entries,
            "independent_approval": "UNVERIFIED",
            "safety": SAFETY,
        },
    )


def invalidate(attempt: Path, state: dict[str, Any], *, interrupted: bool) -> None:
    """Zachowuje odrzucone zamknięcie, odbiera mu znacznik ukończenia."""
    bundle = attempt / "bundle"
    if any((bundle / name).exists() for name in ("COMPLETE.json", "MANIFEST.json", "SHA256SUMS")):
        rejected = Path(tempfile.mkdtemp(prefix="rejected-seal-", dir=attempt))
        for name in ("COMPLETE.json", "MANIFEST.json", "SHA256SUMS", "SUMMARY.json", "SUMMARY.tsv"):
            original = bundle / name
            if original.exists():
                original.rename(rejected / name)
    record(
        state,
        "final_verification",
        "FAIL",
        "Przerwano przygotowanie."
        if interrupted
        else "Niezależny weryfikator odmówił lokalnej spójności; zachowano odrzucone manifesty poza bundle.",
    )
    summary(attempt, state, interrupted=True)


def main() -> int:
    try:
        action, *args = sys.argv[1:]
        if action == "init":
            require(len(args) == 3, "Nieprawidłowe argumenty.")
            print(initialize(*args))
            return 0
        require(len(args) == (2 if action == "stage" else 1), "Nieprawidłowe argumenty.")
        attempt = direct(args[0])
        state = json.loads((attempt / "state.json").read_text())
        if action == "interrupted":
            invalidate(attempt, state, interrupted=True)
            save_state(attempt, state)
            return 0
        if action == "reject":
            invalidate(attempt, state, interrupted=False)
            save_state(attempt, state)
            return 0
        try:
            if action == "stage":
                stage(attempt, args[1], state)
            elif action == "finalize":
                finalize(attempt, state)
            elif action == "seal":
                seal(attempt, state)
            elif action == "handoff":
                handoff(attempt)
            else:
                raise Refusal("Nieznana czynność.")
        except (OSError, ValueError, KeyError, SyntaxError, subprocess.SubprocessError) as exc:
            area = args[1] if action == "stage" else action
            detail = (
                str(exc)
                if isinstance(exc, Refusal)
                else (
                    "ENVIRONMENT_ERROR: brak dostępu, narzędzia lub przekroczony timeout."
                    if isinstance(exc, (OSError, subprocess.SubprocessError))
                    else "INPUT_CONTRACT_ERROR: niekompletne lub nieprawidłowe dane; zachowano częściowe wyniki."
                )
            )
            record(state, area, "FAIL", detail)
            print(detail, file=sys.stderr)
            save_state(attempt, state)
            summary(attempt, state, interrupted=True)
            return 2
        save_state(attempt, state)
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(
            str(exc) if isinstance(exc, Refusal) else "Nie można bezpiecznie przygotować próby.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
