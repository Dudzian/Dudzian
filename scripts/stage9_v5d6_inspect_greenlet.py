"""Reproduce the static inspection of the original V5d6 greenlet wheel.

Reads an already acquired archive without extraction, imports or execution.
The measured report is evidence, never independent omission approval.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path
import stat
from typing import Any
import zipfile

FILENAME = "greenlet-3.5.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"
WHEEL_SHA256 = "975736b002ed080d124cf81a79cb7e05cb26d6b3f5c7a7b651c0fcce70353aa1"
HEADER = "greenlet-3.5.6.data/headers/greenlet.h"
HEADER_SHA256 = "b33e69611490a9e7603add80320c4b65d4e33bea9caff24cbe0884251f48009f"
ORIGIN_URL = (
    "https://files.pythonhosted.org/packages/66/6a/"
    "1594f3869c57c149abdb380492529e04d4c0229b5e4d79572c5bd0aaa673/" + FILENAME
)
DIST_INFO = "greenlet-3.5.6.dist-info"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inspect_original(path: Path) -> dict[str, Any]:
    require(not path.is_symlink() and path.is_file(), "INSPECTION_WHEEL_NOT_REGULAR")
    data = path.read_bytes()
    require(path.name == FILENAME, "INSPECTION_WHEEL_FILENAME")
    require(sha256(data) == WHEEL_SHA256, "INSPECTION_WHEEL_IDENTITY")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        require(len({item.filename for item in members}) == len(members), "INSPECTION_DUPLICATE")
        require(archive.testzip() is None, "INSPECTION_ZIP_CRC")
        files = {item.filename: item for item in members if not item.is_dir()}
        metadata = BytesParser().parsebytes(archive.read(DIST_INFO + "/METADATA"))
        wheel = BytesParser().parsebytes(archive.read(DIST_INFO + "/WHEEL"))
        require(
            not metadata.defects
            and metadata.get_all("Name") == ["greenlet"]
            and metadata.get_all("Version") == ["3.5.6"],
            "INSPECTION_DISTRIBUTION_IDENTITY",
        )
        record_name = DIST_INFO + "/RECORD"
        record_bytes = archive.read(record_name)
        record_rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"))))
        require(
            all(len(row) == 3 for row in record_rows)
            and len({row[0] for row in record_rows}) == len(record_rows),
            "INSPECTION_RECORD_SCHEMA",
        )
        records = {row[0]: row for row in record_rows}
        require(set(records) == set(files), "INSPECTION_RECORD_FILE_SET")
        require(records[record_name] == [record_name, "", ""], "INSPECTION_RECORD_SELF")
        inventory = []
        native = []
        for name, item in sorted(files.items()):
            content = archive.read(item)
            mode = item.external_attr >> 16
            require(stat.S_ISREG(mode), "INSPECTION_MEMBER_TYPE")
            if name != record_name:
                encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest())
                require(
                    records[name]
                    == [name, "sha256=" + encoded.rstrip(b"=").decode(), str(len(content))],
                    "INSPECTION_RECORD_INTEGRITY",
                )
            inventory.append(
                {
                    "member": name,
                    "sha256": sha256(content),
                    "size": len(content),
                    "mode": mode,
                    "type": "regular",
                    "zip_crc32": f"{item.CRC:08x}",
                }
            )
            if content.startswith(b"\x7fELF") or name.endswith((".so", ".pyd", ".dll", ".dylib")):
                native.append(name)
        unsupported = [
            name
            for name in sorted(files)
            if name.split("/")[0].endswith(".data")
            and (len(name.split("/")) < 3 or name.split("/")[1] not in {"purelib", "platlib"})
        ]
        require(unsupported == [HEADER], "INSPECTION_UNSUPPORTED_MEMBERS")
        header = archive.read(HEADER)
        header_item = files[HEADER]
        require(
            sha256(header) == HEADER_SHA256
            and len(header) == 4755
            and header_item.external_attr >> 16 == 0o100644
            and not header.startswith((b"\x7fELF", b"MZ", b"#!"))
            and b"\x00" not in header,
            "INSPECTION_HEADER_IDENTITY",
        )
        header.decode("utf-8")
        return {
            "profile": "stage9-v5d6-greenlet-static-inspection-v1",
            "inspection_date_utc": datetime.now(timezone.utc).date().isoformat(),
            "approval": "NOT_GRANTED",
            "execution": "NOT_EXECUTED",
            "full_operator_wheelhouse": "NOT_PROVIDED_68_OTHER_WHEELS_NOT_INSPECTED",
            "package": {
                "distribution": metadata["Name"],
                "version": metadata["Version"],
                "dist_info": DIST_INFO,
                "filename": FILENAME,
                "sha256": sha256(data),
                "size": len(data),
                "origin_url": ORIGIN_URL,
                "pypi_json_url": "https://pypi.org/pypi/greenlet/3.5.6/json",
                "requires_python": metadata["Requires-Python"],
                "requires_dist": metadata.get_all("Requires-Dist", []),
                "tags": sorted(wheel.get_all("Tag", [])),
                "zip_integrity": "PASS",
                "crc_checked_for_all_files": True,
                "original_file_count": len(files),
                "directory_count": len(members) - len(files),
                "record_file_count": len(records),
                "record_file_set_exact": True,
                "record_sha256": sha256(record_bytes),
                "record_self_entry": "EMPTY_HASH_AND_SIZE",
                "record_sha256_size_checks": len(files) - 1,
                "record_sha256_size_result": "PASS",
                "native_members_requiring_separate_approval": native,
                "unsupported_members": unsupported,
                "additional_unsupported_members": [],
            },
            "header": {
                "distribution": "greenlet",
                "version": "3.5.6",
                "member": HEADER,
                "sha256": sha256(header),
                "size": len(header),
                "mode": header_item.external_attr >> 16,
                "mode_octal": oct(header_item.external_attr >> 16),
                "type": "regular",
                "zip_create_system": header_item.create_system,
                "zip_external_attr": header_item.external_attr,
                "zip_crc32": f"{header_item.CRC:08x}",
                "record_entry": records[HEADER],
                "record_line": ",".join(records[HEADER]),
                "utf8_text": True,
                "elf": False,
                "execute_bits": header_item.external_attr >> 16 & 0o111,
                "static_content_pl": (
                    "Nagłówek C API: deklaracje i makra, #include <Python.h>, osłony C++. "
                    "Makro PyGreenlet_Import odwołuje się do PyCapsule_Import; "
                    "plik odczytano jako dane i nie kompilowano ani nie wykonywano."
                ),
            },
            "original_members": inventory,
            "limitations_pl": [
                "Inspekcja oryginalnego archiwum nie jest niezależną aprobatą pominięcia nagłówka.",
                "Nie zmodyfikowano archiwum ani upstream RECORD; nie instalowano pakietu.",
                "Nie wykonano źródeł, bytecode, importów, bibliotek natywnych ani kompilacji.",
                "URL PyPI i hash bajtów nie zastępują podpisu wydawcy ani niezależnej aprobaty.",
                "Nie dostarczono pełnego oryginalnego wheelhouse 69; tego wyniku nie ekstrapolowano.",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = inspect_original(args.wheel)
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        # Exclusive creation preserves the original wheel even when --output
        # names the input archive, an alias, or an existing report.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError:
        print(json.dumps({"status": "REFUSED", "reason": "INSPECTION_OUTPUT_EXISTS"}))
        return 2
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
