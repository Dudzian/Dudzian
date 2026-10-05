import subprocess

import httpx
import requests


class InvalidSignature(Exception):
    pass


def network(url: str) -> None:
    # ruleid: cryptohunter-http-request-requires-timeout
    requests.get(url)
    # ok: cryptohunter-http-request-requires-timeout
    requests.get(url, timeout=10)
    # ruleid: cryptohunter-tls-verification-disabled
    httpx.get(url, verify=False, timeout=10)


def process(command: str) -> None:
    # ruleid: cryptohunter-subprocess-shell-true
    subprocess.run(command, shell=True, check=False)


def dynamic(value: str) -> object:
    # ruleid: cryptohunter-dynamic-code-execution
    return eval(value)


def verify() -> bool:
    # ruleid: cryptohunter-signature-exception-fail-open
    try:
        raise InvalidSignature
    except InvalidSignature:
        return True
