#!/usr/bin/env python3
"""
send_alert.py — minimalny skrypt do testowego wysłania alertu i logowania wyniku.
Obsługiwane kanały: telegram, email (SMTP), webhook (generic HTTP POST).

Użycie (przykład Telegram):
  python scripts/send_alert.py --channel telegram --message "Paper trading start test"

Wymagania:
  pip install requests python-dotenv
Konfiguracja przez .env lub zmienne środowiskowe (zob. sekcja poniżej).
"""
import argparse
import json
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

try:
    import requests  # type: ignore
except ImportError:
    print("Brak biblioteki 'requests'. Zainstaluj: pip install requests", file=sys.stderr)
    sys.exit(2)

# -- Ładowanie .env (opcjonalnie) --
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # jeśli istnieje .env w katalogu projektu
except Exception:
    pass

ROOT = Path.cwd()
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG = LOGS_DIR / "alerts_audit.jsonl"

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _append_audit(entry: dict) -> None:
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# ------------- Implementacje kanałów -------------
def send_telegram(message: str, token: str | None, chat_id: str | None, timeout: float = 10.0) -> tuple[str, int, dict | None]:
    tok = token or os.getenv("TELEGRAM_BOT_TOKEN")
    cid = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not cid:
        raise RuntimeError("Brak TELEGRAM_BOT_TOKEN lub TELEGRAM_CHAT_ID (ustaw w .env lub jako zmienne środowiskowe).")
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    resp = requests.post(url, data={"chat_id": cid, "text": message}, timeout=timeout)
    try:
        payload = resp.json()
    except Exception:
        payload = None
    status = "OK" if resp.status_code == 200 and payload and payload.get("ok") else "ERROR"
    return status, resp.status_code, payload

def send_webhook(message: str, url: str | None, timeout: float = 10.0) -> tuple[str, int, dict | None]:
    endpoint = url or os.getenv("ALERT_WEBHOOK_URL")
    if not endpoint:
        raise RuntimeError("Brak ALERT_WEBHOOK_URL (ustaw w .env lub podaj --webhook-url).")
    resp = requests.post(endpoint, json={"text": message, "timestamp": _utc_now_iso()}, timeout=timeout)
    try:
        payload = resp.json()
    except Exception:
        payload = {"content": resp.text[:512]}
    status = "OK" if 200 <= resp.status_code < 300 else "ERROR"
    return status, resp.status_code, payload

def send_email(message: str, subject: str | None) -> tuple[str, int, dict | None]:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASS")
    sender = os.getenv("SMTP_FROM")
    recipient = os.getenv("SMTP_TO")

    if not all([host, user, pwd, sender, recipient]):
        raise RuntimeError("Brak konfiguracji SMTP (SMTP_HOST, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO).")

    msg = MIMEText(message, _charset="utf-8")
    msg["Subject"] = subject or "Alert test"
    msg["From"] = sender
    msg["To"] = recipient

    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls()
            server.login(user, pwd)
            server.sendmail(sender, [recipient], msg.as_string())
        return "OK", 250, {"info": "Email accepted by SMTP server"}
    except smtplib.SMTPResponseException as e:
        code = int(getattr(e, "smtp_code", 550))
        return "ERROR", code, {"error": str(e)}
    except Exception as e:
        return "ERROR", 550, {"error": str(e)}

# ------------- CLI -------------
def main():
    parser = argparse.ArgumentParser(description="Wyślij testowy alert i zaloguj wynik do logs/alerts_audit.jsonl")
    parser.add_argument("--channel", required=True, choices=["telegram", "email", "webhook"], help="Kanał alertu")
    parser.add_argument("--message", required=True, help="Treść wiadomości")
    parser.add_argument("--severity", default="info", help="Poziom (np. info|warning|critical)")
    # Telegram
    parser.add_argument("--telegram-token", help="Nadpisz TELEGRAM_BOT_TOKEN")
    parser.add_argument("--telegram-chat-id", help="Nadpisz TELEGRAM_CHAT_ID")
    # Webhook
    parser.add_argument("--webhook-url", help="Nadpisz ALERT_WEBHOOK_URL")
    # Email
    parser.add_argument("--subject", help="Temat emaila (kanał=email)")

    args = parser.parse_args()

    ts = _utc_now_iso()
    entry = {
        "timestamp_utc": ts,
        "channel": args.channel,
        "severity": args.severity,
        "message": args.message,
    }

    try:
        if args.channel == "telegram":
            status, code, payload = send_telegram(args.message, args.telegram_token, args.telegram_chat_id)
        elif args.channel == "webhook":
            status, code, payload = send_webhook(args.message, args.webhook_url)
        else:
            status, code, payload = send_email(args.message, args.subject)

        entry.update({
            "status": status,
            "code": code,
            "response": payload,
        })
    except Exception as e:
        entry.update({
            "status": "ERROR",
            "code": 500,
            "error": str(e),
        })

    _append_audit(entry)

    # Czytelny stdout do podbicia statusu w logu B1
    print(json.dumps(entry, ensure_ascii=False, indent=2))

    # exit code dla CI
    if entry.get("status") != "OK":
        sys.exit(1)

if __name__ == "__main__":
    main()
