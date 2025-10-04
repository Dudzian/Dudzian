#!/usr/bin/env python3
"""
get_telegram_chat_id.py
Pobiera i wypisuje chat_id z ostatnich update'ów Telegram Bota.
Użycie:
  python scripts/get_telegram_chat_id.py --token <BOT_TOKEN>
  # albo z .env / zmiennej środowiskowej TELEGRAM_BOT_TOKEN
  python scripts/get_telegram_chat_id.py
Wymaga: requests, python-dotenv (opcjonalnie)
"""
import os, sys, json, argparse
from datetime import datetime, timezone
try:
    import requests  # type: ignore
except ImportError:
    print("Zainstaluj 'requests': pip install requests", file=sys.stderr)
    sys.exit(2)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

def utcnow():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", help="BOT token (jeśli nie używasz .env/zmiennych środowiskowych)")
    args = ap.parse_args()

    token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Brak tokena. Podaj --token lub ustaw TELEGRAM_BOT_TOKEN w .env", file=sys.stderr)
        sys.exit(1)

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    r = requests.get(url, timeout=10)
    try:
        data = r.json()
    except Exception:
        print("Błąd parsowania JSON. Response:", r.text[:500], file=sys.stderr)
        sys.exit(1)

    if not data.get("ok"):
        print("API zwróciło błąd:", data, file=sys.stderr)
        sys.exit(1)

    results = data.get("result", [])
    if not results:
        print("Brak update'ów. Wskazówki:\n- Napisz wiadomość do bota (DM),\n- albo dodaj go do grupy i napisz cokolwiek,\n- albo opublikuj post na kanale, gdzie jest adminem.\nPotem odpal skrypt ponownie.")
        sys.exit(0)

    # wyciągnij chat z message / edited_message / channel_post / my_chat_member itp.
    chats = []
    def _try(obj, path):
        cur = obj
        for p in path.split("."):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return None
        return cur

    for upd in results:
        for key in ["message", "edited_message", "channel_post", "edited_channel_post", "my_chat_member", "chat_member"]:
            m = upd.get(key)
            if not m: 
                continue
            chat = m.get("chat") if isinstance(m, dict) else None
            if chat and isinstance(chat, dict):
                chats.append(chat)

    # deduplikacja wg id
    uniq = {}
    for c in chats:
        cid = c.get("id")
        if cid is None: 
            continue
        uniq[cid] = {
            "id": cid,
            "type": c.get("type"),
            "title": c.get("title"),
            "username": c.get("username"),
            "first_name": c.get("first_name"),
            "last_name": c.get("last_name"),
        }

    if not uniq:
        print("Znaleziono update'y, ale bez 'chat'. Wyślij nową wiadomość do bota i spróbuj ponownie.")
        sys.exit(0)

    arr = list(uniq.values())

    # Czytelny output
    print(json.dumps({
        "timestamp_utc": utcnow(),
        "found_chats": arr,
        "hint": "Wybierz właściwy 'id' i wpisz do TELEGRAM_CHAT_ID w .env (dla grup często ujemny, np. -100123...)"
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
