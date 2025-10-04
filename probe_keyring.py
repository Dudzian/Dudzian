import sys, runpy, keyring

# Zachowaj oryginalną funkcję
_orig_get = keyring.get_password

def _probe(service, username):
    val = _orig_get(service, username)
    print(f"[PROBE] get_password(service={service!r}, username={username!r}) ->", "HIT" if val else "None")
    return val

# Podmień na nasz „podsłuch”
keyring.get_password = _probe

# Uruchom skrypt z parametrami dry-run
sys.argv = [
    "run_daily_trend.py",
    "--environment", "binance_paper",
    "--secret-namespace", "dudzian.trading",
    "--dry-run",
    "--log-level", "INFO",
]

runpy.run_path("scripts/run_daily_trend.py", run_name="__main__")
