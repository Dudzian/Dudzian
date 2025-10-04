import sys, json, runpy, keyring

orig = keyring.get_password
def probe(service, username):
    val = orig(service, username)
    print(f"[PROBE] get_password({service!r}, {username!r}) ->", "HIT" if val else "MISS")
    return val
keyring.get_password = probe

# uruchamiamy skrypt z Twoimi parametrami
sys.argv = [
    "run_daily_trend.py",
    "--config","config/core.yaml",
    "--environment","binance_paper",
    "--secret-namespace","dudzian.trading",
    "--dry-run",
    "--log-level","DEBUG",
]
runpy.run_path("scripts/run_daily_trend.py", run_name="__main__")
