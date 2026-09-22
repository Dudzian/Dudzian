#!/bin/sh
set -eu

# Offline administrator ceremony.  This installer intentionally does not
# provision keys or accept secrets.  PostgreSQL schema/key provisioning remains
# the separately reviewed offline ceremony.
test "$(id -u)" -eq 0
getent group freshness_verifier_ipc >/dev/null || groupadd --system freshness_verifier_ipc
for user in os_freshness_crypto_verifier os_freshness_runtime; do
    getent passwd "$user" >/dev/null || \
        useradd --system --no-create-home --shell /usr/sbin/nologin "$user"
    usermod --append --groups freshness_verifier_ipc "$user"
done

install -d -o root -g root -m 0755 /usr/lib/cryptohunter /etc/systemd/system
rm -rf /usr/lib/cryptohunter/bot_core /usr/lib/cryptohunter/deployment
cp -a bot_core deployment /usr/lib/cryptohunter/
chown -R root:root /usr/lib/cryptohunter
find /usr/lib/cryptohunter -type d -exec chmod 0755 {} +
find /usr/lib/cryptohunter -type f -exec chmod 0644 {} +
install -o root -g root -m 0644 \
    deployment/systemd/cryptohunter-freshness-verifier.service \
    /etc/systemd/system/cryptohunter-freshness-verifier.service
install -d -o root -g postgres -m 0750 /etc/cryptohunter/postgresql
install -o root -g postgres -m 0640 deployment/postgresql/pg_hba.conf \
    /etc/cryptohunter/postgresql/pg_hba.conf
install -o root -g postgres -m 0640 deployment/postgresql/pg_ident.conf \
    /etc/cryptohunter/postgresql/pg_ident.conf
test -d /etc/postgresql/16/main/conf.d
install -o root -g postgres -m 0640 deployment/postgresql/production-local.conf \
    /etc/postgresql/16/main/conf.d/cryptohunter-production-local.conf

systemctl daemon-reload
systemctl enable cryptohunter-freshness-verifier.service
