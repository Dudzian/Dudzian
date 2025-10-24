.pragma library
.import QtQuick.LocalStorage 2.0 as LocalStorage

const kMaxAuditRows = 200;

var __db = null;

function database() {
    if (__db === null) {
        __db = LocalStorage.LocalStorage.openDatabaseSync(
            "SecurityArtifacts",
            "1.0",
            "Security cache for license audit",
            1024 * 1024
        );
        __db.transaction(function(tx) {
            tx.executeSql(
                "CREATE TABLE IF NOT EXISTS license_audit (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                    "fingerprint TEXT," +
                    "edition TEXT," +
                    "license_id TEXT," +
                    "maintenance_until TEXT," +
                    "recorded_at TEXT" +
                ")"
            );
        });
    }
    return __db;
}

function addLicenseSnapshot(snapshot) {
    if (!snapshot)
        return false;
    var fingerprint = snapshot.fingerprint || "";
    var edition = snapshot.edition || snapshot.profile || "";
    var licenseId = snapshot.license_id || snapshot.licenseId || "";
    var maintenance = snapshot.maintenance_until || snapshot.maintenanceUntil || "";
    var recordedAt = snapshot.recorded_at || snapshot.recordedAt || new Date().toISOString();

    database().transaction(function(tx) {
        tx.executeSql(
            "INSERT INTO license_audit (fingerprint, edition, license_id, maintenance_until, recorded_at) VALUES (?, ?, ?, ?, ?)",
            [fingerprint, edition, licenseId, maintenance, recordedAt]
        );
        if (kMaxAuditRows > 0) {
            tx.executeSql(
                "DELETE FROM license_audit WHERE id NOT IN (SELECT id FROM license_audit ORDER BY id DESC LIMIT ?)",
                [kMaxAuditRows]
            );
        }
    });
    return true;
}

function fetchAudit(limit) {
    var records = [];
    database().transaction(function(tx) {
        var sql = "SELECT fingerprint, edition, license_id, maintenance_until, recorded_at FROM license_audit ORDER BY recorded_at DESC";
        if (limit && limit > 0)
            sql += " LIMIT " + limit;
        var rs = tx.executeSql(sql);
        for (var i = 0; i < rs.rows.length; ++i)
            records.push(rs.rows.item(i));
    });
    return records;
}

function clearAudit() {
    database().transaction(function(tx) {
        tx.executeSql("DELETE FROM license_audit");
    });
}

function totalAuditCount() {
    var count = 0;
    database().transaction(function(tx) {
        var rs = tx.executeSql("SELECT COUNT(*) AS total FROM license_audit");
        if (rs.rows.length > 0)
            count = rs.rows.item(0).total;
    });
    return count;
}
