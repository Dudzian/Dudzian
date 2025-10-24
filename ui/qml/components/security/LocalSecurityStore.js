.pragma library
.import QtQuick.LocalStorage 2.0 as LocalStorage

const kMaxAuditRows = 200;

var __db = null;
var __memoryMode = false;
var __memoryAudit = [];

function __isLocalStorageAvailable() {
    if (typeof LocalStorage === "undefined")
        return false;
    if (LocalStorage === null)
        return false;
    return typeof LocalStorage.openDatabaseSync === "function";
}

function database() {
    if (__db === null) {
        if (__isLocalStorageAvailable()) {
            try {
                __db = LocalStorage.openDatabaseSync(
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
            } catch (error) {
                __db = null;
            }
        }
        if (__db === null) {
            __memoryMode = true;
            __db = { memory: true }; // sentinel oznaczający inicjalizację fallbacku
        }
    }
    return __db;
}

function __appendMemorySnapshot(entry) {
    __memoryAudit.unshift(entry);
    if (kMaxAuditRows > 0 && __memoryAudit.length > kMaxAuditRows)
        __memoryAudit.length = kMaxAuditRows;
}

function addLicenseSnapshot(snapshot) {
    if (!snapshot)
        return false;

    var entry = {
        fingerprint: snapshot.fingerprint || "",
        edition: snapshot.edition || snapshot.profile || "",
        license_id: snapshot.license_id || snapshot.licenseId || "",
        maintenance_until: snapshot.maintenance_until || snapshot.maintenanceUntil || "",
        recorded_at: snapshot.recorded_at || snapshot.recordedAt || new Date().toISOString()
    };

    database();
    if (__memoryMode) {
        __appendMemorySnapshot(entry);
        return true;
    }

    __db.transaction(function(tx) {
        tx.executeSql(
            "INSERT INTO license_audit (fingerprint, edition, license_id, maintenance_until, recorded_at) VALUES (?, ?, ?, ?, ?)",
            [entry.fingerprint, entry.edition, entry.license_id, entry.maintenance_until, entry.recorded_at]
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
    database();
    if (__memoryMode) {
        if (limit && limit > 0)
            return __memoryAudit.slice(0, limit);
        return __memoryAudit.slice();
    }

    var records = [];
    __db.transaction(function(tx) {
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
    database();
    if (__memoryMode) {
        __memoryAudit = [];
        return;
    }
    __db.transaction(function(tx) {
        tx.executeSql("DELETE FROM license_audit");
    });
}

function totalAuditCount() {
    database();
    if (__memoryMode)
        return __memoryAudit.length;
    var count = 0;
    __db.transaction(function(tx) {
        var rs = tx.executeSql("SELECT COUNT(*) AS total FROM license_audit");
        if (rs.rows.length > 0)
            count = rs.rows.item(0).total;
    });
    return count;
}
