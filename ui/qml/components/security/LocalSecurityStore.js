.pragma library
.import QtQuick.LocalStorage 2.0 as LocalStorage

const kMaxAuditRows = 200;

var __db = null;
var __memoryMode = false;
var __memoryStore = [];
var __memoryNextId = 1;

function __resetMemoryStore() {
    __memoryStore = [];
    __memoryNextId = 1;
}

function __activateMemoryMode() {
    __memoryMode = true;
    __db = null;
    __resetMemoryStore();
}

function forceMemoryMode() {
    __activateMemoryMode();
}

function useDiskStorage() {
    __memoryMode = false;
    __db = null;
    __resetMemoryStore();
}

function isMemoryMode() {
    return __memoryMode;
}

function database() {
    if (__memoryMode)
        return null;
    if (__db === null) {
        try {
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
        } catch (err) {
            __activateMemoryMode();
            return null;
        }
    }
    return __db;
}

function __storeMemoryRecord(record) {
    var payload = {
        id: __memoryNextId++,
        fingerprint: record.fingerprint || "",
        edition: record.edition || "",
        license_id: record.license_id || "",
        maintenance_until: record.maintenance_until || "",
        recorded_at: record.recorded_at || new Date().toISOString()
    };
    __memoryStore.unshift(payload);
    if (kMaxAuditRows > 0 && __memoryStore.length > kMaxAuditRows)
        __memoryStore = __memoryStore.slice(0, kMaxAuditRows);
    return payload;
}

function addLicenseSnapshot(snapshot) {
    if (!snapshot)
        return false;
    var fingerprint = snapshot.fingerprint || "";
    var edition = snapshot.edition || snapshot.profile || "";
    var licenseId = snapshot.license_id || snapshot.licenseId || "";
    var maintenance = snapshot.maintenance_until || snapshot.maintenanceUntil || "";
    var recordedAt = snapshot.recorded_at || snapshot.recordedAt || new Date().toISOString();

    var db = database();
    if (db) {
        try {
            db.transaction(function(tx) {
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
        } catch (err) {
            __activateMemoryMode();
        }
    }
    __storeMemoryRecord({
        fingerprint: fingerprint,
        edition: edition,
        license_id: licenseId,
        maintenance_until: maintenance,
        recorded_at: recordedAt
    });
    return true;
}

function fetchAudit(limit) {
    var records = [];
    var db = database();
    if (db) {
        try {
            db.transaction(function(tx) {
                var sql = "SELECT fingerprint, edition, license_id, maintenance_until, recorded_at FROM license_audit ORDER BY recorded_at DESC";
                if (limit && limit > 0)
                    sql += " LIMIT " + limit;
                var rs = tx.executeSql(sql);
                for (var i = 0; i < rs.rows.length; ++i)
                    records.push(rs.rows.item(i));
            });
            return records;
        } catch (err) {
            __activateMemoryMode();
            records = [];
        }
    }
    if (__memoryMode) {
        var copy = __memoryStore.slice();
        if (limit && limit > 0)
            copy = copy.slice(0, limit);
        for (var j = 0; j < copy.length; ++j) {
            records.push({
                fingerprint: copy[j].fingerprint,
                edition: copy[j].edition,
                license_id: copy[j].license_id,
                maintenance_until: copy[j].maintenance_until,
                recorded_at: copy[j].recorded_at
            });
        }
    }
    return records;
}

function clearAudit() {
    var db = database();
    if (db) {
        try {
            db.transaction(function(tx) {
                tx.executeSql("DELETE FROM license_audit");
            });
            return;
        } catch (err) {
            __activateMemoryMode();
        }
    }
    __resetMemoryStore();
}

function totalAuditCount() {
    var count = 0;
    var db = database();
    if (db) {
        try {
            db.transaction(function(tx) {
                var rs = tx.executeSql("SELECT COUNT(*) AS total FROM license_audit");
                if (rs.rows.length > 0)
                    count = rs.rows.item(0).total;
            });
            return count;
        } catch (err) {
            __activateMemoryMode();
        }
    }
    if (__memoryMode)
        count = __memoryStore.length;
    return count;
}
