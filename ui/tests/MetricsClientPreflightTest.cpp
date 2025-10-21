#include <QtTest/QtTest>

#include <QCryptographicHash>
#include <QFile>
#include <QIODevice>
#include <QMultiMap>
#include <QStringList>
#include <QTemporaryDir>

#include "grpc/MetricsClient.hpp"

namespace {

QString writeRootCertificate(const QTemporaryDir& dir, const QString& name = QStringLiteral("root.pem"))
{
    const QString path = dir.filePath(name);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QFAIL("Nie udało się zapisać pliku certyfikatu root CA na potrzeby testu");
    }
    static const QByteArray kCertificate =
        "-----BEGIN CERTIFICATE-----\n"
        "MIIBmjCCAQACCQDK7LE9oM1iPTANBgkqhkiG9w0BAQsFADAUMRIwEAYDVQQDDAlsb2NhbGhvc3Qw\n"
        "HhcNMjQxMDAxMDAwMDAwWhcNMzQwOTI4MDAwMDAwWjAUMRIwEAYDVQQDDAlsb2NhbGhvc3QwXDAN\n"
        "BgkqhkiG9w0BAQEFAANLADBIAkEAy4aTxy5ap39w0DxZ8FDRD75iQ3rZ4NLgr6mNfGXJw9jVdL2Q\n"
        "2v0JS0m/DxGiX1R2jTxc6NVhZrxqyo+ZFy3ODwIDAQABMA0GCSqGSIb3DQEBCwUAA0EAq5Ouv7c3\n"
        "PuYX6+J18NX7YwHtH9g5bbp2o7CIJx9if6W/kiw2J3G1MUrV0egfrS6uBjMZuh1v8C8otINe4oKkg\n"
        "6A==\n"
        "-----END CERTIFICATE-----\n";
    file.write(kCertificate);
    file.close();
    return path;
}

QString computeFingerprint(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QFAIL("Nie udało się otworzyć certyfikatu root CA do obliczenia fingerprintu");
    }
    const QByteArray data = file.readAll();
    file.close();
    return QString::fromUtf8(QCryptographicHash::hash(data, QCryptographicHash::Sha256).toHex());
}

bool containsMessageLike(const QStringList& messages, const QString& needle)
{
    for (const QString& message : messages) {
        if (message.contains(needle, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}

} // namespace

class MetricsClientPreflightTest : public QObject {
    Q_OBJECT

private slots:
    void metadataIncludesTokenRole();
    void metadataOmitsOptionalFields();
    void preflightRejectsMissingEndpoint();
    void preflightValidatesTlsMaterial();
    void preflightDetectsFingerprintMismatch();
    void preflightSucceedsWithValidConfig();
};

void MetricsClientPreflightTest::metadataIncludesTokenRole()
{
    MetricsClient client;
    client.setAuthToken(QStringLiteral("token-xyz"));
    client.setRbacRole(QStringLiteral("ops"));

    const auto metadata = client.authMetadataForTesting();
    QMultiMap<QByteArray, QByteArray> map;
    for (const auto& item : metadata) {
        map.insert(item.first, item.second);
    }

    QCOMPARE(map.value("authorization"), QByteArray("Bearer token-xyz"));
    QCOMPARE(map.value("x-bot-role"), QByteArray("ops"));
    const QList<QByteArray> scopes = map.values("x-bot-scope");
    QCOMPARE(scopes.size(), 1);
    QCOMPARE(scopes.first(), QByteArray("metrics.write"));
}

void MetricsClientPreflightTest::metadataOmitsOptionalFields()
{
    MetricsClient client;

    const auto metadata = client.authMetadataForTesting();
    QMultiMap<QByteArray, QByteArray> map;
    for (const auto& item : metadata) {
        map.insert(item.first, item.second);
    }

    QVERIFY(!map.contains("authorization"));
    QVERIFY(!map.contains("x-bot-role"));
    QCOMPARE(map.value("x-bot-scope"), QByteArray("metrics.write"));
}

void MetricsClientPreflightTest::preflightRejectsMissingEndpoint()
{
    MetricsClient client;
    auto result = client.runPreflightChecklist();
    QVERIFY(!result.ok);
    QVERIFY(!result.errors.isEmpty());
    QVERIFY(result.errors.first().contains(QStringLiteral("Endpoint")));
}

void MetricsClientPreflightTest::preflightValidatesTlsMaterial()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());

    const QString rootPath = writeRootCertificate(dir);

    MetricsClient client;
    client.setEndpoint(QStringLiteral("metrics.example:8443"));

    TelemetryTlsConfig tls;
    tls.enabled = true;
    tls.rootCertificatePath = rootPath;
    tls.clientCertificatePath = dir.filePath(QStringLiteral("client.crt"));
    tls.clientKeyPath = QString();
    client.setTlsConfig(tls);

    const auto result = client.runPreflightChecklist();
    QVERIFY(!result.ok);
    QVERIFY(containsMessageLike(result.errors, QStringLiteral(
        "Konfiguracja mTLS MetricsService wymaga zarówno certyfikatu, jak i klucza klienta.")));
    QVERIFY(containsMessageLike(result.errors, QStringLiteral(
        "Klucz klienta MetricsService nie istnieje")));
}

void MetricsClientPreflightTest::preflightDetectsFingerprintMismatch()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString rootPath = writeRootCertificate(dir);

    MetricsClient client;
    client.setEndpoint(QStringLiteral("metrics.example:8443"));

    TelemetryTlsConfig tls;
    tls.enabled = true;
    tls.rootCertificatePath = rootPath;
    tls.pinnedServerSha256 = QStringLiteral("deadbeef");
    client.setTlsConfig(tls);

    const auto result = client.runPreflightChecklist();
    QVERIFY(!result.ok);
    QVERIFY(containsMessageLike(result.errors, QStringLiteral(
        "Fingerprint root CA nie pasuje do metrics-server-sha256.")));
}

void MetricsClientPreflightTest::preflightSucceedsWithValidConfig()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString rootPath = writeRootCertificate(dir);
    const QString fingerprint = computeFingerprint(rootPath);

    MetricsClient client;
    client.setEndpoint(QStringLiteral("metrics.example:8443"));

    TelemetryTlsConfig tls;
    tls.enabled = true;
    tls.rootCertificatePath = rootPath;
    tls.pinnedServerSha256 = fingerprint;
    client.setTlsConfig(tls);

    const auto result = client.runPreflightChecklist();
    QVERIFY(result.ok);
    QVERIFY(result.errors.isEmpty());
}

QTEST_MAIN(MetricsClientPreflightTest)
#include "MetricsClientPreflightTest.moc"
