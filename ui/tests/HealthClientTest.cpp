#include <QtTest/QtTest>
#include <QTemporaryDir>
#include <QFile>
#include <QMultiMap>
#include <QSet>

#include <algorithm>

#include "grpc/HealthClient.hpp"

namespace {

constexpr const char* kSampleCertificatePem =
    "-----BEGIN CERTIFICATE-----\n"
    "MIIDCzCCAfOgAwIBAgIUXg2adJP0b1IzyHUIygGCz5e4CZswDQYJKoZIhvcNAQEL\n"
    "BQAwFTETMBEGA1UEAwwKdGVzdC5sb2NhbDAeFw0yNTEwMTIxMzUzMThaFw0yNjEw\n"
    "MTIxMzUzMThaMBUxEzARBgNVBAMMCnRlc3QubG9jYWwwggEiMA0GCSqGSIb3DQEB\n"
    "AQUAA4IBDwAwggEKAoIBAQCoTFYSrtuGtDlfV6zjE1YQZz8rDoW58bid8CcMXAA/\n"
    "6sXvCDYFepgjJpD1dDaXLIORNaoteLJLDwd/GbWhK589n/+KzLfnrS+vC+Y9/Zwr\n"
    "9mCJbQ6liPYleqidG98cY50nrru8IiLORWDDWPLMcNJbGuTkg+JUkmVhaLSmWF1u\n"
    "GLN+5IFJ+NXo3fUW5B1swcGYFlta0KelpNaatyMKQZZ7wO4QXp8H/ajBbXpWnE/n\n"
    "JAYOqdBS03+AVpy2Qr0HnCw8NxSur9pqLY4EDepPgowBaMuVv8XgG8+XYJm6y9HH\n"
    "r1sDd7u1Nq+EnqDfuwZ7HRe5sNQXCW8MwOp+kWqVgrGVAgMBAAGjUzBRMB0GA1Ud\n"
    "DgQWBBQgFv08KgCA0frYHzc8GrR6RFyi+jAfBgNVHSMEGDAWgBQgFv08KgCA0frY\n"
    "Hzc8GrR6RFyi+jAPBgNVHRMBAf8EBTADAQH/MA0GCSqGSIb3DQEBCwUAA4IBAQAP\n"
    "LvBg5dsZoS2IZXZMv89QTJsI2pndypdL8iQND7StJstgp57f18rOIqn6M3Hw+RwN\n"
    "NnFs84Jm9qrPRto8jd/sxFFEKYwraLOYadTdIorRYFNdoSYA8dQwh4vUbWLGL5pV\n"
    "EYrSDEItLWPznRLYx2O5NIo1sM6/LWDDSYbwE9EUCxrCFzF9TQT8lVLQDE2KTCbB\n"
    "EiSKc+jb0Y2FlBCTDbfpK3CrvS2nhP+Vh+M9iyacJNHEhbiEkam0wje5/6gDsBoi\n"
    "ouji8dqM2xUMITUW+BWHn/eDOqLss5ZlRbGTNm0Hq3CP7uf0u15r5QxBJjzZYddr\n"
    "WTcLAxCHylb4RYSCze+i\n"
    "-----END CERTIFICATE-----\n";

QString writeCertificate(const QTemporaryDir& dir)
{
    const QString path = dir.filePath(QStringLiteral("root.pem"));
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QFAIL("Nie udało się zapisać certyfikatu testowego");
        return {};
    }
    file.write(kSampleCertificatePem);
    file.close();
    return path;
}

} // namespace

class HealthClientTest : public QObject {
    Q_OBJECT

private slots:
    void metadataUsesDefaultScope();
    void metadataIncludesTokenAndRole();
    void preflightRejectsMissingEndpoint();
    void preflightValidatesTlsFiles();
    void preflightWarnsOnFingerprintWithoutTls();
    void checkFailsWithoutEndpoint();
};

void HealthClientTest::metadataUsesDefaultScope()
{
    HealthClient client;
    const auto metadata = client.authMetadataForTesting();
    bool hasScope = false;
    for (const auto& entry : metadata) {
        if (entry.first == QByteArrayLiteral("x-bot-scope") && entry.second == QByteArrayLiteral("health.read")) {
            hasScope = true;
        }
    }
    QVERIFY(hasScope);
}

void HealthClientTest::metadataIncludesTokenAndRole()
{
    HealthClient client;
    client.setAuthToken(QStringLiteral("secret"));
    client.setRbacRole(QStringLiteral("ops"));
    client.setRbacScopes(QStringList{QStringLiteral("health.read"), QStringLiteral("health.write")});

    const auto metadata = client.authMetadataForTesting();
    QMultiMap<QByteArray, QByteArray> map;
    for (const auto& entry : metadata) {
        map.insert(entry.first, entry.second);
    }

    QCOMPARE(map.value("authorization"), QByteArray("Bearer secret"));
    QCOMPARE(map.value("x-bot-role"), QByteArray("ops"));
    const auto scopes = map.values("x-bot-scope");
    QSet<QByteArray> scopeSet(scopes.begin(), scopes.end());
    QCOMPARE(scopeSet, QSet<QByteArray>({QByteArray("health.read"), QByteArray("health.write")}));
}

void HealthClientTest::preflightRejectsMissingEndpoint()
{
    HealthClient client;
    const auto result = client.runPreflightChecklist();
    QVERIFY(!result.ok);
    QVERIFY(!result.errors.isEmpty());
}

void HealthClientTest::preflightValidatesTlsFiles()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString certPath = writeCertificate(dir);

    HealthClient client;
    client.setEndpoint(QStringLiteral("localhost:50051"));

    GrpcTlsConfig tls;
    tls.enabled = true;
    tls.rootCertificatePath = QStringLiteral("/path/does/not/exist.pem");
    client.setTlsConfig(tls);

    auto result = client.runPreflightChecklist();
    QVERIFY(!result.ok);
    QVERIFY(std::any_of(result.errors.begin(), result.errors.end(), [](const QString& err) {
        return err.contains(QStringLiteral("root CA"), Qt::CaseInsensitive);
    }));

    tls.rootCertificatePath = certPath;
    tls.requireClientAuth = true;
    client.setTlsConfig(tls);

    result = client.runPreflightChecklist();
    QVERIFY(!result.ok);
    QVERIFY(std::any_of(result.errors.begin(), result.errors.end(), [](const QString& err) {
        return err.contains(QStringLiteral("mTLS"), Qt::CaseInsensitive);
    }));
}

void HealthClientTest::preflightWarnsOnFingerprintWithoutTls()
{
    HealthClient client;
    client.setEndpoint(QStringLiteral("localhost:50051"));

    GrpcTlsConfig tls;
    tls.pinnedServerFingerprint = QStringLiteral("deadbeef");
    client.setTlsConfig(tls);

    const auto result = client.runPreflightChecklist();
    QVERIFY(result.ok);
    QVERIFY(std::any_of(result.warnings.begin(), result.warnings.end(), [](const QString& warning) {
        return warning.contains(QStringLiteral("fingerprint"), Qt::CaseInsensitive);
    }));
}

void HealthClientTest::checkFailsWithoutEndpoint()
{
    HealthClient client;
    const auto result = client.check();
    QVERIFY(!result.ok);
    QVERIFY(result.errorMessage.contains(QStringLiteral("HealthService")) || result.errorMessage.isEmpty());
}

QTEST_MAIN(HealthClientTest)
#include "HealthClientTest.moc"

