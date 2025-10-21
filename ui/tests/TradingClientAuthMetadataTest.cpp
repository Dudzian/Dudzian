#include <QtTest/QtTest>
#include <QCryptographicHash>
#include <QFile>
#include <QMultiMap>
#include <QSet>
#include <QSslCertificate>
#include <QStringList>
#include <QTemporaryDir>

#include "grpc/TradingClient.hpp"

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

QString writeCertificate(const QTemporaryDir& dir, const QString& name = QStringLiteral("root.pem"))
{
    const QString path = dir.filePath(name);
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QFAIL("Nie udało się zapisać przykładowego certyfikatu TLS");
        return {};
    }
    file.write(kSampleCertificatePem);
    file.close();
    return path;
}

QString computeFingerprintSha256(const QString& path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        QFAIL("Nie udało się odczytać certyfikatu TLS do obliczenia fingerprintu");
        return {};
    }
    const QByteArray data = file.readAll();
    file.close();
    const QList<QSslCertificate> certs = QSslCertificate::fromData(data, QSsl::Pem);
    if (certs.isEmpty()) {
        QFAIL("Brak certyfikatów w pliku testowym TLS");
        return {};
    }
    return QString::fromLatin1(certs.first().digest(QCryptographicHash::Sha256).toHex());
}

bool containsErrorLike(const QStringList& errors, const QString& needle)
{
    for (const QString& error : errors) {
        if (error.contains(needle, Qt::CaseInsensitive)) {
            return true;
        }
    }
    return false;
}

} // namespace

class TradingClientAuthMetadataTest : public QObject {
    Q_OBJECT
private slots:
    void metadataIncludesTokenRoleScopes();
    void scopesAreNormalized();
    void metadataClearsWhenEmpty();
    void checklistRejectsMissingEndpoint();
    void checklistDetectsFingerprintMismatch();
    void checklistAcceptsPinnedFingerprint();
    void checklistRequiresClientCertificateWhenMtls();
};

void TradingClientAuthMetadataTest::metadataIncludesTokenRoleScopes()
{
    TradingClient client;
    client.setAuthToken(QStringLiteral("token-123"));
    client.setRbacRole(QStringLiteral("operator"));
    client.setRbacScopes(QStringList{QStringLiteral("metrics.write"), QStringLiteral("trading.read")});

    const auto metadata = client.authMetadataForTesting();
    QMultiMap<QByteArray, QByteArray> map;
    for (const auto& entry : metadata) {
        map.insert(entry.first, entry.second);
    }

    QCOMPARE(map.value("authorization"), QByteArray("Bearer token-123"));
    QCOMPARE(map.value("x-bot-role"), QByteArray("operator"));

    const auto scopes = map.values("x-bot-scope");
    QSet<QByteArray> scopeSet = scopes.toSet();
    QCOMPARE(scopeSet, QSet<QByteArray>({QByteArray("metrics.write"), QByteArray("trading.read")}));
}

void TradingClientAuthMetadataTest::scopesAreNormalized()
{
    TradingClient client;
    client.setRbacScopes(QStringList{QStringLiteral(" metrics.write "), QString(), QStringLiteral("metrics.write"),
                                     QStringLiteral("trading.read")});
    const auto metadata = client.authMetadataForTesting();
    QStringList scopes;
    for (const auto& entry : metadata) {
        if (entry.first == QByteArray("x-bot-scope")) {
            scopes.append(QString::fromUtf8(entry.second));
        }
    }
    QCOMPARE(scopes, QStringList{QStringLiteral("metrics.write"), QStringLiteral("trading.read")});
}

void TradingClientAuthMetadataTest::metadataClearsWhenEmpty()
{
    TradingClient client;
    client.setAuthToken(QStringLiteral("temp"));
    client.setRbacRole(QStringLiteral("role"));
    client.setRbacScopes(QStringList{QStringLiteral("scope")});

    client.setAuthToken(QString());
    client.setRbacRole(QString());
    client.setRbacScopes(QStringList{});

    const auto metadata = client.authMetadataForTesting();
    QVERIFY(metadata.isEmpty());
}

void TradingClientAuthMetadataTest::checklistRejectsMissingEndpoint()
{
    TradingClient client;
    client.setEndpoint(QString());

    const auto result = client.runPreLiveChecklist();
    QVERIFY(!result.ok);
    QVERIFY(containsErrorLike(result.errors, QStringLiteral("Endpoint gRPC nie może być pusty")));
}

void TradingClientAuthMetadataTest::checklistDetectsFingerprintMismatch()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString certPath = writeCertificate(dir);

    TradingClient client;
    client.setEndpoint(QStringLiteral("localhost:50051"));

    TradingClient::TlsConfig tls;
    tls.enabled = true;
    tls.rootCertificatePath = certPath;
    tls.pinnedServerFingerprint = QStringLiteral("deadbeef");
    client.setTlsConfig(tls);

    const auto result = client.runPreLiveChecklist();
    QVERIFY(!result.ok);
    QVERIFY(containsErrorLike(result.errors, QStringLiteral("Fingerprint root CA nie pasuje")));
}

void TradingClientAuthMetadataTest::checklistAcceptsPinnedFingerprint()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString certPath = writeCertificate(dir);
    const QString fingerprint = computeFingerprintSha256(certPath);

    TradingClient client;
    client.setEndpoint(QStringLiteral("localhost:50051"));

    TradingClient::TlsConfig tls;
    tls.enabled = true;
    tls.rootCertificatePath = certPath;
    tls.pinnedServerFingerprint = fingerprint;
    client.setTlsConfig(tls);

    const auto result = client.runPreLiveChecklist();
    QVERIFY(result.ok);
    QVERIFY(result.errors.isEmpty());
}

void TradingClientAuthMetadataTest::checklistRequiresClientCertificateWhenMtls()
{
    TradingClient client;
    client.setEndpoint(QStringLiteral("localhost:50051"));

    TradingClient::TlsConfig tls;
    tls.enabled = true;
    tls.requireClientAuth = true;
    tls.rootCertificatePath = QStringLiteral("/tmp/nonexistent_root.pem");
    client.setTlsConfig(tls);

    const auto result = client.runPreLiveChecklist();
    QVERIFY(!result.ok);
    QVERIFY(containsErrorLike(result.errors, QStringLiteral("mTLS wymaga")));
}

QTEST_MAIN(TradingClientAuthMetadataTest)
#include "TradingClientAuthMetadataTest.moc"

