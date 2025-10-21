#include <QtTest/QtTest>
#include <QSignalSpy>

#include <memory>

#include "health/HealthStatusController.hpp"

class FakeHealthClient : public HealthClientInterface {
public:
    void setEndpoint(const QString& endpoint) override { m_endpoint = endpoint; }
    void setTlsConfig(const GrpcTlsConfig& config) override { m_tls = config; }
    void setAuthToken(const QString& token) override { m_token = token; }
    void setRbacRole(const QString& role) override { m_role = role; }
    void setRbacScopes(const QStringList& scopes) override { m_scopes = scopes; }

    QVector<QPair<QByteArray, QByteArray>> authMetadataForTesting() const override { return {}; }

    HealthCheckResult check() override
    {
        ++checkCount;
        return nextResult;
    }

    QString            m_endpoint;
    GrpcTlsConfig      m_tls;
    QString            m_token;
    QString            m_role;
    QStringList        m_scopes;
    HealthCheckResult  nextResult;
    int                checkCount = 0;
};

class HealthStatusControllerTest : public QObject {
    Q_OBJECT

private slots:
    void refreshUpdatesProperties();
    void refreshHandlesFailure();
    void autoRefreshRespectsToggle();
};

void HealthStatusControllerTest::refreshUpdatesProperties()
{
    auto client = std::make_shared<FakeHealthClient>();
    client->nextResult.ok = true;
    client->nextResult.version = QStringLiteral("1.2.3");
    client->nextResult.gitCommit = QStringLiteral("abcdef123456");
    client->nextResult.startedAtUtc = QDateTime::currentDateTimeUtc().addSecs(-3600);

    HealthStatusController controller;
    controller.setHealthClientForTesting(client);
    controller.setAutoRefreshEnabled(false);
    controller.refresh();

    QTRY_VERIFY_WITH_TIMEOUT(controller.healthy(), 1000);
    QCOMPARE(controller.version(), QStringLiteral("1.2.3"));
    QCOMPARE(controller.gitCommitShort(), QStringLiteral("abcdef12"));
    QVERIFY(controller.startedAt().contains(QStringLiteral("T")));
    QVERIFY(controller.statusMessage().contains(QStringLiteral("OK")));
}

void HealthStatusControllerTest::refreshHandlesFailure()
{
    auto client = std::make_shared<FakeHealthClient>();
    client->nextResult.ok = false;
    client->nextResult.errorMessage = QStringLiteral("certificate mismatch");

    HealthStatusController controller;
    controller.setHealthClientForTesting(client);
    controller.setAutoRefreshEnabled(false);
    controller.refresh();

    QTRY_VERIFY_WITH_TIMEOUT(!controller.healthy(), 1000);
    QVERIFY(controller.statusMessage().contains(QStringLiteral("błąd")));
    QVERIFY(!controller.lastCheckedAt().isEmpty());
}

void HealthStatusControllerTest::autoRefreshRespectsToggle()
{
    auto client = std::make_shared<FakeHealthClient>();
    client->nextResult.ok = true;
    client->nextResult.startedAtUtc = QDateTime::currentDateTimeUtc();

    HealthStatusController controller;
    controller.setHealthClientForTesting(client);
    controller.setRefreshIntervalSeconds(5);
    controller.setAutoRefreshEnabled(true);
    controller.refresh();

    QTRY_COMPARE_WITH_TIMEOUT(client->checkCount, 1, 1000);
    QTRY_VERIFY_WITH_TIMEOUT(controller.isTimerActiveForTesting(), 500);

    controller.setAutoRefreshEnabled(false);
    QVERIFY(!controller.isTimerActiveForTesting());
}

QTEST_MAIN(HealthStatusControllerTest)
#include "HealthStatusControllerTest.moc"

