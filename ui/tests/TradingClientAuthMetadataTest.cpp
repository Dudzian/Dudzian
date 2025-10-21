#include <QtTest/QtTest>
#include <QMultiMap>
#include <QSet>
#include <QStringList>

#include "grpc/TradingClient.hpp"

class TradingClientAuthMetadataTest : public QObject {
    Q_OBJECT
private slots:
    void metadataIncludesTokenRoleScopes();
    void scopesAreNormalized();
    void metadataClearsWhenEmpty();
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

QTEST_MAIN(TradingClientAuthMetadataTest)
#include "TradingClientAuthMetadataTest.moc"
