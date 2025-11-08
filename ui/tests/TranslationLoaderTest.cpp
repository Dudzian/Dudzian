#include <QtTest/QtTest>
#include <QCoreApplication>
#include <QTranslator>
#include <QLocale>
#include <QFile>

class TranslationLoaderTest : public QObject {
    Q_OBJECT

private slots:
    void loadsSystemOrFallback();
};

void TranslationLoaderTest::loadsSystemOrFallback()
{
    int argc = 0;
    QCoreApplication app(argc, nullptr);

    const QLocale systemLocale = QLocale::system();
    QStringList candidates;
    candidates << systemLocale.name() << systemLocale.bcp47Name() << systemLocale.name().left(2) << QStringLiteral("en_US")
               << QStringLiteral("en");

    bool loaded = false;
    QTranslator translator;
    for (const QString& candidate : candidates) {
        const QString resourcePath = QStringLiteral(":/i18n/bot_trading_shell_%1.qm").arg(candidate);
        if (QFile::exists(resourcePath) && translator.load(resourcePath)) {
            loaded = true;
            break;
        }
    }

    QVERIFY2(loaded, "No translation resource could be loaded for system locale or fallback");
}

QTEST_GUILESS_MAIN(TranslationLoaderTest)
#include "TranslationLoaderTest.moc"
