#include <QtTest>

#include <QByteArray>
#include <QDir>
#include <QStringList>
#include <QTemporaryDir>
#include <QUrl>

#include "utils/PathUtils.hpp"

using bot::shell::utils::expandPath;
using bot::shell::utils::watchableDirectories;

class PathUtilsTest : public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void expandPath_resolvesHome()
    {
        const QString home = QDir::homePath();
        QCOMPARE(expandPath(QStringLiteral("~")), QDir::cleanPath(home));
        QCOMPARE(expandPath(QStringLiteral("~/data")), QDir::cleanPath(home + QStringLiteral("/data")));
    }

    void expandPath_resolvesEnvironment()
    {
        const QByteArray varName = QByteArrayLiteral("BOT_PATH_UTILS_TEST");
        const QString value = QDir::homePath() + QStringLiteral("/env-test");
        QVERIFY(qputenv(varName.constData(), value.toUtf8()));

        const QString expanded = expandPath(QStringLiteral("$BOT_PATH_UTILS_TEST/cache"));
        QCOMPARE(expanded, QDir::cleanPath(value + QStringLiteral("/cache")));

        const QString braced = expandPath(QStringLiteral("${BOT_PATH_UTILS_TEST}/cfg"));
        QCOMPARE(braced, QDir::cleanPath(value + QStringLiteral("/cfg")));
    }

    void expandPath_resolvesWindowsStyle()
    {
        const QByteArray varName = QByteArrayLiteral("BOT_PATH_UTILS_WIN");
        const QString value = QDir::homePath() + QStringLiteral("/win-test");
        QVERIFY(qputenv(varName.constData(), value.toUtf8()));

        const QString expanded = expandPath(QStringLiteral("%BOT_PATH_UTILS_WIN%/logs"));
        QCOMPARE(expanded, QDir::cleanPath(value + QStringLiteral("/logs")));
    }

    void expandPath_normalizesWindowsDrivePrefixes()
    {
        const QString expanded = expandPath(QStringLiteral("C:/ProgramData/OEM/license.lic"));
        QCOMPARE(expanded, QStringLiteral("C:/ProgramData/OEM/license.lic"));

        const QString backslashVariant = expandPath(QStringLiteral("D:\\Apps\\bot\\config.json"));
        QCOMPARE(backslashVariant, QStringLiteral("D:/Apps/bot/config.json"));
    }

    void expandPath_handlesUncShares()
    {
        const QString unc = expandPath(QStringLiteral("\\\\SERVER\\Share\\licenses"));
        QCOMPARE(unc, QStringLiteral("//SERVER/Share/licenses"));
    }

    void expandPath_resolvesFileUrl()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QUrl url = QUrl::fromLocalFile(dir.path());
        const QString expanded = expandPath(url.toString(QUrl::PreferLocalFile));
        QCOMPARE(expanded, QDir::cleanPath(dir.path()));

        const QString explicitUrl = QStringLiteral("file://%1/data").arg(dir.path());
        const QString expandedExplicit = expandPath(explicitUrl);
        QCOMPARE(expandedExplicit, QDir::cleanPath(dir.path() + QStringLiteral("/data")));
    }

#if defined(Q_OS_UNIX)
    void expandPath_resolvesUserHome()
    {
        const QByteArray userEnv = qgetenv("USER");
        if (userEnv.isEmpty())
            QSKIP("Brak zmiennej USER – pomijam test ~user.");

        const QString username = QString::fromLocal8Bit(userEnv);
        const QString path = expandPath(QStringLiteral("~%1/.config").arg(username));
        const QString expected = QDir::cleanPath(QDir::homePath() + QStringLiteral("/.config"));
        QCOMPARE(path, expected);
    }
#endif

    void expandPath_keepsUnknownVariables()
    {
        const QString original = QStringLiteral("$BOT_PATH_UTILS_UNKNOWN/data");
        QCOMPARE(expandPath(original), QDir::cleanPath(QDir::current().absoluteFilePath(original)));
    }

    void watchableDirectories_existingPath()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QStringList dirs = watchableDirectories(dir.path());
        QCOMPARE(dirs.size(), 1);
        QCOMPARE(QDir(dirs.first()).canonicalPath(), QDir(dir.path()).canonicalPath());
    }

    void watchableDirectories_missingPath()
    {
        QTemporaryDir dir;
        QVERIFY(dir.isValid());

        const QString nested = dir.path() + QStringLiteral("/a/b/c");
        const QStringList dirs = watchableDirectories(nested);
        QCOMPARE(dirs.size(), 1);
        QCOMPARE(QDir(dirs.first()).canonicalPath(), QDir(dir.path()).canonicalPath());
    }
};

QTEST_MAIN(PathUtilsTest)
#include "PathUtilsTest.moc"

