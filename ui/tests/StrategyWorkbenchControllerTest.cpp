#include <QtTest/QtTest>

#include <QFile>
#include <QFileDevice>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>

#include "app/StrategyWorkbenchController.hpp"

class StrategyWorkbenchControllerTest : public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void autoRefreshesWhenConfigChanges();
    void autoRefreshesWhenBridgeScriptChanges();
    void initialRefreshIsTriggeredOnceReady();
    void autoRefreshesWhenPythonExecutableChanges();
    void autoRefreshesWhenPythonExecutableChangesOnDisk();
};

namespace {
QString writeStubBridgeScript(const QString& path)
{
    QFile script(path);
    if (!script.open(QIODevice::WriteOnly | QIODevice::Text))
        return {};

    const QByteArray payload = QByteArrayLiteral(
        "import json\n"
        "import sys\n"
        "\n"
        "CATALOG = {\n"
        "    'engines': ['alpha'],\n"
        "    'definitions': [{\n"
        "        'name': 'alpha-strategy',\n"
        "        'engine': 'alpha',\n"
        "        'risk_profile': 'growth'\n"
        "    }],\n"
        "    'metadata': {},\n"
        "    'blocked': {}\n"
        "}\n"
        "\n"
        "VALIDATION = {\n"
        "    'ok': True,\n"
        "    'issues': [],\n"
        "    'preset': {'strategies': []}\n"
        "}\n"
        "\n"
        "def main():\n"
        "    argv = set(sys.argv[1:])\n"
        "    if '--describe-catalog' in argv:\n"
        "        json.dump(CATALOG, sys.stdout)\n"
        "        sys.stdout.write('\\n')\n"
        "        return 0\n"
        "    if '--preset-wizard' in argv:\n"
        "        json.dump(VALIDATION, sys.stdout)\n"
        "        sys.stdout.write('\\n')\n"
        "        return 0\n"
        "    json.dump({}, sys.stdout)\n"
        "    sys.stdout.write('\\n')\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n");

    script.write(payload);
    script.close();
    return path;
}

QString writePythonWrapper(const QString& path)
{
    QString pythonExec = QStandardPaths::findExecutable(QStringLiteral("python3"));
    if (pythonExec.isEmpty())
        pythonExec = QStandardPaths::findExecutable(QStringLiteral("python"));
    if (pythonExec.isEmpty())
        return {};

    QFile wrapper(path);
    if (!wrapper.open(QIODevice::WriteOnly | QIODevice::Text))
        return {};

    wrapper.write("#!/bin/sh\n");
    wrapper.write("exec ");
    wrapper.write(pythonExec.toUtf8());
    wrapper.write(" \"$@\"\n");
    wrapper.close();

    QFileDevice::Permissions perms = QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner |
                                     QFileDevice::ReadGroup | QFileDevice::ExeGroup |
                                     QFileDevice::ReadOther | QFileDevice::ExeOther;
    if (!QFile::setPermissions(path, perms))
        return {};

    return path;
}
}

void StrategyWorkbenchControllerTest::autoRefreshesWhenConfigChanges()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString configPath = dir.filePath(QStringLiteral("core.yaml"));
    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    StrategyWorkbenchController controller;
    controller.setConfigPath(configPath);
    controller.setScriptPath(scriptPath);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);
    QVERIFY(controller.refreshCatalog());
    QVERIFY(catalogSpy.wait(2000));
    QVERIFY(catalogSpy.count() >= 1);

    QFile configAppend(configPath);
    QVERIFY(configAppend.open(QIODevice::Append | QIODevice::Text));
    configAppend.write("# touch\n");
    configAppend.close();

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= 2);

    QVERIFY(QFile::remove(configPath));
    QFile configRecreate(configPath);
    QVERIFY(configRecreate.open(QIODevice::WriteOnly | QIODevice::Text));
    configRecreate.write("multi_strategy_schedulers: {}\n# recreated\n");
    configRecreate.close();

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= 3);
    QVERIFY(controller.lastError().isEmpty());
}

void StrategyWorkbenchControllerTest::autoRefreshesWhenBridgeScriptChanges()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString configPath = dir.filePath(QStringLiteral("core.yaml"));
    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    StrategyWorkbenchController controller;
    controller.setConfigPath(configPath);
    controller.setScriptPath(scriptPath);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);
    QVERIFY(controller.refreshCatalog());
    QVERIFY(catalogSpy.wait(2000));
    QVERIFY(catalogSpy.count() >= 1);

    QFile scriptUpdate(scriptPath);
    QVERIFY(scriptUpdate.open(QIODevice::Append | QIODevice::Text));
    scriptUpdate.write("# touch\n");
    scriptUpdate.close();

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= 2);
    QVERIFY(controller.lastError().isEmpty());
}

void StrategyWorkbenchControllerTest::initialRefreshIsTriggeredOnceReady()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString configPath = dir.filePath(QStringLiteral("core.yaml"));
    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    StrategyWorkbenchController controller;
    controller.setScriptPath(scriptPath);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);

    controller.setConfigPath(configPath);

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= 1);
}

void StrategyWorkbenchControllerTest::autoRefreshesWhenPythonExecutableChanges()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString configPath = dir.filePath(QStringLiteral("core.yaml"));
    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    StrategyWorkbenchController controller;
    controller.setConfigPath(configPath);
    controller.setScriptPath(scriptPath);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);
    QVERIFY(controller.refreshCatalog());
    QVERIFY(catalogSpy.wait(2000));
    const int baseline = catalogSpy.count();

    QString pythonExec = QStandardPaths::findExecutable(QStringLiteral("python3"));
    if (pythonExec.isEmpty())
        pythonExec = QStandardPaths::findExecutable(QStringLiteral("python"));
    QVERIFY2(!pythonExec.isEmpty(), "Python interpreter not found on PATH");

    controller.setPythonExecutable(pythonExec);

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= baseline + 1);
}

void StrategyWorkbenchControllerTest::autoRefreshesWhenPythonExecutableChangesOnDisk()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString configPath = dir.filePath(QStringLiteral("core.yaml"));
    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    const QString pythonWrapper = writePythonWrapper(dir.filePath(QStringLiteral("python-wrapper.sh")));
    QVERIFY2(!pythonWrapper.isEmpty(), "Failed to prepare python wrapper");

    StrategyWorkbenchController controller;
    controller.setConfigPath(configPath);
    controller.setScriptPath(scriptPath);
    controller.setPythonExecutable(pythonWrapper);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);
    QVERIFY(controller.refreshCatalog());
    QVERIFY(catalogSpy.wait(2000));
    const int baseline = catalogSpy.count();

    QFile wrapperUpdate(pythonWrapper);
    QVERIFY(wrapperUpdate.open(QIODevice::Append | QIODevice::Text));
    wrapperUpdate.write("# touch\n");
    wrapperUpdate.close();

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= baseline + 1);
    QVERIFY(controller.lastError().isEmpty());
}

QTEST_GUILESS_MAIN(StrategyWorkbenchControllerTest)
#include "StrategyWorkbenchControllerTest.moc"
