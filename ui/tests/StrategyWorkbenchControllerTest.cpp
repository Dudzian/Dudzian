#include <QtTest/QtTest>

#include <QDir>
#include <QFile>
#include <QFileDevice>
#include <QProcessEnvironment>
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
    void acceptsTildePaths();
    void autoRefreshesWhenPythonCanonicalTargetChanges();
    void acceptsEnvironmentVariablePaths();
    void reportsReadableErrorsWhenPrerequisitesMissing();
    void reportsPermissionErrorsWhenFilesNotReadable();
    void autoRefreshesWhenConfigPathMaterializesWithinNestedDirectories();
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

void StrategyWorkbenchControllerTest::autoRefreshesWhenConfigPathMaterializesWithinNestedDirectories()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    const QString nestedDir = dir.filePath(QStringLiteral("configs/nested"));
    const QString configPath = nestedDir + QStringLiteral("/core.yaml");

    StrategyWorkbenchController controller;
    controller.setScriptPath(scriptPath);
    controller.setConfigPath(configPath);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);

    QVERIFY(QDir().mkpath(nestedDir));

    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    QVERIFY(catalogSpy.wait(4000));
    QVERIFY(catalogSpy.count() >= 1);
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

void StrategyWorkbenchControllerTest::acceptsTildePaths()
{
    const QString homePath = QDir::homePath();
    if (homePath.isEmpty())
        QSKIP("Home directory is not available");

    QTemporaryDir dir(QDir(homePath).filePath(QStringLiteral("strategy-workbench-XXXXXX")));
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

    QVERIFY(configPath.startsWith(homePath));
    QVERIFY(scriptPath.startsWith(homePath));
    QVERIFY(pythonWrapper.startsWith(homePath));

    const QString configTilde = QStringLiteral("~%1").arg(configPath.mid(homePath.size()));
    const QString scriptTilde = QStringLiteral("~%1").arg(scriptPath.mid(homePath.size()));
    const QString pythonTilde = QStringLiteral("~%1").arg(pythonWrapper.mid(homePath.size()));

    StrategyWorkbenchController controller;
    controller.setConfigPath(configTilde);
    controller.setScriptPath(scriptTilde);
    controller.setPythonExecutable(pythonTilde);

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);
    QVERIFY(controller.refreshCatalog());
    QVERIFY(catalogSpy.wait(2000));
    const int baseline = catalogSpy.count();

    QFile configUpdate(configPath);
    QVERIFY(configUpdate.open(QIODevice::Append | QIODevice::Text));
    configUpdate.write("# touch\n");
    configUpdate.close();

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= baseline + 1);
    QVERIFY(controller.lastError().isEmpty());
}

void StrategyWorkbenchControllerTest::autoRefreshesWhenPythonCanonicalTargetChanges()
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

    const QString pythonLink = dir.filePath(QStringLiteral("python-link.sh"));
    if (!QFile::link(pythonWrapper, pythonLink))
        QSKIP("Symlinks are not supported on this platform");

    StrategyWorkbenchController controller;
    controller.setConfigPath(configPath);
    controller.setScriptPath(scriptPath);
    controller.setPythonExecutable(pythonLink);

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
}

void StrategyWorkbenchControllerTest::acceptsEnvironmentVariablePaths()
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

    QVERIFY(qputenv("WB_CONFIG_PATH", QFile::encodeName(configPath)));
    QVERIFY(qputenv("WB_SCRIPT_PATH", QFile::encodeName(scriptPath)));
    QVERIFY(qputenv("WB_PYTHON_PATH", QFile::encodeName(pythonWrapper)));

    StrategyWorkbenchController controller;
    controller.setConfigPath(QStringLiteral("$WB_CONFIG_PATH"));
    controller.setScriptPath(QStringLiteral("$WB_SCRIPT_PATH"));
    controller.setPythonExecutable(QStringLiteral("$WB_PYTHON_PATH"));

    QSignalSpy catalogSpy(&controller, &StrategyWorkbenchController::catalogChanged);
    QVERIFY(controller.refreshCatalog());
    QVERIFY(catalogSpy.wait(2000));
    QVERIFY(catalogSpy.count() >= 1);

    QFile configAppend(configPath);
    QVERIFY(configAppend.open(QIODevice::Append | QIODevice::Text));
    configAppend.write("# env-touch\n");
    configAppend.close();

    QVERIFY(catalogSpy.wait(3000));
    QVERIFY(catalogSpy.count() >= 2);

    qunsetenv("WB_CONFIG_PATH");
    qunsetenv("WB_SCRIPT_PATH");
    qunsetenv("WB_PYTHON_PATH");
}

void StrategyWorkbenchControllerTest::reportsReadableErrorsWhenPrerequisitesMissing()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Temporary directory not available");

    const QString configPath = dir.filePath(QStringLiteral("core.yaml"));
    const QString scriptPath = writeStubBridgeScript(dir.filePath(QStringLiteral("bridge.py")));
    QVERIFY2(!scriptPath.isEmpty(), "Failed to prepare stub bridge script");

    StrategyWorkbenchController controller;
    controller.setScriptPath(scriptPath);
    controller.setConfigPath(configPath);

    QVERIFY(!controller.refreshCatalog());
    QVERIFY(controller.lastError().contains(QStringLiteral("Plik konfiguracji strategii nie istnieje"))
            || controller.lastError().contains(QStringLiteral("Nie ustawiono ścieżki konfiguracji strategii")));

    QFile configFile(configPath);
    QVERIFY(configFile.open(QIODevice::WriteOnly | QIODevice::Text));
    configFile.write("multi_strategy_schedulers: {}\n");
    configFile.close();

    const QString missingScript = dir.filePath(QStringLiteral("missing_bridge.py"));
    controller.setScriptPath(missingScript);
    QVERIFY(!controller.refreshCatalog());
    QVERIFY(controller.lastError().contains(QStringLiteral("Plik mostka konfiguracji strategii nie istnieje"))
            || controller.lastError().contains(QStringLiteral("Nie ustawiono ścieżki mostka")));

    controller.setScriptPath(scriptPath);

    const QString missingPython = dir.filePath(QStringLiteral("python-missing"));
    controller.setPythonExecutable(missingPython);
    QVERIFY(!controller.refreshCatalog());
    QVERIFY(controller.lastError().contains(QStringLiteral("Interpreter Pythona nie jest dostępny"))
            || controller.lastError().contains(QStringLiteral("Interpreter Pythona nie został odnaleziony"))
            || controller.lastError().contains(QStringLiteral("Nie ustawiono interpretera Pythona")));

    const QString pythonWrapper = writePythonWrapper(dir.filePath(QStringLiteral("python-wrapper.sh")));
    QVERIFY2(!pythonWrapper.isEmpty(), "Failed to prepare python wrapper");
    controller.setPythonExecutable(pythonWrapper);

    QVERIFY(controller.refreshCatalog());
    QVERIFY(controller.lastError().isEmpty());
}

void StrategyWorkbenchControllerTest::reportsPermissionErrorsWhenFilesNotReadable()
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

    QVERIFY(QFile::setPermissions(configPath, QFileDevice::WriteOwner));
    QVERIFY(!controller.refreshCatalog());
    QVERIFY(controller.lastError().contains(QStringLiteral("Plik konfiguracji strategii nie posiada uprawnień do odczytu")));

    QVERIFY(QFile::setPermissions(configPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner));

    QVERIFY(QFile::setPermissions(scriptPath, QFileDevice::WriteOwner));
    QVERIFY(!controller.refreshCatalog());
    QVERIFY(controller.lastError().contains(QStringLiteral("Plik mostka konfiguracji strategii nie posiada uprawnień do odczytu")));

    QVERIFY(QFile::setPermissions(scriptPath,
                                  QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner
                                      | QFileDevice::ReadGroup | QFileDevice::ExeGroup | QFileDevice::ReadOther
                                      | QFileDevice::ExeOther));

    QVERIFY(QFile::setPermissions(pythonWrapper, QFileDevice::ExeOwner));
    QVERIFY(!controller.refreshCatalog());
    QVERIFY(controller.lastError().contains(QStringLiteral("Interpreter Pythona nie posiada uprawnień do odczytu"))
            || controller.lastError().contains(QStringLiteral("Interpreter Pythona nie posiada uprawnień do uruchomienia")));
}

QTEST_GUILESS_MAIN(StrategyWorkbenchControllerTest)
#include "StrategyWorkbenchControllerTest.moc"
