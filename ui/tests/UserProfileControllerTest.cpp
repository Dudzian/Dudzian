#include <QtTest/QtTest>

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSignalSpy>
#include <QTemporaryDir>
#include <QStringList>

#include "app/UserProfileController.hpp"

class UserProfileControllerTest : public QObject
{
    Q_OBJECT

private Q_SLOTS:
    void createsDefaultProfileWhenMissing();
    void toggleFavoriteStrategyUsesCatalogSnapshot();
    void applyThemeUpdatesPaletteAndPersists();
    void createProfileGeneratesUniqueId();
    void renameProfileUpdatesDisplayName();
    void duplicateProfileCopiesFavoritesAndTheme();
    void resetProfileRestoresDefaults();
    void paletteOverridesPersistAndClear();
    void wizardProgressTracksSteps();
};

namespace {
QVariantList makeCatalogDefinitions()
{
    QVariantList list;
    QVariantMap alpha;
    alpha.insert(QStringLiteral("name"), QStringLiteral("alpha-trend"));
    QVariantMap alphaMeta;
    alphaMeta.insert(QStringLiteral("popularity"), 4.5);
    alphaMeta.insert(QStringLiteral("risk_profile"), QStringLiteral("balanced"));
    alphaMeta.insert(QStringLiteral("tags"), QStringList{QStringLiteral("trend"), QStringLiteral("swing")});
    alpha.insert(QStringLiteral("metadata"), alphaMeta);
    list.append(alpha);

    QVariantMap beta;
    beta.insert(QStringLiteral("name"), QStringLiteral("beta-mean-reversion"));
    QVariantMap betaMeta;
    betaMeta.insert(QStringLiteral("score"), 2.0);
    beta.insert(QStringLiteral("metadata"), betaMeta);
    list.append(beta);
    return list;
}
}

void UserProfileControllerTest::createsDefaultProfileWhenMissing()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    const QString path = dir.filePath(QStringLiteral("profiles.json"));
    UserProfileController controller;
    controller.setProfilesPath(path);
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QCOMPARE(controller.activeProfileId(), QStringLiteral("default"));
    QCOMPARE(controller.profiles().size(), 1);
    QVERIFY(controller.save());

    QFile file(path);
    QVERIFY2(file.exists(), "Plik profili nie został zapisany");
}

void UserProfileControllerTest::toggleFavoriteStrategyUsesCatalogSnapshot()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    UserProfileController controller;
    controller.setProfilesPath(dir.filePath(QStringLiteral("profiles.json")));
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QStringList favorites = controller.favoriteStrategies(QString());
    QCOMPARE(favorites.size(), 0);

    QVERIFY(controller.toggleFavoriteStrategy(QString(), QStringLiteral("alpha-trend")));
    favorites = controller.favoriteStrategies(QString());
    QCOMPARE(favorites.size(), 1);
    QCOMPARE(favorites.first(), QStringLiteral("alpha-trend"));

    QVERIFY(controller.toggleFavoriteStrategy(QString(), QStringLiteral("alpha-trend")));
    favorites = controller.favoriteStrategies(QString());
    QCOMPARE(favorites.size(), 0);

    QVERIFY(!controller.toggleFavoriteStrategy(QString(), QStringLiteral("unknown-strategy")));
}

void UserProfileControllerTest::applyThemeUpdatesPaletteAndPersists()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    const QString path = dir.filePath(QStringLiteral("profiles.json"));
    UserProfileController controller;
    controller.setProfilesPath(path);
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QVariantMap palette = controller.activeThemePalette();
    QCOMPARE(palette.value(QStringLiteral("accent")).toString(), QStringLiteral("#4FA3FF"));

    QSignalSpy themeSpy(&controller, &UserProfileController::themePaletteChanged);
    QVERIFY(controller.applyTheme(QString(), QStringLiteral("aurora")));
    QVERIFY(themeSpy.wait(100));

    palette = controller.activeThemePalette();
    QCOMPARE(palette.value(QStringLiteral("accent")).toString(), QStringLiteral("#8C7BFF"));
    QVERIFY(controller.save());

    UserProfileController reloaded;
    reloaded.setProfilesPath(path);
    reloaded.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(reloaded.load());
    QCOMPARE(reloaded.activeProfile().value(QStringLiteral("theme")).toString(), QStringLiteral("aurora"));
}

void UserProfileControllerTest::createProfileGeneratesUniqueId()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    UserProfileController controller;
    controller.setProfilesPath(dir.filePath(QStringLiteral("profiles.json")));
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    const QString firstId = controller.createProfile(QStringLiteral("Scalper"));
    QVERIFY(!firstId.isEmpty());
    const QString secondId = controller.createProfile(QStringLiteral("Scalper"));
    QVERIFY(!secondId.isEmpty());
    QVERIFY(firstId != secondId);
    QCOMPARE(controller.activeProfileId(), secondId);
    QCOMPARE(controller.profiles().size(), 3);
}

void UserProfileControllerTest::renameProfileUpdatesDisplayName()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    UserProfileController controller;
    controller.setProfilesPath(dir.filePath(QStringLiteral("profiles.json")));
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    const QString profileId = controller.createProfile(QStringLiteral("Carry"));
    QVERIFY(!profileId.isEmpty());

    QVERIFY(controller.renameProfile(profileId, QStringLiteral("Carry Trade")));
    const QVariantMap details = controller.profileDetails(profileId);
    QCOMPARE(details.value(QStringLiteral("displayName")).toString(), QStringLiteral("Carry Trade"));
}

void UserProfileControllerTest::duplicateProfileCopiesFavoritesAndTheme()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    UserProfileController controller;
    controller.setProfilesPath(dir.filePath(QStringLiteral("profiles.json")));
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QVERIFY(controller.toggleFavoriteStrategy(QString(), QStringLiteral("alpha-trend")));
    QVERIFY(controller.applyTheme(QString(), QStringLiteral("aurora")));

    const QString duplicateId = controller.duplicateProfile(QString(), QStringLiteral("Trader"));
    QVERIFY(!duplicateId.isEmpty());
    QCOMPARE(controller.activeProfileId(), duplicateId);

    const QVariantMap duplicate = controller.profileDetails(duplicateId);
    QCOMPARE(duplicate.value(QStringLiteral("displayName")).toString(), QStringLiteral("Trader"));
    const QStringList favorites = duplicate.value(QStringLiteral("favorites")).toStringList();
    QCOMPARE(favorites.size(), 1);
    QCOMPARE(favorites.first(), QStringLiteral("alpha-trend"));
    QCOMPARE(duplicate.value(QStringLiteral("theme")).toString(), QStringLiteral("aurora"));
}

void UserProfileControllerTest::resetProfileRestoresDefaults()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    UserProfileController controller;
    controller.setProfilesPath(dir.filePath(QStringLiteral("profiles.json")));
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QVERIFY(controller.toggleFavoriteStrategy(QString(), QStringLiteral("alpha-trend")));
    QVERIFY(controller.applyTheme(QString(), QStringLiteral("aurora")));

    QVERIFY(controller.resetProfile(QString()));
    QCOMPARE(controller.favoriteStrategies(QString()).size(), 0);
    QCOMPARE(controller.activeProfile().value(QStringLiteral("theme")).toString(), QStringLiteral("midnight"));
}

void UserProfileControllerTest::paletteOverridesPersistAndClear()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    const QString path = dir.filePath(QStringLiteral("profiles.json"));
    UserProfileController controller;
    controller.setProfilesPath(path);
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QSignalSpy paletteSpy(&controller, &UserProfileController::themePaletteChanged);
    QVERIFY(controller.setPaletteOverride(QString(), QStringLiteral("accent"), QStringLiteral("#ABCDEF")));
    QVERIFY(paletteSpy.count() > 0);

    QVariantMap overrides = controller.paletteOverrides(QString());
    QCOMPARE(overrides.value(QStringLiteral("accent")).toString(), QStringLiteral("#ABCDEF"));
    QCOMPARE(controller.activeThemePalette().value(QStringLiteral("accent")).toString(), QStringLiteral("#ABCDEF"));

    UserProfileController reloaded;
    reloaded.setProfilesPath(path);
    reloaded.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(reloaded.load());
    QVariantMap reloadedOverrides = reloaded.paletteOverrides(QStringLiteral("default"));
    QCOMPARE(reloadedOverrides.value(QStringLiteral("accent")).toString(), QStringLiteral("#ABCDEF"));

    QVERIFY(controller.clearPaletteOverrides(QString()));
    overrides = controller.paletteOverrides(QString());
    QVERIFY(overrides.isEmpty());
    QCOMPARE(controller.activeThemePalette().value(QStringLiteral("accent")).toString(), QStringLiteral("#4FA3FF"));
}

void UserProfileControllerTest::wizardProgressTracksSteps()
{
    QTemporaryDir dir;
    QVERIFY2(dir.isValid(), "Brak katalogu tymczasowego");

    const QString path = dir.filePath(QStringLiteral("profiles.json"));
    UserProfileController controller;
    controller.setProfilesPath(path);
    controller.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(controller.load());

    QVariantMap progress = controller.activeWizardProgress();
    QVERIFY(progress.value(QStringLiteral("completedSteps")).toStringList().isEmpty());
    QVERIFY(!progress.value(QStringLiteral("completed")).toBool());

    QVERIFY(controller.setWizardStepCompleted(QString(), QStringLiteral("license")));
    progress = controller.activeWizardProgress();
    QStringList steps = progress.value(QStringLiteral("completedSteps")).toStringList();
    QCOMPARE(steps.size(), 1);
    QCOMPARE(steps.first(), QStringLiteral("license"));

    QVERIFY(controller.setWizardStepCompleted(QString(), QStringLiteral("connectivity")));
    progress = controller.activeWizardProgress();
    steps = progress.value(QStringLiteral("completedSteps")).toStringList();
    QCOMPARE(steps.size(), 2);
    QVERIFY(steps.contains(QStringLiteral("connectivity")));

    QVERIFY(controller.markWizardCompleted(QString(), true));
    progress = controller.activeWizardProgress();
    QVERIFY(progress.value(QStringLiteral("completed")).toBool());

    QVERIFY(controller.resetWizardProgress(QString()));
    progress = controller.activeWizardProgress();
    QVERIFY(progress.value(QStringLiteral("completedSteps")).toStringList().isEmpty());
    QVERIFY(!progress.value(QStringLiteral("completed")).toBool());

    QVERIFY(controller.save());

    UserProfileController reloaded;
    reloaded.setProfilesPath(path);
    reloaded.setCatalogSnapshot(makeCatalogDefinitions());
    QVERIFY(reloaded.load());
    QVariantMap restored = reloaded.wizardProgress(QStringLiteral("default"));
    QVERIFY(restored.value(QStringLiteral("completedSteps")).toStringList().isEmpty());
    QVERIFY(!restored.value(QStringLiteral("completed")).toBool());
}

QTEST_MAIN(UserProfileControllerTest)
#include "UserProfileControllerTest.moc"
