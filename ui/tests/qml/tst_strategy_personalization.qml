import QtQuick
import QtTest
import "../../qml/components" as Components
import "../../qml/dashboard" as Dashboard
import "../../qml/settings" as Settings
import "../../qml/views" as Views

TestCase {
    name: "StrategyPersonalization"

    function createUserProfilesStub() {
        const source = `import QtQuick 2.15; QtObject {
            property string activeProfileId: "default"
            property var activeProfile: ({ id: "default", displayName: "Domyślny profil", theme: "midnight", favorites: ["alpha-trend"] })
            property var profiles: ([{
                id: "default",
                displayName: "Domyślny profil",
                theme: "midnight",
                favorites: ["alpha-trend"]
            }])
            property var availableThemes: ["midnight", "aurora", "solarized"]
            property var activeThemePalette: ({
                backgroundPrimary: "#0E1320",
                backgroundOverlay: "#161C2A",
                surfaceMuted: "#242B3D",
                surfaceStrong: "#1F2536",
                surfaceSubtle: "#2C3448",
                textPrimary: "#F5F7FA",
                textSecondary: "#A4ACC4",
                textTertiary: "#7C86A4",
                accent: "#4FA3FF",
                accentMuted: "#3577D4",
                positive: "#3FD0A4",
                negative: "#FF6B6B",
                warning: "#F8C572"
            })
            property var basePalettes: ({
                midnight: {
                    backgroundPrimary: "#0E1320",
                    backgroundOverlay: "#161C2A",
                    surfaceStrong: "#1F2536",
                    surfaceMuted: "#242B3D",
                    surfaceSubtle: "#2C3448",
                    textPrimary: "#F5F7FA",
                    textSecondary: "#A4ACC4",
                    textTertiary: "#7C86A4",
                    accent: "#4FA3FF",
                    accentMuted: "#3577D4",
                    positive: "#3FD0A4",
                    negative: "#FF6B6B",
                    warning: "#F8C572"
                },
                aurora: {
                    backgroundPrimary: "#101926",
                    backgroundOverlay: "#1A2234",
                    surfaceStrong: "#233044",
                    surfaceMuted: "#2D3B52",
                    surfaceSubtle: "#354765",
                    textPrimary: "#F0F7FF",
                    textSecondary: "#B8C4E0",
                    textTertiary: "#8DA1C7",
                    accent: "#8C7BFF",
                    accentMuted: "#6A5DD8",
                    positive: "#4FDDB5",
                    negative: "#FF7F8C",
                    warning: "#FCD77F"
                },
                solarized: {
                    backgroundPrimary: "#002B36",
                    backgroundOverlay: "#073642",
                    surfaceStrong: "#0B3A46",
                    surfaceMuted: "#0F4552",
                    surfaceSubtle: "#13515F",
                    textPrimary: "#FDF6E3",
                    textSecondary: "#EEE8D5",
                    textTertiary: "#C7BFA6",
                    accent: "#B58900",
                    accentMuted: "#9C7A00",
                    positive: "#859900",
                    negative: "#DC322F",
                    warning: "#CB4B16"
                }
            })
            property var overridesStore: ({})
            property var wizardState: ({ "default": { completedSteps: ["license"], completed: false } })
            property var activeWizardProgress: ({ completedSteps: ["license"], completed: false })
            property string lastToggled: ""
            property string lastAppliedTheme: ""
            property string lastCreatedProfileId: ""
            property string lastRenamedProfileId: ""
            property string lastRenamedProfileName: ""
            property string lastDuplicatedProfileId: ""
            property string lastDuplicatedSourceId: ""
            property string lastResetProfileId: ""
            property string lastOverrideRole: ""
            property string lastOverrideColor: ""
            property bool clearedOverrides: false
            signal profilesChanged()
            signal activeProfileChanged()
            signal themePaletteChanged()
            signal catalogIntegrationChanged()
            signal wizardProgressChanged()
            function cloneWizardProgress(id) {
                const state = wizardState[id] || { completedSteps: [], completed: false }
                return ({
                    completedSteps: (state.completedSteps || []).slice(0),
                    completed: state.completed === true
                })
            }
            function cloneProfiles() { return profiles.slice(0) }
            function refreshProfilesProgress() {
                const next = []
                for (let i = 0; i < profiles.length; ++i) {
                    const item = Object.assign({}, profiles[i])
                    item.setupProgress = cloneWizardProgress(item.id)
                    next.push(item)
                }
                profiles = next
            }
            function updateActiveProfileObject() {
                for (let i = 0; i < profiles.length; ++i) {
                    if (profiles[i].id === activeProfileId) {
                        activeProfile = profiles[i]
                        return
                    }
                }
                activeProfile = ({ id: activeProfileId, theme: "midnight", favorites: [] })
            }
            function refreshActivePalette() {
                const profile = activeProfile || { id: activeProfileId, theme: "midnight" }
                const theme = profile.theme || "midnight"
                const base = basePalettes[theme] ? Object.assign({}, basePalettes[theme]) : Object.assign({}, basePalettes.midnight)
                const overrides = paletteOverrides(profile.id)
                for (const key in overrides)
                    base[key] = overrides[key]
                activeThemePalette = base
            }
            function refreshActiveWizardProgress() {
                activeWizardProgress = cloneWizardProgress(activeProfileId)
            }
            function favoriteStrategies(id) {
                const pid = id && id.length > 0 ? id : activeProfileId
                for (let i = 0; i < profiles.length; ++i) {
                    if (profiles[i].id === pid)
                        return profiles[i].favorites || []
                }
                return []
            }
            function recommendedStrategies(id) {
                return [{
                    name: "beta-mean-reversion",
                    engine: "aurora",
                    metadata: {
                        risk_profile: "balanced",
                        tags: ["trend", "hedge"]
                    }
                }]
            }
            function toggleFavoriteStrategy(id, strategy) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                const trimmed = (strategy || "").trim()
                if (trimmed.length === 0)
                    return false
                let updated = false
                const next = []
                for (let i = 0; i < profiles.length; ++i) {
                    const item = Object.assign({}, profiles[i])
                    if (item.id === targetId) {
                        const favorites = (item.favorites || []).slice(0)
                        const idx = favorites.indexOf(trimmed)
                        if (idx === -1)
                            favorites.push(trimmed)
                        else
                            favorites.splice(idx, 1)
                        item.favorites = favorites
                        updated = true
                    }
                    next.push(item)
                }
                if (!updated)
                    return false
                profiles = next
                refreshProfilesProgress()
                lastToggled = trimmed
                updateActiveProfileObject()
                refreshActiveWizardProgress()
                profilesChanged()
                if (targetId === activeProfileId)
                    activeProfileChanged()
                return true
            }
            function ensureWizardEntry(id) {
                if (wizardState[id])
                    return cloneWizardProgress(id)
                const defaults = { completedSteps: [], completed: false }
                wizardState = Object.assign({}, wizardState, { [id]: defaults })
                return cloneWizardProgress(id)
            }
            function wizardProgress(id) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                return cloneWizardProgress(targetId)
            }
            function setWizardStepCompleted(id, stepId, completed) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                const trimmed = (stepId || "").trim()
                if (trimmed.length === 0)
                    return false
                const state = ensureWizardEntry(targetId)
                const steps = state.completedSteps.slice(0)
                const idx = steps.indexOf(trimmed)
                const shouldComplete = completed === undefined ? true : !!completed
                let changed = false
                if (shouldComplete) {
                    if (idx === -1) {
                        steps.push(trimmed)
                        changed = true
                    }
                } else {
                    if (idx !== -1) {
                        steps.splice(idx, 1)
                        changed = true
                    }
                }
                if (!changed)
                    return true
                const updated = {
                    completedSteps: steps,
                    completed: shouldComplete ? state.completed : (steps.length === 0 ? false : state.completed)
                }
                wizardState = Object.assign({}, wizardState, { [targetId]: updated })
                refreshProfilesProgress()
                updateActiveProfileObject()
                if (targetId === activeProfileId) {
                    refreshActiveWizardProgress()
                    wizardProgressChanged()
                }
                profilesChanged()
                return true
            }
            function markWizardCompleted(id, completed) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                const state = ensureWizardEntry(targetId)
                const flag = completed === undefined ? true : !!completed
                if (state.completed === flag)
                    return true
                const updated = {
                    completedSteps: state.completedSteps.slice(0),
                    completed: flag
                }
                wizardState = Object.assign({}, wizardState, { [targetId]: updated })
                refreshProfilesProgress()
                updateActiveProfileObject()
                if (targetId === activeProfileId) {
                    refreshActiveWizardProgress()
                    wizardProgressChanged()
                }
                profilesChanged()
                return true
            }
            function resetWizardProgress(id) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                const state = wizardState[targetId]
                if (state && (!state.completed) && (!state.completedSteps || state.completedSteps.length === 0))
                    return true
                wizardState = Object.assign({}, wizardState, { [targetId]: { completedSteps: [], completed: false } })
                refreshProfilesProgress()
                updateActiveProfileObject()
                if (targetId === activeProfileId) {
                    refreshActiveWizardProgress()
                    wizardProgressChanged()
                }
                profilesChanged()
                return true
            }
            function duplicateProfile(id, name) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                let source = null
                for (let i = 0; i < profiles.length; ++i) {
                    if (profiles[i].id === targetId) {
                        source = profiles[i]
                        break
                    }
                }
                if (!source)
                    return ""
                let baseName = (name || "").trim()
                if (baseName.length === 0) {
                    const original = (source.displayName || "").trim()
                    baseName = original.length > 0 ? `${original} (kopia)` : `${targetId} (kopia)`
                }
                const baseIdRaw = baseName.toLowerCase().replace(/[^a-z0-9_-]+/g, "-") || "profile"
                let candidate = baseIdRaw
                let suffix = 2
                while (profiles.some(p => p.id === candidate)) {
                    candidate = `${baseIdRaw}_${suffix}`
                    suffix += 1
                }
                const duplicate = ({
                    id: candidate,
                    displayName: baseName,
                    theme: source.theme || "midnight",
                    favorites: (source.favorites || []).slice(0)
                })
                profiles = cloneProfiles().concat([duplicate])
                const sourceProgress = cloneWizardProgress(targetId)
                wizardState = Object.assign({}, wizardState, { [candidate]: sourceProgress })
                refreshProfilesProgress()
                activeProfileId = candidate
                lastDuplicatedProfileId = candidate
                lastDuplicatedSourceId = targetId
                updateActiveProfileObject()
                if (overridesStore[targetId])
                    overridesStore = Object.assign({}, overridesStore, { [candidate]: Object.assign({}, overridesStore[targetId]) })
                else if (overridesStore[candidate])
                    overridesStore = Object.assign({}, overridesStore)
                refreshActivePalette()
                refreshActiveWizardProgress()
                profilesChanged()
                wizardProgressChanged()
                activeProfileChanged()
                return candidate
            }
            function applyTheme(id, theme) {
                lastAppliedTheme = theme
                const targetId = id && id.length > 0 ? id : activeProfileId
                const next = []
                for (let i = 0; i < profiles.length; ++i) {
                    const item = Object.assign({}, profiles[i])
                    if (item.id === targetId)
                        item.theme = theme
                    next.push(item)
                }
                profiles = next
                refreshProfilesProgress()
                updateActiveProfileObject()
                refreshActivePalette()
                themePaletteChanged()
                activeProfileChanged()
                return true
            }
            function resetProfile(id) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                let changed = false
                const next = []
                for (let i = 0; i < profiles.length; ++i) {
                    const item = Object.assign({}, profiles[i])
                    if (item.id === targetId) {
                        if ((item.favorites || []).length > 0) {
                            item.favorites = []
                            changed = true
                        }
                        if (item.theme !== "midnight") {
                            item.theme = "midnight"
                            changed = true
                        }
                        if (overridesStore[targetId]) {
                            const nextStore = Object.assign({}, overridesStore)
                            delete nextStore[targetId]
                            overridesStore = nextStore
                            clearedOverrides = true
                            changed = true
                        }
                    }
                    next.push(item)
                }
                if (!changed)
                    return true
                profiles = next
                refreshProfilesProgress()
                lastResetProfileId = targetId
                updateActiveProfileObject()
                if (targetId === activeProfileId) {
                    activeThemePalette = ({
                        backgroundPrimary: "#0E1320",
                        backgroundOverlay: "#161C2A",
                        surfaceMuted: "#242B3D",
                        surfaceStrong: "#1F2536",
                        surfaceSubtle: "#2C3448",
                        textPrimary: "#F5F7FA",
                        textSecondary: "#A4ACC4",
                        accent: "#4FA3FF",
                        accentMuted: "#3577D4",
                        positive: "#3FD0A4",
                        negative: "#FF6B6B",
                        warning: "#F8C572"
                    })
                    themePaletteChanged()
                    activeProfileChanged()
                }
                resetWizardProgress(targetId)
                profilesChanged()
                return true
            }
            function themePalette(theme) {
                const base = basePalettes[theme] || basePalettes.midnight
                return Object.assign({}, base)
            }
            function setActiveProfile(id) {
                for (let i = 0; i < profiles.length; ++i) {
                    if (profiles[i].id === id) {
                        activeProfileId = id
                        updateActiveProfileObject()
                        refreshActivePalette()
                        refreshActiveWizardProgress()
                        wizardProgressChanged()
                        activeProfileChanged()
                        return true
                    }
                }
                return false
            }
            function createProfile(name) {
                const trimmed = (name || "").trim()
                if (trimmed.length === 0)
                    return ""
                const base = trimmed.toLowerCase().replace(/[^a-z0-9_-]+/g, "-") || "profile"
                let candidate = base
                let suffix = 2
                while (profiles.some(p => p.id === candidate)) {
                    candidate = base + "_" + suffix
                    suffix += 1
                }
                const profile = ({ id: candidate, displayName: trimmed, theme: "midnight", favorites: [] })
                profiles = cloneProfiles().concat([profile])
                wizardState = Object.assign({}, wizardState, { [candidate]: { completedSteps: [], completed: false } })
                refreshProfilesProgress()
                lastCreatedProfileId = candidate
                activeProfileId = candidate
                updateActiveProfileObject()
                refreshActivePalette()
                refreshActiveWizardProgress()
                profilesChanged()
                wizardProgressChanged()
                activeProfileChanged()
                return candidate
            }
            function renameProfile(id, displayName) {
                const trimmed = (displayName || "").trim()
                if (trimmed.length === 0)
                    return false
                let changed = false
                const next = []
                for (let i = 0; i < profiles.length; ++i) {
                    const item = Object.assign({}, profiles[i])
                    if (item.id === id) {
                        item.displayName = trimmed
                        changed = true
                    }
                    next.push(item)
                }
                if (!changed)
                    return false
                profiles = next
                refreshProfilesProgress()
                lastRenamedProfileId = id
                lastRenamedProfileName = trimmed
                updateActiveProfileObject()
                profilesChanged()
                if (id === activeProfileId)
                    activeProfileChanged()
                return true
            }
            function removeProfile(id) {
                if (profiles.length <= 1)
                    return false
                const next = profiles.filter(p => p.id !== id)
                if (next.length === profiles.length)
                    return false
                profiles = next
                if (wizardState[id]) {
                    const nextState = Object.assign({}, wizardState)
                    delete nextState[id]
                    wizardState = nextState
                }
                refreshProfilesProgress()
                if (activeProfileId === id) {
                    activeProfileId = profiles[0].id
                    updateActiveProfileObject()
                    refreshActivePalette()
                    refreshActiveWizardProgress()
                    wizardProgressChanged()
                    activeProfileChanged()
                }
                if (overridesStore[id]) {
                    const nextStore = Object.assign({}, overridesStore)
                    delete nextStore[id]
                    overridesStore = nextStore
                }
                profilesChanged()
                return true
            }
            function paletteOverrides(id) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                if (overridesStore[targetId])
                    return Object.assign({}, overridesStore[targetId])
                return {}
            }
            function setPaletteOverride(id, role, color) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                const trimmedRole = (role || "").trim()
                if (trimmedRole.length === 0)
                    return false
                const trimmedColor = (color || "").trim()
                const normalized = trimmedColor.length > 0
                    ? (trimmedColor.startsWith("#") ? trimmedColor.toUpperCase() : ("#" + trimmedColor.toUpperCase()))
                    : ""
                const current = overridesStore[targetId] ? Object.assign({}, overridesStore[targetId]) : {}
                if (normalized.length === 0) {
                    if (!current.hasOwnProperty(trimmedRole))
                        return true
                    delete current[trimmedRole]
                } else {
                    current[trimmedRole] = normalized
                }
                const nextStore = Object.assign({}, overridesStore)
                if (Object.keys(current).length === 0)
                    delete nextStore[targetId]
                else
                    nextStore[targetId] = current
                overridesStore = nextStore
                lastOverrideRole = trimmedRole
                lastOverrideColor = normalized
                if (targetId === activeProfileId) {
                    refreshActivePalette()
                    themePaletteChanged()
                }
                profilesChanged()
                return true
            }
            function clearPaletteOverrides(id) {
                const targetId = id && id.length > 0 ? id : activeProfileId
                if (!overridesStore[targetId])
                    return true
                const nextStore = Object.assign({}, overridesStore)
                delete nextStore[targetId]
                overridesStore = nextStore
                clearedOverrides = true
                if (targetId === activeProfileId) {
                    refreshActivePalette()
                    themePaletteChanged()
                }
                profilesChanged()
                return true
            }
            Component.onCompleted: {
                updateActiveProfileObject()
                refreshActivePalette()
                refreshProfilesProgress()
                refreshActiveWizardProgress()
            }
        }`
        return Qt.createQmlObject(source, testCase, "UserProfilesStub")
    }

    function createWizardStub() {
        const src = `import QtQuick 2.15; QtObject {
            property var steps: [
                { title: "Licencja", description: "Aktywacja", category: "Licencja" },
                { title: "Połączenia", description: "API", category: "Połączenia" },
                { title: "Automatyzacja", description: "Strategie", category: "Strategie" },
                { title: "Marketplace", description: "Presety", category: "Marketplace" },
                { title: "Monitoring", description: "Alerty", category: "Monitoring" }
            ]
            property string startedWith: ""
            function start(id) { startedWith = id; return true }
        }`
        return Qt.createQmlObject(src, testCase, "WizardStub")
    }

    function test_strategyOverviewListsFavorites() {
        const profiles = createUserProfilesStub()
        const component = Qt.createComponent("../../qml/dashboard/StrategyOverviewPanel.qml")
        verify(component.status === Component.Ready, component.errorString())
        const panel = component.createObject(testCase, { userProfiles: profiles })
        verify(panel !== null)
        const favoritesView = panel.findChild("favoritesView")
        verify(favoritesView !== null)
        compare(favoritesView.count, 1)
        panel.destroy()
        profiles.destroy()
    }

    function test_themePersonalizationAppliesTheme() {
        const profiles = createUserProfilesStub()
        const component = Qt.createComponent("../../qml/settings/ThemePersonalization.qml")
        verify(component.status === Component.Ready, component.errorString())
        const personalization = component.createObject(testCase, { userProfiles: profiles })
        verify(personalization !== null)
        const repeater = personalization.findChild("themeRepeater")
        verify(repeater !== null)
        const auroraButton = repeater.itemAt(1).findChild("themeButton_aurora")
        verify(auroraButton !== null)
        auroraButton.clicked()
        compare(profiles.lastAppliedTheme, "aurora")
        personalization.destroy()
        profiles.destroy()
    }

    function test_themePersonalizationAllowsOverrides() {
        const profiles = createUserProfilesStub()
        const component = Qt.createComponent("../../qml/settings/ThemePersonalization.qml")
        verify(component.status === Component.Ready, component.errorString())
        const personalization = component.createObject(testCase, { userProfiles: profiles })
        verify(personalization !== null)

        personalization.applyOverride("accent", "#ABCDEF")
        compare(profiles.lastOverrideRole, "accent")
        compare(profiles.paletteOverrides("default").accent, "#ABCDEF")
        compare(profiles.activeThemePalette.accent, "#ABCDEF")

        const resetButton = personalization.findChild("resetOverrideButton_accent")
        verify(resetButton !== null)
        resetButton.clicked()
        compare(profiles.paletteOverrides("default").hasOwnProperty("accent"), false)

        personalization.applyOverride("accent", "123456")
        const clearButton = personalization.findChild("clearOverridesButton")
        verify(clearButton !== null)
        verify(clearButton.enabled)
        clearButton.clicked()
        compare(profiles.clearedOverrides, true)
        compare(Object.keys(profiles.paletteOverrides("default")).length, 0)

        personalization.destroy()
        profiles.destroy()
    }

    function test_profileManagementWizardProgress() {
        const profiles = createUserProfilesStub()
        const wizard = createWizardStub()
        const component = Qt.createComponent("../../qml/components/UserProfileManagementPanel.qml")
        verify(component.status === Component.Ready, component.errorString())
        const panel = component.createObject(testCase, { userProfiles: profiles, wizardController: wizard })
        verify(panel !== null)

        const listView = panel.findChild("profilesListView")
        verify(listView !== null)
        wait(0)
        let delegate = listView.itemAt(0)
        verify(delegate !== null)

        const progressBar = delegate.findChild("wizardProgressBar_default")
        verify(progressBar !== null)
        compare(progressBar.value, 1)

        profiles.markWizardCompleted("default", true)
        wait(0)

        delegate = listView.itemAt(0)
        verify(delegate !== null)
        const statusLabel = delegate.findChild("wizardStatusLabel_default")
        verify(statusLabel !== null)
        compare(statusLabel.text, "Ukończony")

        panel.destroy()
        wizard.destroy()
        profiles.destroy()
    }

    function test_configurationWizardGallerySummary() {
        const profiles = createUserProfilesStub()
        const wizard = createWizardStub()
        const component = Qt.createComponent("../../qml/components/ConfigurationWizardGallery.qml")
        verify(component.status === Component.Ready, component.errorString())
        const gallery = component.createObject(testCase, { userProfiles: profiles, wizardController: wizard })
        verify(gallery !== null)

        const summaryProgress = gallery.findChild("wizardSummaryProgress")
        verify(summaryProgress !== null)
        compare(summaryProgress.value, 1)

        profiles.resetWizardProgress("default")
        wait(0)
        compare(summaryProgress.value, 0)

        gallery.destroy()
        wizard.destroy()
        profiles.destroy()
    }

    function test_strategyExperienceCombinesSections() {
        const profiles = createUserProfilesStub()
        const app = Qt.createQmlObject('import QtQuick 2.15; QtObject {}', testCase, "AppStub")
        app.userProfiles = profiles
        const wizard = createWizardStub()
        const component = Qt.createComponent("../../qml/views/StrategyExperience.qml")
        verify(component.status === Component.Ready, component.errorString())
        const view = component.createObject(testCase, {
            appController: app,
            configurationWizard: wizard
        })
        verify(view !== null)
        const profilePanel = view.findChild("profileManagementPanel")
        verify(profilePanel !== null)
        const overview = view.findChild("strategyOverviewPanel")
        verify(overview !== null)
        const gallery = view.findChild("configurationWizardGallery")
        verify(gallery !== null)
        const theme = view.findChild("themePersonalization")
        verify(theme !== null)
        view.destroy()
        wizard.destroy()
        profiles.destroy()
        app.destroy()
    }

    function test_profileManagementPanelManagesProfiles() {
        const profiles = createUserProfilesStub()
        const component = Qt.createComponent("../../qml/components/UserProfileManagementPanel.qml")
        verify(component.status === Component.Ready, component.errorString())
        const panel = component.createObject(testCase, { userProfiles: profiles })
        verify(panel !== null)

        const nameField = panel.findChild("newProfileNameField")
        const createButton = panel.findChild("createProfileButton")
        verify(nameField !== null)
        verify(createButton !== null)
        nameField.text = "Scalper"
        createButton.clicked()
        verify(profiles.lastCreatedProfileId.length > 0)
        compare(profiles.profiles.length, 2)
        compare(profiles.activeProfileId, profiles.lastCreatedProfileId)

        const listView = panel.findChild("profilesListView")
        verify(listView !== null)
        listView.currentIndex = 0
        const defaultDelegate = listView.currentItem
        verify(defaultDelegate !== null)
        const defaultNameField = defaultDelegate.findChild("profileNameField_default")
        const saveButton = defaultDelegate.findChild("saveProfileButton_default")
        verify(defaultNameField !== null)
        verify(saveButton !== null)
        defaultNameField.text = "Nowy domyślny"
        saveButton.clicked()
        compare(profiles.lastRenamedProfileId, "default")
        compare(profiles.lastRenamedProfileName, "Nowy domyślny")

        const duplicateButton = defaultDelegate.findChild("duplicateProfileButton_default")
        verify(duplicateButton !== null)
        duplicateButton.clicked()
        verify(profiles.lastDuplicatedProfileId.length > 0)
        compare(profiles.lastDuplicatedSourceId, "default")
        compare(profiles.activeProfileId, profiles.lastDuplicatedProfileId)

        listView.currentIndex = listView.count - 1
        const duplicatedDelegate = listView.currentItem
        verify(duplicatedDelegate !== null)
        const resetButton = duplicatedDelegate.findChild("resetProfileButton_" + profiles.lastDuplicatedProfileId)
        verify(resetButton !== null)
        profiles.toggleFavoriteStrategy(profiles.lastDuplicatedProfileId, "beta-mean-reversion")
        resetButton.clicked()
        compare(profiles.lastResetProfileId, profiles.lastDuplicatedProfileId)

        panel.destroy()
        profiles.destroy()
    }
}
