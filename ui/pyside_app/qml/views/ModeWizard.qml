import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects
import "../components" as Components

Item {
    id: root
    property var designSystem: null
    property var modeWizardController: null
    property var strategyManagementController: null
    property var layoutController: null
    property bool compact: false
    signal launchWizardRequested()

    property var modesModel: modeWizardController ? modeWizardController.modes : []
    property string recommendedModeId: modeWizardController ? modeWizardController.recommendedModeId : ""
    property var recommendationSummary: modeWizardController ? modeWizardController.recommendedModeSummary : ({})
    property var aiProfiles: modeWizardController ? modeWizardController.aiProfiles() : ({})
    property string marketplacePortfolioId: ""
    property string internalSelected: ""
    property string selectedModeId: internalSelected.length > 0 ? internalSelected : recommendedModeId
    property int stepIndex: 0
    property var answers: ({})
    property var presetCandidate: bestPresetForMode()

    function dsColor(token, fallback) {
        if (designSystem && designSystem.color)
            return designSystem.color(token)
        return fallback
    }

    function refreshSelection() {
        root.presetCandidate = root.bestPresetForMode()
        var initial = internalSelected
        if (!initial && recommendedModeId && recommendedModeId.length > 0)
            initial = recommendedModeId
        if (!initial && modesModel && modesModel.length > 0)
            initial = modesModel[0].id
        if (initial)
            selectMode(initial)
    }

    function modeById(modeId) {
        if (!modesModel)
            return null
        for (var i = 0; i < modesModel.length; ++i) {
            if (modesModel[i].id === modeId)
                return modesModel[i]
        }
        return null
    }

    function ensureAnswers(modeId) {
        if (!modeId)
            return {}
        if (!answers[modeId] && modeWizardController && modeWizardController.savedAnswers)
            answers[modeId] = modeWizardController.savedAnswers(modeId) || {}
        return answers[modeId] || {}
    }

    function answerValue(modeId, inputId) {
        var store = ensureAnswers(modeId)
        return store[inputId]
    }

    function setAnswerValue(modeId, inputId, value) {
        if (!modeId || !inputId)
            return
        var store = ensureAnswers(modeId)
        store[inputId] = value
        answers[modeId] = store
    }

    function selectMode(modeId) {
        if (!modeId)
            return
        internalSelected = modeId
        root.presetCandidate = root.bestPresetForMode()
        if (modeWizardController && modeWizardController.setActiveMode)
            modeWizardController.setActiveMode(modeId)
        stepIndex = 0
        ensureAnswers(modeId)
    }

    function bestPresetForMode() {
        if (!strategyManagementController)
            return null
        var presets = strategyManagementController.presets || []
        if (!presets.length)
            return null
        if (root.selectedModeId && root.selectedModeId.length > 0) {
            for (var i = 0; i < presets.length; ++i) {
                var preset = presets[i]
                if (preset.tags && preset.tags.indexOf(root.selectedModeId) !== -1)
                    return preset
            }
        }
        return presets[0]
    }

    Component.onCompleted: refreshSelection()

    Connections {
        target: modeWizardController || null
        enabled: modeWizardController !== null
        function onRecommendationChanged() {
            if (!root.internalSelected)
                root.refreshSelection()
        }
        function onModesChanged() {
            root.refreshSelection()
        }
        function onActiveModeChanged() {
            if (!root.compact)
                root.internalSelected = modeWizardController.activeModeId
        }
        function onResultsChanged() {
            if (root.selectedModeId)
                root.answers[root.selectedModeId] = modeWizardController.savedAnswers(root.selectedModeId) || {}
        }
    }

    Connections {
        target: strategyManagementController || null
        enabled: strategyManagementController !== null
        function onPresetsChanged() {
            root.presetCandidate = root.bestPresetForMode()
        }
    }

    onSelectedModeIdChanged: root.presetCandidate = root.bestPresetForMode()

    ColumnLayout {
        anchors.fill: parent
        spacing: 16

        Item {
            Layout.fillWidth: true
            implicitHeight: summaryContent.implicitHeight + 32

            Rectangle {
                id: summaryBackground
                anchors.fill: parent
                radius: 24
                gradient: Gradient {
                    GradientStop { position: 0; color: designSystem ? designSystem.color("gradientHeroStart") : "#223" }
                    GradientStop { position: 1; color: designSystem ? designSystem.color("gradientHeroEnd") : "#335" }
                }
                opacity: 0.75
            }

            MultiEffect {
                anchors.fill: summaryBackground
                source: summaryBackground
                blurEnabled: true
                blur: 1.0
                blurMax: 30
                saturation: 0.9
                brightness: 0.05
            }

            ColumnLayout {
                id: summaryContent
                anchors.fill: parent
                anchors.margins: 24
                spacing: 8

                Label {
                    text: recommendationSummary && recommendationSummary.title
                          ? qsTr("Rekomendowany tryb: %1").arg(recommendationSummary.title)
                          : qsTr("Wybierz tryb pracy")
                    font.pixelSize: 20
                    font.bold: true
                    color: designSystem ? designSystem.color("textPrimary") : "#fff"
                }

                Label {
                    text: recommendationSummary && recommendationSummary.recommendations
                          && recommendationSummary.recommendations.summary
                          ? recommendationSummary.recommendations.summary
                          : qsTr("Aktywuj kreator, aby AI zaproponowało profil.")
                    color: designSystem ? designSystem.color("textSecondary") : "#d0d4e0"
                    wrapMode: Text.WordWrap
                }

                RowLayout {
                    visible: recommendationSummary && recommendationSummary.badge ? true : false
                    spacing: 8
                    Rectangle {
                        radius: 12
                        color: designSystem ? designSystem.color("surface") : "#fff"
                        opacity: 0.85
                        border.color: designSystem ? designSystem.color("border") : "#ccc"
                        border.width: 1
                        implicitHeight: 26
                        implicitWidth: badgeLabel.implicitWidth + 16
                        Label {
                            id: badgeLabel
                            anchors.centerIn: parent
                            text: recommendationSummary && recommendationSummary.badge ? recommendationSummary.badge : ""
                            color: designSystem ? designSystem.color("textPrimary") : "#111"
                            font.bold: true
                        }
                    }
                    Label {
                        text: recommendationSummary.ai_profile_hint && recommendationSummary.ai_profile_hint.length > 0
                              ? qsTr("Profil AI: %1").arg(recommendationSummary.ai_profile_hint)
                              : ""
                        visible: text.length > 0
                        color: designSystem ? designSystem.color("textSecondary") : "#d0d4e0"
                    }
                }

                Label {
                    text: aiProfiles && Object.keys(aiProfiles).length > 0
                          ? qsTr("Dostępne profile cloud: %1").arg(Object.keys(aiProfiles).join(", "))
                          : qsTr("Brak danych o profilach cloud")
                    color: designSystem ? designSystem.color("textSecondary") : "#d0d4e0"
                    wrapMode: Text.WordWrap
                }

                RowLayout {
                    Layout.topMargin: 4
                    spacing: 12
                    Components.IconButton {
                        designSystem: designSystem
                        text: qsTr("Otwórz kreator")
                        iconName: "mode_wizard"
                        backgroundColor: designSystem ? designSystem.color("accent") : "#00aaff"
                        foregroundColor: designSystem ? designSystem.color("surface") : "#111"
                        onClicked: {
                            if (root.compact)
                                root.launchWizardRequested()
                            else
                                root.stepIndex = 0
                        }
                    }
                    Components.IconButton {
                        designSystem: designSystem
                        text: qsTr("Zastosuj rekomendację")
                        iconName: recommendationSummary && recommendationSummary.icon ? recommendationSummary.icon : "package"
                        subtle: true
                        onClicked: {
                            if (root.recommendedModeId && root.recommendedModeId.length > 0)
                                root.selectMode(root.recommendedModeId)
                        }
                    }
                }

                Rectangle {
                    visible: strategyManagementController !== null && root.presetCandidate !== null
                    Layout.fillWidth: true
                    radius: 18
                    color: designSystem ? designSystem.color("surface") : "#1c2233"
                    opacity: 0.95
                    border.color: designSystem ? designSystem.color("border") : "#2f354a"
                    border.width: 1
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 16
                        spacing: 6
                        Label {
                            text: root.presetCandidate ? qsTr("Marketplace: %1").arg(root.presetCandidate.name || "") : ""
                            font.bold: true
                            color: designSystem ? designSystem.color("textPrimary") : "#fff"
                            visible: root.presetCandidate !== null
                        }
                        Label {
                            text: root.presetCandidate && root.presetCandidate.summary ? root.presetCandidate.summary : qsTr("Brak opisu presetu")
                            color: designSystem ? designSystem.color("textSecondary") : "#c1c8df"
                            wrapMode: Text.WordWrap
                            visible: root.presetCandidate !== null
                        }
                        Item {
                            Layout.fillWidth: true
                            visible: root.presetCandidate && root.presetCandidate.userPreferences && root.presetCandidate.userPreferences.length > 0
                            implicitHeight: personaLayout.implicitHeight + 12
                            Rectangle {
                                id: personaBackdrop
                                anchors.fill: parent
                                radius: 16
                                color: designSystem ? designSystem.color("surfaceMuted") : "#2a3145"
                                opacity: 0.9
                                border.color: designSystem ? designSystem.color("border") : "#3d4560"
                                border.width: 1
                            }
                            MultiEffect {
                                anchors.fill: personaBackdrop
                                source: personaBackdrop
                                blurEnabled: true
                                blur: 1.0
                                blurMax: 16
                                saturation: 0.95
                                brightness: 0.03
                            }
                            ColumnLayout {
                                id: personaLayout
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 6
                                Repeater {
                                    model: root.presetCandidate ? root.presetCandidate.userPreferences : []
                                    delegate: ColumnLayout {
                                        spacing: 2
                                        Label {
                                            text: qsTr("Persona: %1").arg(modelData.persona || qsTr("profil"))
                                            font.bold: true
                                            color: designSystem ? designSystem.color("textPrimary") : "#fff"
                                        }
                                        RowLayout {
                                            spacing: 8
                                            Text {
                                                text: designSystem ? designSystem.iconGlyph("shield") : ""
                                                visible: text.length > 0
                                                font.family: designSystem ? designSystem.fontAwesomeFamily() : ""
                                                font.pixelSize: 14
                                                color: designSystem ? designSystem.color("accent") : "#00aaff"
                                            }
                                            Label {
                                                text: qsTr("Ryzyko: %1").arg(modelData.risk_target || qsTr("brak"))
                                                color: designSystem ? designSystem.color("textSecondary") : "#d0d4e0"
                                            }
                                            Text {
                                                text: designSystem ? designSystem.iconGlyph("package") : ""
                                                visible: text.length > 0
                                                font.family: designSystem ? designSystem.fontAwesomeFamily() : ""
                                                font.pixelSize: 14
                                                color: designSystem ? designSystem.color("accent") : "#00aaff"
                                            }
                                            Label {
                                                text: modelData.recommended_budget
                                                      ? qsTr("Budżet: %1 USD").arg(Number(modelData.recommended_budget).toLocaleString(Qt.locale(), 'f', 0))
                                                      : qsTr("Budżet: brak")
                                                color: designSystem ? designSystem.color("textSecondary") : "#d0d4e0"
                                            }
                                        }
                                        RowLayout {
                                            spacing: 8
                                            Text {
                                                property string fallbackGlyph: "\uf017"
                                                text: designSystem && designSystem.iconGlyph("clock") && designSystem.iconGlyph("clock").length > 0
                                                      ? designSystem.iconGlyph("clock")
                                                      : fallbackGlyph
                                                visible: text.length > 0
                                                font.family: designSystem ? designSystem.fontAwesomeFamily() : "Font Awesome 6 Free"
                                                font.pixelSize: 14
                                                color: designSystem ? designSystem.color("accent") : "#00aaff"
                                            }
                                            Label {
                                                text: qsTr("Horyzont: %1").arg(modelData.holding_period || qsTr("brak"))
                                                color: designSystem ? designSystem.color("textSecondary") : "#d0d4e0"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8
                            TextField {
                                id: wizardPortfolioInput
                                Layout.fillWidth: true
                                placeholderText: qsTr("ID portfela dla presetu")
                                text: root.marketplacePortfolioId
                                onTextChanged: root.marketplacePortfolioId = text
                            }
                            Components.IconButton {
                                designSystem: designSystem
                                text: qsTr("Zastosuj preset")
                                iconName: "strategy_manager"
                                enabled: strategyManagementController !== null && root.presetCandidate !== null && wizardPortfolioInput.text.length > 0
                                onClicked: {
                                    if (strategyManagementController && root.presetCandidate)
                                        strategyManagementController.activateAndAssign(root.presetCandidate.presetId, wizardPortfolioInput.text)
                                }
                            }
                            Components.IconButton {
                                designSystem: designSystem
                                text: qsTr("Otwórz manager")
                                iconName: "package"
                                subtle: true
                                onClicked: {
                                    if (layoutController)
                                        layoutController.setPanelVisibility("strategyManagerPanel", true)
                                }
                            }
                        }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 12
            visible: !compact

            Flickable {
                Layout.fillWidth: true
                Layout.preferredHeight: 150
                contentWidth: cardsRow.implicitWidth
                flickableDirection: Flickable.HorizontalFlick
                clip: true

                RowLayout {
                    id: cardsRow
                    spacing: 12
                    Repeater {
                        model: modesModel
                        delegate: Rectangle {
                            width: 220
                            height: 120
                            radius: 18
                            color: modelData.id === selectedModeId
                                    ? dsColor("surface", "#1c2233")
                                    : dsColor("surfaceMuted", "#2a3145")
                            border.color: modelData.id === selectedModeId
                                          ? dsColor("accent", "#00aaff")
                                          : dsColor("border", "#2f354a")
                            border.width: 1
                            opacity: 0.92

                            ColumnLayout {
                                anchors.fill: parent
                                anchors.margins: 12
                                spacing: 4
                                RowLayout {
                                    spacing: 6
                                    Components.IconGlyph {
                                        source: designSystem.iconSource(modelData.icon || "mode_wizard")
                                        width: 18
                                        height: 18
                                        fillMode: Image.PreserveAspectFit
                                        color: dsColor("textPrimary", "#ffffff")
                                    }
                                    Label {
                                        text: modelData.title
                                        color: dsColor("textPrimary", "#ffffff")
                                        font.bold: true
                                        Layout.fillWidth: true
                                    }
                                }
                                Label {
                                    text: modelData.description
                                    color: dsColor("textSecondary", "#d0d4e0")
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                                Components.IconButton {
                                    designSystem: designSystem
                                    text: modelData.id === selectedModeId ? qsTr("Aktywny") : qsTr("Wybierz")
                                    iconName: modelData.id === selectedModeId ? "shield" : "mode_wizard"
                                    subtle: modelData.id !== selectedModeId
                                    onClicked: root.selectMode(modelData.id)
                                }
                            }
                        }
                    }
                }
            }

            SpinBox {
                id: wizardStepper
                visible: wizardPanel.currentMode && wizardPanel.currentMode.steps && wizardPanel.currentMode.steps.length > 0
                from: 0
                to: wizardPanel.currentMode && wizardPanel.currentMode.steps ? Math.max(0, wizardPanel.currentMode.steps.length - 1) : 0
                value: stepIndex
                stepSize: 1
                editable: false
                onValueChanged: stepIndex = value
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: 16
                color: dsColor("surface", "#1c2233")
                border.color: dsColor("border", "#2f354a")
                border.width: 1
                opacity: 0.95

                ColumnLayout {
                    id: wizardPanel
                    anchors.fill: parent
                    anchors.margins: 16
                    spacing: 8
                    property var currentMode: root.modeById(root.selectedModeId)
                    property var currentStep: currentMode && currentMode.steps && currentMode.steps.length > 0
                            ? currentMode.steps[Math.min(root.stepIndex, currentMode.steps.length - 1)]
                            : null

                    Label {
                        text: wizardPanel.currentStep ? wizardPanel.currentStep.title : qsTr("Brak kroków do wyświetlenia")
                        font.pixelSize: 18
                        font.bold: true
                        color: dsColor("textPrimary", "#ffffff")
                    }

                    Label {
                        text: wizardPanel.currentStep ? wizardPanel.currentStep.description : qsTr("Wybierz tryb, aby zobaczyć kroki kreatora.")
                        color: dsColor("textSecondary", "#d0d4e0")
                        wrapMode: Text.WordWrap
                    }

                    Repeater {
                        model: wizardPanel.currentStep && wizardPanel.currentStep.inputs ? wizardPanel.currentStep.inputs : []
                        delegate: ColumnLayout {
                            property var inputSpec: modelData
                            Layout.fillWidth: true
                            spacing: 6
                            Label {
                                text: inputSpec.label
                                color: dsColor("textPrimary", "#ffffff")
                                font.bold: true
                            }
                            Flow {
                                width: parent.width
                                Layout.fillWidth: true
                                spacing: 8
                                Repeater {
                                    model: inputSpec.options
                                    delegate: Components.IconButton {
                                        designSystem: designSystem
                                        property string optionId: modelData.id
                                        property bool selected: answerValue(root.selectedModeId, inputSpec.id) === optionId
                                        text: modelData.label
                                        iconName: selected ? "shield" : "package"
                                        subtle: !selected
                                        onClicked: setAnswerValue(root.selectedModeId, inputSpec.id, optionId)
                                        ToolTip.visible: hovered && modelData.helper && modelData.helper.length > 0
                                        ToolTip.text: modelData.helper || ""
                                    }
                                }
                            }
                        }
                    }

                    RowLayout {
                        Layout.alignment: Qt.AlignRight
                        spacing: 8
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Wstecz")
                            iconName: "refresh"
                            subtle: true
                            enabled: stepIndex > 0
                            onClicked: stepIndex = Math.max(0, stepIndex - 1)
                        }
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Dalej")
                            iconName: "refresh"
                            subtle: true
                            enabled: wizardPanel.currentMode && wizardPanel.currentMode.steps && stepIndex < wizardPanel.currentMode.steps.length - 1
                            onClicked: stepIndex = Math.min(wizardPanel.currentMode.steps.length - 1, stepIndex + 1)
                        }
                        Components.IconButton {
                            designSystem: designSystem
                            text: qsTr("Zapisz tryb")
                            iconName: "mode_wizard"
                            backgroundColor: dsColor("accent", "#00aaff")
                            foregroundColor: dsColor("surface", "#111111")
                            onClicked: {
                                if (modeWizardController && modeWizardController.saveResult)
                                    modeWizardController.saveResult(root.selectedModeId, ensureAnswers(root.selectedModeId))
                            }
                        }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            visible: compact
            spacing: 8
            Label {
                text: qsTr("Otwórz kreator, aby spersonalizować tryb pracy.")
                color: dsColor("textSecondary", "#d0d4e0")
                wrapMode: Text.WordWrap
            }
            Components.IconButton {
                designSystem: designSystem
                text: qsTr("Konfiguruj tryby pracy")
                iconName: "mode_wizard"
                backgroundColor: dsColor("accent", "#00aaff")
                foregroundColor: dsColor("surface", "#111111")
                onClicked: root.launchWizardRequested()
            }
        }
    }

    function currentMode() {
        return modeById(selectedModeId)
    }
}
