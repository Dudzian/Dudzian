import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: overlay
    property bool enabled: true
    property int displayDuration: 6000
    property int toastLimit: 4
    property real toastWidth: 360
    anchors.fill: parent
    visible: enabled && toastModel.count > 0
    z: 1000

    signal toastDismissed(string id)

    ListModel {
        id: toastModel
    }

    function clear() {
        while (toastModel.count > 0)
            toastModel.remove(0)
    }

    function showToast(id, severity, title, description) {
        if (!enabled)
            return
        var normalizedId = id || ("toast_" + Date.now())
        var payload = {
            id: normalizedId,
            severity: severity,
            title: title || qsTr("Alert"),
            description: description || "",
            createdAt: Date.now()
        }
        var existingIndex = -1
        for (var i = 0; i < toastModel.count; ++i) {
            if (toastModel.get(i).id === normalizedId) {
                existingIndex = i
                break
            }
        }
        if (existingIndex >= 0)
            toastModel.set(existingIndex, payload)
        else
            toastModel.append(payload)

        while (toastModel.count > toastLimit)
            toastModel.remove(0)
    }

    function dismissByIndex(idx) {
        if (idx < 0 || idx >= toastModel.count)
            return
        var id = toastModel.get(idx).id
        toastModel.remove(idx)
        toastDismissed(id)
    }

    ColumnLayout {
        anchors.right: parent.right
        anchors.rightMargin: 24
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 24
        spacing: 12
        Repeater {
            model: toastModel
            delegate: Frame {
                id: toastFrame
                Layout.preferredWidth: overlay.toastWidth
                Layout.fillWidth: false
                padding: 14
                opacity: 0.0
                background: Rectangle {
                    radius: 10
                    color: model.severity === 2 ? Qt.rgba(0.58, 0.16, 0.2, 0.82)
                                                : (model.severity === 1 ? Qt.rgba(0.96, 0.67, 0.17, 0.78)
                                                                        : Qt.rgba(0.16, 0.22, 0.32, 0.82))
                }

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    Label {
                        text: model.title || qsTr("Alert")
                        font.pixelSize: 16
                        font.bold: true
                        color: Qt.rgba(1, 1, 1, 0.95)
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        text: model.description || ""
                        color: Qt.rgba(1, 1, 1, 0.85)
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        Label {
                            Layout.fillWidth: true
                            font.pixelSize: 11
                            color: Qt.rgba(1, 1, 1, 0.65)
                            text: Qt.formatTime(new Date(model.createdAt), Qt.DefaultLocaleShortDate)
                        }
                        Button {
                            id: dismissButton
                            text: qsTr("Zamknij")
                            focusPolicy: Qt.NoFocus
                            onClicked: overlay.dismissByIndex(index)
                        }
                    }
                }

                SequentialAnimation {
                    id: fadeInOut
                    running: overlay.enabled
                    loops: 1
                    PropertyAnimation { target: toastFrame; property: "opacity"; from: 0.0; to: 1.0; duration: 180 }
                    PauseAnimation { duration: overlay.displayDuration }
                    PropertyAnimation { target: toastFrame; property: "opacity"; from: 1.0; to: 0.0; duration: 220 }
                    onFinished: overlay.dismissByIndex(index)
                }

                Component.onCompleted: fadeInOut.restart()
                onModelChanged: fadeInOut.restart()
            }
        }
    }
}
