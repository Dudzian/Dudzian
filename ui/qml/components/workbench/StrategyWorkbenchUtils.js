function formatNumber(value, digits) {
    if (value === null || value === undefined)
        return "–";
    const precision = digits !== undefined ? digits : 2;
    return Number(value).toLocaleString(Qt.locale(), "f", precision);
}

function formatPercent(value, digits) {
    if (value === null || value === undefined)
        return "–";
    const precision = digits !== undefined ? digits : 2;
    return Number(value * 100).toLocaleString(Qt.locale(), "f", precision) + "%";
}

function formatSignedPercent(value, digits) {
    if (value === null || value === undefined)
        return "–";
    const precision = digits !== undefined ? digits : 2;
    const percentValue = Number(value) * 100;
    if (!isFinite(percentValue))
        return "–";
    const magnitude = Math.abs(percentValue).toLocaleString(Qt.locale(), "f", precision);
    if (percentValue > 0)
        return "+" + magnitude + "%";
    if (percentValue < 0)
        return "-" + magnitude + "%";
    return "0%";
}

function formatBoolean(value) {
    return value ? qsTr("Tak") : qsTr("Nie");
}

function formatText(value, placeholder) {
    if (value === null || value === undefined || value === "")
        return placeholder !== undefined ? placeholder : "–";
    return value;
}

function formatDuration(value) {
    if (value === null || value === undefined)
        return "–";
    const seconds = Number(value);
    if (!isFinite(seconds))
        return "–";
    if (seconds < 0)
        return qsTr("n/d");
    const rounded = Math.round(seconds);
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const secs = rounded % 60;
    if (hours > 0)
        return qsTr("%1 h %2 min").arg(hours).arg(minutes);
    if (minutes > 0)
        return secs > 0 ? qsTr("%1 min %2 s").arg(minutes).arg(secs)
                         : qsTr("%1 min").arg(minutes);
    return qsTr("%1 s").arg(secs);
}

function formatTimestamp(value) {
    if (!value)
        return "–";
    const date = new Date(value);
    if (isNaN(date.getTime()))
        return value;
    const formatted = Qt.formatDateTime(date, Qt.DefaultLocaleShortDate);
    if (formatted && formatted.length > 0)
        return formatted;
    return date.toISOString();
}

function formatCurrencyPair(baseCurrency, quoteCurrency) {
    const base = baseCurrency || "";
    const quote = quoteCurrency || "";
    if (!base && !quote)
        return "–";
    if (!base)
        return quote;
    if (!quote)
        return base;
    return base + "/" + quote;
}
