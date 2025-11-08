const blockTemplates = {
    data_feed: [
        { key: "symbol", type: "string", defaultValue: "BTCUSDT" },
        { key: "interval", type: "enum", defaultValue: "1h", options: ["1m", "5m", "1h", "4h", "1d"] },
        { key: "source", type: "enum", defaultValue: "sandbox", options: ["sandbox", "live"] }
    ],
    filter: [
        { key: "risk_limit", type: "number", defaultValue: 2, min: 0, max: 100, decimals: 2 },
        { key: "lookback", type: "number", defaultValue: 30, min: 1, max: 365, step: 1 },
        { key: "enabled", type: "boolean", defaultValue: true }
    ],
    signal: [
        { key: "threshold", type: "number", defaultValue: 0.5, min: 0, max: 1, decimals: 2 },
        { key: "lookback", type: "number", defaultValue: 14, min: 1, max: 365, step: 1 },
        { key: "enabled", type: "boolean", defaultValue: true }
    ],
    allocator: [
        { key: "allocation", type: "number", defaultValue: 10, min: 0, max: 100, decimals: 2 },
        { key: "max_positions", type: "number", defaultValue: 5, min: 1, max: 100, step: 1 }
    ],
    execution: [
        { key: "venue", type: "string", defaultValue: "sandbox" },
        { key: "account", type: "string", defaultValue: "paper" },
        { key: "slippage", type: "number", defaultValue: 0.1, min: 0, max: 10, decimals: 2 },
        { key: "quantity", type: "number", defaultValue: 1, min: 0, max: 1000, decimals: 4 }
    ]
};

function clone(value) {
    if (value === null || typeof value !== "object")
        return value;
    if (Array.isArray(value))
        return value.map(clone);
    const copy = {};
    for (const key in value) {
        if (!Object.prototype.hasOwnProperty.call(value, key))
            continue;
        copy[key] = clone(value[key]);
    }
    return copy;
}

export function parameterConfig(type) {
    return blockTemplates[type] ? blockTemplates[type].map(clone) : [];
}

export function defaultParams(type) {
    const config = blockTemplates[type];
    if (!config)
        return {};
    const result = {};
    config.forEach((entry) => {
        result[entry.key] = clone(entry.defaultValue);
    });
    return result;
}

function normalizeValue(entry, raw) {
    switch (entry.type) {
    case "number": {
        if (typeof raw === "number")
            return raw;
        if (typeof raw === "string" && raw.trim().length > 0) {
            const parsed = Number(raw);
            if (!Number.isNaN(parsed))
                return parsed;
        }
        return entry.defaultValue !== undefined ? entry.defaultValue : 0;
    }
    case "boolean":
        if (typeof raw === "boolean")
            return raw;
        if (typeof raw === "string")
            return raw === "true" || raw === "1";
        if (typeof raw === "number")
            return raw !== 0;
        return entry.defaultValue === undefined ? false : entry.defaultValue;
    case "enum": {
        const options = entry.options || [];
        const value = typeof raw === "string" || typeof raw === "number" ? String(raw) : entry.defaultValue;
        if (options.indexOf(value) !== -1)
            return value;
        return entry.defaultValue !== undefined ? entry.defaultValue : (options.length > 0 ? options[0] : "");
    }
    case "string":
    default:
        if (raw === null || raw === undefined)
            return entry.defaultValue !== undefined ? entry.defaultValue : "";
        return String(raw);
    }
}

export function mergeParams(type, params) {
    const config = blockTemplates[type];
    if (!config) {
        return params ? clone(params) : {};
    }
    const normalized = {};
    const input = params || {};
    config.forEach((entry) => {
        const key = entry.key;
        const raw = Object.prototype.hasOwnProperty.call(input, key) ? input[key] : entry.defaultValue;
        normalized[key] = normalizeValue(entry, raw);
    });
    for (const key in input) {
        if (!Object.prototype.hasOwnProperty.call(input, key))
            continue;
        if (!Object.prototype.hasOwnProperty.call(normalized, key))
            normalized[key] = clone(input[key]);
    }
    return normalized;
}

export function summaryOrder(type) {
    const config = blockTemplates[type];
    if (!config)
        return [];
    return config.map((entry) => entry.key);
}
