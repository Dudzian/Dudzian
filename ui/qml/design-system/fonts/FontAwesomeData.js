function solidDataUrl(ciSnapshot) {
    // CI: brak bundlowania fontów w snapshotach, aby uniknąć zależności od assetów runtime.
    if (ciSnapshot === true)
        return ""
    if (typeof SOLID_FONT_DATA_URL !== "undefined")
        return SOLID_FONT_DATA_URL
    return ""
}
