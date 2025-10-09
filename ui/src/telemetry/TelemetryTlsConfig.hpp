#pragma once

#include <QString>

struct TelemetryTlsConfig {
    bool enabled = false;
    QString rootCertificatePath;
    QString clientCertificatePath;
    QString clientKeyPath;
    QString serverNameOverride;
    QString pinnedServerSha256;
};

