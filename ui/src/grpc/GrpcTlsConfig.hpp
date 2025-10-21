#pragma once

#include <QString>

struct GrpcTlsConfig {
    bool     enabled = false;
    bool     requireClientAuth = false;
    QString  rootCertificatePath;
    QString  clientCertificatePath;
    QString  clientKeyPath;
    QString  serverNameOverride;
    QString  targetNameOverride;
    QString  pinnedServerFingerprint;
};

