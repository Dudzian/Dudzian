#pragma once

#include <QString>
#include <QStringList>

namespace bot::shell::utils {

//! Expand environment placeholders (e.g. $VAR or %VAR%) without normalizing the result.
QString expandEnvironmentPlaceholders(const QString& text);

//! Expand paths supporting '~' and relative segments using current working directory.
QString expandPath(const QString& path);

//! Return directories that can be watched for changes for a potentially non-existent path.
QStringList watchableDirectories(const QString& directoryPath);

} // namespace bot::shell::utils
