#include "PathUtils.hpp"

#include <QByteArray>
#include <QDir>
#include <QFileInfo>
#include <QtGlobal>
#include <QUrl>

#if defined(Q_OS_UNIX)
#    include <algorithm>
#    include <errno.h>
#    include <limits>
#    include <pwd.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

namespace {

QString expandEnvironmentVariables(const QString& input)
{
    QString result;
    result.reserve(input.size());

    int index = 0;
    while (index < input.size()) {
        const QChar ch = input.at(index);
        if (ch == QLatin1Char('$')) {
            if (index + 1 < input.size() && input.at(index + 1) == QLatin1Char('{')) {
                const int end = input.indexOf(QLatin1Char('}'), index + 2);
                if (end > index + 2) {
                    const QString name = input.mid(index + 2, end - index - 2);
                    const QByteArray nameBytes = name.toUtf8();
                    if (!name.isEmpty() && qEnvironmentVariableIsSet(nameBytes.constData())) {
                        result.append(qEnvironmentVariable(nameBytes.constData()));
                        index = end + 1;
                        continue;
                    }
                    result.append(input.mid(index, end - index + 1));
                    index = end + 1;
                    continue;
                }
            } else {
                int end = index + 1;
                while (end < input.size()) {
                    const QChar candidate = input.at(end);
                    if (!candidate.isLetterOrNumber() && candidate != QLatin1Char('_'))
                        break;
                    ++end;
                }
                if (end > index + 1) {
                    const QString name = input.mid(index + 1, end - index - 1);
                    const QByteArray nameBytes = name.toUtf8();
                    if (!name.isEmpty() && qEnvironmentVariableIsSet(nameBytes.constData())) {
                        result.append(qEnvironmentVariable(nameBytes.constData()));
                        index = end;
                        continue;
                    }
                    result.append(input.mid(index, end - index));
                    index = end;
                    continue;
                }
            }
        } else if (ch == QLatin1Char('%')) {
            const int end = input.indexOf(QLatin1Char('%'), index + 1);
            if (end > index + 1) {
                const QString name = input.mid(index + 1, end - index - 1);
                const QByteArray nameBytes = name.toUtf8();
                if (!name.isEmpty() && qEnvironmentVariableIsSet(nameBytes.constData())) {
                    result.append(qEnvironmentVariable(nameBytes.constData()));
                    index = end + 1;
                    continue;
                }
                result.append(input.mid(index, end - index + 1));
                index = end + 1;
                continue;
            }
        }

        result.append(ch);
        ++index;
    }

    return result;
}

#if defined(Q_OS_UNIX)
QString resolveUserHomeDirectory(const QString& user)
{
    const QString trimmed = user.trimmed();
    if (trimmed.isEmpty())
        return {};

    const QByteArray userBytes = trimmed.toUtf8();

#    if defined(_POSIX_THREAD_SAFE_FUNCTIONS)
    struct passwd pwd;
    struct passwd* result = nullptr;
    long bufSize = sysconf(_SC_GETPW_R_SIZE_MAX);
    if (bufSize < 0)
        bufSize = 16384;

    const long clamped = std::min<long>(bufSize, std::numeric_limits<int>::max());

    QByteArray buffer;
    buffer.resize(static_cast<int>(clamped));

    const int rc = getpwnam_r(userBytes.constData(), &pwd, buffer.data(), static_cast<size_t>(buffer.size()), &result);
    if (rc == 0 && result && result->pw_dir)
        return QString::fromLocal8Bit(result->pw_dir);
    return {};
#    else
    errno = 0;
    struct passwd* pwd = getpwnam(userBytes.constData());
    if (pwd && pwd->pw_dir)
        return QString::fromLocal8Bit(pwd->pw_dir);
    return {};
#    endif
}
#else
QString resolveUserHomeDirectory(const QString& user)
{
    Q_UNUSED(user);
    return {};
}
#endif

} // namespace

namespace bot::shell::utils {

QString expandPath(const QString& path)
{
    if (path.trimmed().isEmpty())
        return {};

    QString expanded = expandEnvironmentVariables(path.trimmed());

    if (expanded.startsWith(QStringLiteral("file:"), Qt::CaseInsensitive)) {
        const QUrl url = QUrl::fromUserInput(expanded);
        if (url.isValid() && url.scheme().compare(QStringLiteral("file"), Qt::CaseInsensitive) == 0) {
            QString local = url.toLocalFile();
            if (local.isEmpty())
                local = url.path();
            if (!local.isEmpty())
                expanded = local;
        }
    }
    if (expanded == QStringLiteral("~")) {
        expanded = QDir::homePath();
    } else if (expanded.startsWith(QStringLiteral("~/"))) {
        expanded = QDir::homePath() + expanded.mid(1);
    } else if (expanded.startsWith(QLatin1Char('~'))) {
        const int slashIndex = expanded.indexOf(QLatin1Char('/'), 1);
        const QString user = slashIndex == -1 ? expanded.mid(1) : expanded.mid(1, slashIndex - 1);
        if (!user.isEmpty()) {
            const QString home = resolveUserHomeDirectory(user);
            if (!home.isEmpty()) {
                const QString remainder = slashIndex == -1 ? QString() : expanded.mid(slashIndex);
                expanded = home + remainder;
            }
        }
    }

    const auto isAsciiLetter = [](QChar ch) {
        return (ch >= QLatin1Char('a') && ch <= QLatin1Char('z')) || (ch >= QLatin1Char('A') && ch <= QLatin1Char('Z'));
    };

    const bool isDrivePrefixed = expanded.size() >= 2 && isAsciiLetter(expanded.at(0)) && expanded.at(1) == QLatin1Char(':');
    const bool isUncPath = expanded.startsWith(QStringLiteral("\\\\"));

    if (isDrivePrefixed || isUncPath) {
        QString normalized = expanded;
        normalized.replace(QLatin1Char('\\'), QLatin1Char('/'));

        if (isDrivePrefixed && normalized.size() == 2)
            normalized.append(QLatin1Char('/'));

        return QDir::cleanPath(normalized);
    }

    QFileInfo info(expanded);
    if (!info.isAbsolute())
        expanded = QDir::current().absoluteFilePath(expanded);

    return QDir::cleanPath(expanded);
}

QStringList watchableDirectories(const QString& directoryPath)
{
    QString cleaned = directoryPath.trimmed();
    if (cleaned.isEmpty())
        return {};

    QDir dir(cleaned);
    QStringList result;

    if (dir.exists()) {
        result.append(dir.absolutePath());
        return result;
    }

    QDir cursor(dir);
    QString previous;
    while (true) {
        const QString current = cursor.absolutePath();
        if (current.isEmpty() || current == previous)
            break;

        if (cursor.exists()) {
            result.append(cursor.absolutePath());
            break;
        }

        previous = current;
        if (!cursor.cdUp())
            break;
    }

    return result;
}

} // namespace bot::shell::utils
