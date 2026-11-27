/* Trusted Linux measurement of the stage9_v5d10_launch.py process view. */
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <inttypes.h>
#include <limits.h>
#include <linux/capability.h>
#include <linux/magic.h>
#include <linux/securebits.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <sys/wait.h>
#include <unistd.h>

#include "stage9_v5d11_policy.h"

#define PROFILE "cryptohunter-observer-preflight-v5d11-a"
#define PROFILE_LABEL PROFILE " (enforce)"
#define AUTHORITY "/run/stage9-v5d11/launch-authorization.json"
#define MOUNTINFO_LIMIT (4U * 1024U * 1024U)

struct settings {
    const char *stage;
    uid_t uid;
    gid_t gid;
    const char *socket;
    const char *root;
    const char *receipt_sha;
    const char *approval_sha;
    const char *authority;
};

struct namespace_ids {
    uintmax_t mount;
    uintmax_t pid;
    uintmax_t network;
};

static char mountinfo[MOUNTINFO_LIMIT + 1];

static void write_exact(int fd, const void *buffer, size_t count)
{
    const char *data = buffer;
    while (count > 0) {
        ssize_t length = write(fd, data, count);
        if (length < 0 && errno == EINTR)
            continue;
        if (length <= 0)
            _exit(126);
        data += length;
        count -= (size_t)length;
    }
}

static void fail(const char *code) __attribute__((noreturn));

static void fail(const char *code)
{
    write_exact(STDERR_FILENO, code, strlen(code));
    write_exact(STDERR_FILENO, "\n", 1);
    _exit(125);
}

static bool digest_string(const char *value)
{
    return strlen(value) == 64 && strspn(value, "0123456789abcdef") == 64;
}

static bool absolute_path(const char *value)
{
    const char allowed[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.";
    if (value[0] != '/' || strlen(value) >= PATH_MAX)
        return false;
    for (const char *part = value + 1;;) {
        size_t length = strspn(part, allowed);
        if (length == 0 || (length == 1 && part[0] == '.') ||
            (length == 2 && part[0] == '.' && part[1] == '.'))
            return false;
        if (part[length] == '\0')
            return true;
        if (part[length] != '/')
            return false;
        part += length + 1;
    }
}

static void join(char result[PATH_MAX], const char *left, const char *right)
{
    int length = snprintf(result, PATH_MAX, "%s%s", left, right);
    if (length <= 0 || length >= PATH_MAX)
        fail("BOOTSTRAP_PATH_LENGTH");
}

static unsigned int account_id(const char *value)
{
    if (*value == '\0' || strspn(value, "0123456789") != strlen(value))
        fail("BOOTSTRAP_ACCOUNT_ID");
    errno = 0;
    unsigned long result = strtoul(value, NULL, 10);
    if (errno != 0 || result == 0 || result >= UINT_MAX)
        fail("BOOTSTRAP_ACCOUNT_ID");
    return (unsigned int)result;
}

static struct stat path_stat(const char *path)
{
    struct stat info;
    if (lstat(path, &info) != 0)
        fail("BOOTSTRAP_PATH_STAT");
    return info;
}

static void check_pipe(int fd, int access)
{
    struct statfs fs;
    int flags = fcntl(fd, F_GETFL);
    if (flags < 0 || (flags & O_ACCMODE) != access || (flags & O_NONBLOCK) != 0 ||
        fstatfs(fd, &fs) != 0 || fs.f_type != PIPEFS_MAGIC)
        fail("BOOTSTRAP_STDIO_PIPE_REQUIRED");
}

static void check_stage(const char *stage)
{
    char resolved[PATH_MAX];
    if (realpath(stage, resolved) == NULL || strcmp(stage, resolved) != 0)
        fail("BOOTSTRAP_STAGE_PATH");
    struct stat directory = path_stat(resolved);
    if (!S_ISDIR(directory.st_mode) || directory.st_uid != 0 ||
        (directory.st_mode & 07777) != 0755)
        fail("BOOTSTRAP_STAGE_OWNER_MODE");
    *strrchr(resolved, '/') = '\0';
    struct stat parent = path_stat(resolved);
    if (!S_ISDIR(parent.st_mode) || parent.st_uid != 0 || (parent.st_mode & 07777) != 0700)
        fail("BOOTSTRAP_STAGE_PARENT_OWNER_MODE");
    for (char *slash = strrchr(resolved, '/'); slash != NULL; slash = strrchr(resolved, '/')) {
        if (slash == resolved)
            break;
        *slash = '\0';
        struct stat ancestor = path_stat(resolved);
        if (!S_ISDIR(ancestor.st_mode) || ancestor.st_uid != 0 || (ancestor.st_mode & 0022) != 0)
            fail("BOOTSTRAP_STAGE_ANCESTOR_OWNER_MODE");
    }
}

static uintmax_t namespace_inode(const char *path)
{
    struct stat info;
    if (stat(path, &info) != 0 || info.st_ino == 0)
        fail("BOOTSTRAP_NAMESPACE_STAT");
    return info.st_ino;
}

static struct namespace_ids read_namespaces(void)
{
    return (struct namespace_ids){
        .mount = namespace_inode("/proc/self/ns/mnt"),
        .pid = namespace_inode("/proc/self/ns/pid"),
        .network = namespace_inode("/proc/self/ns/net"),
    };
}

static bool same_inode(struct stat left, struct stat right)
{
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

static void isolate_descriptors(void)
{
    check_pipe(STDIN_FILENO, O_RDONLY);
    check_pipe(STDOUT_FILENO, O_WRONLY);
    check_pipe(STDERR_FILENO, O_WRONLY);
    struct stat input, output, error;
    if (fstat(STDIN_FILENO, &input) != 0 || fstat(STDOUT_FILENO, &output) != 0 ||
        fstat(STDERR_FILENO, &error) != 0 || same_inode(input, output) ||
        same_inode(input, error) || same_inode(output, error) ||
        syscall(SYS_close_range, 3U, UINT_MAX, 0U) != 0)
        fail("BOOTSTRAP_INHERITED_DESCRIPTORS");
}

static struct stat bind_one(const char *stage, const char *path, bool is_socket)
{
    char target[PATH_MAX];
    char source[64];
    join(target, stage, path);
    struct stat placeholder = path_stat(target);
    if (!S_ISREG(placeholder.st_mode) || placeholder.st_uid != 0 || placeholder.st_nlink != 1)
        fail("BOOTSTRAP_BIND_PLACEHOLDER");
    int fd = open(path, O_PATH | O_NOFOLLOW | O_CLOEXEC);
    struct stat expected;
    if (fd < 0 || fstat(fd, &expected) != 0 || expected.st_uid != 0)
        fail("BOOTSTRAP_BIND_SOURCE");
    if (is_socket ? !S_ISSOCK(expected.st_mode) :
        (!S_ISCHR(expected.st_mode) || major(expected.st_rdev) != 1 || minor(expected.st_rdev) != 3))
        fail("BOOTSTRAP_BIND_SOURCE_TYPE");
    int length = snprintf(source, sizeof(source), "/proc/self/fd/%d", fd);
    unsigned long flags = MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NOEXEC;
    if (is_socket)
        flags |= MS_NODEV;
    if (length <= 0 || (size_t)length >= sizeof(source) ||
        mount(source, target, NULL, MS_BIND, NULL) != 0 ||
        mount(NULL, target, NULL, flags, NULL) != 0)
        fail("BOOTSTRAP_SINGLE_OBJECT_MOUNT");
    close(fd);
    if (!same_inode(expected, path_stat(target)))
        fail("BOOTSTRAP_BIND_IDENTITY_CHANGED");
    return expected;
}

static struct stat enter_view(const struct settings *settings)
{
    if (mount(settings->stage, settings->stage, NULL, MS_BIND, NULL) != 0)
        fail("BOOTSTRAP_STAGE_BIND");

    char proc[PATH_MAX];
    join(proc, settings->stage, "/proc");
    struct stat proc_directory = path_stat(proc);
    if (!S_ISDIR(proc_directory.st_mode) || proc_directory.st_uid != 0 ||
        mount("proc", proc, "proc", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC, NULL) != 0)
        fail("BOOTSTRAP_PRIVATE_PROC");

    bind_one(settings->stage, "/dev/null", false);
    struct stat socket = bind_one(settings->stage, settings->socket, true);
    if (mount(NULL, settings->stage, NULL, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
              NULL) != 0)
        fail("BOOTSTRAP_READ_ONLY_ROOT");

    /* Putting the old root over the new root avoids a put-old directory in the view. */
    if (chdir(settings->stage) != 0 || syscall(SYS_pivot_root, ".", ".") != 0 ||
        umount2(".", MNT_DETACH) != 0 || chdir("/") != 0)
        fail("BOOTSTRAP_PIVOT_ROOT");
    struct statvfs filesystem;
    if (statvfs("/", &filesystem) != 0 || (filesystem.f_flag & ST_RDONLY) == 0 || getpid() != 1 ||
        !same_inode(socket, path_stat(settings->socket)))
        fail("BOOTSTRAP_FINAL_VIEW");
    return socket;
}

static void drop_privilege(uid_t uid, gid_t gid)
{
    for (int capability = 0; capability < 64; ++capability) {
        if (prctl(PR_CAPBSET_DROP, capability, 0L, 0L, 0L) != 0 && errno != EINVAL)
            fail("BOOTSTRAP_BOUNDING_CAPABILITIES");
    }
    unsigned long securebits = SECBIT_NOROOT | SECBIT_NOROOT_LOCKED | SECBIT_KEEP_CAPS_LOCKED |
        SECBIT_NO_CAP_AMBIENT_RAISE | SECBIT_NO_CAP_AMBIENT_RAISE_LOCKED;
    if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL, 0L, 0L, 0L) != 0 ||
        prctl(PR_SET_SECUREBITS, securebits, 0L, 0L, 0L) != 0 || setgroups(0, NULL) != 0 ||
        setresgid(gid, gid, gid) != 0 || setresuid(uid, uid, uid) != 0)
        fail("BOOTSTRAP_DROP_IDENTITY");

    struct __user_cap_header_struct header = {.version = _LINUX_CAPABILITY_VERSION_3, .pid = 0};
    struct __user_cap_data_struct caps[2] = {{0}};
    if (syscall(SYS_capset, &header, caps) != 0 || syscall(SYS_capget, &header, caps) != 0)
        fail("BOOTSTRAP_CAPABILITIES");
    for (size_t index = 0; index < sizeof(caps) / sizeof(caps[0]); ++index) {
        if (caps[index].effective != 0 || caps[index].permitted != 0 || caps[index].inheritable != 0)
            fail("BOOTSTRAP_CAPABILITIES");
    }
    for (int capability = 0; capability < 64; ++capability) {
        int bounding = prctl(PR_CAPBSET_READ, capability, 0L, 0L, 0L);
        if (bounding != 0 && !(bounding == -1 && errno == EINVAL))
            fail("BOOTSTRAP_FINAL_BOUNDING_CAPABILITIES");
        int ambient = prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_IS_SET, capability, 0L, 0L);
        if (ambient != 0 && !(ambient == -1 && errno == EINVAL))
            fail("BOOTSTRAP_FINAL_AMBIENT_CAPABILITIES");
    }
    uid_t real, effective, saved;
    gid_t real_group, effective_group, saved_group;
    if (getresuid(&real, &effective, &saved) != 0 || real != uid || effective != uid || saved != uid ||
        getresgid(&real_group, &effective_group, &saved_group) != 0 || real_group != gid ||
        effective_group != gid || saved_group != gid || getgroups(0, NULL) != 0 ||
        prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) != 0 ||
        prctl(PR_GET_DUMPABLE, 0L, 0L, 0L, 0L) != 0 ||
        prctl(PR_SET_PDEATHSIG, SIGKILL, 0L, 0L, 0L) != 0)
        fail("BOOTSTRAP_FINAL_IDENTITY");
}

static size_t read_bounded(int fd, char *buffer, size_t size)
{
    size_t total = 0;
    while (total < size) {
        ssize_t length = read(fd, buffer + total, size - total);
        if (length < 0 && errno == EINTR)
            continue;
        if (length < 0)
            fail("BOOTSTRAP_EVIDENCE_READ");
        if (length == 0)
            break;
        total += (size_t)length;
    }
    if (total == size)
        fail("BOOTSTRAP_EVIDENCE_LIMIT");
    buffer[total] = '\0';
    return total;
}

static size_t read_object(const char *name, char *buffer, size_t size)
{
    int fd = open(name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0)
        fail("BOOTSTRAP_EVIDENCE_OPEN");
    size_t total = read_bounded(fd, buffer, size);
    close(fd);
    return total;
}

static void apply_policy(int attr)
{
    const char command[] = "changeprofile " PROFILE;
    if (write(attr, command, sizeof(command)) != (ssize_t)sizeof(command) || close(attr) != 0)
        fail("BOOTSTRAP_APPARMOR_TRANSITION");
    if (prctl(PR_SET_NO_NEW_PRIVS, 1L, 0L, 0L, 0L) != 0 ||
        prctl(PR_GET_NO_NEW_PRIVS, 0L, 0L, 0L, 0L) != 1 || stage9_v5d11_install_seccomp() != 0 ||
        stage9_v5d11_probe_seccomp() != 0)
        fail("BOOTSTRAP_SECCOMP_APPLICATION");

    char label[128];
    size_t length = read_object("/proc/self/attr/apparmor/current", label, sizeof(label));
    if (length > 0 && label[length - 1] == '\n')
        label[--length] = '\0';
    if (strcmp(label, PROFILE_LABEL) != 0)
        fail("BOOTSTRAP_APPARMOR_LABEL");
    int probe = open("/proc/self/environ", O_RDONLY | O_CLOEXEC);
    if (probe >= 0 || errno != EACCES)
        fail("BOOTSTRAP_APPARMOR_DENIAL");
}

static void emit_measurement(struct namespace_ids host, struct stat socket)
{
    struct namespace_ids isolated = read_namespaces();
    if (host.mount == isolated.mount || host.pid == isolated.pid || host.network == isolated.network)
        fail("BOOTSTRAP_NAMESPACES_NOT_PRIVATE");
    size_t length = read_object("/proc/self/mountinfo", mountinfo, sizeof(mountinfo));
    if (length == 0 || mountinfo[length - 1] != '\n')
        fail("BOOTSTRAP_MOUNTINFO_FORMAT");

    char header[512];
    int count = snprintf(header, sizeof(header),
        "{\"protocol\":\"stage9-v5d11-bootstrap-v1\",\"mountinfo_bytes\":%zu,"
        "\"namespaces\":{\"mnt\":%ju,\"pid\":%ju,\"net\":%ju},"
        "\"apparmor_label\":\"" PROFILE_LABEL "\",\"socket_identity\":[%ju,%ju,%ju,%ju,%ju]}\n",
        length, isolated.mount, isolated.pid, isolated.network, (uintmax_t)socket.st_dev,
        (uintmax_t)socket.st_ino, (uintmax_t)socket.st_uid, (uintmax_t)socket.st_gid,
        (uintmax_t)(socket.st_mode & 07777));
    if (count <= 0 || (size_t)count >= sizeof(header))
        fail("BOOTSTRAP_FRAME");
    write_exact(STDOUT_FILENO, header, (size_t)count);
    write_exact(STDOUT_FILENO, mountinfo, length);
}

static void parked_gate(void)
{
    char gate[69];
    size_t size = read_bounded(STDIN_FILENO, gate, sizeof(gate));
    if (size == 0 || (size == 7 && memcmp(gate, "CANCEL\n", 7) == 0))
        _exit(0);
    if (size != 68 || memcmp(gate, "GO ", 3) != 0 || gate[67] != '\n')
        fail("BOOTSTRAP_GATE_PROTOCOL");
    gate[67] = '\0';
    if (!digest_string(gate + 3))
        fail("BOOTSTRAP_GATE_PROTOCOL");

    /* Measurement accepts no release. Review never becomes an exec decision here. */
    fail("BOOTSTRAP_NATIVE_RELEASE_DISABLED");
}

static void worker(const struct settings *settings, struct namespace_ids host)
{
    char dumpable[8];
    read_object("/proc/sys/fs/suid_dumpable", dumpable, sizeof(dumpable));
    /* A UID change must never transiently permit same-UID host ptrace. */
    if (strcmp(dumpable, "0\n") != 0 && strcmp(dumpable, "2\n") != 0)
        fail("BOOTSTRAP_HOST_DUMPABILITY_REJECTED");
    char label[128];
    size_t length = read_object("/proc/self/attr/apparmor/current", label, sizeof(label));
    if (length > 0 && label[length - 1] == '\n')
        label[length - 1] = '\0';
    if (strcmp(label, "unconfined") != 0)
        fail("BOOTSTRAP_EXISTING_APPARMOR_CONFINEMENT");
    int attr = open("/proc/self/attr/apparmor/current", O_WRONLY | O_NOFOLLOW | O_CLOEXEC);
    if (attr < 0 || prctl(PR_SET_PDEATHSIG, SIGKILL, 0L, 0L, 0L) != 0)
        fail("BOOTSTRAP_APPARMOR_ATTR");
    struct stat socket = enter_view(settings);
    struct stat authority = path_stat(settings->authority);
    if (!S_ISREG(authority.st_mode) || authority.st_uid != 0 || authority.st_nlink != 1 ||
        (authority.st_mode & 07777) != 0444 || authority.st_size != 0)
        fail("BOOTSTRAP_AUTHORITY_PLACEHOLDER");
    drop_privilege(settings->uid, settings->gid);
    apply_policy(attr);
    emit_measurement(host, socket);
    parked_gate();
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--dump-seccomp") == 0)
        return stage9_v5d11_dump_seccomp(STDOUT_FILENO) == 0 ? 0 : 125;
    if (argc != 9 || getuid() != 0 || geteuid() != 0)
        fail("BOOTSTRAP_SUPERVISOR_REQUIRED");
    const struct settings settings = {
        .stage = argv[1], .uid = account_id(argv[2]), .gid = account_id(argv[3]), .socket = argv[4],
        .root = argv[5], .receipt_sha = argv[6], .approval_sha = argv[7], .authority = argv[8],
    };
    if (!absolute_path(settings.stage) || !absolute_path(settings.root) ||
        (strcmp(settings.socket, "/run/libvirt/virtqemud-sock-ro") != 0 &&
         strcmp(settings.socket, "/run/libvirt/libvirt-sock-ro") != 0) ||
        strcmp(settings.authority, AUTHORITY) != 0 || !digest_string(settings.receipt_sha) ||
        !digest_string(settings.approval_sha))
        fail("BOOTSTRAP_SETTINGS");
    isolate_descriptors();
    check_stage(settings.stage);
    umask(077);
    pid_t supervisor = getppid();
    if (supervisor <= 1 || prctl(PR_SET_PDEATHSIG, SIGKILL, 0L, 0L, 0L) != 0 ||
        getppid() != supervisor)
        fail("BOOTSTRAP_SUPERVISOR_GONE");
    struct namespace_ids host = read_namespaces();
    if (unshare(CLONE_NEWNS | CLONE_NEWPID | CLONE_NEWNET) != 0 ||
        mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL) != 0)
        fail("BOOTSTRAP_NAMESPACE_CREATE");
    pid_t child = fork();
    if (child < 0)
        fail("BOOTSTRAP_PID_NAMESPACE_FORK");
    if (child == 0) {
        worker(&settings, host);
        _exit(125);
    }
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
    int status;
    for (;;) {
        if (waitpid(child, &status, 0) == child)
            return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        if (errno != EINTR)
            return 125;
    }
}
