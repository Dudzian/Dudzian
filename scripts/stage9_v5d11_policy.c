/* V5d11 policy A v1, Linux x86-64 LP64. The same filter survives qemu-img exec. */
#define _GNU_SOURCE
#include "stage9_v5d11_policy.h"

#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sched.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <unistd.h>

#if !defined(__x86_64__) || defined(__ILP32__)
#error "V5d11 profile A requires Linux x86-64 LP64"
#endif

#define DENY (SECCOMP_RET_ERRNO | EUCLEAN)
#define PERMIT BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)
#define REFUSE BPF_STMT(BPF_RET | BPF_K, DENY)
#define ARG_LO(n) offsetof(struct seccomp_data, args[n])
#define ARG_HI(n) (ARG_LO(n) + sizeof(uint32_t))
#define LOAD(offset) BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (offset))
#define MATCH(value, yes, no) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (value), (yes), (no))
#define MASK(bits) BPF_STMT(BPF_ALU | BPF_AND | BPF_K, (bits))
#define ALLOW(call) MATCH(SYS_##call, 0, 1), PERMIT

/* Existing /dev/null is opened O_RDWR by CPython subprocess. Read-only mounts
 * and AppArmor enforce path-specific write permission, and this filter forbids
 * file creation, truncation and unnamed temporary files outright. */
#define OPEN(call, flags) \
    MATCH(SYS_##call, 0, 4), LOAD(ARG_LO(flags)), \
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, O_CREAT | O_TRUNC | O_APPEND | (O_TMPFILE & ~O_DIRECTORY), 1, 0), \
    PERMIT, REFUSE

#define STREAM(call) \
    MATCH(SYS_##call, 0, 9), \
    LOAD(ARG_LO(0)), MATCH(AF_UNIX, 0, 6), \
    LOAD(ARG_LO(1)), MASK(~(SOCK_CLOEXEC | SOCK_NONBLOCK)), MATCH(SOCK_STREAM, 0, 3), \
    LOAD(ARG_LO(2)), MATCH(0, 0, 1), PERMIT, REFUSE

#define THREAD_FLAGS \
    (CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD | CLONE_SYSVSEM | \
     CLONE_SETTLS | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID)
#define FORK_FLAGS (CLONE_CHILD_SETTID | CLONE_CHILD_CLEARTID | SIGCHLD)
#define SPAWN_FLAGS (CLONE_VM | CLONE_VFORK | SIGCHLD)

static struct sock_filter profile_a[] = {
    LOAD(offsetof(struct seccomp_data, arch)),
    MATCH(AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    LOAD(offsetof(struct seccomp_data, nr)),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, 0x40000000U, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),

    OPEN(open, 1), OPEN(openat, 2),
    STREAM(socket), STREAM(socketpair),
    /* AppArmor limits the UNIX destination to the socket in the private view. */
    ALLOW(connect), ALLOW(getsockopt), ALLOW(getsockname), ALLOW(getpeername),
    ALLOW(shutdown), ALLOW(recvfrom),
    /* send(2) is sendto with no destination; never allow ancillary data. */
    MATCH(SYS_sendto, 0, 6),
    LOAD(ARG_HI(4)), MATCH(0, 0, 3), LOAD(ARG_LO(4)), MATCH(0, 0, 1), PERMIT, REFUSE,

    MATCH(SYS_clone3, 0, 1), BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | ENOSYS),
    MATCH(SYS_clone, 0, 9),
    LOAD(ARG_HI(0)), MATCH(0, 0, 5), LOAD(ARG_LO(0)),
    MATCH(THREAD_FLAGS, 4, 0), MATCH(FORK_FLAGS, 3, 0),
    MATCH(SPAWN_FLAGS, 2, 0), MATCH(SIGCHLD, 1, 0), REFUSE, PERMIT,
    /* No flags can be passed to vfork. AppArmor constrains program execution. */
    ALLOW(vfork), ALLOW(execve),
    MATCH(SYS_kill, 0, 8),
    LOAD(ARG_LO(0)), MATCH(0, 5, 0),
    BPF_JUMP(BPF_JMP | BPF_JGT | BPF_K, 4194304U, 4, 0),
    LOAD(ARG_LO(1)), MATCH(SIGKILL, 1, 0), MATCH(SIGTERM, 0, 1), PERMIT, REFUSE,
    MATCH(SYS_tgkill, 0, 6),
    LOAD(ARG_LO(2)), MATCH(SIGABRT, 2, 0), MATCH(SIGKILL, 1, 0),
    MATCH(SIGTERM, 0, 1), PERMIT, REFUSE,

    /* Executable mappings must be file-backed and non-writable. AppArmor
     * then admits only the exact pinned file. No W-to-X transition is allowed. */
    MATCH(SYS_mmap, 0, 9), LOAD(ARG_LO(2)),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_EXEC, 0, 5),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_WRITE, 5, 0),
    LOAD(ARG_LO(3)), BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, MAP_ANONYMOUS, 3, 0),
    LOAD(ARG_LO(4)), MATCH(UINT32_MAX, 1, 0), PERMIT, REFUSE,
    MATCH(SYS_mprotect, 0, 4), LOAD(ARG_LO(2)),
    BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, PROT_EXEC, 1, 0),
    PERMIT, REFUSE,

    MATCH(SYS_fcntl, 0, 12), LOAD(ARG_LO(1)),
    MATCH(F_GETFD, 9, 0), MATCH(F_SETFD, 8, 0), MATCH(F_GETFL, 7, 0),
    MATCH(F_SETFL, 6, 0), MATCH(F_DUPFD_CLOEXEC, 5, 0),
    MATCH(F_GETLK, 4, 0), MATCH(F_SETLK, 3, 0), MATCH(F_OFD_GETLK, 2, 0),
    MATCH(F_OFD_SETLK, 1, 0), REFUSE, PERMIT,
    /* This is the V5d1 image guard's FS_IOC_GETFLAGS, a read of inode flags. */
    MATCH(SYS_ioctl, 0, 6),
    LOAD(ARG_HI(1)), MATCH(0, 0, 3), LOAD(ARG_LO(1)), MATCH(0x80086601U, 0, 1), PERMIT, REFUSE,
    MATCH(SYS_prctl, 0, 7), LOAD(ARG_LO(0)),
    MATCH(PR_SET_NAME, 4, 0), MATCH(PR_GET_NAME, 3, 0),
    MATCH(PR_GET_NO_NEW_PRIVS, 2, 0), MATCH(PR_GET_DUMPABLE, 1, 0), REFUSE, PERMIT,

    ALLOW(access), ALLOW(faccessat), ALLOW(faccessat2),
    ALLOW(read), ALLOW(readv), ALLOW(pread64), ALLOW(preadv), ALLOW(preadv2),
    ALLOW(write), ALLOW(writev), ALLOW(lseek), ALLOW(getdents64),
    ALLOW(stat), ALLOW(lstat), ALLOW(fstat), ALLOW(newfstatat), ALLOW(statx),
    ALLOW(statfs), ALLOW(fstatfs), ALLOW(readlink), ALLOW(readlinkat), ALLOW(getcwd),
    ALLOW(close), ALLOW(close_range), ALLOW(dup), ALLOW(dup2), ALLOW(dup3),
    ALLOW(pipe), ALLOW(pipe2), ALLOW(poll), ALLOW(ppoll), ALLOW(select), ALLOW(pselect6),
    ALLOW(epoll_create1), ALLOW(epoll_ctl), ALLOW(epoll_wait), ALLOW(epoll_pwait),
    ALLOW(eventfd2), ALLOW(futex), ALLOW(brk), ALLOW(munmap), ALLOW(mremap), ALLOW(madvise),
    ALLOW(arch_prctl), ALLOW(set_tid_address), ALLOW(set_robust_list), ALLOW(rseq),
    ALLOW(rt_sigaction), ALLOW(rt_sigprocmask), ALLOW(rt_sigreturn), ALLOW(sigaltstack),
    ALLOW(wait4), ALLOW(waitid), ALLOW(exit), ALLOW(exit_group),
    ALLOW(getpid), ALLOW(getppid), ALLOW(gettid), ALLOW(getuid), ALLOW(geteuid),
    ALLOW(getgid), ALLOW(getegid), ALLOW(getresuid), ALLOW(getresgid), ALLOW(getgroups),
    ALLOW(uname), ALLOW(sysinfo), ALLOW(getrusage), ALLOW(getcpu), ALLOW(sched_getaffinity),
    ALLOW(sched_yield), ALLOW(clock_gettime), ALLOW(clock_getres), ALLOW(gettimeofday),
    ALLOW(nanosleep), ALLOW(clock_nanosleep), ALLOW(getrandom), ALLOW(getrlimit),
    /* Query only. Both halves of the new-limit user pointer must be zero. */
    MATCH(SYS_prlimit64, 0, 6),
    LOAD(ARG_HI(2)), MATCH(0, 0, 3), LOAD(ARG_LO(2)), MATCH(0, 0, 1), PERMIT, REFUSE,

    REFUSE,
};

_Static_assert(sizeof(profile_a) / sizeof(profile_a[0]) < 4096, "classic BPF limit");
_Static_assert(sizeof(struct sock_filter) == 8, "unexpected Linux BPF layout");

int stage9_v5d11_install_seccomp(void) {
    struct sock_fprog program = {
        .len = sizeof(profile_a) / sizeof(profile_a[0]),
        .filter = profile_a,
    };
    int no_new_privs = prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0);
    if (no_new_privs != 1) {
        if (no_new_privs == 0)
            errno = EPERM;
        return -1;
    }
    long result = syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_TSYNC, &program);
    if (result > 0)
        errno = EBUSY;
    return result == 0 ? 0 : -1;
}

int stage9_v5d11_probe_seccomp(void) {
    /* Invalid addresses and FDs prevent effects even when the filter is absent. */
    const struct {
        long syscall_number;
        long first;
        long second;
        long third;
    } probes[] = {
        {SYS_openat, -1, 0, O_CREAT},
        {SYS_execveat, -1, 0, 0},
        {SYS_socket, AF_INET, 0, 0},
        {SYS_recvmsg, -1, 0, 0},
        {SYS_sendmsg, -1, 0, 0},
        {SYS_setns, -1, 0, 0},
        {SYS_ptrace, -1, 0, 0},
        {SYS_io_uring_enter, -1, 0, 0},
        {SYS_ioctl, -1, 0x40086602U, 0},  /* FS_IOC_SETFLAGS */
        {SYS_setresuid, -1, -1, -1},
        {SYS_mprotect, 0, 0, PROT_EXEC},
        {SYS_mmap, 0, 0, PROT_WRITE | PROT_EXEC},
    };
    for (size_t index = 0; index < sizeof(probes) / sizeof(probes[0]); ++index) {
        errno = 0;
        if (syscall(probes[index].syscall_number, probes[index].first,
                    probes[index].second, probes[index].third, 0L, 0L, 0L) != -1 ||
            errno != EUCLEAN) {
            errno = EPERM;
            return -1;
        }
    }
    return 0;
}

int stage9_v5d11_dump_seccomp(int fd) {
    const unsigned char *cursor = (const unsigned char *)profile_a;
    size_t remaining = sizeof(profile_a);
    while (remaining > 0) {
        ssize_t written = write(fd, cursor, remaining);
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0) {
            if (written == 0)
                errno = EIO;
            return -1;
        }
        cursor += written;
        remaining -= (size_t)written;
    }
    return 0;
}
