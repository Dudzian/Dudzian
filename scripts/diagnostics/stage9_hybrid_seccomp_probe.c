/* TEST_ONLY: rzeczywisty, nieuprzywilejowany probe samej fazy B.
 * Nie tworzy socketu/listenera, nie uruchamia Pythona i nie sprawdza AppArmor.
 * Kompilacja: cc -static -O2 -Wall -Wextra -Werror plik.c -o probe
 * Zmiana PROBE_DENY_ERRNO służy wyłącznie różnicowemu przypisaniu odmowy.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#if !defined(__x86_64__) || defined(__ILP32__)
#error "Probe TEST_ONLY wymaga natywnego Linux x86-64 LP64"
#endif

#ifndef PROBE_DENY_ERRNO
#define PROBE_DENY_ERRNO EACCES
#endif
#ifndef PROBE_FILTER_FLAGS
#define PROBE_FILTER_FLAGS SECCOMP_FILTER_FLAG_TSYNC
#endif
#ifndef PROBE_X32_ACTION
#define PROBE_X32_ACTION SECCOMP_RET_KILL_PROCESS
#endif

_Static_assert(PROBE_DENY_ERRNO > 0 && PROBE_DENY_ERRNO <= 4095,
               "Niepoprawny errno odmowy TEST_ONLY");

struct probe {
    const char *name;
    long number;
    char report[96];
    size_t report_size;
};

/* Wskaźniki i FD są celowo nieprawidłowe. Brak filtra nigdy nie prowadzi do
 * otwarcia programu, pliku, połączenia ani dostępu do obcego procesu.
 * Dla fork/vfork/clone natychmiast kończymy obie gałęzie, jeśli filtr zawiedzie.
 */
#define CASE(name) {#name, SYS_##name, {0}, 0}
static struct probe probes[] = {
    CASE(execve), CASE(execveat), CASE(fork), CASE(vfork), CASE(clone),
    CASE(clone3), CASE(sendmsg), CASE(sendmmsg), CASE(recvmsg), CASE(recvmmsg),
    CASE(io_uring_setup), CASE(io_uring_enter), CASE(io_uring_register),
    CASE(ptrace), CASE(process_vm_readv), CASE(process_vm_writev),
    CASE(pidfd_getfd), CASE(openat), CASE(prctl), CASE(seccomp),
    CASE(socket), CASE(bind), CASE(listen), CASE(accept4), CASE(mmap),
    CASE(mprotect), CASE(write)
};
#undef CASE

static void finish(int status) __attribute__((noreturn));
static void finish(int status) {
    syscall(SYS_exit_group, status);
    __builtin_unreachable();
}

static void emit(const char *message, size_t size) {
    /* Po seal dozwolony jest wyłącznie write na dokładnym FD 1. */
    while (size > 0) {
        long written = syscall(SYS_write, 1, message, size);
        if (written <= 0)
            finish(3);
        message += written;
        size -= (size_t)written;
    }
}

#define EMIT(text) emit(text, sizeof(text) - 1)

static int seal(void) {
    const unsigned int deny = SECCOMP_RET_ERRNO | PROBE_DENY_ERRNO;
    struct sock_filter code[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JSET | BPF_K, 0x40000000U, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, PROBE_X32_ACTION),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_exit, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_exit_group, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_write, 0, 5),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, args[0]) + 4),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 0, 0, 3),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
                 offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, 1),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        BPF_STMT(BPF_RET | BPF_K, deny)
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(code) / sizeof(code[0])),
        .filter = code
    };
    if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 ||
        prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0 ||
        prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 ||
        prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1)
        return -1;
    /* TSYNC może zwrócić dodatni TID błędnego wątku: każdy wynik != 0 blokuje. */
    return syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER,
                   PROBE_FILTER_FLAGS, &program) == 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    int x32;
    char x32_report[64];
    int x32_size;
    if (argc != 2 || (strcmp(argv[1], "--probe") != 0 &&
                      strcmp(argv[1], "--probe-x32") != 0)) {
        EMIT("BLOCKED: wymagane --probe albo --probe-x32; brak listenera\n");
        finish(64);
    }
    x32 = strcmp(argv[1], "--probe-x32") == 0;
    x32_size = snprintf(x32_report, sizeof(x32_report), "ODMOWA:x32:%u\n",
                        (unsigned int)(PROBE_X32_ACTION & SECCOMP_RET_DATA));
    if (x32_size <= 0 || (size_t)x32_size >= sizeof(x32_report))
        finish(2);
    /* Wszystkie operacje formatowania wykonujemy przed filtrem. */
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
        int size = snprintf(probes[i].report, sizeof(probes[i].report),
                            "ODMOWA:%s:%d\n", probes[i].name, PROBE_DENY_ERRNO);
        if (size <= 0 || (size_t)size >= sizeof(probes[i].report))
            finish(2);
        probes[i].report_size = (size_t)size;
    }
    if (seal() != 0) {
        EMIT("BLOCKED: ustanowienie ochrony nieudane; brak listenera\n");
        finish(2);
    }
    if (x32) {
        errno = 0;
        long result = syscall(SYS_write | 0x40000000U, 1, "", 0);
        if ((PROBE_X32_ACTION & SECCOMP_RET_ACTION_FULL) == SECCOMP_RET_ERRNO &&
            result == -1 && errno == (PROBE_X32_ACTION & SECCOMP_RET_DATA)) {
            emit(x32_report, (size_t)x32_size);
            finish(0);
        }
        EMIT("BLAD: x32 nie zakonczyl procesu\n");
        finish(3);
    }
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); ++i) {
        errno = 0;
        /* -1 jako pierwszy argument wyklucza prawidłowy PID/FD/flagę clone.
         * write na -1 bada ograniczenie FD, nie dozwolony kanał raportu.
         */
        long result = syscall(probes[i].number, -1L, 0L, 0L, 0L, 0L, 0L);
        if (result != -1 || errno != PROBE_DENY_ERRNO) {
            EMIT("BLAD: brak przypisanej odmowy filtra\n");
            finish(3);
        }
        emit(probes[i].report, probes[i].report_size);
    }
    EMIT("TEST_ONLY: lokalny filtr sprawdzony; AppArmor/Python/AF_UNIX NOT_PROVEN\n");
    finish(0);
}
