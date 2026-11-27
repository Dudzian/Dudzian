#ifndef STAGE9_V5D11_POLICY_H
#define STAGE9_V5D11_POLICY_H

/* Linux x86-64 profile A. Each function returns 0 or -1 and sets errno. */
int stage9_v5d11_install_seccomp(void);
int stage9_v5d11_probe_seccomp(void);
int stage9_v5d11_dump_seccomp(int fd);

#endif
