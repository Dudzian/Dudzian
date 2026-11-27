/* Immutable image proof after entering the private mount view and dropping privileges. */
#ifndef STAGE9_V5D11_IMAGES_H
#define STAGE9_V5D11_IMAGES_H

#include <linux/fs.h>
#include <sys/ioctl.h>

#define IMAGE_ROOT "/var/lib/libvirt/images/"
#define IMAGE_MAX_BYTES (UINTMAX_C(128) * 1024 * 1024 * 1024)

static uintmax_t image_number(const char **input, char delimiter)
{
    if (**input < '0' || **input > '9')
        fail("BOOTSTRAP_IMAGE_IDENTITY_SYNTAX");
    char *tail;
    errno = 0;
    uintmax_t number = strtoumax(*input, &tail, 10);
    if (errno != 0 || *tail != delimiter || number == 0)
        fail("BOOTSTRAP_IMAGE_IDENTITY_SYNTAX");
    *input = tail + (delimiter != '\0');
    return number;
}

static void verify_image_directory(int fd)
{
    struct stat info;
    if (fd < 0 || fstat(fd, &info) != 0 || !S_ISDIR(info.st_mode) ||
        info.st_uid != 0 || (info.st_mode & 022) != 0)
        fail("BOOTSTRAP_IMAGE_ANCESTOR");
}

static int open_protected_image(const char *path)
{
    char name[PATH_MAX];
    size_t size = strlen(path);
    memcpy(name, path, size + 1);
    char *part = name + 1;
    const int dir_flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW;
    int parent = open("/", dir_flags);
    verify_image_directory(parent);
    for (char *slash = strchr(part, '/'); slash != NULL; slash = strchr(part, '/')) {
        *slash = '\0';
        int child = openat(parent, part, dir_flags);
        verify_image_directory(child);
        if (close(parent) != 0)
            fail("BOOTSTRAP_IMAGE_CLOSE");
        parent = child;
        part = slash + 1;
    }
    int fd = openat(parent, part, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (close(parent) != 0 || fd < 0)
        fail("BOOTSTRAP_IMAGE_OPEN");
    return fd;
}

static void check_image_view(char *const *arguments, size_t count, gid_t gid)
{
    if (count % 2 != 0)
        fail("BOOTSTRAP_IMAGE_ARGUMENTS");
    const char *previous = "";
    for (size_t index = 0; index < count; index += 2) {
        const char *path = arguments[index];
        const char *identity = arguments[index + 1];
        if (!absolute_path(path) || strncmp(path, IMAGE_ROOT, sizeof(IMAGE_ROOT) - 1) != 0 ||
            strcmp(previous, path) >= 0)
            fail("BOOTSTRAP_IMAGE_PATH_REJECTED");
        previous = path;
        uintmax_t device = image_number(&identity, ':');
        uintmax_t inode = image_number(&identity, ':');
        uintmax_t size = image_number(&identity, '\0');
        if (size > IMAGE_MAX_BYTES)
            fail("BOOTSTRAP_IMAGE_SIZE_REJECTED");
        int fd = open_protected_image(path);
        struct stat info;
        struct statfs fs;
        struct statvfs mount;
        /* Linux advertises the same superblock magic for ext2/3/4; the
         * supervisor checks ext4 in host mountinfo for this exact inode. */
        unsigned long flags = 0;
        if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) || info.st_uid != 0 ||
            info.st_gid != gid || info.st_nlink != 1 || (info.st_mode & 07777) != 0440 ||
            (uintmax_t)info.st_dev != device || (uintmax_t)info.st_ino != inode ||
            (uintmax_t)info.st_size != size || fstatfs(fd, &fs) != 0 || fs.f_type != EXT4_SUPER_MAGIC ||
            fstatvfs(fd, &mount) != 0 || (mount.f_flag & ST_RDONLY) == 0 ||
            ioctl(fd, FS_IOC_GETFLAGS, &flags) != 0 || (flags & FS_IMMUTABLE_FL) == 0)
            fail("BOOTSTRAP_IMAGE_IMMUTABILITY_REJECTED");
        if (close(fd) != 0)
            fail("BOOTSTRAP_IMAGE_CLOSE");
    }
}

#endif
