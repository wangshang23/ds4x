#include "ds4_memory.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static const uint64_t DS4_GIB = 1024ull * 1024ull * 1024ull;

bool ds4_parse_gib_arg(const char *text, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!text || !text[0] || !bytes) return false;

    size_t len = strlen(text);
    if (len > 2 && (text[len - 2] == 'g' || text[len - 2] == 'G') &&
        (text[len - 1] == 'b' || text[len - 1] == 'B')) {
        len -= 2;
    }
    if (len == 0 || len >= 32) return false;
    for (size_t i = 0; i < len; ++i) {
        if (!isdigit((unsigned char)text[i])) return false;
    }

    char number[32];
    memcpy(number, text, len);
    number[len] = '\0';
    errno = 0;
    const unsigned long long value = strtoull(number, NULL, 10);
    if (errno != 0 || value == 0 || value > UINT64_MAX / DS4_GIB) {
        return false;
    }
    *bytes = (uint64_t)value * DS4_GIB;
    return true;
}

bool ds4_memory_lock_acquire(ds4_memory_lock *lock, uint64_t bytes) {
    if (!lock) return false;
    lock->ptr = NULL;
    lock->bytes = 0;
    if (bytes == 0) return true;
    if (bytes > (uint64_t)SIZE_MAX) return false;

    void *ptr = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr, "ds4: simulated-memory mmap %.2f GiB failed: %s\n",
                (double)bytes / (double)DS4_GIB, strerror(errno));
        return false;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_size > 0 ? (uint64_t)page_size : 4096ull;
    volatile unsigned char *data = (volatile unsigned char *)ptr;
    for (uint64_t offset = 0; offset < bytes; offset += page) {
        data[offset] = (unsigned char)(offset / page);
    }
    data[bytes - 1u] = 1;
    if (mlock(ptr, (size_t)bytes) != 0) {
        fprintf(stderr, "ds4: simulated-memory mlock %.2f GiB failed: %s\n",
                (double)bytes / (double)DS4_GIB, strerror(errno));
        munmap(ptr, (size_t)bytes);
        return false;
    }

    lock->ptr = ptr;
    lock->bytes = bytes;
    fprintf(stderr, "ds4: simulated used memory: locked %.2f GiB\n",
            (double)bytes / (double)DS4_GIB);
    return true;
}

void ds4_memory_lock_release(ds4_memory_lock *lock) {
    if (!lock || !lock->ptr || lock->bytes == 0) return;
    munlock(lock->ptr, (size_t)lock->bytes);
    munmap(lock->ptr, (size_t)lock->bytes);
    lock->ptr = NULL;
    lock->bytes = 0;
}
