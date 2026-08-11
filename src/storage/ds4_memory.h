#ifndef DS4_MEMORY_H
#define DS4_MEMORY_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} ds4_memory_lock;

bool ds4_parse_gib_arg(const char *text, uint64_t *bytes);
bool ds4_memory_lock_acquire(ds4_memory_lock *lock, uint64_t bytes);
void ds4_memory_lock_release(ds4_memory_lock *lock);

#endif
