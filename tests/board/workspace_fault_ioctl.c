#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>

struct __attribute__((packed)) gem_create {
    uint32_t handle, flags;
    uint64_t size, obj_addr, dma_addr, sram_size;
    int32_t domain;
    uint32_t core_mask;
};

static _Thread_local int mode, remaining, target, injected;
static _Thread_local unsigned long long request_size;

void workspace_fault_arm(int phase, int count, int domain) {
    mode = phase;
    remaining = count;
    target = domain;
    injected = 0;
    request_size = 0;
}

int workspace_fault_disarm(void) {
    int count = injected;
    mode = remaining = injected = 0;
    return count;
}

static int is_attention_graph(void) {
    void* frames[24];
    int count = backtrace(frames, 24);
    for (int i = 1; i < count; ++i) {
        Dl_info info;
        if (dladdr(frames[i], &info) && info.dli_sname &&
            strstr(info.dli_sname, "attention5GraphC"))
            return 1;
    }
    return 0;
}

int ioctl(int fd, unsigned long request, ...) {
    va_list args;
    va_start(args, request);
    void* arg = va_arg(args, void*);
    va_end(args);
    if (remaining && request == _IOWR('d', 0x42, struct gem_create)) {
        const struct gem_create* gem = arg;
        if (!gem->handle && gem->domain == target &&
            (mode == 1 || is_attention_graph()) &&
            (!request_size || gem->size == request_size)) {
            request_size = gem->size;
            --remaining;
            ++injected;
            fprintf(stderr, "injected GEM ENOMEM phase=%d domain=%d size=%llu attempt=%d\n",
                    mode, target, request_size, injected);
            errno = ENOMEM;
            return -1;
        }
    }
    int (*real_ioctl)(int, unsigned long, ...) = dlsym(RTLD_NEXT, "ioctl");
    return real_ioctl(fd, request, arg);
}
