#ifndef IRIS_PLATFORM_H
#define IRIS_PLATFORM_H

/* Portability macro for __attribute__((unused)) */
#ifdef __GNUC__
#define IRIS_UNUSED __attribute__((unused))
#else
#define IRIS_UNUSED
#endif

/* Iris platform abstraction layer.
 * Wraps POSIX APIs used by the codebase so they compile on Windows.
 * On non-Windows, this header simply includes the standard POSIX headers
 * and defines no-op inline wrappers. */

#ifdef _WIN32

/* Allow POSIX names like read/write/close without deprecation warnings */
#if defined(_MSC_VER) && !defined(_CRT_NONSTDC_NO_DEPRECATE)
#define _CRT_NONSTDC_NO_DEPRECATE 1
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <io.h>
#include <process.h>
#include <direct.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* strcasecmp / strdup                                                */
/* ------------------------------------------------------------------ */
#define iris_strcasecmp _stricmp
/* MSVC calls it _strdup; define the POSIX name only for our files */
#if defined(_MSC_VER) && !defined(strdup)
#define strdup _strdup
#endif

/* POSIX standard file descriptor constants */
#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
#define STDERR_FILENO 2
#endif

/* isatty → _isatty on MSVC */
#if defined(_MSC_VER)
#define isatty _isatty
#endif

/* strcasecmp — POSIX → MSVC */
#if defined(_MSC_VER) && !defined(strcasecmp)
#define strcasecmp _stricmp
#endif

/* unlink — POSIX → MSVC */
#if defined(_MSC_VER) && !defined(unlink)
#define unlink _unlink
#endif

/* EAGAIN / EWOULDBLOCK — POSIX errno codes */
#ifndef EAGAIN
#define EAGAIN    11
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK 140
#endif

/* ------------------------------------------------------------------ */
/* File / path helpers                                                */
/* ------------------------------------------------------------------ */

static inline int iris_dir_exists(const char *path) {
    DWORD attr = GetFileAttributesA(path);
    return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
}

/* Return 1 if path exists (file or dir) */
static inline int iris_path_exists(const char *path) {
    DWORD attr = GetFileAttributesA(path);
    return attr != INVALID_FILE_ATTRIBUTES;
}

static inline const char *iris_get_temp_dir(void) {
    static char buf[MAX_PATH];
    DWORD n = GetTempPathA(MAX_PATH, buf);
    if (n == 0 || n > MAX_PATH) return "C:\\Windows\\Temp";
    /* Strip trailing backslash if present */
    size_t len = strlen(buf);
    if (len > 0 && buf[len-1] == '\\') buf[len-1] = '\0';
    return buf;
}

/* mkdtemp emulation: create a temp directory from a template ending in XXXXXX.
 * The template is modified in place on success (just like POSIX mkdtemp). */
static inline int iris_mkdtemp(char *template) {
    size_t len = strlen(template);
    if (len < 6 || strcmp(template + len - 6, "XXXXXX") != 0) { errno = EINVAL; return -1; }
    /* Replace XXXXXX with a random 6-char hex string */
    const char *hex = "0123456789abcdef";
    for (int i = 0; i < 6; i++) {
        template[len - 6 + i] = hex[rand() & 0x0F];
    }
    if (CreateDirectoryA(template, NULL)) return 0;
    /* If name collides, retry a few times */
    for (int attempt = 0; attempt < 10; attempt++) {
        for (int i = 0; i < 6; i++) {
            template[len - 6 + i] = hex[rand() & 0x0F];
        }
        if (CreateDirectoryA(template, NULL)) return 0;
    }
    errno = EEXIST;
    return -1;
}

/* mkstemps emulation: create a temp file from a template ending in XXXXXXsuffix.
 * Returns fd (but we only need the name, so return a valid fd to /dev/null-ish). */
static inline int iris_mkstemps(char *template, int suffix_len) {
    size_t len = strlen(template);
    int base = (int)len - 6 - suffix_len;
    if (base < 0 || strncmp(template + base, "XXXXXX", 6) != 0) { errno = EINVAL; return -1; }
    const char *hex = "0123456789abcdef";
    char path[MAX_PATH];
    for (int attempt = 0; attempt < 20; attempt++) {
        for (int i = 0; i < 6; i++) {
            template[base + i] = hex[rand() & 0x0F];
        }
        strcpy(path, template);
        HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_NEW,
                               FILE_ATTRIBUTE_NORMAL, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
            return 1; /* caller only needs the name, fd-like return */
        }
    }
    errno = EEXIST;
    return -1;
}

static inline int iris_unlink(const char *path) {
    return DeleteFileA(path) ? 0 : -1;
}

/* ------------------------------------------------------------------ */
/* mmap abstraction                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    HANDLE hFile;
    HANDLE hMap;
    void  *data;
    size_t size;
} iris_mmap_t;

static inline int iris_mmap_open(const char *path, iris_mmap_t *out) {
    memset(out, 0, sizeof(*out));
    out->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (out->hFile == INVALID_HANDLE_VALUE) return -1;

    LARGE_INTEGER li;
    if (!GetFileSizeEx(out->hFile, &li)) { CloseHandle(out->hFile); return -1; }
    out->size = (size_t)li.QuadPart;

    out->hMap = CreateFileMappingA(out->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!out->hMap) { CloseHandle(out->hFile); return -1; }

    out->data = MapViewOfFile(out->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!out->data) { CloseHandle(out->hMap); CloseHandle(out->hFile); return -1; }
    return 0;
}

static inline void iris_mmap_close(iris_mmap_t *m) {
    if (m->data) UnmapViewOfFile(m->data);
    if (m->hMap) CloseHandle(m->hMap);
    if (m->hFile && m->hFile != INVALID_HANDLE_VALUE) CloseHandle(m->hFile);
    memset(m, 0, sizeof(*m));
}

/* Portable unmap: works with just data pointer and size (Windows ignores size) */
static inline void iris_munmap(void *data, size_t size) {
    (void)size;
    if (data) UnmapViewOfFile(data);
}

/* ------------------------------------------------------------------ */
/* Timing                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    double sec;
    double usec;
} iris_timeval_t;

static inline int iris_gettimeofday(iris_timeval_t *tv) {
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    double total_sec = (double)now.QuadPart / (double)freq.QuadPart;
    tv->sec = floor(total_sec);
    tv->usec = (total_sec - tv->sec) * 1000000.0;
    return 0;
}

static inline double iris_time_ms(void) {
    iris_timeval_t tv;
    iris_gettimeofday(&tv);
    return tv.sec * 1000.0 + tv.usec / 1000.0;
}

/* clock_gettime(CLOCK_MONOTONIC, ...) replacement */
typedef struct {
    double tv_sec;
    double tv_nsec;
} iris_timespec_t;

static inline int iris_clock_monotonic(iris_timespec_t *ts) {
    iris_timeval_t tv;
    iris_gettimeofday(&tv);
    ts->tv_sec = tv.sec;
    ts->tv_nsec = tv.usec * 1000.0;
    return 0;
}

/* Windows has sysconf(_SC_NPROCESSORS_ONLN) as an MSVC CRT extension */
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(__cpuid)
#endif

/* ------------------------------------------------------------------ */
/* Thread abstraction (replacement for pthreads on Windows)           */
/* ------------------------------------------------------------------ */

typedef HANDLE iris_thread_t;

static inline int iris_thread_create(iris_thread_t *thread,
                                     unsigned (__stdcall *func)(void*),
                                     void *arg) {
    *thread = (HANDLE)_beginthreadex(NULL, 0, func, arg, 0, NULL);
    return (*thread != NULL) ? 0 : -1;
}

static inline int iris_thread_join(iris_thread_t thread) {
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

/* ------------------------------------------------------------------ */
/* getopt_long emulation (minimal)                                    */
/* ------------------------------------------------------------------ */

/* We provide a minimal getopt_long implementation for Windows.
 * The option table is built by parsing a struct option array in main.c.
 * This is kept intentionally minimal — only what main.c uses. */

#define no_argument       0
#define required_argument 1
#define optional_argument 2

typedef struct {
    const char *name;
    int has_arg;
    int *flag;
    int val;
} iris_option_t;

extern int   iris_optind;
extern int   iris_optopt;
extern char *iris_optarg;

int iris_getopt_long(int argc, char * const argv[],
                     const char *shortopts,
                     const iris_option_t *longopts,
                     int *longindex);

#else /* !_WIN32 — POSIX / macOS / Linux */

/* ------------------------------------------------------------------ */
/* Native POSIX includes                                              */
/* ------------------------------------------------------------------ */

#define _GNU_SOURCE
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <time.h>
#include <pthread.h>
#include <dirent.h>
#include <getopt.h>   /* available on macOS+Linux, including in mingw-w64 */

/* ------------------------------------------------------------------ */
/* Aliases — all map to the POSIX originals                           */
/* ------------------------------------------------------------------ */

#define iris_strcasecmp strcasecmp

/* Portable unmap wrapper */
static inline void iris_munmap(void *data, size_t size) {
    munmap(data, size);
}

/* mmap — native POSIX, no wrapper needed (callers use sys/mman.h directly) */

static inline const char *iris_get_temp_dir(void) {
    return "/tmp";
}

/* mkdtemp / mkstemps — native POSIX */
static inline int iris_mkdtemp(char *template) { return mkdtemp(template) ? 0 : -1; }
static inline int iris_mkstemps(char *template, int suffix_len) {
    return mkstemps(template, suffix_len);
}
static inline int iris_unlink(const char *path) { return unlink(path); }

/* Timing — native POSIX */
typedef struct timeval  iris_timeval_t;
typedef struct timespec iris_timespec_t;

static inline int iris_gettimeofday(iris_timeval_t *tv) {
    return gettimeofday(tv, NULL);
}

static inline double iris_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static inline int iris_clock_monotonic(iris_timespec_t *ts) {
    return clock_gettime(CLOCK_MONOTONIC, ts);
}

/* Threads — native pthreads */
typedef pthread_t iris_thread_t;

static inline int iris_thread_create(iris_thread_t *thread,
                                     void *(*func)(void*),
                                     void *arg) {
    return pthread_create(thread, NULL, func, arg);
}

static inline int iris_thread_join(iris_thread_t thread) {
    return pthread_join(thread, NULL);
}

/* getopt_long — native */
typedef struct option iris_option_t;
#define iris_getopt_long getopt_long

extern int optind;
extern int optopt;
extern char *optarg;

/* Path helpers */
static inline int iris_dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static inline int iris_path_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

#endif /* _WIN32 */

#endif /* IRIS_PLATFORM_H */
