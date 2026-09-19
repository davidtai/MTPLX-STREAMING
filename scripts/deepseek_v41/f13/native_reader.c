/*
 * native_reader.c -- F13 native burst reader prototype for the DSV4.1 Q4
 * streaming-decode reader-burst microbenchmark.
 *
 * Model: a fixed pool of N worker pthreads served by a mutex+condvar job
 * queue. One ctypes call per burst (submit_batch) enqueues all jobs and
 * broadcasts; the worker pthreads run pread() completely outside the CPython
 * GIL, so I/O starts the instant a worker wakes -- no Python thread handoff is
 * needed to reach the first read. wait_all() blocks on a completion condvar
 * (and is called through ctypes.CDLL, which releases the GIL), so the Python
 * main thread's post-submit graph-building work overlaps the reads.
 *
 * Timing: submit_batch stamps g_submit_ts (CLOCK_MONOTONIC); each worker stamps
 * just before its pread and min-reduces into g_first_pread_ts. The whole
 * submit->first-pread delay is computed in one clock domain and read back via
 * get_first_pread_delay_ns() -- the native analogue of the Python variants'
 * first-preadv-delay, and the direct test of the GIL-handoff hypothesis.
 *
 * Build (see build_native.sh):
 *   clang -O2 -Wall -shared -fPIC -o libreader.dylib native_reader.c -lpthread
 *
 * CPU-only: no MLX, no Metal, read-only pread on a caller-supplied fd.
 */
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>

typedef struct {
    int fd;             /* open, read-only descriptor (shared, thread-safe pread) */
    long long offset;   /* absolute file offset of this sub-read */
    long long length;   /* bytes to read */
    unsigned char *dest;/* destination pointer (numpy uint8 view over an mmap buf) */
    int err;            /* 0 on success, else the captured errno */
} job_t;

static pthread_mutex_t g_mtx     = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_work_cv = PTHREAD_COND_INITIALIZER; /* workers wait for jobs   */
static pthread_cond_t  g_done_cv = PTHREAD_COND_INITIALIZER; /* main waits for finish    */

static job_t     *g_jobs = NULL;   /* caller-owned array for the current batch */
static int        g_njobs = 0;
static int        g_next = 0;      /* next unclaimed job index                 */
static int        g_completed = 0; /* jobs finished in the current batch        */
static long long  g_submit_ts = 0;
static long long  g_first_pread_ts = 0;
static int        g_first_set = 0;
static int        g_nthreads = 0;
static pthread_t *g_threads = NULL;
static int        g_shutdown = 0;

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

/* Read exactly job->length bytes, looping over short reads; capture errno. */
static void do_job(job_t *j) {
    long long off = j->offset;
    long long rem = j->length;
    unsigned char *p = j->dest;
    j->err = 0;
    while (rem > 0) {
        ssize_t n = pread(j->fd, p, (size_t)rem, (off_t)off);
        if (n < 0) {
            if (errno == EINTR) continue;
            j->err = errno;
            return;
        }
        if (n == 0) { j->err = EIO; return; } /* unexpected EOF */
        p += n; off += (long long)n; rem -= (long long)n;
    }
}

static void *worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_mtx);
        while (!g_shutdown && g_next >= g_njobs)
            pthread_cond_wait(&g_work_cv, &g_mtx);
        if (g_shutdown) { pthread_mutex_unlock(&g_mtx); break; }
        int idx = g_next++;
        pthread_mutex_unlock(&g_mtx);

        long long t = now_ns();               /* stamp before entering pread */
        pthread_mutex_lock(&g_mtx);
        if (!g_first_set || t < g_first_pread_ts) {
            g_first_pread_ts = t;
            g_first_set = 1;
        }
        pthread_mutex_unlock(&g_mtx);

        do_job(&g_jobs[idx]);

        pthread_mutex_lock(&g_mtx);
        g_completed++;
        if (g_completed >= g_njobs)
            pthread_cond_signal(&g_done_cv);
        pthread_mutex_unlock(&g_mtx);
    }
    return NULL;
}

/* Spawn a fixed pool of nthreads workers. Returns 0, or -1 if already inited. */
int reader_init(int nthreads) {
    pthread_mutex_lock(&g_mtx);
    if (g_threads != NULL) { pthread_mutex_unlock(&g_mtx); return -1; }
    g_shutdown = 0;
    g_njobs = 0; g_next = 0; g_completed = 0;
    g_first_set = 0; g_first_pread_ts = 0; g_submit_ts = 0;
    g_nthreads = nthreads;
    g_threads = (pthread_t *)malloc(sizeof(pthread_t) * (size_t)nthreads);
    if (g_threads == NULL) { pthread_mutex_unlock(&g_mtx); return -2; }
    pthread_mutex_unlock(&g_mtx);
    for (int i = 0; i < nthreads; i++)
        pthread_create(&g_threads[i], NULL, worker, NULL);
    return 0;
}

/* Enqueue njobs and wake the pool. Fast; returns immediately (I/O runs on
 * the worker pthreads). jobs must stay valid until wait_all() returns. */
void submit_batch(job_t *jobs, int njobs) {
    pthread_mutex_lock(&g_mtx);
    g_jobs = jobs;
    g_njobs = njobs;
    g_next = 0;
    g_completed = 0;
    g_first_set = 0;
    g_first_pread_ts = 0;
    g_submit_ts = now_ns();
    pthread_cond_broadcast(&g_work_cv);
    pthread_mutex_unlock(&g_mtx);
}

/* Block until every job in the current batch has completed. */
void wait_all(void) {
    pthread_mutex_lock(&g_mtx);
    while (g_completed < g_njobs)
        pthread_cond_wait(&g_done_cv, &g_mtx);
    pthread_mutex_unlock(&g_mtx);
}

/* submit -> earliest-pread delay in ns for the last batch (-1 if none). */
long long get_first_pread_delay_ns(void) {
    pthread_mutex_lock(&g_mtx);
    long long d = g_first_set ? (g_first_pread_ts - g_submit_ts) : -1;
    pthread_mutex_unlock(&g_mtx);
    return d;
}

/* errno captured for job i (0 == ok), or -1 if i is out of range. */
int get_job_err(int i) {
    if (i < 0 || i >= g_njobs) return -1;
    return g_jobs[i].err;
}

/* Signal shutdown and join every worker. */
void reader_shutdown(void) {
    pthread_mutex_lock(&g_mtx);
    g_shutdown = 1;
    pthread_cond_broadcast(&g_work_cv);
    pthread_mutex_unlock(&g_mtx);
    for (int i = 0; i < g_nthreads; i++)
        pthread_join(g_threads[i], NULL);
    free(g_threads);
    g_threads = NULL;
    g_nthreads = 0;
}
