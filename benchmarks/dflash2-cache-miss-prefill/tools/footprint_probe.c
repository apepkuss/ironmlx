/* Read-only macOS SDK libproc counters; no task ports or privileges. */
#include <errno.h>
#include <libproc.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/proc_info.h>
#include <sys/resource.h>

#define LIMIT 16384
struct record {
    int32_t pid, ppid, pgid, error;
    uint64_t start, footprint, resident, lifetime_peak, user_ns, system_ns;
};

/* Membership = root + its group + recursive descendants + previously observed
 * descendants with the SAME kernel start identity (even after reparenting).
 * Individual counters are collected sequentially, not atomically. */
int b1_snapshot(int root, int group, const int32_t *known,
                const uint64_t *starts, int known_count,
                struct record *out, int capacity) {
    int *pids = calloc(LIMIT, sizeof(int));
    struct proc_bsdinfo *bsd = calloc(LIMIT, sizeof(*bsd));
    unsigned char *selected = calloc(LIMIT, 1);
    if (!pids || !bsd || !selected) { free(pids); free(bsd); free(selected); return -ENOMEM; }
    int bytes = proc_listpids(PROC_ALL_PIDS, 0, pids, LIMIT * sizeof(int));
    if (bytes <= 0 || bytes >= LIMIT * (int)sizeof(int)) {
        int result = bytes <= 0 ? -errno : -EOVERFLOW;
        free(pids); free(bsd); free(selected); return result;
    }
    int count = bytes / sizeof(int);
    for (int i = 0; i < count; i++) {
        if (pids[i] <= 0 || proc_pidinfo(pids[i], PROC_PIDTBSDINFO, 0,
                &bsd[i], sizeof(bsd[i])) != sizeof(bsd[i])) continue;
        selected[i] = pids[i] == root || bsd[i].pbi_pgid == (uint32_t)group;
        for (int j = 0; !selected[i] && j < known_count; j++) {
            if (pids[i] != known[j]) continue;
            struct rusage_info_v4 usage = {0};
            if (!proc_pid_rusage(pids[i], RUSAGE_INFO_V4, (rusage_info_t *)&usage)
                    && usage.ri_proc_start_abstime == starts[j]) selected[i] = 1;
        }
    }
    int changed;
    do {
        changed = 0;
        for (int i = 0; i < count; i++) {
            if (selected[i] || !bsd[i].pbi_pid) continue;
            for (int j = 0; j < count; j++) {
                if (selected[j] && bsd[i].pbi_ppid == (uint32_t)pids[j]) {
                    selected[i] = 1; changed = 1; break;
                }
            }
        }
    } while (changed);
    int written = 0;
    for (int i = 0; i < count; i++) {
        if (!selected[i]) continue;
        if (written == capacity) { written = -EOVERFLOW; break; }
        struct record *r = &out[written++];
        memset(r, 0, sizeof(*r));
        r->pid = pids[i]; r->ppid = bsd[i].pbi_ppid; r->pgid = bsd[i].pbi_pgid;
        struct rusage_info_v4 u = {0};
        if (proc_pid_rusage(pids[i], RUSAGE_INFO_V4, (rusage_info_t *)&u)) {
            r->error = errno; continue;
        }
        if (u.ri_proc_exit_abstime) { r->error = ESRCH; continue; }
        r->start = u.ri_proc_start_abstime; r->footprint = u.ri_phys_footprint;
        r->resident = u.ri_resident_size; r->lifetime_peak = u.ri_lifetime_max_phys_footprint;
        r->user_ns = u.ri_user_time; r->system_ns = u.ri_system_time;
    }
    free(pids); free(bsd); free(selected); return written;
}
