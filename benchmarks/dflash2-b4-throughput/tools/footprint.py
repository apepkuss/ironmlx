"""Observed aggregate process-tree footprint, never sum of per-PID maxima."""
import ctypes
import errno
import json
import threading
import time


class Record(ctypes.Structure):
    _fields_ = [(k, ctypes.c_int32) for k in ('pid', 'ppid', 'pgid', 'error')] + [
        (k, ctypes.c_uint64) for k in ('start', 'footprint', 'resident',
                                     'lifetime_peak', 'user_ns', 'system_ns')]


class Probe:
    def __init__(self, library, root):
        self.root = root
        self.known = {}
        self.library = ctypes.CDLL(str(library))
        self.call = self.library.b1_snapshot
        self.call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.POINTER(Record), ctypes.c_int]
        self.call.restype = ctypes.c_int

    def sample(self):
        started = time.monotonic()
        pids = list(self.known)
        known = (ctypes.c_int32 * len(pids))(*pids)
        starts = (ctypes.c_uint64 * len(pids))(*(self.known[p] for p in pids))
        records = (Record * 1024)()
        count = self.call(self.root, self.root, known, starts, len(pids), records, 1024)
        if count < 0:
            raise OSError(-count, 'libproc process-tree snapshot')
        processes, errors, exited = [], [], []
        for r in records[:count]:
            row = {key: getattr(r, key) for key, _ in Record._fields_}
            if r.error:
                (exited if r.error == errno.ESRCH else errors).append(row)
            else:
                self.known[r.pid] = r.start
                processes.append(row)
        return dict(monotonic_s=started, ended_monotonic_s=time.monotonic(),
            processes=processes, errors=errors, exited_during_sample=exited,
            tree_footprint_bytes=sum(r['footprint'] for r in processes),
            tree_resident_bytes=sum(r['resident'] for r in processes))


class FootprintCollector:
    def __init__(self, path, library, root, interval=0.25):
        self.path, self.probe, self.interval = path, Probe(library, root), interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error, self.count = None, 0

    def _run(self):
        try:
            with self.path.open('x') as destination:
                while not self.stop_event.is_set():
                    start = time.monotonic()
                    row = self.probe.sample()
                    destination.write(json.dumps(row) + '\n')
                    destination.flush()
                    self.count += 1
                    if row['errors']:
                        raise PermissionError('Selected process counter unavailable')
                    self.stop_event.wait(max(0, self.interval - (time.monotonic() - start)))
        except BaseException as error:
            self.error = type(error).__name__ + ': ' + str(error)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.error = 'Footprint collector did not stop'
        return dict(samples=self.count, error=self.error, interval_s=self.interval)


def window(rows, start, end):
    # Only complete snapshots inside the declared wrapper window, no padding.
    inside = [r for r in rows if start <= r['monotonic_s'] and r['ended_monotonic_s'] <= end]
    if not inside or any(r['errors'] or not r['processes'] for r in inside):
        raise ValueError('Missing/incomplete process-tree samples in window')
    gaps = [inside[0]['monotonic_s'] - start, end - inside[-1]['ended_monotonic_s']]
    gaps += [b['monotonic_s'] - a['monotonic_s'] for a, b in zip(inside, inside[1:])]
    peak = max(inside, key=lambda r: r['tree_footprint_bytes'])
    return dict(samples=len(inside), observed_peak_footprint_bytes=peak['tree_footprint_bytes'],
        peak_monotonic_s=peak['monotonic_s'], peak_processes=peak['processes'],
        observed_peak_resident_bytes=max(r['tree_resident_bytes'] for r in inside),
        maximum_gap_s=max(gaps),
        maximum_snapshot_duration_s=max(r['ended_monotonic_s']-r['monotonic_s'] for r in inside))


def read_samples(path, live=False):
    raw = path.read_text()
    if raw and not raw.endswith('\n'):
        if not live:
            raise ValueError('Unfinished JSONL snapshot')
        raw = raw[:raw.rfind('\n')+1]
    return [json.loads(line) for line in raw.splitlines()]
