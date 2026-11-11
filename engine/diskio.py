"""Positional parallel reads from a file on Windows, optionally bypassing the OS cache
(FILE_FLAG_NO_BUFFERING) to measure true disk streaming."""
import ctypes, ctypes.wintypes as wt, threading
from concurrent.futures import ThreadPoolExecutor

k32 = ctypes.WinDLL('kernel32', use_last_error=True)
k32.CreateFileW.restype = wt.HANDLE
k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, wt.LPVOID, wt.DWORD, wt.DWORD, wt.HANDLE]
k32.ReadFile.argtypes = [wt.HANDLE, wt.LPVOID, wt.DWORD, ctypes.POINTER(wt.DWORD), wt.LPVOID]
k32.ReadFile.restype = wt.BOOL


class OVERLAPPED(ctypes.Structure):
    _fields_ = [('Internal', ctypes.c_size_t), ('InternalHigh', ctypes.c_size_t),
                ('Offset', wt.DWORD), ('OffsetHigh', wt.DWORD), ('hEvent', wt.HANDLE)]


class Reader:
    def __init__(self, path, nobuf=True, threads=8):
        self.path, self.flags = path, (0x20000000 if nobuf else 0) | 0x08000000  # + RANDOM_ACCESS hint
        self.tl = threading.local()
        self.pool = ThreadPoolExecutor(threads)

    def _h(self):
        h = getattr(self.tl, 'h', None)
        if h is None:
            h = k32.CreateFileW(self.path, 0x80000000, 1, None, 3, self.flags, None)
            if h in (None, wt.HANDLE(-1).value):
                raise OSError(ctypes.get_last_error())
            self.tl.h = h
        return h

    def _read(self, off, size, ptr):
        ov = OVERLAPPED(0, 0, off & 0xffffffff, off >> 32, None)
        n = wt.DWORD(0)
        if not k32.ReadFile(self._h(), ptr, size, ctypes.byref(n), ctypes.byref(ov)) or n.value != size:
            raise OSError(ctypes.get_last_error(), f'read {off} {size} got {n.value}')

    def read_many(self, jobs):
        """jobs: list of (file_offset, size, dest_ptr). Blocks until all done."""
        list(self.pool.map(lambda j: self._read(*j), jobs))


def submit_many(reader, jobs):
    """non-blocking variant: returns futures."""
    return [reader.pool.submit(reader._read, *j) for j in jobs]


class PJob:
    """a read job of the PriorityReader; result() runs it inline if no worker has started it yet (no waiting in the queue)"""
    __slots__ = ('args', 'prio', 'state', 'done', 'err', 'reader')

    def __init__(self, reader, prio, args):
        self.reader, self.prio, self.args, self.state, self.err = reader, prio, args, 0, None
        self.done = threading.Event()

    def __lt__(self, o):
        return self.prio < o.prio

    def _run(self):
        try:
            self.reader._read(*self.args)
        except Exception as ex:
            self.err = ex
        self.state = 2
        self.done.set()

    def result(self):
        with self.reader.lock:
            steal = self.state == 0
            if steal:
                self.state = 1
        if steal:
            self.reader.stolen += 1
            self._run()
        else:
            self.done.wait()
        if self.err:
            raise self.err

    def done_(self):
        return self.state == 2


class PriorityReader(Reader):
    """Reader whose background reads are served by priority (smaller = sooner) instead of first come, first served"""
    def __init__(self, path, nobuf=True, threads=8):
        super().__init__(path, nobuf=nobuf, threads=1)
        import heapq
        self.heap, self.lock, self.cv, self.stolen = [], threading.Lock(), threading.Condition(threading.Lock()), 0
        self._heapq = heapq
        for _ in range(threads):
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        hq = self._heapq
        while True:
            with self.cv:
                while not self.heap:
                    self.cv.wait()
                job = hq.heappop(self.heap)
            with self.lock:
                if job.state != 0:
                    continue
                job.state = 1
            job._run()

    def psubmit(self, jobs, prio):
        out = [PJob(self, prio, j) for j in jobs]
        with self.cv:
            for jb in out:
                self._heapq.heappush(self.heap, jb)
            self.cv.notify_all()
        return out
