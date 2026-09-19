"""Fixed four-worker demand-first queue for the isolated packed reader.

Priority controls queued work only; active positional reads finish normally.
The reader retains writer joins and memoryview ownership through completion.
"""
from concurrent.futures import Future
from itertools import count
from queue import PriorityQueue
import threading


class PriorityReads:
    def __init__(self):
        self.queue=PriorityQueue()
        self.sequence=count()
        self.lock=threading.Lock()
        self.closed=False
        self.threads=tuple(threading.Thread(target=self.work,name=f'packed-read-{i}',daemon=True) for i in range(4))
        for thread in self.threads:thread.start()

    def submit(self,fn,job,*,priority):
        future=Future()
        with self.lock:
            if self.closed:raise RuntimeError('read queue is closed')
            self.queue.put((priority,next(self.sequence),(future,fn,job)))
        return future

    def work(self):
        while True:
            _,_,task=self.queue.get()
            if task is None:
                self.queue.task_done()
                return
            future,fn,job=task
            try:
                if future.set_running_or_notify_cancel():
                    try:result=fn(job)
                    except BaseException as error:future.set_exception(error)
                    else:
                        future.set_result(result)
                        result=None
            finally:
                future=fn=job=task=None
                self.queue.task_done()

    def shutdown(self,wait=True):
        with self.lock:
            if not self.closed:
                self.closed=True
                for _ in self.threads:self.queue.put((2,next(self.sequence),None))
        if wait:
            for thread in self.threads:thread.join()
