# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

# First-Come-First-Served Queue
import time
from collections import deque
from typing import List, TextIO

from Client import Client
from Job import Job


class FCFSQueue:
    def __init__(self):
        self.deque = deque()

    def append(self, job):
        self.deque.append(job)

    def pop(self):
        return self.deque.popleft()

    def len(self) -> int:
        return len(self.deque)

    @staticmethod
    def name() -> str:
        return "FCFS"


# Server that consumes a queue of tasks of a fixed size (`queue_size`), with a fixed concurrency (MPL)
class Server:
    def __init__(self, mpl: int, sim_name: str, client: Client, rho: float, queue_size: int,
                 retry_queue_size: int):
        self.start_time = 0  # to be set by each simulation
        self.busy: int = 0
        self.queue: FCFSQueue = FCFSQueue()
        self.sim_name: str = sim_name
        self.mpl: int = mpl
        self.jobs: List[Job | None] = [None for _ in range(mpl)]
        self.client: Client = client
        self.rho: float = rho
        self.queue_size: int = queue_size
        self.retry_queue_size: int = retry_queue_size
        self.file: TextIO | None = None
        self.retries: int = 0
        self.dropped: int = 0  # cumulative number

    def job_done(self, t: float, n: int) -> List:
        assert (self.busy > 0)
        completed = self.jobs[n]
        if completed.max_retries > completed.retries_left:  # a retried job is completed
            self.retries -= 1

        end_time = time.time()
        runtime = end_time - self.start_time
        self.file.write("%f,%f,%f,%s,%d,%d,%d,%f\n" % (t, self.rho, t - completed.created_t, self.sim_name,
                                                       self.queue.len(), self.retries, self.dropped, runtime))

        events = []
        if self.queue.len() > 0:
            next_job = self.queue.pop()
            self.jobs[n] = next_job
            events = [(t + next_job.size, self.job_done, n)]
        else:
            self.busy -= 1
            self.jobs[n] = None

        done_event = self.client.done(t, completed)
        if done_event is not None:
            events.append(done_event)

        return events

    def offer(self, job: Job, t: float) -> (float, List, int):
        if job.max_retries > job.retries_left and self.queue.len() < self.queue_size:
            if self.retries < self.retry_queue_size:
                self.retries += 1
            else:
                self.retries += 1
                # self.dropped += 1  # there is not enough space in the virtual retries queue
                #return None

        if self.busy < self.mpl:
            # The server isn't entirely busy, so we can start on the job immediately
            self.busy += 1
            for i in range(self.mpl):
                if self.jobs[i] is None:
                    self.jobs[i] = job
                    return t + job.size, self.job_done, i
            # Should never get here because jobs slots should always be available if busy < mpl
            assert False
        else:
            # The server is busy, so try to enqueue the job, if there is enough space in the queue
            if self.queue.len() < self.queue_size:
                self.queue.append(job)
            else:
                self.dropped += 1
            return None
