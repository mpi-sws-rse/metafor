# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import random
from abc import ABC, abstractmethod
from typing import Type

from simulator.job import Job


class Client(ABC):

    def __init__(self, rho: float, job_type: Type[Job]):
        self.rate_tps: float = rho / job_type.mean()
        self.job_type: Type[Job] = job_type
        self.server = None

    @abstractmethod
    def generate(self, t: float, payload: float):
        pass

    @abstractmethod
    def done(self, t: float, event: Job):
        pass


# Open loop load generation client. Creates an unbounded concurrency
class OpenLoopClient(Client):
    def __init__(self, rho: float, job_type: Type[Job]):
        super().__init__(rho, job_type)

    def generate(self, t: float, payload):
        job = self.job_type(t)
        next_t = t + random.expovariate(self.rate_tps)
        offered = self.server.offer(job, t)
        if offered is None:
            return [(next_t, self.generate, None)]
        else:
            return [(next_t, self.generate, None), offered]

    def done(self, t: float, event: Job):
        return None


# Open loop load generation client, with timeout. Creates an unbounded concurrency
class OpenLoopClientWithTimeout(OpenLoopClient):
    def __init__(
        self, rho: float, job_type: Type[Job], timeout: float, max_retries: int
    ):
        super().__init__(rho, job_type)
        self.timeout = timeout
        self.max_retries: int = max_retries

    def generate(self, t: float, payload):
        job = self.job_type(t, self.max_retries, self.max_retries)
        next_t = t + random.expovariate(self.rate_tps)
        offered = self.server.offer(job, t)
        if offered is None:
            return [(next_t, self.generate, None)]
        else:
            return [(next_t, self.generate, None), offered]

    def done(self, t: float, event: Job):
        if t - event.created_t > self.timeout and event.retries_left > 0:
            # Offer another job as a replacement for the timed-out one
            return self.server.offer(
                self.job_type(t, self.max_retries, event.retries_left - 1), t
            )
        else:
            return None
