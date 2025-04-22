# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import random
from abc import ABC, abstractmethod
from typing import Type

from Job import Job


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
    def __init__(self, rho: float, job_type: Type[Job], timeout: float, max_retries: int,
                 rho_fault: float, rho_reset: float, fault_start, fault_duration):
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
    def __init__(self, rho: float, job_type: Type[Job], timeout: float,
                 max_retries: int, rho_fault: float, rho_reset: float,
                 fault_start: float, fault_duration: float):
        super().__init__(rho, job_type, timeout, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
        self.timeout = timeout
        self.max_retries: int = max_retries
        self.rho_fault : float = rho_fault
        self.rho_reset : float = rho_reset
        self.rate_tps_fault : float = rho_fault / job_type.mean()
        self.rate_tps_reset : float = rho_reset / job_type.mean()
        self.fault_start : float = fault_start
        self.fault_duration : float = fault_duration


    def generate(self, t: float, payload):
        job = self.job_type(t, self.max_retries, self.max_retries)
        if t < self.fault_start[0]: # must be modified when there are more instances of faults
            next_t = t + random.expovariate(self.rate_tps)
        elif t >= self.fault_start[0] and t < self.fault_start[0] + self.fault_duration:
            next_t = t + random.expovariate(self.rate_tps_fault)
        elif t >= self.fault_start[0] + self.fault_duration:
            next_t = t + random.expovariate(self.rate_tps_reset)
        offered = self.server.offer(job, t)
        if offered is None:
            return [(next_t, self.generate, None)]
        else:
            return [(next_t, self.generate, None), offered]

    def done(self, t: float, event: Job):
        if t - event.created_t > self.timeout and event.retries_left > 0:
            # Offer another job as a replacement for the timed-out one
            return self.server.offer(self.job_type(t, self.max_retries, event.retries_left - 1), t)
        else:
            return None
