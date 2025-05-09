# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import random
from abc import ABC
from typing import Type


import logging
logger = logging.getLogger(__name__)


class Distribution(ABC):
    def __init__(self):
        pass

    def sample(self) -> float:
        return 0

class ExponentialDistribution(Distribution):
    def __init__(self, mean: float):
        self.mean = mean

    def sample(self) -> float:
        return random.expovariate(self.mean)


class JobStatus:
    CREATED = 0
    ENQUEUED = 1
    PROCESSING = 2
    COMPLETED = 3
    DROPPED = 4

    @staticmethod
    def __str__(status):
        m = { JobStatus.CREATED: 'created', 
             JobStatus.ENQUEUED: 'enqueued',
             JobStatus.PROCESSING: 'processing',
             JobStatus.COMPLETED: 'completed',
             JobStatus.DROPPED: 'dropped'
             }
        return m[status]


class Job(ABC):
    def __init__(self, name: str, timestamp: float, max_retries: int = 0, retries_left: int = 0):
        self.created_t: float = timestamp
        self.completed_t: float = 0.0
        self.name = name
        self.status = JobStatus.CREATED
        self.max_retries: int = max_retries
        self.retries_left: int = retries_left
        self.size: float = 0

    def __str__(self):
        return "[%s: created %f, status: %s]" % (self.name, self.created_t, JobStatus.__str__(self.status))
    
    @staticmethod
    def mean() -> float:
        pass



# Job with unimodal exponentially distributed latency
def exp_job(mean: float) -> Type[Job]:
    class ExpJob(Job):
        def __init__(self, t: float, max_retries: int = 0, retries_left: int = 0):
            super().__init__(t, max_retries, retries_left)
            self.size = random.expovariate(1.0 / mean)

        @staticmethod
        def mean() -> float:
            return mean

    return ExpJob


# Job with bimodal exponentially distributed latency
def bimod_job(mean_1: float, mean_2: float, p: float) -> Type[Job]:
    class BiModJob(Job):
        def __init__(self, t: float, max_retries: int = 0, retries_left: int = 0):
            super().__init__(t, max_retries, retries_left)
            if random.random() > p:
                self.size = random.expovariate(1.0 / mean_1)
            else:
                self.size = random.expovariate(1.0 / mean_2)

        @staticmethod
        def mean() -> float:
            return (1.0 - p) * mean_1 + p * mean_2

    return BiModJob
