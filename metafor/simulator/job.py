# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import random
from abc import ABC
from typing import Type, List

import logging
logger = logging.getLogger(__name__)
import uuid

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


class WeibullDistribution(Distribution):
    def __init__(self, mean: float):
        self.mean = mean

    def sample(self) -> float:
        return random.weibullvariate(1.0/self.mean,1.0)


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
    def __init__(self, name: str, timestamp: float, max_retries: int = 0, retries_left: int = 0,
                 request_id: str | None = None, attempt_id: int = 0):
        self.created_t: float = timestamp
        self.completed_t: float = 0.0
        self.name = name
        self.status = JobStatus.CREATED
        self.max_retries: int = max_retries
        self.retries_left: int = retries_left
        self.size: float = 0
        self.request_id = request_id or str(uuid.uuid4())
        self.attempt_id = attempt_id

    def __str__(self):
        return "[%s: created %f, status: %s]" % (self.name, self.created_t, JobStatus.__str__(self.status))
    
    @staticmethod
    def mean() -> float:
        pass
    

    def clone_for_branch(self, t:float) -> "Job":
        """
        Create a new Job instance for a DAG branch.
        Shares request identity, but has independent execution state.
        """
        new_job = self.__class__(
            name=self.name,
            timestamp=self.created_t,
            max_retries=self.max_retries,
            retries_left=self.retries_left,
        )

        # Reset execution-specific fields
        new_job.status = JobStatus.CREATED
        new_job.created_t = self.created_t
        new_job.size = 0.0  # must be resampled by server

        # Logical request identity
        new_job.request_id = self.request_id

        # New attempt for branch
        new_job.attempt_id = self.attempt_id + 1

        return new_job



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


# Job with Weibull distributed latency
def wei_job(mean: float) -> Type[Job]:
    class WeiJob(Job):
        def __init__(self, t: float, max_retries: int = 0, retries_left: int = 0):
            super().__init__(t, max_retries, retries_left)
            self.size = random.weibullvariate(mean, 1.0)

        @staticmethod
        def mean() -> float:
            return mean

    return WeiJob


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
