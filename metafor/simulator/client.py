# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import random
from abc import ABC, abstractmethod
from typing import Type

from metafor.simulator.job import Job, Distribution, JobStatus


import logging
logger = logging.getLogger(__name__)

class Client(ABC):

    def __init__(self, name: str, apiname: str, distribution: Distribution):
        self.name = name
        self.apiname = apiname
        self.distribution: float = distribution
        self.server = None

    def set_server(self, s):
        self.server = s

    # set new distribution
    def set_distribution(self, d: Distribution):
        self.distribution = d


    @abstractmethod
    def generate(self, t: float, payload: float):
        pass

    @abstractmethod
    def done(self, t: float, event: Job):
        pass

    def print(self):
        print("Client ", self.name, " api ", self.apiname)


# Open loop load generation client. Creates an unbounded concurrency
class OpenLoopClient(Client):
    def __init__(self, name: str, apiname: str, distribution: Distribution):
        super().__init__(name, apiname, distribution)

    def generate(self, t: float, payload):
        job = Job(t)
        next_t = t + self.distribution.sample()
        assert self.server is not None, "Server is not set for client " + self.name
        offered = self.server.offer(job, t)
        if offered is None:
            return [(next_t, self.generate, None)]
        else:
            return [(next_t, self.generate, None), offered]

    def done(self, t: float, event: Job):
        return None


# Open loop load generation client, with timeout. Creates an unbounded concurrency
class OpenLoopClientWithTimeout(OpenLoopClient):
    def __init__(self, name: str, apiname: str, distribution: Distribution, timeout: float, max_retries: int):
        super().__init__(name, apiname, distribution)
        self.timeout = timeout
        self.max_retries: int = max_retries



    def generate(self, t: float, payload=None):
        job = Job(name=self.apiname, timestamp=t, max_retries=self.max_retries, retries_left=self.max_retries)
        logger.info("Job %f %s" % (t, self.apiname))
        next_t = t + self.distribution.sample()
        assert self.server is not None, "Server is not set for client " + self.name

        offered = self.server.offer(job, t)
        if offered is None:
            return [(next_t, self.generate, None)]
        else:
            return [(next_t, self.generate, None), (t + self.timeout, self.on_timeout, job), offered]

    def on_timeout(self, t, job):
        logger.info("on timeout called at %f with job %s" % (t, job))
        if job.status != JobStatus.COMPLETED and job.status != JobStatus.DROPPED:
            # we have timed out: generate another instance, if retries left

            if job.retries_left > 0:
                logger.info("Job %f timeout, retrying %d at %f " % (job.created_t, job.max_retries - job.retries_left, t))
                job.retries_left -= 1
                offered = self.server.offer(job, t)
                if offered is None:
                    return []
                else:
                    return [(t + self.timeout, self.on_timeout, job), offered]
        else:
            # this job has been completed or dropped already
            return []
        
    def done(self, t: float, event: Job):
        # job completed
        #if t - event.created_t > self.timeout and event.retries_left > 0:
        #    # Offer another job as a replacement for the timed-out one
        #    return self.server.offer(self.job_type(t, self.max_retries, event.retries_left - 1), t)
        #else:
        event.status = JobStatus.COMPLETED
        return None
