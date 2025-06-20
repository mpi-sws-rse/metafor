# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import random
from abc import ABC, abstractmethod
from typing import Type

from metafor.simulator.job import Job, Distribution, JobStatus

import logging
logger = logging.getLogger(__name__)

class Client(ABC):

    def __init__(self, name: str, apiname: str, distribution: Distribution, rho: float, job_type: Type[Job]):
        self.name = name
        self.apiname = apiname
        self.distribution: float = distribution
        self.rate_tps: float = rho / job_type.mean()
        self.job_type: Type[Job] = job_type
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
    def __init__(self, name: str, apiname: str, distribution: Distribution, rho: float, job_type: Type[Job]):
        super().__init__(name, apiname, distribution, rho, job_type)

    def generate(self, t: float, payload):
        job = self.job_type(t)
        next_t = t + self.distribution(self.rate_tps).sample()
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
    """
    Function implementing Client api with retries and timeouts.
    If the job is not complete before $timeout$ elapses, it is retried $max_retries$ number
    of times.
    We use two modes - Normal mode and Fault mode.
    In the current setup, the server utlization rate is set to rho during inital phase and 
    the late phase of the simulation (Normal mode). In between, a request storm occurs which
    is modelled using a fault rate and fault duration capturing the intensity and duration
    of the fault (Fault mode). 
    
    """
    def __init__(self, name: str, apiname: str, distribution: Distribution, 
                 rho: float, job_type: Type[Job], timeout: float, max_retries: int,
                 rho_fault: float, rho_reset: float, fault_start: float, fault_duration: float):
        super().__init__(name, apiname, distribution, rho, job_type)
        self.timeout = timeout
        self.max_retries: int = max_retries
        self.rho_fault : float = rho_fault
        self.rho_reset : float = rho_reset
        self.rate_tps_fault : float = rho_fault / job_type.mean()
        self.rate_tps_reset : float = rho_reset / job_type.mean()
        self.fault_start : float = fault_start
        self.fault_duration : float = fault_duration


    def generate(self, t: float, payload=None):
        """
        This function is invoked when client requests a new job (job creation).
        For normal mode, the arrival rate of a new job is $rho$. For
        fault mode, the arrival rate is given by $rho_fault$. Higher rate 
        signify smaller intervals between job arrival, leading to incrase in 
        number of jobs arriving into the system.     

        Args:
            t : timeout

        Returns:
            A new job if no job is currently being processed by the server.
            If a job is already being processed by the server, then a retry 
            job is also scheduled at timeout.
        """
        job = Job(name=self.apiname, timestamp=t, max_retries=self.max_retries, retries_left=self.max_retries)
        
        logger.info("Job %f %s" % (t, self.apiname))

        if t < self.fault_start[0]: # must be modified when there are more instances of faults
            next_t = t + self.distribution(self.rate_tps).sample()
            #logger.info(" client rate %f   arrival   %f" % (self.rate_tps,self.distribution(self.rate_tps).sample()))
        elif t >= self.fault_start[0] and t < self.fault_start[0] + self.fault_duration:
            next_t = t + self.distribution(self.rate_tps_fault).sample()
        elif t >= self.fault_start[0] + self.fault_duration:
            next_t = t + self.distribution(self.rate_tps_reset).sample()


        assert self.server is not None, "Server is not set for client " + self.name
        offered = self.server.offer(job, t)
        if offered is None:
            return [(next_t, self.generate, None)]
        else:
            return [(next_t, self.generate, None), (t + self.timeout, self.on_timeout, job), offered]

    def on_timeout(self, t, job):
        """
        This function is invoked when a job is under processing after timeout. 
        
        Args:
            t : timeout
            job : a retry job

        Returns:
            A retry job is scheduled at timeout if job is still under processing else None.
        """
        logger.info("on timeout called at %f with job %s" % (t, job))
        if job.status != JobStatus.COMPLETED:
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
        """
        This function is invoked when a job is completed.
        """
        # job completed
        #if t - event.created_t > self.timeout and event.retries_left > 0:
        #    # Offer another job as a replacement for the timed-out one
        #    return self.server.offer(self.job_type(t, self.max_retries, event.retries_left - 1), t)
        #else:
        event.status = JobStatus.COMPLETED
        return None
