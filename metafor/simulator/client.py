import random
from abc import ABC, abstractmethod
from typing import Type

from metafor.simulator.job import Job, Distribution, JobStatus, RetryOrigin, MixtureOfExponentials
from collections import deque

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
        self.retries : int = 0
        self.rho_fault : float = rho_fault
        self.rho_reset : float = rho_reset
        self.rate_tps_fault : float = rho_fault / job_type.mean()
        self.rate_tps_reset : float = rho_reset / job_type.mean()
        
        self.fault_start : float = fault_start
        self.fault_duration : float = fault_duration
        self.num_complete_jobs = 0

        self.completed_request_ids: deque[str] = deque(maxlen=1000)

        
        self._use_mixture = False
        self.it_list = []
        

    def _next_interarrival(self, t: float) -> float:
        """
        Return the next inter-arrival time, respecting fault windows.
        Works for both plain ExponentialDistribution and MixtureOfExponentials.
        """
        if isinstance(self.distribution, MixtureOfExponentials):
            # Mixture handles rho internally via set_rho()
            self.distribution.set_rho(self.rate_tps)
            return self.distribution.sample()

        else:
            # Original class-based path — unchanged behaviour
            if t < self.fault_start[0]:
                return self.distribution(self.rate_tps).sample()
            elif t < self.fault_start[0] + self.fault_duration:
                return self.distribution(self.rate_tps_fault).sample()
            else:
                return self.distribution(self.rate_tps_reset).sample()
            
            
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
        job = Job(
            self.apiname,
            t,
            self.max_retries,
            self.max_retries,
            None,     
            0,
        )
        job.client = self
        
        logger.info(" New Job %s with id %s created at %f" % (self.apiname, job.request_id, t ))

        # ---- single call replaces the branching distribution logic ----
        iat =  self._next_interarrival(t)
        next_t = t + iat
        self.it_list.append(iat)

        #assert self.server is not None, "Server is not set for client " + self.name
        
        if self.server is None:
            return
        
        # Get the Job to be processed
        offered = self.server.offer(job, t)


        events = [(next_t, self.generate, payload)]

        if offered:
            events.append((t + self.timeout, self.on_timeout, job))
            if isinstance(offered, list):
                events.extend(offered)
            else:
                events.append(offered)
                
        return events
        
    
    def on_timeout(self, t, job):
        """
        This function is invoked when a job is under processing after timeout. 
        
        Args:
            t : timeout
            job : a retry job

        Returns:
            A retry job is scheduled at timeout if job is still under processing else None.
        """
        # check by request_id, not object status
        if job.request_id in self.completed_request_ids:
            return None
        
        logger.info("Client timeout called at %f with job %s with id %s" % (t, job.name, job.request_id))
        
        ####################################################
        # DESIGN : We use independent retries per server (from the client’s perspective, 
        # there is one request.)
        
        if job.status in {JobStatus.COMPLETED, JobStatus.DROPPED, JobStatus.FORWARDED}:
            return None
        
        if job.retries_left <= 0:
            job.status = JobStatus.DROPPED
            logger.info(    
                f"Client request {job.request_id} failed after retries"
            )
            return None
        
        job.retries_left -= 1
        self.retries += 1
        
        retry = job.clone_for_retry(t)
        retry.retry_origin = RetryOrigin.CLIENT
        retry.client = self
        logger.info(
            f"Client retry {retry.request_id} attempt {retry.attempt_id}"
        )



        offered = self.server.offer(retry, t)
        events = []

        # schedule next timeout
        events.append((t + self.timeout, self.on_timeout, retry))

        if offered:
            if isinstance(offered, list):
                events.extend(offered)
            else:
                events.append(offered)

        return events


    def on_complete(self, t, job):
        
        if job.request_id in self.completed_request_ids:
            logger.info(f"Duplicate on_complete ignored for {job.request_id}")
            return None
        
        job.status = JobStatus.COMPLETED
        job.completed_t = t

        latency = t - job.created_t
        self.completed_request_ids.append(job.request_id)
        self.num_complete_jobs += 1

        logger.info(
            f"Client completed request {job.request_id} "
            f"latency={latency:.4f}"
        )

        return None


