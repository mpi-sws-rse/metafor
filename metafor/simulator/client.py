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
        job = Job(
            name=self.apiname, 
            timestamp=t, 
            max_retries=self.max_retries, 
            retries_left=self.max_retries,
            request_id=None,      # auto-generated
            attempt_id=0,
            )
        
        logger.info(" New Job %s with id %s created at %f" % (self.apiname, job.request_id, t ))

        if t < self.fault_start[0]: # must be modified when there are more instances of faults
            next_t = t + self.distribution(self.rate_tps).sample()
            #logger.info(" client rate %f   arrival   %f" % (self.rate_tps,self.distribution(self.rate_tps).sample()))
        elif t >= self.fault_start[0] and t < self.fault_start[0] + self.fault_duration:
            next_t = t + self.distribution(self.rate_tps_fault).sample()
        elif t >= self.fault_start[0] + self.fault_duration:
            next_t = t + self.distribution(self.rate_tps_reset).sample()


        assert self.server is not None, "Server is not set for client " + self.name
        
        if self.server is None:
            return
        
        # Get the Job to be processed
        offered = self.server.offer(job, t)
        
        if offered is None:
            # No Job is to be processed
            return [(next_t, self.generate, payload)]
        else:
            return [(next_t, self.generate, payload), (t + self.timeout, self.on_timeout, job), offered]

    def on_timeout(self, t, job):
        """
        This function is invoked when a job is under processing after timeout. 
        
        Args:
            t : timeout
            job : a retry job

        Returns:
            A retry job is scheduled at timeout if job is still under processing else None.
        """
        logger.info("on timeout called at %f with job %s with id %s" % (t, job.name, job.request_id))
        
        ####################################################
        # Modifying on_timeout to check all servers in the chain for the 
        # job’s status, as jobs may be in the queue or processing in
        # downstream servers.
        # DESIGN : We use a single, fixed retry budget per job (end-to-end), 
        # not independent retries per server (from the client’s perspective, 
        # there is one request.)
        
        retried_jobs = []
       

        if job.status in {JobStatus.COMPLETED, JobStatus.DROPPED}:
            return None
        
        if job.retries_left <= 0:
            job.status = JobStatus.DROPPED
            return None
        
        # DESIGN : A retry is schedules only at the current server and the downstream servers
        #if job.status != JobStatus.COMPLETED:
        if job.retries_left > 0:
            logger.info(f"Job {job.created_t} with id {job.request_id} timeout, retrying {job.max_retries - job.retries_left} with attempt {job.attempt_id} at {t} on server {self.server.id}")
            job.retries_left -= 1
            job.attempt_id += 1
            job.status = JobStatus.CREATED
            
            job_copy = job.clone_for_branch(t)
            offered = self.server.offer(job_copy, t)  # Retry at first server immediately
            if offered is not None:
                retried_jobs.append(offered)
                retried_jobs.append((t + self.timeout, self.on_timeout, job_copy))
                # below downstreaming jobs are handled in job_done in server
                # for ds in self.server.downstream_server:
                #     offered = ds.offer(job, t)  # Retry at downstream servers
                #     if offered is not None:
                #         retried_jobs.append(offered)
                
                    
           
            #return retried_jobs
            #t = t + service_time
        return retried_jobs
    
        
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
