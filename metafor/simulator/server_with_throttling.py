# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import math
import numpy as np
import pandas
import time
from collections import deque
from typing import List, TextIO, Optional

from metafor.simulator.client_multi import Client, OpenLoopClientWithTimeout
from metafor.simulator.job import Distribution, Job, JobStatus
from metafor.simulator.server_multi import Server

import logging
logger = logging.getLogger(__name__)


class ServerWithThrottling(Server):
    """
    Server that consumes a queue of tasks of a fixed size (`queue_size`), with a fixed concurrency (MPL)    
    """

    def __init__(
            self, 
            id : int,
            name: str, 
            queue_size: int, 
            thread_pool: int,
            service_time_distribution: dict[str, Distribution],
            retry_queue_size: int, 
            client: OpenLoopClientWithTimeout, 
            throttle:bool, 
            ts:float, 
            ap:float,
            downstream_server: Optional['Server']
    ):
            super().__init__(
                id,
                name, 
                queue_size, 
                thread_pool,
                service_time_distribution,
                retry_queue_size, 
                client,
                downstream_server
            )
            self.throttle=throttle
            self.ts = ts
            self.ap = ap

        


    def offer(self, job: Job, t: float):
        """
        This function gets invoked when a job has to be processed.
        If the server is available, the job is processed otherwise it is added
        to the queue (waiting). If the queue is full, the job is dropped. The retries
        mechanism is implemented by the timeout() in client.py  

        Args:
            job: the current job to be processed
            t : timestamp
            
        Returns:
            If the job is processed, it returns the completed job else None.
        """

        
        if job.max_retries > job.retries_left: 
            # this is a retried job
            self.retries += 1
            current=self.downstream_server
            while current is not None:
                current.retries += 1
                current=current.downstream_server

        if self.busy < self.thread_pool:
            # The server isn't entirely busy, so we can start on the job immediately
            self.busy += 1
            #print("busy ",self.busy,"  ",self.thread_pool)
            for i in range(self.thread_pool):
                if self.jobs[i] is None:
                    logger.info("Processing %s at %f on server %d" % (job.name, t, self.id))
                    self.jobs[i] = job
                    job.status = JobStatus.PROCESSING
                    service_time = self.service_time_distribution[job.name].sample()
                    #logger.info("server rate %f   service time  %f" % (self.service_time_distribution[job.name].mean,service_time))
                    job.size = service_time
                    logger.info("Processing %s at %f on server %d" % (job.name, t, self.id))
                    
                    return t + service_time, self.job_done, i
            # Should never get here because jobs slots should always be available if busy < thread_pool
            assert False
        else:
            
            if self.queue.len() < self.ts*self.queue_size:
                job.status = JobStatus.ENQUEUED
                logger.info("Enqueueing %s at %f" % (job.name, t))
                self.queue.append(job)        
            elif self.queue.len() < self.queue_size:
                #admission_prob = 0.5 
                if np.random.random() < self.ap:
                    job.status = JobStatus.ENQUEUED
                    logger.info("Enqueueing %s at %f" % (job.name, t))
                    self.queue.append(job)
                else:
                    job.status = JobStatus.DROPPED
                    logger.info("Dropped %s at %f" % (job.name, t))
                    self.dropped += 1
                    return t + self.client.timeout, self.client.on_timeout, job
            
            else:
                job.status = JobStatus.DROPPED
                logger.info("Dropped %s at %f" % (job.name, t))
                self.dropped += 1
                return t + self.client.timeout, self.client.on_timeout, job
            return None
    
