# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import math
import numpy as np
import pandas
import time
from collections import deque
from typing import List, TextIO, Optional

from metafor.simulator.client import Client, OpenLoopClientWithTimeout
from metafor.simulator.job import Distribution, Job, JobStatus


import logging
logger = logging.getLogger(__name__)



#########################################
# Allow Context to log metrics for multiple servers, 
# possibly with a server identifier in the result dictionary.
# Extend analyze to report metrics per server (e.g., 
# queue lengths and latencies for each server)

class Context:
    """
    manage the output from the simulation
    """
    def __init__(self, id: int,server_id: int):
        self.id = id
        self.server_id = server_id
        self.result = []
    
    def write(self, l: List):
        self.result.append(l)
        
    def close(self):
        pass

    def queue_lengths(self):
        data = []
        for r in self.result:
            if r['server'] == self.server_id:
                data.append((r['timestamp'], r['queue_length']))
        df = pandas.DataFrame(data)
        print(df)
        return df

    def latency(self):
        data = []
        for r in self.result:
            if r['server'] == self.server_id:
                data.append((r['timestamp'], r['latency']))
        df = pandas.DataFrame(data)
        print(df)
        return df

    def analyze(self):
        servers = set(r['server'] for r in self.result)
        queue_dfs = []
        latency_dfs = []
        print(f"\nAnalyzing {server}")
        queue_dfs.append(self.queue_lengths())
        latency_dfs.append(self.latency())
        return queue_dfs, latency_dfs

  

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


class Server:
    """
    Server that consumes a queue of tasks of a fixed size (`queue_size`), with a fixed concurrency (MPL)    
    """

    def __init__(self, id:int, name: str, queue_size: int, thread_pool: int,
                 service_time_distribution: dict[str, Distribution],
                 retry_queue_size: int, client: OpenLoopClientWithTimeout,
                 downstream_server: Optional['Server'] = None):
        self.id = id
        self.start_time = 0  # to be set by each simulation
        self.busy: int = 0

        self.queue: FCFSQueue = FCFSQueue()
        self.queue_size: int = queue_size

        self.service_time_distribution = service_time_distribution
        
        self.sim_name: str = name

        self.thread_pool: int = thread_pool
        self.retry_queue_size: int = retry_queue_size

        self.jobs: List[Job | None] = [None for _ in range(thread_pool)]
        self.client: Client = client
        # self.rho: float = rho
        # self.file: TextIO | None = None
        self.context = None

        # statistics
        self.retries: int = 0
        self.dropped: int = 0  # cumulative number
        self.downstream_server: Optional['Server'] = downstream_server
    

    def set_context(self, c: Context):
        self.context = c
        #self.server = self
        #if self.downstream_server:
        #    self.downstream_server.set_context(c)  # Share context with downstream

    def print(self):
        print("DES Server: ", self.sim_name, "[q = ", self.queue_size, " threads=", self.thread_pool, "]")
        print("Rates: ", self.service_time_distribution)


    def job_done(self, t: float, n: int) -> List:
        """
        This function is invoked when a job is completed
        Once the job completes, the current context is added to the log
        and jobs waiting in queue are processed.
        
        Args:
            t : timestamp
            n :  number of threads (less than thread_pool)
            
        Returns:
            If queue is not empty, it returns the dequeued job else None.

        """
        #print(" Job DONE ",n)
        #assert (self.busy > 0)
        completed = self.jobs[n]
        if completed is None:
            return
        #assert completed is not None

        if self.downstream_server is None:
            # This ensures that a Job is completed when it has completed on the last 
            # downstream server
            completed.status = JobStatus.COMPLETED
            completed.completed_t = t

            if completed.max_retries > completed.retries_left:  # a retried job is completed
                self.retries -= 1
                current=self.downstream_server
                while current is not None:
                    current.retries -= 1
                    current=current.downstream_server
                
            logger.info("Completing %s at %f on server %d" % (completed.name, t,self.id))

            end_time = time.time()
            runtime = end_time - self.start_time
            assert self.context is not None, "Context not set: cannot output results"

        #################################################
        # Ensure metrics like latency account for the total time
        # a job spends across all servers.
        self.context.write(
            {'server': self.id,
            'timestamp': t,
            'latency' : t - completed.created_t,
            'queue_length' : self.queue.len(),
            'retries' : self.retries,
            'dropped' : self.dropped,
            'runtime' : time.time() - self.start_time,
            'retries_left' : self.jobs[n].retries_left,
            'service_time' : self.jobs[n].size,
            })
            #[t, t - completed.created_t, self.queue.len(), self.retries, self.dropped])
        # self.file.write("%f,%f,%d,%d,%d,%f\n" % (t, t - completed.created_t,
        #                                              self.queue.len(), self.retries, self.dropped, runtime))
        events = []
        
        # In job_done, forward completed jobs to downstream_server.offer if downstream_server exists.
        if self.downstream_server is not None:
            #service_time = self.service_time_distribution[self.jobs[n].name].sample()
            # self.jobs[n].size = service_time
            offered = self.downstream_server.offer(completed, t)
            # we do not add service time to t as it is already being done at l:258
            if offered is not None:
                events.append(offered)
        #     #events.append((t + service_time, self.downstream_server.job_done, n))
            
            
        
        if self.queue.len() > 0:
            next_job = self.queue.pop()
            next_job.status = JobStatus.PROCESSING
            logger.info("Dequeueing %s created %f at %f" % (next_job.name, next_job.created_t, t))

            self.jobs[n] = next_job
            service_time = self.service_time_distribution[next_job.name].sample()
            #logger.info("server rate %f   service time  %f" % (self.service_time_distribution[next_job.name].mean,service_time))
            next_job.size = service_time
            events.append((t + service_time, self.job_done, n))

        else:
            self.busy -= 1
            self.jobs[n] = None

        return events

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
                
           
        ##########################################################
        # Ensure dropped jobs or retries are handled consistently, 
        # considering the downstream server’s state.
        # Should we have a thread pool for each server?
        #
        if self.busy < self.thread_pool:
            # The server isn't entirely busy, so we can start on the job immediately
            self.busy += 1
            for i in range(self.thread_pool):
                if self.jobs[i] is None:
                    logger.info("Processing %s at %f on server %d" % (job.name, t, self.id))
                    self.jobs[i] = job
                    job.status = JobStatus.PROCESSING
                    service_time = self.service_time_distribution[job.name].sample()
                    #logger.info("server rate %f   service time  %f" % (self.service_time_distribution[job.name].mean,service_time))
                    job.size = service_time
                    logger.info("Processing %s at %f on server %d" % (job.name, t, self.id))
                    #print("job  offered at server",self.id,"  ",self.downstream_server)
                    # if self.downstream_server is not None:
                    #     offered = self.downstream_server.offer(job, t)
                    #     print("downstream offered ")
                    #     return [t + service_time, self.downstream_server.job_done, i],  offered
                    # else:
                    return t + service_time, self.job_done, i
                
            # Should never get here because jobs slots should always be available if busy < thread_pool
            raise ValueError("No free job slots despite busy < thread_pool")
            #assert False
        else:
            # The server is busy, so try to enqueue the job, if there is enough space in the queue
            if self.queue.len() < self.queue_size:
                job.status = JobStatus.ENQUEUED
                logger.info("Enqueueing %s at %f" % (job.name, t))
                self.queue.append(job)
            else:
                job.status = JobStatus.DROPPED
                logger.info("Dropped %s at %f" % (job.name, t))
                self.dropped += 1
                return t + self.client.timeout, self.client.on_timeout, job
            return None
