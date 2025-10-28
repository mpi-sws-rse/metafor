# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import math
import numpy as np
import pandas
import time
from collections import deque
from typing import List, TextIO

from metafor.simulator.client import Client, OpenLoopClientWithTimeout
from metafor.simulator.job import Distribution, Job, JobStatus


import logging
logger = logging.getLogger(__name__)


class Context:
    """
    manage the output from the simulation
    """
    def __init__(self, id: int):
        self.id = id
        self.result = []
    
    def write(self, l: List):
        self.result.append(l)
        
    def close(self):
        pass

    def queue_lengths(self):
        data = []
        for r in self.result:
            data.append((r['timestamp'], r['queue_length']))
        df = pandas.DataFrame(data)
        print(df)

    def latency(self):
        data = []
        for r in self.result:
            data.append((r['timestamp'], r['latency']))
        df = pandas.DataFrame(data)
        print(df)

    def analyze(self):
        print("Queue lengths")
        self.queue_lengths()

        print("Latencies")
        self.latency()

    """
    # Print the mean value, the variance, and the standard deviation at each stat point in each second
    def mean_variance_std_dev(self, max_t: float, num_runs: int, step_time: int, mean_t: float):
        # XXX: DO NOT USE: IN PROGRESS
        num_datapoints = math.ceil(max_t / step_time)
        latency_dateset = np.zeros((num_datapoints))
        runtime_dateset = np.zeros((num_datapoints))
        df = pandas.DataFrame(self.result)
        latency_ave = [0]
        latency_var = [0]
        latency_std = [0]
        runtime = [0]
        for step in range(num_datapoints):
            latency_ave.append(np.mean(latency_dateset[:, step]))
            latency_var.append(np.var(latency_dateset[:, step]))
            latency_std.append(np.std(latency_dateset[:, step]))
            runtime.append(np.sum(runtime_dateset[:, step]))
        return latency_ave,  latency_var, latency_std, runtime
    """

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



class LIFOStack:
    def __init__(self):
        self.deque = deque()

    def append(self, job):
        """Add a job to the top of the stack."""
        self.deque.append(job)

    def pop(self):
        """Remove and return the job from the top of the stack."""
        return self.deque.pop()

    def len(self) -> int:
        """Return the number of jobs in the stack."""
        return len(self.deque)

    @staticmethod
    def name() -> str:
        """Return the name of this scheduling discipline."""
        return "LIFO"



class Server:
    """
    Server that consumes a queue of tasks of a fixed size (`queue_size`), with a fixed concurrency (MPL)    
    """

    def __init__(self, name: str, queue_size: int, thread_pool: int,
                 service_time_distribution: dict[str, Distribution],
                 retry_queue_size: int, client: OpenLoopClientWithTimeout, throttle:bool):
        self.start_time = 0  # to be set by each simulation
        self.busy: int = 0

        self.queue: FCFSQueue = FCFSQueue()
        #self.queue: LIFOStack = LIFOStack()
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
        self.throttle: bool = throttle

    def set_context(self, c: Context):
        self.context = c

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
        
        assert (self.busy > 0)
        completed = self.jobs[n]
        assert completed is not None

        
        completed.status = JobStatus.COMPLETED
        completed.completed_t = t

        if completed.max_retries > completed.retries_left:  # a retried job is completed
            self.retries -= 1
        logger.info("Completing %s at %f" % (completed.name, t))

        end_time = time.time()
        runtime = end_time - self.start_time
        assert self.context is not None, "Context not set: cannot output results"
    
        self.context.write(
            {'timestamp': t,
             'latency' : t - completed.created_t,
             'queue_length' : self.queue.len(),
             'retries' : self.retries,
             'dropped' : self.dropped,
             'runtime' : runtime,
             'retries_left' : self.jobs[n].retries_left,
             'service_time' : self.jobs[n].size,
             })
             #[t, t - completed.created_t, self.queue.len(), self.retries, self.dropped])
        # self.file.write("%f,%f,%d,%d,%d,%f\n" % (t, t - completed.created_t,
        #                                              self.queue.len(), self.retries, self.dropped, runtime))
        
        events = []
        if self.queue.len() > 0:
            next_job = self.queue.pop()
            next_job.status = JobStatus.PROCESSING
            logger.info("Dequeueing %s created %f at %f" % (next_job.name, next_job.created_t, t))

            self.jobs[n] = next_job
            service_time = self.service_time_distribution[next_job.name].sample()
            #logger.info("server rate %f   service time  %f" % (self.service_time_distribution[next_job.name].mean,service_time))
            next_job.size = service_time
            events = [(t + service_time, self.job_done, n)]
        else:
            self.busy -= 1
            self.jobs[n] = None

        # done_event = self.client.done(t, completed)
        # if done_event is not None:
        #     events.append(done_event)

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

        if job.max_retries > job.retries_left: # this is a retried job
            self.retries += 1
            # if self.retries < self.retry_queue_size:
            #    self.retries += 1
            #else:
            #    self.retries += 1
            #    #self.dropped += 1  # there is not enough space in the virtual retries queue
            #    #return None

        if self.busy < self.thread_pool:
            # The server isn't entirely busy, so we can start on the job immediately
            self.busy += 1
            #print("busy ",self.busy,"  ",self.thread_pool)
            for i in range(self.thread_pool):
                if self.jobs[i] is None:
                    logger.info("Processing %s at %f" % (job.name, t))
                    self.jobs[i] = job
                    job.status = JobStatus.PROCESSING
                    service_time = self.service_time_distribution[job.name].sample()
                    #logger.info("server rate %f   service time  %f" % (self.service_time_distribution[job.name].mean,service_time))
                    job.size = service_time
                    logger.info("Processing %s at %f" % (job.name, t))
                    return t + service_time, self.job_done, i
            # Should never get here because jobs slots should always be available if busy < thread_pool
            assert False
        else:
            if self.throttle is False:    
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
            
            else:
                # Throttling strategy kicks in  as soon as queuue is 70% full
                # The server is busy, so try to enqueue the job, if there is enough space in the queue
                if self.queue.len() < 0.7*self.queue_size:
                    job.status = JobStatus.ENQUEUED
                    logger.info("Enqueueing %s at %f" % (job.name, t))
                    self.queue.append(job)        
                elif self.queue.len() < self.queue_size:
                    admission_prob = 0.1 
                    if np.random.random() < admission_prob:
                        job.status = JobStatus.ENQUEUED
                        logger.info("Enqueueing %s at %f" % (job.name, t))
                        self.queue.append(job)
                    else:
                        job.status = JobStatus.DROPPED
                        logger.info("Dropped %s at %f" % (job.name, t))
                        self.dropped += 1
                        return t, self.client.on_timeout, job
                
                else:
                    job.status = JobStatus.DROPPED
                    logger.info("Dropped %s at %f" % (job.name, t))
                    self.dropped += 1
                    return t + self.client.timeout, self.client.on_timeout, job
                return None
