# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import math
import numpy as np
import pandas
import time
from collections import deque, defaultdict
from typing import List, TextIO, Optional

from metafor.simulator.client import Client, OpenLoopClientWithTimeout
from metafor.simulator.job import Distribution, Job, JobStatus, RetryOrigin, DropReason

import logging
logger = logging.getLogger(__name__)


class JoinTracker:
    def __init__(self, dag: dict):
        self.fan_in = defaultdict(int)
        for src, dsts in dag.items():
            for d in dsts:
                self.fan_in[d] += 1

        self.pending = {}
        self.latencies = defaultdict(list)
        self.representative_job = {}

    def is_join_node(self, server_id: int) -> bool:
        return self.fan_in[server_id] > 1

    def record(self, request_id, server_id, latency, job) -> tuple[bool, Job]:
        key = (request_id, server_id)
        if key not in self.pending:
            self.pending[key] = self.fan_in[server_id]
            self.representative_job[key] = job
        else:
            # merge attempt history
            for sid, count in job.server_attempts.items():
                self.representative_job[key].server_attempts[sid] = max(
                    self.representative_job[key].server_attempts[sid], count
                )

        self.latencies[key].append(latency)
        self.pending[key] -= 1
        done = self.pending[key] == 0
        return done, self.representative_job[key]

    def true_latency(self, request_id, server_id) -> float:
        return max(self.latencies[(request_id, server_id)])

    def cleanup(self, request_id, server_id):
        key = (request_id, server_id)
        del self.pending[key]
        del self.latencies[key]
        del self.representative_job[key]


#########################################
# Allow Context to log metrics for multiple servers, 
# possibly with a server identifier in the result dictionary.
# Extend analyze to report metrics per server (e.g., 
# queue lengths and latencies for each server)

class Context:
    """
    manage the output from the simulation
    """
    def __init__(self, id: int,server_id: int, join_tracker: JoinTracker):
        self.id = id
        self.server_id = server_id
        self.join_tracker = join_tracker
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
        #servers = set(r['server'] for r in self.result)
        queue_dfs = []
        latency_dfs = []
        print(f"\nAnalyzing {self.server_id}")
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



class TokenBucket:
    def __init__(self, capacity: float, refill_rate: float):
        """
        capacity    : maximum tokens (burst size)
        refill_rate : tokens added per unit simulation time
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity      # start full
        self._last_refill_t = 0.0

    def refill(self, t: float):
        elapsed = (t - self._last_refill_t)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self._last_refill_t = t

    def consume(self, t: float, n: float = 1.0) -> bool:
        """Refill first, then try to consume n tokens. Returns True if admitted."""
        self.refill(t)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False
    


class Server:
    """
    Server that consumes a queue of tasks of a fixed size (`queue_size`), with a fixed concurrency (MPL)    
    """

    def __init__(
        self, 
        id:int, 
        name: str, 
        queue_size: int, 
        thread_pool: int,
        service_dist: Distribution,
        client: OpenLoopClientWithTimeout,
        downstream_server: Optional[List['Server']] = None,
        timeout=None,
        max_retries=0,
        retry_delay=0.0,
        token_bucket: TokenBucket | None = None,
    ):
        self.id = id
        self.start_time = 0  # to be set by each simulation
        self.busy = 0

        self.queue = FCFSQueue()
        self.queue_size = queue_size

        self.service_dist = service_dist
        
        self.sim_name = name

        self.thread_pool = thread_pool

        self.jobs = [None for _ in range(thread_pool)]
        self.client= client
        # self.rho: float = rho
        # self.file: TextIO | None = None
        self.context = None

        # statistics
        self.retries = 0
        #self.dropped = 0  # cumulative number
        self.downstream_server = downstream_server

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.token_bucket = token_bucket

        self.dropped_queue_full: int = 0
        self.dropped_token_bucket: int = 0
        self.token_data = []
        
    
    @property
    def dropped(self) -> int:
        return self.dropped_queue_full + self.dropped_token_bucket

    def set_context(self, c: Context):
        self.context = c
        #self.server = self
        #if self.downstream_server:
        #    self.downstream_server.set_context(c)  # Share context with downstream

    def print(self):
        print("DES Server: ", self.sim_name, "[q = ", self.queue_size, " threads=", self.thread_pool, "]")
        print("Rates: ", self.service_dist)

    def _drain_queue(self, t: float, n: int) -> list:
        """
        Free thread slot n and assign it to the next queued job
        if one exists. Returns any new events to add to the 
        simulator queue.
        """

        # Always clear the slot first so the invariant holds before any
        # new assignment — callers must not touch self.jobs[n] after this.
        self.jobs[n] = None

        if self.queue.len() > 0:
            next_job = self.queue.pop()
            next_job.status = JobStatus.PROCESSING
            logger.info(
                "Dequeueing %s id %s created %f at %f on server %d"
                % (next_job.name, next_job.request_id, next_job.created_t, t, self.id)
            )
            self.jobs[n] = next_job
            # busy is unchanged: one job left, one arrived — net zero
            service_time = self.service_dist.sample()
            next_job.size = service_time
            return [(t + service_time, self.job_done, n)]
        else:
            # No work waiting — slot genuinely goes idle
            # The key idea is: offer is the only writer of busy += 1, 
            # and _drain_queue is the only writer of busy -= 1.
            self.busy -= 1
            return []


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
            return []
        completed.completed_t = t

        
        
        tracker = self.context.join_tracker
        
        # ── Join gate (any node with fan-in > 1) ──────────────────
        if tracker.is_join_node(self.id):
            all_done, completed = tracker.record(
                completed.request_id, self.id, t - completed.created_t, completed
            )
            if not all_done:
                # Branch arrived but others still pending.
                # Free this thread — do NOT forward downstream yet.
                # self.busy -= 1
                # self.jobs[n] = None
                completed.status = JobStatus.FORWARDED  # branch waiting at gate
                return self._drain_queue(t, n)  # still service queued jobs
            true_lat = tracker.true_latency(completed.request_id, self.id)
            tracker.cleanup(completed.request_id, self.id)
        else:
            true_lat = t - completed.created_t

        if self.downstream_server is None:
            true_lat = t - completed.created_t

        # logger.info("Completing %s with id %s at %f on server %d" 
        #             % (completed.name, completed.request_id, t,self.id))
        logger.info("Completing %s id %s attempt %s at %f on server %d" 
                    % (completed.name, completed.request_id, completed.attempt_id, t, self.id))

        end_time = time.time()
        runtime = end_time - self.start_time
        assert self.context is not None, "Context not set: cannot output results"

        #################################################
        # Ensure metrics like latency account for the total time
        # a job spends across all servers.
        self.context.write(
            {'server': self.id,
            'timestamp': t,
            'latency' : true_lat,
            'queue_length' : self.queue.len(),
            'retries' : self.retries,
            'dropped' : self.dropped,
            'runtime' : time.time() - self.start_time,
            'retries_left' : self.max_retries - completed.server_attempts[self.id],
            'service_time' : self.jobs[n].size,
            'throughput' : completed.client.num_complete_jobs if self.downstream_server is None else 0.0,
            'request_id' : self.jobs[n].request_id,
            'attempt_id' : self.jobs[n].attempt_id,
            'retry_origin': completed.retry_origin.value,
            'client_retries_used': completed.client.max_retries - completed.retries_left,
            'server_retries_used': completed.server_attempts[self.id],
            'dropped_queue_full':    self.dropped_queue_full,
            'dropped_token_bucket':  self.dropped_token_bucket,
            })
            #[t, t - completed.created_t, self.queue.len(), self.retries, self.dropped])
        # self.file.write("%f,%f,%d,%d,%d,%f\n" % (t, t - completed.created_t,
        #                                              self.queue.len(), self.retries, self.dropped, runtime))
        
        #assert completed is not None

        events = []
        
        # In job_done, forward completed jobs to downstream_server.offer if downstream_server exists.
        # if self.downstream_server is not None:
        #     offered = self.downstream_server.offer(completed, t)
        #     if offered is not None:
        #         events.append(offered)
        print("Server", self.id, "completed job", completed.request_id)
        
        # ── Forward downstream ─────────────────────────────────────
        if self.downstream_server:
            completed.status = JobStatus.FORWARDED  # ← job moving to next server
            print("Forwarding to:", [ds.id for ds in self.downstream_server])

            for ds in self.downstream_server:
                branch_job = completed.clone_for_branch(t)
                offered = ds.offer(branch_job, t)
                if offered:
                    if isinstance(offered, list):
                        events.extend(offered)
                    else:
                        events.append(offered)
        
        # ── Notify client (leaf only) ──────────────────────────────
        else:
            print("Forwarding to:  [Client]")
            #completed.status = JobStatus.COMPLETED
            #completed.client.num_complete_jobs += 1
            client_event = completed.client.on_complete(t, completed)
            if client_event:
                if isinstance(client_event, list):
                    events.extend(client_event)
                else:
                    events.append(client_event)

        
        events.extend(self._drain_queue(t, n))

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
        self.token_data.append(self.token_bucket.tokens)
        # Token bucket check — before queue or thread logic
        if self.token_bucket is not None:
            if not self.token_bucket.consume(t):
                job.status = JobStatus.DROPPED
                job.drop_reason = DropReason.TOKEN_BUCKET
                self.dropped_token_bucket += 1
                #self.dropped += 1
                logger.info("Token bucket dropped %s id %s at %f on server %d"
                            % (job.name, job.request_id, t, self.id))
                return None

        events = []                    
           
        ##########################################################
        # Ensure dropped jobs or retries are handled consistently, 
        # considering the downstream server’s state.
        # Should we have a thread pool for each server?
        #
        if self.busy < self.thread_pool:
            # The server isn't entirely busy, so we can start on the job immediately
            self.busy += 1
            #print("busy $$$$$  ")
            for i in range(self.thread_pool):
                if self.jobs[i] is None:
                    #print("processing $$$$$  ")
                    #logger.info("Processing %s with id %s at %f on server %d" % (job.name, job.request_id, t, self.id))
                    self.jobs[i] = job
                    job.status = JobStatus.PROCESSING
                    #service_time = self.service_time_distribution[job.name].sample()
                    service_time = self.service_dist.sample()
                    #logger.info("server rate %f   service time  %f" % (self.service_time_distribution[job.name].mean,service_time))
                    job.size = service_time
                    logger.info("Processing %s with id %s at %f on server %d" % (job.name, job.request_id, t, self.id))
                    
                    events = [(t + service_time, self.job_done, i)]

                    # schedule server-level timeout
                    if self.timeout is not None:
                        events.append((t + self.timeout, self.on_timeout, (job, i)))

                    return events
                
            # Should never get here because jobs slots should always be available if busy < thread_pool
            raise ValueError("No free job slots despite busy < thread_pool")
            #assert False
        else:
            # The server is busy, so try to enqueue the job, if there is enough space in the queue
            if self.queue.len() < self.queue_size:
                job.status = JobStatus.ENQUEUED
                logger.info("Enqueueing %s with id %s at %f on server %d" % (job.name, job.request_id, t, self.id))
                self.queue.append(job)
            else:
                job.status = JobStatus.DROPPED
                job.drop_reason = DropReason.QUEUE_FULL
                self.dropped_queue_full += 1
                logger.info("Dropped %s with id %s at %f  on server %d" % (job.name, job.request_id, t, self.id))
                #self.dropped += 1
                #return t + self.client.timeout, self.client.on_timeout, job
            return None



    def on_timeout(self, t, payload):

        job, thread_id = payload

        # job finished or dropped
        if job.status in {JobStatus.COMPLETED, JobStatus.DROPPED, JobStatus.FORWARDED}:
            return None
        
        #Because the thread may now contain another job.
        if self.jobs[thread_id] is None or self.jobs[thread_id].request_id != job.request_id:
            return None

        attempts = job.server_attempts[self.id]

        if attempts >= self.max_retries:
            return None

        

        job.server_attempts[self.id] += 1
        self.retries += 1

        retry_job = job.clone_for_retry(t)
        retry_job.retry_origin = RetryOrigin.SERVER

        logger.info(
            f" Retry {attempts+1} for request {job.request_id} on Server {self.id}, "
        )

        offered = self.offer(retry_job, t + self.retry_delay)

        if offered:
            if isinstance(offered, list):
                return offered
            return [offered]

        return None
