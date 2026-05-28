import math
import numpy as np
import pandas
import time
from collections import deque, defaultdict
from typing import List, TextIO, Optional
from collections import deque
from metafor.simulator.client import Client, OpenLoopClientWithTimeout
from metafor.simulator.job import Distribution, Job, JobStatus, RetryOrigin, DropReason

import logging
logger = logging.getLogger(__name__)


class Context:
    def __init__(self, id: int, server_id: int):
        self.id = id
        self.server_id = server_id
        self.result = []
        self.recent_latencies = deque(maxlen=100)

    def write(self, l: List):
        self.result.append(l)
        self.recent_latencies.append(l['latency'])

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
        self.tokens = capacity
        self._last_refill_t = 0.0

    def refill(self, t: float):
        elapsed = t - self._last_refill_t
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self._last_refill_t = t

    def consume(self, t: float, n: float = 1.0) -> bool:
        """Refill first, then try to consume n tokens. Returns True if admitted."""
        self.refill(t)
        if self.tokens >= n:
            self.tokens -= n
            return True
        return False


# ──────────────────────────────────────────────────────────────────────────────
# JoinTracker — synchronous fan-out join barrier
# ──────────────────────────────────────────────────────────────────────────────

class JoinTracker:
    """
    Tracks pending branch completions for a synchronous fan-out at one server.

    When server S fans out to N downstream branches, a single JoinTracker is
    created and stored in each branch job's response_callback tuple (in the
    position that normally holds the integer slot index).  Every branch — on
    success OR on final drop — calls _on_branch_response(t, (tracker, None)).
    Only when pending reaches 0 does the join fire.

    Design invariants
    -----------------
    * One JoinTracker per fan-out event (per job, per server slot).
    * Thread slot `slot_n` at the fanning-out server is held until join fires.
    * `upstream_cb` is the (cb_fn, cb_slot, cb_next) tuple this server received
      from its own upstream; None if this server is the root.
    * The DES event loop processes events in time order, so the last call to
      branch_done() happens at t = max(all branch completion times), which is
      the correct join timestamp.
    """

    __slots__ = ('slot_n', 'upstream_cb', 'pending', 'request_id', 'attempt_id')

    def __init__(self, slot_n: int, upstream_cb, n_branches: int, 
                 request_id: str, attempt_id: int):
        self.slot_n = slot_n
        self.upstream_cb = upstream_cb
        self.pending = n_branches
        #print(request_id)
        self.request_id = request_id   
        self.attempt_id = attempt_id

    def branch_done(self) -> bool:
        """
        Decrement pending count.  Returns True iff this was the last branch.
        """
        self.pending -= 1
        return self.pending == 0

    @property
    def timeout_key(self):
        """Canonical key used in Server.timed_out_requests."""
        return (self.request_id, self.attempt_id)


# ──────────────────────────────────────────────────────────────────────────────

class Server:
    """
    Server that consumes a queue of tasks of a fixed size (`queue_size`),
    with a fixed concurrency (MPL).

    Fan-out behaviour (len(downstream_server) > 1)
    ───────────────────────────────────────────────
    When job_done fires on a server with multiple downstream targets, a
    JoinTracker is created for that job's thread slot.  One cloned branch
    job is forwarded to each downstream server; each branch job carries
    _on_branch_response (not _on_downstream_response) as its callback.

    _on_branch_response decrements the JoinTracker.  Only when all branches
    have returned (pending == 0) does the join fire:
      • _write_metrics is called with t = join completion time
      • _drain_queue releases the thread slot
      • the upstream callback chain is propagated

    This gives a synchronous, all-or-nothing join: the fanning-out server's
    slot is held until max(branch latencies), matching the JoinTracker
    semantics visible in the Gantt diagram.

    Metric recording policy (unchanged from chain topology)
    ───────────────────────
    Leaf server (no downstream_server):
        _write_metrics inside job_done (own service completion).
    Non-leaf / fan-out server:
        _write_metrics inside _on_downstream_response or _on_branch_response,
        whichever fires last — i.e. when the full round-trip completes.
    """

    def __init__(
        self,
        id: int,
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
        network_dist=None,
    ):
        self.id = id
        self.start_time = 0
        self.busy = 0

        self.queue = FCFSQueue()
        self.queue_size = queue_size
        self.service_dist = service_dist
        self.sim_name = name
        self.thread_pool = thread_pool
        self.jobs = [None for _ in range(thread_pool)]
        self.client = client
        self.context = None

        self.retries = 0
        self.downstream_server = downstream_server
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.token_bucket = token_bucket

        self.dropped_queue_full: int = 0
        self.dropped_token_bucket: int = 0
        self.completed_jobs: int = 0
        self.throughput: int = 0
        self.last_latency: float = 0.0
        self.global_recent_latencies = deque(maxlen=200)
        self.stale_dropped = 0

        self.forwarded_request_ids: deque[str] = deque(maxlen=5000)
        self.network_dist = network_dist

        self.timed_out_requests: deque[str] = deque(maxlen=5000)

    @property
    def dropped(self) -> int:
        return self.dropped_queue_full + self.dropped_token_bucket

    def set_context(self, c: Context):
        self.context = c

    def print(self):
        print("DES Server: ", self.sim_name,
              "[q = ", self.queue_size, " threads=", self.thread_pool, "]")
        print("Rates: ", self.service_dist)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────


    def _timeout_key(self, job: Job) -> tuple:
        """Canonical key identifying one specific attempt of a request."""
        return (job.request_id, job.attempt_id)

    def _slot_is_stale(self, slot_n: int, expected_key: tuple) -> bool:
        """
        Returns True if the job currently in slot_n is NOT the one we expect,
        meaning either the slot was freed and reassigned, or the attempt was
        already timed out.

        FIX: previously only checked request_id, so retries (same request_id,
        higher attempt_id) were incorrectly treated as stale and discarded.
        Now checks the full (request_id, attempt_id) pair.
        """
        job = self.jobs[slot_n]
        if job is None:
            return True
        return self._timeout_key(job) in self.timed_out_requests

    def _drain_queue(self, t: float, n: int) -> list:
        self.jobs[n] = None

        if self.queue.len() > 0:
            next_job = self.queue.pop()
            if hasattr(next_job, 'created_t') and \
               (t - next_job.created_t) > next_job.client.max_retries * next_job.client.timeout:
                self.stale_dropped = getattr(self, 'stale_dropped', 0) + 1
                return self._drain_queue(t, n)
            next_job.status = JobStatus.PROCESSING
            logger.info(
                "Dequeueing %s trace=%s id=%s created %f at %f on server %d"
                % (next_job.name, next_job.trace_id, next_job.request_id, 
                   next_job.created_t, t, self.id)
            )
            self.jobs[n] = next_job
            service_time = self.service_dist.sample()
            next_job.size = service_time
            return [(t + service_time, self.job_done, n)]
        else:
            self.busy -= 1
            return []

    def _write_metrics(self, t: float, job: Job, is_leaf: bool):
        assert self.context is not None, "Context not set: cannot output results"

        true_lat = t - job.created_t
        self.last_latency = true_lat
        self.global_recent_latencies.append(true_lat)
        self.completed_jobs += 1
        self.throughput = job.client.num_complete_jobs

        self.context.write({
            'server':               self.id,
            'timestamp':            t,
            'latency':              true_lat,
            'queue_length':         self.queue.len(),
            'retries':              self.retries,
            'dropped':              self.dropped,
            'runtime':              time.time() - self.start_time,
            'retries_left':         self.max_retries - job.server_attempts[self.id],
            'service_time':         job.size,
            'throughput':           job.client.num_complete_jobs if is_leaf else 0.0,
            'trace_id':             job.trace_id,
            'request_id':           job.request_id,
            'attempt_id':           job.attempt_id,
            'retry_origin':         job.retry_origin.value,
            'client_retries_used':  job.client.max_retries - job.retries_left,
            'server_retries_used':  job.server_attempts[self.id],
            'dropped_queue_full':   self.dropped_queue_full,
            'dropped_token_bucket': self.dropped_token_bucket,
        })

    # ──────────────────────────────────────────────────────────────────────────
    # Network / callback path
    # ──────────────────────────────────────────────────────────────────────────

    def _network_deliver(self, t: float, payload) -> list:
        ds, job = payload
        offered = ds.offer(job, t)
        if offered:
            return offered if isinstance(offered, list) else [offered]
        if job.status == JobStatus.ENQUEUED:
            # Accepted into queue — not a rejection.
            return []
        return [(t, self._on_rejection, (job, ds))]

    def _on_rejection(self, t: float, payload) -> list:
        """
        Called when a downstream drop is received.  Schedules a retry
        if retries remain.

        Fan-out extension: if all retries are exhausted AND this job belongs
        to a fan-out branch (identified by JoinTracker in response_callback),
        fire the branch callback so the join counter is decremented and the
        parent server's slot is not held indefinitely.
        """
        job, ds = payload
        attempts = job.server_attempts[ds.id]
        if attempts >= self.max_retries:
            # No retries left.
            # If this is a fan-out branch, unblock the JoinTracker.
            if job.response_callback is not None:
                cb_fn, cb_arg, cb_next = job.response_callback
                if isinstance(cb_arg, JoinTracker):
                    logger.info(
                        "Branch final-drop for id %s: attempt=%d:  notifying JoinTracker at t=%f"
                        % (job.request_id,job.attempt_id, t)
                    )
                    return [(t, cb_fn, (cb_arg, cb_next))]
            return []

        job.server_attempts[ds.id] += 1
        self.retries += 1

        retry_job = job.clone_for_retry(t)
        retry_job.retry_origin = RetryOrigin.SERVER

        #backoff = self.retry_delay * (2 ** attempts)
        backoff = self.retry_delay
        return [(t + backoff, self._network_deliver, (ds, retry_job))]

    def _on_downstream_response(self, t: float, payload) -> list:
        """
        Fires when the downstream response arrives back (single-downstream path).

        Sequence: capture job → write metrics → release slot → propagate upstream.
        """
        slot_n, upstream_cb = payload

        # Slot may be None if on_timeout already freed it (late-arriving callback).
        if self.jobs[slot_n] is None or \
                self._timeout_key(self.jobs[slot_n]) in self.timed_out_requests:
            logger.info(
                "Server %d slot %d: late callback at t=%f discarded (slot freed by timeout)"
                % (self.id, slot_n, t)
            )
            return []

        # assert self.jobs[slot_n] is not None, (
        #     f"Server {self.id} slot {slot_n} was already freed before "
        #     f"response arrived at t={t}"
        # )

        job = self.jobs[slot_n]
        req_id = job.request_id

        logger.info(
            "Upstream response received with trace=%s id=%s at server %d slot %d at t=%f"
            % (job.trace_id, req_id, self.id, slot_n, t)
        )

        self._write_metrics(t, job, is_leaf=False)
        events = self._drain_queue(t, slot_n)

        if upstream_cb is not None:
            cb_fn, cb_slot, cb_next = upstream_cb
            net_delay = self.network_dist.sample() if self.network_dist else 0.0
            logger.info(
                "Propagating upstream callback with trace=%s id=%s to next server at t=%f network delay=%f"
                % (job.trace_id, req_id, t, net_delay)
            )
            events.append((t + net_delay, cb_fn, (cb_slot, cb_next)))
        else:
            net_delay = self.network_dist.sample() if self.network_dist else 0.0
            logger.info(
                "Root server %d notifying client trace=%s id=%s at t=%f network delay=%f"
                % (self.id, job.trace_id, req_id, t, net_delay)
            )
            events.append((t + net_delay, job.client.on_complete, job))

        return events

    def _on_branch_response(self, t: float, payload) -> list:
        """
        Fires when one branch of a fan-out completes (success or final drop).

        Decrements the shared JoinTracker.  When the last branch responds
        (pending == 0), this method acts identically to _on_downstream_response:
        writes metrics at the join completion time, releases the thread slot,
        and propagates the upstream callback.

        Parameters
        ----------
        t : float
            Current simulation time.  Because the DES event loop processes
            events in strict time order, the call that reaches pending == 0
            always arrives at t == max(all branch completion times), which is
            the correct join timestamp without needing a stored max.
        payload : (JoinTracker, None)
            The JoinTracker instance shared by all branches of this fan-out.
            The second element is always None (placeholder for the cb_next
            position that the existing callback-tuple protocol expects).
        """
        tracker, _ = payload

        if not tracker.branch_done():
            # More branches still in flight — nothing to do yet.
            return []

        # ── All branches done: fire the join ──────────────────────────────────
        slot_n = tracker.slot_n
        upstream_cb = tracker.upstream_cb

        # Guard: parent slot may have been freed by a timeout while branches were in flight.
        if self.jobs[slot_n] is None  or tracker.request_id in self.timed_out_requests:
            logger.info(
                "Server %d: join complete at t=%f but slot %d already freed by timeout — discarding"
                % (self.id, t, slot_n)
            )
            return []

        # assert self.jobs[slot_n] is not None, (
        #     f"Server {self.id}: slot {slot_n} was freed before join completed "
        #     f"at t={t}.  This indicates a double-free or missing JoinTracker."
        # )

        job = self.jobs[slot_n]
        req_id = job.request_id

        logger.info(
            "Fan-out join complete for trace=%s id=%s at server %d slot %d at t=%f"
            % (job.trace_id, req_id, self.id, slot_n, t)
        )

        # t is the join completion time (latest branch).
        self._write_metrics(t, job, is_leaf=False)

        # Release thread slot; dequeue next waiting job at this server.
        events = self._drain_queue(t, slot_n)

        if upstream_cb is not None:
            cb_fn, cb_slot, cb_next = upstream_cb
            net_delay = self.network_dist.sample() if self.network_dist else 0.0
            logger.info(
                "Join: propagating upstream callback trace=%s id=%s at t=%f network delay=%f"
                % (job.trace_id, req_id, t, net_delay)
            )
            events.append((t + net_delay, cb_fn, (cb_slot, cb_next)))
        else:
            # Root server with fan-out (uncommon but valid) — notify client.
            net_delay = self.network_dist.sample() if self.network_dist else 0.0
            logger.info(
                "Join: root server %d notifying client for id %s at t=%f"
                % (self.id, req_id, t)
            )
            events.append((t + net_delay, job.client.on_complete, job))

        return events

    # ──────────────────────────────────────────────────────────────────────────
    # Core event handlers
    # ──────────────────────────────────────────────────────────────────────────

    def job_done(self, t: float, n: int) -> List:
        """
        Invoked when a job finishes its service time on this server.

        Single downstream (chain):
            Unchanged from original — clones one branch job, sets
            _on_downstream_response as its callback, forwards.

        Multiple downstream (fan-out):
            Creates a JoinTracker for this slot.  Clones one branch job per
            downstream server, each carrying _on_branch_response as its
            callback.  The thread slot is held until all branches return
            and _on_branch_response fires the join.

        Leaf (no downstream):
            Unchanged — writes metrics immediately, frees slot, fires callback.
        """
        completed = self.jobs[n]
        if completed is None:
            return []
        completed.completed_t = t

        if completed.request_id in self.forwarded_request_ids:
            return self._drain_queue(t, n)

        logger.info("Completing %s trace=%s id=%s at %f on server %d"
                    % (completed.name, completed.trace_id, 
                       completed.request_id, t, self.id))

        self.forwarded_request_ids.append(completed.request_id)

        events = []

        # ── Non-leaf: forward downstream ──────────────────────────────────────
        if self.downstream_server:
            completed.status = JobStatus.FORWARDED
            n_branches = len(self.downstream_server)

            if n_branches == 1:
                # ── Single downstream: original chain behaviour ───────────────
                branch_job = completed.clone_for_branch(t)
                upstream_cb = completed.response_callback
                branch_job.response_callback = (
                    self._on_downstream_response, n, upstream_cb
                )
                net_delay = self.network_dist.sample() if self.network_dist else 0.0
                events.append((
                    t + net_delay,
                    self._network_deliver,
                    (self.downstream_server[0], branch_job),
                ))

            else:
                # ── Fan-out: synchronous join via JoinTracker ─────────────────
                #
                # One JoinTracker per fan-out event.  Each branch job receives
                # the same tracker instance in its response_callback so that
                # _on_branch_response can decrement the shared counter.
                #
                # NOTE: clone_for_branch must produce independent job copies
                # (separate request_id / attempt_id) so that each downstream
                # server tracks its own service independently.  Verify this
                # in Job.clone_for_branch if you observe duplicate-id warnings.
                
                tracker = JoinTracker(
                    slot_n=n,
                    upstream_cb=completed.response_callback,
                    n_branches=n_branches,
                    request_id=completed.request_id,
                    attempt_id=completed.attempt_id,
                )
                logger.info(
                    "Fan-out from server %d slot %d: trace=%s %d branches at t=%f"
                    % (self.id, n, completed.trace_id, n_branches, t)
                )
                for ds in self.downstream_server:
                    branch_job = completed.clone_for_branch(t)
                    # cb_arg position holds the JoinTracker (not a slot int).
                    # _on_branch_response detects this via isinstance(cb_arg, JoinTracker).
                    branch_job.response_callback = (
                        self._on_branch_response, tracker, None
                    )
                    net_delay = self.network_dist.sample() if self.network_dist else 0.0
                    events.append((
                        t + net_delay,
                        self._network_deliver,
                        (ds, branch_job),
                    ))

            # Thread slot is NOT released here — held until callback fires.
            return events

        # ── Leaf node ─────────────────────────────────────────────────────────
        else:
            self._write_metrics(t, completed, is_leaf=True)
            events.extend(self._drain_queue(t, n))

            if completed.response_callback is not None:
                cb_fn, cb_slot, cb_next = completed.response_callback
                net_delay2 = self.network_dist.sample() if self.network_dist else 0.0
                logger.info(
                    "Leaf server %d firing upstream callback trace=%s id=%s at t=%f network delay=%f"
                    % (self.id, completed.trace_id, completed.request_id, t, net_delay2)
                )
                events.append((t + net_delay2, cb_fn, (cb_slot, cb_next)))
            else:
                # Single-server edge case: leaf is also root.
                net_delay = self.network_dist.sample() if self.network_dist else 0.0
                events.append((t + net_delay, completed.client.on_complete, completed))

            return events

    def offer(self, job: Job, t: float):
        if self.token_bucket is not None:
            if not self.token_bucket.consume(t):
                job.status = JobStatus.DROPPED
                job.drop_reason = DropReason.TOKEN_BUCKET
                self.dropped_token_bucket += 1
                logger.info("Token bucket dropped trace=%s id=%s at %f on server %d"
                            % (job.trace_id, job.request_id, t, self.id))
                return None

        if self.busy < self.thread_pool:
            self.busy += 1
            for i in range(self.thread_pool):
                if self.jobs[i] is None:
                    self.jobs[i] = job
                    job.status = JobStatus.PROCESSING
                    service_time = self.service_dist.sample()
                    job.size = service_time
                    logger.info("Processing trace=%s id=%s at %f on server %d"
                                % (job.trace_id, job.request_id, t, self.id))
                    events = [(t + service_time, self.job_done, i)]
                    if self.timeout is not None:
                        events.append((t + self.timeout, self.on_timeout, (job, i)))
                    return events
            raise ValueError("No free job slots despite busy < thread_pool")
        else:
            if self.queue.len() < self.queue_size:
                job.status = JobStatus.ENQUEUED
                logger.info("Enqueueing trace=%s id=%s at %f on server %d"
                            % (job.trace_id, job.request_id, t, self.id))
                self.queue.append(job)
            else:
                job.status = JobStatus.DROPPED
                job.drop_reason = DropReason.QUEUE_FULL
                self.dropped_queue_full += 1
                logger.info("Dropped trace=%s id=%s at %f on server %d"
                            % (job.trace_id, job.request_id, t, self.id))
            return None

    def on_timeout(self, t, payload):
        job, thread_id = payload

        # if job.request_id in self.forwarded_request_ids:
        #     return None
        # removed: forwarded_request_ids check — it fires before timeout ever could ──


        if job.status in {JobStatus.COMPLETED, JobStatus.DROPPED}:
            return None

        if self.jobs[thread_id] is None or \
                self.jobs[thread_id].request_id != job.request_id:
            return None

        
        # Mark as timed out so late-arriving downstream callbacks discarded.
        self.timed_out_requests.append(self._timeout_key(job))

        # Free the slot NOW — we are abandoning the downstream call.
        # _drain_queue sets jobs[thread_id]=None and decrements busy.
        events = self._drain_queue(t, thread_id)
        attempts = job.server_attempts[self.id]

        if attempts >= self.max_retries:
            logger.info(
                f"Server {self.id} slot {thread_id}: timeout, max retries exhausted "
                f"for {job.request_id}  attempt={job.attempt_id} at t={t}"
            )
            return events  # slot freed, no retry

        # Schedule a retry — this works for both PROCESSING and FORWARDED jobs.
        # For FORWARDED jobs, the retry re-enters this server's service queue
        # and will be forwarded downstream again when it completes.
        job.server_attempts[self.id] += 1
        self.retries += 1

        retry_job = job.clone_for_retry(t)
        retry_job.retry_origin = RetryOrigin.SERVER

        logger.info(
            f"Server {self.id} slot {thread_id}: timeout → retry attempt "
            f"{retry_job.attempt_id} for id={job.request_id} "
            f"(was {'FORWARDED' if job.status == JobStatus.FORWARDED else 'PROCESSING'}) "
            f"at t={t}"
        )
        # logger.info(
        #     f" Retry {attempts+1} for request {job.request_id} on Server {self.id}, "
        # )

        offered = self.offer(retry_job, t + self.retry_delay)

        if offered:
            #return offered if isinstance(offered, list) else [offered]
            events.extend(offered if isinstance(offered, list) else [offered])
        return events
