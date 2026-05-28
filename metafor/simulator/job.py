import random
from abc import ABC
from typing import Type, List

import logging
logger = logging.getLogger(__name__)
import uuid
from collections import defaultdict
import copy
from enum import Enum
import math
from itertools import accumulate
import bisect

class Distribution(ABC):
    def __init__(self):
        self.name = "ExponentialDistribution"
        pass

    def sample(self) -> float:
        return 0

class ExponentialDistribution(Distribution):

    def __init__(self, rate: float):
        self.rate = rate            # λ — the expovariate parameter
        self.name = "ExponentialDistribution"

    @property
    def mean(self) -> float:
        return 1.0 / self.rate      # true mean of the distribution

    def sample(self) -> float:
        return random.expovariate(self.rate)


class MixtureOfExponentials:
    DEFAULT_WEIGHTS = (0.70, 0.20, 0.10)
    DEFAULT_MULTIPLIERS = (1.5, 0.8, 0.15)

    def __init__(
        self,
        base_rate: float,
        weights: tuple[float, ...] | None = None,
        multipliers: tuple[float, ...] | None = None,
    ):
        weights = weights if weights is not None else self.DEFAULT_WEIGHTS
        multipliers = multipliers if multipliers is not None else self.DEFAULT_MULTIPLIERS

        if not math.isclose(sum(weights), 1.0, rel_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {sum(weights):.6f}")
        self.name= "MixtureOfExponentials"
        self.weights = weights

        # Normalise multipliers so that E[X] = 1/base_rate exactly.
        raw_mean_scale = sum(w / m for w, m in zip(weights, multipliers))
        self._multipliers = tuple(m / raw_mean_scale for m in multipliers)

        self._cumulative = tuple(accumulate(weights))
        self.rates: list[float] = []
        self.set_rho(base_rate)

    @property
    def mean(self) -> float:
        return sum(w / r for w, r in zip(self.weights, self.rates))

    def set_rho(self, base_rate: float) -> None:
        """
        Rescale all component rates to a new base rate, preserving mixture shape.
        Safe to call at any point during simulation — takes effect on the next sample().
        """
        if base_rate <= 0:
            raise ValueError(f"base_rate must be positive, got {base_rate}")
        self.rates = [base_rate * m for m in self._multipliers]

    def sample(self) -> float:
        u = random.random()
        k = bisect.bisect_left(self._cumulative, u)
        k = min(k, len(self.rates) - 1)
        return random.expovariate(self.rates[k])



class NormalLoadMixture(MixtureOfExponentials):
    DEFAULT_WEIGHTS      = [0.75, 0.20, 0.05]
    DEFAULT_MULTIPLIERS  = [1.5,  0.8,  0.10]
    # mostly fast, rare slow bursts

class FaultLoadMixture(MixtureOfExponentials):
    DEFAULT_WEIGHTS      = [0.50, 0.35, 0.15]
    DEFAULT_MULTIPLIERS  = [10.0,  0.06,  1.0]
    # higher peak rate, heavier tail — models request storm

class ResetLoadMixture(MixtureOfExponentials):
    DEFAULT_WEIGHTS      = [0.80, 0.15, 0.05]
    DEFAULT_MULTIPLIERS  = [1.2,  0.7,  0.20]
    # settling back toward normal, moderate tail

    
class WeibullDistribution(Distribution):
    def __init__(self, rate: float):
        self.rate = rate

    @property
    def mean(self) -> float:
        return 1.0 / self.rate

    def sample(self) -> float:
        return random.weibullvariate(self.mean, 1.0)  # scale=mean, shape=1




class NormalDisttribution(Distribution):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self):
        return max(0, random.gauss(self.mean, self.std))


class LogNormalDistribution(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        return random.lognormvariate(self.mu, self.sigma)


class RetryOrigin(Enum):
    NONE = "none"       # original attempt
    CLIENT = "client"   # client timeout fired
    SERVER = "server"   # server-level timeout fired

class DropReason(Enum):
    NONE = "none"           # not dropped
    QUEUE_FULL = "queue_full"
    TOKEN_BUCKET = "token_bucket"

class JobStatus:
    CREATED = 0
    ENQUEUED = 1
    PROCESSING = 2
    COMPLETED = 3
    DROPPED = 4
    FORWARDED = 5

    @staticmethod
    def __str__(status):
        m = { JobStatus.CREATED: 'created', 
             JobStatus.ENQUEUED: 'enqueued',
             JobStatus.PROCESSING: 'processing',
             JobStatus.COMPLETED: 'completed',
             JobStatus.DROPPED: 'dropped',
             JobStatus.FORWARDED: 'forwarded'
             }
        return m[status]


class Job(ABC):
    def __init__(
            self, 
            name: str, 
            timestamp: float, 
            max_retries: int = 0, 
            retries_left: int = 0,
            request_id: str | None = None,
            attempt_id: int = 0,
            trace_id:   str | None = None,
    ):
        self.created_t: float = timestamp
        self.completed_t: float = 0.0
        self.name = name
        self.status = JobStatus.CREATED
        self.max_retries: int = max_retries
        self.retries_left: int = retries_left
        self.size: float = 0
        self.request_id = request_id or str(uuid.uuid4())
        # trace_id is the single stable identifier for the entire end-to-end
        # request lifetime.  It is set once at creation (== request_id for the
        # root job) and copied verbatim through every clone — branch or retry.
        # Unlike request_id, it never changes, so you can grep a single value
        # to reconstruct the full S1→S2→S3 (→S4→S5) trace from the logs.
        self.trace_id   = trace_id or self.request_id
        self.attempt_id = attempt_id
        self.client = None

        # retry attempts per server
        self.server_attempts = defaultdict(int)
        self.retry_origin: RetryOrigin = RetryOrigin.NONE
        self.response_callback = None  # (fn, slot_n, next_callback) | None

    def __str__(self):
        return "[%s: created %f, status: %s]" % (self.name, self.created_t, JobStatus.__str__(self.status))
    
    @staticmethod
    def mean() -> float:
        """Return the mean service time for this job type."""
        pass
    
    def clone_for_branch(self, t: float) -> "Job":
        """
        Produce an independent branch copy for fan-out forwarding.

        Rules
        ─────
        created_t   preserved  — keeps end-to-end latency measurement correct
                                 across the full request lifetime.
        request_id  NEW uuid   — each branch must have a unique identity so
                                 that downstream servers (e.g. S5 with fan-in
                                 from S3 and S4) do not discard the second
                                 arrival via the forwarded_request_ids guard.
        attempt_id  reset to 0 — this is a fresh attempt on the new branch;
                                 retry counting starts from zero.
        server_attempts  copied independently — avoids cross-branch mutation.
        response_callback  None — upstream Server sets the correct
                                  JoinTracker-bearing callback after cloning.
        """
        new = copy.copy(self)
        new.completed_t = 0
        new.status = JobStatus.CREATED

        # Independent identity — critical for fan-in deduplication at S5.
        new.request_id = str(uuid.uuid4())
        new.attempt_id = 0
        # trace_id is the one stable handle for the whole end-to-end request.
        # It must survive every hop so logs can be reconstructed per trace.
        new.trace_id   = self.trace_id

        # Independent retry tracking per branch.
        new.server_attempts = self.server_attempts.copy()

        # Explicitly cleared; Server.job_done sets the correct callback.
        new.response_callback = None

        return new
    
    def clone_for_retry(self, t: float) -> "Job":

        new = copy.copy(self)

        new.completed_t = 0
        new.status = JobStatus.CREATED

        new.server_attempts = self.server_attempts.copy()

        new.trace_id   = self.trace_id   # same logical request, new attempt
        new.attempt_id += 1

        return new

    def is_retry(self):
        return self.attempt_id > 0


# Job with unimodal exponentially distributed latency
def exp_job(mean: float) -> Type[Job]:
    class ExpJob(Job):
        def __init__(self, t: float, max_retries: int = 0, retries_left: int = 0):
            super().__init__(t, max_retries, retries_left)
            self.size = random.expovariate(1.0 / mean)
            self.name = "ExponentialDistribution"

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
