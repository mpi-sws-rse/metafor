# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import math
import numpy as np
import pandas
import time
from collections import deque
from typing import List, TextIO, Optional

from metafor.simulator.client import Client, OpenLoopClientWithTimeout
from metafor.simulator.job import Distribution, Job, JobStatus
from metafor.simulator.server import Server

import logging
logger = logging.getLogger(__name__)

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



class ServerWithLIFO(Server):
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
            self.queue: LIFOStack = LIFOStack()
