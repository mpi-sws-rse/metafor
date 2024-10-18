from dataclasses import dataclass
from typing import List


@dataclass
class CTMCParameters:
    main_queue_size: int
    retry_queue_size: int
    lambdaa: float
    lambdaas: List[float]  # arrival rates for different types of jobs
    state_num: int
    mu_retry_base: float
    mu_drop_base: float
    mu0_p: int
    mu0_ps: List[float]  # processing rates for different types of jobs
    timeout: int
    timeouts: List[int]  # timeouts for different types of jobs
    max_retries: int
    retries: List[int]  # retries for different types of jobs
    thread_pool: int
    alpha: float
