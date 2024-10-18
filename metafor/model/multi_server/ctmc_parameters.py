from dataclasses import dataclass
from typing import List


@dataclass
class MultiServerCTMCParameters:
    server_num: int  # number of servers
    main_queue_sizes: List[int]
    retry_queue_sizes: List[int]
    lambda_init: List[float]
    mu0_ps: List[float]
    timeouts: List[int]
    max_retries: List[int]
    thread_pools: List[int]
    config_set: List[List[float]]
    lambda_fault: List[float]
    fault_time: float
    lambda_reset: List[float]
    reset_time: float
    sub_tree_list: List[List[int]]
    parent_list: List[List[int]]
    q_min_list: List[int]
    q_max_list: List[int]
    o_min_list: List[int]
    o_max_list: List[int]
    state_num: List[int]
    state_num_prod: int


