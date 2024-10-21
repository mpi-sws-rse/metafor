from dataclasses import dataclass, field
from typing import List


@dataclass
class PlotParameters:
    step_time: int = 500
    sim_time: int = 6000
    qlen_max: int = 150
    qlen_step: int = 10
    olen_max: int = 60
    olen_step: int = 10
    retry_max: int = 10
    retry_step: int = 1
    lambda_max: float = 12
    lambda_min: float = 8
    lambda_step: float = 0.5
    mu_max: float = 12
    mu_min: float = 8
    mu_step: float = 0.5
    reset_lambda_max: float = 10
    reset_lambda_min: float = 6
    reset_lambda_step: float = 0.5
    lambda_fault: List[float] = field(default_factory=lambda: [20])
    lambda_reset: List[float] = field(default_factory=lambda: [8, 0.25])
    config_set: List[List[float]] = field(
        default_factory=lambda: [[8, 0.5], [8, 4.5], [8, 0.25]]
    )
    start_time_fault: int = 200
    duration_fault: int = 200
    reset_time: float = 150
    timeout_max: int = 15
    timeout_min: int = 5
