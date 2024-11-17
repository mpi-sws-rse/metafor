import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Any

import numpy as np
import numpy.typing as npt

class CTMCRepresentation:
    EXPLICIT = 0
    COO = 1
    CSC = 2
    LINOP = 3

class CTMC(ABC):

    def __init__(self):
        self.pi: Optional[npt.NDArray[np.float64]] = None

    @abstractmethod
    def get_init_state(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def compute_stationary_distribution(self) -> npt.NDArray[np.float64]:
        pass

    def get_stationary_distribution(self) -> npt.NDArray[np.float64]:
        if self.pi is None:
            self.pi = self.compute_stationary_distribution()
        return self.pi

    @abstractmethod
    def main_queue_size_average(self, pi: npt.NDArray[np.float64]):
        pass

    @abstractmethod
    def main_queue_size_variance(self, pi: npt.NDArray[np.float64], mean_queue_length):
        pass

    @abstractmethod
    def main_queue_size_std(self, pi, mean_queue_length):
        pass

    def main_queue_size_analysis(self, pi) -> Dict[str, float]:
        avg = self.main_queue_size_average(pi)
        var = self.main_queue_size_variance(pi, avg)
        std = self.main_queue_size_std(pi, avg)
        return {"avg": avg, "var": var, "std": std}

    @abstractmethod
    def retry_queue_size_average(self, pi: npt.NDArray[np.float64]):
        pass

    @abstractmethod
    def latency_average(self, pi: npt.NDArray[np.float64], req_type: int = 0):
        pass

    @abstractmethod
    def finite_time_analysis(self, pi0: npt.NDArray[np.float64], analyses: dict[str, Callable[[Any], Any]],
                             sim_time: int, sim_step: int):
        pass
