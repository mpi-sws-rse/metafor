import math
from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
import numpy.typing as npt


class CTMC(ABC):

    def __init__(self):
        self.pi: Optional[npt.NDArray[np.float64]] = None

    @abstractmethod
    def get_stationary_distribution(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def main_queue_size_average(self, pi: npt.NDArray[np.float64]):
        pass

    @abstractmethod
    def main_queue_size_variance(self, pi: npt.NDArray[np.float64], mean_queue_length):
        pass

    def main_queue_size_std(self, pi, mean_queue_length):
        return math.sqrt(self.main_queue_size_variance(pi, mean_queue_length))

    def main_queue_size_analysis(self, pi) -> Dict[str, float]:
        avg = self.main_queue_size_average(pi)
        var = self.main_queue_size_variance(pi, avg)
        std = self.main_queue_size_std(pi, avg)
        return {"avg": avg, "var": var, "std": std}
