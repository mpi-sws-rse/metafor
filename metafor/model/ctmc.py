import math
from abc import ABC, abstractmethod
from typing import Optional

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
