from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import numpy.typing as npt


class CTMC(ABC):

    def __init__(self):
        self.pi: Optional[npt.NDArray[np.float64]] = None

    @abstractmethod
    def main_queue_size_average(self, pi: npt.NDArray[np.float64]) -> float:
        pass

    @abstractmethod
    def get_stationary_distribution(self) -> npt.NDArray[np.float64]:
        pass
