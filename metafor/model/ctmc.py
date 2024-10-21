from abc import ABC, abstractmethod
from typing import List


class CTMC(ABC):

    def __init__(self):
        self.pi: List[float] = None

    @abstractmethod
    def main_queue_average_size(self, pi) -> float:
        pass

    @abstractmethod
    def get_stationary_distribution(self) -> List[float]:
        pass
