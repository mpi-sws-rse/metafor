from abc import ABC, abstractmethod


class CTMC(ABC):

    @abstractmethod
    def main_queue_average_size(self, pi) -> float:
        pass
