import math
from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, Any

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix


class CTMCRepresentation:
    EXPLICIT = 0
    COO = 1
    CSC = 2
    CSR = 3
    LINOP = 3

class Matrix:
    def __init__(self, dim: int, representation: CTMCRepresentation = CTMCRepresentation.EXPLICIT):
        self.representation = representation  # not used yet
        self.dim = dim
        match self.representation:
            case CTMCRepresentation.EXPLICIT:
                try:
                    self.Q = np.zeros((self.dim, self.dim))
                except np._core._exceptions._ArrayMemoryError:
                    raise "State space (%d states) too large for an explicit representation. Try a sparse " \
                          "representation." % (self.dim)
            case CTMCRepresentation.COO | CTMCRepresentation.CSC | CTMCRepresentation.CSR:
                # since our matrices have a few entries in each row, we can optimize this representation as
                # an array of (column, data)
                # this will also improve row sum
                self.rows = []
                self.columns = []
                self.data = []
            case _:
                raise NotImplementedError

    def set(self, i, j, value):
        match self.representation:
            case CTMCRepresentation.EXPLICIT:
                self.Q[i][j] = value
            case CTMCRepresentation.COO | CTMCRepresentation.CSC | CTMCRepresentation.CSR :
                self.rows.append(i)
                self.columns.append(j)
                self.data.append(value)
            case _:
                raise NotImplementedError

    def matrix(self):
        match self.representation:
            case CTMCRepresentation.EXPLICIT:
                return self.Q
            case CTMCRepresentation.COO:
                return coo_matrix((self.data, (self.rows, self.columns)), shape=(self.dim, self.dim))
            case CTMCRepresentation.CSC:
                return csc_matrix((self.data, (self.rows, self.columns)), shape=(self.dim, self.dim))
            case CTMCRepresentation.CSR:
                return csr_matrix((self.data, (self.rows, self.columns)), shape=(self.dim, self.dim))
            case _:
                raise NotImplementedError

    def sum(self, index):
        match self.representation:
            case CTMCRepresentation.EXPLICIT:
                return np.sum(self.Q[index, :])
            case CTMCRepresentation.COO:
                sum = 0.0
                for (i, j, v) in zip(self.rows, self.columns, self.data):
                    if i == index:
                        sum += v
                return sum
            case _:
                raise NotImplementedError
            
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
