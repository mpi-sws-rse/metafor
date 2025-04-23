# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/simple_collapse_sim/sim.py

from math import sqrt
from dataclasses import dataclass
from typing import List


@dataclass
class StatData:
    start_t: float
    rho: float
    run_name: str
    service_time: float = 0.0
    qlen: int = 0
    retries: int = 0
    dropped: int = 0
    runtime: float = 0.0

    @staticmethod
    def header() -> str:
        return "t,rho,service_time,name,qlen,retries,dropped,runtime"

    def key(self) -> int:
        return int(self.start_t) + hash(self.run_name)

    def as_csv(self) -> str:
        return f"{int(self.start_t)},{self.rho},{self.service_time},{self.run_name},{self.qlen},{self.retries}," \
               f"{self.dropped},{self.runtime}"


# Calculate the mean of an array of values
def avg(v):
    return sum(v) / len(v)


def stat_data_average(stat_data_v: List[StatData]) -> StatData:
    data: StatData = StatData(stat_data_v[0].start_t, stat_data_v[0].rho, stat_data_v[0].run_name)
    data.service_time = avg([stat_data.service_time for stat_data in stat_data_v])
    data.qlen = avg([stat_data.qlen for stat_data in stat_data_v])
    data.retries = avg([stat_data.retries for stat_data in stat_data_v])
    data.dropped = avg([stat_data.dropped for stat_data in stat_data_v])
    data.runtime = avg([stat_data.runtime for stat_data in stat_data_v])
    return data


def stat_data_variance(stat_data_v: List[StatData]) -> StatData:
    data: StatData = StatData(stat_data_v[0].start_t, stat_data_v[0].rho, stat_data_v[0].run_name)
    average_data: StatData = stat_data_average(stat_data_v)

    data.service_time = avg([abs(stat_data.service_time - average_data.service_time) ** 2 for stat_data in stat_data_v])
    data.qlen = avg([abs(stat_data.qlen - average_data.qlen) ** 2 for stat_data in stat_data_v])
    data.retries = avg([abs(stat_data.retries - average_data.retries) ** 2 for stat_data in stat_data_v])
    data.dropped = avg([abs(stat_data.dropped - average_data.dropped) ** 2 for stat_data in stat_data_v])
    data.runtime = avg([abs(stat_data.runtime - average_data.runtime) ** 2 for stat_data in stat_data_v])
    return data


def stat_data_std_deviation(stat_data_v: List[StatData]) -> StatData:
    data: StatData = StatData(stat_data_v[0].start_t, stat_data_v[0].rho, stat_data_v[0].run_name)
    variance_data: StatData = stat_data_variance(stat_data_v)

    data.service_time = sqrt(variance_data.service_time)
    data.qlen = sqrt(variance_data.qlen)
    data.retries = sqrt(variance_data.retries)
    data.dropped = sqrt(variance_data.dropped)
    data.runtime = sqrt(variance_data.runtime)
    return data
