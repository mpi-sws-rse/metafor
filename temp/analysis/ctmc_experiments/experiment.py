import itertools
from abc import abstractmethod

import time
from typing import Any, Iterable

from dsl.dsl import Program


# A parameter name is a tuple of strings, starting with "server" or "source"
# Typical parameter name would be
# `server, server_name, qsize`
# `server, server_name, api, api_name, processing_rate`
# `source:source_name:arrival_rate`
class ParameterName:
    def __init__(self, name: tuple[str]):
        assert name[0] == "server" or name[0] == "source"

        self.name = name


class Parameter:
    def __init__(self, name: ParameterName, values: Iterable[Any]):
        self.name = name
        self.values = values

    def param_name(self):
        return self.name

    def __iter__(self):
        return self.values.__iter__()

    def as_list(self):
        return list(self.values)


def nested_map(keys, value):
    if len(keys) == 1:
        return {keys[0]: value}
    return {keys[0]: nested_map(keys[1:], value)}


def create_nested_map(pairs):
    result = {}
    for keys, value in pairs:
        current_map = nested_map(keys, value)
        # Merge current_map into the result map
        temp = result
        for key in keys[:-1]:  # Traverse to the second last key
            temp = temp.setdefault(key, {})
        temp[keys[-1]] = value  # Set the last key to the value
    return result

def extract_name_val(nestedmap):
    def extract_worker(name, nestedmap):
        if isinstance(nestedmap, dict):
            assert(len(nestedmap) == 1)
            for k, v in nestedmap.items():
                return extract_worker(name + (k, ), v)
        else:
            return (name, nestedmap)
    return extract_worker((), nestedmap)

class ParameterList:
    def __init__(self, paramlist: list[Parameter]):
        self.params = paramlist
        self.product = itertools.product(*paramlist)

    def __iter__(self):
        return self

    def __next__(self):
        tup = self.product.__next__()
        return create_nested_map(zip(map(lambda p: p.param_name(), self.params), tup))


class Experiment:

    @abstractmethod
    def build(self, params) -> Program:
        pass

    @abstractmethod
    def analyze(self, param_setting, _program):
        return []

    @abstractmethod
    def show(self, _results):
        pass

    def update(self, p: Program, param):
        server_params = param.get("server", None)
        source_params = param.get("source", None)

        if server_params is not None:
            for server_name, smap in server_params.items():
                s = p.get_server(server_name)
                if s is not None:
                    if "qsize" in smap:
                        s.qsize = smap["qsize"]
                    if "orbit_size" in smap:
                        s.orbit_size = smap["orbit_size"]
                    if "thread_pool" in smap:
                        s.thread_pool = smap["thread_pool"]
                    if "api" in smap:
                        api_map = smap["api"]
                        for api_name, api_update_map in api_map.items():
                            api = s.apis.get(api_name, None)
                            if api is not None:
                                if "processing_rate" in api_update_map:
                                    api.processing_rate = api_update_map[
                                        "processing_rate"
                                    ]
                                # downstream call modifications TBD
        if source_params is not None:
            for source_name, source_map in source_params.items():
                s = p.get_source(source_name)
                if s is not None:
                    if "arrival_rate" in source_map:
                        s.arrival_rate = source_map["arrival_rate"]
                    if "timeout" in source_map:
                        s.timeout = source_map["timeout"]
                    if "retries" in source_map:
                        s.retries = source_map["retries"]
        return p

    def sweep(self, plist: ParameterList):
        # NOTE: sweep mutates program structure
        results = []
        print("\n")
        start = time.time()
        for params in plist:
            print("==========================================================")
            print("Running experiment with parameters ", params)
            program = self.build(params)
            program.print()
            analyze_start = time.time()
            results.append(self.analyze(params, program))
            print("Analysis time = ", time.time() - analyze_start, " seconds")
        # print("Sweep: \n", results)
        print("Sweep finished in ", time.time() - start, " seconds.\n\n")
        print("==========================================================")
        self.show(results)
