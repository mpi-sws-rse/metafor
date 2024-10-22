import itertools
from abc import abstractmethod

from typing import Any, Iterable

from dsl.dsl import Program


# A parameter name is a tuple of strings, starting with "server" or "source"
# Typical parameter name would be
# `server, server_name, qsize`
# `server, server_name, api, api_name, processing_rate`
# `source:source_name:arrival_rate`
class ParameterName:
    def __init__(self, name: tuple[str]):
        assert(name[0] == "server" or name[0] == "source")

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

# class ServerParameter(Parameter):
#     def __init__(self, servername: str, paramname: str, values: Iterable[Any]):
#         super().__init__(paramname, values)
#         self.servername = servername

#     def get_name(self):
#         return self.servername
    
# class SourceParameter(Parameter):
#     def __init__(self, sourcename: str, paramname: str, values: Iterable[Any]):
#         super().__init__(paramname, values)
#         self.sourcename = sourcename

#     def get_name(self):
#         return self.sourcename


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
        server_params = param.get('server', None)
        source_params = param.get('source', None)

        if server_params is not None:
            for server_name, smap in server_params.items():
                s = p.get_server(server_name)
                if s is not None:
                    if 'qsize' in smap:
                        s.qsize = smap['qsize']
                    if 'orbit_size' in smap:
                        s.orbit_size = smap['orbit_size']
                    if 'thread_pool' in smap:
                        s.thread_pool = smap['thread_pool']
                    if 'api' in smap:
                        api_map = smap['api']
                        for api_name, api_update_map in api_map.items():
                            api = s.apis.get(api_name, None)
                            if api is not None:
                                if 'processing_rate' in api_update_map:
                                    api.processing_rate = api_update_map['processing_rate']
                                # downstream call modifications TBD
        if source_params is not None:
            for source_name, source_map in source_params.items():
                s = p.get_source(source_name)
                if s is not None:
                    if 'arrival_rate' in source_map:
                        s.arrival_rate = source_map['arrival_rate']
                    if 'timeout' in source_map:
                        s.timeout = source_map['timeout']
                    if 'retries' in source_map:
                        s.retries = source_map['retries']
        return p

    def sweep(self, plist: ParameterList):
        results = []
        for params in plist:
            print("Running experiment with parameters ", params)
            program = self.build(params)
            results.append(self.analyze(params, program))
        print("Sweep: \n", results)
        self.show(results)


# def test_fiddle_program():
#     apis = { 'rd': Work(10, []) }
#     s_conf = fiddle.Config(Server, name="server", apis=apis, qsize=100, orbit_size=20, thread_pool=1)
#     rd_src_conf = fiddle.Config(Source, name="reader", api_name="rd", arrival_rate=9.5, timeout=9, retries=3)
#     p_conf = fiddle.Config(Program, name="single_server")

#     s = fiddle.build(s_conf)
#     rd_src = fiddle.build(rd_src_conf)
#     p = fiddle.build(p_conf)
#     p.add_server(s)
#     p.add_source(rd_src)
#     p.connect("reader", "server")
#     p.print()

#     s_conf.qsize = 200
#     rd_src_conf.arrival_rate = 3
#     s = fiddle.build(s_conf)
#     rd_src = fiddle.build(rd_src_conf)
#     p = fiddle.build(p_conf)
#     p.add_server(s)
#     p.add_source(rd_src)
#     p.connect("reader", "server")
#     p.print()
