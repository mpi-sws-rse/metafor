
from itertools import dropwhile, takewhile
from typing import Dict, Optional
from metafor.analysis.experiment import Experiment, Parameter, ParameterList
from metafor.dsl.dsl import Constants, DependentCall, Program, Server, Source, Work
import math
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution

class Visualizer:
    def __init__(self, p: Program, num_x = 20, num_y = 20):
        self.p = p
        self.num_points_x = num_x
        self.num_points_y = num_y
        self.experiment = Experiment()

    def _compute_params(self, p: Program, qsize, osize):
        server = p.get_root_server()
        arrival_rates, service_rates, timeouts, retries, _ = p.get_params(server)
        arrival_rate = sum(arrival_rates)
        service_rate = 0
        timeout = 0
        max_retries = 0
        for lambdaa_i, mu0_p_i, timeout_i, retries_i in zip(
                arrival_rates, service_rates, timeouts, retries
        ):
            service_rate += mu0_p_i * (lambdaa_i / arrival_rate)
            timeout += timeout_i * (lambdaa_i / arrival_rate)
            max_retries += retries_i * (lambdaa_i / arrival_rate)
        thread_pool = server.thread_pool
        mu_retry_base = max_retries / ((max_retries + 1) * timeout)
        mu_drop_base = 1 / ((max_retries + 1) * timeout)
        tail_prob = self._tail_prob_computer(qsize, service_rate, timeout, thread_pool)
        # tail_prob = self._tail_prob_computer(p, server, qsize, osize)
        return {
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'mu_retry_base': mu_retry_base,
            'mu_drop_base': mu_drop_base,
            'tail_prob': tail_prob,
            'thread_pool' : thread_pool,
        }

    # Multi server
    def _get_service_rate(self, server: Server) -> Work:
        # we assume exactly one API call
        assert(len(server.apis) == 1), "Server %s has more than one API call" % server.name
        apiname = list(server.apis)[0]
        apiwork: Work = server.get_work(apiname)
        return apiwork.processing_rate
    
    # Multi server
    def _get_exogeneous_arrival_rate(self, p: Program, server: Server) -> float:
        # return the arrival rate if there are additional sources pushing API calls into this server
        sources = p.get_sources(server)
        return sum([s.arrival_rate for s in sources])

    # Multi server
    def _compute_effective_service_rate(self, _p: Program, server: Server, downstream: list[Server], qsizes: Dict[str, int]):
        # calculate the effective processing rate of a server making waiting dependent calls
        # assuming the queue sizes `qsizes` mapping server names to queue lengths
        # `server` is the server for which the rate is computed.
        # `downstream` is the list of transitively called servers
        #
        # We assume there is exactly one API call being handled
        # essentially, the effective service rate is the minimum over all service rates of the server and its 
        # downstream dependencies

        rate = self._get_service_rate(server) * min(1, qsizes.get(server.name, 0), server.thread_pool)
        # now we iterate over the downstream servers and find the minimal processing rate: that is the effective rate
        # for our server
        for s in downstream:
            srate = self._get_service_rate(s) * min(1, qsizes.get(s.name, 0), s.thread_pool)
            # we make sure the min is at least one, to avoid being stuck if the queue size is 0
            rate = min(rate, srate)
        return rate
    
    # Multi server
    def _compute_effective_arrival_rate(self, p: Program, server: Server, all_servers: list[Server], qsizes: Dict[str, int]):
        # calculate the effective arrival rate at a server
        # `upstream` is the list of all upstream servers
        print("In compute_effective_arrival_rate: ")
        print("Server: ", server)
        print("All servers: ", list(map(lambda s: s.name, all_servers)))

        rate = self._get_exogeneous_arrival_rate(p, server)
        upstream = list(takewhile(lambda s: s.name != server.name, all_servers))
        print("Upstream: ", list(map(lambda s: s.name, upstream)))

        # compute the rate of jobs added by the upstream/ancestor nodes
        added_lambda = 0
        for node in upstream:
            effective_arr_rate_node = added_lambda + self._get_exogeneous_arrival_rate(p, node)
            node_downstream = list(dropwhile(lambda s: s.name != node.name, upstream))[1:] # take downstream nodes, and drop `node` from the list
            effective_proc_rate_node = self._compute_effective_service_rate(p, node, node_downstream, qsizes)
            added_lambda = min(effective_arr_rate_node, effective_proc_rate_node)

        rate += added_lambda
        return rate

    def _compute_retries_and_timeout(self, p: Program, all_servers: list[Server], s: Server):
        if s.name == p.get_root_server().name:
            # for the root, the retry/timeouts are driven by the exogenous source
            sources = p.get_sources(s)
            assert len(sources) == 1
            return sources[0].retries, sources[0].timeout
        else:
            # for the other servers, the retry/timeouts are given by the dependent calls from the upstream server
            # find the caller
            caller = p.get_root_server()
            for server in all_servers[1:]:
                if server.name == s.name:
                    break
                else:
                    caller = server
            assert (len(caller.apis) == 1), "Server %s has more than one API call" % server.name
            print("Caller: ", caller)
            apiname = list(server.apis)[0]
            downstream = caller.get_work(apiname).downstream
            assert (len(downstream) == 1)
            return (downstream[0].retry, downstream[0].timeout)

    # Multi server
    def _compute_params_general(self, p: Program, server: Server, qsizes: Dict[str, int], osizes: Dict[str, int]):
        all_servers = self._get_topological_list(p)
        
        downstream = list(dropwhile(lambda s: s.name != server.name, all_servers))[1:]
        arrival_rate = self._compute_effective_arrival_rate(p, server, all_servers, qsizes)
        service_rate = self._compute_effective_service_rate(p, server, downstream, qsizes)
        thread_pool = server.thread_pool
        
        """def _compute_retries_and_timeout(p: Program, all_servers: list[Server], s: Server):
            if s.name == p.get_root_server().name:
                # for the root, the retry/timeouts are driven by the exogenous source
                sources = p.get_sources(s)
                assert len(sources) == 1
                return sources[0].retries, sources[0].timeout
            else:
                # for the other servers, the retry/timeouts are given by the dependent calls from the upstream server
                # find the caller
                caller = p.get_root_server()
                for server in all_servers[1:]:
                    if server.name == s.name:
                        break
                    else:
                        caller = server
                assert(len(caller.apis) == 1), "Server %s has more than one API call" % server.name
                print("Caller: ", caller)
                apiname = list(server.apis)[0]
                downstream = caller.get_work(apiname).downstream
                assert(len(downstream) == 1)
                return (downstream[0].retry, downstream[0].timeout)"""
                
                
        (retries, timeout) = self._compute_retries_and_timeout(p, all_servers, server)

        mu_retry_base = retries / ((retries + 1) * timeout)
        mu_drop_base = 1 / ((retries + 1) * timeout)
        #tail_prob = [0.5] * server.qsize # XXX TODO
        # TODO: calculate tail prob self._tail_prob_computer(total_ind, mu0_p, timeout, thread_pool, qsize, osize)[node_id]
        tail_prob = self._tail_prob_computer_general(p, server, qsizes, osizes)
        return{
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'mu_retry_base': mu_retry_base,
            'mu_drop_base': mu_drop_base,
            'tail_prob': tail_prob,
            'thread_pool': thread_pool,
        }

    def visualize(self, param: Optional[ParameterList] = None, qrange=None, orange=None, show_equilibrium=True):
        num_servers = self.p.get_number_of_servers()

        # we assume there are up to two servers for the moment :-)
        if num_servers == 1:
            server = self.p.get_root_server()
            qsize = qrange if qrange is not None else server.qsize
            osize = orange if orange is not None else server.orbit_size
        
            if param is None:
                self.viz_2d(self.p, qsize, osize, show_equilibrium=show_equilibrium)
            else:
                for pval in param:
                    modified_program = self.experiment.update(self.p, pval)
                    modified_program.print()
                    self.viz_2d(modified_program, qsize, osize, show_equilibrium)
        else:
            assert False, "Visualize called with a multi-server program. Please call `visualize_general` instead"


    def visualize_general(self, param: Optional[ParameterList], 
                          qrange: Dict[str, int],
                          orange: Dict[str, int],                  
                          show_equilibrium=False):
        if param is None:
            self.viz_general(self.p, qrange, orange, show_equilibrium=show_equilibrium)
        else:
            for pval in param:
                modified_program = self.experiment.update(self.p, pval)
                modified_program.print()
                self.viz_general(modified_program, qrange, orange, show_equilibrium)


    def _check_program(self, p: Program):
        root = p.get_root_server()
        # 1. check that all source -> server connections are to the root server
        cxns = p.connections
        for c in cxns:
            assert c[1] == root.name, "All connections should be to root"
        # 2. check that the servers are arranged in a line 
        
        s = root
        callees = p.get_callees(s)
        while len(callees) == 1:
            s = list(callees)[0]
            # 3. check that dependent calls wait until the downstream calls are done 
            # (i.e., call_type is WAIT_UNTIL_DONE for all dependent calls)
            for work in s.apis.values():
                assert len(work.downstream) <= 1, "More than one downstream work"
                if len(work.downstream) == 1:
                    assert work.downstream[0].call_type == Constants.WAIT_UNTIL_DONE, "server is not waiting for a dependent call"
            callees = p.get_callees(s)
        assert len(callees) == 0, "Server %s calls multiple other servers: program is not linear" % s.name
        


    def _get_topological_list(self, p: Program) -> list[Server]:
        # get a topological list of servers
        root = p.get_root_server()

        servers = [root]
        callees = p.get_callees(root)
        # we assume that the servers are a linear chain: | callees | == 1
        while len(callees) == 1:
            s = list(callees)[0]
            servers.append(s)
            callees = p.get_callees(s)
        # at this point servers is topologically sorted, with root at the beginning
        return servers
    

    def viz_general_server(self, p: Program, server: Server, qsizes: Dict[str, int], osizes: Dict[str, int], show_equilibrium=True):
        # visualize the dynamics of `server` in the program `p`, 
        # assuming queue bounds for the other servers are given by the map `qsizes` (for queue size) and `osizes` (for orbit size)
        params = self._compute_params_general(p, server, qsizes, osizes)

        qsize = qsizes[server.name]
        osize = osizes[server.name]
        if qsize > osize:
            x_to_y_range = int(qsize / osize)
        else:
            x_to_y_range = 1
            assert False, "For visualization, set queue size > orbit size (revisit this assumption)"
    
        params = self._compute_params_general(p, server, qsizes, osizes)
        arrival_rate = params['arrival_rate']
        service_rate = params['service_rate']
        mu_retry_base = params['mu_retry_base']
        mu_drop_base = params['mu_drop_base']
        tail_prob = params['tail_prob']
        thread_pool = params['thread_pool']
        
        # Downsample the i and j ranges for better visibility
        i_values = np.linspace(0, qsize/x_to_y_range, self.num_points_x, endpoint=False)  #
        j_values = np.linspace(0, osize, self.num_points_y, endpoint=False)  #
    
        # Create meshgrid for i and j values
        I, J = np.meshgrid(i_values, j_values)
    
        # Create arrays for the horizontal (U) and vertical (V) components
        U = np.zeros(I.shape)  # Horizontal component
        V = np.zeros(I.shape)  # Vertical component
        
        # Compute magnitudes and angles for each (i, j)
        for idx_i, i in enumerate(i_values):
            for idx_j, j in enumerate(j_values):
                U[idx_j, idx_i] = self.q_rate_computer(
                    int(i * x_to_y_range), 
                    int(j), 
                    arrival_rate, 
                    service_rate, 
                    mu_retry_base,
                    thread_pool)
                V[idx_j, idx_i] = self.o_rate_computer(
                    int(i * x_to_y_range), 
                    int(j), 
                    arrival_rate, 
                    mu_retry_base, 
                    mu_drop_base,
                    tail_prob)
        # Compute magnitude (for color) and angle (for arrow direction)
        magnitude = np.sqrt(U ** 2 + V ** 2)  # Magnitude of the vector
        angle = np.arctan2(V, U)  # Angle of the vector (atan2 handles f_x=0 correctly)

        # Find the maximum absolute values
        max_mag = np.max(magnitude)

        # Normalize the horizontal (U) and vertical (V) components by the maximum values
        # magnitude_normalized = (magnitude / max_mag)

        # Define a fixed maximum arrow length for visibility
        fixed_max_length =  qsize / (x_to_y_range * max(self.num_points_x, self.num_points_y))


        # Flatten the arrays for plotting
        I_flat = I.flatten()
        J_flat = J.flatten()
        U_flat = np.cos(angle).flatten() * fixed_max_length # Normalize the direction to length fixed_max_length
        V_flat = np.sin(angle).flatten() * fixed_max_length # Normalize the direction to length fixed_max_length
        # magnitude_flat = magnitude_normalized.flatten()
        magnitude_flat = magnitude.flatten()

        # Plotting
        fig, ax = plt.subplots(figsize=(9, 6))

        # Create a colormap for the arrow colors based on the magnitude
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.min(magnitude_flat), vmax=np.max(magnitude_flat))
        colors = cmap(norm(magnitude_flat))

        # Plot the arrows using the fixed length and color by magnitude
        _ = ax.quiver(I_flat, J_flat, U_flat, V_flat, color=colors,
                        angles='xy', scale_units='xy', scale=1, width=0.003)

        # Add a colorbar based on the magnitude
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(magnitude_flat)  # Link the data to the ScalarMappable
        cbar = plt.colorbar(sm, ax=ax)  # Attach the colorbar to the current axis

        if show_equilibrium:
            # Create a circle at the (almost) equilibrium point
            res, obj_val = self.equilibrium_computer(qsize, osize, arrival_rate, service_rate, mu_retry_base, mu_drop_base,
                                        thread_pool,
                                        tail_prob)
            if abs(obj_val) < .01:
                print("found an almost equilibrium point")
                circle = plt.Circle((res.x[0]/x_to_y_range, res.x[1]), .01 * qsize/x_to_y_range, color='red', fill=True)
                ax.add_artist(circle)

        # Get current tick positions on the x-axis
        xticks = ax.get_xticks()

        # Re-scale the tick labels to the correct numbers
        scaled_xticks = xticks * x_to_y_range
        scaled_xticks.astype(int)

        # Set the new scaled tick labels
        ax.set_xticklabels(scaled_xticks)

        # Set labels for the axes
        ax.set_xlabel('Queue length')
        ax.set_ylabel('Orbit length')

        # Display the plot
        plt.show()
        # plt.savefig("2D")

    def viz_general(self, p: Program, qsizes: Dict[str, int], osizes: Dict[str, int], show_equilibrium=True):
        p.print()
        # We currently assume that the servers are arranged linearly, and only the first server has exogeneous arrivals
        self._check_program(p)

        servers = self._get_topological_list(p)
        print('Servers:', servers)
        # check that qrange and orange both have an entry for each server
        # and if not, add values for each server
        for server in servers:
            q = qsizes.get(server.name, None)
            if q is None:
                qsizes[server.name] = server.qsize - 1
            o = osizes.get(server.name, None)
            if o is None:
                osizes[server.name] = server.orbit_size - 1

        for server in servers:
            self.viz_general_server(p, server, qsizes, osizes, show_equilibrium)
 


    def viz_2d(self, p, qsize, osize, show_equilibrium=True):
        p.print()
        params = self._compute_params(p, qsize, osize)        
        arrival_rate = params['arrival_rate']
        service_rate = params['service_rate']
        mu_retry_base = params['mu_retry_base']
        mu_drop_base = params['mu_drop_base']
        tail_prob = params['tail_prob']
        thread_pool = params['thread_pool']

        if qsize > osize:
            x_to_y_range = int(qsize/osize) # ensure that qsize is a multiple of osize
        else:
            assert False, "For visualization, set queue size > orbit size (revisit this assumption)"

        i_max = qsize/x_to_y_range # used to make the plot x&y coordinates of arrow sizes reasonable
        j_max = osize


        # Downsample the i and j ranges for better visibility
        i_values = np.linspace(0, i_max, self.num_points_x, endpoint=False)  #
        j_values = np.linspace(0, j_max, self.num_points_y, endpoint=False)  #
        
        # Create meshgrid for i and j values
        I, J = np.meshgrid(i_values, j_values)
        
        # Create arrays for the horizontal (U) and vertical (V) components
        U = np.zeros(I.shape)  # Horizontal component
        V = np.zeros(I.shape)  # Vertical component

        # Compute magnitudes and angles for each (i, j)
        for idx_i, i in enumerate(i_values):
            for idx_j, j in enumerate(j_values):
                U[idx_j, idx_i] = self.q_rate_computer(int(i*x_to_y_range), int(j), arrival_rate, service_rate, mu_retry_base,
                                                       thread_pool)
                V[idx_j, idx_i] = self.o_rate_computer(int(i*x_to_y_range), int(j), arrival_rate, mu_retry_base, mu_drop_base,
                                                       tail_prob)
                """U[idx_j, idx_i], V[idx_j, idx_i] = self.dominant_trans_finder(int(i), int(j), arrival_rate, service_rate, mu_retry_base, mu_drop_base,
                                      thread_pool, tail_prob)"""

        # Compute magnitude (for color) and angle (for arrow direction)
        magnitude = np.sqrt(U ** 2 + V ** 2)  # Magnitude of the vector
        angle = np.arctan2(V, U)  # Angle of the vector (atan2 handles f_x=0 correctly)

        # Find the maximum absolute values
        max_mag = np.max(magnitude)

        # Normalize the horizontal (U) and vertical (V) components by the maximum values
        # magnitude_normalized = (magnitude / max_mag)

        # Define a fixed maximum arrow length for visibility
        fixed_max_length =  i_max / max(self.num_points_x, self.num_points_y)


        # Flatten the arrays for plotting
        I_flat = I.flatten()
        J_flat = J.flatten()
        U_flat = np.cos(angle).flatten() * fixed_max_length # Normalize the direction to length fixed_max_length
        V_flat = np.sin(angle).flatten() * fixed_max_length # Normalize the direction to length fixed_max_length
        # magnitude_flat = magnitude_normalized.flatten()
        magnitude_flat = magnitude.flatten()

        # Plotting
        fig, ax = plt.subplots(figsize=(9, 6))

        # Create a colormap for the arrow colors based on the magnitude
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=np.min(magnitude_flat), vmax=np.max(magnitude_flat))
        colors = cmap(norm(magnitude_flat))

        # Plot the arrows using the fixed length and color by magnitude
        _ = ax.quiver(I_flat, J_flat, U_flat, V_flat, color=colors,
                           angles='xy', scale_units='xy', scale=1, width=0.003)

        # Add a colorbar based on the magnitude
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(magnitude_flat)  # Link the data to the ScalarMappable
        cbar = plt.colorbar(sm, ax=ax)  # Attach the colorbar to the current axis

        if show_equilibrium:
            # Create a circle at the (almost) equilibrium point
            res, obj_val = self.equilibrium_computer(qsize, osize, arrival_rate, service_rate, mu_retry_base, mu_drop_base,
                                        thread_pool,
                                        tail_prob)
            if abs(obj_val) < .01:
                print("found an almost equilibrium point")
                circle = plt.Circle((res.x[0]/x_to_y_range, res.x[1]), .01 * i_max, color='red', fill=True)
                ax.add_artist(circle)

        # Get current tick positions on the x-axis
        xticks = ax.get_xticks()

        # Re-scale the tick labels to the correct numbers
        scaled_xticks = xticks * x_to_y_range
        scaled_xticks.astype(int)

        # Set the new scaled tick labels
        ax.set_xticklabels(scaled_xticks)

        # Set labels for the axes
        ax.set_xlabel('Queue length')
        ax.set_ylabel('Orbit length')

        # Display the plot
        plt.show()
        # plt.savefig("2D")

    def equilibrium_computer(self, qsize, osize, arrival_rate, service_rate, mu_retry_base, mu_drop_base, thread_pool, tail_prob):
        bounds = [(0, qsize), (0, osize)]
        # x0 = [17000, 0]
        # constraints = []

        def objective(x):
            y1 = arrival_rate - service_rate * min(x[0],thread_pool) + mu_retry_base * x[1]
            y2 = arrival_rate * tail_prob[int(x[0])] - mu_retry_base * (1-tail_prob[int(x[0])]) * x[1] - mu_drop_base * x[1]
            return y1 ** 2 + y2 ** 2

        # def round_solution(x, convergence):
        #     return np.round(x).astype(int)

        #result = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=constraints,  options={'gtol': 1e-19})
        result =  differential_evolution(objective, bounds=bounds, strategy='best1bin')
        print("found solution is", result.x)
        print("obj value is", objective(result.x))
        print(result.message)
        return result, objective(result.x)

    def q_rate_computer(self, q, o, arrival_rate, service_rate, mu_retry_base, thread_pool):
        # compute the algebraic sum of rates along the x axis
        return arrival_rate - service_rate * min(q, thread_pool) + mu_retry_base * o

    def o_rate_computer(self, q, o, arrival_rate, mu_retry_base, mu_drop_base, tail_prob):
        # compute the algebraic sum of rates along the y axis
        return arrival_rate * tail_prob[q] - mu_retry_base * (1-tail_prob[q]) * o - mu_drop_base * o

    """def dominant_trans_finder(self, q, o, arrival_rate, service_rate, mu_retry_base, mu_drop_base, thread_pool,
                              tail_prob):
        # find the dominant transition
        if arrival_rate*(1-tail_prob[q])+ mu_retry_base*tail_prob[q]*o > max(service_rate * min(q,thread_pool), mu_retry_base*tail_prob[q]*o):
            u = 1
            v = 0
        elif service_rate * min(q,thread_pool) > max(arrival_rate, mu_retry_base*o):
            u = -1
            v = 0
        elif arrival_rate*tail_prob[q]+ mu_retry_base*tail_prob[q]*o > max(service_rate * min(q,thread_pool), mu_retry_base*(1-tail_prob[q])*o, arrival_rate*(1-tail_prob[q])+ mu_retry_base*tail_prob[q]*o):
            u = 1
            v = 1
        elif mu_retry_base*(1-tail_prob[q])*o > max(arrival_rate, service_rate * min(q,thread_pool), mu_retry_base*tail_prob[q]*o):
            u = 1
            v = -1
        else:
            assert False, "Unreachable point in visualize"
        return (u, v)
        """


    def _tail_prob_computer(self, qsize: float, service_rate: float, timeout: float, thread_pool: float) -> list[float]:
        """Compute the timeout probabilities for the case that service time is distributed exponentially."""

        tail_seq = [0]  # The timeout prob is zero when there is no job in the queue!
        """current_sum = 0
        last = 1
        for job_num in range(
                1, qsize
        ):  # compute the timeout prob for all different queue sizes.
            mu = min(job_num, thread_pool) * service_rate  # to remain close to the math symbol
            mu_x_timeout = mu * timeout
            exp_mu_timeout = math.exp(-mu_x_timeout)
            if exp_mu_timeout == 0:
                return [0] * qsize
            last = last * mu_x_timeout / job_num
            current_sum = current_sum + last
            tail_seq.append(current_sum * exp_mu_timeout)"""
        # exact method is unstable for large values...we overapproximate using chebyshev ineq!
        for job_num in range(1, qsize):  # compute the timeout prob for all different queue sizes.
            service_rate_effective = min(job_num, thread_pool) * service_rate
            ave = job_num / service_rate_effective
            var = job_num / (service_rate_effective**2)
            sigma = math.sqrt(var)
            if timeout - ave > sigma:
                k_inv = sigma / (timeout - ave)
                tail_seq.append(k_inv ** 2)
            else:
                tail_seq.append(1)
        return tail_seq


    def _tail_prob_computer_general(self, p: Program, server: Server, qsizes: Dict[str, int], osizes: Dict[str, int])  -> list[float]:
        """Compute the timeout probabilities for the case that service time is distributed exponentially."""
        all_servers = self._get_topological_list(p)
        downstream = list(dropwhile(lambda s: s.name != server.name, all_servers))[1:]
        arrival_rate = self._compute_effective_arrival_rate(p, server, all_servers, qsizes)
        service_rate = self._compute_effective_service_rate(p, server, downstream, qsizes)
        thread_pool = server.thread_pool
        tail_prob = []
        for job_num in range(0, qsizes[server.name]):  # compute the timeout prob for all different queue sizes.
            ave = 0
            var = 0
            ind = 0
            for node in downstream:
                downstream_node = downstream[ind:]
                service_rate_node= self._compute_effective_service_rate(p, node, downstream_node, qsizes)
                if service_rate_node != 0:
                    ave += qsizes[node.name] / service_rate_node
                    var += qsizes[node.name] * 1 / (service_rate_node ** 2)
                ind += 1  #
            sigma = math.sqrt(var)
            (_, timeout) = self._compute_retries_and_timeout(p, all_servers, server)
            if timeout - ave > sigma:
                k_inv = sigma / (timeout - ave)
                tail_prob.append(k_inv ** 2)
            else:
                tail_prob.append(1)
        return tail_prob

import unittest
from metafor.dsl.dsl import Server, Source, Work, Program
from numpy import linspace

class TestViz(unittest.TestCase):
    def program(self, retry_when_full=False):
        api = { "insert": Work(10, [],), "delete": Work(10, []) }
        server = Server("52", api, qsize=200, orbit_size=20, thread_pool=1)
        src1 = Source('client-i', 'insert', 4.75, timeout=9, retries=3)
        src2 = Source('client-d', 'delete', 4.75, timeout=9, retries=3)

        p = Program("Service52", retry_when_full=retry_when_full)

        p.add_server(server)
        p.add_sources([src1, src2])
        p.connect('client-i', '52')
        p.connect('client-d', '52')
        return p
    
    def multi_server_program(self):
        api1 = { "insert": Work(10, [ DependentCall(
            "tail", "head", "insert", Constants.WAIT_UNTIL_DONE, 3, 4
        )],) }
        server1 = Server("head", api1, qsize=200, orbit_size=20, thread_pool=10)
        
        api2 = { "insert": Work(10, [])}
        server2 = Server("tail", api2, 100, 10, thread_pool=1)

        src = Source('client-i', 'insert', 4.75, timeout=9, retries=3)

        p = Program("MultiServerProgram")

        p.add_server(server1)
        p.add_server(server2)
        p.add_sources([src])
        p.connect('client-i', 'head')
        return p

    def test_viz(self):
        v = Visualizer(self.program())
        v.visualize(param = None, show_equilibrium=False)
        
        v = Visualizer(self.program())
        p = Parameter(("server", "52", "api", "insert", "processing_rate"), linspace(9.75, 10.25, 5))
        v.visualize(param=ParameterList([p]))

    def test_viz_with_retry_when_full(self):
        v = Visualizer(self.program(retry_when_full=True))
        v.visualize(param=None, show_equilibrium=False)

        v = Visualizer(self.program())
        p = Parameter(("server", "52", "api", "insert", "processing_rate"), linspace(9.75, 10.25, 5))
        v.visualize(param=ParameterList([p]))

    def test_viz2(self):
        v = Visualizer(self.multi_server_program())
        v.visualize_general(param=None, qrange={}, orange={}, show_equilibrium=False)


if __name__ == '__main__':
    unittest.main()
