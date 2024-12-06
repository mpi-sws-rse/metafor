
from typing import Optional
from metafor.analysis.experiment import Experiment, Parameter, ParameterList
from metafor.dsl.dsl import Program
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

    def _compute_params(self, p: Program, qsize, _osize):
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
        return {
            'arrival_rate': arrival_rate,
            'service_rate': service_rate,
            'mu_retry_base': mu_retry_base,
            'mu_drop_base': mu_drop_base,
            'tail_prob': tail_prob,
            'thread_pool' : thread_pool,
        }

    def visualize(self, param: Optional[ParameterList] = None, qrange=None, orange=None, show_equilibrium=True):
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
            assert False, "For visualization, set queue size > orbit size"

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
        fixed_max_length = .05 * i_max


        # Flatten the arrays for plotting
        I_flat = I.flatten()
        J_flat = J.flatten()
        U_flat = np.cos(angle).flatten() * fixed_max_length # Normalize the direction to length fixed_max_length
        V_flat = np.sin(angle).flatten() * fixed_max_length # Normalize the direction to length fixed_max_length
        # magnitude_flat = magnitude_normalized.flatten()
        magnitude_flat = magnitude.flatten()

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))

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


    def _tail_prob_computer(self, qsize: float, service_rate: float, timeout: float, thread_pool: float):
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

import unittest
from metafor.dsl.dsl import Server, Source, Work, Program
class TestViz(unittest.TestCase):
    def program(self):
        api = { "insert": Work(10, [],), "delete": Work(10, []) }
        server = Server("52", api, qsize=200, orbit_size=20, thread_pool=1)
        src1 = Source('client-i', 'insert', 4.75, timeout=9, retries=3)
        src2 = Source('client-d', 'delete', 4.75, timeout=9, retries=3)

        p = Program("Service52")

        p.add_server(server)
        p.add_sources([src1, src2])
        p.connect('client-i', '52')
        p.connect('client-d', '52')
        return p

    def test_viz(self):
        from numpy import linspace
        v = Visualizer(self.program())
        v.visualize(param = None, show_equilibrium=False)

        v = Visualizer(self.program())
        p = Parameter(("server", "52", "api", "insert", "processing_rate"), linspace(9.75, 10.25, 5))
        v.visualize(param=ParameterList([p]))


if __name__ == '__main__':
    unittest.main()
