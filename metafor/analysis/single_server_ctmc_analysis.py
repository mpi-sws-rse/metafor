import copy
import math
from typing import List

import numpy as np
import scipy

from model.single_server.ctmc import SingleServerCTMC
from utils.plot import (
    plot_results,
    trigger_plot_generator,
    plot_bar_data,
    plot_results_latency,
    plot_results_reset,
)
from utils.plot_parameters import PlotParameters


def average_lengths_analysis(
    ctmc: SingleServerCTMC, file_name: str, plot_params: PlotParameters
):
    print("Started the average lengths analysis")

    pi_q_new = np.zeros(ctmc.state_num)
    pi_q_new[0] = 1  # Initially the queue is empty
    pi_q_seq = [copy.copy(pi_q_new)]  # Initializing the initial distribution

    Q = ctmc.generator_mat_exact()
    (
        main_queue_avg_len,
        main_queue_var_len,
        main_queue_std_len,
        retry_queue_avg_len,
        runtime,
    ) = ctmc.server_population_queue_model(Q, pi_q_seq, plot_params)

    print("Main queue average length")
    print(main_queue_avg_len)
    print("Retry queue average length")
    print(retry_queue_avg_len)

    plot_results(
        plot_params.step_time,
        main_queue_avg_len,
        main_queue_var_len,
        main_queue_std_len,
        runtime,
        file_name,
    )

    ctmc.reach_prob_computation(Q, plot_params)

    print("Finished the average lengths analysis\n")


def fault_scenario_plot_generator(
    variable1: str,
    lambda_fault: int,
    fault_start_time: int,
    fault_duration: int,
    low: int,
    high: int,
    step: int,
    qlen0: int,
    retry_queue_size: int,
    lambdaas: List[int],
    mu0_ps: List[int],
    timeouts: List[int],
    retries: List[int],
    thread_pool: int,
    alpha: float,
    nominal_value: int,
    file_name: str,
):
    input_seq = []
    mean_rt_seq = []
    std_rt_seq = []
    lower_bound_rt_seq = []
    upper_bound_rt_seq = []
    qlen = 0
    olen = 0
    x_axis_label = ""
    main_color = ""
    fade_color = ""
    for val in np.arange(low, high + step, step):
        ctmc_before_fault = None
        ctmc_during_fault = None
        ctmc_after_fault = None
        if variable1 == "qlen":
            ctmc_before_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_during_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [lambda_fault] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_after_fault = SingleServerCTMC(
                val,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            qlen = val
            x_axis_label = "Queue length"
            main_color = "#A9A9A9"
            fade_color = "#D3D3D3"
        elif variable1 == "olen":
            ctmc_before_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_during_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [lambda_fault] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_after_fault = SingleServerCTMC(
                qlen0, val, lambdaas, mu0_ps, timeouts, retries, thread_pool, alpha
            )
            olen = val
            x_axis_label = "Orbit length"
            main_color = "tab:brown"
            fade_color = "#A52A2A"
        elif variable1 == "to":
            ctmc_before_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_during_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [lambda_fault] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_after_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                [val] * len(mu0_ps),
                retries,
                thread_pool,
                alpha,
            )
            x_axis_label = "Timeout"
            main_color = "tab:red"
            fade_color = "#FFCCCB"
        elif variable1 == "max_retries":
            ctmc_before_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_during_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [lambda_fault] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_after_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                val,
                thread_pool,
                alpha,
            )
            x_axis_label = "Maximum number of retries"
        elif variable1 == "lambda":
            ctmc_before_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_during_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [lambda_fault] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_after_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [val] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            x_axis_label = "Arrival rate"
            main_color = "#301934"  # 'tab:blue'
            fade_color = "#D8BFD8"  # "#ADD8E6"
        elif variable1 == "mu":
            ctmc_before_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_during_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [lambda_fault] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            ctmc_after_fault = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                [val] * len(lambdaas),
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            x_axis_label = "Processing rate"
            main_color = "#FF1493"  # 'tab:green'
            fade_color = "#FFB6C1"  # "#90EE90"
        Q_before_fault = ctmc_before_fault.generator_mat_exact()
        Q_during_fault = ctmc_during_fault.generator_mat_exact()
        Q_after_fault = ctmc_after_fault.generator_mat_exact()
        pi_0 = np.zeros(ctmc_before_fault.state_num)
        pi_0[0] = 1  # initially, no job in the buffer
        pi_before_fault = np.matmul(
            pi_0, scipy.linalg.expm(Q_before_fault * fault_start_time)
        )
        pi_after_fault = np.matmul(
            pi_before_fault, scipy.linalg.expm(Q_during_fault * fault_duration)
        )
        if qlen == 0:
            qlen = qlen0
        if olen == 0:
            olen = retry_queue_size
        mean_rt = ctmc_after_fault.hitting_time_average_us(
            Q_after_fault, pi_after_fault, math.floor(0.1 * qlen)
        )
        var_rt = ctmc_after_fault.hitting_time_variance_us(
            Q_after_fault, pi_after_fault, math.floor(0.1 * qlen)
        )
        std_rt = np.sqrt(var_rt)
        mean_rt_seq.append(mean_rt)
        std_rt_seq.append(std_rt)
        lower_bound_rt_seq.append(mean_rt - std_rt)
        upper_bound_rt_seq.append(mean_rt + std_rt)
        input_seq.append(val)
    plot_results_reset(
        input_seq,
        mean_rt_seq,
        lower_bound_rt_seq,
        upper_bound_rt_seq,
        x_axis_label,
        "Return time",
        "reset_" + variable1 + "_varied_" + file_name,
        main_color,
        fade_color,
        nominal_value,
    )


def fault_scenario_analysis(
    ctmc: SingleServerCTMC, file_name: str, plot_params: PlotParameters
):
    print("Started the fault scenario analysis")

    timeout_min = plot_params.timeout_min
    timeout_max = plot_params.timeout_max
    mu_min = plot_params.mu_min
    mu_max = plot_params.mu_max
    mu_step = plot_params.mu_step
    lambda_fault = plot_params.lambda_fault[0]
    start_time_fault = plot_params.start_time_fault
    duration_fault = plot_params.duration_fault
    reset_lambda_min = plot_params.reset_lambda_min
    reset_lambda_max = plot_params.reset_lambda_max
    reset_lambda_step = plot_params.reset_lambda_step

    # fault scenario plots
    trigger_plot_generator(
        lambda_fault, start_time_fault, duration_fault, ctmc.lambdaa, 10, 1000, file_name
    )

    fault_scenario_plot_generator(
        "mu",
        lambda_fault,
        start_time_fault,
        duration_fault,
        mu_min,
        mu_max,
        mu_step,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        10,
        file_name,
    )
    fault_scenario_plot_generator(
        "to",
        lambda_fault,
        start_time_fault,
        duration_fault,
        timeout_min,
        timeout_max,
        1,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        9,
        file_name,
    )
    fault_scenario_plot_generator(
        "lambda",
        lambda_fault,
        start_time_fault,
        duration_fault,
        reset_lambda_min,
        reset_lambda_max,
        reset_lambda_step,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        9.5,
        file_name,
    )
    print("Finished the fault scenario analysis\n")


def latency_plot_generator(
    variable1: str,
    low: int,
    high: int,
    step: int,
    qlen0: int,
    retry_queue_size: int,
    lambdaas: List[float],
    mu0_ps: List[int],
    timeouts: List[int],
    retries: List[int],
    thread_pool: int,
    alpha: float,
    file_name: str,
    job_type: int,
):
    input_seq = []
    mean_latency_seq = []
    std_latency_seq = []
    lower_bound_latency_seq = []
    upper_bound_latency_seq = []
    T_mixing_seq = []
    low_regime_prob_seq = []
    mid_regime_prob_seq = []
    high_regime_prob_seq = []
    mean_rt_seq_su = []
    mean_rt_seq_us = []
    std_rt_seq_su = []
    std_rt_seq_us = []
    lower_bound_rt_seq_su = []
    lower_bound_rt_seq_us = []
    upper_bound_rt_seq_su = []
    upper_bound_rt_seq_us = []
    qlen = 0
    olen = 0
    x_axis_label = ""
    main_color = ""
    fade_color = ""
    for val in np.arange(low, high + step, step):
        ctmc = None
        if variable1 == "qlen":
            ctmc = SingleServerCTMC(
                val,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            qlen = val
            x_axis_label = "Queue length"
            main_color = "#A9A9A9"
            fade_color = "#D3D3D3"
        elif variable1 == "olen":
            ctmc = SingleServerCTMC(
                qlen0, val, lambdaas, mu0_ps, timeouts, retries, thread_pool, alpha
            )
            olen = val
            x_axis_label = "Orbit length"
            main_color = "tab:brown"
            fade_color = "#A52A2A"
        elif variable1 == "to":
            ctmc = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                [val] * len(mu0_ps),
                retries,
                thread_pool,
                alpha,
            )
            x_axis_label = "Timeout"
            main_color = "tab:red"
            fade_color = "#FFCCCB"
        elif variable1 == "max_retries":
            ctmc = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                mu0_ps,
                timeouts,
                [val] * len(mu0_ps),
                thread_pool,
                alpha,
            )
            x_axis_label = "Maximum number of retries"
        elif variable1 == "lambda":
            ctmc = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                [val] * len(mu0_ps),
                mu0_ps,
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            x_axis_label = "Arrival rate"
            main_color = "#301934"  # 'tab:blue'
            fade_color = "#D8BFD8"  # "#ADD8E6"
        elif variable1 == "mu":
            ctmc = SingleServerCTMC(
                qlen0,
                retry_queue_size,
                lambdaas,
                [val] * len(lambdaas),
                timeouts,
                retries,
                thread_pool,
                alpha,
            )
            x_axis_label = "Processing rate"
            main_color = "#FF1493"  # 'tab:green'
            fade_color = "#FFB6C1"  # "#90EE90"
        Q = ctmc.generator_mat_exact()
        eigenvalues = np.linalg.eigvals(Q)
        sorted_eigenvalues = np.sort(eigenvalues.real)[::-1]
        sorted_eigenvalues[1] = sorted_eigenvalues[1]
        T_mixing = 1 / abs(sorted_eigenvalues[1]) * math.log2(100)
        QT = np.transpose(Q)
        ns = scipy.linalg.null_space(QT)
        pi_ss = ns / np.linalg.norm(ns, ord=1)
        if sum(pi_ss) < -0.001:
            pi_ss = -pi_ss
        if qlen == 0:
            qlen = qlen0
        if olen == 0:
            olen = retry_queue_size

        unstable_ind = ctmc._index_composer(int(0.9 * qlen), 0)
        stable_ind = ctmc._index_composer(int(0.1 * qlen), olen - 1)
        pi_unstable = np.zeros(ctmc.state_num)
        pi_unstable[unstable_ind] = 1
        pi_stable = np.zeros(ctmc.state_num)
        pi_stable[stable_ind] = 1
        mean_return_time_su = ctmc.hitting_time_average_su(
            Q, pi_stable, math.floor(0.9 * qlen)
        )
        mean_return_time_us = ctmc.hitting_time_average_us(
            Q, pi_unstable, math.floor(0.1 * qlen)
        )
        var_return_time_su = ctmc.hitting_time_variance_su(
            Q, pi_stable, math.floor(0.9 * qlen)
        )
        var_return_time_us = ctmc.hitting_time_variance_us(
            Q, pi_stable, math.floor(0.1 * qlen)
        )
        std_return_time_su = np.sqrt(var_return_time_su)
        std_return_time_us = np.sqrt(var_return_time_us)
        T_mixing_seq.append(T_mixing)
        mean_rt_seq_su.append(mean_return_time_su)
        mean_rt_seq_us.append(mean_return_time_us)
        std_rt_seq_su.append(std_return_time_su)
        std_rt_seq_us.append(std_return_time_us)
        lower_bound_rt_seq_su.append(mean_return_time_su - std_return_time_su)
        lower_bound_rt_seq_us.append(mean_return_time_us - std_return_time_us)
        upper_bound_rt_seq_su.append(mean_return_time_su + std_return_time_su)
        upper_bound_rt_seq_us.append(mean_return_time_us + std_return_time_us)
        low_regime_prob_seq.append(
            ctmc.prob_dist_accumulator(pi_ss, 0, math.ceil(qlen * 0.1), 0, olen)
        )
        mid_regime_prob_seq.append(
            ctmc.prob_dist_accumulator(
                pi_ss, math.ceil(qlen * 0.1), math.ceil(qlen * 0.9), 0, olen
            )
        )
        high_regime_prob_seq.append(
            ctmc.prob_dist_accumulator(pi_ss, math.ceil(qlen * 0.9), qlen, 0, olen)
        )
        mean_latency = ctmc.latency_average(pi_ss, job_type)
        var_latency = ctmc.latency_var(pi_ss, job_type)
        std_latency = ctmc.main_queue_size_std(var_latency)
        mean_latency_seq.append(mean_latency)
        std_latency_seq.append(std_latency)
        lower_bound_latency_seq.append(mean_latency - std_latency)
        upper_bound_latency_seq.append(mean_latency + std_latency)
        input_seq.append(val)
    job_info = ""
    if job_type != -1:
        job_info = (
            "lambda_" + str(lambdaas[job_type]) + "_mu_" + str(mu0_ps[job_type]) + "_"
        )
    plot_bar_data(
        step,
        input_seq,
        low_regime_prob_seq,
        high_regime_prob_seq,
        mean_rt_seq_su,
        mean_rt_seq_us,
        x_axis_label,
        "bar_" + variable1 + "_varied_" + job_info + file_name,
        "blue",
        "gray",
        "red",
    )
    plot_results_latency(
        input_seq,
        mean_latency_seq,
        lower_bound_latency_seq,
        upper_bound_latency_seq,
        x_axis_label,
        "Latency",
        "latency_" + variable1 + "_varied_" + job_info + file_name,
        main_color,
        fade_color,
    )


def latency_analysis(
    ctmc: SingleServerCTMC,
    file_name: str,
    plot_params: PlotParameters,
    job_type: int = -1,
):
    job_details = ""
    if job_type != -1:
        job_details = " for job " + str(job_type)
    print("Started latency analysis" + job_details)

    timeout_min = plot_params.timeout_min
    timeout_max = plot_params.timeout_max
    mu_min = plot_params.mu_min
    mu_max = plot_params.mu_max
    mu_step = plot_params.mu_step
    lambda_min = plot_params.lambda_min
    lambda_max = plot_params.lambda_max
    lambda_step = plot_params.lambda_step
    qlen_max = plot_params.qlen_max
    qlen_step = plot_params.qlen_step
    olen_max = plot_params.olen_max
    olen_step = plot_params.olen_step

    # latency analysis for retrial orbits
    print("Creating the latency plots")
    print("For the timeout")
    latency_plot_generator(
        "to",
        timeout_min,
        timeout_max,
        1,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )
    print("For the processing rate")
    latency_plot_generator(
        "mu",
        mu_min,
        mu_max,
        mu_step,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )
    print("For the arrival rate")
    latency_plot_generator(
        "lambda",
        lambda_min,
        lambda_max,
        lambda_step,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )
    print("For the main queue length")
    latency_plot_generator(
        "qlen",
        80,
        qlen_max,
        qlen_step,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )
    print("For the orbit length")
    latency_plot_generator(
        "olen",
        10,
        olen_max,
        olen_step,
        ctmc.main_queue_size,
        ctmc.retry_queue_size,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )

    # M/M/1 plots
    print("Creating the latency M/M/1 plots")
    print("For the processing rate")
    latency_plot_generator(
        "mu",
        mu_min,
        mu_max,
        mu_step,
        ctmc.main_queue_size,
        1,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )
    print("For the arrival rate")
    latency_plot_generator(
        "lambda",
        lambda_min,
        lambda_max,
        lambda_step,
        ctmc.main_queue_size,
        1,
        ctmc.lambdaas,
        ctmc.mu0_ps,
        ctmc.timeouts,
        ctmc.retries,
        ctmc.thread_pool,
        ctmc.alpha,
        file_name,
        job_type,
    )
    print("Finished latency analysis" + job_details + "\n")
