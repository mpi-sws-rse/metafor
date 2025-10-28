from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def plot_results(
    step_time: int,
    latency_ave_len_seq: List[float],
    latency_var_len_seq: List[float],
    latency_std_len_seq: List[float],
    runtime_seq: List[float],
    main_queue_ave_len_seq: List[float],
    main_queue_var_len_seq: List[float],
    main_queue_std_len_seq: List[float],
    figure_name: str,
):
    """This function plots variations of four quantities over different time bounds:
    (1) mean queue length, (2) variance of queue length, (3) standard deviation of queue length, (4) runtime.
    """
    time = [i * step_time for i in list(range(0, len(main_queue_ave_len_seq)))]
    # Create 4x1 sub plots
    plt.rcParams["figure.figsize"] = [6, 10]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.5, hspace=0.5)

    color="tab:blue"
    ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
    ax1.plot(time, latency_ave_len_seq, color=color)
    ax1.set_xlabel("Time bound (ms)", fontsize=8)
    ax1.set_ylabel("Average Latency", fontsize=8, color=color)
    ax1.grid("on")
    ax1.set_xlim(0, max(time))
    ax1.tick_params(axis='y', labelcolor=color)

    ax11 = ax1.twinx()
    color="tab:red"
    ax1.set_ylabel("Average LateArrival rate", color=color)
    avg_arr_rate = np.ones(len(time))
    avg_arr_rate[np.arange(int(0.45*len(time)),int(0.55*len(time)))] = 10
    ax11.plot(time, avg_arr_rate, color=color)
    ax11.tick_params(axis='y',labelcolor=color)
    
    # ax2 = plt.subplot(ax[1, 0])  # row 3, col 0
    # ax2.plot(time, runtime_seq, color="tab:green")
    # ax2.set_xlabel("Time bound (ms)", fontsize=8)
    # ax2.set_ylabel("Runtime (sec)", fontsize=8)
    # ax2.grid("on")
    # ax2.set_xlim(0, max(time))

    ax3 = plt.subplot(ax[1, 0])  # row 4, col 0
    color="tab:blue"
    ax3.plot(time, main_queue_ave_len_seq, color="tab:blue")
    ax3.set_xlabel("Time bound (ms)", fontsize=8)
    ax3.set_ylabel("Average queue length", fontsize=8,color=color)
    ax3.grid("on")
    ax3.set_xlim(0, max(time))
    ax3.tick_params(axis='y', labelcolor=color)

    ax31 = ax3.twinx()
    color="tab:red"
    ax31.set_ylabel("Average LateArrival rate", color=color)
    avg_arr_rate = np.ones(len(time))
    avg_arr_rate[np.arange(int(0.45*len(time)),int(0.55*len(time)))] = 10
    ax31.plot(time, avg_arr_rate, color=color)
    ax31.tick_params(axis='y',labelcolor=color)
    #plt.show()
    plt.savefig(figure_name)
    plt.close()

    return


    ax = plt.GridSpec(7, 1)
    ax.update(wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
    ax1.plot(time, latency_ave_len_seq, color="tab:blue")
    ax1.set_xlabel("Time bound (ms)", fontsize=8)
    ax1.set_ylabel("Average Latency", fontsize=8)
    ax1.grid("on")
    ax1.set_xlim(0, max(time))

    ax2 = plt.subplot(ax[1, 0])  # row 1, col 0
    ax2.plot(time, latency_var_len_seq, color="tab:red")
    ax2.set_xlabel("Time bound (ms)", fontsize=8)
    ax2.set_ylabel("Variance of Latency", fontsize=8)
    ax2.grid("on")
    ax2.set_xlim(0, max(time))

    ax3 = plt.subplot(ax[2, 0])  # row 2, col 0
    ax3.plot(time, latency_std_len_seq, color="tab:purple")
    ax3.set_xlabel("Time bound (ms)", fontsize=8)
    ax3.set_ylabel("Standard deviation of Latency", fontsize=8)
    ax3.grid("on")
    ax3.set_xlim(0, max(time))
    
    ax4 = plt.subplot(ax[3, 0])  # row 3, col 0
    ax4.plot(time, runtime_seq, color="tab:green")
    ax4.set_xlabel("Time bound (ms)", fontsize=8)
    ax4.set_ylabel("Runtime (sec)", fontsize=8)
    ax4.grid("on")
    ax4.set_xlim(0, max(time))

    ax5 = plt.subplot(ax[4, 0])  # row 4, col 0
    ax5.plot(time, main_queue_ave_len_seq, color="tab:blue")
    ax5.set_xlabel("Time bound (ms)", fontsize=8)
    ax5.set_ylabel("Average queue length", fontsize=8)
    ax5.grid("on")
    ax5.set_xlim(0, max(time))

    ax6 = plt.subplot(ax[5, 0])  # row 5, col 0
    ax6.plot(time, main_queue_var_len_seq, color="tab:red")
    ax6.set_xlabel("Time bound (ms)", fontsize=8)
    ax6.set_ylabel("Variance of queue length", fontsize=8)
    ax6.grid("on")
    ax6.set_xlim(0, max(time))

    ax7 = plt.subplot(ax[6, 0])  # row 6, col 0
    ax7.plot(time, main_queue_std_len_seq, color="tab:purple")
    ax7.set_xlabel("Time bound (ms)", fontsize=8)
    ax7.set_ylabel("Standard deviation of queue length", fontsize=8)
    ax7.grid("on")
    ax7.set_xlim(0, max(time))


    plt.savefig(figure_name)
    plt.close()


def plot_main_queue_average_length(step_time, main_queue_ave_len_seq):
    """This function plots the average queue length for different time bounds."""
    time = [i * step_time for i in list(range(0, len(main_queue_ave_len_seq)))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, main_queue_ave_len_seq, color="tab:red")
    ax.set_xlabel("Time bound (ms)")
    ax.set_ylabel("Main queue average length")
    ax.grid("on")
    ax.set_xlim(0, max(time))


def plot_main_queue_var_length(step_time, main_queue_var_len_seq):
    """This function plots the variance of queue length for different time bounds."""
    time = [i * step_time for i in list(range(0, len(main_queue_var_len_seq)))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, main_queue_var_len_seq, color="tab:blue")
    ax.set_xlabel("Time bound (ms)")
    ax.set_ylabel("Variance of the queue length")
    ax.grid("on")
    ax.set_xlim(0, max(time))


def plot_main_queue_std_length(step_time, main_queue_std_len_seq):
    """This function plots the standard deviation of queue length for different time bounds."""
    time = [i * step_time for i in list(range(0, len(main_queue_std_len_seq)))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, main_queue_std_len_seq, color="tab:purple")
    ax.set_xlabel("Time bound (ms)")
    ax.set_ylabel("Standard deviation of the queue length")
    ax.grid("on")
    ax.set_xlim(0, max(time))


def plot_runtime(step_time, runtime_seq):
    """This function plots the runtime of our method for different time bounds."""
    time = [i * step_time for i in list(range(0, len(runtime_seq)))]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time, runtime_seq, color="tab:green")
    ax.set_xlabel("Time bound (ms)")
    ax.set_ylabel("Runtime (sec)")
    ax.grid("on")
    ax.set_xlim(0, max(time))


def plot_results_latency(
    input_seq: List[float],
    mean_seq: List[float],
    lower_bound_seq: List[float],
    upper_bound_seq: List[float],
    x_axis: str,
    y_axis: str,
    figure_name: str,
    color1: str,
    color2: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    plt.figure()  # row 0, col 0
    plt.plot(input_seq, mean_seq, color=color1)
    plt.fill_between(
        input_seq, lower_bound_seq, upper_bound_seq, color=color2, alpha=0.4
    )
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.grid("on")
    plt.xlim(min(input_seq), max(input_seq))
    plt.subplots_adjust(left=0.15, right=.95, top=.95, bottom=0.15)
    plt.savefig(figure_name)
    plt.close()


def plot_bar_data(
    step_size: float,
    input_seq: List[float],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    mean_rt_seq_su,
    mean_rt_seq_us,
    x_axis: str,
    figure_name: str,
    color1: str,
    color2: str,
    color3: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()
    bar_width = 0.2 * step_size
    max_rt = max(
        [min(mean_rt_seq_su[i], mean_rt_seq_us[i]) for i in range(len(input_seq))]
    )

    for i in range(len(input_seq)):
        sum_prob = low_regime_prob_seq[i] + high_regime_prob_seq[i]
        ax.bar(
            input_seq[i],
            low_regime_prob_seq[i] / sum_prob,
            width=bar_width,
            color=color1,
            label=f"Normalized Prob for low mode" if i == 0 else "",
        )
        ax.bar(
            input_seq[i],
            high_regime_prob_seq[i] / sum_prob,
            width=bar_width,
            bottom=np.array(low_regime_prob_seq[i] / sum_prob)
            + (min(mean_rt_seq_su[i], mean_rt_seq_us[i]) / max_rt),
            color=color3,
            label=f"Normalized Prob for high mode" if i == 0 else "",
        )
        ax.bar(
            input_seq[i],
            min(mean_rt_seq_su[i], mean_rt_seq_us[i]) / max_rt,
            width=bar_width,
            bottom=low_regime_prob_seq[i] / sum_prob,
            color=color2,
            alpha=0.6,
            label=f"Normalized average hitting time" if i == 0 else "",
        )
        ax.yaxis.set_visible(False)

    # Show legend
    ax.legend()
    # Show plot
    plt.tight_layout()
    plt.xlabel(x_axis, fontsize=14)
    plt.grid("on")
    plt.ylim(0, 3)
    fig.subplots_adjust(left=0.15, right=.95, top=.95, bottom=0.15)
    plt.savefig(figure_name)
    plt.close()


def plot_results_reset(
    input_seq: List[float],
    mean_seq: List[float],
    lower_bound_seq: List[float],
    upper_bound_seq: List[float],
    x_axis: str,
    y_axis: str,
    figure_name: str,
    color1: str,
    color2: str,
    nominal_value,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    plt.figure()  # row 0, col 0
    plt.plot(input_seq, mean_seq, color=color1)
    plt.fill_between(
        input_seq, lower_bound_seq, upper_bound_seq, color=color2, alpha=0.4
    )
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.grid("on")
    plt.xlim(min(input_seq), max(input_seq))
    plt.axvline(x=nominal_value, color="k", linestyle="--", linewidth=2)

    plt.savefig(figure_name)
    plt.close()


def trigger_plot_generator(
    lambda_fault,
    fault_start_time,
    fault_duration,
    lambda0,
    time_step,
    sim_time,
    file_name: str,
):
    input_seq = []
    time_seq = []
    for time in np.arange(0, sim_time, time_step):
        time_seq.append(time)
        if time < fault_start_time:
            input_seq.append(lambda0)
        elif fault_start_time <= time < fault_start_time + fault_duration:
            input_seq.append(lambda_fault)
        else:
            input_seq.append(lambda0)
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    plt.figure()  # row 0, col 0
    plt.plot(time_seq, input_seq, color="tab:purple")
    plt.xlabel("Time (ms)", fontsize=14)
    plt.ylabel("Arrival rate", fontsize=14)
    plt.grid("on")
    plt.xlim(min(time_seq), max(time_seq))

    plt.savefig("fault_scenario_" + file_name)
    plt.close()


def plot_heatmap(
    input_seq: List[List[float]],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    T_mixing_seq,
    x_axis: str,
    y_axis: str,
    figure_name: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    lambdaa_seq = np.unique(np.array([input_seq[i][0] for i in range(len(input_seq))]))
    mu_seq = np.unique(np.array([input_seq[i][1] for i in range(len(input_seq))]))
    z = np.array(
        [
            min(low_regime_prob_seq[i], high_regime_prob_seq[i]) * T_mixing_seq[i]
            for i in range(len(input_seq))
        ]
    )
    Z = z.reshape(len(lambdaa_seq), len(mu_seq))
    plt.figure(figsize=(6, 5))
    plt.imshow(
        Z,
        origin="lower",
        cmap="viridis",
        extent=[lambdaa_seq.min(), lambdaa_seq.max(), mu_seq.min(), mu_seq.max()],
    )
    plt.colorbar(label="Metastability metric")

    # fig, ax = plt.subplots()
    # ax.set_xlim(0, max(x) * 1.1)  # Set x-axis limits
    # ax.set_ylim(0, max(y) * 1.1)  # Set y-axis limits
    plt.tight_layout()
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    # fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(figure_name)
    plt.close()


def plot_Nyquist(
    input_seq: List[float],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    T_mixing_seq,
    x_axis: str,
    y_axis: str,
    figure_name: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    x = T_mixing_seq
    y = [
        min(low_regime_prob_seq[i], high_regime_prob_seq[i])
        for i in range(len(input_seq))
    ]
    z = input_seq
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(z), vmax=max(z))

    points = np.array([x, y]).T.reshape(
        -1, 1, 2
    )  # Reshape the x and y points for LineCollection
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.array(z))  # Color based on z
    lc.set_linewidth(2)  # Set the line width

    fig, ax = plt.subplots()
    ax.add_collection(lc)  # Add the line collection to the plot
    ax.autoscale()  # Autoscale the plot limits to fit the data
    ax.set_xlim(0, max(x) * 1.1)  # Set x-axis limits
    ax.set_ylim(0, 0.5)  # Set y-axis limits

    # Add colorbar to show the mapping of z values to colors
    plt.colorbar(lc, label="Arrival rate")

    # Region 1: A square in the lower-left corner
    ax.axvspan(
        2000,
        max(x) * 1.1,
        ymin=0.5,
        ymax=1,
        facecolor="red",
        alpha=0.3,
        label="Metastable region",
    )

    # Region 2: The complement (upper-right corner)
    # ax.axvspan(0, 2000, ymin=0, ymax=.25, facecolor='blue', alpha=0.1, label='')
    plt.legend(loc="upper center")

    plt.tight_layout()
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    # plt.title('Sequence of (x, y) points with color based on z')
    plt.grid("on")
    fig.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(figure_name)
    plt.close()


def plot_quiver(
    input_seq: List[float],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    T_mixing_seq,
    x_axis: str,
    y_axis: str,
    figure_name: str,
):
    x = T_mixing_seq
    y = [
        min(low_regime_prob_seq[i], high_regime_prob_seq[i])
        for i in range(len(input_seq))
    ]
    z = input_seq
    plt.figure(figsize=(6, 6))
    plt.quiver(z, np.zeros_like(z), x, y, angles="xy", scale_units="xy", scale=1)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.savefig(figure_name)
    plt.close()


def plot_scatter(
    input_seq: List[float],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    T_mixing_seq,
    x_axis: str,
    y_axis: str,
    figure_name: str,
):
    x = T_mixing_seq
    y = [
        min(low_regime_prob_seq[i], high_regime_prob_seq[i])
        for i in range(len(input_seq))
    ]
    z = input_seq
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker="o", label="Trajectory")

    for i in range(1, len(z), 1):
        plt.arrow(
            x[i - 1],
            y[i - 1],
            x[i] - x[i - 1],
            y[i] - y[i - 1],
            head_width=0.05,
            head_length=0.1,
            fc="r",
            ec="r",
        )

    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.savefig(figure_name)
    plt.close()


def plot_2D(
    input_seq: List[float],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    T_mixing_seq,
    x_axis: str,
    y_axis: str,
    figure_name: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True
    x = T_mixing_seq
    y = [
        min(low_regime_prob_seq[i], high_regime_prob_seq[i])
        for i in range(len(input_seq))
    ]
    z = input_seq

    plt.scatter(x, y, c=z, cmap="viridis", s=50)
    plt.colorbar(label="z")
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.grid(True)
    plt.savefig(figure_name)
    plt.close()


def plot_mixing_time(
    input_seq: List[float],
    low_regime_prob_seq: List[float],
    high_regime_prob_seq: List[float],
    T_mixing_seq,
    x_axis: str,
    y_axis: str,
    figure_name: str,
    color1: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    plt.figure()  # row 0, col 0
    output_seq = [
        min(low_regime_prob_seq[i], high_regime_prob_seq[i]) * T_mixing_seq[i]
        for i in range(len(input_seq))
    ]
    plt.plot(input_seq, output_seq, color=color1)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.grid("on")
    plt.xlim(min(input_seq), max(input_seq))

    plt.savefig(figure_name)
    plt.close()


def plot_metastability_index(
    input_seq: List[float],
    output_seq: List[float],
    x_axis: str,
    y_axis: str,
    figure_name: str,
    color: str,
):
    plt.rc("font", size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    plt.figure()  # row 0, col 0
    plt.plot(input_seq, output_seq, color=color)
    plt.xlabel(x_axis, fontsize=14)
    plt.ylabel(y_axis, fontsize=14)
    plt.grid("on")
    plt.xlim(min(input_seq), max(input_seq))

    plt.savefig(figure_name)
    plt.close()
