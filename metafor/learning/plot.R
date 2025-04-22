# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/plot.R

library(ggplot2)
library(tidyverse)

width=6
height=3
dpi=100

# ecdf = empirical cumulative distribution function

plot_results <- function(name) {
	d = read.table(paste0(name, "_results.csv"), header=TRUE, sep=",")
	g = ggplot(d, aes(service_time)) + stat_ecdf(aes(color=name)) +
		ylab("Cumulative Density") +
		xlab("Client-Observed Latency")
	ggsave(paste0(name, "_ecdf.png"), dpi=dpi, width=width, height=height)

	g = ggplot(d, aes(service_time)) + stat_ecdf(aes(color=name)) +
		ylab("Cumulative Density") +
		xlab("Client-Observed Latency") +
		coord_cartesian(ylim = c(0.9, 1.0))
	ggsave(paste0(name, "_ecdf_zoomed.png"), dpi=dpi, width=width, height=height)

	gp = ggplot(d, aes(service_time)) + geom_freqpoly(aes(color=name),binwidth=1) + coord_cartesian(xlim = c(0,30)) + xlim(0,30)
	ggsave(paste0(name, "_pdf.png"), dpi=dpi, width=width, height=height)

	gq = ggplot(d, aes(x=t, y=qlen, color=name)) + geom_line() +
		ylab("Queue Length") +
		xlab("Time")
	ggsave(paste0(name, "_qlen.png"), dpi=dpi, width=width, height=height)

	gd <- ggplot(d, aes(x=t, y=dropped, color=name)) + geom_line()
    ggsave(paste0(name, "_dropped.png"), dpi=dpi, width=width, height=height)

    ggr <- ggplot(d, aes(x=t, y=retries, color=name)) + geom_line()
    ggsave(paste0(name, "_retries.png"), dpi=dpi, width=width, height=height)
}

runs=10

plot_results("mean_exp")
plot_results("variance_exp")
plot_results("std_deviation_exp")

for (i in 1:runs) {
    results_file_name = paste0(i, "_exp")
    if (file.exists(paste0(results_file_name, "_results.csv"))) {
        plot_results(results_file_name)
    }
}