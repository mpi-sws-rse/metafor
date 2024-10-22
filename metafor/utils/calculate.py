from math import exp


def tail_prob_computer(qsize: float, service_rate: float, timeout: float):
    """This function computes the timeout probabilities for the case that service time is distributed exponentially."""
    mu = service_rate  # to remain close to the math symbol
    mu_x_timeout = mu * timeout
    exp_mu_timeout = exp(-mu_x_timeout)
    if exp_mu_timeout == 0:
        return [0] * qsize

    tail_seq = [0]  # The timeout prob is zero when there is no job in the queue!
    current_sum = 0
    last = 1
    for job_num in range(
        1, qsize
    ):  # compute the timeout prob for all different queue sizes.
        last = last * mu_x_timeout / job_num
        current_sum = current_sum + last
        tail_seq.append(current_sum * exp_mu_timeout)
    return tail_seq
