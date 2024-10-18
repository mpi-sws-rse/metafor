from math import exp, factorial


def tail_prob_computer(qsize: float, service_rate: float, timeout: float):
    """This function computes the timeout probabilities for the case that service time is distributed exponentially."""
    mu = service_rate  # to remain close to the math symbol

    tail_seq = [0]  # The timeout prob is zero when there is no job in the queue!
    mu_x_timeout = mu * timeout
    exp_mu_timeout = exp(- mu_x_timeout)
    current_sum = 0
    last = 1
    for job_num in range(1, qsize):  # compute the timeout prob for all different queue sizes.
        last = last * mu_x_timeout / job_num
        current_sum = current_sum + last
        tail_seq.append(current_sum * exp_mu_timeout)
    return tail_seq


#def tail_prob_computer_basic(main_queue_size, mu0_p, timeout):
#    """BUGGY"""
#    """This function computes the timeout probabilities for the case that service time is distributed exponentially."""
#    tail_seq = [0]  # The timeout prob is zero when there is no job in the queue!
#    for job_num in range(1, main_queue_size):  # compute the timeout prob for all different queue sizes.
#        val = 0  # compute "val" which is the timeout probability when queue size is equal to "job_num".
#        for k in range(0, job_num):
#            val += exp(-mu0_p * timeout) * ((mu0_p * timeout) ** k) / factorial(k)
#        tail_seq.append(val)
#    return tail_seq

#def test_tail_prob(self):
#    q1 = tail_prob_computer_basic(5, 2, .5)
#    q2 = tail_prob_computer(5, 2, .5)
#    print("[basic]  q1 = ", q1)
#    print("[clever] q2 = ", q2)
#    q = zip(q1, q2)
#    assert (all(map(lambda a: abs(a[0] - a[1]) < 0.001, q)))