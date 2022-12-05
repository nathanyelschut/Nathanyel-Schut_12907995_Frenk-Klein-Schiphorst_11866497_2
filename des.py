"""Module for doing discrete event simulations in the context of
queueing systems."""

# Authors: Nathanyel Schut & Frenk Klein Schiphorst
# Date: 02/12/2022

import random

import numpy as np
import simpy
from scipy.stats import norm


def source(env: simpy.Environment, n_customers: int, mean_arrival_time: float,
           mean_service_time: float, servers: simpy.Resource,
           use_priority: bool, service_time_distribution: str):
    """Function for generating customers at exponentially distributed
    inter arrival times. Based on the Simpy tutorial 'Bank renege'.

    Args:
        env (simpy.Environment): simulation environment
        n_customers (int): total number of customers to be simulated
        mean_arrival_time (float): average time between two arrivals
        mean_service_time (float): average time of service
        servers (simpy.Recource): simpy recource for handling requests
        use_priority (bool): whether to use priority queues
        service_time_distribution (str): distribution of the service
            times. Options are 'D' (deterministic), 'M' (Markov) and
            'H' (hyperexponential). All distributions have the same
            mean service time

    Yields:
        simpy event: event which is triggered after some exponentially
            distributed inter arrival time.
    """
    # Global variable to keep track of the waiting times
    global wait_times
    wait_times = []

    for _ in range(n_customers):
        c = customer(env, servers, mean_service_time, use_priority,
                     service_time_distribution)
        env.process(c)
        t = random.expovariate(1.0 / mean_arrival_time)
        yield env.timeout(t)


def customer(env: simpy.Environment, servers: simpy.Resource,
             mean_service_time: float, use_priority: bool,
             service_time_distribution: str):
    """Simulates a customer arriving, being serviced and leaving.
    Also based on the Simpy tutorial 'Bank renege'.

    Args:
        env (simpy.Environment): simulation environment
        servers (simpy.Recource): simpy recource for handling requests
        mean_service_time (float): average time of service
        use_priority (bool): whether to use priority queues
        service_time_distribution (str): distribution of the service
            times. Options are 'D' (deterministic), 'M' (Markov) and
            'H' (hyperexponential). All distributions have the same
            mean service time

    Yields:
        simpy event: request of being serviced
        simpy event: event which is triggered after service time has
            expired
    """
    # Arrive
    arrive = env.now

    # Service
    if service_time_distribution == 'M':
        service_time = random.expovariate(1.0 / mean_service_time)

    elif service_time_distribution == 'D':
        service_time = mean_service_time

    elif service_time_distribution == 'H':
        X = random.random()

        if X < 0.25:
            service_time = random.expovariate(1 / (2 * mean_service_time))

        else:
            service_time = random.expovariate(3 / (2 * mean_service_time))

    if use_priority:
        precision = 4
        priority = round(service_time * 10**precision)

        req = servers.request(priority=priority)

    else:
        req = servers.request()

    yield req

    wait = env.now - arrive

    wait_times.append(wait)

    yield env.timeout(service_time)

    # Leave
    servers.release(req)


def get_mean_wait_times(n_servers: int, n_customers: int, n_simulations: int,
                        mean_arrival_time: float, mean_service_time: float,
                        use_priority: bool = False,
                        service_time_distribution: str = 'M') -> np.ndarray:
    """Function for performing multiple simulations of a queueing
    system.

    Args:
        n_servers (int): number of servers to handle requests
        n_customers (int): total number of customers to be simulated
        n_simulations (int): total number of simulations to be done
        mean_arrival_time (float): average time between two arrivals
        mean_service_time (float): average time of service
        use_priority (bool): whether to use priority queues. Defaults
            to False.
        service_time_distribution (str): distribution of the service
            times. Options are 'D' (deterministic), 'M' (Markov) and
            'H' (hyperexponential). All distributions have the same
            mean service time

    Returns:
        numpy.ndarray: array of waiting times where indices i, j denote
            the simulation number and customer number respectively.
    """
    wait_times_array = np.zeros((n_simulations, n_customers))

    for i in range(n_simulations):
        # Setup and start the simulation
        env = simpy.Environment()

        # Start processes and run
        if use_priority:
            servers = simpy.PriorityResource(env, capacity=n_servers)
        else:
            servers = simpy.Resource(env, capacity=n_servers)

        env.process(source(env, n_customers, mean_arrival_time,
                           mean_service_time, servers, use_priority,
                           service_time_distribution))

        env.run()

        # Add waiting times to an array before resetting the list.
        wait_times_array[i, :] = wait_times

    return wait_times_array


def get_confidence_interval(p: float, estimates: np.ndarray) -> float:
    """Function for calculating confidence intervals based on a list of
    estimates.

    Args:
        p (float): confidence level. For example, 0.95 indicates 95%
            confidence
        estimates (numpy.ndarray): 2 dimensional numpy array with
            waiting times per simulation per customer

    Returns:
        float: confidence interval
    """
    phi = 0.5*(p+1)
    lamda = norm.ppf(phi)

    estimates = np.mean(estimates, axis=1)

    S_sq = 0
    X_bar = np.mean(estimates)
    j_max = np.size(estimates)

    for j in range(j_max):
        X_j = estimates[j]

        S_sq += ((X_j - X_bar)**2) / (j_max - 1)

    a = lamda * np.sqrt(S_sq) / np.sqrt(j_max)
    return a
