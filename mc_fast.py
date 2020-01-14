# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0

To do
-----
Multiproccesing
outer measure function


"""
import sys
import time
import numpy as np
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration
# from lqmc.hubbard import HubbardModel
# from lqmc.configuration import Configuration


def stdout(string):
    sys.__stdout__.write(string)
    sys.__stdout__.flush()


def clear_line(width=100):
    stdout("\r" + " " * width + "\r")


def assert_params(u, t, dtau, verbose=True):
    check_val = u * t * dtau**2
    if verbose:
        print(f"Check-value {check_val:.2f} should be smaller than 0.1!")
    assert check_val < 0.1
    if verbose:
        print("OK!")


def compute_m(ham, config, lamb, dtau, sigma):
    n = ham.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham)
    # Create the V_l matrix
    v = np.zeros((n, n), dtype=config.dtype)

    # fill diag(V_l) with values of last time slice and compute B-product
    lmax = config.n_t - 1
    np.fill_diagonal(v, config[:, lmax])
    exp_v = expm(sigma * lamb * v)
    b = np.dot(exp_k, exp_v)

    b_prod = b
    for l in reversed(range(1, lmax)):
        # Fill V_l with new values and compute B(l)
        np.fill_diagonal(v, config[:, l])
        exp_v = expm(sigma * lamb * v)
        b = np.dot(exp_k, exp_v)

        b_prod = np.dot(b_prod, b)

    # compute M matrix
    return np.eye(n) + b_prod


def mc_loop(n_sites, n_t):
    for i in range(n_sites):
        for l in range(n_t):
            yield i, l


def warmup(model, config, dtau, lamb, sweeps=None):
    if sweeps is None:
        sweeps = int(0.5 * model.n_sites * config.n_t)

    ham = model.ham_kinetic()
    print(ham)
    # Calculate m-matrices
    m_up = compute_m(ham, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Store copy of current configuration
    old_config = config.copy()
    acc = False
    # QMC loop
    stdout("Warmup sweep")
    for sweep in range(sweeps):
        for i, l in mc_loop(model.n_sites, config.n_t):
            clear_line(100)
            stdout(f"Warmup sweep: {sweep+1}/{sweeps}, accepted: {acc}")
            # Update Configuration
            config.update(i, l)

            # Calculate m-matrices and ratio of the configurations
            # Accept move with metropolis acceptance ratio.
            m_up = compute_m(ham, config, lamb, dtau, sigma=+1)
            m_dn = compute_m(ham, config, lamb, dtau, sigma=-1)
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            r = np.random.rand()  # Random number between 0 and 1
            # print(f"Probability: {ratio:.2f}")
            # print(f"Random number: {r:.2f}")
            if r < ratio:
                acc = True
                # Move accepted:
                # Continue using the new configuration
                old = new
                old_config = config.copy()
                # print("Move accepted!")
            else:
                acc = False
                # Move not accepted:
                # Revert to the old configuration
                config = old_config
                # print("Move not accepted :(")
    print()
    return config


def measure_gf(model, config, dtau, lamb, sweeps=3200):
    ham = model.ham_kinetic()
    # Calculate m-matrices
    m_up = compute_m(ham, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Initialize total and temp greens functions
    gf_up, gf_dn = 0, 0
    g_tmp_up = np.linalg.inv(m_up)
    g_tmp_dn = np.linalg.inv(m_dn)

    # Store copy of current configuration
    old_config = config.copy()

    # QMC loop
    number = 0
    stdout("Measurement sweep")
    for sweep in range(sweeps):
        clear_line(100)
        stdout(f"Measurement sweep: {sweep + 1}/{sweeps}")
        for i, l in mc_loop(model.n_sites, config.n_t):
            # Update Configuration
            config.update(i, l)
            # Calculate m-matrices and ratio of the configurations
            # Accept move with metropolis acceptance ratio.
            m_up = compute_m(ham, config, lamb, dtau, sigma=+1)
            m_dn = compute_m(ham, config, lamb, dtau, sigma=-1)
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            r = np.random.rand()  # Random number between 0 and 1
            # print(f"Probability: {ratio:.2f}")
            # print(f"Random number: {r:.2f}")
            if r < ratio:
                # Move accepted:
                # Continue using the new configuration and update temp greens function
                old = new
                old_config = config.copy()
                g_tmp_up = np.linalg.inv(m_up)
                g_tmp_dn = np.linalg.inv(m_dn)
                # print("Move accepted!")
            else:
                # Move not accepted:
                # Revert to the old configuration
                config = old_config
                # print("Move not accepted :(")

            # Add temp greens function to total gf after each step
            gf_up += g_tmp_up
            gf_dn += g_tmp_dn
            number += 1
    print()
    # Return the normalized gfs for each spin
    return np.array([gf_up, gf_dn]) / number


def save(model, t, n_t, gf):
    file = f"data\\gf2_t={t}_nt={n_t}_{model.param_str()}"
    np.save(file, gf)


def filling(g_sigma):
    return 1 - np.diagonal(g_sigma)


def main(calc=True):
    temp = 2
    beta = 1 / temp

    n_sites = 20
    u, t = 2, 1
    tau = beta
    n_tau = 10

    if calc:
        dtau = tau / n_tau
        assert_params(u, t, dtau)

        lamb = np.arccosh(np.exp(u * dtau / 2.))  # Paper factor
        # lamb = 0.5 * np.exp(-u * dtau / 4.)

        model = HubbardModel(u=u, t=t, mu=u / 2)
        model.build(n_sites)

        t0 = time.time()

        config = Configuration(model.n_sites, n_tau)
        config = warmup(model, config, dtau, lamb, sweeps=100)
        gf = measure_gf(model, config, dtau, lamb, sweeps=100)

        t = time.time() - t0
        mins, secs = divmod(t, 60)
        print(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
        print()
        save(model, tau, n_tau, gf)
    else:
        gf = np.load("data\\gf_t=2_nt=20_u=2_t=1_mu=1.0.npy")

    n_up, n_dn = filling(gf[0]), filling(gf[1])

    print(f"<n↑> = {np.mean(n_up):.3f}  {n_up}")
    print(f"<n↓> = {np.mean(n_dn):.3f}  {n_dn}")
    print(f"<n>  = {np.mean(n_up + n_dn):.3f}")


if __name__ == "__main__":
    main(calc=True)
