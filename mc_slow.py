# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0

To do
-----
- Multiproccesing
- Outer measure function
- Logging
- M-matrix inner func (??)
"""
import time
import itertools
import numpy as np
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration


def updateln(string):
    print("\r" + string, end="", flush=True)


def check_params(u, t, dtau):
    check_val = u * t * dtau**2
    if check_val < 0.1:
        print(f"Check-value {check_val:.2f} is smaller than 0.1!")
    else:
        print(f"Check-value {check_val:.2f} should be smaller than 0.1!")


def compute_m(ham_kin, config, lamb, dtau, sigma):
    n = ham_kin.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham_kin)
    # Create the V_l matrix
    v = np.zeros((n, n), dtype=config.dtype)

    # fill diag(V_l) with values of last time slice and compute B-product
    lmax = config.n_t - 1
    np.fill_diagonal(v, config[:, lmax])
    exp_v = expm(sigma * lamb * v)
    b = np.dot(exp_k, exp_v)

    b_prod = b
    for l in reversed(range(1, lmax)):
        # Fill V_l with new values, compute B(l) and multiply with total product
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


def warmup(model, config, dtau, lamb, sweeps=200):
    ham_kin = model.ham_kinetic()

    # Calculate m-matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Store copy of current configuration
    old_config = config.copy()
    acc = False
    # QMC loop
    updateln("Warmup sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.n_t)):
            updateln(f"Warmup sweep: {sweep+1}/{sweeps}, accepted: {acc}")
            # Update Configuration
            config.update(i, l)

            # Calculate m-matrices and ratio of the configurations
            # Accept move with metropolis acceptance ratio.
            m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
            m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            r = np.random.rand()  # Random number between 0 and 1
            if r < ratio:
                # Move accepted:
                # Continue using the new configuration
                acc = True
                old = new
                old_config = config.copy()
            else:
                # Move not accepted:
                # Revert to the old configuration
                acc = False
                config = old_config
    print()
    return config


def measure_gf(model, config, dtau, lamb, sweeps=800):
    ham_kin = model.ham_kinetic()

    # Calculate m-matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Initialize total and temp greens functions
    gf_up, gf_dn = 0, 0
    g_tmp_up = np.linalg.inv(m_up)
    g_tmp_dn = np.linalg.inv(m_dn)

    # Store copy of current configuration
    old_config = config.copy()

    # QMC loop
    acc = False
    number = 0
    updateln("Measurement sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.n_t)):
            updateln(f"Measurement sweep: {sweep+1}/{sweeps}, accepted: {acc}")
            # Update Configuration
            config.update(i, l)
            # Calculate m-matrices and ratio of the configurations
            # Accept move with metropolis acceptance ratio.
            m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
            m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            r = np.random.rand()  # Random number between 0 and 1
            if r < ratio:
                # Move accepted:
                # Update temp greens function and continue using the new configuration
                g_tmp_up = np.linalg.inv(m_up)
                g_tmp_dn = np.linalg.inv(m_dn)

                acc = True
                old = new
                old_config = config.copy()
            else:
                # Move not accepted:
                # Revert to the old configuration
                acc = False
                config = old_config

            # Add temp greens function to total gf after each step
            gf_up += g_tmp_up
            gf_dn += g_tmp_dn
            number += 1
    print()
    # Return the normalized gfs for each spin
    return np.array([gf_up, gf_dn]) / number


def save(model, beta, n_tau, gf):
    file = f"data\\gf2_t={beta}_nt={n_tau}_{model.param_str()}"
    np.save(file, gf)


def measure(u, t, beta, n_tau, n_sites):
    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites)
    dtau = beta / n_tau
    check_params(u, t, dtau)

    lamb = np.arccosh(np.exp(u * dtau / 2.))  # Paper factor
    # lamb = 0.5 * np.exp(-u * dtau / 4.)

    t0 = time.time()
    config = Configuration(model.n_sites, n_tau)
    config = warmup(model, config, dtau, lamb, sweeps=200)
    gf = measure_gf(model, config, dtau, lamb, sweeps=800)
    t = time.time() - t0

    mins, secs = divmod(t, 60)
    print(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
    print()
    save(model, beta, n_tau, gf)
    return gf


def filling(g_sigma):
    return 1 - np.diagonal(g_sigma)


def main():
    n_sites = 4
    u, t = 2, 1
    temp = 2
    beta = 1 / temp
    n_tau = 10

    gf = measure(u, t, beta, n_tau, n_sites)
    # gf = np.load("data\\gf_t=2_nt=20_u=2_t=1_mu=1.0.npy")

    n_up, n_dn = filling(gf[0]), filling(gf[1])
    print(f"<n↑> = {np.mean(n_up):.3f}  {n_up}")
    print(f"<n↓> = {np.mean(n_dn):.3f}  {n_dn}")
    print(f"<n>  = {np.mean(n_up + n_dn):.3f}")


if __name__ == "__main__":
    main()
