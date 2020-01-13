# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0
"""
import numpy as np
from scipy.linalg import expm
from hubbard import HubbardModel
from configuration import Configuration


def assert_params(u, t, dtau, verbose=True):
    check_val = u * t * dtau**2
    if verbose:
        print(f"Check-value {check_val} should be smaller than 0.1!")
    assert check_val < 0.1
    if verbose:
        print("OK!\n")


def compute_b(model, config, lamb, dtau, l, sigma):
    ham = model.ham_kinetic()
    # check if there is a better way to calculate the matrix exponential
    tmp1 = expm(dtau * ham)
    v_l = model.build_v(l, config=config.config)
    # V_l is diagonal -> more effective way to calculate exp
    tmp2 = expm(sigma * lamb * v_l)
    return np.dot(tmp1, tmp2)


def compute_m(model, config, lamb, dtau, sigma):
    n = model.n_sites
    # Create the matrix m
    m = np.eye(n, dtype=np.float64)
    m[0, 0] = 0

    lmax = config.n_t - 1
    bs = compute_b(model, config, lamb, dtau, lmax, sigma)
    for l in reversed(range(1, lmax)):
        b = compute_b(model, config, lamb, dtau, l, sigma)
        bs = np.dot(bs, b)
    return m + bs


def mc_loop(n_sites, n_t):
    for i in range(n_sites):
        for l in range(n_t):
            yield i, l


def warmup(model, config, dtau, lamb, sweeps=None):
    if sweeps is None:
        sweeps = int(0.5 * model.n_sites * config.n_t)

    # Calculate m-matrices
    m_up = compute_m(model, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(model, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Store copy of current configuration
    old_config = config.copy()

    # QMC loop
    for sweep in range(sweeps):
        print(f"Warmup sweep: {sweep+1}/{sweeps}")
        for i, l in mc_loop(model.n_sites, config.n_t):
            # Update Configuration
            config.update(i, l)

            # Calculate m-matrices and ratio of the configurations
            # Accept move with metropolis acceptance ratio.
            m_up = compute_m(model, config, lamb, dtau, sigma=+1)
            m_dn = compute_m(model, config, lamb, dtau, sigma=-1)
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            r = np.random.rand()  # Random number between 0 and 1
            # print(f"Probability: {ratio:.2f}")
            # print(f"Random number: {r:.2f}")
            if r < ratio:
                # Move accepted:
                # Continue using the new configuration
                old = new
                old_config = config.copy()
                # print("Move accepted!")
            else:
                # Move not accepted:
                # Revert to the old configuration
                config = old_config
                # print("Move not accepted :(")
    return config


def measure_gf(model, config, dtau, lamb, sweeps=3200):
    # Calculate m-matrices
    m_up = compute_m(model, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(model, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Initialize total and temp greens functions
    gf_up, gf_dn = 0, 0
    g_tmp_up = np.linalg.inv(m_up)
    g_tmp_dn = np.linalg.inv(m_dn)

    # Store copy of current configuration
    old_config = config.copy()

    # QMC loop
    number = 0
    for sweep in range(sweeps):
        print(f"Measurement sweep: {sweep + 1}/{sweeps}")
        for i, l in mc_loop(model.n_sites, config.n_t):
            # Update Configuration
            config.update(i, l)
            # Calculate m-matrices and ratio of the configurations
            # Accept move with metropolis acceptance ratio.
            m_up = compute_m(model, config, lamb, dtau, sigma=+1)
            m_dn = compute_m(model, config, lamb, dtau, sigma=-1)
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
    # Return the normalized gfs for each spin
    return np.array([gf_up, gf_dn]) / number


def filling(g_sigma):
    return 1 - np.diagonal(g_sigma)


def main():
    n_sites = 4
    u, t = 2, 1
    tau = 2
    n_tau = 10
    dtau = tau / n_tau

    assert_params(u, t, dtau)

    # lamb = np.arccosh(np.exp(u * dtau / 2.))
    lamb = 0.5 * np.exp(- u * dtau / 4.)  # Paper factor

    model = HubbardModel(u=u, t=t, mu=u/2)
    model.build(n_sites)

    config = Configuration(model.n_sites, n_tau)
    config = warmup(model, config, dtau, lamb, sweeps=100)
    gf = measure_gf(model, config, dtau, lamb, sweeps=200)

    n_up, n_dn = filling(gf[0]), filling(gf[1])
    print()
    print("n↑:  ", n_up)
    print("<n↑>:", np.mean(n_up))
    print()
    print("n↓:  ", n_dn)
    print("<n↓>:", np.mean(n_dn))


if __name__ == "__main__":
    main()
