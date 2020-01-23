# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0

To do
-----
- Improve saving and loading
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration, local_gf


def plot_gf_tau(beta, gf):
    tau = np.linspace(0, beta, gf.shape[0])
    fig, ax = plt.subplots()
    ax.grid()
    # ax.set_xlim(0, beta)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    for i, y in enumerate(gf.T):
        ax.plot(tau, y, label="$G_{" + f"{i}, {i}" + "}$")
    ax.legend()
    return fig, ax


def compute_b(config, exp_k, dtau, lamb, l, sigma):
    exp_v = np.zeros((config.n_sites, config.n_sites), dtype=np.float64)
    np.fill_diagonal(exp_v, np.exp(+1 * sigma * lamb * dtau * config[:, l]))
    return np.dot(exp_k, exp_v)


def compute_m(config, exp_k, dtau, lamb, sigma):
    b_prod = compute_b(config, exp_k, dtau, lamb, 0, sigma)
    for l in range(1, config.time_steps):
        b_l = compute_b(config, exp_k, dtau, lamb, l, sigma)
        b_prod = np.dot(b_l, b_prod)
    # compute M matrix
    return np.eye(config.n_sites) + b_prod


def compute_gf_slices(config, g_beta, exp_k, dtau, lamb, sigma):
    # First index of g[0, :, :] represents the time slices
    g = np.zeros((config.time_steps, config.n_sites, config.n_sites), dtype=np.float64)
    g[-1, :, :] = g_beta
    for l in reversed(range(1, config.time_steps)):
        b = compute_b(config, exp_k, dtau, lamb, l, sigma)
        b_inv = np.linalg.inv(b)
        g[l-1, ...] = np.dot(np.dot(b_inv, g[l, ...]), b)
    return g


def iter_indices(sweeps, config):
    for sweep in range(sweeps):
        for i in range(config.n_sites):
            for l in range(config.time_steps):
                yield sweep, i, l


def iteration_step():
    pass


def warmup_loop(config, exp_k, dtau, lamb, sweeps=100):
    m_up = compute_m(config, exp_k, dtau, lamb, sigma=+1)
    m_dn = compute_m(config, exp_k, dtau, lamb, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)
    print("", end="", flush=True)
    for sweep, i, l in iter_indices(sweeps, config):
        # Update Configuration
        config.update(i, l)
        m_up = compute_m(config, exp_k, dtau, lamb, sigma=+1)
        m_dn = compute_m(config, exp_k, dtau, lamb, sigma=-1)
        new = np.linalg.det(m_up) * np.linalg.det(m_dn)
        ratio = new / old
        acc = np.random.rand() < ratio
        print(f"\rWarmup  {(sweep, i, l)} {ratio:.3f} -> {acc}", end="", flush=True)
        if acc:
            # Move accepted: Continue using the new configuration
            old = new + 0.0
        else:
            # Move not accepted: Revert to the old configuration
            config.update(i, l)
    print()


def measure_loop(config, exp_k, dtau, lamb, sweeps=1):
    m_up = compute_m(config, exp_k, dtau, lamb, sigma=+1)
    m_dn = compute_m(config, exp_k, dtau, lamb, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)

    # Initialize total and temp greens functions
    g_tau_up, g_tau_dn = 0, 0
    g_beta_up = np.linalg.inv(m_up)
    g_beta_dn = np.linalg.inv(m_dn)
    g_tau_tmp_up = compute_gf_slices(config, g_beta_up, exp_k, dtau, lamb, sigma=+1)
    g_tau_tmp_dn = compute_gf_slices(config, g_beta_dn, exp_k, dtau, lamb, sigma=-1)
    # QMC loop
    number = 0
    print("", end="", flush=True)
    for sweep, i, l in iter_indices(sweeps, config):
        # Update Configuration
        config.update(i, l)

        m_up = compute_m(config, exp_k, dtau, lamb, sigma=+1)
        m_dn = compute_m(config, exp_k, dtau, lamb, sigma=-1)
        new = np.linalg.det(m_up) * np.linalg.det(m_dn)
        ratio = new / old

        acc = np.random.rand() < ratio
        print(f"\rMeasure {(sweep, i, l)} {ratio:.3f} -> {acc}", end="", flush=True)
        if acc:
            # Move accepted: Continue using the new configuration
            g_beta_up = np.linalg.inv(m_up)
            g_beta_dn = np.linalg.inv(m_dn)
            g_tau_tmp_up = compute_gf_slices(config, g_beta_up, exp_k, dtau, lamb, sigma=+1)
            g_tau_tmp_dn = compute_gf_slices(config, g_beta_dn, exp_k, dtau, lamb, sigma=-1)

            old = new + 0.0
        else:
            # Move not accepted: Revert to the old configuration
            config.update(i, l)

        # Add temp greens function to total gf after each step
        g_tau_up += g_tau_tmp_up
        g_tau_dn += g_tau_tmp_dn
        number += 1
    print()
    # Return the normalized gfs for each spin
    return np.array([g_tau_up, g_tau_dn]) / number


def main():
    # Model parameters
    n_sites = 3
    u, t = 0.001, 1
    temp = 1 / 4
    beta = 1 / temp
    # Simulation parameters
    time_steps = 20
    warmup = 1000
    sweeps = 1
    dtau = beta / time_steps

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites, cycling=False)
    ham_kin = model.ham_kinetic()

    lamb = np.arccosh(np.exp(model.u * dtau / 2.))
    exp_k = expm(-1 * dtau * ham_kin)
    print(exp_k)
    config = Configuration(n_sites, time_steps)
    config.initialize()

    print(config)
    warmup_loop(config, exp_k, dtau, lamb, warmup)
    print(config)
    g_tau = measure_loop(config, exp_k, dtau, lamb, sweeps)
    print(config)
    gf_tau_up, gf_tau_dn = local_gf(g_tau)
    plot_gf_tau(beta, gf_tau_up)
    plt.show()


if __name__ == "__main__":
    main()
