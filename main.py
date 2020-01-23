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
from lqmc import HubbardModel, measure
from lqmc.tools import save_gf_tau, load_gf_tau, print_filling, local_gf


def tau2iw_dft(gf_tau, beta):
    r""" Discrete Fourier transform of the real Green's function `gf_tau`.

    Parameters
    ----------
    gf_tau : (..., N_tau) float np.ndarray
        The Green's function at imaginary times :math:`τ \in [0, β]`.
    beta : float
        The inverse temperature :math:`beta = 1/k_B T`.
    Returns
    -------
    gf_iw : (..., {N_iw - 1}/2) float np.ndarray
        The Fourier transform of `gf_tau` for positive fermionic Matsubara
        frequencies :math:`iω_n`.
    """
    # expand `gf_tau` to [-β, β] to get symmetric function
    gf_tau_full_range = np.concatenate((-gf_tau[:-1, ...], gf_tau), axis=0)
    dft = np.fft.ihfft(gf_tau_full_range[:-1, ...], axis=0)
    # select *fermionic* Matsubara frequencies
    gf_iw = -beta * dft[1::2, ...]
    return gf_iw


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


def main():
    # Model parameters
    n_sites = 10
    u, t = 6, 1
    temp = 200
    beta = 1 / temp
    # Simulation parameters
    time_steps = 50
    warmup = 500
    sweeps = 5000
    cores = 1  # None to use all cores of the cpu

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites)
    try:
        g_tau = load_gf_tau(model, beta, time_steps, sweeps)
        print("GF data loaded")
    except FileNotFoundError:
        print("Found no data...")
        g_tau = measure(model, beta, time_steps, warmup, sweeps, cores=cores)
        save_gf_tau(model, beta, time_steps, sweeps, g_tau)
        print("Saving")

    gf_up, gf_dn = local_gf(g_tau)
    gf_omega_up = tau2iw_dft(gf_up, beta)
    gf_omega_dn = tau2iw_dft(gf_dn, beta)

    plot_gf_tau(beta, gf_up)
    plot_gf_tau(beta, gf_dn)
    plot_gf_tau(beta, gf_up + gf_dn)

    print('fillings:')
    print_filling(g_tau[0][0], g_tau[1][0])
    print_filling(g_tau[0][5], g_tau[1][5])
    print_filling(g_tau[0][7], g_tau[1][7])
    print('G_iw:')
    g_iw = tau2iw_dft(gf_up + gf_dn, beta)
    print(g_iw)
    print('Im(G_iw)')
    print(g_iw.imag)

    plt.show()


if __name__ == "__main__":
    main()
