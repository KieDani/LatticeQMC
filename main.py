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
from lqmc.tools import check_params, save_gf_tau, load_gf_tau, print_filling


def get_local_gf_tau(g_tau):
    """ Returns the local elements (diagonal) of the Green's function matrix
    Parameters
    ----------
    g_tau: (..., M, N, N) array_like
        Green's function matrices for M time steps and N sites

    Returns
    -------
    gf_tau: (..., M, N) np.ndarray
    """
    return np.diagonal(g_tau, axis1=-2, axis2=-1)


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
    ax.set_ylim(-2, 2)
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
    n_sites = 3
    u, t = 4, 1
    temp = 4.
    beta = 1 / temp
    # Simulation parameters
    time_steps = 15
    warmup = 200
    sweeps = 200
    cores = 1  # None to use all cores of the cpu

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites, cycling=False)
    print(model.ham_kinetic())

    try:
        g_tau = load_gf_tau(model, beta, time_steps, sweeps)
        print("GF data loaded")
    except FileNotFoundError:
        print("Found no data...")
        g_tau = measure(model, beta, time_steps, warmup, sweeps, cores=cores, det_mode=True)
        #save_gf_tau(model, beta, time_steps, sweeps, g_tau)
        print("Saving")

    gf_up, gf_dn = get_local_gf_tau(g_tau)
    #gf_omega_up = tau2iw_dft(gf_up, beta)
    #gf_omega_dn = tau2iw_dft(gf_dn, beta)

    plot_gf_tau(beta, gf_up)
    plot_gf_tau(beta, gf_dn)
    plot_gf_tau(beta, gf_up + gf_dn)

    print('fillings:')
    print(g_tau)
    print_filling(g_tau[0][0], g_tau[1][0])
    print_filling(g_tau[0][5], g_tau[1][5])
    print_filling(g_tau[0][time_steps-1], g_tau[1][time_steps-1])
    print('G_iw:')
    g_iw = tau2iw_dft(gf_up + gf_dn, beta)
    print(g_iw)
    print('Im(G_iw)')
    print(g_iw.imag)
    plt.plot(range(0,int(time_steps/2)), g_iw.imag[:,0])

    plt.show()


if __name__ == "__main__":
    main()
