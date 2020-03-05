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
from lqmc import HubbardModel, LatticeQMC, LqmcProcessManager
from lqmc.tools import local_gf, print_filling, save_gf_tau, load_gf_tau
from lqmc.gftools import tau2iw_dft


def measure(model, temp, time_steps, warmup, sweeps, cores=None):
    """ Runs the lqmc warmup and measurement loop for the given model.
    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    temp: float
        The temperature in .math'\beta = 1/T'.
    time_steps: int
        Number of time steps from .math'0' to .math'\beta'
    sweeps: int, optional
        Total number of sweeps (warmup + measurement)
    warmup: int, optional
        The ratio of sweeps used for warmup. The default is '0.2'.
    cores: int, optional
        Number of processes to use. If not specified one process per core is used.
    Returns
    -------
    gf: (2, N, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    if cores is not None and cores == 1:
        solver = LatticeQMC(model, time_steps, warmup, sweeps, det_mode=True)
        solver.set_temperature(temp)
        gf_tau = solver.run()
    else:
        manager = LqmcProcessManager(cores)
        manager.init(model, temp, time_steps, warmup, sweeps)
        manager.start()
        manager.run()
        gf_data = manager.recv_all()
        manager.join()
        manager.terminate()
        gf_tau = np.sum(gf_data, axis=0) / manager.cores
    return gf_tau


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
    save = False
    # Model parameters
    n_sites = 5
    u, t = 0, 1
    temp = 2
    beta = 1 / temp
    # Simulation parameters
    time_steps = 15
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
        g_tau = measure(model, temp, time_steps, warmup, sweeps, cores=cores)
        if save:
            save_gf_tau(model, beta, time_steps, sweeps, g_tau)
            print("Saving")

    print('fillings:')
    print_filling(g_tau[0][0], g_tau[1][0])
    print_filling(g_tau[0][5], g_tau[1][5])
    print_filling(g_tau[0][7], g_tau[1][7])

    gf_up, gf_dn = local_gf(g_tau)

    g_iw = tau2iw_dft(gf_up + gf_dn, beta)
    print('G_iw:')
    print(g_iw)

    plot_gf_tau(beta, gf_up)
    plot_gf_tau(beta, gf_dn)
    plt.show()


if __name__ == "__main__":
    main()
