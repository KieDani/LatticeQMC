# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0

To do
-----
- Multiproccesing
- M-matrix inner func (??)
- Improve saving and loading
"""
import time
import logging
import numpy as np
from lqmc import HubbardModel, check_params, print_filling
from lqmc.lqmc import LatticeQMC
from lqmc.muliprocessing import LqmcProcessManager

# Configure basic logging for lqmc-loop
# logging.basicConfig(filename="lqmc.log", filemode="w", format='%(message)s', level=logging.DEBUG)


def save_gf_tau(model, beta, time_steps, gf):
    """ Save time depended Green's function measurement data

    To Do
    -----
    Improve saving and loading
    """
    shape = model.lattice.shape
    model_str = f"u={model.u}_t={model.t}_mu={model.mu}_w={shape[0]}_h={shape[1]}"
    file = f"data\\gf_t={beta}_nt={time_steps}_{model_str}.npy"
    np.save(file, gf)


def load_gf_tau(model, beta, time_steps):
    """ Load saved time depended Green's function measurement data

    To Do
    -----
    Improve saving and loading
    """
    file = f"data\\gf_t={beta}_nt={time_steps}_{model.param_str()}.npy"
    return np.load(file)


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


def measure_single_process(model, beta, time_steps, sweeps, warmup_ratio=0.2):
    t0 = time.time()

    solver = LatticeQMC(model, beta, time_steps, sweeps, warmup_ratio)
    print("Warmup:     ", solver.warm_sweeps)
    print("Measurement:", solver.meas_sweeps)
    check_params(model.u, model.t, solver.dtau)
    solver.warmup_loop()
    gf_tau = solver.measure_loop()

    t = time.time() - t0
    mins, secs = divmod(t, 60)
    print(f"\nTotal time: {int(mins):0>2}:{int(secs):0>2} min")
    print()
    return gf_tau


def measure_multi_process(model, beta, time_steps, sweeps, warmup_ratio=0.2):
    manager = LqmcProcessManager()
    manager.init(model, beta, time_steps, sweeps, warmup_ratio)

    manager.start()
    manager.run()
    manager.join()
    gf_data = manager.recv_all()
    manager.terminate()

    gf_tau = np.sum(gf_data, axis=0) / manager.cores
    return gf_tau


def measure(model, beta, time_steps, sweeps, warmup_ratio=0.2, mp=True):
    """ Runs the lqmc warmup and measurement loop for the given model.

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    beta: float
        The inverse temperature .math'\beta = 1/T'.
    time_steps: int
        Number of time steps from .math'0' to .math'\beta'
    sweeps: int, optional
        Total number of sweeps (warmup + measurement)
    warmup_ratio: float, optional
        The ratio of sweeps used for warmup. The default is '0.2'.
    mp: bool, optional
        Flag if multiprocessing should be used. The default is True.

    Returns
    -------
    gf: (2, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    if mp:
        gf_tau = measure_multi_process(model, beta, time_steps, sweeps, warmup_ratio)
    else:
        gf_tau = measure_single_process(model, beta, time_steps, sweeps, warmup_ratio)

    # Revert the time axis to be '[0, \beta]'
    return gf_tau[::-1, ...]






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


def get_local_gf_tau(g_tau):
    """ Returns the local elements (diagonal) of the Green's function matrix

    Parameters
    ----------
    g_tau: (..., N, N) array_like
        Green's function matrices

    Returns
    -------
    gf_tau: (..., N) np.ndarray
    """
    return np.diagonal(g_tau, axis1=-2, axis2=-1)


def plot_gf_tau(beta, gf):
    tau = np.linspace(0, beta, gf.shape[0])
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_xlim(0, beta)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$G(\tau)$")
    for i, y in enumerate(gf.T):
        ax.plot(tau, y, label="$G_{" + f"{i}, {i}" + "}$")
    ax.legend()









def main():
    # Model parameters
    n_sites = 5
    u, t = 2, 1
    temp = 2
    beta = 1 / temp
    # Simulation parameters
    sweeps = 1000
    time_steps = 25

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites)

    gf_tau = measure(model, beta, time_steps, sweeps, mp=True)
    save_gf_tau(model, beta, time_steps, gf_tau)
    # gf_tau = load_gf_tau(model, beta, time_steps)

    gf_tau_up, gf_tau_dn = gf_tau
    print_filling(gf_tau_up[5], gf_tau_dn[5])
    print()


if __name__ == "__main__":
    main()
