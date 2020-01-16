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
from lqmc import HubbardModel, Configuration
from lqmc.lqmc import warmup_loop, measure_loop, check_params, print_filling, LatticeQMC

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


def measure(model, beta, time_steps, sweeps=1000, warmup_ratio=0.2, fast=True):
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
    fast: bool, optional
        Flag if the fast algorithm should be used. The default is True.

    Returns
    -------
    gf: (2, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    dtau = beta / time_steps
    check_params(model.u, model.t, dtau)

    warmup_sweeps = int(sweeps * warmup_ratio)
    measure_sweeps = sweeps - warmup_sweeps

    t0 = time.time()
    config = Configuration(model.n_sites, time_steps)
    config = warmup_loop(model, config, dtau, warmup_sweeps, fast=fast)
    gf = measure_loop(model, config, dtau, measure_sweeps, fast=fast)
    t = time.time() - t0

    mins, secs = divmod(t, 60)
    print(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
    print()
    save_gf_tau(model, beta, time_steps, gf)
    return gf


def main():
    # Model parameters
    n_sites = 5
    u, t = 2, 1
    temp = 2
    beta = 1 / temp

    # Simulation parameters
    sweeps = 100
    time_steps = 10

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites)

    solver = LatticeQMC(model, beta, time_steps, sweeps)
    solver.warmup_loop()
    gf_tau_up, gf_tau_dn = solver.measure_loop()

    # gf_tau_up, gf_tau_dn = measure(model, beta, time_steps, sweeps, fast=True)
    # gf_tau_up, gf_tau_dn = load_gf_tau(model, beta, time_steps)

    print_filling(gf_tau_up[0], gf_tau_dn[0])
    print()


if __name__ == "__main__":
    main()
