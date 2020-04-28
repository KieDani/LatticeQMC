# -*- coding: utf-8 -*-
"""
Created on 10 Nov 2018
@author: Dylan Jones

project: LatticeQMC
version: 1.0
"""
from .lattice import Lattice
from .configuration import Configuration
from .hubbard import HubbardModel
from .lqmc import LatticeQMC
from .tools import *
from .multiprocessing import ParallelProcessManager, SerialProcessManager


def measure(model, beta, time_steps, warmup, sweeps, cores=None, det_mode=False):
    r""" Runs the lqmc warmup and measurement loop for the given model.

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance with N sites.
    beta: float
        The inverse temperature .math'\beta = 1/T'.
    time_steps: int
        Number of time steps from .math'0' to .math'\beta'
    warmup: int, optional
        The ratio of sweeps used for warmup. The default is '0.2'.
    sweeps: int, optional
        Total number of sweeps (warmup + measurement)
    cores: int, optional
        Number of processes to use. If not specified one process per core is used.
    det_mode: bool, optional
        Flag for the calculation mode. If 'True' the slow algorithm via
        the determinants is used. The default is 'False' (faster).

    Returns
    -------
    gf_up: (N, N) np.ndarray
        Measured Green's function .math'G_{\uparrow}' of the up-spin.
    gf_dn: (N, N) np.ndarray
        Measured Green's function .math'G_{\downarrow}' of the down-spin.
    """
    if cores is not None and cores == 1:
        solver = LatticeQMC(model, beta, time_steps, warmup, sweeps, det_mode)
        gf_up, gf_dn = solver.run()
    else:
        check_params(model.u, model.t, beta / time_steps)
        manager = ParallelProcessManager(model, beta, time_steps, warmup, procs=cores)
        manager.set_jobs(sweeps)
        manager.run()
        gf_up, gf_dn = manager.get_result()
    return gf_up, gf_dn


def measure_betas(model, betas, time_steps, warmup=500, sweeps=5000,
                  cores=-1, caching=True, det_mode=False):
    r""" Runs the lqmc loop for the given model and beta-array.

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance with N sites.
    betas: (M) array_like of float
        The inverse temperature array .math'\beta = 1/T' for wich the
        Green's functions are computed.
    time_steps: int
        Number of time steps from .math'0' to .math'\beta'
    warmup: int, optional
        The ratio of sweeps used for warmup. The default is '0.2'.
    sweeps: int, optional
        Total number of sweeps (warmup + measurement)
    cores: int, optional
        Number of processes to use. If not specified one process per core is used.
    caching: bool, optional
        Flag if data should be saved in a temporary file while calculating.
        The default is True.
    det_mode: bool, optional
        Flag for the calculation mode. If 'True' the slow algorithm via
        the determinants is used. The default is 'False' (faster).

    Returns
    -------
    gf_up: (M, N, N) np.ndarray
        Measured Green's function .math'G_{\uparrow}' of the up-spin.
    gf_dn: (M, N, N) np.ndarray
        Measured Green's function .math'G_{\downarrow}' of the down-spin.
    """
    manager = SerialProcessManager(model, time_steps, warmup, sweeps, det_mode, cores, caching)
    manager.set_jobs(betas)
    manager.run(sleep=1.0)
    gf_data = manager.get_result()
    gf_up, gf_dn = np.swapaxes(gf_data, 0, 1)
    manager.terminate()
    return gf_up, gf_dn
