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
from .muliprocessing import LqmcProcess, LqmcProcessManager
from .tools import *


def measure(model, beta, time_steps, warmup, sweeps, cores=None, det_mode=False):
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
    warmup: int, optional
        The ratio of sweeps used for warmup. The default is '0.2'.
    cores: int, optional
        Number of processes to use. If not specified one process per core is used.
    det_mode: bool, optional
        Flag for the calculation mode. If 'True' the slow algorithm via
        the determinants is used. The default i9s 'False' (faster).

    Returns
    -------
    gf: (2, N, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    check_params(model.u, model.t, beta / time_steps)
    if cores is not None and cores == 1:
        solver = LatticeQMC(model, beta, time_steps, warmup, sweeps, det_mode=det_mode)
        gf_tau = solver.run()
    else:
        manager = LqmcProcessManager(cores)
        manager.init(model, beta, time_steps, warmup, sweeps, det_mode=det_mode)
        manager.start()
        manager.run()
        gf_data = manager.recv_all()
        manager.join()
        manager.terminate()
        gf_tau = np.sum(gf_data, axis=0) / manager.cores
    return gf_tau
