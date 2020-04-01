import time
import numpy as np
import matplotlib.pyplot as plt
from lqmc.lqmc import LatticeQMC
from lqmc import HubbardModel
from lqmc.multiprocessing import ParallelProcessManager, SerialProcessManager
from lqmc.tools import check_params, save_gf_tau, load_gf_tau


def print_result(gf_up, gf_dn):
    print("-"*50)
    print("filling:")
    filling_up = 1 - np.diagonal(gf_up)
    filling_dn = 1 - np.diagonal(gf_dn)
    print(f"<n↑> =", filling_up)
    print(f"<n↓> =", filling_dn)
    print("double occupancy rate:")
    do_rate = filling_up * filling_dn
    print(do_rate)
    print("local moment:")
    loc_moment = filling_up + filling_dn - 2 * do_rate
    print(loc_moment)


def pfrmt_series(temps, data):
    lines = list()
    for i, temp in enumerate(temps):
        gf_up, gf_dn = data[i]
        filling_up = 1 - np.diagonal(gf_up)
        filling_dn = 1 - np.diagonal(gf_dn)
        lines.append(f"t = {temp}")
        lines.append(f"  <n↑> = {filling_up}")
        lines.append(f"  <n↓> = {filling_dn}")
    return "\n".join(lines)


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
        the determinants is used. The default is 'False' (faster).
    Returns
    -------
    gf: (2, N, N) np.ndarray
        Measured Green's function .math'G' of the up- and down-spin channel.
    """
    if cores is not None and cores == 1:
        solver = LatticeQMC(model, beta, time_steps, warmup, sweeps, det_mode)
        gf = solver.run()
    else:
        check_params(model.u, model.t, beta / time_steps)
        manager = ParallelProcessManager(model, beta, time_steps, warmup, procs=cores)
        manager.set_jobs(sweeps)
        manager.run()
        gf = manager.get_result()
    return gf


def measure_temps(model, temps, time_steps, warmup=500, sweeps=5000, cores=-1, det_mode=False):
    manager = SerialProcessManager(model, time_steps, warmup, sweeps, det_mode, procs=cores)
    manager.set_jobs(1 / temps)
    manager.run()
    gf_data = manager.get_result()
    manager.terminate()
    return gf_data


def main():
    # Model parameters
    n_sites = 5
    u, t = 4, 1
    temp = 4
    beta = 1 / temp
    # Simulation parameters
    time_steps = 15
    warmup = 500
    sweeps = 10000
    cores = -1  # None to use all cores of the cpu

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites)

    temps = np.arange(0.5, 10, step=0.5)
    gf_data = measure_temps(model, temps, time_steps, warmup, sweeps, cores)
    string = pfrmt_series(temps, gf_data)
    print('-'*50)
    print(string)

    # gf_up, gf_dn = measure(model, beta, time_steps, warmup, sweeps, cores=cores)
    # print_result(gf_up, gf_dn)

    # try:
    #     g_tau = load_gf_tau(model, beta, time_steps, sweeps)
    #     print("GF data loaded")
    # except FileNotFoundError:
    #     print("Found no data...")
    #     g_tau = measure(model, beta, time_steps, warmup, sweeps, cores=cores)
    #     save_gf_tau(model, beta, time_steps, sweeps, g_tau)
    #     print("Saving")


if __name__ == "__main__":
    main()
