# coding: utf-8
"""
Created on 15 Jan 2020
author: Dylan Jones

project: LatticeQMC
version: 1.0
"""
import time
import numpy as np
from lqmc import HubbardModel
from lqmc.lqmc import LatticeQMC, print_filling, check_params
from lqmc.muliprocessing import LqmcProcessManager

# logging.basicConfig(filename="lqmc_mp.log", filemode="w", level=logging.INFO)


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
    shape = model.lattice.shape
    model_str = f"u={model.u}_t={model.t}_mu={model.mu}_w={shape[0]}_h={shape[1]}"
    file = f"data\\gf_t={beta}_nt={time_steps}_{model_str}.npy"
    return np.load(file)


def compute_single_process(model, beta, time_steps, sweeps, warmup_ratio=0.2):
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


def compute_multi_process(model, beta, time_steps, sweeps, warmup_ratio=0.2):
    manager = LqmcProcessManager()
    manager.init(model, beta, time_steps, sweeps, warmup_ratio)

    manager.start()
    manager.run()
    manager.join()
    gf_data = manager.recv_all()
    manager.terminate()

    gf_tau = np.sum(gf_data, axis=0) / manager.cores
    return gf_tau


def compute(model, beta, time_steps, sweeps, warmup_ratio=0.2, mp=True):
    if mp:
        gf_tau = compute_multi_process(model, beta, time_steps, sweeps, warmup_ratio)
    else:
        gf_tau = compute_single_process(model, beta, time_steps, sweeps, warmup_ratio)
    # save_gf_tau(model, beta, time_steps, gf_tau)
    return gf_tau


def main():
    n_sites = 5
    u, t = 2, 1
    temp = 2
    beta = 1 / temp
    time_steps = 10
    sweeps = 1000

    model = HubbardModel(u, t, mu=u / 2)
    model.build(n_sites)

    gf_up, gf_dn = compute(model, beta, time_steps, sweeps, mp=True)
    # gf_up, gf_dn = load_gf_tau(model, beta, time_steps)
    print_filling(gf_up[0], gf_dn[0])


if __name__ == "__main__":
    main()
