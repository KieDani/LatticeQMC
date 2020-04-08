# coding: utf-8
"""
Created on 08 Apr 2020
author: Dylan Jones
"""
import os
import numpy as np
from lqmc import HubbardModel, measure_betas


def get_datadir(shape, time_steps, warmup, sweeps):
    w, h = shape
    root = os.path.join('data', f'{w}x{h}_nt={time_steps}_warm={warmup}_meas={sweeps}')
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    return root


def main():
    # Simulation parameters
    time_steps = 50
    warmup = 500
    sweeps = 5000
    cores = -1  # None to use all cores of the cpu

    temps = np.linspace(2, 20, 20)
    betas = 1 / temps

    model = HubbardModel()
    model.build_square(4)

    folder = get_datadir(model.lattice.shape, time_steps, warmup, sweeps)

    u_vals = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
    for u in u_vals:
        model.set_params(u=u)
        file = os.path.join(folder, f'u={model.u}.npz')
        try:
            np.load(file, allow_pickle=True)
        except FileNotFoundError:
            line = '-' * 80
            print(line)
            print(f'Computing u={model.u}')
            print(line)
            gf_up, gf_dn = measure_betas(model, betas, time_steps, warmup, sweeps, cores)
            np.savez(file, beta=betas, gf_up=gf_up, gf_dn=gf_dn)
            print()


if __name__ == "__main__":
    main()
