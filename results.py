# coding: utf-8
"""
Created on 07 Apr 2020
author: Dylan Jones
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from lqmc import filling, local_moment


def get_datadir(shape, time_steps, warmup, sweeps):
    w, h = shape
    root = os.path.join('data', f'{w}x{h}_nt={time_steps}_warm={warmup}_meas={sweeps}')
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    return root


def pfrmt_series(temps, data):
    lines = list()
    for i, temp in enumerate(temps):
        gf_up, gf_dn = data[i]
        filling_up = 1 - np.diagonal(gf_up)
        filling_dn = 1 - np.diagonal(gf_dn)
        lines.append(f"T={temp} beta = {1/temp:.2f}")
        lines.append(f"   <n↑> = {filling_up}")
        lines.append(f"   <n↓> = {filling_dn}")
    return "\n".join(lines)


def cleanup(data, lim=(0, 1)):
    data[(data < lim[0]) | (data > lim[1])] = np.nan
    return data


def main():
    folder = get_datadir((6, 1), 50, 500, 5000)
    files = list(os.listdir(folder))

    fig, ax = plt.subplots()

    for file in files:
        u = os.path.splitext(file)[0].split('=')[1]
        u = float(u)
        path = os.path.join(folder, file)
        data = np.load(path, allow_pickle=True)
        beta = data['beta']
        gf_up = data['gf_up']
        gf_dn = data['gf_dn']

        temps = 1 / beta
        n_up, n_dn = filling(gf_up), filling(gf_dn)
        n_up = np.mean(n_up, axis=1)
        n_dn = np.mean(n_dn, axis=1)

        loc_filling = (n_up + n_dn) / 2
        loc_filling = cleanup(loc_filling)
        loc_moment = np.mean(local_moment(gf_up, gf_dn), axis=1)
        loc_moment = cleanup(loc_moment)

        line = ax.plot(temps, loc_filling, label=f'U={u}')[0]
        col = line.get_color()
        ax.plot(temps, loc_moment, color=col, ls='--')

    ax.set_xlabel('T')
    ax.set_ylabel(r'$\langle m_z^2 \rangle$')
    ax.legend()
    ax.set_ylim(0.4, 1)
    ax.grid()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
