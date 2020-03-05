# coding: utf-8
"""
Created on 22 Jan 2020
author: Dylan Jones

project: LatticeQMC
version: 1.0
"""
import numpy as np
from lqmc import HubbardModel
from lqmc.gftools import pole_gf_tau
import matplotlib.pyplot as plt


def decompose(a):
    xi, rv = np.linalg.eigh(a)
    return rv, xi, np.linalg.inv(rv)


def reconstruct(rv, xi, rv_inv, diag=False):
    if diag:
        return ((np.transpose(rv_inv) * rv) @ xi[..., np.newaxis])[..., 0]
    else:
        return (rv * xi[..., np.newaxis, :]) @ rv_inv


def noninter_gf(ham, beta):
    rv, xi, rv_inv = decompose(ham)
    tau = np.linspace(0, beta, num=2049)
    # append axis, as we don't want the sum here
    diag_gf_tau = pole_gf_tau(tau, xi[..., np.newaxis], weights=1, beta=beta)
    gf_tau = reconstruct(rv, diag_gf_tau, rv_inv)
    gf_tau = np.moveaxis(gf_tau, 0, -1)  # Convert to shape (site, site, tau)
    return tau, gf_tau


def main():
    # Model parameters
    n_sites = 10
    u, t = 0, 1
    temp = 1 / 4
    beta = 1 / temp

    model = HubbardModel(u=u, t=t, mu=u / 2)
    model.build(n_sites, cycling=False)
    ham = model.ham_kinetic()
    tau, gf_tau = noninter_gf(ham, beta)

    fig, ax = plt.subplots()
    for site in range(model.n_sites):
        ax.plot(tau, gf_tau[site, site], label=f"site {site}")
    ax.set_xlabel(r"$\tau$")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
