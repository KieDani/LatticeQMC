# coding: utf-8
"""
Created on 04 Mar 2020
author: Dylan Jones
"""
import numpy as np
import matplotlib.pyplot as plt
from lqmc.logging import read_log_file


class LqmcLog:

    def __init__(self):
        self.lines = read_log_file()
        self.n = len(self.lines)

    def __getitem__(self, item):
        return self.lines[0]

    def iter_section(self, name):
        start = name.upper()
        end = "END " + name.upper()
        idx = 0
        while idx < self.n and start not in self.lines[idx]:
            idx += 1
        idx += 1
        line = self.lines[idx]
        while idx < self.n and end not in line:
            yield line
            idx += 1
            line = self.lines[idx]

    def get_warmup_config_stats(self):
        logs = list(self.iter_section("warmup"))
        n = len(logs)
        stats = np.zeros((n, 2), dtype="float")
        for i in range(n):
            head, msg = logs[i].split(" - ", 1)
            string = msg.strip().split(" -- ")[-1]
            mean, var = string.split(" ")
            stats[i] = float(mean), float(var)
        return stats.T

    def plot_warmup_config_stats(self):
        stats = self.get_warmup_config_stats()
        fig, ax = plt.subplots()
        ax.set_xlabel("Step")
        ax.plot(stats[0], label="MC mean")
        ax.plot(stats[1], label="MC var")
        ax.legend()
        plt.show()


def main():
    log = LqmcLog()
    log.plot_warmup_config_stats()


if __name__ == "__main__":
    main()
