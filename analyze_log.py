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

    @staticmethod
    def split_head(line):
        head, msg = line.split(" - ", 1)
        return head.strip(), msg.strip()

    def iter_section(self, name):
        start = name.upper()
        end = "END " + name.upper()
        idx = 0
        while idx < self.n and start not in self.lines[idx]:
            idx += 1
        idx += 1
        line = self.lines[idx]
        while idx < self.n-1 and end not in line:
            yield line
            idx += 1
            line = self.lines[idx]

    def get_param(self, name):
        for line in self.iter_section("init"):
            _, msg = self.split_head(line)
            if name in msg:
                _, string = msg.split("=")
                return float(string)
        return None

    def print_init(self):
        for line in self.iter_section("init"):
            print(line)

    def get_config_states(self, status="warmup"):
        logs = list(self.iter_section(status))
        n = len(logs)
        stats = np.zeros((n, 2), dtype="float")
        for i in range(n):
            head, msg = self.split_head(logs[i])
            string = msg.strip().split(" -- ")[-1]
            mean, var = string.split(" ")
            stats[i] = float(mean), float(var)
        sweep = np.arange(n) / int(self.get_param("time_steps") * self.get_param("sites"))
        return sweep, stats.T

    def plot_config_stats(self, status="warmup"):
        sweep, stats = self.get_config_states(status)
        fig, ax = plt.subplots()
        ax.set_xlabel("Sweep")
        ax.plot(sweep, stats[0], label="MC mean")
        ax.plot(sweep, stats[1], label="MC var")
        ax.legend()
        plt.show()


def main():
    log = LqmcLog()
    log.print_init()
    log.plot_config_stats("warmup")


if __name__ == "__main__":
    main()
