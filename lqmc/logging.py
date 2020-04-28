# coding: utf-8
"""
Created on 03 Mar 2020
author: Dylan Jones
"""
import logging
from logging import getLogger, INFO, DEBUG, WARNING, ERROR

FILE = "lqmc.log"

BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)


class ConsoleFormatter(logging.Formatter):

    _COLOR = "\033[1;%dm"
    _RESET = "\033[0m"
    _BOLD = "\033[1m"

    COLORS = {'DEBUG': WHITE,
              'INFO': WHITE,
              'WARNING': YELLOW,
              'ERROR': RED,
              'CRITICAL': RED}

    def __init__(self, fmt, color=True, datefmt=None):
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record):
        string = super().format(record)
        if self.color:
            lvl = record.levelname
            if lvl == "CRITICAL":
                string = (self._COLOR + self._BOLD) % (30 + self.COLORS[lvl]) + string + self._RESET
            else:
                string = self._COLOR % (30 + self.COLORS[lvl]) + string + self._RESET
        return string


class ConsoleHandler(logging.StreamHandler):

    def __init__(self, level=logging.INFO, lvlname=True):
        super().__init__()
        fmt = '[%(levelname)-5s] %(message)s' if lvlname else '%(message)s'
        formatter = ConsoleFormatter(fmt)
        self.setFormatter(formatter)
        self.setLevel(level)


class FileHandler(logging.FileHandler):

    def __init__(self, filename, level=logging.DEBUG, mode="w", datefmt='%H:%M:%S'):
        super().__init__(filename, mode=mode)
        formatter = logging.Formatter('[%(levelname)-5s] %(asctime)s - %(message)s', datefmt)
        self.setFormatter(formatter)
        self.setLevel(level)


def get_logger(name='lqmc', file=FILE, console_lvl=INFO, file_lvl=DEBUG):
    logger = getLogger(name)
    ch = ConsoleHandler(console_lvl)
    fh = FileHandler(file, file_lvl)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def read_log_file(file=FILE):
    with open(file, "r") as f:
        lines = f.readlines()
    return [line[:-1] for line in lines]
