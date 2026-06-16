from vmdpy import VMD
import numpy as np


def compute_vmd(signal, K=5):

    alpha = 2000
    tau = 0
    DC = 0
    init = 1
    tol = 1e-7

    modes, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)

    return modes.T   # shape (time, modes)