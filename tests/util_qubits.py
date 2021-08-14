from mpc4quantum import QExperiment

import numpy as np
import qutip as qt


def blackman(ts, t0, tf, dt):
    """
    Evaluate Blackman pulse to resolution dt using 1D linear interpolation.
    """
    M = int((tf - t0) / dt)
    t_interp = np.linspace(t0, tf, M)
    f_interp = np.blackman(M)
    return np.interp(ts, t_interp, f_interp, left=0, right=0)


class RWA_Qubit:
    def __init__(self, w0, w1, wR, A0):
        self.dim_s = 2
        self.dim_x = self.dim_s ** 2
        self.dim_u = 1

        self._w0 = w0
        self._w1 = w1
        self._wR = wR
        self._A0 = A0

        H0 = 1 / 2 * (self._w0 - self._wR) * qt.sigmaz()
        H1 = 1 / 2 * self._A0 * qt.sigmax()
        self.H_list = [H0, H1]
        self.QE = QExperiment(H0, [H1])

        # Two pulses?
        # H2 = 1 / 2 * self._A0 * qt.sigmay()
        # self.H_list = [H0, H1, H2]
        # self.QE = QExperiment(H0, [H1, H2])

    def u1(self, ts, args):
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt']) * np.cos((self._w1 - self._wR) * ts)


class RWA_Transmon:
    """
    Driven on resonance and in the frame of the drive.
    """
    def __init__(self, delta, A0):
        self.dim_s = 3
        self.dim_x = self.dim_s ** 2
        self.dim_u = 2
        self._delta = delta
        self._A0 = A0

        H0 = delta * qt.basis(3, 2).proj()
        HX = 1 / 2 * self._A0 * (qt.create(3) + qt.destroy(3))
        HY = 1j / 2 * self._A0 * (qt.create(3) - qt.destroy(3))
        self.H_list = [H0, HX, HY]
        self.QE = QExperiment(H0, [HX, HY])

    @staticmethod
    def u1(ts, args):
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt'])