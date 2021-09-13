from mpc4quantum import QExperiment

import numpy as np
import qutip as qt

# The qubits are supposed to represent the true physical system, unlike the model which  is our idealization.


def blackman(ts, t0, tf, dt):
    """
    Evaluate Blackman pulse to resolution dt using 1D linear interpolation.
    """
    M = int((tf - t0) / dt)
    t_interp = np.linspace(t0, tf, M)
    f_interp = np.blackman(M)
    return np.interp(ts, t_interp, f_interp, left=0, right=0)


class RWA_Coupled:
    def __init__(self):
        self.dim_u = 3
        self.dim_s = 4
        self.dim_x = self.dim_s ** 2

        I_op = qt.identity(2)
        H0_z12 = qt.Qobj(qt.tensor(qt.sigmaz(), qt.sigmaz()).full())
        H_y1 = qt.Qobj(qt.tensor(qt.sigmay(), I_op).full())
        H_y2 = qt.Qobj(qt.tensor(I_op, qt.sigmay()).full())
        # H_x1 = qt.Qobj(qt.tensor(qt.sigmax(), I_op).full())
        # H_x2 = qt.Qobj(qt.tensor(I_op, qt.sigmax()).full())
        H_z1 = qt.Qobj(qt.tensor(qt.sigmaz(), I_op).full())
        # H_z2 = qt.Qobj(qt.tensor(I_op, qt.sigmaz()).full())
        self.H_list = [H0_z12, H_y1, H_y2, H_z1]

        self.QE = QExperiment(self.H_list[0], [self.H_list[i] for i in range(1, len(self.H_list))])


class RWA_Qubit:
    def __init__(self, wQ, wD, wR):
        """
        Wrapper for qubit simulations in a rotating frame after the rotating wave approximation.

        :param wQ: Qubit frequency
        :param wD: Drive frequency
        :param wR: Rotating wave frequency
        """
        self.dim_s = 2
        self.dim_x = self.dim_s ** 2
        self.dim_u = 1

        self._w0 = wQ
        self._w1 = wD
        self._wR = wR

        H0 = 1 / 2 * (self._w0 - self._wR) * qt.sigmaz()
        H1 = 1 / 2 * qt.sigmax()
        self.H_list = [H0, H1]
        self.QE = QExperiment(H0, [H1])

        # Two pulses?
        # H2 = 1 / 2 * self._A0 * qt.sigmay()
        # self.H_list = [H0, H1, H2]
        # self.QE = QExperiment(H0, [H1, H2])

    def u1(self, ts, args):
        # Blackman pulse (truncated Gaussian) with modifications for the rotating frame
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt']) * np.cos((self._wD - self._wR) * ts)


class RWA_Transmon:
    """
    Driven on resonance and in the frame of the drive.
    """
    def __init__(self, alpha):
        """
        Transmon qubit in a frame rotating exactly at the qubit frequency after the rotating wave approximation.

        :param alpha: Anharmonicity alpha
        """
        self.dim_s = 3
        self.dim_x = self.dim_s ** 2
        self.dim_u = 2
        self._delta = alpha

        H0 = alpha * qt.basis(3, 2).proj()
        HX = 1 / 2 * (qt.create(3) + qt.destroy(3))
        HY = 1j / 2 * (qt.create(3) - qt.destroy(3))
        self.H_list = [H0, HX, HY]
        self.QE = QExperiment(H0, [HX, HY])

    @staticmethod
    def u1(ts, args):
        # Blackman pulse (truncated gaussian)
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt'])
