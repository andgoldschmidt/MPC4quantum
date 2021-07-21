from mpc4quantum import *

import numpy as np
import qutip as qt
from unittest import TestCase


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
        self._w0 = w0
        self._w1 = w1
        self._wR = wR
        self._A0 = A0

        H0 = 1 / 2 * (self._w0 - self._wR) * qt.sigmaz()
        H1 = 1 / 2 * self._A0 * qt.sigmax()
        H2 = 1 / 2 * self._A0 * qt.sigmay()

        self.QE = QExperiment(H0, [H2, H1])

    def u1(self, ts, args):
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt']) * np.cos((self._w1 - self._wR) * ts)


class TestAll(TestCase):
    def test_1(self):
        # Parameters
        # ==========
        u_dim = 2
        order = 2
        control_sat = 25

        # Clock
        # =====
        n_steps = 25
        dt = 0.01
        horizon = 15
        clock = KeepTime(dt, horizon, n_steps)

        # Experiment
        # ==========
        freqs = {'w0': np.pi * (1 + np.random.randn() * .1),
                 'w1': np.pi,
                 'wR': np.pi}
        qubit = RWA_Qubit(**freqs, A0=1)
        e1 = qubit.QE
        x0_train = qt.basis(2, 0).proj().data.toarray().flatten()

        # Training data
        # -------------
        args = {'t0': 0, 'tf': 25, 'dt': clock.dt, 'A': 1}
        ts_train = np.arange(args['t0'], args['tf'], args['dt'])
        u = qubit.u1(ts_train, args)
        lib_fns = lifting.create_library(order, u_dim)[1:]
        u1 = np.vstack([u, np.zeros_like(u)])
        u2 = np.vstack([np.zeros_like(u), u])
        X2 = []
        X1 = []
        UX1 = []
        for us_train in [u1, u2]:
            xs_train = e1.simulate(x0_train, ts_train, us_train)
            #     xs_train = xs_train + 5e-1 * np.random.randn(*xs_train.shape)
            X2.append(xs_train[:, 1:])
            X1.append(xs_train[:, :-1])
            lift_us = np.vstack([f(us_train) for f in lib_fns])
            UX1.append(lifting.krtimes(lift_us[:, :-1], xs_train[:, :-1]))
        X2 = np.hstack(X2)
        X1 = np.hstack(X1)
        UX1 = np.hstack(UX1)
        a_model = DiscrepDMDc.from_data(X2, X1, UX1, rcond=1e-6)

        # Wrap model
        # ==========
        w_model = lifting.WrapModel(*a_model.get_discrete(), u_dim, order)
        A, B = w_model.get_model_along_traj(xs_train, us_train, ts_train)
