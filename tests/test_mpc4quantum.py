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
        x_dim = 4
        order = 1

        # Clock
        # =====
        n_steps = 25
        dt = 0.1
        horizon = 15
        clock = StepClock(dt, horizon, n_steps)

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
        lib_fns = create_library(order, u_dim)[1:]
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
            UX1.append(krtimes(lift_us[:, :-1], xs_train[:, :-1]))
        X2 = np.hstack(X2)
        X1 = np.hstack(X1)
        UX1 = np.hstack(UX1)
        training_model = DiscrepDMDc.from_data(X2, X1, UX1, rcond=1e-6)

        # Cost
        # ====
        # Manually form cost matrices
        Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        Qf = Q * 0
        R = 0.001 * np.identity(u_dim)

        rho0 = qt.basis(2, 0).proj()
        rho1 = ((qt.basis(2, 0) + qt.basis(2, 1)) / np.sqrt(2)).proj()  # qt.basis(2, 1).proj()
        initial_state = rho0.data.toarray().flatten()
        target_state = rho1.data.toarray().flatten()

        # Benchmarks
        # ----------
        X_bm = np.hstack([target_state.reshape(-1, 1)] * (clock.n_steps + 1))
        U_bm = np.hstack([np.zeros([u_dim, 1])] * clock.n_steps)

        # Model (state-independent)
        # =====
        model1 = DiscrepDMDc.from_bootstrap(x_dim, x_dim, u_dim * x_dim, training_model.A)

        # Model predictive control
        # ========================
        data, model2, exit_code = mpc(initial_state, u_dim, order, X_bm, U_bm, clock, e1, model1, Q, R, Qf)
        print(exit_code)
