from mpc4quantum import *
from train_model import train_model

import numpy as np
import qutip as qt
from unittest import TestCase

# Diagnostic imports
import os
import matplotlib.pyplot as plt
cmap = plt.get_cmap('tab10')


class TestAll(TestCase):
    # https://stackoverflow.com/questions/65945243/
    # https://github.com/cvxpy/cvxpy/issues/45
    # http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms (for bad reg. sigma)

    def test_1(self):
        for order in [2]:  # range(1, 5):
            np.random.seed(1)
            # Parameters
            # ==========
            # order = 1

            # Clock
            # -----
            n_steps = 25
            dt = 0.5
            horizon = 25
            clock = StepClock(dt, horizon, n_steps)

            # Experiment
            # ==========
            freqs = {'w0': np.pi,  # * (1 + np.random.randn() * .001),
                     'w1': np.pi,
                     'wR': np.pi}
            qubit = RWA_Qubit(**freqs, A0=1)

            # Trained model
            # -------------
            training_model, best_rcond = train_model(clock, qubit, order)

            # Cost
            # ====
            # Manually form cost matrices
            Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
            Qf = Q * 0
            R = .01 * np.identity(qubit.dim_u)

            rho0 = qt.basis(qubit.dim_s, 0).proj()
            rho1 = qt.basis(qubit.dim_s, 1).proj()
            initial_state = rho0.data.toarray().flatten()
            target_state = rho1.data.toarray().flatten()

            # Benchmarks
            # ----------
            X_bm = np.hstack([target_state.reshape(-1, 1)] * (clock.n_steps + 1))
            U_bm = np.hstack([np.zeros([qubit.dim_u, 1])] * clock.n_steps)

            # Model (state-independent)
            # =====
            model1 = DiscrepDMDc.from_bootstrap(qubit.dim_x, qubit.dim_x, training_model.dim_u, training_model.A)

            # Model predictive control
            # ========================
            data, model2, exit_code = mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE, model1,
                                          Q, R, Qf, streaming=False)
            print('done')

            # DIAGNOSTIC
            # path = '../playground/singleU_dt_{}{}/'.format(*str(dt).split('.'))
            # if not os.path.exists(path):
            #     os.makedirs(path)
            #
            # xs, us = data
            # fig, ax = plt.subplots(1)
            # for row in xs[:, :-1]:
            #     ax.plot(clock.ts_sim, row.real, marker='o', markerfacecolor='None')
            # ax.set_ylim([-1.1, 1.1])
            # fig.savefig(path + 'traj_order_{}.png'.format(order), transparent=True)
            # fig.tight_layout()
            # fig.show()
            #
            # fig, ax = plt.subplots(1, figsize=(4, 3))
            # for row in us:
            #     ax.step(np.hstack([clock.ts_sim, clock.ts_sim[-1]]), np.hstack([row, row[-1]]), where='post')
            # max_u = np.max(np.abs(us))
            # ax.set_ylim([-max_u - 1, max_u + 1])
            # fig.savefig(path + 'control_order_{}.png'.format(order), transparent=True)
            # fig.tight_layout()
            # fig.show()
