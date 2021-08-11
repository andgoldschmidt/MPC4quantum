from mpc4quantum import *
from tests import *

import numpy as np
import qutip as qt
from unittest import TestCase

# Diagnostic imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = plt.get_cmap('tab10')


class TestAll(TestCase):
    # https://stackoverflow.com/questions/65945243/
    # https://github.com/cvxpy/cvxpy/issues/45
    # http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms (for bad reg. sigma)

    def test_2(self):
        # Parameters
        # ==========
        sat = 1
        order = 4

        # Clock
        # -----
        n_train = 25
        dt = 0.5

        # Experiment
        # ==========
        freqs = {'w0': np.pi,  # * (1 + np.random.randn() * .001),
                 'w1': np.pi,
                 'wR': np.pi}
        qubit = RWA_Qubit(**freqs, A0=sat)

        # Exact solution
        # ==============
        measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s) for j in
                        range(qubit.dim_s)]
        A_cts_list = [vectorize_me(op, measure_list) for op in qubit.H_list]
        A_init = discretize_homogeneous(A_cts_list, dt, order)

        # Predict dynamics
        # ================
        rho0 = qt.basis(qubit.dim_s, 0).proj()
        initial_state = rho0.data.toarray().flatten()

        # Control
        # -------
        pulse_width = n_train * dt
        ts_train = np.arange(0, pulse_width * 1 / dt, dt)
        args1 = {'t0': 0, 'tf': pulse_width, 'dt': dt, 'A': 1}
        us = qubit.u1(ts_train, args1)[None, :]
        lib_fns = create_library(order, qubit.dim_u)[1:]

        # Model
        # -----
        model1 = DiscrepDMDc.from_bootstrap(qubit.dim_x, qubit.dim_x, qubit.dim_x * len(lib_fns), A_init)

        # Prediction
        # ----------
        n_steps = len(ts_train)
        xs = [None] * (n_steps + 1)
        xs[0] = initial_state[:, None]
        for i in range(n_steps):
            lift_us = np.vstack([f(us[:, i].reshape(-1, 1)) for f in lib_fns])
            uxs = krtimes(lift_us, xs[i])
            xs[i + 1] = model1.predict(xs[i], uxs)
            # xs[i + 1] = A_init @ np.vstack([xs[i], uxs])
        xs = np.hstack(xs)

    def test_1(self):
        for order in range(1, 5):
            np.random.seed(1)
            # Parameters
            # ==========
            sat = 1

            # Clock
            # -----
            n_steps = 15
            n_train = 30
            dt = 0.5
            horizon = 15
            clock = StepClock(dt, horizon, n_steps)

            # Experiment
            # ==========
            freqs = {'w0': np.pi, # * (1 + np.random.randn() * .001),
                     'w1': np.pi,
                     'wR': np.pi}
            qubit = RWA_Qubit(**freqs, A0=sat)

            # Exact model
            # -------------
            training_model, best_rcond = train_model(n_train * dt, clock, qubit, order)
            A_init = training_model.A
            # measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag()
            #                 for i in range(qubit.dim_s) for j in range(qubit.dim_s)]
            # A_cts_list = [vectorize_me(op, measure_list) for op in qubit.H_list]
            # A_init = discretize_homogeneous(A_cts_list, dt, order)

            # Cost
            # ====
            # Manually form cost matrices
            Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
            Qf = Q * 0
            r_val = 1e-9  # 0.1
            R = r_val * np.identity(qubit.dim_u)

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
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            model1 = DiscrepDMDc.from_bootstrap(qubit.dim_x, qubit.dim_x, qubit.dim_x * dim_lift, A_init)

            # Model predictive control
            # ========================
            try:
                data, model2, exit_code = mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE, model1,
                                              Q, R, Qf, sat=sat, streaming=False)
            except Exception as e:
                print(e)
                # continue

            # DIAGNOSTIC
            # **********
            path = '../playground/2021_08_11_badTraining/' \
                   'steps{}_horiz{}_sat{}_dt{}{}_R{}/'.format(n_steps, horizon, sat, *str(dt).split('.'),
                                                              *str(int(abs(np.log10(r_val)))))
            transparent = False
            show_plots = False
            if not os.path.exists(path):
                os.makedirs(path)
            fig, axes = plt.subplots(2, 1)
            imshow_args = {'norm': mpl.colors.CenteredNorm(), 'cmap': plt.get_cmap('RdBu_r')}
            ax = axes[0]
            ax.imshow(model2.A.real, **imshow_args)
            ax.set_title('Real')
            ax = axes[1]
            im = ax.imshow(model2.A.imag, **imshow_args)
            ax.set_title('Imag')
            fig.subplots_adjust(right=0.8, hspace=0)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            fig.savefig(path + 'ops_order_{}.png'.format(order), transparent=transparent)
            # fig.show()

            xs, us = data
            fig, axes = plt.subplots(2, 1)
            ax = axes[0]
            for row in xs[:, :-1]:
                ax.plot(clock.ts_sim, row.real, marker='o', markerfacecolor='None')
            ax.set_ylim([-1.1, 1.1])
            ax = axes[1]
            infidelity = [1 - qt.fidelity(qt.Qobj(x.reshape(2, 2)), rho1) for x in xs.T]
            ax.plot(np.hstack([clock.ts_sim, clock.ts_sim[-1] + clock.dt]), infidelity)
            ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(path + 'traj_order_{}.png'.format(order), transparent=transparent)
            # fig.show()

            fig, ax = plt.subplots(1, figsize=(4, 3))
            for row in us:
                ax.step(np.hstack([clock.ts_sim, clock.ts_sim[-1]+ clock.dt]), np.hstack([row, row[-1]]), where='post')
            max_u = np.max(np.abs(us))
            ax.set_ylim([-max_u * 1.1, max_u * 1.1])
            fig.tight_layout()
            fig.savefig(path + 'control_order_{}.png'.format(order), transparent=transparent)
            # fig.show()
