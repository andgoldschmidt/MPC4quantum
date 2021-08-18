import mpc4quantum as m4q
from tests import *

import numpy as np
import qutip as qt
from unittest import TestCase

# Diagnostic imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = plt.get_cmap('tab10')

# Default args for plot_operator
imshow_args = {'norm': mpl.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3), 'cmap': plt.get_cmap('RdBu_r')}


def plot_operator(A, dim_x, args=imshow_args):
    dim_l = A.shape[1]//A.shape[0]
    fig, axes = plt.subplots(2, dim_l)
    fig.subplots_adjust(hspace=0)
    for i in range(dim_l):
        Ai = A.reshape(dim_x, -1, dim_x)[:, i, :]
        ax = axes[0, i]
        im = ax.imshow(Ai.real, **imshow_args)
        ax = axes[1, i]
        im = ax.imshow(Ai.imag, **imshow_args)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.8, hspace=0)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig, axes


class TestAll(TestCase):
    def test_3level_drag(self):
        for order in range(3, 4):
            np.random.seed(1)
            # Parameters
            # ==========
            # order = 2
            sat = 1
            # Clock
            # -----
            n_steps = 8
            dt = 0.5
            horizon = 8
            clock = m4q.StepClock(dt, horizon, n_steps)

            # Experiment
            # ==========
            freqs = {'delta': -0.06, 'A0': 1}
            qubit = RWA_Transmon(**freqs)

            # Initial model
            # -------------
            measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s)
                            for j in range(qubit.dim_s)]
            A_cts_list = [m4q.vectorize_me(op, measure_list) for op in qubit.H_list]
            A_init = m4q.discretize_homogeneous(A_cts_list, dt, order)

            # Cost
            # ====
            # Manually form cost matrices_new
            Q = np.zeros((qubit.dim_x, qubit.dim_x))
            Q[0, 0] = 1
            Q[4, 4] = 1
            Q[8, 8] = 0
            Qf = Q * 1
            # Break symmetry
            R = np.array([[1e-3, 0], [0, 1e-3]])  # 1e-3 * np.identity(qubit.dim_u)

            # Avoid local min.--perturb. initial state slightly
            Rx = qt.qip.operations.rx(1e-4)
            rho0 = qt.basis(qubit.dim_s, 0).proj()
            rho0.data[:2, :2] = Rx.dag() * rho0[:2, :2] * Rx
            rho1 = qt.basis(qubit.dim_s, 1).proj()
            initial_state = rho0.data.toarray().flatten()
            target_state = rho1.data.toarray().flatten()

            # Benchmarks
            # ----------
            X_bm = np.hstack([target_state.reshape(-1, 1)] * (clock.n_steps + 1))
            t1 = np.linspace(0, n_steps * dt, n_steps, endpoint=False)
            # u1 = qubit.u1(t1, {'A': 1, 't0': t1[0], 'tf': t1[-1], 'dt': dt})
            U_bm = np.vstack([np.zeros_like(t1), np.zeros_like(t1)])

            # Model (state-independent)
            # =====
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            model1 = DiscrepDMDc.from_bootstrap(qubit.dim_x, qubit.dim_x, qubit.dim_x * dim_lift, A_init)

            # Model predictive control
            # ========================
            data, model2, exit_code = m4q.mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE, model1,
                                              Q, R, Qf, sat=sat)

            # Save diagnostics
            # ================
            # if diagnostic_plots:
            path = './../playground/2021_08_16_DRAG_minus/'
            if not os.path.exists(path):
                os.makedirs(path)

            transparent = False
            if not os.path.exists(path):
                os.makedirs(path)

            fig, axes = plot_operator(model2.A, qubit.dim_x)
            fig.savefig(path + 'ops_order_{}.png'.format(order), transparent=transparent)
            xs, us = data
            fig, axes = plt.subplots(2, 1)
            ax = axes[0]
            ilabels = [0, 4, 8]
            for irow, row in enumerate(xs[:, :-1]):
                ilabel = 'r{}'.format(np.base_repr(irow, base=3).zfill(2))
                ax_args = {'marker': '.', 'markerfacecolor': 'None', 'label': ilabel, 'alpha': 1}
                if irow in ilabels:
                    ax.plot(clock.ts_sim, row.real, **ax_args)
            # lines = ax.get_lines()
            # ax.legend([lines[l] for l in ilabels], [lines[l].get_label() for l in ilabels], ncol=len(ilabels))
            ax.legend()
            ax.set_ylim([-1.1, 1.1])
            ax = axes[1]
            infidelity = [1 - qt.fidelity(qt.Qobj(x.reshape(qubit.dim_s, qubit.dim_s)), rho1) for x in xs.T]
            ax.plot(np.hstack([clock.ts_sim, clock.ts_sim[-1] + clock.dt]), infidelity)
            ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(path + 'traj_order_{}.png'.format(order), transparent=transparent)

            fig, axes = plt.subplots(len(us), figsize=(3 * len(us), 3))
            for irow, row in enumerate(us):
                ax = axes[irow]
                ax.step(np.hstack([clock.ts_sim, clock.ts_sim[-1] + clock.dt]), np.hstack([row, row[-1]]), where='post',
                        color=cmap(irow))
                max_u = np.max(np.abs(us))
                ax.set_ylim([-max_u * 1.1, max_u * 1.1])
            fig.tight_layout()
            fig.savefig(path + 'control_order_{}.png'.format(order), transparent=transparent)

    def test_2level_vectorization(self):
        """
        This test provides a minimal example for generating a trajectory using the model constructed from the
        vectorization utilities in mpc4quantum.vectorization.

        You can adjust the order to see whether the model has been discretized to sufficient order in dt.
        """
        # Parameters
        # ==========
        sat = 1
        order = 2

        # Clock
        # -----
        n_train = 25
        dt = 0.5

        # Experiment
        # ==========
        freqs = {'w0': np.pi,
                 'w1': np.pi,
                 'wR': np.pi}
        qubit = RWA_Qubit(**freqs, A0=sat)

        # Exact solution
        # ==============
        measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s) for j in
                        range(qubit.dim_s)]
        A_cts_list = [m4q.vectorize_me(op, measure_list) for op in qubit.H_list]
        A_init = m4q.discretize_homogeneous(A_cts_list, dt, order)

        # Predict dynamics
        # ================
        rho0 = qt.basis(qubit.dim_s, 0).proj()
        x0 = rho0.data.toarray().flatten()[:, None]

        # Control
        # -------
        pulse_width = n_train * dt
        ts = np.arange(0, pulse_width * 1 / dt, dt)
        args = {'t0': 0, 'tf': pulse_width, 'dt': dt, 'A': 1}
        us = qubit.u1(ts, args)[None, :]

        # Library functions for actuation-extension
        lib_fns = create_library(order, qubit.dim_u)[1:]

        # Prediction
        # ----------
        n_steps = len(ts)
        xs = [None] * (n_steps + 1)
        xs[0] = x0
        lift_us = np.vstack([f(us) for f in lib_fns])
        for i in range(n_steps):
            uxs = krtimes(lift_us[:, i].reshape(-1, 1), xs[i])
            xs[i + 1] = A_init @ np.vstack([xs[i], uxs])
        xs_lin = np.hstack(xs)

        # Comparison
        # ==========
        xs_me = qubit.QE.simulate(x0, ts, us)

        # Not sure where mesolve reports the value--end, start, or middle of time step?
        num_close = np.sum(np.abs(xs_me - (xs_lin[:, 1:] + xs_lin[:, :-1]) / 2) < .1)
        num_total = np.sum(np.ones_like(xs_me, dtype=int))
        # Say 90% of the data points are close, as a rough check
        assert num_close / num_total > .9

    def test_2level_control(self):
        """
        This code demonstrates some MPC controls and corresponding trajectories for a state transfer.\
        """
        for order in range(2, 5):
            np.random.seed(1)
            # Parameters
            # ==========
            sat = 1

            # Clock
            # -----
            n_steps = 15
            dt = 0.5
            horizon = 15
            clock = m4q.StepClock(dt, horizon, n_steps)

            # Experiment
            # ==========
            freqs = {'w0': np.pi, 'w1': np.pi, 'wR': np.pi}
            qubit = RWA_Qubit(**freqs, A0=sat)

            # Exact model
            # -------------
            # Construct |i><j| basis.
            measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s)
                            for j in range(qubit.dim_s)]
            A_cts_list = [m4q.vectorize_me(op, measure_list) for op in qubit.H_list]
            A_init = m4q.discretize_homogeneous(A_cts_list, dt, order)

            # Cost
            # ====
            # Manually form cost matrices
            Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
            Qf = Q
            r_val = 1e-2
            R = r_val * np.identity(qubit.dim_u)

            # Set control problem
            Rx = qt.qip.operations.rx(1e-4)
            rho0 = Rx.dag() * qt.basis(qubit.dim_s, 0).proj() * Rx
            rho1 = qt.basis(qubit.dim_s, 1).proj()
            initial_state = rho0.data.toarray().flatten()
            target_state = rho1.data.toarray().flatten()

            # Benchmarks
            # ----------
            X_bm = np.hstack([target_state.reshape(-1, 1)] * (clock.n_steps + 1))
            U_bm = np.hstack([np.zeros([qubit.dim_u, 1])] * clock.n_steps)

            # Model (n.b. we're not using any data-driven machinery here.)
            # =====
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            model1 = m4q.DMDc(qubit.dim_x, qubit.dim_x, qubit.dim_x * dim_lift, A_init)

            # Model predictive control
            # ========================
            try:
                data, model2, exit_code = m4q.mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE,
                                                  model1, Q, R, Qf, sat=sat)
            except Exception as e:
                print(e)
                # continue

            # DIAGNOSTIC
            # **********
            path = './../playground/2021_08_14_TwoLvl/' \
                   'steps{}_horiz{}_sat{}_dt{}{}_R{}/'.format(n_steps, horizon, sat, *str(dt).split('.'),
                                                              *str(int(abs(np.log10(r_val)))))
            transparent = False
            if not os.path.exists(path):
                os.makedirs(path)

            fig, axes = plot_operator(model2.A, qubit.dim_x)
            fig.savefig(path + 'ops_order_{}.png'.format(order), transparent=transparent)

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

            fig, ax = plt.subplots(1, figsize=(4, 3))
            for row in us:
                ax.step(np.hstack([clock.ts_sim, clock.ts_sim[-1]+ clock.dt]), np.hstack([row, row[-1]]), where='post')
            max_u = np.max(np.abs(us))
            ax.set_ylim([-max_u * 1.1, max_u * 1.1])
            fig.tight_layout()
            fig.savefig(path + 'control_order_{}.png'.format(order), transparent=transparent)
