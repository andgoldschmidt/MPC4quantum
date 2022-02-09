import mpc4quantum as m4q
from tests import *

import numpy as np
import qutip as qt
from scipy.linalg import block_diag
from unittest import TestCase

# Diagnostic imports
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = plt.get_cmap('tab10')


# TODO: Cost values need a to_string or ?
# Set root
rootname = '2021_09_13_OSQP'


# Default args for plot_operator
imshow_args = {'norm': mpl.colors.SymLogNorm(vmin=-1, vmax=1, linthresh=1e-3), 'cmap': plt.get_cmap('RdBu_r')}


def plot_operator(A, dim_x, args=imshow_args):
    dim_l = A.shape[1]//A.shape[0]
    fig, axes = plt.subplots(2, dim_l)
    fig.subplots_adjust(hspace=0)
    for i in range(dim_l):
        Ai = A.reshape(dim_x, -1, dim_x)[:, i, :]
        ax = axes[0, i]
        _ = ax.imshow(Ai.real, **args)
        ax = axes[1, i]
        im = ax.imshow(Ai.imag, **args)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.8, hspace=0)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig, axes


# ***********************
# Gate synthesis examples
# ***********************
class TestGateSynth(TestCase):
    def test_NOT_gate(self):
        for order in range(1, 5):
            # Parameters
            # ==========
            sat = 1
            du = .25
            clock = m4q.StepClock(dt=0.05, horizon=15, n_steps=50)

            # Experiment
            # ==========
            freqs = {'w0': np.pi, 'w1': np.pi, 'wR': np.pi}
            qubit = RWA_Qubit(**freqs)
            gate_synth = m4q.QSynthesis(qubit.H_list[0], [qubit.H_list[1]])

            # Model
            # =====
            As_cts_list = [-1j * np.kron(np.kron(h, np.identity(2)) - np.kron(np.identity(2), h.conj()), np.identity(4))
                           for h in qubit.H_list]
            A_init = m4q.discretize_homogeneous(As_cts_list, clock.dt, order)

            # DMDc object
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            dim_process = qubit.dim_x ** 2
            model1 = m4q.DMDc(dim_process, dim_process, dim_process * dim_lift, A_init)

            # Operations
            Rx = qt.qip.operations.rx(1e-3)
            U0 = Rx * qt.identity(2)
            Uf = qt.sigmax()
            p0 = qt.tensor(U0, U0.conj()).full().flatten()
            pf = qt.tensor(Uf, Uf.conj()).full().flatten()

            # Benchmarks
            # ==========
            P_bm = np.hstack([pf.reshape(-1, 1)] * (clock.horizon + 1))
            U_bm = np.hstack([np.ones([qubit.dim_u, 1]) * .5] * clock.horizon)

            # Cost
            # ====
            Q = np.identity(len(p0))
            Qf = Q * 1e1
            R = np.identity(qubit.dim_u) * 1e-2

            # MPC
            # ===
            def exit_condition(p2, p1, u1):
                return ((p1 - pf).conj().T @ Q @ (p1 - pf)).real < 1e-2

            data, model2, exit_code = m4q.mpc(p0, qubit.dim_u, order, P_bm, U_bm, clock, gate_synth, model1, Q, R, Qf,
                                              sat=sat, du=du, exit_condition=exit_condition)
            ps, us = data

            # Save diagnostics
            # ================
            path = './../playground/{}_NOT_gate/sat_{}_du_{}_{}/'.format(rootname, sat, du, clock.to_string())
            if not os.path.exists(path):
                os.makedirs(path)
            transparent = False
            if np.any(us):
                tsp1 = np.hstack([clock.ts_sim, clock.ts_sim[-1] + clock.dt])

                fig, ax = plt.subplots(1)
                ax.step(tsp1, np.hstack([us[0], us[0, -1]]), where='post')
                ax.set_ylim(-1.1, 1.1)
                fig.savefig(path + 'control_order_{}.png'.format(order), transparent=transparent)

                def plot_state_prediction(psi0, name):
                    psi0 = psi0 / psi0.norm()
                    rho0 = psi0.proj().full().reshape(-1, 1)
                    xs = [None] * ps.shape[1]
                    for i, p in enumerate(ps.T):
                        xs[i] = p.reshape((2 ** 2, 2 ** 2)) @ rho0
                    xs = np.hstack(xs)
                    fig, axes = plt.subplots(1, 4, figsize=[3 * 4 + 3, 3])
                    for i in range(4):
                        ax = axes[i]
                        ax.plot(tsp1, xs[i].real, ls='-', color=cmap(i))
                        ax.plot(tsp1, xs[i].imag, ls='--', color=cmap(i))
                        ax.set_ylim([-1.1, 1.1])
                    fig.tight_layout()
                    fig.savefig(path + '{}_order_{}.png'.format(name, order), transparent=transparent)

                # Example state preparations based on gate
                state = qt.basis(2, 0)
                plot_state_prediction(state, '1psi0')
                state = -3 * qt.basis(2, 1) + qt.basis(2, 0)
                plot_state_prediction(state, 'n3psi1_1psi0')

                fig, ax = plt.subplots(1)
                ax.plot(tsp1, [((p - pf).conj().T @ Q @ (p - pf)).real for p in ps.T])
                ax.set_ylim([1e-3, 1e1])
                ax.set_yscale('log')
                fig.savefig(path + 'cost_order_{}.png'.format(order), transparent=transparent)


# *******************
# Functionality tests
# *******************
class TestFunctionality(TestCase):
    def test_discretization(self):
        """
        Euler discretization to first order should be (1 + A*dt) and N*dt.
        If the timestep is one, then the operators should be approximately equal.

        """
        dt = 1
        qubits = RWA_Crosstalk(0)
        measure_list = [qt.basis(2, i) * qt.basis(2, j).dag() for i in range(2) for j in range(2)]

        # Vectorize
        A_cts_list_1 = [m4q.vectorize_me(op, measure_list) for op in qubits.H_list_1]
        A_cts_list_2 = [m4q.vectorize_me(op, measure_list) for op in qubits.H_list_2]

        # Order
        order = 1

        # Discretize
        # A = [[A1, 0], [0, A2]] acts on [x1, x2]
        # N1 = [[N1, 0], [0, 0]] acts on u1*[x1, x2]
        # N2 = [[0, 0], [0, N2]] acts on u2*[x1, x2]
        # Put all u1 controls first, then all u2 controls
        A_cts_list = [block_diag(A_cts_list_1[0], A_cts_list_2[0])]
        n1, _ = A_cts_list_1[0].shape
        n2, _ = A_cts_list_2[0].shape
        for i in range(1, len(A_cts_list_1)):
            N1 = np.block([[A_cts_list_1[i], np.zeros((n1, n2))],
                           [np.zeros((n2, n1)), np.zeros((n2, n2))]])
            A_cts_list.append(N1)
        for i in range(1, len(A_cts_list_2)):
            N2 = np.block([[np.zeros((n1, n1)), np.zeros((n1, n2))],
                           [np.zeros((n2, n1)), A_cts_list_2[i]]])
            A_cts_list.append(N2)
        A_dst = m4q.discretize_homogeneous(A_cts_list, dt, order)

        # Construct theoretical equivalent for our simple case
        A_cts_compare = A_cts_list
        A_cts_compare[0] = A_cts_compare[0] + np.identity(A_cts_compare[0].shape[0])
        A_cts_compare = np.hstack(A_cts_compare)

        assert np.isclose(A_dst.real, A_cts_compare.real).all()
        assert np.isclose(A_dst.imag, A_cts_compare.imag).all()

    def test_partialTrace(self):
        """
        This test looks at whether our partial trace code works.
        """
        for ndim in [2, 4]:
            # Tensor product space
            a = qt.rand_dm(ndim)
            b = qt.rand_dm(ndim)
            c = qt.tensor(a, b)

            ab_vec = m4q.QCoupledExperiment.lift(c.full().flatten())
            true_ab_vec = np.hstack([a.full().flatten(), b.full().flatten()])
            assert np.isclose(true_ab_vec, ab_vec).all()

            c_vec = m4q.QCoupledExperiment.proj(ab_vec)
            lp_c = qt.Qobj(c_vec.reshape(ndim**2, ndim**2))
            assert np.isclose(c, lp_c).all()

            # Full space
            d = qt.rand_dm(ndim**2)
            d_ab_vec = m4q.QCoupledExperiment.lift(d.full().flatten())
            d_vec = m4q.QCoupledExperiment.proj(d_ab_vec)
            lp_d = qt.Qobj(d_vec.reshape(ndim ** 2, ndim ** 2))
            assert not np.isclose(d, lp_d).all()

    def test_vectorization(self):
        """
        This test provides a minimal example for generating a trajectory using the model constructed from the
        vectorization utilities in mpc4quantum.vectorization.

        You can adjust the order to see whether the model has been discretized to sufficient order in dt.
        """
        # Parameters
        # ==========
        order = 2
        n_train = 25
        dt = 0.5

        # Experiment
        # ==========
        freqs = {'w0': np.pi, 'w1': np.pi, 'wR': np.pi}
        qubit = RWA_Qubit(**freqs)

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


# **************************
# State preparation examples
# **************************
class TestStatePrep(TestCase):
    def test_crosstalk(self):
        """
        This one is a bit hacky because we want to pretend our model only operates on the qubit subspaces,
        even though we have a crosstalk which lifts us to the full space of the coupled qubits.
        """
        # Simulation
        # **********
        qubits = RWA_Crosstalk(0)

        # Basis (single qubit)
        measure_list = [qt.basis(2, i) * qt.basis(2, j).dag() for i in range(2) for j in range(2)]

        # Vectorize
        A_cts_list_1 = [m4q.vectorize_me(op, measure_list) for op in qubits.H_list_1]
        A_cts_list_2 = [m4q.vectorize_me(op, measure_list) for op in qubits.H_list_2]

        # Model
        # *****
        # Clock
        # =====
        clock = m4q.StepClock(dt=0.5, horizon=20, n_steps=50)
        sat = 2 * np.pi * 0.1
        du = 0.25
        clock.measure_freq = 2

        # Order
        # =====
        order = 1

        # Discretize
        # ==========
        # A = [[A1, 0], [0, A2]] acts on [x1, x2]
        # N1 = [[N1, 0], [0, 0]] acts on u1 [x1, x2]
        # N2 = [[0, 0], [0, N2]] acts on u2 [x1, x2]
        # Put all u1 controls first, then all u2 controls
        A_cts_list = [block_diag(A_cts_list_1[0], A_cts_list_2[0])]
        n1, _ = A_cts_list_1[0].shape
        n2, _ = A_cts_list_2[0].shape
        for i in range(1, len(A_cts_list_2)):
            N2 = np.block([[np.zeros((n1, n1)), np.zeros((n1, n2))],
                           [np.zeros((n2, n1)), A_cts_list_2[i]]])
            A_cts_list.append(N2)
        for i in range(1, len(A_cts_list_1)):
            N1 = np.block([[A_cts_list_1[i], np.zeros((n1, n2))],
                           [np.zeros((n2, n1)), np.zeros((n2, n2))]])
            A_cts_list.append(N1)
        A_dst = m4q.discretize_homogeneous(A_cts_list, clock.dt, order)

        # DMD
        # ===
        # The model state is a stacked qubit 1 and qubit 2 instead of 4^2
        model_dim_x = 2 * 4
        dim_lift = len(m4q.create_library(order, qubits.dim_u)[1:])
        model1 = m4q.DMDc(model_dim_x, model_dim_x, dim_lift * model_dim_x, A_dst)

        # Objective
        # *********
        # Stack model states
        # ------------------
        Rx1 = qt.qip.operations.rx(-1e-3)
        Rx2 = qt.qip.operations.rx(1e-3)
        rho1_init = Rx1 * qt.basis(2, 0).proj() * Rx1.dag()
        rho2_init = Rx2 * qt.basis(2, 0).proj() * Rx2.dag()
        rho1_targ = qt.basis(2, 1).proj()
        rho2_targ = qt.basis(2, 0).proj()  # ((qt.basis(2, 0) + qt.basis(2, 1))/ np.sqrt(2)).proj()
        # Initial state (in the experiment)
        initial_state = qt.tensor(rho1_init, rho2_init).full().flatten()
        # Target state (for model benchmarks)
        target_state = np.hstack([rho1_targ.full().flatten(), rho2_targ.full().flatten()])

        # Benchmarks
        # ----------
        X_bm = np.hstack([target_state[:, None]] * (clock.n_steps + clock.horizon + 1))
        U_bm = np.hstack([np.zeros((qubits.dim_u, 1))] * (clock.n_steps + clock.horizon))

        # Cost
        # ----
        Q = block_diag(np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
                       np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]))
        qf_val = 1
        Qf = Q * qf_val
        r_val = 1e-3
        R = np.identity(qubits.dim_u) * r_val

        # MPC
        # ***
        data, model2, exit_code = m4q.mpc(initial_state, qubits.dim_u, order, X_bm, U_bm, clock, qubits.QE, model1,
                                          Q, R, Qf, sat=sat, du=du, warm_start=False)

        xs, us = data
        xs_proj = np.vstack([qubits.QE.lift(xi) for xi in xs.T]).T
        x1 = xs_proj[:4, :]
        x2 = xs_proj[4:, :]
        ts = clock.ts_sim[::clock.measure_freq]
        tsplus1 = np.hstack([clock.ts_sim, clock.ts_sim[-1] + clock.dt])

        for xs, targ in zip([x1, x2], [rho1_targ, rho2_targ]):
            xs = np.atleast_2d(xs[:, :-1][:, ::clock.measure_freq])
            fig, axes = plt.subplots(2, 1, figsize=(6, 4))
            ax = axes[0]
            for row in xs:
                ax.plot(ts, row.real, marker='.', markerfacecolor='None')
            ax.set_ylim([-1.1, 1.1])

            ax = axes[1]
            infidelity = [1 - qt.fidelity(qt.Qobj(x.reshape(2, 2)), targ) for x in xs.T]
            ax.plot(ts, infidelity)
            ax.set_ylim([5e-4, 1.5])
            ax.set_yscale('log')

        for row in us:
            fig, ax = plt.subplots(1, figsize=(6, 2))
            ax.step(tsplus1, np.hstack([row, row[-1]]), where='post')
            max_u = np.max(np.abs(us))
            ax.set_ylim([-max_u * 1.1, max_u * 1.1])

        fig, axes = plot_operator(A_dst, 8)

    def test_CNOT_state(self):
        # Hamiltonian
        qubits = RWA_Coupled()

        # Counting basis
        # ^^^^^^^^^^^^^^
        proj_list = [qt.basis(2, i) * qt.basis(2, j).dag() for i in range(2) for j in range(2)]
        measure_list = [qt.Qobj(qt.tensor(i, j).full()) for i in proj_list for j in proj_list]
        # Spin basis
        # ^^^^^^^^^^
        # proj_list = [qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
        # measure_list = [qt.tensor(i, j) for i in proj_list for j in proj_list]

        # Vectorize
        A_cts_list = [m4q.vectorize_me(H, measure_list) for H in qubits.H_list]

        # Parameters
        # **********
        sat = 2 * np.pi * 0.05
        clock = m4q.StepClock(dt=0.25, horizon=50, n_steps=200)
        du = 1 * sat

        for order in range(1, 2):
            np.random.seed(1)
            # Discretize
            # ----------
            A_dst = m4q.discretize_homogeneous(A_cts_list, clock.dt, order)

            # Model
            # -----
            dim_lift = len(m4q.create_library(order, qubits.dim_u)[1:])
            model1 = m4q.DMDc(qubits.dim_x, qubits.dim_x, qubits.dim_x * dim_lift, A_dst)

            # Objective
            # *********
            # States
            # ------
            Rx1 = qt.qip.operations.rx(-1e-2)
            Rx2 = qt.qip.operations.rx(1e-2)
            rho0 = qt.tensor(Rx1 * qt.basis(2, 0).proj() * Rx1.dag(), Rx2 * qt.basis(2, 0).proj() * Rx2.dag())
            rho1 = qt.tensor(qt.basis(2, 0).proj(), qt.basis(2, 1).proj())
            initial_state = rho0.full().flatten()
            target_state = rho1.full().flatten()

            # Benchmarks
            # ----------
            # Benchmarks are shifted during the run
            incline = [min(1., 2 * n / clock.n_steps) for n in range(clock.n_steps + clock.horizon + 1)]
            # [1] * (clock.n_steps + clock.horizon + 1)
            X_bm = np.hstack([target_state[:, None] * i for i in incline])
            # X_bm = np.hstack([target_state[:, None]] * (clock.n_steps + clock.horizon + 1))
            U_bm = np.hstack([np.zeros((qubits.dim_u, 1))] * (clock.n_steps + clock.horizon))

            # Cost
            # ----
            Q = np.zeros((qubits.dim_x, qubits.dim_x))
            nonzero_cost = [0, 5, 10, 15]
            for i in nonzero_cost:
                Q[i, i] = 1
            qf_val = 1
            Qf = Q * qf_val
            r_val = 1e-3  # / sat ** 2
            R = np.identity(qubits.dim_u) * r_val

            # MPC
            # ***
            data, model2, exit_code = m4q.mpc(initial_state, qubits.dim_u, order, X_bm, U_bm, clock, qubits.QE, model1,
                                              Q, R, Qf, sat=sat, du=du)

            xs, us = data
            tsplus1 = np.hstack([clock.ts_sim, clock.ts_sim[-1] + clock.dt])

            # Save diagnostics
            # ================
            path = './../playground/{}_CNOT_v3/'.format(rootname) + \
                   '{}_sat_{}_du_{}_Qf_{}_R_{}/'.format(clock.to_string(), m4q.val_to_str(sat), m4q.val_to_str(du),
                                                        m4q.val_to_str(qf_val), m4q.val_to_str(r_val))
            if not os.path.exists(path):
                os.makedirs(path)
            transparent = False

            fig, axes = plt.subplots(1, qubits.dim_u, figsize=[4 * 3 + 2, 3])
            for i in range(qubits.dim_u):
                ax = axes[i]
                ax.step(tsplus1, np.hstack([us[i], us[i][-1]]), where='post')
            fig.savefig(path + 'control_order_{}.png'.format(order), transparent=transparent)

            fig, ax = plt.subplots(1, figsize=[4,3])
            full_rho1 = qt.Qobj(rho1.full())
            infidelity = [1 - qt.fidelity(qt.Qobj(x.reshape(qubits.dim_s, qubits.dim_s)), full_rho1) for x in xs.T]
            ax.plot(tsplus1, infidelity)
            ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(path + 'cost_order_{}.png'.format(order), transparent=transparent)

            fig, axes = plt.subplots(1, 4, figsize=[4 * 4 + 3, 3])
            j = 0
            for i in range(qubits.dim_x):
                if i in nonzero_cost:
                    ax = axes[j]
                    ax.plot(tsplus1, xs[i].real, c=cmap(j))
                    j = j + 1
                    ax.set_ylim([-.1, 1.1])
            fig.savefig(path + 'traj_order_{}.png'.format(order), transparent=transparent)

    def test_DRAG_state(self):
        """
        Produce a control that looks like the familiar DRAG scheme.
        """
        # Parameters
        # ==========
        clock = m4q.StepClock(dt=0.25, horizon=16, n_steps=20)
        # Note: Often sat is defined relative to anharmonicity (~1/10)
        sat = 2 * np.pi * 0.25
        du = 0.5 * sat

        # Experiment
        # ==========
        anharm = -2 * np.pi * 0.1 * (1 / clock.dt)
        qubit = RWA_Transmon(alpha=anharm)

        # Initial model
        # -------------
        measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s)
                        for j in range(qubit.dim_s)]
        A_cts_list = [m4q.vectorize_me(op, measure_list) for op in qubit.H_list]

        for order in range(1, 3):
            np.random.seed(1)
            A_init = m4q.discretize_homogeneous(A_cts_list, clock.dt, order)

            # Cost
            # ====
            Q = np.zeros((qubit.dim_x, qubit.dim_x))
            Q[0, 0] = 1
            Q[4, 4] = 1
            Q[8, 8] = 0
            qf_val = 1
            Qf = Q * qf_val
            # Break symmetry?
            r_val = 1e-3 / sat ** 2
            R = np.array([[r_val, 0], [0, r_val]])

            # Avoid local min.--perturb. initial state slightly
            Rx = qt.qip.operations.rx(1e-4)
            rho0 = qt.basis(qubit.dim_s, 0).proj()
            rho0.data[:2, :2] = Rx.dag() * rho0[:2, :2] * Rx
            rho1 = qt.basis(qubit.dim_s, 1).proj()
            initial_state = rho0.full().flatten()
            target_state = rho1.full().flatten()

            # Benchmarks
            # ----------
            # Benchmarks are shifted along horizons during the run
            X_bm = np.hstack([target_state[:, None]] * (clock.n_steps + clock.horizon + 1))
            U_bm = np.hstack([np.zeros((qubit.dim_u, 1))] * (clock.n_steps + clock.horizon))

            # Model (n.b. we're not using any data-driven machinery here.)
            # =====
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            model1 = m4q.DMDc(qubit.dim_x, qubit.dim_x, qubit.dim_x * dim_lift, A_init)

            # Model predictive control
            # ========================
            data, model2, exit_code = m4q.mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE, model1, Q,
                                              R, Qf, sat=sat, du=du)

            # Save diagnostics
            # ================
            path = './../playground/{}_DRAG/alpha_{}/'.format(rootname, m4q.val_to_str(anharm)) + \
                   '{}_sat_{}_du_{}_Qf_{}_R_{}/'.format(clock.to_string(), m4q.val_to_str(sat), m4q.val_to_str(du),
                                                        m4q.val_to_str(qf_val), m4q.val_to_str(r_val))
            if not os.path.exists(path):
                os.makedirs(path)
            transparent=False

            # fig, axes = plot_operator(model2.A, qubit.dim_x)
            # fig.savefig(path + 'ops_order_{}.png'.format(order), transparent=transparent)

            xs, us = data
            fig, axes = plt.subplots(2, 1)
            ax = axes[0]
            ilabels = [0, 4, 8]
            for irow, row in enumerate(xs[:, :-1]):
                ilabel = 'r{}'.format(np.base_repr(irow, base=3).zfill(2))
                ax_args = {'marker': '.', 'markerfacecolor': 'None', 'label': ilabel, 'alpha': 1}
                if irow in ilabels:
                    ax.plot(clock.ts_sim, row.real, **ax_args)
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
            ax.axhline(0,clock.ts_sim[0], clock.ts_sim[-1] + clock.dt)
            fig.tight_layout()
            fig.savefig(path + 'control_order_{}.png'.format(order), transparent=transparent)

    def test_NOT_state(self):
        """
        This code demonstrates some MPC controls and corresponding trajectories for a state transfer.

        For parameters, e.g.: First fix dt = 1ns. Then make the values of the other variables match experimental values
         based on this scaling.
        """
        # Parameters
        # ==========
        clock = m4q.StepClock(dt=1, horizon=10, n_steps=20)
        sat = 2 * np.pi * 0.1
        du = 0.5 * sat

        # Experiment
        # ==========
        wq = 2 * np.pi * 4
        freqs = {'wQ': wq, 'wD': wq, 'wR': wq}
        qubit = RWA_Qubit(**freqs)

        # Exact model
        # -----------
        # Construct |i><j| basis.
        measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s)
                        for j in range(qubit.dim_s)]
        A_cts_list = [m4q.vectorize_me(op, measure_list) for op in qubit.H_list]
        for order in range(1, 3):
            np.random.seed(1)
            A_init = m4q.discretize_homogeneous(A_cts_list, clock.dt, order)

            # # Add discrepancy to qubit
            # # ------------------------
            freqs['wQ'] = wq * .99
            qubit = RWA_Qubit(**freqs)

            # Cost
            # ====
            Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
            qf_val = 1
            Qf = Q * qf_val
            r_val = 1e-2 / sat ** 2
            R = r_val * np.identity(qubit.dim_u)

            # Set control problem
            Rx = qt.qip.operations.rx(1e-4)
            rho0 = Rx * qt.basis(qubit.dim_s, 0).proj() * Rx.dag()
            rho1 = qt.basis(qubit.dim_s, 1).proj()
            initial_state = rho0.data.toarray().flatten()
            target_state = rho1.data.toarray().flatten()

            # Benchmarks
            # ----------
            # Benchmarks are shifted along horizons during the run
            X_bm = np.hstack([target_state[:, None]] * (clock.n_steps + clock.horizon + 1))
            U_bm = np.hstack([np.zeros((qubit.dim_u, 1))] * (clock.n_steps + clock.horizon))

            # Model (n.b. we're not using any data-driven machinery here.)
            # =====
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            model1 = m4q.DMDc(qubit.dim_x, qubit.dim_x, qubit.dim_x * dim_lift, A_init)

            # Model predictive control
            # ========================
            data, model2, exit_code = m4q.mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE, model1, Q,
                                              R, Qf, sat=sat, du=du)

            # DIAGNOSTIC
            # **********
            path = './../playground/{}_NOT/wQ_{}/'.format(rootname, int(wq)) + \
                   '{}_sat_{}_du_{}_Qf_{}_R_{}/'.format(clock.to_string(), m4q.val_to_str(sat), m4q.val_to_str(du),
                                                        m4q.val_to_str(qf_val), m4q.val_to_str(r_val))
            transparent = False
            if not os.path.exists(path):
                os.makedirs(path)

            # fig, axes = plot_operator(model2.A, qubit.dim_x)
            # fig.savefig(path + 'ops_order_{}.png'.format(order), transparent=transparent)

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

    def test_NOT_state_freq(self):
        """
        This code demonstrates some MPC controls and corresponding trajectories for a state transfer.

        For parameters, e.g.: First fix dt = 1ns. Then make the values of the other variables match experimental values
         based on this scaling.
        """
        # Parameters
        # ==========
        clock = m4q.StepClock(dt=0.2, horizon=50, n_steps=100)
        sat = 2 * np.pi * 0.1
        du = 0.1 * sat
        clock.measure_freq = 5

        # Experiment
        # ==========
        wq = 2 * np.pi * 4
        freqs = {'wQ': wq, 'wD': wq, 'wR': wq}
        qubit = RWA_Qubit(**freqs)

        # Exact model
        # -----------
        # Construct |i><j| basis.
        measure_list = [qt.basis(qubit.dim_s, i) * qt.basis(qubit.dim_s, j).dag() for i in range(qubit.dim_s)
                        for j in range(qubit.dim_s)]
        A_cts_list = [m4q.vectorize_me(op, measure_list) for op in qubit.H_list]
        for order in range(1, 3):
            np.random.seed(1)
            A_init = m4q.discretize_homogeneous(A_cts_list, clock.dt, order)

            # # Add discrepancy to qubit
            # # ------------------------
            freqs['wQ'] = wq * .99
            qubit = RWA_Qubit(**freqs)

            # Cost
            # ====
            Q = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
            qf_val = 1
            Qf = Q * qf_val
            r_val = 1e-2
            R = r_val * np.identity(qubit.dim_u)

            # Set control problem
            Rx = qt.qip.operations.rx(1e-4)
            rho0 = Rx * qt.basis(qubit.dim_s, 0).proj() * Rx.dag()
            rho1 = qt.basis(qubit.dim_s, 1).proj()
            initial_state = rho0.data.toarray().flatten()
            target_state = rho1.data.toarray().flatten()

            # Benchmarks
            # ----------
            # Benchmarks are shifted along horizons during the run
            X_bm = np.hstack([target_state[:, None]] * (clock.n_steps + clock.horizon + 1))
            U_bm = np.hstack([np.zeros((qubit.dim_u, 1))] * (clock.n_steps + clock.horizon))

            # Model (n.b. we're not using any data-driven machinery here.)
            # =====
            dim_lift = len(create_library(order, qubit.dim_u)[1:])
            model1 = m4q.DMDc(qubit.dim_x, qubit.dim_x, qubit.dim_x * dim_lift, A_init)

            # Model predictive control
            # ========================
            data, model2, exit_code = m4q.mpc(initial_state, qubit.dim_u, order, X_bm, U_bm, clock, qubit.QE, model1, Q,
                                              R, Qf, sat=sat, du=du)

            # DIAGNOSTIC
            # **********
            path = './../playground/{}_mfreq_NOT/wQ_{}/'.format(rootname, int(wq)) + \
                   '{}_sat_{}_du_{}_Qf_{}_R_{}/'.format(clock.to_string(), m4q.val_to_str(sat), m4q.val_to_str(du),
                                                        m4q.val_to_str(qf_val), m4q.val_to_str(r_val))
            transparent = False
            if not os.path.exists(path):
                os.makedirs(path)

            # fig, axes = plot_operator(model2.A, qubit.dim_x)
            # fig.savefig(path + 'ops_order_{}.png'.format(order), transparent=transparent)

            xs, us = data
            xs = np.atleast_2d(xs[:, :-1][:, ::clock.measure_freq])
            ts = clock.ts_sim[::clock.measure_freq]
            fig, axes = plt.subplots(2, 1)
            ax = axes[0]
            for row in xs:
                ax.plot(ts, row.real, marker='o', markerfacecolor='None')
            ax.set_ylim([-1.1, 1.1])
            ax = axes[1]
            infidelity = [1 - qt.fidelity(qt.Qobj(x.reshape(2, 2)), rho1) for x in xs.T]
            ax.plot(ts, infidelity)
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
