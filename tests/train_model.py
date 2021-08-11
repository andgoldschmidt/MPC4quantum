from mpc4quantum import *

import qutip as qt
import matplotlib.pyplot as plt
cmap = plt.get_cmap('tab10')


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
        # H2 = 1 / 2 * self._A0 * qt.sigmay()

        self.H_list = [H0, H1]
        self.QE = QExperiment(H0, [H1])
        # self.H_list = [H0, H1, H2]
        # self.QE = QExperiment(H0, [H1, H2])

    def u1(self, ts, args):
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt']) * np.cos((self._w1 - self._wR) * ts)


def train_model(pulse_width, clock, qubit, order):
    # User inputs
    # ===========
    x0_train = qt.basis(qubit.dim_s, 0).proj().data.toarray().flatten()
    lib_fns = create_library(order, qubit.dim_u)[1:]  # ignore 0th power

    ts_train = np.arange(0, pulse_width * 2, clock.dt)
    args1 = {'t0': 0, 'tf': pulse_width, 'dt': clock.dt, 'A': 1}
    u1 = qubit.u1(ts_train, args1)
    u1 = u1[None, :]

    # Two pulses
    # ----------
    # u1 = np.vstack([u, np.zeros_like(u)])
    # u2 = np.vstack([np.zeros_like(u), u])

    # Training data
    # =============
    training_list = [u1]
    # training_list = [u1, u2]
    n_training = len(training_list)

    # Simulate data
    # -------------
    X2 = []
    X1 = []
    U1 = []
    UX1 = []
    for us_train in training_list:
        xs_train = qubit.QE.simulate(x0_train, ts_train, us_train)
        # xs_train = xs_train + 1e-2 * np.random.randn(*xs_train.shape)
        X2.append(xs_train[:, 1:])
        X1.append(xs_train[:, :-1])
        U1.append(np.vstack([f(us_train) for f in lib_fns])[:, :-1])
        UX1.append(krtimes(U1[-1], X1[-1]))
    X2 = np.hstack(X2)
    X1 = np.hstack(X1)
    U1 = np.hstack(U1)
    UX1 = np.hstack(UX1)

    # Training models (hyper-parameter optimization)
    # ===============
    smallest_loss = np.inf
    best_rcond = None
    best_model = None
    # User-defined resolution
    # -----------------------
    for rcond in np.logspace(-6, -1, 10):
        current_model = DiscrepDMDc.from_data(X2, X1, UX1, rcond=rcond)

        # Reshape data
        # ^^^^^^^^^^^^
        # dimension of variable, number of training experiments, number of training timesteps
        if n_training > 1:
            X2_reshape = X2.reshape(X2.shape[0], n_training, -1)
            X1_reshape = X1.reshape(X1.shape[0], n_training, -1)
            U1_reshape = U1.reshape(U1.shape[0], n_training, -1)
        else:
            X2_reshape = X2[:, None, :]
            X1_reshape = X1[:, None, :]
            U1_reshape = U1[:, None, :]

        loss = 0
        for a_training in range(n_training):
            loss += get_prediction_loss(current_model, X2_reshape[:, a_training, :], X1_reshape[:, a_training, :],
                                        U1_reshape[:, a_training, :])

        if loss < smallest_loss:
            smallest_loss = loss
            best_model = current_model
            best_rcond = rcond
    return best_model, best_rcond


def get_prediction_loss(current_model, X2, X1, U1):
    X2_predict = [None] * (X2.shape[1] + 1)
    X2_predict[0] = X1[:, 0].reshape(-1, 1)
    for i in range(X2.shape[1]):
        current_ux = krtimes(U1[:, i].reshape(-1, 1), X2_predict[i])
        X2_predict[i + 1] = current_model.predict(X2_predict[i], current_ux)
    X2_predict = np.hstack(X2_predict[1:])

    # # DIAGNOSTIC
    # # **********
    # fig, axes = plt.subplots(2, 1)
    # fake_ts = np.arange(X2.shape[1])
    # ax = axes[0]
    # for i in range(X2.shape[0]):
    #     ax.plot(fake_ts, X2_predict[i].real, c=cmap(i))
    #     ax.plot(fake_ts, X2[i].real, alpha=0.5, lw=3, c=cmap(i))
    # ax = axes[1]
    # for i in range(U1.shape[0]):
    #     ax.plot(fake_ts, U1[i], c='k', ls='--')
    # ax.set_ylim([-1, 1])
    # fig.show()

    return np.linalg.norm(X2 - X2_predict, 2)