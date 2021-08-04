from mpc4quantum import *

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
        # H2 = 1 / 2 * self._A0 * qt.sigmay()

        self.QE = QExperiment(H0, [H1])

    def u1(self, ts, args):
        return args['A'] * blackman(ts, args['t0'], args['tf'], args['dt']) * np.cos((self._w1 - self._wR) * ts)


def train_model(clock, qubit, order):
    # Inputs
    x0_train = qt.basis(qubit.dim_s, 0).proj().data.toarray().flatten()

    args = {'t0': 0, 'tf': 25, 'dt': clock.dt, 'A': 1}
    ts_train = np.arange(args['t0'], 50, args['dt'])

    lib_fns = create_library(order, qubit.dim_u)[1:]  # ignore 0th power
    u = qubit.u1(ts_train, args)
    u1 = u[None, :]
    # u1 = np.vstack([u, np.zeros_like(u)])
    # u2 = np.vstack([np.zeros_like(u), u])

    # Simulate data
    X2 = []
    X1 = []
    U1 = []
    UX1 = []
    for us_train in [u1]:  # , u2]:
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

    # Training (hyper-parameter optimization)
    smallest_loss = np.inf
    best_rcond = None
    best_model = None
    for rcond in np.logspace(-6, -1, 5):
        current_model = DiscrepDMDc.from_data(X2, X1, UX1, rcond=rcond)
        X2_predict = [None] * (X2.shape[1] + 1)
        X2_predict[0] = X1[:, 0].reshape(-1, 1)
        for i in range(X2.shape[1]):
            current_ux = krtimes(U1[:, i].reshape(-1, 1), X2_predict[i])
            X2_predict[i + 1] = current_model.predict(X2_predict[i], current_ux)
        X2_predict = np.hstack(X2_predict[1:])
        loss = np.linalg.norm(X2 - X2_predict, 2)

        # DIAGNOSTIC
        # for i in range(qubit.dim_x):
        #     plt.plot(ts_train[:-1], X2_predict[i].real, c=cmap(i))
        #     plt.plot(ts_train[:-1], X2[i].real, alpha=0.5, lw=3, c=cmap(i))
        # ax2 = plt.twinx()
        # ax2.plot(ts_train[:-1], U1[0], c='k', ls='--')
        # plt.title(str(rcond) + ': ' + str(loss))
        # plt.ylim([-1, 1])
        # plt.show()

        if loss < smallest_loss:
            smallest_loss = loss
            best_model = current_model
            best_rcond = rcond
    return best_model, best_rcond