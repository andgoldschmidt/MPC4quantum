import numpy as np
# TODO:
#   1.  Decide if U=None is permissible.
#   2.  Insert dimension / operator checks and warnings


class DMDc:
    """
    DMDc interface (the interface acts as a minimal container for read-only DMDc models).
    """
    def __init__(self, dim_y, dim_x, dim_u, A0):
        """
        A DMDc linear control model. The read-only version of the class can predict future states using the linear
        control model and return views on the linear control operators. The fully-implemented version of the class must
        update the fit model based on single iterations of new data (streaming data).

        :param dim_y: Output state dimension.
        :param dim_x: Input state dimension.
        :param dim_u: Input control dimension.
        :param A0: Initial model of shape (dim_y, dim_x * dim_u).
        """
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.A = A0

        # Default discount factor; if the half-life is k iterations, then the discount is 2^(-1/k).
        self.discount = 1

        # Default rcond for pinv regularization.
        self.rcond = 1e-15

    @classmethod
    def from_data(cls, Y, X, U, **kwargs):
        """
        Fit a DMDc model using snapshots matrices.

        :param Y: Output state data of shape (dim_y, n_snapshots).
        :param X: Input state data of shape (dim_x, n_snapshots)
        :param U: Input control data of shape (dim_u, n_snapshots)
        :return: A DMDc object.
        """
        raise NotImplementedError()

    @classmethod
    def from_bootstrap(cls, dim_y, dim_x, dim_u, A0, **kwargs):
        """
        Fit a DMDc model by bootstrapping to an initial model.

        :param dim_y: Output state dimension.
        :param dim_x: Input state dimension.
        :param dim_u: Input control dimension.
        :param A0: Initial model of shape (dim_y, dim_x * dim_u).
        :return: A DMDc object.
        """
        raise NotImplementedError()

    @classmethod
    def from_randn(cls, dim_y, dim_x, dim_u, **kwargs):
        """
        Fit a DMDc model by bootstrapping to a random model.

        :param dim_y: Output state dimension.
        :param dim_x: Input state dimension.
        :param dim_u: Input control dimension.
        :return: A DMDc object.
        """
        raise NotImplementedError()

    def fit_iteration(self, next_y, next_x, next_u):
        """
        Streaming update of the DMDc model based on the next data.

        :param next_y: Next ouput state.
        :param next_x: Next input state.
        :param next_u: Next control state.
        :return: The updated linear control model as returned by self.get_discrete().
        """
        raise NotImplementedError()

    def predict(self, current_x, current_u):
        """
        Make a prediction of the output data using the current model and the provided input data. Can predict from
        single iterations or full snapshot matrices.

        :param current_x: The input state of shape dim_x or (dim_x, n_snapshots)
        :param current_u: The input control of shape dim_u or (dim_u, n_snapshots)
        :return: The predicted output state of shape (dim_y, 1) or (dim_u, n_snapshots)
        """
        A_x, A_u = self.get_discrete()
        current_x = current_x.reshape(self.dim_x, -1)
        current_u = current_u.reshape(self.dim_u, -1)
        return A_x @ current_x + A_u @ current_u

    def get_discrete(self):
        """
        Return the linear control model A, B such that y = A @ x + B @ u.

        :return: Two operators of shapes (dim_y, dim_x) and (dim_y, dim_u), respectively.
        """
        A_x = self.A[:self.dim_y, :self.dim_x]
        A_u = self.A[:self.dim_y, self.dim_x:]
        return A_x, A_u

    # def get_continuous_from_discrete(self):
    #     raise NotImplementedError()


class DiscrepDMDc(DMDc):
    """
    Offline DMDc implementation. Uses discprepancies to enable bootstrapping.

    Notes:
        1.  Consider improvements with streaming ideas to avoid repeat pinv.

    TODO:
        1. Need to implement a copy method.
    """
    def __init__(self, dim_y, dim_x, dim_u, A0, **kwargs):
        super().__init__(dim_y, dim_x, dim_u, A0)
        self.initialization = kwargs

        # Update available kwargs
        self.Y = kwargs['Y'] if 'Y' in kwargs else None
        self.X = kwargs['X'] if 'X' in kwargs else None
        self.U = kwargs['U'] if 'U' in kwargs else None
        self.discount = kwargs['discount'] if 'discount' in kwargs else self.discount
        self.rcond = kwargs['rcond'] if 'rcond' in kwargs else self.rcond
        self.min_rank = dim_x

        # Saved data
        self.iA = [A0]
        self._save = False
        self._iteration = 0
        self._isave = 10

    @classmethod
    def from_randn(cls, dim_y, dim_x, dim_u, **kwargs):
        """
        Fit a DMDc model by bootstrapping to a (real) random normal model of a specified standard deviation.

        :param dim_y: Output state dimension.
        :param dim_x: Input state dimension.
        :param dim_u: Input control dimension.
        :param kwargs: Requires sigma, the standard deviation of the normal distribution for the random operator.
        :return: A DMDc object.
        """
        sigma = kwargs['sigma']
        A0 = np.random.randn(dim_y, dim_x + dim_u) * sigma
        return cls(dim_y, dim_x, dim_u, A0, **{'sigma': sigma})

    @classmethod
    def from_bootstrap(cls, dim_y, dim_x, dim_u, A0, **kwargs):
        # docstring inherited
        return cls(dim_y, dim_x, dim_u, A0)

    @classmethod
    def from_data(cls, Y, X, U=None, **kwargs):
        """
        Fit a DMDc model using snapshots matrices.

        :param Y: Output state data of shape (dim_y, n_snapshots).
        :param X: Input state data of shape (dim_x, n_snapshots)
        :param U: Input control data of shape (dim_u, n_snapshots)
        :param kwargs: Requires rcond, the rcond to use in pinv for the fit.
        :return: A DMDc object.
        """
        rcond = kwargs['rcond']
        dim_y = Y.shape[0]
        dim_x = X.shape[0]
        if U is None:
            dim_u = 0
            Z = X
        else:
            dim_u = U.shape[0]
            Z = np.vstack([X, U])
        # Singular values less than or equal to rcond * largest_singular_value are set to zero
        A0 = Y @ np.linalg.pinv(Z, rcond=rcond)
        return cls(dim_y, dim_x, dim_u, A0, **{'Y': Y, 'X': X, 'U': U, 'rcond': rcond})

    @staticmethod
    def _update_stack(val, stack, discount):
        val = val.reshape(-1, 1)
        return val if stack is None else np.hstack([discount * stack, val])

    def fit_iteration(self, next_y, next_x, next_u=np.array([])):
        # docstring inherited
        # Update data matrices
        self.Y = self._update_stack(next_y, self.Y, self.discount)
        self.X = self._update_stack(next_x, self.X, self.discount)
        self.U = self._update_stack(next_u, self.U, self.discount)

        # Add the current discrepancy model to the previous model.
        # TODO: set step size for update?
        if np.linalg.matrix_rank(self.X) >= self.min_rank:
            current_Y = self.predict(self.X, self.U)
            current_Z = np.vstack([self.X, self.U])
            A1 = (self.Y - current_Y) @ np.linalg.pinv(current_Z, rcond=self.rcond)
            self.A = self.A + A1

        # Save iteration
        self._iteration += 1
        if self._save and self._iteration % self._isave == 0:
            self.iA.append(np.copy(self.A))

        # Return the current linear model
        return self.get_discrete()


class OnlineDMDc(DMDc):
    """
    Online DMDc for overconstrained systems.
    C.f. Online Dynamic Mode Decomposition for Time-Varying Systems by Zhang et al.

    Notes:
        1.  Consider taking the SVD of the data matrix X in order to include a rank truncation for noisy data. Can
            this be applied iteratively?
        2.  Robust regularization of the recursive least squares algorithm? Or, how to regularize the rank-1 updates?
    """

    def __init__(self, dim_y, dim_x, dim_u, P0, A0, **kwargs):
        super().__init__(dim_y, dim_x, dim_u, A0)
        self.initialization = kwargs

        # Store P
        self.P = P0

        # Saved data
        self.iP = [P0]
        self.iA = [A0]
        self._save = False
        self._iteration = 0
        self._isave = 10

    @classmethod
    def from_randn(cls, dim_y, dim_x, dim_u, **kwargs):
        """
        Fit a DMDc model by bootstrapping to a (real) random normal model of a specified standard deviation.

        :param dim_y: Output state dimension.
        :param dim_x: Input state dimension.
        :param dim_u: Input control dimension.
        :param kwargs:
            Requires:
                sigma, the standard deviation of the normal distribution for the random operator.
                alpha, the update rate multiplying the estimated P operator (unknown covariance) (try: 1e2)
        :return: A DMDc object.
        """
        sigma = kwargs['sigma']
        alpha = kwargs['alpha']
        dim_z = dim_x + dim_u
        P0 = alpha * np.identity(dim_z)
        A0 = np.random.randn(dim_y, dim_z) * sigma
        return cls(dim_y, dim_x, dim_u, P0, A0, **kwargs)

    @classmethod
    def from_bootstrap(cls, dim_y, dim_x, dim_u, A0, **kwargs):
        """
        Fit a DMDc model by bootstrapping to an initial model.

        :param dim_y: Output state dimension.
        :param dim_x: Input state dimension.
        :param dim_u: Input control dimension.
        :param A0: Initial model of shape (dim_y, dim_x * dim_u).
        :param kwargs:
            Requires alpha, the update rate multiplying the estimated P operator (unknown covariance) (try: 1e2)
        :return: A DMDc object.
        """
        alpha = kwargs['alpha']
        P0 = alpha * np.identity(dim_x + dim_u)
        return cls(dim_y, dim_x, dim_u, P0, A0, **kwargs)

    @classmethod
    def from_data(cls, Y, X, U=None, **kwargs):
        # docstring inherited
        dim_y = Y.shape[0]
        dim_x = X.shape[0]
        if U is None:
            dim_u = 0
            Z = X
        else:
            dim_u = U.shape[0]
            Z = np.vstack([X, U])
        # We assume this is full rank and not ill-conditioned.
        P0 = np.linalg.pinv(Z @ Z.T)
        A0 = Y @ Z.T @ P0
        return cls(dim_y, dim_x, dim_u, P0, A0, **{'Y': Y, 'X': X, 'U': U})

    def fit_iteration(self, next_y, next_x, next_u=np.array([])):
        # docstring inherited
        # Compute state and output with useful operator products
        next_y = next_y.reshape(-1, 1)
        next_z = np.vstack([next_x.reshape(-1, 1), next_u.reshape(-1, 1)])
        Az = self.A @ next_z
        Pz = self.P @ next_z
        gamma = 1 / (1 + next_z.T @ Pz)

        # Online update
        self.A = self.A + gamma * (next_y - Az) @ Pz.T
        self.P = (self.P - gamma * Pz @ Pz.T) / self.discount
        self._iteration += 1
        if self._save and self._iteration % self._isave == 0:
            self.iA.append(np.copy(self.A))
            self.iP.append(np.copy(self.P))

        # Return the current linear model
        return self.get_discrete()
