from abc import ABC, abstractmethod
import numpy as np
from qutip import mesolve, propagator, Qobj, tensor
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class Experiment(ABC):
    """
    The Experiment class is an interface for simulating ODEs with external controls.
    """
    def __init__(self):
        # Save the last simulation
        self.ts = None
        self.us = None
        self.xs = None

    @abstractmethod
    def f(self, t, x, u):
        """
        The function evaluation of the ODE.

        :param t: Current time
        :param x: Current state
        :param u: Current control
        :return: The time derivative of the state.
        """

    @staticmethod
    def lift(x):
        # If present, 2nd dimension of :return: should match :param: x.
        return x

    @staticmethod
    def proj(z):
        # If present, 2nd dimension of :return: should match :param: z.
        return z

    @abstractmethod
    def simulate(self, x0, ts, us):
        """
        Solve the initial value problem starting at x0 for the times spanning the first and last values of ts. The
        output is evaluated at all times in ts.

        :param x0: The initial state.
        :param ts: The time series to evaluate the output states.
        :param us: The control can be either a function of time or an ndarray of the length of ts.
        :return: The solution state of the initial value problem.
        """


class CExperiment(Experiment):
    """
    The CExperiment class implements solve_ivp to solve classical experiments.
    """
    def __init__(self):
        # Save the last simulation
        super().__init__()
        self._interpolation = 'linear'

    @abstractmethod
    def f(self, t, x, u):
        """
        The function evaluation of the ODE.

        :param t: Current time
        :param x: Current state
        :param u: Current control
        :return: The time derivative of the state.
        """

    def _f(self, t, x):
        return self.f(t, x, self.us(t)).flatten()

    def simulate(self, x0, ts, us):
        """
        Solve the initial value problem starting at x0 for the times spanning the first and last values of ts. The
        output is evaluated at all times in ts.

        :param x0: The initial state.
        :param ts: The time series to evaluate the output states.
        :param us: The control can be either a function of time or an ndarray of the length of ts.
        :return: The solution state of the initial value problem.
        """
        self.ts = ts
        # Control function or series must map times to inputs
        self.us = us if callable(us) else interp1d(ts, us, kind=self._interpolation)
        res = solve_ivp(self._f, [self.ts[0], self.ts[-1]], x0, t_eval=self.ts)
        self.xs = res.y
        return res.y


class VanDerPol(CExperiment):
    """
    To define a new kind of experiment, we need to specify the ODE function, f. This class simulates the Van der Pol
    oscillator.

    We can also use this instance to add system-specific characteristics. For example, in this class we defined a lift
    function in order to standardize a Koopman linearization of the Van der Pol oscillator.

    We also add LQR set/get methods for interactin with the LQR algorothm.
    """
    def __init__(self, mu):
        super().__init__()
        self.dim_x = 2
        self.dim_u = 1
        self.mu = mu

        # Set/Get for LQR
        self.Q = None
        self.R = None
        self.Qf = None
        self.target = None
        self._set_cost = False
        self._set_target = False

    def f(self, t, x, u):
        x1, x2 = x
        return np.block([
            x2,
            -x1 + self.mu * (1 - x1 ** 2) * x2 + u
        ])

    @staticmethod
    def lift(x):
        x1, x2 = x
        z = np.vstack([x1, x2, x1 ** 2, x1 ** 2 * x2])
        return z if x.ndim > 1 else z.flatten()

    @staticmethod
    def proj(z):
        return z[:2, :] if z.ndim > 1 else z[:2]


class Rotor(CExperiment):
    """
    A simple controlled rotation matrix.
    """
    def __init__(self, epsilon):
        super().__init__()
        self.dim_x = 2
        self.dim_u = 1
        self.epsilon = epsilon

    def f(self, t, x, u):
        x1, x2 = x
        omega = 1 + self.epsilon * u
        return np.block([
            omega * x2,
            -omega * x1
        ])


# Utility functions for interp1d controls
def _wrap_u(u_func, index_u):
    def new_func(ts, args):
        return u_func(ts)[index_u]
    return new_func


def _wrap_us(us):
    if callable(us):
        if hasattr(us, 'y'):
            # Assume us is an interp1d object
            u_dim = us.y.reshape(-1, len(us.x)).shape[0]
            us = np.array([_wrap_u(us, i) for i in range(u_dim)])
        else:
            raise AttributeError('QExperiment expected an interp1d object.')
    else:
        us = np.atleast_2d(us)
        u_dim = us.shape[0]
    return us, u_dim


class QExperiment(Experiment):
    """
    The QExperiment class implements qutip's mesolve to solve quantum state preparation experiments.

    TODO:
     - sigma doesn't make much sense, what you want is a confusion matrix acting on the measurement result.
     - the confusion matrix retains the density matrix structure but with adjusted probability estimates.
    """
    def __init__(self, H0, H1_list):
        super().__init__()
        self.H0 = H0
        self.H1_list = H1_list
        self._me_args = {}
        self._sigma = 0

    def f(self, t, x, u):
        return self.H0 * x + np.sum([H1 * x * u1 for H1, u1 in zip(self.H1_list, u)], axis=0)

    def set_sigma(self, sigma):
        self._sigma = sigma

    def set(self, key, value):
        """
        Set a keyword argument for QuTip's mesolve.
        """
        self._me_args[key] = value

    def simulate(self, x0, ts, us):
        self.set('rho0', Qobj(x0.reshape(*self.H0.shape)))
        self.ts = ts
        self.set('tlist', self.ts)
        # Check is 'us' is a function or an ndarray
        self.us, u_dim = _wrap_us(us)
        self.set('H', [self.H0] + [[self.H1_list[i_row], self.us[i_row]] for i_row in range(u_dim)])
        res = mesolve(**self._me_args)
        self.xs = np.array(res.expect) if 'e_ops' in self._me_args \
            else np.vstack([s.full().flatten() for s in res.states]).T
        return self.xs + (np.random.randn(*self.xs.shape) + 1j * np.random.randn(*self.xs.shape)) * self._sigma


class QCoupledExperiment(QExperiment):
    """
    The QExperiment class implements qutip's mesolve to solve quantum state preparation experiments. We also
    use partial traces to get only the qubit subspaces as the lifted state (I know this is a bit counter to the
    intent of lifting refering to the creation of many Koopman observables which live in the model space, but
    this will let us use models defined only in the qubit spaces.
    """
    def __init__(self, H0, H1_list):
        super().__init__(H0, H1_list)

    @staticmethod
    def lift(rhoAB_vec):
        # Get dimensions
        dim_AB2 = len(rhoAB_vec)
        dim_AB = isqrt(dim_AB2)
        # Make an assumption about identical qubits
        dim_A2 = dim_AB
        dim_A = isqrt(dim_A2)
        dim_B = dim_A

        # Partial traces
        rhoAB_square = rhoAB_vec.reshape(dim_A * dim_B, dim_A * dim_B)

        # Partial trace over B
        rhoA_square = np.zeros((dim_A, dim_A), dtype=np.complex128)
        idA = np.identity(dim_A)
        for i in range(dim_B):
            ket = np.zeros((dim_B, 1))
            ket[i] = 1
            right = np.kron(idA, ket)
            left = np.kron(idA, ket.T)
            rhoA_square += left @ rhoAB_square @ right

        # Partial trace over A
        rhoB_square = np.zeros((dim_B, dim_B), dtype=np.complex128)
        idB = np.identity(dim_B)
        for i in range(dim_A):
            ket = np.zeros((dim_A, 1))
            ket[i] = 1
            right = np.kron(ket, idB)
            left = np.kron(ket.T, idB)
            rhoB_square += left @ rhoAB_square @ right

        # Stack
        rhoA_vec = rhoA_square.flatten()
        rhoB_vec = rhoB_square.flatten()
        rhoA_rhoB_vec = np.hstack([rhoA_vec, rhoB_vec])
        return rhoA_rhoB_vec

    @staticmethod
    def proj(rhoA_rhoB_vec):
        # Get dimensions
        dim_A2_B2 = len(rhoA_rhoB_vec)
        # Make an assumption about identical qubits
        dim_2A2 = dim_A2_B2
        dim_A2 = dim_2A2 // 2
        dim_A = isqrt(dim_A2)
        dim_B = dim_A

        # Split into two operators
        rhoA_vec = rhoA_rhoB_vec[:dim_A**2]
        rhoB_vec = rhoA_rhoB_vec[dim_A**2:]
        rhoA_square = rhoA_vec.reshape(dim_A, dim_A)
        rhoB_square = rhoB_vec.reshape(dim_B, dim_B)

        # Compute the tensor product
        rhoAB_square = np.kron(rhoA_square, rhoB_square)
        rhoAB_vec = rhoAB_square.flatten()
        return rhoAB_vec


def split_blocks(bmatrix, nrows, ncols):
    """
    Split a block matrix into sub-blocks.
    https://stackoverflow.com/questions/11105375/
    """
    r, h = bmatrix.shape
    return bmatrix.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)


def isqrt(n):
    """
    Integer square root (can replace with math.isqrt if Python >= 3.8).
    https://stackoverflow.com/questions/15390807/
    """
    if n > 0:
        x = 1 << (n.bit_length() + 1 >> 1)
        while True:
            y = (x + n // x) >> 1
            if y >= x:
                return x
            x = y
    elif n == 0:
        return 0
    else:
        raise ValueError("Square root not defined for negative numbers.")


class QSynthesis(Experiment):
    """
    The QSynthesis class implements qutip's propogator to solve quantum gate synthesis.
    """
    def __init__(self, H0, H1_list):
        super().__init__()
        self.H0 = H0
        self.H1_list = H1_list
        self._prop_args = {}

    def f(self, t, x, u):
        return self.H0 * x + np.sum([H1 * x * u1 for H1, u1 in zip(self.H1_list, u)], axis=0)

    def set(self, key, value):
        """
        Set a keyword argument for QuTip's propagator.
        """
        self._prop_args[key] = value

    @staticmethod
    def lift(U):
        """
        By way of analogy, the process matrix P is the density matrix rho for a unitary operator U.
        It is defined under a presumed vectorization of rho using numpy's flatten s.t. P = U 'otimes' U^*.

        :param U: A unitary matrix of shape (n^2,).
        :return: The flat process operator (n^4,) associated to the provided unitary.
        """
        #
        n = isqrt(U.shape[0])
        U = U.reshape(n, n)
        return np.kron(U, U.conj()).flatten()

    @staticmethod
    def proj(P):
        """
        Shape the process matrix P = U 'otimes' U^* into a single propagator, U.

        :param P: The process matrix of shape (n^4,).
        :return: A unitary operator equivalent to P (up to global phase).
        """
        # Shape the initial condition (U 'otimes' U^*) into a single propagator, U.
        n = isqrt(isqrt(P.shape[0]))
        blocks = split_blocks(P.reshape(n ** 2, n ** 2), n, n)
        U = np.zeros((n, n))
        # Look for a nonzero block, and divide out the prefactor from the kronecker product.
        for i, val in enumerate([np.any(b) for b in blocks]):
            if val:
                # Complex square root (allow negative numbers)
                U = blocks[i].conj() / np.lib.scimath.sqrt(blocks[i].flatten()[i])
                break
        return U.flatten()

    def simulate(self, x0, ts, us):
        # The initial condition is a single unitary U (its lifted state is U 'otimes' U^*)
        n = self.H0.shape[1]
        x0 = Qobj(self.proj(x0).reshape(n, n))

        # Check is 'us' is a function or an ndarray
        self.us, u_dim = _wrap_us(us)
        self.set('H', [self.H0] + [[self.H1_list[i_row], self.us[i_row]] for i_row in range(u_dim)])
        self.ts = ts

        # Cases: Avoid hitting a qutip 'feature' for ts <= 2.
        if len(self.ts) > 2:
            self.set('t', self.ts)
            # Unitary mode 'single' avoids the need to check if dtype is Qobj or memoryview
            self.xs = propagator(**self._prop_args, unitary_mode='single')
        else:
            self.xs = [None] * len(self.ts)
            for i, t in enumerate(self.ts):
                self.set('t', t)
                self.xs[i] = propagator(**self._prop_args, unitary_mode='single')

        # Append the initial condition to the resulting ndarrays via multiplication, U(t, t0) @ U(t0, 0)
        if len(self.xs) > 0:
            #  Full multiplication, states might be in tensor product spaces. Qobj to match output of propagator.
            self.xs = [Qobj(xi.full() @ x0.full()) for xi in self.xs]

        # Lift the results (U 'otimes' U^*)
        self.xs = np.vstack([self.lift(xi.full().flatten()) for xi in self.xs]).T
        return self.xs
