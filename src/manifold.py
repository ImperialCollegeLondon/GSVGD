"""
Adapted from the Pymanopt library:
    https://www.pymanopt.org/
"""
import torch
import abc
import functools


def multiprod(A, B):
    """
    Inspired by MATLAB multiprod function by Paolo de Leva. A and B are
    assumed to be arrays containing M matrices, that is, A and B have
    dimensions A: (M, N, P), B:(M, P, Q). multiprod multiplies each matrix
    in A with the corresponding matrix in B, using matrix multiplication.
    so multiprod(A, B) has dimensions (M, N, Q).
    """

    # First check if we have been given just one matrix
    if len(A.shape) == 2:
        return A @ B

    # Approx 5x faster, only supported by numpy version >= 1.6:
    return torch.einsum("ijk,ikl->ijl", A, B)


def multitransp(A):
    """
    Inspired by MATLAB multitransp function by Paolo de Leva. A is assumed to
    be an array containing M matrices, each of which has dimension N x P.
    That is, A is an M x N x P array. Multitransp then returns an array
    containing the M matrix transposes of the matrices in A, each of which
    will be P x N.
    """
    # First check if we have been given just one matrix
    if A.ndim == 2:
        return A.T
    return torch.transpose(A, 2, 1)


class Manifold(metaclass=abc.ABCMeta):
    """
    Abstract base class setting out a template for manifold classes. If you
    would like to extend Pymanopt with a new manifold, then your manifold
    should inherit from this class.
    Not all methods are required by all solvers. In particular, first order
    gradient based solvers such as
    :py:mod:`pymanopt.solvers.steepest_descent` and
    :py:mod:`pymanopt.solvers.conjugate_gradient` require
    :py:func:`egrad2rgrad` to be implemented but not :py:func:`ehess2rhess`.
    Second order solvers such as :py:mod:`pymanopt.solvers.trust_regions`
    will require :py:func:`ehess2rhess`.
    All of these methods correspond closely to methods in
    `Manopt <http://www.manopt.org>`_. See
    http://www.manopt.org/tutorial.html#manifolds for more details on manifolds
    in Manopt, which are effectively identical to those in Pymanopt (all of the
    methods in this class have equivalents in Manopt with the same name).
    """

    def __init__(self, name, dimension, point_layout=1):
        assert isinstance(dimension, (int, torch.int)), "dimension must be an integer"
        assert (isinstance(point_layout, int) and point_layout > 0) or (
            isinstance(point_layout, (list, tuple))
            and all(torch.Tensor(point_layout) > 0)
        ), (
            "'point_layout' must be a positive integer or a sequence of "
            "positive integers"
        )

        self._name = name
        self._dimension = dimension
        self._point_layout = point_layout

    def __str__(self):
        """Returns a string representation of the particular manifold."""
        return self._name

    def _get_class_name(self):
        return self.__class__.__name__

    @property
    def dim(self):
        """The dimension of the manifold"""
        return self._dimension

    @property
    def point_layout(self):
        """The number of elements a point on a manifold consists of.
        For most manifolds, which represent points as (potentially
        multi-dimensional) arrays, this will be 1, but other manifolds might
        represent points as tuples or lists of arrays. In this case,
        `point_layout` describes how many elements such tuples/lists contain.
        """
        return self._point_layout

    # Manifold properties that subclasses can define

    @property
    def typicaldist(self):
        """Returns the "scale" of the manifold. This is used by the
        trust-regions solver to determine default initial and maximal
        trust-region radii.
        """
        raise NotImplementedError(
            "Manifold class '{:s}' does not provide a 'typicaldist'".format(
                self._get_class_name()
            )
        )

    # Abstract methods that subclasses must implement

    @abc.abstractmethod
    def inner(self, X, G, H):
        """Returns the inner product (i.e., the Riemannian metric) between two
        tangent vectors `G` and `H` in the tangent space at `X`.
        """

    @abc.abstractmethod
    def proj(self, X, G):
        """Projects a vector `G` in the ambient space on the tangent space at
        `X`.
        """

    @abc.abstractmethod
    def norm(self, X, G):
        """Computes the norm of a tangent vector `G` in the tangent space at
        `X`.
        """

    @abc.abstractmethod
    def rand(self):
        """Returns a random point on the manifold."""

    @abc.abstractmethod
    def randvec(self, X):
        """Returns a random vector in the tangent space at `X`. This does not
        follow a specific distribution.
        """

    @abc.abstractmethod
    def zerovec(self, X):
        """Returns the zero vector in the tangent space at X."""

    # Methods which are only required by certain solvers

    def _raise_not_implemented_error(method):
        """Method decorator which raises a NotImplementedError with some meta
        information about the manifold and method if a decorated method is
        called.
        """

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            raise NotImplementedError(
                "Manifold class '{:s}' provides no implementation for "
                "'{:s}'".format(self._get_class_name(), method.__name__)
            )

        return wrapper

    @_raise_not_implemented_error
    def dist(self, X, Y):
        """Returns the geodesic distance between two points `X` and `Y` on the
        manifold."""

    @_raise_not_implemented_error
    def egrad2rgrad(self, X, G):
        """Maps the Euclidean gradient `G` in the ambient space on the tangent
        space of the manifold at `X`. For embedded submanifolds, this is simply
        the projection of `G` on the tangent space at `X`.
        """

    @_raise_not_implemented_error
    def ehess2rhess(self, X, G, H, U):
        """Converts the Euclidean gradient `G` and Hessian `H` of a function at
        a point `X` along a tangent vector `U` to the Riemannian Hessian of `X`
        along `U` on the manifold.
        """

    @_raise_not_implemented_error
    def retr(self, X, G):
        """Computes a retraction mapping a vector `G` in the tangent space at
        `X` to the manifold.
        """

    @_raise_not_implemented_error
    def exp(self, X, U):
        """Computes the Lie-theoretic exponential map of a tangent vector `U`
        at `X`.
        """

    @_raise_not_implemented_error
    def log(self, X, Y):
        """Computes the Lie-theoretic logarithm of `Y`. This is the inverse of
        `exp`.
        """

    @_raise_not_implemented_error
    def transp(self, X1, X2, G):
        """Computes a vector transport which transports a vector `G` in the
        tangent space at `X1` to the tangent space at `X2`.
        """

    @_raise_not_implemented_error
    def pairmean(self, X, Y):
        """Returns the intrinsic mean of two points `X` and `Y` on the
        manifold, i.e., a point that lies mid-way between `X` and `Y` on the
        geodesic arc joining them.
        """


class Grassmann(Manifold):
    """
    Factory class for the Grassmann manifold. This is the manifold of p-
    dimensional subspaces of n dimensional real vector space. Initiation
    requires the dimensions n, p to be specified. Optional argument k
    allows the user to optimize over the product of k Grassmanns.
    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1 (Note that this is different to manopt!).
    """

    #   I have chaned the retraction to one using the polar decomp as am now
    #   implementing vector transport. See comment below (JT)

    #   April 17, 2013 (NB) :
    #       Retraction changed to the polar decomposition, so that the vector
    #       transport is now correct, in the sense that it is compatible with
    #       the retraction, i.e., transporting a tangent vector G from U to V
    #       where V = Retr(U, H) will give Z, and transporting GQ from UQ to VQ
    #       will give ZQ: there is no dependence on the representation, which
    #       is as it should be. Notice that the polar factorization requires an
    #       SVD whereas the qfactor retraction requires a QR decomposition,
    #       which is cheaper. Hence, if the retraction happens to be a
    #       bottleneck in your application and you are not using vector
    #       transports, you may want to replace the retraction with a qfactor.

    def __init__(self, n, p, k=1):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError(
                "Need n >= p >= 1. Values supplied were n = %d " "and p = %d." % (n, p)
            )
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)

        if k == 1:
            name = "Grassmann manifold Gr({:d}, {:d})".format(n, p)
        elif k >= 2:
            name = "Product Grassmann manifold Gr({:d}, {:d})^{:d}".format(n, p, k)
        dimension = int(k * (n * p - p ** 2))
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return torch.sqrt(self._p * self._k)

    # Geodesic distance for Grassmann
    def dist(self, X, Y):
        u, s, v = torch.linalg.svd(multiprod(multitransp(X), Y))
        s[s > 1] = 1
        s = torch.arccos(s)
        return torch.norm(s)

    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the Grassmann this is the Frobenius inner product.
        return torch.tensordot(G, H, axes=G.ndim)

    def proj(self, X, U):
        return U - multiprod(X, multiprod(multitransp(X), U))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Convert Euclidean into Riemannian Hessian.
        PXehess = self.proj(X, ehess)
        XtG = multiprod(multitransp(X), egrad)
        HXtG = multiprod(H, XtG)
        return PXehess - HXtG

    def retr(self, X, G):
        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        # Calculate 'thin' SVD of X + G
        u, s, vt = torch.linalg.svd(X + G, full_matrices=False)
        return multiprod(u, vt)


    def norm(self, X, G):
        # Norm on the tangent space is simply the Euclidean norm.
        return torch.norm(G)

    # Generate random Grassmann point using qr of random normally distributed
    # matrix.
    def rand(self):
        if self._k == 1:
            X = torch.randn(self._n, self._p)
            q, r = torch.qr(X)
            return q

        X = torch.zeros((self._k, self._n, self._p))
        for i in range(self._k):
            X[i], r = torch.qr(torch.randn(self._n, self._p))
        return X

    def randvec(self, X):
        U = torch.randn(*torch.shape(X))
        U = self.proj(X, U)
        U = U / torch.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def exp(self, X, U):
        u, s, vt = torch.linalg.svd(U, full_matrices=False)
        cos_s = torch.cos(s).unsqueeze(-2)
        sin_s = torch.sin(s).unsqueeze(-2)

        Y = multiprod(multiprod(X, multitransp(vt) * cos_s), vt) + multiprod(
            u * sin_s, vt
        )

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        if self._k == 1:
            Y, unused = torch.qr(Y)
            return Y
        else:
            for i in range(self._k):
                Y[i], unused = torch.qr(Y[i])
            return Y

    def log(self, X, Y):
        ytx = multiprod(multitransp(Y), X)
        At = multitransp(Y) - multiprod(ytx, multitransp(X))
        Bt = torch.linalg.solve(ytx, At)
        u, s, vt = torch.linalg.svd(multitransp(Bt), full_matrices=False)
        arctan_s = torch.arctan(s).unsqueeze(-2)

        U = multiprod(u * arctan_s, vt)
        return U

    def zerovec(self, X):
        if self._k == 1:
            return torch.zeros((self._n, self._p))
        return torch.zeros((self._k, self._n, self._p))


class Stiefel(Manifold):
    """
    Factory class for the Grassmann manifold. This is the manifold of p-
    dimensional subspaces of n dimensional real vector space. Initiation
    requires the dimensions n, p to be specified. Optional argument k
    allows the user to optimize over the product of k Grassmanns.
    Elements are represented as n x p matrices (if k == 1), and as k x n x p
    matrices if k > 1 (Note that this is different to manopt!).
    """

    #   I have chaned the retraction to one using the polar decomp as am now
    #   implementing vector transport. See comment below (JT)

    #   April 17, 2013 (NB) :
    #       Retraction changed to the polar decomposition, so that the vector
    #       transport is now correct, in the sense that it is compatible with
    #       the retraction, i.e., transporting a tangent vector G from U to V
    #       where V = Retr(U, H) will give Z, and transporting GQ from UQ to VQ
    #       will give ZQ: there is no dependence on the representation, which
    #       is as it should be. Notice that the polar factorization requires an
    #       SVD whereas the qfactor retraction requires a QR decomposition,
    #       which is cheaper. Hence, if the retraction happens to be a
    #       bottleneck in your application and you are not using vector
    #       transports, you may want to replace the retraction with a qfactor.

    def __init__(self, n, p, k=1):
        self._n = n
        self._p = p
        self._k = k

        if n < p or p < 1:
            raise ValueError(
                "Need n >= p >= 1. Values supplied were n = %d " "and p = %d." % (n, p)
            )
        if k < 1:
            raise ValueError("Need k >= 1. Value supplied was k = %d." % k)

        if k == 1:
            name = "Grassmann manifold Gr({:d}, {:d})".format(n, p)
        elif k >= 2:
            name = "Product Grassmann manifold Gr({:d}, {:d})^{:d}".format(n, p, k)
        dimension = int(k * (n * p - p ** 2))
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return torch.sqrt(self._p * self._k)

    # Geodesic distance for Grassmann
    def dist(self, X, Y):
        u, s, v = torch.linalg.svd(multiprod(multitransp(X), Y))
        s[s > 1] = 1
        s = torch.arccos(s)
        return torch.norm(s)

    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        # For the Grassmann this is the Frobenius inner product.
        return torch.tensordot(G, H, axes=G.ndim)

    def proj(self, X, U):
        return U - multiprod(X, multiprod(multitransp(X), U))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Convert Euclidean into Riemannian Hessian.
        PXehess = self.proj(X, ehess)
        XtG = multiprod(multitransp(X), egrad)
        HXtG = multiprod(H, XtG)
        return PXehess - HXtG

    def retr(self, X, G):
        # Calculate 'thin' qr decomposition of X + G
        # XNew, r = np.linalg.qr(X + G)

        # We do not need to worry about flipping signs of columns here,
        # since only the column space is important, not the actual
        # columns. Compare this with the Stiefel manifold.

        # Compute the polar factorization of Y = X+G
        q, r = torch.qr(X + G)
        sign_mat = torch.sign(torch.sign(torch.diagonal(r, dim1=1, dim2=2)) + 0.5).to(X.device)
        
        return torch.matmul(q, torch.diag_embed(sign_mat))

    def norm(self, X, G):
        # Norm on the tangent space is simply the Euclidean norm.
        return torch.norm(G)

    # Generate random Grassmann point using qr of random normally distributed
    # matrix.
    def rand(self):
        if self._k == 1:
            X = torch.randn(self._n, self._p)
            q, r = torch.qr(X)
            return q

        X = torch.zeros((self._k, self._n, self._p))
        for i in range(self._k):
            X[i], r = torch.qr(torch.randn(self._n, self._p))
        return X

    def randvec(self, X):
        U = torch.randn(*torch.shape(X))
        U = self.proj(X, U)
        U = U / torch.norm(U)
        return U

    def transp(self, x1, x2, d):
        return self.proj(x2, d)

    def exp(self, X, U):
        u, s, vt = torch.linalg.svd(U, full_matrices=False)
        cos_s = torch.cos(s).unsqueeze(-2)
        sin_s = torch.sin(s).unsqueeze(-2)

        Y = multiprod(multiprod(X, multitransp(vt) * cos_s), vt) + multiprod(
            u * sin_s, vt
        )

        # From numerical experiments, it seems necessary to
        # re-orthonormalize. This is overall quite expensive.
        if self._k == 1:
            Y, unused = torch.qr(Y)
            return Y
        else:
            for i in range(self._k):
                Y[i], unused = torch.qr(Y[i])
            return Y

    def log(self, X, Y):
        ytx = multiprod(multitransp(Y), X)
        At = multitransp(Y) - multiprod(ytx, multitransp(X))
        Bt = torch.linalg.solve(ytx, At)
        u, s, vt = torch.linalg.svd(multitransp(Bt), full_matrices=False)
        arctan_s = torch.arctan(s).unsqueeze(-2)

        U = multiprod(u * arctan_s, vt)
        return U

    def zerovec(self, X):
        if self._k == 1:
            return torch.zeros((self._n, self._p))
        return torch.zeros((self._k, self._n, self._p))
