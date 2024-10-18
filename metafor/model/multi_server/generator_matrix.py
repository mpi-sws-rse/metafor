from scipy.sparse.linalg import LinearOperator


class GeneratorMatrix(LinearOperator):
    def __init__(self, shape, matvec, rmatvec=None, dtype=None):
        super().__init__(dtype=dtype, shape=shape)
        self._matvec_func = matvec
        self._rmatvec_func = rmatvec

    # Matrix-vector multiplication
    def _matvec(self, x):
        return self._matvec_func(x)

    # Adjoint matrix-vector multiplication (optional)
    def _rmatvec(self, x):
        if self._rmatvec_func is not None:
            return self._rmatvec_func(x)
        else:
            raise NotImplementedError("rmatvec is not implemented")
