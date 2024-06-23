import numpy as np

rng = np.random.default_rng()


class Value:
    def __init__(self, func=None):
        self.func = func
        self.value = np.array([])

    def compute(self):
        if self.func:
            self.func.compute()

    def diff(self, D):
        if self.func:
            self.func.diff(D)


class Func:
    def __call__(self, inputs):
        self.inputs = inputs
        self.output = Value(self)
        return self.output

    def compute(self):
        pass

    def diff(self, D):
        pass


class LeakyReLU(Func):
    def __init__(self, slope):
        self.slope = slope

    def compute(self):
        x = self.inputs[0]
        x.compute()
        self.output.value = np.where(
            x.value > 0, x.value, x.value * self.slope
        )

    def diff(self, D):
        _D = np.where(self.output.value > 0, 1.0, self.slope).T
        self.inputs[0].diff(D * _D)


class Linear(Func):
    def __init__(self, m, n):
        self.W = rng.random((m, n))
        self.b = rng.random((m, 1))

    def compute(self):
        x = self.inputs[0]
        x.compute()
        self.output.value = self.W @ x.value + self.b

    def diff(self, D):
        self.inputs[0].diff(D @ self.W)
        n = self.inputs[0].value.size
        _D = self.inputs[0].value.reshape((1, n))
        self.W -= D.T @ _D
        self.b -= D.T
