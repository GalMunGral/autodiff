import numpy as np
from tqdm import tqdm
import random

from autodiff import Value, Linear, LeakyReLU
from plot import plot


X = Value()
L1 = Linear(10, 2)
R1 = LeakyReLU(0.01)
L2 = Linear(50, 10)
R2 = LeakyReLU(0.01)
L3 = Linear(10, 50)
R3 = LeakyReLU(0.01)
L4 = Linear(1, 10)
Y = L4([R3([L3([R2([L2([R1([L1([X])])])])])])])


def f_target(x, y):
    return np.arcsin(x * y) + np.sin(0.2**x + 5 * y)


def f_approx(x, y):
    X.value = np.array([x, y]).reshape(2, 1)
    Y.compute()
    return Y.value[0][0]


samples = []
for i in range(1000):
    x0 = random.uniform(-1, 1)
    x1 = random.uniform(-1, 1)
    samples.append((x0, x1, f_target(x0, x1)))


rate = 0.0001
steps = 10000000

for i in tqdm(range(steps)):
    x0, x1, y = random.choice(samples)
    d = f_approx(x0, x1) - y
    Y.diff(np.array([[(d / np.abs(d)) * rate]]))

plot(f_target, f_approx, lower=-1, upper=1, levels=20)
