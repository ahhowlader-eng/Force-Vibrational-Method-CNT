import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import math
import time

tic = time.time()

# =========================
# Parameters
# =========================
n = 999
m = np.zeros((n+1, n+1))

# =========================
# Percolation network
# =========================
for i in range(n):
    for j in range(n):
        if np.random.rand() <= 0.95:
            m[i, j] = 1

# Lattice deletions
m[2::3, 0::2] = 0
m[0::3, 1::2] = 0

# =========================
# Geometry
# =========================
acc = 1.42
aa = np.sqrt(3) * acc
a  = np.sqrt(3) / 2

x, y = np.meshgrid(np.arange(1, n+2), np.arange(1, n+2))
nn = x.shape[0]

x = (a * x) * acc
y = (y + np.tile([0, 0.5], (nn, nn//2))) * acc

# Remove empty sites
mask0 = (m == 0)
x[mask0] = 0
y[mask0] = 0

# =========================
# Plot lattice
# =========================
plt.figure(1)
plt.plot(x, y, 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)

# =========================
# Chiral vector & cut
# =========================
ch1 = 10
ch2 = 0
tc  = 250

ch = aa * np.sqrt(ch1**2 + ch1*ch2 + ch2**2)
dr = math.gcd((2*ch2 + ch1), (2*ch1 + ch2))
t  = tc * (np.sqrt(3)/dr) * ch
r  = ch / (2*np.pi)

x0, y0 = x[0, 0], y[0, 0]
xv = [x0, x0 + ch - 0.005, x0 + ch - 0.005, x0, x0]
yv = [y0, y0, y0 + t - 0.005, y0 + t - 0.005, y0]

# =========================
# inpolygon equivalent
# =========================
points = np.vstack((x.ravel(), y.ravel())).T
path = Path(np.vstack((xv, yv)).T)
inside = path.contains_points(points)
inside = inside.reshape(x.shape)

# =========================
# Plot cut region
# =========================
plt.figure(2)
plt.plot(xv, yv, 'k-')
plt.plot(x[inside], y[inside], 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)

# =========================
# Count atoms inside
# =========================
LN = np.sum(inside[:n, :n])

# =========================
# Reduced lattice
# =========================
lnn = 2 * (2 * ch * ch) / (aa * aa * dr)
lnn = int(round(lnn))

xx = x[:(3*tc)+12, :lnn-20]
yy = y[:(3*tc)+12, :lnn-20]

mi = m[:(3*tc)+12, :lnn-20]

# Boundary cleanup
mi[:6, :] = 0
mi[(3*tc)+7:(3*tc)+12, :] = 0

# =========================
# Neighbor marking (value = 2)
# =========================
for i in range(1, (3*tc)+11):
    for j in range(1, (lnn-20)-1):
        if (j % 2 == 0) and (i % 3 == 1):
            if mi[i,j] == 1 and mi[i,j-1] == 0: mi[i,j-1] = 2
            if mi[i,j] == 1 and mi[i,j+1] == 0: mi[i,j+1] = 2
            if mi[i,j] == 1 and mi[i-1,j] == 0: mi[i-1,j] = 2

for i in range(1, (3*tc)+11):
    for j in range(1, (lnn-20)-1):
        if (j % 2 == 1) and (i % 3 == 1):
            if mi[i,j] == 1 and mi[i,j-1] == 0: mi[i,j-1] = 2
            if mi[i,j] == 1 and mi[i,j+1] == 0: mi[i,j+1] = 2
            if mi[i,j] == 1 and mi[i+1,j] == 0: mi[i+1,j] = 2

for i in range(1, (3*tc)+11):
    for j in range(1, (lnn-20)-1):
        if (j % 2 == 0) and (i % 3 == 0):
            if mi[i,j] == 1 and mi[i-1,j-1] == 0: mi[i-1,j-1] = 2
            if mi[i,j] == 1 and mi[i-1,j+1] == 0: mi[i-1,j+1] = 2
            if mi[i,j] == 1 and mi[i+1,j] == 0: mi[i+1,j] = 2

for i in range(1, (3*tc)+11):
    for j in range(1, (lnn-20)-1):
        if (j % 2 == 1) and (i % 3 == 2):
            if mi[i,j] == 1 and mi[i+1,j-1] == 0: mi[i+1,j-1] = 2
            if mi[i,j] == 1 and mi[i+1,j+1] == 0: mi[i+1,j+1] = 2
            if mi[i,j] == 1 and mi[i-1,j] == 0: mi[i-1,j] = 2

# =========================
# Replication
# =========================
M = np.tile(mi, (1, 3))
M[:, :(lnn-20)-3] = 0
M[:, (2*(lnn-20))+4:3*(lnn-20)] = 0

# =========================
# Save
# =========================
np.savez(
    'n100_per_10000_5_exact.npz',
    M=M,
    LN=LN,
    mi=mi,
    xx=xx,
    yy=yy
)

print(f"Elapsed time: {time.time()-tic:.2f} s")
