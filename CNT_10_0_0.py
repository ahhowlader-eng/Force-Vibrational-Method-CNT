import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import time

t0 = time.time()

# -------------------------------
# Parameters
# -------------------------------
n = 999

# -------------------------------
# Percolation lattice
# -------------------------------
m = np.zeros((n+1, n+1), dtype=int)

for i in range(n):
    for j in range(n):
        d = np.random.rand()
        if d <= 1:        # bond probability (same as MATLAB)
            m[i, j] = 1

# Honeycomb removal
m[2:n:3, 0:n:2] = 0
m[0:n:3, 1:n:2] = 0

# -------------------------------
# Geometry (graphene lattice)
# -------------------------------
acc = 1.42
aa = np.sqrt(3) * acc
a = np.sqrt(3) / 2

x, y = np.meshgrid(np.arange(1, n+2), np.arange(1, n+2))
nn = x.shape[0]

x = a * x * acc
y = (y + np.tile([0, 0.5], (nn, nn // 2))) * acc

# Remove deleted atoms
mask = (m == 0)
x[mask] = 0
y[mask] = 0

# -------------------------------
# Plot graphene lattice
# -------------------------------
plt.figure()
plt.plot(x, y, 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)
plt.show()

# -------------------------------
# CNT parameters (n,m)
# -------------------------------
ch1, ch2 = 10, 0
tc = 250

ch = aa * np.sqrt(ch1**2 + ch1*ch2 + ch2**2)
dr = np.gcd(2*ch2 + ch1, 2*ch1 + ch2)
t = tc * (np.sqrt(3) / dr) * ch
r = ch / (2 * np.pi)

# CNT unit cell polygon
x0, y0 = x[0, 0], y[0, 0]
xv = [x0, x0 + ch - 0.005, x0 + ch - 0.005, x0, x0]
yv = [y0, y0, y0 + t - 0.005, y0 + t - 0.005, y0]

poly = Path(np.column_stack((xv, yv)))
points = np.column_stack((x.flatten(), y.flatten()))
inpoly = poly.contains_points(points).reshape(x.shape)

# -------------------------------
# Plot CNT unit cell
# -------------------------------
plt.figure()
plt.plot(xv, yv, 'r-')
plt.plot(x[inpoly], y[inpoly], 'k.', markersize=5)
plt.axis('equal')
plt.axis([0, 150, 0, 150])
plt.grid(True)
plt.show()

# -------------------------------
# Count atoms inside CNT
# -------------------------------
LN = np.sum(inpoly[:n, :n])

# Theoretical atom count
lnn = int(round(2 * (2 * ch * ch) / (aa * aa * dr)))

# -------------------------------
# Extract CNT lattice
# -------------------------------
rows = 3 * tc + 12
cols = lnn - 20

xx = x[:rows, :cols]
yy = y[:rows, :cols]
mi = m[:rows, :cols].copy()

# Boundary removal
mi[:6, :] = 0
mi[-6:, :] = 0

# -------------------------------
# Nearest-neighbor connectivity
# -------------------------------
for i in range(1, rows-1):
    for j in range(1, cols-1):
        if j % 2 == 0 and i % 3 == 1:
            if mi[i, j] == 1 and mi[i, j-1] == 0:
                mi[i, j-1] = 2
            if mi[i, j] == 1 and mi[i, j+1] == 0:
                mi[i, j+1] = 2
            if mi[i, j] == 1 and mi[i-1, j] == 0:
                mi[i-1, j] = 2

for i in range(1, rows-1):
    for j in range(1, cols-1):
        if j % 2 == 1 and i % 3 == 1:
            if mi[i, j] == 1 and mi[i, j-1] == 0:
                mi[i, j-1] = 2
            if mi[i, j] == 1 and mi[i, j+1] == 0:
                mi[i, j+1] = 2
            if mi[i, j] == 1 and mi[i+1, j] == 0:
                mi[i+1, j] = 2

for i in range(1, rows-1):
    for j in range(1, cols-1):
        if j % 2 == 0 and i % 3 == 0:
            if mi[i, j] == 1 and mi[i-1, j-1] == 0:
                mi[i-1, j-1] = 2
            if mi[i, j] == 1 and mi[i-1, j+1] == 0:
                mi[i-1, j+1] = 2
            if mi[i, j] == 1 and mi[i+1, j] == 0:
                mi[i+1, j] = 2

for i in range(1, rows-1):
    for j in range(1, cols-1):
        if j % 2 == 1 and i % 3 == 2:
            if mi[i, j] == 1 and mi[i+1, j-1] == 0:
                mi[i+1, j-1] = 2
            if mi[i, j] == 1 and mi[i+1, j+1] == 0:
                mi[i+1, j+1] = 2
            if mi[i, j] == 1 and mi[i-1, j] == 0:
                mi[i-1, j] = 2

# -------------------------------
# Periodic replication
# -------------------------------
M = np.tile(mi, (1, 3))
M[:, :cols-3] = 0
M[:, 2*cols+3 : 3*cols] = 0

# -------------------------------
# Save results
# -------------------------------
np.savez('n100_per_10000_0_exact.npz', M=M, LN=LN)

print("Elapsed time:", time.time() - t0)
