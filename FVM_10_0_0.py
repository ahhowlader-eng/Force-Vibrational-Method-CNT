# Load data
import numpy as np
from scipy.io import loadmat

data = loadmat('n100_per_10000_0_exact.mat')
M  = data['M']
LN = int(data['LN'][0][0])

LY = 754
LX = 52

# Constants
Kin  = 245 * 0.983412556
Kra  = 365 * 0.983412556
Kout = 98.2 * 0.983412556

K2in  = -32.3 * 0.949368617
K2ra  = 88   * 0.949368617
K2out = -4   * 0.949368617

K3in  = -52.5 * 0.931880372
K3ra  = 30    * 0.931880372
K3out = 1.5   * 0.931880372

K4in  = 22.9 * 0.877326506
K4ra  = -19.2 * 0.877326506
K4out = -5.8 * 0.877326506

M1 = 1.993e-26

# Allocate arrays
Kin2 = np.zeros((LY+8, LX+8))
Kra2 = np.zeros_like(Kin2)
Kout2 = np.zeros_like(Kin2)

# nearest-neighbor spring assignment
for X in range(20,40):
    for Y in range(1, LY+7):
        if M[Y, X] <= 0:
            Kin2[Y, X] = 0.0
            Kra2[Y, X] = 0.0
            Kout2[Y, X] = 0.0
        else:
            if (M[Y, X-1] > 0 or
                M[Y-1, X-1] > 0 or
                M[Y+1, X-1] > 0):
                Kin2[Y, X]  = Kin
                Kra2[Y, X]  = Kra
                Kout2[Y, X] = Kout

# Periodic boundary
Kin2[:, 17:20] = Kin2[:, 37:40]

# Random force initialization
SQM = np.sqrt(M1)
F0 = 1.0

FLX = np.zeros((LY+8, LX+8))
FLY = np.zeros_like(FLX)
FLZ = np.zeros_like(FLX)

for X in range(20,40):
    for Y in range(LY+8):
        FLX[Y,X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
        FLY[Y,X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
        FLZ[Y,X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())

# Time integration loop
TAU = 0.01e-13
WS = 1e12
WE = 3.25e14
WSTEP = 1e12
DIV = 5e12

for W in np.arange(WS, WE+WSTEP, WSTEP):
    DN = int(round(W/DIV)) + 1
    ECST = int(2*np.pi*DN / (W*TAU))

    for N in range(ECST):
        t = W * N * TAU

        for X in range(20,40):
            for Y in range(3, LY+5):

                if M[Y,X] <= 0:
                    continue

                VDX = FLX[Y,X] * np.cos(t)
                VDY = FLY[Y,X] * np.cos(t)
                VDZ = FLZ[Y,X] * np.cos(t)

                # ---- your force expressions go here ----

# Context
# Arrays (float64)
UX, UY, UZ = ...     # displacements
VX, VY, VZ = ...     # velocities
M = ...              # lattice mask

Kin2, Kra2, Kout2 = ...
K2in2, K2ra2, K2out2 = ...
K3in2, K3ra2, K3out2 = ...
K4in2, K4ra2, K4out2 = ...

M1 = 1.993e-26

# Index ranges
X=20:40 , Y=3:LY+5

# External driving term
VDX = FLX[Y, X] * np.cos(t)

# Nearest-neighbor (1st NN) force — X direction
# left
VDX -= (Kin2[Y, X] / M1) * (UX[Y, X] - UX[Y, X-1])
VDX -= (Kra2[Y, X] / M1) * (UX[Y, X] - UX[Y-1, X-1])
VDX -= (Kra2[Y, X] / M1) * (UX[Y, X] - UX[Y+1, X-1])

# right
VDX -= (Kin2[Y, X+1] / M1) * (UX[Y, X] - UX[Y, X+1])
VDX -= (Kra2[Y+1, X+1] / M1) * (UX[Y, X] - UX[Y+1, X+1])
VDX -= (Kra2[Y-1, X+1] / M1) * (UX[Y, X] - UX[Y-1, X+1])

# Second-nearest neighbors (2nd NN)
VDX -= (K2in2[Y, X] / M1) * (UX[Y, X] - UX[Y, X-2])
VDX -= (K2in2[Y, X+2] / M1) * (UX[Y, X] - UX[Y, X+2])

VDX -= (K2ra2[Y, X] / M1) * (UX[Y, X] - UX[Y-2, X])
VDX -= (K2ra2[Y+2, X] / M1) * (UX[Y, X] - UX[Y+2, X])

# Third-nearest neighbors (3rd NN)
VDX -= (K3in2[Y, X] / M1) * (UX[Y, X] - UX[Y, X-3])
VDX -= (K3in2[Y, X+3] / M1) * (UX[Y, X] - UX[Y, X+3])

VDX -= (K3ra2[Y, X] / M1) * (UX[Y, X] - UX[Y-3, X])
VDX -= (K3ra2[Y+3, X] / M1) * (UX[Y, X] - UX[Y+3, X])

# Fourth-nearest neighbors (4th NN)
VDX -= (K4in2[Y, X] / M1) * (UX[Y, X] - UX[Y, X-4])
VDX -= (K4in2[Y, X+4] / M1) * (UX[Y, X] - UX[Y, X+4])

VDX -= (K4ra2[Y, X] / M1) * (UX[Y, X] - UX[Y-4, X])
VDX -= (K4ra2[Y+4, X] / M1) * (UX[Y, X] - UX[Y+4, X])

# Velocity update
VX[Y, X] += VDX * TAU
UX[Y, X] += VX[Y, X] * TAU

# Y-DIRECTION FORCE BLOCK (VDY)

# External driving
VDY = FLY[Y, X] * np.cos(t)

# Nearest-neighbor (1st NN)
# left
VDY -= (Kra2[Y, X] / M1) * (UY[Y, X] - UY[Y-1, X-1])
VDY -= (Kin2[Y, X] / M1) * (UY[Y, X] - UY[Y, X-1])
VDY -= (Kra2[Y, X] / M1) * (UY[Y, X] - UY[Y+1, X-1])

# right
VDY -= (Kra2[Y+1, X+1] / M1) * (UY[Y, X] - UY[Y+1, X+1])
VDY -= (Kin2[Y, X+1] / M1) * (UY[Y, X] - UY[Y, X+1])
VDY -= (Kra2[Y-1, X+1] / M1) * (UY[Y, X] - UY[Y-1, X+1])

# Second-nearest neighbors (2nd NN)
VDY -= (K2ra2[Y, X] / M1) * (UY[Y, X] - UY[Y-2, X])
VDY -= (K2ra2[Y+2, X] / M1) * (UY[Y, X] - UY[Y+2, X])

VDY -= (K2in2[Y, X] / M1) * (UY[Y, X] - UY[Y, X-2])
VDY -= (K2in2[Y, X+2] / M1) * (UY[Y, X] - UY[Y, X+2])

# Third-nearest neighbors (3rd NN)
VDY -= (K3ra2[Y, X] / M1) * (UY[Y, X] - UY[Y-3, X])
VDY -= (K3ra2[Y+3, X] / M1) * (UY[Y, X] - UY[Y+3, X])

VDY -= (K3in2[Y, X] / M1) * (UY[Y, X] - UY[Y, X-3])
VDY -= (K3in2[Y, X+3] / M1) * (UY[Y, X] - UY[Y, X+3])

# Fourth-nearest neighbors (4th NN)
VDY -= (K4ra2[Y, X] / M1) * (UY[Y, X] - UY[Y-4, X])
VDY -= (K4ra2[Y+4, X] / M1) * (UY[Y, X] - UY[Y+4, X])

VDY -= (K4in2[Y, X] / M1) * (UY[Y, X] - UY[Y, X-4])
VDY -= (K4in2[Y, X+4] / M1) * (UY[Y, X] - UY[Y, X+4])

# Velocity & displacement update
VY[Y, X] += VDY * TAU
UY[Y, X] += VY[Y, X] * TAU

# Z-DIRECTION FORCE BLOCK (VDZ)

# External driving
VDZ = FLZ[Y, X] * np.cos(t)

# Nearest neighbors (1st NN)
# # left
VDZ -= (Kout2[Y, X] / M1) * (UZ[Y, X] - UZ[Y, X-1])
VDZ -= (Kout2[Y, X] / M1) * (UZ[Y, X] - UZ[Y-1, X-1])
VDZ -= (Kout2[Y, X] / M1) * (UZ[Y, X] - UZ[Y+1, X-1])

# right
VDZ -= (Kout2[Y, X+1] / M1) * (UZ[Y, X] - UZ[Y, X+1])
VDZ -= (Kout2[Y+1, X+1] / M1) * (UZ[Y, X] - UZ[Y+1, X+1])
VDZ -= (Kout2[Y-1, X+1] / M1) * (UZ[Y, X] - UZ[Y-1, X+1])

# Second-nearest neighbors (2nd NN)
VDZ -= (K2out2[Y, X] / M1) * (UZ[Y, X] - UZ[Y, X-2])
VDZ -= (K2out2[Y, X+2] / M1) * (UZ[Y, X] - UZ[Y, X+2])

VDZ -= (K2out2[Y, X] / M1) * (UZ[Y, X] - UZ[Y-2, X])
VDZ -= (K2out2[Y+2, X] / M1) * (UZ[Y, X] - UZ[Y+2, X])

# Third-nearest neighbors (3rd NN)
VDZ -= (K3out2[Y, X] / M1) * (UZ[Y, X] - UZ[Y, X-3])
VDZ -= (K3out2[Y, X+3] / M1) * (UZ[Y, X] - UZ[Y, X+3])

VDZ -= (K3out2[Y, X] / M1) * (UZ[Y, X] - UZ[Y-3, X])
VDZ -= (K3out2[Y+3, X] / M1) * (UZ[Y, X] - UZ[Y+3, X])

# Fourth-nearest neighbors (4th NN)
VDZ -= (K4out2[Y, X] / M1) * (UZ[Y, X] - UZ[Y, X-4])
VDZ -= (K4out2[Y, X+4] / M1) * (UZ[Y, X] - UZ[Y, X+4])

VDZ -= (K4out2[Y, X] / M1) * (UZ[Y, X] - UZ[Y-4, X])
VDZ -= (K4out2[Y+4, X] / M1) * (UZ[Y, X] - UZ[Y+4, X])

# Velocity & displacement update
VZ[Y, X] += VDZ * TAU
UZ[Y, X] += VZ[Y, X] * TAU

