import numpy as np
import time

tic = time.time()

# =========================
# Load data
# =========================
data = np.load('n100_per_1000_0_exact.npz')
M  = data['M']
LN = int(data['LN'])
mi = data['mi']
xx = data['xx']
yy = data['yy']

LX = 52
LY = 79

# =========================
# Force constants
# =========================
Kin   = 245 * 0.983412556
Kra   = 365 * 0.983412556
Kout  = 98.2 * 0.983412556

K2in  = -32.3 * 0.949368617
K2ra  = 88   * 0.949368617
K2out = -4   * 0.949368617

K3in  = -52.5 * 0.931880372
K3ra  = 30    * 0.931880372
K3out = 1.5   * 0.931880372

K4in  = 22.9  * 0.877326506
K4ra  = -19.2 * 0.877326506
K4out = -5.8  * 0.877326506

M1 = 1.993e-26
F0 = 1.0
TOTAL_ATOM = LN

# =========================
# Allocate arrays
# =========================
def z():
    return np.zeros((LY+8, LX+8))

Kin1 = z(); Kin2 = z()
Kin21 = z(); Kin22 = z(); Kin23 = z()
Kin31 = z(); Kin32 = z()
Kin411 = z(); Kin412 = z(); Kin413 = z()
Kin421 = z(); Kin422 = z(); Kin423 = z()
Kin431 = z(); Kin432 = z(); Kin433 = z()
Kin441 = z(); Kin442 = z(); Kin443 = z()

Kra1 = z(); Kra2 = z()
Kra21 = z(); Kra22 = z(); Kra23 = z()
Kra31 = z(); Kra32 = z()
Kra411 = z(); Kra412 = z(); Kra413 = z()
Kra421 = z(); Kra422 = z(); Kra423 = z()
Kra431 = z(); Kra432 = z(); Kra433 = z()
Kra441 = z(); Kra442 = z(); Kra443 = z()

Kout1 = z(); Kout2 = z()
Kout21 = z(); Kout22 = z(); Kout23 = z()
Kout31 = z(); Kout32 = z()
Kout411 = z(); Kout412 = z(); Kout413 = z()
Kout421 = z(); Kout422 = z(); Kout423 = z()
Kout431 = z(); Kout432 = z(); Kout433 = z()
Kout441 = z(); Kout442 = z(); Kout443 = z()

# =========================
# First neighbor (X-direction)
# =========================
for X in range(20, 40):
    for Y in range(1, LY+7):
        if M[Y, X] <= 0:
            continue
        if (M[Y, X-1] > 0 or
            M[Y-1, X-1] > 0 or
            M[Y+1, X-1] > 0):
            Kin2[Y, X] = Kin
            Kra2[Y, X] = Kra
            Kout2[Y, X] = Kout

# Periodic copy
Kin2[:,17:20] = Kin2[:,37:40]
Kin2[:,40:43] = Kin2[:,20:23]
Kra2[:,17:20] = Kra2[:,37:40]
Kra2[:,40:43] = Kra2[:,20:23]
Kout2[:,17:20] = Kout2[:,37:40]
Kout2[:,40:43] = Kout2[:,20:23]

# =========================
# First neighbor (Y-direction)
# =========================
for X in range(20, 40):
    for Y in range(1, LY+8):
        if M[Y, X] <= 0:
            continue
        if M[Y-1, X] > 0:
            Kin1[Y, X] = Kin
            Kra1[Y, X] = Kra
            Kout1[Y, X] = Kout

Kin1[:,17:20] = Kin1[:,37:40]
Kin1[:,40:43] = Kin1[:,20:23]
Kra1[:,17:20] = Kra1[:,37:40]
Kra1[:,40:43] = Kra1[:,20:23]
Kout1[:,17:20] = Kout1[:,37:40]
Kout1[:,40:43] = Kout1[:,20:23]

# =========================
# Time integration parameters
# =========================
TAU = 0.01e-13
SQM = np.sqrt(M1)
TM  = TAU / M1
WS  = 1e12

P1 = 8
WE = 3.25e14
WSTEP = 5e12
DIV = 5e12

W = 2.5431e14
DN = int(round(W / DIV)) + 1
ECST = int(2.0 * np.pi * DN / (W * TAU))

# =========================
# Main loop
# =========================
for P in range(P1 + 1):

    U0X = z(); U0Y = z(); U0Z = z()
    U1X = z(); U1Y = z(); U1Z = z()

    VDX = z(); VDY = z(); VDZ = z()

    FLX = z(); FLY = z(); FLZ = z()

    for X in range(20, 40):
        for Y in range(LY+8):
            FLX[Y,X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
            FLY[Y,X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
            FLZ[Y,X] = F0 * SQM * np.cos(2*np.pi*np.random.rand())

    for N in range(ECST + 1):
        c = np.cos(W * N * TAU)

        for X in range(20, 40):
            for Y in range(3, LY+5):

                if M[Y,X] <= 0:
                    continue

                VDX[Y,X] = FLX[Y,X] * c
                VDY[Y,X] = FLY[Y,X] * c
                VDZ[Y,X] = FLZ[Y,X] * c

                # -------- parity condition --------
                if (X % 2 == 1) and (Y % 3 == 2):
                    VDX[Y,X] += Kin1[Y,X] * (U0X[Y-1,X] - U0X[Y,X])
                    VDY[Y,X] += Kra1[Y,X] * (U0Y[Y-1,X] - U0Y[Y,X])
                    VDZ[Y,X] += Kout1[Y,X] * (U0Z[Y-1,X] - U0Z[Y,X])

p1 = (X % 2 == 1) and (Y % 3 == 2)
p2 = (X % 2 == 0) and (Y % 3 == 1)
p3 = (X % 2 == 1) and (Y % 3 == 0)
p4 = (X % 2 == 0) and (Y % 3 == 2)

for N in range(ECST + 1):
    c = np.cos(W * N * TAU)

    for X in range(20, 40):
        for Y in range(3, LY+5):

            if M[Y, X] <= 0:
                continue

            VDX[Y,X] = FLX[Y,X] * c
            VDY[Y,X] = FLY[Y,X] * c
            VDZ[Y,X] = FLZ[Y,X] * c

            p1 = (X % 2 == 1) and (Y % 3 == 2)
            p2 = (X % 2 == 0) and (Y % 3 == 1)
            p3 = (X % 2 == 1) and (Y % 3 == 0)
            p4 = (X % 2 == 0) and (Y % 3 == 2)

            # ==================================================
            # 1st NEIGHBORS
            # ==================================================
            if p1:
                VDX[Y,X] += Kin1[Y,X]*(U0X[Y-1,X]-U0X[Y,X]) \
                          + Kin2[Y,X]*(U0X[Y,X-1]-U0X[Y,X])
                VDY[Y,X] += Kra1[Y,X]*(U0Y[Y-1,X]-U0Y[Y,X]) \
                          + Kra2[Y,X]*(U0Y[Y,X-1]-U0Y[Y,X])
                VDZ[Y,X] += Kout1[Y,X]*(U0Z[Y-1,X]-U0Z[Y,X]) \
                          + Kout2[Y,X]*(U0Z[Y,X-1]-U0Z[Y,X])

            if p2:
                VDX[Y,X] += Kin1[Y+1,X]*(U0X[Y+1,X]-U0X[Y,X]) \
                          + Kin2[Y,X]*(U0X[Y,X-1]-U0X[Y,X])
                VDY[Y,X] += Kra1[Y+1,X]*(U0Y[Y+1,X]-U0Y[Y,X]) \
                          + Kra2[Y,X]*(U0Y[Y,X-1]-U0Y[Y,X])
                VDZ[Y,X] += Kout1[Y+1,X]*(U0Z[Y+1,X]-U0Z[Y,X]) \
                          + Kout2[Y,X]*(U0Z[Y,X-1]-U0Z[Y,X])

            if p3:
                VDX[Y,X] += Kin1[Y+1,X]*(U0X[Y+1,X]-U0X[Y,X]) \
                          + Kin2[Y,X+1]*(U0X[Y,X+1]-U0X[Y,X])
                VDY[Y,X] += Kra1[Y+1,X]*(U0Y[Y+1,X]-U0Y[Y,X]) \
                          + Kra2[Y,X+1]*(U0Y[Y,X+1]-U0Y[Y,X])
                VDZ[Y,X] += Kout1[Y+1,X]*(U0Z[Y+1,X]-U0Z[Y,X]) \
                          + Kout2[Y,X+1]*(U0Z[Y,X+1]-U0Z[Y,X])

            if p4:
                VDX[Y,X] += Kin1[Y,X]*(U0X[Y-1,X]-U0X[Y,X]) \
                          + Kin2[Y,X+1]*(U0X[Y,X+1]-U0X[Y,X])
                VDY[Y,X] += Kra1[Y,X]*(U0Y[Y-1,X]-U0Y[Y,X]) \
                          + Kra2[Y,X+1]*(U0Y[Y,X+1]-U0Y[Y,X])
                VDZ[Y,X] += Kout1[Y,X]*(U0Z[Y-1,X]-U0Z[Y,X]) \
                          + Kout2[Y,X+1]*(U0Z[Y,X+1]-U0Z[Y,X])

            # ==================================================
            # 2nd NEIGHBORS
            # ==================================================
            if p1 or p3:
                VDX[Y,X] += Kin21[Y,X]*(U0X[Y-1,X-1]-U0X[Y,X]) \
                          + Kin22[Y,X]*(U0X[Y+1,X-1]-U0X[Y,X]) \
                          + Kin23[Y,X]*(U0X[Y+2,X]-U0X[Y,X])

                VDY[Y,X] += Kra21[Y,X]*(U0Y[Y-1,X-1]-U0Y[Y,X]) \
                          + Kra22[Y,X]*(U0Y[Y+1,X-1]-U0Y[Y,X]) \
                          + Kra23[Y,X]*(U0Y[Y+2,X]-U0Y[Y,X])

                VDZ[Y,X] += Kout21[Y,X]*(U0Z[Y-1,X-1]-U0Z[Y,X]) \
                          + Kout22[Y,X]*(U0Z[Y+1,X-1]-U0Z[Y,X]) \
                          + Kout23[Y,X]*(U0Z[Y+2,X]-U0Z[Y,X])

            if p2 or p4:
                VDX[Y,X] += Kin21[Y,X]*(U0X[Y-1,X+1]-U0X[Y,X]) \
                          + Kin22[Y,X]*(U0X[Y+1,X+1]-U0X[Y,X]) \
                          + Kin23[Y,X]*(U0X[Y-2,X]-U0X[Y,X])

                VDY[Y,X] += Kra21[Y,X]*(U0Y[Y-1,X+1]-U0Y[Y,X]) \
                          + Kra22[Y,X]*(U0Y[Y+1,X+1]-U0Y[Y,X]) \
                          + Kra23[Y,X]*(U0Y[Y-2,X]-U0Y[Y,X])

                VDZ[Y,X] += Kout21[Y,X]*(U0Z[Y-1,X+1]-U0Z[Y,X]) \
                          + Kout22[Y,X]*(U0Z[Y+1,X+1]-U0Z[Y,X]) \
                          + Kout23[Y,X]*(U0Z[Y-2,X]-U0Z[Y,X])

            # ==================================================
            # 3rd NEIGHBORS
            # ==================================================
            VDX[Y,X] += Kin31[Y,X]*(U0X[Y,X-2]-U0X[Y,X]) \
                      + Kin32[Y,X]*(U0X[Y,X+2]-U0X[Y,X])
            VDY[Y,X] += Kra31[Y,X]*(U0Y[Y,X-2]-U0Y[Y,X]) \
                      + Kra32[Y,X]*(U0Y[Y,X+2]-U0Y[Y,X])
            VDZ[Y,X] += Kout31[Y,X]*(U0Z[Y,X-2]-U0Z[Y,X]) \
                      + Kout32[Y,X]*(U0Z[Y,X+2]-U0Z[Y,X])

            # ==================================================
            # 4th NEIGHBORS
            # ==================================================
            VDX[Y,X] += (
                Kin411[Y,X]*(U0X[Y-2,X-1]-U0X[Y,X]) +
                Kin412[Y,X]*(U0X[Y-1,X-2]-U0X[Y,X]) +
                Kin413[Y,X]*(U0X[Y+1,X-2]-U0X[Y,X]) +
                Kin421[Y,X]*(U0X[Y+2,X-1]-U0X[Y,X]) +
                Kin422[Y,X]*(U0X[Y+1,X+2]-U0X[Y,X]) +
                Kin423[Y,X]*(U0X[Y-1,X+2]-U0X[Y,X]) +
                Kin431[Y,X]*(U0X[Y-2,X+1]-U0X[Y,X]) +
                Kin432[Y,X]*(U0X[Y-1,X+2]-U0X[Y,X]) +
                Kin433[Y,X]*(U0X[Y+1,X+2]-U0X[Y,X]) +
                Kin441[Y,X]*(U0X[Y+2,X+1]-U0X[Y,X]) +
                Kin442[Y,X]*(U0X[Y+1,X-2]-U0X[Y,X]) +
                Kin443[Y,X]*(U0X[Y-1,X-2]-U0X[Y,X])
            )

            VDY[Y,X] += (
                Kra411[Y,X]*(U0Y[Y-2,X-1]-U0Y[Y,X]) +
                Kra412[Y,X]*(U0Y[Y-1,X-2]-U0Y[Y,X]) +
                Kra413[Y,X]*(U0Y[Y+1,X-2]-U0Y[Y,X]) +
                Kra421[Y,X]*(U0Y[Y+2,X-1]-U0Y[Y,X]) +
                Kra422[Y,X]*(U0Y[Y+1,X+2]-U0Y[Y,X]) +
                Kra423[Y,X]*(U0Y[Y-1,X+2]-U0Y[Y,X]) +
                Kra431[Y,X]*(U0Y[Y-2,X+1]-U0Y[Y,X]) +
                Kra432[Y,X]*(U0Y[Y-1,X+2]-U0Y[Y,X]) +
                Kra433[Y,X]*(U0Y[Y+1,X+2]-U0Y[Y,X]) +
                Kra441[Y,X]*(U0Y[Y+2,X+1]-U0Y[Y,X]) +
                Kra442[Y,X]*(U0Y[Y+1,X-2]-U0Y[Y,X]) +
                Kra443[Y,X]*(U0Y[Y-1,X-2]-U0Y[Y,X])
            )

            VDZ[Y,X] += (
                Kout411[Y,X]*(U0Z[Y-2,X-1]-U0Z[Y,X]) +
                Kout412[Y,X]*(U0Z[Y-1,X-2]-U0Z[Y,X]) +
                Kout413[Y,X]*(U0Z[Y+1,X-2]-U0Z[Y,X]) +
                Kout421[Y,X]*(U0Z[Y+2,X-1]-U0Z[Y,X]) +
                Kout422[Y,X]*(U0Z[Y+1,X+2]-U0Z[Y,X]) +
                Kout423[Y,X]*(U0Z[Y-1,X+2]-U0Z[Y,X]) +
                Kout431[Y,X]*(U0Z[Y-2,X+1]-U0Z[Y,X]) +
                Kout432[Y,X]*(U0Z[Y-1,X+2]-U0Z[Y,X]) +
                Kout433[Y,X]*(U0Z[Y+1,X+2]-U0Z[Y,X]) +
                Kout441[Y,X]*(U0Z[Y+2,X+1]-U0Z[Y,X]) +
                Kout442[Y,X]*(U0Z[Y+1,X-2]-U0Z[Y,X]) +
                Kout443[Y,X]*(U0Z[Y-1,X-2]-U0Z[Y,X])
            )
