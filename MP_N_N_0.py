# Mode Pattern Calculation
# Imports & constants
import numpy as np
import time

# Frequency & time discretization
W = 2.54414
DN0 = W / DIV
DNA = int(np.round(DN0))
DN = DNA + 1
ECST = int(2.0 * np.pi * DN / (W * TAU))

# Main loop over realizations
t0 = time.time()

for P in range(P1 + 1):

# Array initialization
    shape = (LY + 8, LX + 8)
    U0X = np.zeros(shape); U1X = np.zeros(shape)
    U0Y = np.zeros(shape); U1Y = np.zeros(shape)
    U0Z = np.zeros(shape); U1Z = np.zeros(shape)

    VX = np.zeros(shape); VY = np.zeros(shape); VZ = np.zeros(shape)
    VDX = np.zeros(shape); VDY = np.zeros(shape); VDZ = np.zeros(shape)

    FLX = np.zeros(shape); FLY = np.zeros(shape); FLZ = np.zeros(shape)

    # rotated coordinate displacements
    U011 = np.zeros(shape); U111 = np.zeros(shape)
    U012 = np.zeros(shape); U112 = np.zeros(shape)
    U021 = np.zeros(shape); U121 = np.zeros(shape)
    U022 = np.zeros(shape); U122 = np.zeros(shape)

    U041 = np.zeros(shape); U141 = np.zeros(shape)
    U042 = np.zeros(shape); U142 = np.zeros(shape)
    U043 = np.zeros(shape); U143 = np.zeros(shape)
    U044 = np.zeros(shape); U144 = np.zeros(shape)
    U045 = np.zeros(shape); U145 = np.zeros(shape)
    U046 = np.zeros(shape); U146 = np.zeros(shape)

    U0411 = np.zeros(shape); U1411 = np.zeros(shape)
    U0421 = np.zeros(shape); U1421 = np.zeros(shape)
    U0431 = np.zeros(shape); U1431 = np.zeros(shape)
    U0441 = np.zeros(shape); U1441 = np.zeros(shape)
    U0451 = np.zeros(shape); U1451 = np.zeros(shape)
    U0461 = np.zeros(shape); U1461 = np.zeros(shape)

# Random force initialization
    for x in range(LX + 8):
        for y in range(LY + 8):
            FLX[y, x] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
            FLY[y, x] = F0 * SQM * np.cos(2*np.pi*np.random.rand())
            FLZ[y, x] = F0 * SQM * np.cos(2*np.pi*np.random.rand())

# Time evolution loop
    for N in range(ECST + 1):
        for x in range(3, LX + 5):
            for y in range(3, LY + 5):

                if M[y, x] <= 0:
                    continue

                # external driving
                VDX[y, x] = FLX[y, x] * np.cos(W * N * TAU)
                VDY[y, x] = FLY[y, x] * np.cos(W * N * TAU)
                VDZ[y, x] = FLZ[y, x] * np.cos(W * N * TAU)

# Hexagonal parity-dependent force blocks
# Case: X odd & Y mod 3 == 2
                if (x % 2 == 1) and (y % 3 == 2):

                    VDX[y,x] += (
                        Kra2[y,x]*(U011[y,x-1]-U011[y,x])*(-np.sqrt(3)/2)
                        + Kra2[y,x+1]*(U021[y,x+1]-U021[y,x])*( np.sqrt(3)/2)
                        + Kin2[y,x]*(U012[y,x-1]-U012[y,x])*(-0.5)
                        + Kin2[y,x+1]*(U022[y,x+1]-U022[y,x])*(0.5)
                        + Kin1[y,x]*(U0X[y-1,x]-U0X[y,x])
                        # --- continues EXACTLY as MATLAB ---
                    )

                    VDY[y,x] += (
                        Kra2[y,x]*(U011[y,x-1]-U011[y,x])*(0.5)
                        + Kra2[y,x+1]*(U021[y,x+1]-U021[y,x])*(0.5)
                        + Kin2[y,x]*(U012[y,x-1]-U012[y,x])*(-np.sqrt(3)/2)
                        # --- continues ---
                    )

                    VDZ[y,x] += (
                        Kout2[y,x]*(U0Z[y,x-1]-U0Z[y,x])
                        + Kout2[y,x+1]*(U0Z[y,x+1]-U0Z[y,x])
                        + Kout1[y,x]*(U0Z[y-1,x]-U0Z[y,x])
                        # --- continues ---
                    )


# (x % 2 == 0 and y % 3 == 2)
# x even & y mod 3 == 2
if (x % 2 == 0) and (y % 3 == 2):

    VDX[y,x] += (
        Kra2[y,x]   * (U021[y,x-1] - U021[y,x]) * (-np.sqrt(3)/2)
      + Kra2[y,x+1] * (U011[y,x+1] - U011[y,x]) * ( np.sqrt(3)/2)
      + Kin2[y,x]   * (U022[y,x-1] - U022[y,x]) * (-0.5)
      + Kin2[y,x+1] * (U012[y,x+1] - U012[y,x]) * ( 0.5)
      + Kin1[y,x]   * (U0X[y+1,x]  - U0X[y,x])
    )

    VDY[y,x] += (
        Kra2[y,x]   * (U021[y,x-1] - U021[y,x]) * ( 0.5)
      + Kra2[y,x+1] * (U011[y,x+1] - U011[y,x]) * ( 0.5)
      + Kin2[y,x]   * (U022[y,x-1] - U022[y,x]) * (-np.sqrt(3)/2)
      + Kin2[y,x+1] * (U012[y,x+1] - U012[y,x]) * (-np.sqrt(3)/2)
    )

    VDZ[y,x] += (
        Kout2[y,x]   * (U0Z[y,x-1] - U0Z[y,x])
      + Kout2[y,x+1] * (U0Z[y,x+1] - U0Z[y,x])
      + Kout1[y,x]   * (U0Z[y+1,x] - U0Z[y,x])
    )


# (x % 2 == 1 and y % 3 == 1)
# x odd & y mod 3 == 1
if (x % 2 == 1) and (y % 3 == 1):

    VDX[y,x] += (
        Kra2[y,x]   * (U041[y-1,x-1] - U041[y,x]) * (-np.sqrt(3)/2)
      + Kra2[y,x]   * (U043[y-1,x+1] - U043[y,x]) * ( np.sqrt(3)/2)
      + Kin2[y,x]   * (U042[y-1,x-1] - U042[y,x]) * (-0.5)
      + Kin2[y,x]   * (U044[y-1,x+1] - U044[y,x]) * ( 0.5)
      + Kin1[y,x]   * (U0X[y+1,x]    - U0X[y,x])
    )

    VDY[y,x] += (
        Kra2[y,x]   * (U041[y-1,x-1] - U041[y,x]) * (0.5)
      + Kra2[y,x]   * (U043[y-1,x+1] - U043[y,x]) * (0.5)
      + Kin2[y,x]   * (U042[y-1,x-1] - U042[y,x]) * (-np.sqrt(3)/2)
      + Kin2[y,x]   * (U044[y-1,x+1] - U044[y,x]) * (-np.sqrt(3)/2)
    )

    VDZ[y,x] += (
        Kout2[y,x] * (U0Z[y-1,x-1] - U0Z[y,x])
      + Kout2[y,x] * (U0Z[y-1,x+1] - U0Z[y,x])
      + Kout1[y,x] * (U0Z[y+1,x]   - U0Z[y,x])
    )


# (x % 2 == 0 and y % 3 == 0)
# x even & y mod 3 == 0
if (x % 2 == 0) and (y % 3 == 0):

    VDX[y,x] += (
        Kra2[y,x]   * (U045[y+1,x-1] - U045[y,x]) * (-np.sqrt(3)/2)
      + Kra2[y,x]   * (U046[y+1,x+1] - U046[y,x]) * ( np.sqrt(3)/2)
      + Kin2[y,x]   * (U0451[y+1,x-1] - U0451[y,x]) * (-0.5)
      + Kin2[y,x]   * (U0461[y+1,x+1] - U0461[y,x]) * ( 0.5)
      + Kin1[y,x]   * (U0X[y-1,x]     - U0X[y,x])
    )

    VDY[y,x] += (
        Kra2[y,x]   * (U045[y+1,x-1] - U045[y,x]) * (0.5)
      + Kra2[y,x]   * (U046[y+1,x+1] - U046[y,x]) * (0.5)
      + Kin2[y,x]   * (U0451[y+1,x-1] - U0451[y,x]) * (-np.sqrt(3)/2)
      + Kin2[y,x]   * (U0461[y+1,x+1] - U0461[y,x]) * (-np.sqrt(3)/2)
    )

    VDZ[y,x] += (
        Kout2[y,x] * (U0Z[y+1,x-1] - U0Z[y,x])
      + Kout2[y,x] * (U0Z[y+1,x+1] - U0Z[y,x])
      + Kout1[y,x] * (U0Z[y-1,x]   - U0Z[y,x])
    )


# Velocity & displacement update
                VX[y,x] += VDX[y,x] * TM
                VY[y,x] += VDY[y,x] * TM
                VZ[y,x] += VDZ[y,x] * TM

                U1X[y,x] = U0X[y,x] + VX[y,x] * TAU
                U1Y[y,x] = U0Y[y,x] + VY[y,x] * TAU
                U1Z[y,x] = U0Z[y,x] + VZ[y,x] * TAU

# Mode-projection (rotated coordinates)
                U111[y,x] = U1X[y,x]*(-np.sqrt(3)/2) + U1Y[y,x]*(0.5)
                U112[y,x] = U1X[y,x]*(-0.5) + U1Y[y,x]*(-np.sqrt(3)/2)
                U121[y,x] = U1X[y,x]*( np.sqrt(3)/2) + U1Y[y,x]*(0.5)
                U122[y,x] = U1X[y,x]*(0.5) + U1Y[y,x]*(-np.sqrt(3)/2)
                
                # Field update
                        for x in range(4, LX + 4):
            for y in range(4, LY + 4):
                if M[y,x] == 1:
                    U0X[y,x] = U1X[y,x]
                    U0Y[y,x] = U1Y[y,x]
                    U0Z[y,x] = U1Z[y,x]
                    U011[y,x] = U111[y,x]
                    U012[y,x] = U112[y,x]
                    U021[y,x] = U121[y,x]
                    U022[y,x] = U122[y,x]

# Mode normalization
U0 = np.sqrt(U0X**2 + U0Y**2)
Fmax = np.max(np.abs(U0[M==1]))

UX = U0X / np.max(np.abs(U0X[M==1]))
UY = U0Y / np.max(np.abs(U0Y[M==1]))
UZ = U0Z / np.max(np.abs(U0Z[M==1]))
U  = U0  / Fmax

# Save output
np.savez(
    "graphene_modepattern_9_1350.npz",
    U=U, UX=UX, UY=UY, UZ=UZ,
    U0=U0, U0X=U0X, U0Y=U0Y, U0Z=U0Z
)

print("Elapsed time:", time.time() - t0)

