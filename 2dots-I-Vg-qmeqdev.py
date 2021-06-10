import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

import qmeq

# Quantum dot parameters in meV
vgate = 0.

e1, e2 = 0., 0.0001  # onsite

t = 2.  # hopping
U11 = 15.0  # intradot Coulomb
U12 = 1.5  # interdot Coulomb

D11 = 0.
D22 = D11
D12 = D11/2.

# Lead parameters in meV
vbias = 1.
temp = 0.1
dband = 60.0
# Tunneling amplitudes in meV
gamL, gamR = 1., 1.
tL, tR = np.sqrt(gamL / (2 * np.pi)), np.sqrt(gamR / (2 * np.pi))

nsingle = 4  # number of single-particle states

# 0 is up, 1 is down
hsingle = {(0, 0): e1 - vgate, (1, 1): e1 - vgate, (2, 2): e2 - vgate, (3, 3): e2 - vgate,
           (0, 2): -t, (1, 3): -t}

coulomb = {(0, 1, 1, 0): U11, (2, 3, 3, 2): U11,  # intra
           (0, 2, 2, 0): U12, (0, 3, 3, 0): U12, (1, 2, 2, 1): U12, (1, 3, 3, 1): U12}

pairing = {(0, 1): -D11, (2, 3): -D22, (0, 3): -D12, (1,2): D12 }

# The coupling matrix has indices(lead−spin , level)
tleads = {(0, 0): tL, (0, 2): tL, (1, 0): tR, (1, 2): tR,  # spin up fpr L and R
          (2, 1): tL, (2, 3): tL, (3, 1): tR, (3, 3): tR}  # spin down fpr L and R

nleads = 4

#        L,up        R,up         L,down      R,down
mulst = {0: vbias / 2, 1: -vbias / 2, 2: vbias / 2, 3: -vbias / 2}
tlst = {0: temp, 1: temp, 2: temp, 3: temp}

system = qmeq.BuilderSBase(nsingle=nsingle, hsingle=hsingle, coulomb=coulomb, pairing = pairing,
                         nleads=nleads, tleads=tleads, mulst=mulst, tlst=tlst, dband=dband,
                         kerntype="Pauli")

# system.solve()
# print()
# print('Pauli current:')
# print(system.current)
# print(system.energy_current)
# print()
# print('Current continuity:')
# print(np.sum(system.current))


####### plot current vs vgate

vgate_start = -5.
vgate_stop = 20.
vgate_nsteps = 500
vgate_values = np.linspace(vgate_start, vgate_stop, vgate_nsteps)

Itoplot = []

for jj in range(len(vgate_values)):
    system.change(hsingle={(0, 0): e1 - vgate_values[jj], (1, 1): e1 - vgate_values[jj], (2, 2): e2 - vgate_values[jj],
                           (3, 3): e2 - vgate_values[jj],
                           (0, 2): -t, (1, 3): -t})
    system.solve()
    Itoplot.append(system.current[0])

fig = plt.figure(figsize=(8, 6))  # ,     ,dpi=600)#

ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

# change the thickness of all the subplot axes
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(1.)

ax.plot(vgate_values, Itoplot, c="blue")

ax.set_ylabel('$I $', fontsize=20)
ax.set_xlabel(r"$ V_g $", fontsize=20)

ax.set_title('QmeQ-Pauli', fontsize=20)

# ax.set_ylim(-0.01,0.4)

plt.show()