from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import qmeq

# Quantum dot parameters in meV
vgate = 0.

e1, e2 = 0., 0.0001  # onsite

t = 2.  # hopping
U11 = 15.0  # intradot Coulomb
U12 = 1.5  # interdot Coulomb
D11 = 1.
D22 = D11
D12 = D11 / 2.

# Lead parameters in meV
vbias = 0.
temp = 0.15
dband = 60.0
# Tunneling amplitudes in meV
gamL, gamR = 0.5, 0.5
tL, tR = np.sqrt(gamL / (2 * np.pi)), np.sqrt(gamR / (2 * np.pi))

nsingle = 4  # number of single-particle states

# 0 is up, 1 is down
hsingle = {(0, 0): e1 - vgate, (1, 1): e1 - vgate, (2, 2): e2 - vgate, (3, 3): e2 - vgate,
           (0, 2): -t, (1, 3): -t}

coulomb = {(0, 1, 1, 0): U11, (2, 3, 3, 2): U11,  # intra
           (0, 2, 2, 0): U12, (0, 3, 3, 0): U12, (1, 2, 2, 1): U12, (1, 3, 3, 1): U12}

pairing = {(0, 1): -D11, (2, 3): -D22, (0, 3): -D12, (1, 2): D12}

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

def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]
    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    #
    for j1 in range(vgpnt):
        system.change(hsingle = {(0, 0): e1-vglst[j1], (1, 1): e1-vglst[j1], (2, 2): e2-vglst[j1], (3, 3): e2-vglst[j1],
                              (0, 2): -t,    (1,3): -t })
        system.solve(masterq=False)
        for j2 in range(vpnt):
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2,
                                 2: vlst[j2]/2, 3: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = (system.current[0]
                          + system.current[2])
            #
            system.add(mulst={0: dV/2, 1: -dV/2,
                              2: dV/2, 3: -dV/2})
            system.solve(qdq=False)
            stab_cond[j1, j2] = (system.current[0]
                               + system.current[2]
                               - stab[j1, j2])/dV
    #
    return stab, stab_cond

def stab_plot(stab_cond, vlst, vglst, U, gam, title, fname='fig.pdf'):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1],
                                         vlst[0], vlst[-1]])/U
    fig = plt.figure(figsize=(6,4.2))
    p = plt.subplot(1, 1, 1)
    p.set_xlabel('$V_{g}/U$', fontsize=20)
    p.set_ylabel('$V/U$', fontsize=20)
    p.set_title(title, fontsize=20)
    p_im = plt.imshow(stab_cond.T, extent=[xmin, xmax, ymin, ymax],
                                   aspect='auto',
                                   origin='lower',
                                   cmap=plt.get_cmap('Spectral'))
    cbar = plt.colorbar(p_im)
    cbar.set_label('Conductance $\mathrm{d}I/\mathrm{d}V$', fontsize=20)
    fig.savefig(fname, bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.show()

vpnt, vgpnt = 201, 201
vlst = np.linspace(-2*U11, 2*U11, vpnt)
vglst = np.linspace(-1.*U11, 1.5*U11, vgpnt)
stab, stab_cond = stab_calc(system, 0., vlst, vglst)
stab_plot(stab_cond, vlst, vglst, U11, gamL, 'Pauli, $\Delta=0$', 'stab.pdf')