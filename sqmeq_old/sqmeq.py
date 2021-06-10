"""
Definition of auxiliary functions and classes in order to have superconductivity in qmeq

"""

import qmeqdev as qmeq
import numpy as np
from scipy import linalg as sla

def construct_Tplus(leads, tleads, Tplus_=None):
    """
        Constructs the many-body tunneling amplitude matrix Tplus from single particle
        tunneling amplitudes.

        Parameters
        ----------
        leads : LeadsTunneling
            LeadsTunneling object.
        tleads : dict
            Dictionary containing single particle tunneling amplitudes.
            tleads[(lead, state)] = tunneling amplitude.
        Tplus_ : None or ndarray
            nbaths by nmany by nmany numpy array containing old values of Tplus.
            The values in tleads are added to Tplus_.

        Returns
        -------
        Tplus : ndarray
            nleads by nmany by nmany numpy array containing many-body tunneling amplitudes.
            The returned Tba corresponds to Fock basis.
        """
    si, mtype = leads.si, leads.mtype
    if Tplus_ is None:
        Tplus = np.zeros((si.nleads, si.nmany, si.nmany), dtype=mtype)
    else:
        Tplus = Tplus_
    # Iterate over many-body states
    for j1 in range(si.nmany):
        state = si.get_state(j1)
        # Iterate over single particle states
        for j0 in tleads:
            (j3, j2), tamp = j0, tleads[j0]
            # Calculate fermion sign for added/removed electron in a given state
            fsign = np.power(-1, sum(state[0:j2]))
            if state[j2] == 0:
                statep = list(state)
                statep[j2] = 1
                ind = si.get_ind(statep)
                if ind is None:
                    continue
                Tplus[j3, ind, j1] += fsign * tamp
    return Tplus

def construct_ham_pairing(qd, pairing):
    ham_pairing = np.zeros((qd.si.nmany, qd.si.nmany), dtype=qd.mtype)
    for j1 in range(qd.si.nmany):
        state = qd.si.get_state(j1)
        for j2 in pairing:
            (m, n), Delta = j2, pairing[j2]
            if m < n and state[m] == 0 and state[n] == 0:
                statep = list(state)
                statep[m] = 1
                statep[n] = 1
                j1p = qd.si.get_ind(statep)
                ham_pairing[j1p, j1] = Delta * (-1)**int(sum(state[0:n])) * (-1)**int(sum(state[0:m]))
                ham_pairing[j1, j1p] = ham_pairing[j1p, j1].conjugate()
    return ham_pairing


class SBuilder(qmeq.Builder):
    def __init__(self,
                 nsingle=0, hsingle={}, coulomb={},
                 nleads=0, tleads={}, mulst={}, tlst={}, dband={},
                 indexing=None, kpnt=None,
                 kerntype='Pauli', symq=True, norm_row=0, solmethod=None,
                 itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                 mtype_qd=complex, mtype_leads=complex,
                 symmetry=None, herm_hs=True, herm_c=False, m_less_n=True, pairing = {}):

        super().__init__(nsingle=nsingle, hsingle=hsingle, coulomb=coulomb,
                 nleads=nleads, tleads=tleads, mulst=mulst, tlst=tlst, dband=dband,
                 indexing=indexing, kpnt=kpnt,
                 kerntype=kerntype, symq=symq, norm_row=norm_row, solmethod=solmethod,
                 itype=itype, dqawc_limit=dqawc_limit, mfreeq=mfreeq, phi0_init=phi0_init,
                 mtype_qd=mtype_qd, mtype_leads=mtype_leads,
                 symmetry=symmetry, herm_hs=herm_hs, herm_c=herm_c, m_less_n=m_less_n)

        self.Tplus0 = construct_Tplus(self.leads, self.leads.tleads)
        self.Tminus0 =  np.zeros((self.nleads, self.si.nmany, self.si.nmany), dtype=self.leads.mtype)
        for lead in range(self.nleads):
            self.Tminus0[lead] = self.Tplus0[lead].conjugate().T

        self.pairing = pairing
        self.ham_pairing = construct_ham_pairing(self.qd, self.pairing)


        self.ham = np.zeros((self.si.nmany, self.si.nmany), dtype=self.qd.mtype)
        counter = 0
        for charge in range(self.si.ncharge):
            self.ham[counter: counter + len(self.qd.hamlst[charge]), counter: counter + len(self.qd.hamlst[charge])] = self.qd.hamlst[charge]
            counter += len(self.qd.hamlst[charge])

        self.ham = self.ham + self.ham_pairing

    def change_ham(self, hsingle=None, coulomb=None, tleads=None, mulst=None, tlst=None, dlst=None, pairing=None):
        #change through Builder_base
        self.change(hsingle=hsingle, coulomb=coulomb, tleads=tleads, mulst=mulst, tlst=tlst, dlst=dlst)
        #change on spot
        self.ham = np.zeros((self.si.nmany, self.si.nmany), dtype=self.qd.mtype)
        counter = 0
        for charge in range(self.si.ncharge):
            self.ham[counter: counter + len(self.qd.hamlst[charge]), counter: counter + len(self.qd.hamlst[charge])] = self.qd.hamlst[charge]
            counter += len(self.qd.hamlst[charge])
        if pairing:
            self.pairing = pairing
            self.ham_pairing = construct_ham_pairing(self.qd, self.pairing)
        self.ham = self.ham + self.ham_pairing

    def diag_ham(self):
        self.mbvals, self.mbvecs = np.linalg.eigh(self.ham)

    def get_mbstates_charges(self):
        self.mbstates_charges = []
        for mbstate in range(self.si.nmany):
            maxind = np.argmax(abs(self.mbvecs[:,mbstate]))
            self.mbstates_charges.append( sum(self.si.get_state(maxind)) )

    def get_mbstates_parities(self):
        self.mbstates_parities = []
        for mbstate in range(self.si.nmany):
            maxind = np.argmax(abs(self.mbvecs[:, mbstate]))
            self.mbstates_parities.append(sum(self.si.get_state(maxind)) % 2)

    def rotate_Tpm(self):
        self.Tplus = np.zeros((self.si.nleads, self.si.nmany, self.si.nmany), dtype=self.leads.mtype)
        self.Tminus = np.zeros((self.si.nleads, self.si.nmany, self.si.nmany), dtype=self.leads.mtype)
        for lead in range(self.si.nleads):
            self.Tplus[lead] = self.mbvecs.conjugate().T @ self.Tplus0[lead] @  self.mbvecs
            self.Tminus[lead] = self.Tplus[lead].conjugate().T





