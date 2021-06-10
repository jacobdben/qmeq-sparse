"""
This is written for a version of qmeq before refactor_approach.
Notice that the functions for generating the Pauli factors and the kernel
are defined outsided of the classe's body and there is no KernelHandler class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import qmeq
import numpy as np
import itertools


def generate_spaulifct(self):
    """
    Make factors used for generating Pauli master equation kernel.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.paulifct : array
        (Modifies) Factors used for generating Pauli master equation kernel.
    """
    (E, Tba, si, mulst, tlst, dlst) = (self.qd.Ea, self.leads.Tba, self.si,
                                       self.leads.mulst, self.leads.tlst, self.leads.dlst)
    itype = self.funcp.itype
    ########################################################################################## changed 2 to 4 in the line below
    paulifct = np.zeros((si.nleads, si.ndm1, 4), dtype=qmeq.mytypes.doublenp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c, b in itertools.product(si.statesdm[ccharge], si.statesdm[bcharge]):
            cb = si.get_ind_dm1(c, b, bcharge)
            Ecb = E[c]-E[b]
            for l in range(si.nleads):
                xcb = ( Tba[l, c, b]*Tba[l, c, b].conjugate() ).real
                rez = qmeq.specfunc.func_pauli(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                paulifct[l, cb, 0] = xcb*rez[0] # Go from b to c adding an electron to the dot
                paulifct[l, cb, 1] = xcb*rez[1] # Go from c to b removing an electron from the dot
                ######################################################################### processes that are possible because of superconducting pairing
                xbc = (Tba[l, b, c]*Tba[l, b, c].conjugate()).real
                rez = qmeq.specfunc.func_pauli(-Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype)
                paulifct[l, cb, 2] = xbc * rez[0] # Go from c to b adding an electron to the dot
                paulifct[l, cb, 3] = xbc * rez[1] # Go from b to c removing an electron from the dot
    self.paulifct = paulifct
    return 0

# ---------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------
def generate_kern_spauli(self):
    """
    Generate Pauli master equation kernel.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.kern : array
        (Modifies) Kernel matrix for Pauli master equation.
    self.bvec : array
        (Modifies) Right hand side column vector for master equation.
        The entry funcp.norm_row is 1 representing normalization condition.
    """
    (paulifct, si) = (self.paulifct, self.si)

    self.kern_ext = np.zeros((si.npauli+1, si.npauli), dtype=qmeq.mytypes.doublenp)
    self.kern = self.kern_ext[0:-1, :]

    qmeq.approach.base.pauli.generate_norm_vec(self, si.npauli)
    kern = self.kern
    for charge in range(si.ncharge):
        for b in si.statesdm[charge]:
            bb = si.get_ind_dm0(b, b, charge)
            bb_bool = si.get_ind_dm0(b, b, charge, 2)
            if bb_bool:
                for a in si.statesdm[charge-1]:
                    aa = si.get_ind_dm0(a, a, charge-1)
                    ba = si.get_ind_dm1(b, a, charge-1)
                    for l in range(si.nleads):
                        ################################################################### the 2nd terms in the next two lines
                        kern[bb, bb] -= paulifct[l, ba, 1] + paulifct[l, ba, 2]
                        kern[bb, aa] += paulifct[l, ba, 0] + paulifct[l, ba, 3]
                for c in si.statesdm[charge+1]:
                    cc = si.get_ind_dm0(c, c, charge+1)
                    cb = si.get_ind_dm1(c, b, charge)
                    for l in range(si.nleads):
                        ################################################################## the 2nd terms in the next two lines
                        kern[bb, bb] -= paulifct[l, cb, 0] + paulifct[l, cb, 3]
                        kern[bb, cc] += paulifct[l, cb, 1] + paulifct[l, cb, 2]
    return 0

def generate_current_spauli(self):
    """
    Calculates currents using Pauli master equation approach.

    Parameters
    ----------
    self : Approach
        Approach object.

    self.current : array
        (Modifies) Values of the current having nleads entries.
    self.energy_current : array
        (Modifies) Values of the energy current having nleads entries.
    self.heat_current : array
        (Modifies) Values of the heat current having nleads entries.
    """
    (phi0, E, paulifct, si) = (self.phi0, self.qd.Ea, self.paulifct, self.si)
    current = np.zeros(si.nleads, dtype=qmeq.mytypes.doublenp)
    energy_current = np.zeros(si.nleads, dtype=qmeq.mytypes.doublenp)
    for charge in range(si.ncharge-1):
        ccharge = charge+1
        bcharge = charge
        for c in si.statesdm[ccharge]:
            cc = si.get_ind_dm0(c, c, ccharge)
            for b in si.statesdm[bcharge]:
                bb = si.get_ind_dm0(b, b, bcharge)
                cb = si.get_ind_dm1(c, b, bcharge)
                for l in range(si.nleads):
                    fct1 = +phi0[bb]*paulifct[l, cb, 0]
                    fct2 = -phi0[cc]*paulifct[l, cb, 1]
                    ############################################################### added fct3, fct4
                    fct3 = -phi0[bb]*paulifct[l, cb, 3]
                    fct4 = +phi0[cc]*paulifct[l, cb, 2]
                    current[l] += fct1 + fct2 + fct3 + fct4
                    ############################################################## added fct3, fct4, not sure about his line
                    energy_current[l] += -(E[b]-E[c])*(fct1 + fct2 + fct3 + fct4)
    self.current = current
    self.energy_current = energy_current
    self.heat_current = energy_current - current*self.leads.mulst
    return 0

class ApproachPySPauli(qmeq.approach.base.pauli.ApproachPyPauli):


    kerntype = 'pySPauli'
    generate_fct = staticmethod(generate_spaulifct)
    generate_kern = staticmethod(generate_kern_spauli)
    generate_current = staticmethod(generate_current_spauli)

