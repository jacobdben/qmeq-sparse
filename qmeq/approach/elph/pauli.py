"""Module containing python functions, which generate first order Pauli kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp

from ...specfunc.specfunc_elph import FuncPauliElPh

from ..aprclass import ApproachElPh
from ..base.pauli import ApproachPauli as ApproachPauliBase


# ---------------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------------
class ApproachPauli(ApproachElPh):

    kerntype = 'pyPauli'

    def get_kern_size(self):
        return self.si.npauli

    def generate_fct(self):
        ApproachPauliBase.generate_fct(self)

        E, Vbbp = self.qd.Ea, self.baths.Vbbp
        si, kh = self.si_elph, self.kernel_handler
        ndm0, ncharge, nbaths, statesdm = si.ndm0, si.ncharge, si.nbaths, si.statesdm

        func_pauli = FuncPauliElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                                   self.baths.bath_func, self.funcp.eps_elph)

        paulifct = np.zeros((nbaths, ndm0), dtype=doublenp)
        for charge in range(ncharge):
            # The diagonal elements b=bp are excluded, because they do not contribute
            for b, bp in itertools.permutations(statesdm[charge], 2):
                bbp_bool = si.get_ind_dm0(b, bp, charge, maptype=2)
                if not bbp_bool:
                    continue
                bbp = si.get_ind_dm0(b, bp, charge)
                Ebbp = E[b]-E[bp]
                for l in range(nbaths):
                    xbbp = 0.5*(Vbbp[l, b, bp]*Vbbp[l, b, bp].conjugate() +
                                Vbbp[l, bp, b].conjugate()*Vbbp[l, bp, b]).real
                    func_pauli.eval(Ebbp, l)
                    paulifct[l, bbp] = xbbp*func_pauli.val

        self.paulifct_elph = paulifct

    def generate_kern(self):
        ApproachPauliBase.generate_kern(self)

    def generate_coupling_terms(self, b, bp, bcharge):
        ApproachPauliBase.generate_coupling_terms(self, b, bp, bcharge)

        paulifct = self.paulifct_elph
        si, si_elph, kh = self.si, self.si_elph, self.kernel_handler
        nbaths, statesdm = si.nbaths, si.statesdm

        acharge = bcharge

        bb = si.get_ind_dm0(b, b, bcharge)
        for a in statesdm[acharge]:
            aa = si.get_ind_dm0(a, a, acharge)
            ab = si_elph.get_ind_dm0(a, b, bcharge)
            ba = si_elph.get_ind_dm0(b, a, acharge)
            if aa == -1 or ba == -1:
                continue

            fctm, fctp = 0, 0
            for l in range(nbaths):
                fctm -= paulifct[l, ab]
                fctp += paulifct[l, ba]
            kh.set_matrix_element_pauli(fctm, fctp, bb, aa)

    def generate_current(self):
        ApproachPauliBase.generate_current(self)
