"""Module containing python functions, which generate first order 1vN kernel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import complexnp
from ...mytypes import doublenp

from ...specfunc.specfunc import func_1vN
from ..aprclass import Approach


# ---------------------------------------------------------------------------------------------------
# 1 von Neumann approach
# ---------------------------------------------------------------------------------------------------
class Approach1vN(Approach):

    kerntype = 'py1vN'

    def generate_fct(self):
        """
        Make factors used for generating 1vN, Redfield master equation kernels.

        Parameters
        ----------
        self.phi1fct : array
            (Modifies) Factors used for generating 1vN, Redfield master equation kernels.
        self.phi1fct_energy : array
            (Modifies) Factors used to calculate energy and heat currents in 1vN, Redfield approaches.
        """
        E, si = self.qd.Ea, self.si,
        mulst, tlst, dlst = self.leads.mulst, self.leads.tlst, self.leads.dlst
        itype, limit = self.funcp.itype, self.funcp.dqawc_limit
        ndm1, ncharge, nleads, statesdm = si.ndm1, si.ncharge, si.nleads, si.statesdm

        phi1fct = np.zeros((nleads, ndm1, 2), dtype=complexnp)
        phi1fct_energy = np.zeros((nleads, ndm1, 2), dtype=complexnp)
        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)
                Ecb = E[c]-E[b]
                for l in range(nleads):
                    rez = func_1vN(Ecb, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, limit)
                    phi1fct[l, cb, 0] = rez[0]
                    phi1fct[l, cb, 1] = rez[1]
                    phi1fct_energy[l, cb, 0] = rez[2]
                    phi1fct_energy[l, cb, 1] = rez[3]
        self.phi1fct = phi1fct
        self.phi1fct_energy = phi1fct_energy

    def generate_coupling_terms(self, b, bp, bcharge):
        Tba, phi1fct = self.leads.Tba, self.phi1fct
        si, kh = self.si, self.kernel_handler
        nleads, statesdm = si.nleads, si.statesdm

        acharge = bcharge-1
        ccharge = bcharge+1

        # --------------------------------------------------
        for a, ap in itertools.product(statesdm[acharge], statesdm[acharge]):
            if kh.is_included(a, ap, acharge):
                bpa = si.get_ind_dm1(bp, a, acharge)
                bap = si.get_ind_dm1(b, ap, acharge)
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += (+ Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bpa, 0].conjugate()
                                - Tba[l, b, a]*Tba[l, ap, bp]*phi1fct[l, bap, 0])
                kh.set_matrix_element(fct_aap, b, bp, bcharge, a, ap, acharge)
        # --------------------------------------------------
        for bpp in statesdm[bcharge]:
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for a in statesdm[acharge]:
                    bpa = si.get_ind_dm1(bp, a, acharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, a]*Tba[l, a, bpp]*phi1fct[l, bpa, 1].conjugate()
                for c in statesdm[ccharge]:
                    cbp = si.get_ind_dm1(c, bp, bcharge)
                    for l in range(nleads):
                        fct_bppbp += +Tba[l, b, c]*Tba[l, c, bpp]*phi1fct[l, cbp, 0]
                kh.set_matrix_element(fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)
            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for a in statesdm[acharge]:
                    ba = si.get_ind_dm1(b, a, acharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, a]*Tba[l, a, bp]*phi1fct[l, ba, 1]
                for c in statesdm[ccharge]:
                    cb = si.get_ind_dm1(c, b, bcharge)
                    for l in range(nleads):
                        fct_bbpp += -Tba[l, bpp, c]*Tba[l, c, bp]*phi1fct[l, cb, 0].conjugate()
                kh.set_matrix_element(fct_bbpp, b, bp, bcharge, b, bpp, bcharge)
        # --------------------------------------------------
        for c, cp in itertools.product(statesdm[ccharge], statesdm[ccharge]):
            if kh.is_included(c, cp, ccharge):
                cbp = si.get_ind_dm1(c, bp, bcharge)
                cpb = si.get_ind_dm1(cp, b, bcharge)
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += (+ Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cbp, 1]
                                - Tba[l, b, c]*Tba[l, cp, bp]*phi1fct[l, cpb, 1].conjugate())
                kh.set_matrix_element(fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    def generate_current(self):
        """
        Calculates currents using 1vN approach.

        Parameters
        ----------
        self.phi1 : array
            (Modifies) Values of first order density matrix elements
            stored in nleads by ndm1 numpy array.
        self.current : array
            (Modifies) Values of the current having nleads entries.
        self.energy_current : array
            (Modifies) Values of the energy current having nleads entries.
        self.heat_current : array
            (Modifies) Values of the heat current having nleads entries.
        """
        E, Tba = self.qd.Ea, self.leads.Tba
        phi1fct, phi1fct_energy = self.phi1fct, self.phi1fct_energy

        si = self.si
        ndm0, ndm1, npauli = si.ndm0, si.ndm1, si.npauli
        ncharge, nleads, statesdm = si.ncharge, si.nleads, si.statesdm

        kh = self.kernel_handler
        kh.set_phi0(self.phi0)

        phi1 = np.zeros((nleads, ndm1), dtype=complexnp)
        current = np.zeros(nleads, dtype=complexnp)
        energy_current = np.zeros(nleads, dtype=complexnp)

        for charge in range(ncharge-1):
            ccharge = charge+1
            bcharge = charge
            for c, b in itertools.product(statesdm[ccharge], statesdm[bcharge]):
                cb = si.get_ind_dm1(c, b, bcharge)

                for l in range(nleads):
                    fct1 = phi1fct[l, cb, 0]
                    fct2 = phi1fct[l, cb, 1]
                    fct1h = phi1fct_energy[l, cb, 0]
                    fct2h = phi1fct_energy[l, cb, 1]

                    for bp in si.statesdm[bcharge]:
                        if not kh.is_included(bp, b, bcharge):
                            continue
                        phi0bpb = kh.get_phi0_element(bp, b, bcharge)

                        phi1[l, cb] += Tba[l, c, bp]*phi0bpb*fct1
                        current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1
                        energy_current[l] += Tba[l, b, c]*Tba[l, c, bp]*phi0bpb*fct1h

                    for cp in statesdm[ccharge]:
                        if not kh.is_included(c, cp, ccharge):
                            continue
                        phi0ccp = kh.get_phi0_element(c, cp, ccharge)

                        phi1[l, cb] += Tba[l, cp, b]*phi0ccp*fct2
                        current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2
                        energy_current[l] += Tba[l, b, c]*phi0ccp*Tba[l, cp, b]*fct2h

        self.phi1 = phi1
        self.current = np.array(-2*current.imag, dtype=doublenp)
        self.energy_current = np.array(-2*energy_current.imag, dtype=doublenp)
        self.heat_current = self.energy_current - self.current*self.leads.mulst
# ---------------------------------------------------------------------------------------------------
