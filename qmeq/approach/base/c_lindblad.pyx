# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order Lindblad kernel.
   For docstrings see documentation of module lindblad."""

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import doublenp
from ...mytypes import complexnp

# Cython imports

cimport numpy as np
cimport cython

from libc.math cimport sqrt

from ...specfunc.c_specfunc cimport func_pauli

from ..c_aprclass cimport Approach
from ..c_aprclass cimport KernelHandler

# ---------------------------------------------------------------------------------------------------
# Lindblad approach
# ---------------------------------------------------------------------------------------------------
cdef class ApproachLindblad(Approach):

    kerntype = 'Lindblad'

    cpdef generate_fct(self):
        cdef complex_t [:, :, :] Tba = self.leads.Tba
        cdef double_t [:] E = self.qd.Ea
        cdef double_t [:] mulst = self.leads.mulst
        cdef double_t [:] tlst = self.leads.tlst
        cdef double_t [:, :] dlst = self.leads.dlst

        cdef KernelHandler kh = self._kernel_handler
        cdef long_t nleads = kh.nleads

        cdef long_t itype = self.funcp.itype

        cdef long_t b, a, l
        cdef double_t Eba

        cdef complex_t [:, :, :] tLba = np.zeros((nleads, kh.nmany, kh.nmany), dtype=complexnp)
        cdef np.ndarray[double_t, ndim=1] rez = np.zeros(2, dtype=doublenp)

        for i in range(kh.ndm1):
            b = kh.all_ba[i, 0]
            a = kh.all_ba[i, 1]
            Eba = E[b]-E[a]
            for l in range(nleads):
                # fct1 = fermi_func((E[b]-E[a]-mulst[l])/tlst[l])
                # fct2 = 1-fct1
                func_pauli(Eba, mulst[l], tlst[l], dlst[l, 0], dlst[l, 1], itype, rez)
                tLba[l, b, a] = sqrt(rez[0])*Tba[l, b, a]
                tLba[l, a, b] = sqrt(rez[1])*Tba[l, a, b]

        self.tLba = tLba

    cdef void set_coupling(self):
        self._tLba = self.tLba

    cdef void generate_coupling_terms(self,
                long_t b, long_t bp, long_t bcharge,
                KernelHandler kh) nogil:

        cdef long_t bpp, a, ap, c, cp
        cdef complex_t fct_aap, fct_bppbp, fct_bbpp, fct_ccp

        cdef long_t i, j, l
        cdef long_t nleads = kh.nleads
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef complex_t [:, :, :] tLba = self._tLba

        cdef long_t acharge = bcharge-1
        cdef long_t ccharge = bcharge+1

        cdef long_t acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
        cdef long_t bcount = kh.statesdm_count[bcharge]
        cdef long_t ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

        # --------------------------------------------------
        for i in range(acount):
            for j in range(acount):
                a = statesdm[acharge, i]
                ap = statesdm[acharge, j]
                if not kh.is_included(a, ap, acharge):
                    continue
                fct_aap = 0
                for l in range(nleads):
                    fct_aap += tLba[l, b, a]*tLba[l, bp, ap].conjugate()
                kh.set_matrix_element(1j*fct_aap, b, bp, bcharge, a, ap, acharge)

        # --------------------------------------------------
        for i in range(bcount):
            bpp = statesdm[bcharge, i]
            if kh.is_included(bpp, bp, bcharge):
                fct_bppbp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, a, b].conjugate()*tLba[l, a, bpp]
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    for l in range(nleads):
                        fct_bppbp += -0.5*tLba[l, c, b].conjugate()*tLba[l, c, bpp]
                kh.set_matrix_element(1j*fct_bppbp, b, bp, bcharge, bpp, bp, bcharge)

            # --------------------------------------------------
            if kh.is_included(b, bpp, bcharge):
                fct_bbpp = 0
                for j in range(acount):
                    a = statesdm[acharge, j]
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, a, bpp].conjugate()*tLba[l, a, bp]
                for j in range(ccount):
                    c = statesdm[ccharge, j]
                    for l in range(nleads):
                        fct_bbpp += -0.5*tLba[l, c, bpp].conjugate()*tLba[l, c, bp]
                kh.set_matrix_element(1j*fct_bbpp, b, bp, bcharge, b, bpp, bcharge)

        # --------------------------------------------------
        for i in range(ccount):
            for j in range(ccount):
                c = statesdm[ccharge, i]
                cp = statesdm[ccharge, j]
                if not kh.is_included(c, cp, ccharge):
                    continue
                fct_ccp = 0
                for l in range(nleads):
                    fct_ccp += tLba[l, b, c]*tLba[l, bp, cp].conjugate()
                kh.set_matrix_element(1j*fct_ccp, b, bp, bcharge, c, cp, ccharge)
        # --------------------------------------------------

    cpdef generate_current(self):
        cdef double_t [:] E = self.qd.Ea
        cdef complex_t [:, :, :] tLba = self.tLba

        cdef long_t i, j, k, l
        cdef long_t a, b, bp, c
        cdef long_t acharge, bcharge, ccharge
        cdef long_t acount, bcount, ccount
        cdef complex_t fcta, fctc, phi0bbp

        cdef KernelHandler kh = self._kernel_handler
        cdef long_t [:, :] statesdm = kh.statesdm
        cdef long_t nleads = kh.nleads
        kh.set_phi0(self.phi0)

        cdef np.ndarray[complex_t, ndim=1] current = np.zeros(nleads, dtype=complexnp)
        cdef np.ndarray[complex_t, ndim=1] energy_current = np.zeros(nleads, dtype=complexnp)

        for bcharge in range(kh.ncharge):
            acharge = bcharge-1
            ccharge = bcharge+1

            acount = kh.statesdm_count[acharge] if acharge >= 0 else 0
            bcount = kh.statesdm_count[bcharge]
            ccount = kh.statesdm_count[ccharge] if ccharge <= kh.ncharge else 0

            for i in range(bcount):
                for j in range(bcount):
                    b = statesdm[bcharge, i]
                    bp = statesdm[bcharge, j]
                    if not kh.is_included(b, bp, bcharge):
                        continue

                    phi0bbp = kh.get_phi0_element(b, bp, bcharge)
                    for l in range(nleads):
                        for k in range(acount):
                            a = statesdm[acharge, k]
                            fcta = tLba[l, a, b]*phi0bbp*tLba[l, a, bp].conjugate()
                            current[l] = current[l] - fcta
                            energy_current[l] = energy_current[l] + (E[a]-0.5*(E[b]+E[bp]))*fcta
                        for k in range(ccount):
                            c = statesdm[ccharge, k]
                            fctc = tLba[l, c, b]*phi0bbp*tLba[l, c, bp].conjugate()
                            current[l] = current[l] + fctc
                            energy_current[l] = energy_current[l] + (E[c]-0.5*(E[b]+E[bp]))*fctc

        self.current = np.array(current.real, dtype=doublenp)
        self.energy_current = np.array(energy_current.real, dtype=doublenp)
        self.heat_current = self.energy_current - self.current*self.leads.mulst
# ---------------------------------------------------------------------------------------------------
