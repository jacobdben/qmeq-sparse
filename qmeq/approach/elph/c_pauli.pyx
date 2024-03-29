# cython: boundscheck=False
# cython: cdivision=True
# cython: infertypes=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False

"""Module containing cython functions, which generate first order Pauli kernel.
   For docstrings see documentation of module pauli."""

# Python imports

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import itertools

from ...mytypes import complexnp
from ...mytypes import doublenp

# Cython imports

cimport numpy as np
cimport cython

from ...specfunc.c_specfunc_elph cimport FuncPauliElPh

from ..c_aprclass cimport ApproachElPh
from ..c_aprclass cimport KernelHandler

from ..base.c_pauli cimport ApproachPauli as ApproachPauliBase

# ---------------------------------------------------------------------------------------------------------
# Pauli master equation
# ---------------------------------------------------------------------------------------------------------
cdef class ApproachPauli(ApproachElPh):

    kerntype = 'Pauli'
    no_coherences = True

    def get_kern_size(self):
        return self.si.npauli

    cpdef generate_fct(self):
        ApproachPauliBase.generate_fct(self)

        cdef double_t [:] E = self.qd.Ea
        cdef complex_t [:, :, :] Vbbp = self.baths.Vbbp

        cdef KernelHandler kh = self._kernel_handler.elph
        cdef long_t nbaths = kh.nbaths

        cdef long_t b, bp, bbp, bcharge, l, i
        cdef double_t Ebbp, xbbp

        func_pauli = FuncPauliElPh(self.baths.tlst_ph, self.baths.dlst_ph,
                                   self.baths.bath_func, self.funcp.eps_elph)

        cdef double_t [:, :] paulifct = np.zeros((nbaths, kh.ndm0), dtype=doublenp)

        for i in range(kh.ndm0):
            b = kh.all_bbp[i, 0]
            bp = kh.all_bbp[i, 1]
            bcharge = kh.all_bbp[i, 2]

            # The diagonal elements b=bp are excluded, because they do not contribute
            if b == bp:
                continue

            bbp = kh.get_ind_dm0(b, bp, bcharge)
            Ebbp = E[b]-E[bp]
            for l in range(nbaths):
                xbbp = 0.5*(Vbbp[l, b, bp]*Vbbp[l, b, bp].conjugate() +
                            Vbbp[l, bp, b].conjugate()*Vbbp[l, bp, b]).real
                func_pauli.eval(Ebbp, l)
                paulifct[l, bbp] = xbbp*func_pauli.val

        self.paulifct_elph = paulifct

    cdef void set_coupling(self):
        ApproachPauliBase.set_coupling(self)
        self._paulifct_elph = self.paulifct_elph

    cdef void generate_coupling_terms(self,
                long_t b, long_t bp, long_t bcharge,
                KernelHandler kh) nogil:

        ApproachPauliBase.generate_coupling_terms(self, b, bp, bcharge, kh)

        cdef long_t bb, a, aa, ab, ba
        cdef double_t fctm, fctp

        cdef long_t i, l
        cdef long_t nbaths = kh.nbaths
        cdef long_t [:, :] statesdm = kh.statesdm

        cdef double_t [:, :] paulifct = self._paulifct_elph

        cdef long_t bcount = kh.statesdm_count[bcharge]

        bb = kh.get_ind_dm0(b, b, bcharge)
        for i in range(bcount):
            a = statesdm[bcharge, i]

            aa = kh.get_ind_dm0(a, a, bcharge)
            if aa == -1: # if not kh.is_included(a, a, bcharge)
                continue

            ab = kh.elph.get_ind_dm0(a, b, bcharge)
            if ab == -1: # if not kh.elph.is_included(a, b, bcharge)
                continue

            ba = kh.elph.get_ind_dm0(b, a, bcharge)
            fctm, fctp = 0, 0
            for l in range(nbaths):
                fctm -= paulifct[l, ab]
                fctp += paulifct[l, ba]
            kh.set_matrix_element_pauli(fctm, fctp, bb, aa)

    cpdef generate_current(self):
        ApproachPauliBase.generate_current(self)
