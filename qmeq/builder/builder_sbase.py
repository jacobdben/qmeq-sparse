"""Module containing BuilderSBase and BuilderSManyBody classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .builder_base import BuilderBase
from .builder_base import attribute_map
from .builder_base import BuilderManyBody
from ..indexing import StateIndexingDM

from ..qdot import QuantumDot
from ..qdot import QuantumDotS
from ..leadstun import LeadsTunneling
from .funcprop import FunctionProperties

# -----------------------------------------------------------
# Python modules
from ..approach.sbase.pauli import ApproachPauli as ApproachPyPauli

# Cython compiled modules

try:
    from ..approach.sbase.c_pauli import ApproachPauli

except ImportError:
    print("WARNING: Cannot import Cython compiled modules for the approaches (builder_sbase.py).")
    ApproachPauli = ApproachPyPauli

# -----------------------------------------------------------

attribute_map_sbase = dict(pairing='qd')
attribute_map.update(attribute_map_sbase)

class BuilderSBase(BuilderBase):

    def __init__(self,
                 nsingle=0, hsingle={}, coulomb={}, pairing={},
                 nleads=0, tleads={}, mulst={}, tlst={}, dband={},
                 indexing=None, kpnt=None,
                 kerntype='Pauli', symq=True, norm_row=0, solmethod=None,
                 itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                 mtype_qd=complex, mtype_leads=complex,
                 symmetry="parity", herm_hs=True, herm_c=False, m_less_n=True):

        self._init_copy_data(locals())
        self._init_validate_data()
        self._init_set_globals()
        self._init_set_approach_class()
        self._init_create_setup()
        self._init_create_appr()

    def _init_set_globals(self):
        self.globals = globals()

    def _init_create_setup(self):
        data = self.data
        self.funcp = FunctionProperties(symq=data.symq, norm_row=data.norm_row, solmethod=data.solmethod,
                                        itype=data.itype, dqawc_limit=data.dqawc_limit,
                                        mfreeq=data.mfreeq, phi0_init=data.phi0_init,
                                        mtype_qd=data.mtype_qd, mtype_leads=data.mtype_leads,
                                        kpnt=data.kpnt, dband=data.dband)

        icn = self.Approach.indexing_class_name
        self.si = self.globals[icn](data.nsingle, data.indexing, data.symmetry)
        self.qd = QuantumDotS(data.hsingle, data.coulomb, data.pairing, self.si,
                             data.herm_hs, data.herm_c, data.m_less_n, data.mtype_qd)
        self.leads = LeadsTunneling(data.nleads, data.tleads, self.si,
                                    data.mulst, data.tlst, data.dband, data.mtype_leads)

    def add(self, hsingle=None, coulomb=None, pairing=None, tleads=None, mulst=None, tlst=None, dlst=None):
        """
        See add() in parent + pairing
        """
        if not (hsingle is None and coulomb is None):
            self.qd.add(hsingle, coulomb)
        if not (pairing is None):
            self.qd.add_pairing(pairing)
        if not (tleads is None and mulst is None and tlst is None and dlst is None):
            self.leads.add(tleads, mulst, tlst, dlst)

    def change(self, hsingle=None, coulomb=None, pairing=None, tleads=None, mulst=None, tlst=None, dlst=None):
        """
        see change() in parent + pairing
        """
        if not (hsingle is None and coulomb is None):
            self.qd.change(hsingle, coulomb)
        if not (pairing is None):
            self.qd.change_pairing(pairing)
        if not (tleads is None and mulst is None and tlst is None and dlst is None):
            self.leads.change(tleads, mulst, tlst, dlst)



class BuilderManyBodyS(BuilderSBase, BuilderManyBody):
    """
    Class for building the system for stationary transport calculations,
    using many-body states as an input.

    For missing descriptions of attributes use help(Builder).

    Attributes
    ----------
    Ea : array
        nmany by 1 array containing many-body Hamiltonian eigenvalues.
    Na : array
        nmany by 1 array containing parities of many-body states. 0 for even, 1 for odd
    Tba_plus : array
        nleads by nmany by nmany array, which contains many-body tunneling amplitude matrix
        for adding electrons. Per lead : sum_i ( t * <b|d_i^\dagger|a> )
    """

    def __init__(self,
                 Ea=None, Na=[0], Tba_plus=None,
                 mulst={}, tlst={}, dband={}, kpnt=None,
                 kerntype='Pauli', symq=True, norm_row=0, solmethod=None,
                 itype=0, dqawc_limit=10000, mfreeq=False, phi0_init=None,
                 mtype_qd=complex, mtype_leads=complex,
                 symmetry='parity', herm_hs=True, herm_c=False, m_less_n=True):

        nleads = Tba_plus.shape[0] if Tba_plus is not None else 0

        BuilderSBase.__init__(self,
            nleads=nleads, mulst=mulst, tlst=tlst, dband=dband, kpnt=kpnt,
            kerntype=kerntype, symq=symq, norm_row=norm_row, solmethod=solmethod,
            itype=itype, dqawc_limit=dqawc_limit, mfreeq=mfreeq, phi0_init=phi0_init,
            mtype_qd=mtype_qd, mtype_leads=mtype_leads,
            symmetry=symmetry, herm_hs=herm_hs, herm_c=herm_c, m_less_n=m_less_n,
            indexing='charge')

        self._init_state_indexing(Na, Ea)

        self.qd.Ea = Ea
        self.leads.Tba = Tba_plus