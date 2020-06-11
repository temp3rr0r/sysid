"""
Default imports.
"""
# from sysid.ss import StateSpaceDiscreteLinear, StateSpaceDataList, StateSpaceDataArray
# from sysid.subspace import subspace_det_algo1, prbs, nrms

# Uncomment for CUDA numpy
# from sysid.ss_cupy import StateSpaceDiscreteLinear, StateSpaceDataList, StateSpaceDataArray
# from sysid.subspace_cupy import subspace_det_algo1, prbs, nrms

# Uncomment for CUDA numpy fp16
from sysid.ss_cupy_fp16 import StateSpaceDiscreteLinear, StateSpaceDataList, StateSpaceDataArray
from sysid.subspace_cupy_fp16 import subspace_det_algo1, prbs, nrms

# Uncomment for CPU numpy
# from sysid.ss_numpy import StateSpaceDiscreteLinear, StateSpaceDataList, StateSpaceDataArray
# from sysid.subspace_numpy import subspace_det_algo1, prbs, nrms

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
