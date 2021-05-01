'''
   create projection matrix. This is a simpler version than the one in 
   operator.py
 
   used from engine.py to create a mapping between fes.
'''
import numpy as np
from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   comm = MPI.COMM_WORLD
   from petram.helper.mpi_recipes import  gather_vector, allgather_vector
   from mfem.common.mpi_debug import nicePrint   
else:
   import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Projection')

def fes_mapping(element1, order1, dim1, dim2):
    '''
    gives automatic element mapping for domain-boundary coupling
    '''
    msg = ("mapping not supported from " + element1 +
           " dim: "+str(dim1) + " " + str(dim2))
    if dim1 == dim2:
        ret = element1, order1

    elif element1.startswith("ND"):
        if dim1 == 3 and dim2 == 2:
            ret = "ND_FECollection", order1
        elif dim1 == 2 and dim2 == 1:
            ret = "L2_FECollection", order1 - 1
        elif dim1 == 2 and dim2 == 3:
            ret = "ND_FECollection", order1
        else:
            assert False, msg

    elif element1.startswith("RT"):
        if dim1 == 3 and dim2 == 2:
            ret = "L2_FECollection", order1 - 1
        elif dim1 == 2 and dim2 == 1:
            ret = "L2_FECollection", order1 - 1
        else:
            assert False, msg

    elif element1.startswith("L2"):
        if dim1 == 2 and dim2 == 3:
            ret = "RT_FECollection", order1 + 1
        elif dim1 == 1 and dim2 == 2:
            ret = "ND_FECollection", order1 + 1
        else:
            assert False, msg

    elif element1.startswith("H1"):
        ret = "H1_FECollection", order1
    else:
        assert False, msg

    return ret

def simple_projection(fes1, fes2, sel_mode, tol=1e-5):
    mesh1 = fes1.GetMesh()
    mesh2 = fes1.GetMesh()
    dim1 = mesh1.Dimension()
    dim2 = mesh2.Dimension()

    projmode = ""
    if dim2 == 3:
       if sel_mode == "domain":
           projmode = "volume"
       elif sel_mode == "boundary":
           projmode = "surface"
    elif dim2 == 2:
       if sel_mode == "domain":
           projmode = "surface"
       elif sel_mode == "boundary":
           projmode = "edge"
    elif dim2 == 1:
       if sel_mode == "domain":
           projmode = "edge"
       elif sel_mode == "boundary":
           projmode = "vertex"
    assert projmode != "", "unknow projection mode"
    if sel_mode == "domain":
        if dim1 == dim2:
            idx1 = np.unique(mesh1.GetAttributeArray())
        elif dim1 == dim2+1:
            idx1 = np.unique(mesh1.GetBdrAttributeArray())
        else:
            assert False, "unsupported mode"
        idx2 = np.unique(mesh2.GetAttributeArray())
    else:
        if dim1 == dim2:
            idx1 = np.unique(mesh1.GetBdrAttributeArray())
        elif dim1 == dim2-1:
            idx1 = np.unique(mesh1.GetAttributeArray())
        idx2 = np.unique(mesh2.GetBdrAttributeArray())

    if use_parallel:
        idx1 = list(idx1)
        idx2 = list(idx2)
        idx1 = list(set(sum(comm.allgather(idx1), [])))
        idx2 = list(set(sum(comm.allgather(idx2), [])))
    idx = np.intersect1d(idx1, idx2)
    idx1 = list(idx)
    idx2 = list(idx)

    if use_parallel:
        # we may not need this?
        idx1 =  list(set(sum(comm.allgather(idx1), [])))
        idx2 =  list(set(sum(comm.allgather(idx2), [])))

    dprint1("projection index ", idx1, idx2)
        
    from petram.helper.dof_map import notrans

    sdim1 = mesh1.SpaceDimension()
    sdim2 = mesh2.SpaceDimension()        

    trans1 = notrans
    trans2 = notrans
    
    from petram.helper.dof_map import projection_matrix as pm
    # matrix to transfer unknown from trail to test
    M, row, col = pm(idx2, idx1, fes2, [], fes2=fes1,
                     trans1=trans2, trans2=trans1,
                     mode=projmode, tol=tol, filldiag=False,
                     old_mapping=False)
    return M