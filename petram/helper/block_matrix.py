from __future__ import print_function
from functools import reduce
'''
Utility class to handle BlockMatrix made from scipy-sparse and
Hypre with the same interface
'''
import itertools

import numpy as np
import scipy
from scipy.sparse import coo_matrix, spmatrix, lil_matrix, csc_matrix

from petram.mfem_config import use_parallel
import mfem.common.chypre as chypre
from mfem.common.mpi_debug import nicePrint

if use_parallel:
    from petram.helper.mpi_recipes import *
    from mfem.common.parcsr_extra import *
    import mfem.par as mfem
    default_kind = 'hypre'
else:
    import mfem.ser as mfem
    default_kind = 'scipy'

from petram.solver.solver_utils import make_numpy_coo_matrix
from petram.helper.matrix_file import write_coo_matrix, write_vector

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('BlockMatrix')
format_memory_usage = debug.format_memory_usage


class One(object):
    '''
    An identity matrix (used in P and mimic 1*X = X)
    '''

    def __init__(self, ref):
        self._shape = ref.shape
        self._is_hypre = False
        if hasattr(ref, "GetColPartArray"):
            self._cpart = ref.GetColPartArray()
            self._is_hypre = True
        if hasattr(ref, "GetRowPartArray"):
            self._rpart = ref.GetRowPartArray()
            self._is_hypre = True

    def __repr__(self):
        return "IdentityMatrix"

    @property
    def shape(self): return self._shape

    @property
    def nnz(self):
        return self.shape[0]

    def true_nnz(self):
        return self.shape[0]

    def GetColPartArray(self):
        return self._cpart

    def GetRowPartArray(self):
        return self._rpart

    def __mul__(self, other):
        return other

    def transpose(self):
        return self

    def dot(self, other):
        return other

    @property
    def isHypre(self):
        return self._is_hypre


class ScipyCoo(coo_matrix):
    def true_nnz(self):
        if hasattr(self, "eliminate_zeros"):
            self.eliminate_zeros()
        return self.nnz

    def __add__(self, other):
        ret = super(ScipyCoo, self).__add__(other)
        return convert_to_ScipyCoo(ret)

    def __sub__(self, other):
        ret = super(ScipyCoo, self).__sub__(other)
        return convert_to_ScipyCoo(ret)

    def setDiag(self, idx, value=1.0):

        ret = self.tolil()
        idx = np.array(idx, dtype=int, copy=False)
        ret[idx, idx] = value
        # for i in idx:
        #   ret[i,i] = value
        ret = ret.tocoo()
        self.data = ret.data
        self.row = ret.row
        self.col = ret.col
    '''
    def resetDiagImag(self, idx):
        ret = self.tolil()

        for i in idx:
           ret[i,i] = ret[i,i].real
        ret = ret.tocoo()
        self.data = ret.data
        self.row  = ret.row
        self.col  = ret.col
    '''

    def resetRow(self, rows, inplace=True):
        rows = np.array(rows, dtype=int, copy=False)
        #ret = self.tolil()
        #ret[rows, :] = 0.0
        #ret.data[np.in1d(ret.row in rows)] = 0.0
        ret = self.tocsr()
        ret[rows, :] = ret[rows, :] * 0.0
        ret = ret.tocoo()

        if inplace:
            self.data = ret.data
            self.row = ret.row
            self.col = ret.col
            return self
        else:
            return ret

    def resetCol(self, cols, inplace=True):
        cols = np.array(cols, dtype=int, copy=False)
        #ret.data[np.in1d(ret.col in cols)] = 0.0
        #ret = self.tolil()
        ret = self.tocsc()
        ret[:, cols] = ret[:, cols] * 0.0
        ret = ret.tocoo()

        if inplace:
            self.data = ret.data
            self.row = ret.row
            self.col = ret.col
            return self
        else:
            return ret

    def selectRows(self, nonzeros):
        m = self.tocsr()
        ret = (m[nonzeros, :]).tocoo()
        return convert_to_ScipyCoo(ret)

    def selectCols(self, nonzeros):
        m = self.tocsc()
        ret = (m[:, nonzeros]).tocoo()
        return convert_to_ScipyCoo(ret)

    def dot(self, P):
        return convert_to_ScipyCoo(super(ScipyCoo, self).dot(P))

    def rap(self, P):
        PP = P.conj().transpose()
        return convert_to_ScipyCoo(PP.dot(self.dot(P)))

    def conj(self, inplace=False):
        if inplace:
            np.conj(self.data, out=self.data)
            return self
        else:
            return self.conj()

    def elimination_matrix(self, nonzeros):
        '''
        # P elimination matrix for column vector
        [1 0  0][x1]    [x1]
        [0 0  1][x2]  = [x3]
                [x3]

        P^t (transpose does reverse operation)
           [1 0][x1]    [x1]
           [0 0][x3]  = [0]
           [0,1]        [x3]

        # P^t does elimination matrix for horizontal vector
                  [1,0]
        [x1 x2 x3][0,0]  = [x1, x3]
                  [0,1]

        # P does reverse operation for horizontal vector
               [1,0 0]
        [x1 x3][0,0 1]  = [x1, 0  x3]
        '''
        ret = lil_matrix((len(nonzeros), self.shape[0]))
        for k, z in enumerate(nonzeros):
            ret[k, z] = 1.
        return convert_to_ScipyCoo(ret.tocoo())

    def get_global_coo(self):
        '''
        global representation:
           zero on non-root node
        '''
        try:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank
            if myid != 0:
                return coo_matrix(self.shape)
            else:
                return self
        except BaseException:
            return self

    def get_mfem_sparsemat(self):
        '''
        generate mfem::SparseMatrix using the same data
        '''
        if np.iscomplexobj(self):
            csr_r = self.tocsr().real
            csr_r.eliminate_zeros()
            csr_i = self.tocsr().imag
            csr_i.eliminate_zeros()
            return mfem.SparseMatrix(csr_r), mfem.SparseMatrix(csr_i)
        else:
            csr_r = self.tocsr()
            csr_r.eliminate_zeros()
            csr_i = None
            return mfem.SparseMatrix(csr_r), None

    def __repr__(self):
        return "ScipyCoo" + str(self.shape)

    def __str__(self):
        return "ScipyCoo" + str(self.shape) + super(ScipyCoo, self).__str__()

    @property
    def isHypre(self):
        return False

    '''
    two function to provde the same interface as CHypre
    '''

    def GetColPartArray(self):
        return (0, self.shape[1], self.shape[1])

    def GetRowPartArray(self):
        return (0, self.shape[0], self.shape[0])
    GetPartitioningArray = GetRowPartArray

    def eliminate_RowsCols(self, B, tdof, inplace=True, diagpolicy=1):
        '''
        daigpolicy = 0  # DiagOne
        daigpolicy = 1  # DiagKeep

        Note: policy is controled from engine::filL_BCeliminate_matrix
        '''
        print("inplace flag off copying matrix")
        # A + Ae style elimination
        idx = np.in1d(self.col, tdof)
        idx2 = np.in1d(self.row, tdof)

        if diagpolicy == 0:
            diagAe = self.diagonal()[tdof] - 1
            diagA = 1
        else:
            diagAe = 0
            diagA = self.diagonal()[tdof]

        aidx = np.logical_or(idx, idx2)
        #aidx = []
        AeCol = self.col[aidx]
        AeRow = self.row[aidx]
        AeData = self.data[aidx]
        Ae2 = coo_matrix((AeData, (AeRow, AeCol)),
                         shape=self.shape, dtype=self.dtype)
        Ae2c = Ae2.tocsr()
        Ae2c[tdof, tdof] = diagAe
        Ae2 = Ae2c.tocoo()

        if inplace:
            target = self
            target_b = B.tolil()
        else:
            target = self.copy()
            target_b = B.copy().tolil()

        target.data[idx] = 0
        target.data[idx2] = 0
        target.eliminate_zeros()
        lil2 = target.tolil()
        lil2[tdof, tdof] = diagA

        # if diagpolicy == 1:
        # target_b[tdof, 0] = target_b[tdof, 0].toarray().flatten() * diagA
        target_b[tdof, 0] = diagA

        coo = lil2.tocoo()
        target.data = coo.data
        target.row = coo.row
        target.col = coo.col

        coo_b = convert_to_ScipyCoo(target_b)

        return Ae2, target, coo_b

    def get_elements(self, tdof):
        slil = self.tolil()
        value = slil[tdof, :].toarray()
        return value

    def set_elements(self, tdof, m):
        slil = self.tolil()
        slil[tdof, :1] = m
        coo = slil.tocoo()
        self.data = coo.data
        self.row = coo.row
        self.col = coo.col

    def copy_element(self, tdof, m):
        mlil = m.tolil()
        value = mlil[tdof, 0].toarray()

        slil = self.tolil()
        slil[tdof, :1] = value
        coo = slil.tocoo()
        self.data = coo.data
        self.row = coo.row
        self.col = coo.col


def convert_to_ScipyCoo(mat):
    if isinstance(mat, np.ndarray):
        mat = coo_matrix(mat)
    if isinstance(mat, spmatrix):
        if not isinstance(mat, coo_matrix):
            mat = mat.tocoo()
        mat.__class__ = ScipyCoo
    return mat


class BlockMatrix(object):
    def __init__(self, shape, kind=default_kind, complex=False):
        '''
        kind : scipy
                  stores scipy sparse or numpy array
               hypre
        '''
        self.block = [[None] * shape[1] for x in range(shape[0])]
        self.kind = kind
        self.shape = shape
        self.complex = complex

        self._rsize = None
        self._csize = None
        self._grsize = None
        self._gcsize = None
        self._rp = None
        self._cp = None

    def cp_size_hint(self, size_hint):
        size_hint.check_element_sizes()

        self._rsize = size_hint._rsize
        self._csize = size_hint._csize
        self._grsize = size_hint._grsize
        self._gcsize = size_hint._gcsize
        self._rp = size_hint._rp
        self._cp = size_hint._cp

    @property
    def rsize(self):
        if self._grsize is None:
            self.check_element_sizes()
        return self._rsize

    @property
    def csize(self):
        if self._grsize is None:
            self.check_element_sizes()
        return self._csize

    @property
    def grsize(self):
        if self._grsize is None:
            self.check_element_sizes()
        return self._grsize

    @property
    def gcsize(self):
        if self._grsize is None:
            self.check_element_sizes()
        return self._gcsize

    @property
    def rp(self):
        return self._rp

    @property
    def cp(self):
        return self._cp

    @property
    def is_zero(self):
        '''
        check if block is completely zero
        '''
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    return False
        return True

    def __getitem__(self, idx):
        try:
            r, c = idx
        except BaseException:
            r = idx
            c = 0
        if isinstance(r, slice):
            new_r = range(self.shape[0])[r]
            ret = BlockMatrix((len(new_r), self.shape[1]))
            for i, r in enumerate(new_r):
                for j in range(self.shape[1]):
                    ret[i, j] = self[r, j]
            return ret
        elif isinstance(c, slice):
            new_c = range(self.shape[1])[c]
            ret = BlockMatrix((self.shape[0], len(new_c)))
            for i in range(self.shape[0]):
                for j, c in enumerate(new_c):
                    ret[i, j] = self[i, c]
            return ret
        else:
            return self.block[r][c]

    def __setitem__(self, idx, v):
        try:
            r, c = idx
        except BaseException:
            r = idx
            c = 0
        if v is not None:
            if isinstance(v, chypre.CHypreMat):
                if v.isComplex():
                    self.complex = True
            elif isinstance(v, chypre.CHypreVec):
                if v.isComplex():
                    self.complex = True
            elif v is None:
                pass
            else:
                v = convert_to_ScipyCoo(v)
                if np.iscomplexobj(v):
                    self.complex = True
        self.block[r][c] = v

    def __add__(self, v):
        if self.shape != v.shape:
            raise ValueError("Block format is inconsistent")

        shape = self.shape
        complex = self.complex or v.complex
        ret = BlockMatrix(shape, kind=self.kind, complex=complex)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if self[i, j] is None and v[i, j] is None:
                    ret[i, j] = None
                elif self[i, j] is None:
                    ret[i, j] = v[i, j]
                elif v[i, j] is None:
                    ret[i, j] = self[i, j]
                else:
                    ret[i, j] = self[i, j] + v[i, j]
        return ret

    def __sub__(self, v):
        if self.shape != v.shape:
            raise ValueError("Block format is inconsistent")
        shape = self.shape
        complex = self.complex or v.complex
        ret = BlockMatrix(shape, kind=self.kind, complex=complex)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if self[i, j] is None and v[i, j] is None:
                    ret[i, j] = None
                elif self[i, j] is None:
                    ret[i, j] = -v[i, j]
                elif v[i, j] is None:
                    ret[i, j] = self[i, j]
                else:
                    ret[i, j] = self[i, j] - v[i, j]
        return ret

    def __mul__(self, other):  # note other is scalar.. this is not matrix multiplicaiton
        shape = self.shape

        ret = BlockMatrix(shape, kind=self.kind, complex=self.complex)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if self[i, j] is not None:
                    ret[i, j] = self[i, j] * other
        return ret

    def __neg__(self):
        shape = self.shape
        ret = BlockMatrix(shape, kind=self.kind, complex=self.complex)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if self[i, j] is not None:
                    ret[i, j] = -self[i, j]
        return ret

    def __repr__(self):
        txt = ["BlockMatrix" + str(self.shape)]
        for i in range(self.shape[0]):
            txt.append(str(i) + " : " + "  ".join([self.block[i][j].__repr__()
                                                   for j in range(self.shape[1])]))
        return "\n".join(txt) + "\n"

    def format_nnz(self):
        txt = []
        for i in range(self.shape[0]):
            txt.append(str(i) + " : " + ",  ".join([str(self[i, j].nnz)
                                                    if hasattr(self[i, j], "nnz") else "unknown"
                                                    for j in range(self.shape[1])]))
        return "\nnon-zero elements (nnz)\n" + "\n".join(txt)

    def normsq(self):
        shape = self.shape
        assert shape[1] == 1, "multilpe vectors are not supported"
        norm = [0]*shape[0]

        for i in range(shape[0]):
            # for j in range(shape[1]):
            for j in [0]:
                v = self[i, j]
                if isinstance(v, chypre.CHypreVec):
                    vec = v.toarray()
                elif isinstance(v, ScipyCoo):
                    vec = v.toarray()
                else:
                    assert False, "not supported"
                norm[i] += np.abs(np.sum(vec*np.conj(vec)))

        if use_parallel:
            norm = np.sum(allgather(norm), 0)

        return norm

    def norm(self):
        return np.sqrt(self.normsq())

    def average_norm(self, sq=False):
        shape = self.shape
        assert shape[1] == 1, "multilpe vectors are not supported"

        if sq:
            norm = self.normsq()
        else:
            norm = self.norm()

        length = [0]*shape[0]

        for i in range(shape[0]):
            for j in range(shape[1]):
                v = self[i, j]
                length[i] += np.prod(v.shape)  # shape is (1,x) or (x, 1)

        if use_parallel:
            length = np.sum(allgather(length), 0)

        length = np.array(length)
        return norm/length

    def print_nnz(self):
        print(self.format_nnz())

    def format_true_nnz(self):
        txt = []
        for i in range(self.shape[0]):
            txt.append(str(i) + " : " + ",  ".join([str(self[i, j].true_nnz())
                                                    if hasattr(self[i, j], "true_nnz") else "unknown"
                                                    for j in range(self.shape[1])]))
        return "non-zero elements (true nnz)\n" + "\n".join(txt)

    def print_true_nnz(self):
        print(self.format_ture_nnz())

    def print_row_part(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    print(i, j, self[i, j].GetRowPartArray())

    def print_col_part(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    print(i, j, self[i, j].GetColPartArray())

    def get_local_row_height(self, i):
        for j in range(self.shape[1]):
            blk = self.block[i][j]
            if blk is not None:
                return blk.shape[0]
        return None

    def get_local_col_width(self, j):
        for i in range(self.shape[0]):
            blk = self.block[i][j]
            if blk is not None:
                return blk.shape[1]
        return None

    def get_local_row_heights(self):
        return [self.get_local_row_height(i) for i in range(self.shape[0])]

    def get_local_col_widths(self):
        return [self.get_local_col_width(j) for j in range(self.shape[1])]

    def save_to_file(self, file):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                name = file + '_' + str(i) + '_' + str(j)
                v = self[i, j]
                if isinstance(v, chypre.CHypreMat):
                    m = v.get_local_coo()
                    write_coo_matrix(name, m)
                elif isinstance(v, chypre.CHypreVec):
                    m = coo_matrix(v.toarray()).transpose()
                    write_coo_matrix(name, m)
                elif isinstance(v, ScipyCoo):
                    write_coo_matrix(name, v)
                elif v is None:
                    continue
                else:
                    assert False, "Don't know how to write file for " + \
                        str(type(v))

    def transpose(self):
        ret = BlockMatrix((self.shape[1], self.shape[0]), kind=self.kind)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    ret[j, i] = self[i, j].transpose()
        return ret

    def add_to_element(self, i, j, v):
        if self[i, j] is None:
            self[i, j] = v
        else:
            self[i, j] = self[i, j] + v

    def dot(self, mat):
        if self.shape[1] != mat.shape[0]:
            raise ValueError("Block format is inconsistent")

        shape = (self.shape[0], mat.shape[1])
        ret = BlockMatrix(shape, kind=self.kind)

        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(self.shape[1]):
                    if self[i, k] is None:
                        continue
                    elif mat[k, j] is None:
                        continue
                    elif ret[i, j] is None:
                        ret[i, j] = self[i, k].dot(mat[k, j])
                    else:
                        # print "dot is called here", self[i, k], mat[k, j]
                        ret[i, j] = ret[i, j] + self[i, k].dot(mat[k, j])
                    # try:
                    #    ret[i,j].shape
                    # except:
                    #    ret[i,j] = coo_matrix([[ret[i,j]]])
        return ret

    def get_subblock(self, mask1, mask2):
        ret = BlockMatrix((sum(mask1), sum(mask2)), kind=self.kind,
                          complex=self.complex)
        imask1 = [x for x in range(len(mask1)) if mask1[x]]
        imask2 = [x for x in range(len(mask2)) if mask2[x]]
        for i, ii in enumerate(imask1):
            for j, jj in enumerate(imask2):
                ret[i, j] = self[ii, jj]
        return ret

    def get_block_id(self, ignore_none=True):
        blocks = sum(self.block, [])
        if ignore_none:
            return [id(b) for b in blocks if b is not None]
        else:
            return [id(b) for b in blocks]

    def check_shared_id(self, target):
        ids = target.get_block_id()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is None:
                    continue
                if id(self[i, j]) in ids:
                    print(i, j, "shared")

    def eliminate_empty_rowcol(self):
        '''
        collect empty row first. (no global communicaiton this step)

        share empty rows..

        apply it to all node

        '''
        from functools import reduce
        ret = BlockMatrix(self.shape, kind=self.kind)
        P2 = BlockMatrix(self.shape, kind=self.kind)

        dprint1(self.format_true_nnz())

        for i in range(self.shape[0]):
            nonzeros = []
            mat = None
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    if isinstance(self[i, j], ScipyCoo):
                        coo = self[i, j]
                        csr = coo.tocsr()
                        num_nonzeros = np.diff(csr.indptr)
                        knonzero = np.where(num_nonzeros != 0)[0]
                        if mat is None:
                            mat = coo
                    elif isinstance(self[i, j], chypre.CHypreMat):
                        coo = self[i, j].get_local_coo()
                        if hasattr(coo, "eliminate_zeros"):
                            coo.eliminate_zeros()
                        csr = coo.tocsr()
                        num_nonzeros = np.diff(csr.indptr)
                        knonzero = np.where(num_nonzeros != 0)[0]
                        if mat is None:
                            mat = self[i, j]
                    elif isinstance(self[i, j], chypre.CHypreVec):
                        if self[i, j].isAllZero():
                            knonzero = []
                        else:
                            knonzero = [0]
                    elif (isinstance(self[i, j], np.ndarray) and
                          self[i, j].ndim == 2):
                        knonzero = [k for k in range(self[i, j].shape[0])
                                    if any(self[i, j][k, :])]
                        self[i, j] = convert_to_ScipyCoo(self[i, j])
                        mat = self[i, j]
                    else:
                        raise ValueError("Unsuported Block Element" +
                                         str(type(self[i, j])))
                    nonzeros.append(knonzero)
                else:
                    nonzeros.append([])
            knonzeros = reduce(np.union1d, nonzeros)
            knonzeros = np.array(knonzeros, dtype=np.int64)
            # share nonzero to eliminate column...
            if self.kind == 'hypre':
                if isinstance(self[i, i], chypre.CHypreMat):
                    gknonzeros = self[i, i].GetRowPartArray()[0] + knonzeros
                else:
                    gknonzeros = knonzeros
                gknonzeros = allgather_vector(gknonzeros)
                gknonzeros = np.unique(gknonzeros)
                dprint2('nnz', coo.nnz, len(knonzero))
            else:
                gknonzeros = knonzeros

            if mat is not None:
                # hight and row partitioning of mat is used
                # to construct P2
                if len(gknonzeros) < self[i, i].shape[0]:
                    P2[i, i] = self[i, i].elimination_matrix(gknonzeros)
                else:
                    P2[i, i] = One(self[i, i])
            # what if common zero rows differs from common zero col?
            for j in range(self.shape[1]):
                if ret[i, j] is not None:
                    ret[i, j] = ret[i, j].selectRows(gknonzeros)
                elif self[i, j] is not None:
                    ret[i, j] = self[i, j].selectRows(gknonzeros)
                else:
                    pass
                if ret[j, i] is not None:
                    ret[j, i] = ret[j, i].selectCols(gknonzeros)
                elif self[j, i] is not None:
                    ret[j, i] = self[j, i].selectCols(gknonzeros)
                else:
                    pass

        return ret, P2

    def reformat_distributed_mat(self, mat, ksol, ret, mask, alpha=1, beta=0):
        '''
        reformat distributed matrix into blockmatrix (columne vector)
        '''
        L = []
        idx = 0
        imask = [x for x in range(len(mask[0])) if mask[0][x]]
        jmask = [x for x in range(len(mask[1])) if mask[1][x]]

        for j in jmask:
            for i in imask:
                if self[i, j] is not None:
                    if self.kind == 'scipy':
                        l = self[i, j].shape[1]
                    else:
                        part = self[i, j].GetColPartArray()
                        l = part[1] - part[0]
                    ref = self[i, j]
                    break
            L.append(l)
            v = mat[idx:idx + l, ksol]
            idx = idx + l

            ret.set_element_from_distributed_mat(
                v, j, 0, ref, alpha=alpha, beta=beta)
        return ret

    def set_element_from_distributed_mat(self, v, i, j, ref, alpha=1, beta=0):
        if self.kind == 'scipy':
            if alpha == 1 and beta == 0:
                self[i, j] = v.reshape(-1, 1)
            else:
                self[i, j] = self[i, j]*beta + alpha * v.reshape(-1, 1)
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            if ref.isHypre:
                v = np.ascontiguousarray(v)
                if np.iscomplexobj(v):
                    if alpha == 1 and beta == 0:
                        rv = ToHypreParVec(v.real)
                        iv = ToHypreParVec(v.imag)
                        self[i, j] = chypre.CHypreVec(rv, iv)
                    else:
                        self[i, j] *= beta
                        realpart = self[i, j][0].GetDataArray()
                        realpart += v.real*alpha
                        imagpart = self[i, j][1].GetDataArray()
                        imagpart += v.imag*alpha
                else:
                    rv = ToHypreParVec(v)
                    if alpha == 1 and beta == 0:
                        self[i, j] = chypre.CHypreVec(rv, None)
                    else:
                        self[i, j] *= beta
                        realpart = self[i, j][0].GetDataArray()
                        realpart += v*alpha
            else:
                assert False, "bug. this mode is not supported"

    def reformat_central_mat(self, mat, ksol, ret, mask, alpha=1, beta=0):
        '''
        reformat central matrix into blockmatrix (columne vector)
        so that matrix can be multiplied from the right of this

        self is a block diagonal matrix

        by default, ret is replaced by a new data
        with alpha != 1 or beta!=0, it will be set to
             old data * beta + new_data * alpha
        '''
        L = []
        idx = 0
        imask = [x for x in range(len(mask[0])) if mask[0][x]]
        jmask = [x for x in range(len(mask[1])) if mask[1][x]]

        for j in jmask:
            for i in imask:
                if self[i, j] is not None:
                    l = self[i, j].shape[1]
                    break
            L.append(l)
            ref = self[i, j]

            if mat is not None:
                v = mat[idx:idx + l, ksol]
            else:
                v = None   # slave node (will recive data)
            idx = idx + l
            ret.set_element_from_central_mat(
                v, j, 0, ref, alpha=alpha, beta=beta)
        return ret

    def set_element_from_central_mat(self, v, i, j, ref, alpha=1, beta=0):
        '''
        set element using vector in root node
        row partitioning is taken from column partitioning
        of ref
        '''
        if self.kind == 'scipy':
            #print("here", type(self[i,j]))
            # if isinstance(self[i,j], ScipyCoo):
            #   print("here", self[i,j].shape, self[i,j])
            if alpha == 1 and beta == 0:
                self[i, j] = v.reshape(-1, 1)
            else:
                self[i, j] = self[i, j]*beta + alpha * v.reshape(-1, 1)

        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD

            if ref.isHypre:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD

                part = ref.GetColPartArray()
                v = comm.bcast(v)
                start_col = part[0]
                end_col = part[1]

                v = np.ascontiguousarray(v[start_col:end_col])
                if np.iscomplexobj(v):
                    if alpha == 1 and beta == 0:
                        rv = ToHypreParVec(v.real)
                        iv = ToHypreParVec(v.imag)
                        self[i, j] = chypre.CHypreVec(rv, iv)
                    else:
                        self[i, j] *= beta
                        realpart = self[i, j][0].GetDataArray()
                        realpart += v.real*alpha
                        imagpart = self[i, j][1].GetDataArray()
                        imagpart += v.imag*alpha
                else:
                    if alpha == 1 and beta == 0:
                        rv = ToHypreParVec(v)
                        self[i, j] = chypre.CHypreVec(rv, None)
                    else:
                        self[i, j] *= beta
                        realpart = self[i, j][0].GetDataArray()
                        realpart += v*alpha
            else:
                # slave node gets the copy
                v = comm.bcast(v)
                if alpha == 1 and beta == 0:
                    self[i, j] = v.reshape(-1, 1)
                else:
                    self[i, j] = self[i, j]*beta + alpha * v.reshape(-1, 1)

    def get_squaremat_from_right(self, r, c):
        size = self[r, c].shape
        if self.kind == 'scipy':
            return ScipyCoo((size[1], size[1]))
        else:
            # this will return CHypreMat
            return self[r, c].get_squaremat_from_right()
    #
    #  methods for coo format
    #

    def gather_densevec(self):
        '''
        gather vector data to head node as dense data (for rhs)
        '''
        if self.kind == 'scipy':
            if self.complex:
                M = scipy.sparse.bmat(self.block, format='coo',
                                      dtype='complex').toarray()
            else:
                M = scipy.sparse.bmat(self.block, format='coo',
                                      dtype='float').toarray()
            return M
        else:
            data = []
            for i in range(self.shape[0]):
                if isinstance(self[i, 0], chypre.CHypreVec):
                    data.append(self[i, 0].GlobalVector())
                elif isinstance(self[i, 0], np.ndarray):
                    data.append(self[i, 0].flatten())
                elif isinstance(self[i, 0], ScipyCoo):
                    data.append(self[i, 0].toarray().flatten())
                else:
                    raise ValueError("Unsupported element" + str((i, 0)) +
                                     ":" + str(type(self[i, 0])))
            return np.hstack(data).reshape(-1, 1)

    def add_empty_square_block(self, r, c):
        '''
        add a square matrix to r, c element

        note: this is used to add diagnal block so that essential BC elimination
              works when diagnaol block is not specified in the model tree

              it is assuemd that the size of matrix can be inferred from
              off-diagonal blocks

              It generate a matrix whose memory for diag elements is allocated.

              If the size cannot be inferred, it does not do anything. This
              means form has zero col/row blocks. This happens when time-dependent
              solver fills the matrix (row corresponding to d/dt)
        '''
        #dprint1("adding empty block ", r, c)
        if self[r, c] is not None:
            assert False, "block is already filled"
        roffset = []
        coffset = []
        rsize = -1
        csize = -1
        for i in range(self.shape[0]):
            if self[i, c] is not None:
                csize = self[i, c].shape[1]
                if self.kind != 'scipy':
                    cp = self[i, c].GetColPartArray()
                    shape = self[i, c].shape
        for j in range(self.shape[1]):
            if self[r, j] is not None:
                rsize = self[r, j].shape[0]
                if self.kind != 'scipy':
                    rp = self[r, j].GetRowPartArray()
                    gsize = self[r, j].shape[0]

        if rsize < 0 or csize < 0:
            # this case does not fill the dialgona matrix
            return

        if rsize != csize:
            assert False, "matrix is not squre"
        if self.kind == 'scipy':
            m = scipy.sparse.eye(rsize, format='coo', dtype=float) * 0
            self[r, c] = convert_to_ScipyCoo(m)
        else:
            from mfem.common.chypre import SquareCHypreMat
            m = SquareCHypreMat(gsize, rp, real=True)
            self[r, c] = m

    def get_global_offsets(self, convert_real=False,
                           interleave=True):
        '''
        build matrix in coordinate format
        '''
        roffset = []
        coffset = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    roffset.append(self[i, j].shape[0])
                    break
        for j in range(self.shape[1]):
            for i in range(self.shape[0]):
                if self[i, j] is not None:
                    coffset.append(self[i, j].shape[1])
                    break
        #coffset = [self[0, j].shape[1] for j in range(self.shape[1])]
        if self.complex and convert_real:
            if interleave:
                roffset = np.vstack((roffset, roffset)).flatten()
                coffset = np.vstack((roffset, roffset)).flatten()
            else:
                roffset = np.hstack((roffset, roffset))
                coffset = np.hstack((roffset, roffset))

        roffsets = np.hstack([0, np.cumsum(roffset)])
        coffsets = np.hstack([0, np.cumsum(coffset)])
        return roffsets, coffsets

    def get_global_coo(self, dtype='float'):
        roffsets, coffsets = self.get_global_offsets()
        col = []
        row = []
        data = []
        glcoo = coo_matrix((roffsets[-1], coffsets[-1]), dtype=dtype)
        dprint1("roffset(get_global_coo)", roffsets)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is None:
                    continue
                gcoo = self[i, j].get_global_coo()
                row.append(gcoo.row + roffsets[i])
                col.append(gcoo.col + coffsets[j])
                data.append(gcoo.data)
        glcoo.col = np.hstack(col)
        glcoo.row = np.hstack(row)
        glcoo.data = np.hstack(data)

        return glcoo

    #
    #  methods for distributed csr format
    #
    def get_local_partitioning(self, convert_real=True,
                               interleave=True,
                               merge_realimag=False):
        '''
        build matrix in coordinate format
        '''
        roffset = np.zeros(self.shape[0], dtype=int)
        coffset = np.zeros(self.shape[1], dtype=int)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    rp = self[i, j].GetRowPartArray()
                    if (roffset[i] != 0 and
                            roffset[i] != rp[1] - rp[0]):
                        assert False, 'row partitioning is not consistent'
                    roffset[i] = rp[1] - rp[0]
                    if use_parallel and not isinstance(
                            self[i, j], chypre.CHypreMat):
                        from mpi4py import MPI
                        myid = MPI.COMM_WORLD.rank
                        if myid != 0:
                            roffset[i] = 0
        for j in range(self.shape[1]):
            for i in range(self.shape[0]):
                if self[i, j] is not None:
                    cp = self[i, j].GetColPartArray()
                    if (coffset[j] != 0 and
                            coffset[j] != cp[1] - cp[0]):
                        assert False, 'col partitioning is not consistent'
                    coffset[j] = cp[1] - cp[0]
                    if use_parallel and not isinstance(
                            self[i, j], chypre.CHypreMat):
                        if myid != 0:
                            coffset[i] = 0

        #coffset = [self[0, j].shape[1] for j in range(self.shape[1])]
        if self.complex and convert_real:
            if merge_realimag:
                roffset = roffset * 2
                coffset = coffset * 2
            else:
                if interleave:
                    roffset = np.repeat(roffset, 2)
                    coffset = np.repeat(coffset, 2)
                else:
                    roffset = np.hstack((roffset, roffset))
                    coffset = np.hstack((coffset, coffset))

        roffsets = np.hstack([0, np.cumsum(roffset)])
        coffsets = np.hstack([0, np.cumsum(coffset)])
        return roffsets, coffsets

    def get_local_partitioning_v(self, convert_real=True,
                                 interleave=True, size_hint=None):
        '''
        build matrix in coordinate format
        '''
        roffset = np.zeros(self.shape[0], dtype=int)
        if size_hint is not None:
            hint_shape = size_hint.shape
            # print "called with hint:", hint_shape, self.shape
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    rp = self[i, j].GetPartitioningArray()
                    if (roffset[i] != 0 and
                            roffset[i] != rp[1] - rp[0]):
                        assert False, 'row partitioning is not consistent'
                    roffset[i] = rp[1] - rp[0]
                elif size_hint is not None:
                    for k in range(hint_shape[1]):
                        if size_hint[i, k] is not None:
                            rp = size_hint[i, k].GetPartitioningArray()
                            if (roffset[i] != 0 and
                                    roffset[i] != rp[1] - rp[0]):
                                assert False, 'row partitioning is not consistent'
                            roffset[i] = rp[1] - rp[0]

        #coffset = [self[0, j].shape[1] for j in range(self.shape[1])]
        if self.complex and convert_real:
            if interleave:
                roffset = np.repeat(roffset, 2)
            else:
                roffset = np.hstack((roffset, roffset))

        roffsets = np.hstack([0, np.cumsum(roffset)])
        return roffsets

    def gather_blkvec_interleave(self, size_hint=None, blk_format=None):
        '''
        Construct MFEM::BlockVector

        This routine ordered unkonws in the following order
           Re FFE1, Im FES1, ReFES2, Im FES2, ...
        If self.complex is False, it assembles a nomal block
           vector
        This routine is used together with get_global_blkmat_interleave(self):
        '''
        assert blk_format is None, "block structure merging is not yet implemented"

        roffsets = self.get_local_partitioning_v(convert_real=True,
                                                 interleave=True,
                                                 size_hint=size_hint)
        dprint1("roffsets(vector)", roffsets)
        offset = mfem.intArray(list(roffsets))

        vec = mfem.BlockVector(offset)
        vec._offsets = offset  # in order to keep it from freed

        data = []
        ii = 0
        jj = 0

        # Here I don't like that I am copying the data between two vectors..
        # But, avoiding this takes the large rearangement of program flow...
        for i in range(self.shape[0]):
            if self[i, 0] is not None:
                if isinstance(self[i, 0], chypre.CHypreVec):
                    vec.GetBlock(ii).Assign(self[i, 0][0].GetDataArray())
                    if self.complex:
                        if self[i, 0][1] is not None:
                            vec.GetBlock(ii +
                                         1).Assign(self[i, 0][1].GetDataArray())
                        else:
                            vec.GetBlock(ii + 1).Assign(0.0)
                elif isinstance(self[i, 0], ScipyCoo):
                    arr = np.atleast_1d(self[i, 0].toarray().squeeze())
                    vec.GetBlock(ii).Assign(np.real(arr))
                    if self.complex:
                        vec.GetBlock(ii + 1).Assign(np.imag(arr))
                else:
                    assert False, "not implemented, " + str(type(self[i, 0]))
            else:
                vec.GetBlock(ii).Assign(0.0)
                if self.complex:
                    vec.GetBlock(ii + 1).Assign(0.0)
            ii = ii + 2 if self.complex else ii + 1

        return vec

    def get_global_blkmat_interleave(self, blk_format=None):
        '''
        This routine ordered unkonws in the following order
           FFE1, FES2,
        If it is complex
           Re FFE1, Im FES1, ReFES2, Im FES2, ...
        '''
        assert blk_format is None, "block structure merging is not yet implemented"

        roffsets, coffsets = self.get_local_partitioning(convert_real=True,
                                                         interleave=True)
        dprint1("Generating MFEM BlockMatrix: shape = " +
                str((len(roffsets) - 1, len(coffsets) - 1)))
        dprint1(
            "Generating MFEM BlockMatrix: roffset/coffset = ",
            roffsets,
            coffsets)

        ro = mfem.intArray(list(roffsets))
        co = mfem.intArray(list(coffsets))
        glcsr = mfem.BlockOperator(ro, co)

        ii = 0

        for i in range(self.shape[0]):
            jj = 0
            for j in range(self.shape[1]):
                if self[i, j] is not None:
                    if use_parallel:
                        if isinstance(self[i, j], chypre.CHypreMat):
                            gcsr = self[i, j]
                            cp = self[i, j].GetColPartArray()
                            rp = self[i, j].GetRowPartArray()
                            s = self[i, j].shape
                            # if (cp == rp).all() and s[0] == s[1]:
                            if gcsr[0] is not None:
                                csr = ToScipyCoo(gcsr[0]).tocsr()
                                gcsr[0] = ToHypreParCSR(csr, col_starts=cp)
                            if gcsr[1] is not None:
                                csr = ToScipyCoo(gcsr[1]).tocsr()
                                gcsr[1] = ToHypreParCSR(csr, col_starts=cp)
                                gcsrm = ToHypreParCSR(-csr, col_starts=cp)
                            dprint2(i, j, s, rp, cp)
                        else:
                            assert False, "unsupported block element " + \
                                str(type(self[i, j]))
                    else:
                        if isinstance(self[i, j], ScipyCoo):
                            gcsr = self[i, j].get_mfem_sparsemat()
                            if gcsr[1] is not None:
                                gcsrm = mfem.SparseMatrix(gcsr[1])
                                gcsrm *= -1.
                        else:
                            assert False, "unsupported block element " + \
                                type(self[i, j])

                    glcsr.SetBlock(ii, jj, gcsr[0])
                    if self.complex:
                        glcsr.SetBlock(ii + 1, jj + 1, gcsr[0])
                    if gcsr[1] is not None:
                        glcsr.SetBlock(ii + 1, jj, gcsr[1])
                        glcsr.SetBlock(ii, jj + 1, gcsrm)
                jj = jj + 2 if self.complex else jj + 1
            ii = ii + 2 if self.complex else ii + 1

        return glcsr

    def gather_blkvec_merged(self,
                             size_hint=None,
                             symmetric=False,
                             blk_format=None):
        '''
        Construct MFEM::BlockVector

        This routine ordered unkonws in the following order
           (Re_FES1_node1, Im_FES1_node1, Re_FES1_node2, Im_FES1_node2, ...)

        '''
        assert self.complex, "this format is complex only"

        roffsets = self.get_local_partitioning_v(size_hint=size_hint)
        if size_hint is not None:
            self.cp_size_hint(size_hint)

        rsizes = np.sum(np.diff(roffsets).reshape(-1, 2), 1)
        if blk_format is None:
            blk_format = [(x,) for x in range(len(rsizes))]

        rsizes = [np.sum(rsizes[np.array(x)]) for x in blk_format]
        roffsets = np.hstack([0, np.cumsum(rsizes)])

        dprint1("roffsets(vector)", roffsets)
        offset = mfem.intArray(list(roffsets))

        vec = mfem.BlockVector(offset)
        vec._offsets = offset  # in order to keep it from freed

        data = []
        ii = 0
        jj = 0

        # Here I don't like that I am copying the data between two vectors..
        # But, avoiding this takes the large rearangement of program flow...
        for i, rows in enumerate(blk_format):
            vec_r, vec_i = self.get_merged_submblocks_v(rows, symmetric)
            vv = np.hstack([vec_r, vec_i])
            vec.GetBlock(i).Assign(vv)
        return vec

    def get_merged_submblocks_v(self, rows, symmetric):
        m = len(rows)
        arr_r = []
        arr_i = []

        for i in rows:
            if self[i, 0] is not None:
                if isinstance(self[i, 0], chypre.CHypreVec):
                    rr = self[i, 0][0].GetDataArray()
                    if self[i, 0][1] is not None:
                        if symmetric:
                            ii = -self[i, 0][1].GetDataArray()
                        else:
                            ii = self[i, 0][1].GetDataArray()
                    else:
                        ii = rr * 0
                    arr_r.append(rr)
                    arr_i.append(ii)
                elif isinstance(self[i, 0], ScipyCoo):
                    arr = np.atleast_1d(self[i, 0].toarray().squeeze())
                    arr_r.append(np.real(arr))
                    if symmetric:
                        arr_i.append(-np.imag(arr))
                    else:
                        arr_i.append(np.imag(arr))
                else:
                    assert False, "not implemented, " + str(type(self[i, 0]))
            else:
                arr_r.append(np.zeros((self.rsize[i],)))
                arr_i.append(np.zeros((self.rsize[i],)))
        if m > 1:
            return np.hstack(arr_r), np.hstack(arr_i)
        else:
            return arr_r[0], arr_i[0]

    def get_global_blkmat_merged(self, symmetric=False, blk_format=None):
        '''
        This routine ordered unkonws in the following order
           Re FFE1, Im FES1, ReFES2, Im FES2, ...
        output is ComplexOperator
        '''
        assert self.complex, "this format is complex only"

        roffsets, coffsets = self.get_local_partitioning()

        rsizes = np.sum(np.diff(roffsets).reshape(-1, 2), 1)
        csizes = np.sum(np.diff(coffsets).reshape(-1, 2), 1)

        assert len(rsizes) == len(csizes), "block matrix is not square"
        if blk_format is None:
            blk_format = [(x,) for x in range(len(rsizes))]

        rsizes = [np.sum(rsizes[np.array(x)]) for x in blk_format]
        csizes = [np.sum(csizes[np.array(x)]) for x in blk_format]

        roffsets = np.hstack([0, np.cumsum(rsizes)])
        coffsets = np.hstack([0, np.cumsum(csizes)])
        dprint1("Generating MFEM BlockMatrix: shape = " +
                str((len(roffsets) - 1, len(coffsets) - 1)))
        # nicePrint(
        #    "Generating MFEM BlockMatrix: roffset/coffset = ",
        #    roffsets,
        #    coffsets)

        ro = mfem.intArray(list(roffsets))
        co = mfem.intArray(list(coffsets))
        glcsr = mfem.BlockOperator(ro, co)

        for jjj in range(len(blk_format)):
            for iii in range(len(blk_format)):
                rows = blk_format[iii]
                cols = blk_format[jjj]
                gcsa, gcsb = self.get_merged_submblocks(rows, cols)
                if gcsa is None and gcsb is None:
                    continue

                #Hermitian = False if symmetric else True
                Hermitian = True if symmetric else False
                if use_parallel:
                    gcsr = mfem.ComplexHypreParMatrix(
                        gcsa, gcsb, False, False, Hermitian)

                else:
                    gcsr = mfem.ComplexSparseMatrix(
                        gcsa, gcsb, False, False, Hermitian)

                gcsr._real_operator = gcsa
                gcsr._imag_operator = gcsb

                glcsr.SetBlock(iii, jjj, gcsr)

        return glcsr

    def check_element_sizes(self):
        m = self.shape[0]
        n = self.shape[1]

        rsize = [-1]*m
        csize = [-1]*n
        grsize = [-1]*m
        gcsize = [-1]*n

        self._rp = [None]*m
        self._cp = [None]*m

        for i, j in itertools.product(range(m), range(n)):
            mat = None
            if self[i, j] is None:
                continue
            mat = self[i, j]

            if use_parallel:
                if isinstance(mat, chypre.CHypreMat):
                    cp = mat.GetColPartArray()
                    rp = mat.GetRowPartArray()
                    rs = rp[1] - rp[0]
                    cs = cp[1] - cp[0]

                    if gcsize[j] == -1:
                        cs2 = allgather(cs)
                        cs2 = np.hstack([0, np.cumsum(cs2)])
                        gcsize[j] = cs2[-1]
                        self._cp[j] = cp

                    if grsize[i] == -1:
                        rs2 = allgather(rs)
                        rs2 = np.hstack([0, np.cumsum(rs2)])
                        grsize[i] = rs2[-1]
                        self._rp[i] = rp

                elif isinstance(mat, chypre.CHypreVec):
                    rp = mat.GetPartitioningArray()
                    rs = rp[1] - rp[0]

                    if grsize[i] == -1:
                        rs2 = allgather(rs)
                        rs2 = np.hstack([0, np.cumsum(rs2)])
                        grsize[i] = rs2[-1]
                        self._rp[i] = rp
                    cs = 1
                    gcsize[j] = 1

                else:
                    assert False, "must be CHypreMat or CHypreVec"

            else:
                assert isinstance(mat, ScipyCoo),  "must be ScipyCoo"
                rs, cs = mat.shape
                grsize[i] = rs
                gcsize[j] = cs

            if rsize[i] != -1:
                assert rsize[i] == rs, "Inconsistent Matrix Shape"
            if csize[j] != -1:
                assert csize[j] == cs, "Inconsistent Matrix Shape"
            rsize[i] = rs
            csize[j] = cs

        self._rsize = rsize
        self._csize = csize
        self._grsize = grsize
        self._gcsize = gcsize

    def get_merged_submblocks(self, rows, cols):
        '''
        return subblock. merge them if rows/cols have more than 1 elements
        '''
        m = len(rows)
        n = len(cols)

        if use_parallel:
            if m == 1 and n == 1:
                i, j = rows[0], cols[0]
                mat = self[i, j]
                if mat is None:
                    return None, None
                gcsa = mat[0]
                gcsb = mat[1]
                if gcsa is None:
                    gcsa = empty_hypremat((self.rsize[i], self.gcsize[j]),
                                          self.cp[j])
                if gcsb is None:
                    gcsb = empty_hypremat((self.rsize[i], self.gcsize[j]),
                                          self.cp[j])
            else:
                arr_r = mfem.hypreparmatArray2D(m, n)
                arr_r.Assign(None)
                arr_i = mfem.hypreparmatArray2D(m, n)
                has_real = False
                has_imag = False
                for i, j in itertools.product(range(m), range(n)):
                    ir = rows[i]
                    ic = cols[j]

                    # real part
                    if self[ir, ic] is None:
                        # if ir == ic:
                        empty = empty_hypremat((self.rsize[ir], self.gcsize[ic]),
                                               self.cp[ic])
                        empty.thisown = False
                        arr_r[i, j] = empty
                    elif self[ir, ic][0] is None:
                        # if ir == ic:
                        empty = empty_hypremat((self.rsize[ir], self.gcsize[ic]),
                                               self.cp[ic])
                        empty.thisown = False
                        arr_r[i, j] = empty
                    else:
                        arr_r[i, j] = self[ir, ic][0]

                    #mat = arr_r[i, j]
                    # if mat is not None:
                    #    cp = mat.GetColPartArray()
                    #    rp = mat.GetRowPartArray()
                    #
                    #    nicePrint(i, j, ir, ic, (mat.M(), mat.N()), (mat.Height(), mat.Width()), rp, cp, mat.NumRows(), mat.NumCols())

                    # imag part
                    if self[ir, ic] is None:
                        # if ir == ic:
                        empty = empty_hypremat((self.rsize[ir], self.gcsize[ic]),
                                               self.cp[ic])
                        empty.thisown = False
                        arr_i[i, j] = empty
                    elif self[ir, ic][1] is None:
                        # if ir == ic:
                        empty = empty_hypremat((self.rsize[ir], self.gcsize[ic]),
                                               self.cp[ic])
                        empty.thisown = False
                        arr_i[i, j] = empty
                    else:
                        arr_i[i, j] = self[ir, ic][1]

                gcsa = mfem.HypreParMatrixFromBlocks(arr_r)
                gcsb = mfem.HypreParMatrixFromBlocks(arr_i)
        else:
            if m == 1 and n == 1:
                mat = self[rows[0], cols[0]]
                if mat is None:
                    return None, None

                csra = mat.real.tocsr()
                csrb = mat.imag.tocsr()
                csra.eliminate_zeros()
                csrb.eliminate_zeros()
                gcsa = mfem.SparseMatrix(csra)
                gcsb = mfem.SparseMatrix(csrb)
            else:
                offset1 = [self.rsize[i] for i in rows]
                offset2 = [self.csize[i] for i in cols]
                offset1 = mfem.intArray(
                    list(np.hstack([0, np.cumsum(offset1)])))
                offset2 = mfem.intArray(
                    list(np.hstack([0, np.cumsum(offset2)])))

                mr = mfem.BlockMatrix(offset1, offset2)
                mi = mfem.BlockMatrix(offset1, offset2)

                for i, j in itertools.product(range(m), range(n)):
                    ir = rows[i]
                    ic = cols[j]
                    if self[ir, ic] is None:
                        continue
                    csra = self[ir, ic].real.tocsr()
                    csrb = self[ir, ic].imag.tocsr()
                    csra.eliminate_zeros()
                    csrb.eliminate_zeros()
                    gcsa = mfem.SparseMatrix(csra)
                    gcsb = mfem.SparseMatrix(csrb)
                    mr.SetBlock(i, j, gcsa)
                    mi.SetBlock(i, j, gcsb)

                gcsa = mr.CreateMonolithic()
                gcsb = mi.CreateMonolithic()

        return gcsa, gcsb


def empty_hypremat(local_size, cp):
    csr = scipy.sparse.csr_matrix(
        (local_size[0], local_size[1]), dtype=np.float64)
    cp3 = [cp[0], cp[1], local_size[1]]
    return ToHypreParCSR(csr, col_starts=cp3)
