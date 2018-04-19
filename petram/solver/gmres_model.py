from .solver_model import LinearSolverModel, LinearSolver
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GMRESModel')

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   from mfem.common.parcsr_extra import *
   import mfem.par as mfem
   default_kind = 'hypre'
   
   from mpi4py import MPI                               
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   smyid = '{:0>6d}'.format(myid)   
else:
   import mfem.ser as mfem
   default_kind = 'scipy'
   
SparseSmootherCls = {"Jacobi": ("DSmoother", 0),
                     "l1Jacobi": ("DSmoother", 1),
                     "lumpedJacobi": ("DSmoother", 2),
                     "GS": ("GSSmoother", 0),
                     "forwardGS": ("GSSmoother", 1),
                     "backwardGS": ("GSSmoother", 2),}                                          

class GMRES(LinearSolverModel):
    has_2nd_panel = False
    accept_complex = False
    def init_solver(self):
        pass
    
    def panel1_param(self):
        from petram.pi.widget_smoother import WidgetSmoother       
        return [["log_level",   self.log_level,  400, {}],
                ["max  iter.",  self.maxiter,  400, {}],
                ["rel. tol",    self.reltol,  300,  {}],
                ["abs. tol.",   self.abstol,  300, {}],
                ["restart(kdim)", self.kdim,     400, {}],
                [None, None, 99, {"UI":WidgetSmoother, "span":(1,2)}],
                ["write matrix",  self.write_mat,   3, {"text":""}],]     
    
    def get_panel1_value(self):
        num_matrix = self.parent.get_num_matrix() # this will set _mat_weight
        all_dep_vars = self.root()['Phys'].all_dependent_vars(num_matrix)
        
        names = [x[0] for x in  self.preconditioners]
        for n in all_dep_vars:
           if not n in names:
              self.preconditioners.append((n, ['None', 'None']))
        return (long(self.log_level), long(self.maxiter),
                self.reltol, self.abstol, long(self.kdim),
                self.preconditioners, self.write_mat)
    
    def import_panel1_value(self, v):
        self.log_level = long(v[0])
        self.maxiter = long(v[1])
        self.reltol = v[2]
        self.abstol = v[3]
        self.kdim = long(v[4])
        self.preconditioners = v[5]
        self.write_mat = bool(v[6])
        
    def attribute_set(self, v):
        v = super(GMRES, self).attribute_set(v)
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol']  = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] =   50
        v['printit'] = 1
        v['preconditioner'] = ''
        v['preconditioners'] = []        
        v['write_mat'] = False        
        return v
    
    def verify_setting(self):
        if not self.parent.assemble_real:
            for phys in self.get_phys():
                if phys.is_complex():
                    return False, "GMRES does not support complex.", "A complex problem must be converted to a real value problem"
        return True, "", ""

    def linear_system_type(self, assemble_real, phys_complex):
        #if not phys_complex: return 'block'
        return 'blk_interleave'
        #return None


    def real_to_complex(self, solall, M):
        if use_parallel:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank


           offset = M.RowOffsets().ToList()
           of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o))) for o in offset]
           if myid != 0: return           

        else:
           offset = M.RowOffsets()
           of = offset.ToList()
           
        rows = M.NumRowBlocks()
        s = solall.shape
        nb = rows/2
        i = 0
        pt = 0
        result = np.zeros((s[0]/2, s[1]), dtype='complex')
        for j in range(nb):
           l = of[i+1]-of[i]
           result[pt:pt+l,:] = (solall[of[i]:of[i+1],:]
                             +  1j*solall[of[i+1]:of[i+2],:])
           i = i+2
           pt = pt + l

        return result

    def allocate_solver(self, datatype='D'):
        solver = GMRESSolver(self, int(self.maxiter),
                             self.abstol, self.reltol, int(self.kdim))

        #solver.AllocSolver(datatype)
        return solver

class GMRESSolver(LinearSolver):
    is_iterative = True
    def __init__(self, gui, maxiter, abstol, reltol, kdim):
        self.maxiter = maxiter
        self.abstol = abstol
        self.reltol = reltol
        self.kdim = kdim
        LinearSolver.__init__(self, gui)

    def SetOperator(self, opr, dist=False, name = None):
        self.Aname = name
        self.A = opr                     
                             
    def Mult(self, b, x=None, case_base=0):
        if use_parallel:
            return self.solve_parallel(self.A, b, x)
        else:
            return self.solve_serial(self.A, b, x)
                             
    def solve_parallel(self, A, b, x=None):
        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank
        nproc    = MPI.COMM_WORLD.size
        from petram.helper.mpi_recipes import gather_vector
        
        def get_block(Op, i, j):
            try:
               return Op._linked_op[(i,j)]
            except KeyError:
               return None

        offset = A.RowOffsets()
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()
        
        if self.gui.write_mat:
           for i in range(cols):
              for j in range(rows):
                 m = get_block(A, i, j)
                 if m is None: continue
                 m.Print('matrix_'+str(i)+'_'+str(j))
           for i, bb  in enumerate(b):
              for j in range(rows):
                 v = bb.GetBlock(j)
                 v.Print('rhs_'+str(i)+'_'+str(j)+'.'+smyid)

        M = mfem.BlockDiagonalPreconditioner(offset)
        
        prcs = dict(self.gui.preconditioners)
        name = self.Aname
        if self.gui.parent.is_complex():
           name = sum([[n, n] for n in name], [])
           
        for k, n in enumerate(name):
           prc = prcs[n][1]
           if prc == "None": continue
           A0 = get_block(A, k, k)
           invA0 = mfem.HypreSmoother(A0)
           invA0.SetType(getattr(mfem.HypreSmoother, prc))
           invA0.iterative_mode = False
           M.SetDiagonalBlock(k, invA0)
           
        '''
        We should support Shur complement type preconditioner
        if offset.Size() > 2:
            B =  get_block(A, 1, 0)
            MinvBt = get_block(A, 0, 1)
            #Md = mfem.HypreParVector(MPI.COMM_WORLD,
            #                        A0.GetGlobalNumRows(),
            #                        A0.GetRowStarts())
            Md = mfem.Vector()
            A0.GetDiag(Md)
            MinvBt.InvScaleRows(Md)
            S = mfem.ParMult(B, MinvBt)
            invS = mfem.HypreBoomerAMG(S)
            invS.iterative_mode = False
            M.SetDiagonalBlock(1, invS)
        '''
        maxiter = int(self.maxiter)
        atol = self.abstol
        rtol = self.reltol
        kdim = int(self.kdim)
        printit = 1

        sol = []

        solver = mfem.GMRESSolver(MPI.COMM_WORLD)
        solver.SetKDim(kdim)
        #solver = mfem.MINRESSolver(MPI.COMM_WORLD)
        solver.SetAbsTol(atol)
        solver.SetRelTol(rtol)
        solver.SetMaxIter(maxiter)
        solver.SetOperator(A)
        solver.SetPreconditioner(M)
        solver.SetPrintLevel(1)

        # solve the problem and gather solution to head node...
        # may not be the best approach

        for bb in b:
           rows = MPI.COMM_WORLD.allgather(np.int32(bb.Size()))
           rowstarts = np.hstack((0, np.cumsum(rows)))
           dprint1("rowstarts/offser",rowstarts, offset.ToList())
           xx = mfem.BlockVector(offset)
           if x is None:
              xx.Assign(0.0)
           else:
              xx = x
              #for j in range(cols):
              #   dprint1(x.GetBlock(j).Size())
              #   dprint1(x.GetBlock(j).GetDataArray())
              #assert False, "must implement this"
           solver.Mult(bb, xx)
           s = []
           for i in range(offset.Size()-1):
               v = xx.GetBlock(i).GetDataArray()
               vv = gather_vector(v)
               if myid == 0:
                   s.append(vv)
               else:
                   pass
           if myid == 0:               
               sol.append(np.hstack(s))
        if myid == 0:                              
            sol = np.transpose(np.vstack(sol))
            return sol
        else:
            return None
        
    def solve_serial(self, A, b, x=None):

        def get_block(Op, i, j):
            try:
               return Op._linked_op[(i,j)]
            except KeyError:
               return None
                      
        offset = A.RowOffsets()
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()
        if self.gui.write_mat:
           for i in range(cols):
              for j in range(rows):
                 m = get_block(A, i, j)
                 if m is None: continue
                 m.Print('matrix_'+str(i)+'_'+str(j))
           for i, bb  in enumerate(b):
              for j in range(rows):
                 v = bb.GetBlock(j)
                 v.Print('rhs_'+str(i)+'_'+str(j))

        M = mfem.BlockDiagonalPreconditioner(offset)

        prcs = dict(self.gui.preconditioners)
        name = self.Aname
        if self.gui.parent.is_complex():
           name = sum([[n, n] for n in name], [])

           
        '''   
        this if block does a generalized version of this...
        M1 = mfem.GSSmoother(get_block(A, 0, 0))
        M1.iterative_mode = False
        M.SetDiagonalBlock(0, M1)
        '''
        for k, n in enumerate(name):
           
           prc = prcs[n][0]
           if prc == "None": continue
           A0 = get_block(A, k, k)
           cls = getattr(mfem, SparseSmootherCls[prc][0])
           arg = SparseSmootherCls[prc][1]           
           invA0 = cls(A0, arg)
           invA0.iterative_mode = False
           M.SetDiagonalBlock(k, invA0)

        '''
        We should support Shur complement type preconditioner
        if offset.Size() > 2:
            B =  get_block(A, 1, 0)
            MinvBt = get_block(A, 0, 1)
            Md = mfem.Vector(get_block(A, 0, 0).Height())
            get_block(A, 0, 0).GetDiag(Md)
            for i in range(Md.Size()):
                if Md[i] != 0.:
                    MinvBt.ScaleRow(i, 1/Md[i])
                else:
                    assert False, "diagnal element of matrix is zero"
            S = mfem.Mult(B, MinvBt)
            S.iterative_mode = False
            SS = mfem.DSmoother(S)
            SS.iterative_mode = False            
            M.SetDiagonalBlock(1, SS)
        '''

        '''
        int GMRES(const Operator &A, Vector &x, const Vector &b, Solver &M,
          int &max_iter, int m, double &tol, double atol, int printit)
        '''
        maxiter = int(self.maxiter)
        atol = self.abstol
        rtol = self.reltol
        kdim = int(self.kdim)
        printit = 1

        sol = []

        solver = mfem.GMRESSolver()
        solver.SetKDim(kdim)
        #solver = mfem.MINRESSolver()
        solver.SetAbsTol(atol)
        solver.SetRelTol(rtol)
        solver.SetMaxIter(maxiter)

        solver.SetOperator(A)
        solver.SetPreconditioner(M)
        solver.SetPrintLevel(1)

        for bb in b:
           if x is None:           
              xx = mfem.Vector(bb.Size())
              xx.Assign(0.0)
           else:
              xx = x
              #for j in range(cols):
              #   print x.GetBlock(j).Size()
              #   print x.GetBlock(j).GetDataArray()                 
              #assert False, "must implement this"
           solver.Mult(bb, xx)
           sol.append(xx.GetDataArray().copy())
        sol = np.transpose(np.vstack(sol))
        return sol
    

         
        
