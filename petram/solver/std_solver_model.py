import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdSolver')
rprint = debug.regular_print('StdSolver')

class StdSolver(Solver):
    can_delete = True
    has_2nd_panel = False

    def attribute_set(self, v):
        super(StdSolver, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        return [["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                ["clear working directory",
                 self.clear_wdir,  3, {"text":""}],
                ["initialize solution only",
                 self.init_only,  3, {"text":""}], 
                ["convert to real matrix (complex prob.)",
                 self.assemble_real,  3, {"text":""}],
                ["save parallel mesh",
                 self.save_parmesh,  3, {"text":""}],
                ["use cProfiler",
                 self.use_profiler,  3, {"text":""}],]

    def get_panel1_value(self):
        return (self.init_setting,
                self.phys_model,
                self.clear_wdir,
                self.init_only,               
                self.assemble_real,
                self.save_parmesh,
                self.use_profiler)        
    
    def import_panel1_value(self, v):
        self.init_setting = str(v[0])        
        self.phys_model = str(v[1])
        self.clear_wdir = v[2]
        self.init_only = v[3]        
        self.assemble_real = v[4]
        self.save_parmesh = v[5]
        self.use_profiler = v[6]                

    def get_editor_menus(self):
        return []
    
    def get_possible_child(self):
        choice = []
        try:
            from petram.solver.mumps_model import MUMPS
            choice.append(MUMPS)
        except ImportError:
            pass

        try:
            from petram.solver.gmres_model import GMRES
            choice.append(GMRES)
        except ImportError:
            pass

        try:
            from petram.solver.strumpack_model import SpSparse
            choice.append(SpSparse)
        except ImportError:
            pass
        return choice

    def allocate_solver_instance(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = StandardSolver(self, engine)
        return instance
    
    @debug.use_profiler
    def run(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = StandardSolver(self, engine)

        # We dont use probe..(no need...)
        #instance.configure_probes(self.probe)

        #self.init_only = True
        finished = instance.init(self.init_only)
        while not finished:
            finished = instance.solve()

        instance.save_solution(ksol = 0,
                               skip_mesh = False, 
                               mesh_only = False,
                               save_parmesh=self.save_parmesh)
        
        print(debug.format_memory_usage())

from petram.solver.solver_model import SolverInstance

class StandardSolver(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
         
    def init(self, init_only=False):
        def get_matrix_weight(timestep_config, timestep_weight):
            return [1, 0, 0]            
        
        engine = self.engine
                      
        phys_target = self.get_phys()
        num_matrix= engine.run_set_matrix_weight(phys_target, get_matrix_weight)
        
        engine.set_formblocks(phys_target, num_matrix)
        
        for p in phys_target:
            engine.run_mesh_extension(p)
        
        engine.run_alloc_sol(phys_target)
        
        inits = self.get_init_setting()
        if len(inits) == 0:
            # in this case alloate all fespace and initialize all
            # to zero
            engine.run_apply_init(phys_target, 0)
        else:
            for init in inits:
                init.run(engine)
        engine.run_apply_essential(phys_target)
        
        self.assemble()
        A, X, RHS, Ae, B, M = self.blocks        
        self.sol = X[0]
        
        if init_only:
            self.save_solution()            
            return True
        else:
            return False

    def compute_A(self, M, B, X):
        '''
        M[0] x = B
        '''
        return M[0]
    
    def compute_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        return B

    def assemble(self):
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_verify_setting(phys_target, self.gui)
        engine.run_assemble_mat(phys_target)
        engine.run_assemble_rhs(phys_target)
        self.blocks = self.engine.run_assemble_blocks(self.compute_A, self.compute_rhs)
        #A, X, RHS, Ae, B, M = blocks
        self.assembled = True
        
    def assemble_rhs(self):
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_assemble_rhs(phys_target)
        B = self.engine.run_update_B_blocks()
        blocks = list(self.blocks)
        blocks[4] = B
        self.blocks = tuple(blocks)
        self.assembled = True

    def solve(self):
        engine = self.engine

        if not self.assembled:
            assert False, "assmeble must have been called"
            
        A, X, RHS, Ae, B, M = self.blocks        
        AA = engine.finalize_matrix(A, not self.phys_real, format = self.ls_type)
        BB = engine.finalize_rhs([RHS], not self.phys_real, format = self.ls_type)

        linearsolver = self.allocate_linearsolver(AA.dtype == 'complex')

        linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)        
        solall = linearsolver.Mult(BB, case_base=0)
        
        #linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        #solall = linearsolver.Mult(BB, case_base=0)
            
        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_model.real_to_complex(solall, AA)
        
        self.sol = A.reformat_central_mat(solall, 0)
        return True


        
        
