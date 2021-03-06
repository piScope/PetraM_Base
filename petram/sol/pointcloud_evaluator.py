import numpy as np
import parser
import scipy
import six
import weakref
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.sol.evaluator_agent import EvaluatorAgent
from petram.sol.bdr_nodal_evaluator import process_iverts2nodals
from petram.sol.bdr_nodal_evaluator import eval_at_nodals, get_emesh_idx

class PointcloudEvaluator(EvaluatorAgent):
    def __init__(self, attrs, pc_type=None, pc_param=None):
        '''
           attrs = [1,2,3]
           plane = [ax, ay, az, c]

           cut-plane is defined as
           ax * x + ay * y + ax * z + c = 0
        '''
        super(PointcloudEvaluator, self).__init__()
        self.attrs = set(attrs)
        self.pc_type  = pc_type
        self.pc_param = pc_param
        
    def preprocess_geometry(self, attrs, emesh_idx=0, pc_type=None,
                            pc_param=None):

        from petram.helper.geom import generate_pc_from_cpparam

        self.attrs = attrs
        if pc_param is not None:
            pc_param = self.pc_param
            pc_type =  self.pc_type
            
        if pc_type == 'cutplane': # cutplane
            param = {"origin": pc_param[0], "e1":pc_param[1], "e2":pc_param[2],
                     "x":pc_param[3], "y":pc_param[4]}
            points = generate_pc_from_cpparam(**param)
            
        elif pc_type == 'line': 
            sp = np.array(pc_param[0])
            ep = np.array(pc_param[1])
            num = pc_param[2]
            
            ii = np.linspace(0, 1., num)
            points = np.vstack([ sp * (1-i) + ep * i  for i in ii])

        elif pc_type == 'xyz':
            points = pc_param


        self.ans_shape = points.shape
        self.ans_points = points
        self.points = points.reshape(-1, points.shape[-1])

        mesh = self.mesh()[emesh_idx]

        v = mfem.Vector()
        mesh.GetVertices(v)
        vv = v.GetDataArray()
        vv = vv.reshape(3, -1)
        max_mesh_ptx = np.max(vv, 1)
        min_mesh_ptx = np.min(vv, 1)

        max_ptx = np.max(self.points, 0)
        min_ptx = np.min(self.points, 0)

        out_of_range = False

        for i in range(len(max_mesh_ptx)):
           if max_mesh_ptx[i] < min_ptx[i]: out_of_range = True
           if min_mesh_ptx[i] > max_ptx[i]: out_of_range = True

        if out_of_range:
            counts = 0
            elem_ids = np.zeros(len(self.points), dtype=int)-1
            int_points = [None]*len(self.points)
            print("skipping mesh")
        else:
            counts, elem_ids, int_points = mesh.FindPoints(self.points, warn=False)
            print("FindPoints found " + str(counts) + " points")
        attrs = [ mesh.GetAttribute(id) if id != -1 else -1 for id in elem_ids]
        attrs = np.array([ i if i in self.attrs else -1 for i in attrs])

        self.elem_ids = elem_ids
        self.masked_attrs = attrs
        self.int_points = int_points
        self.counts = counts

        idx = np.where(attrs != -1)[0]
        self.locs = self.points[idx]

        self.valid_idx = idx
        self.emesh_idx = emesh_idx
        self.knowns = WKD()        

    def eval_at_points(self, expr, solvars, phys):
        from petram.helper.variables import Variable, var_g, NativeCoefficientGenBase, CoefficientVariable
    
        variables = []
        st = parser.expr(expr)
        code= st.compile('<string>')
        names = code.co_names

        g = {}
        #print solvars.keys()
        for key in phys._global_ns.keys():
           g[key] = phys._global_ns[key]
        for key in solvars.keys():
           g[key] = solvars[key]

        ll_name = []
        ll_value = []
        var_g2 = var_g.copy()

        new_names = []
        name_translation = {}
        for n in names:
           if (n in g and isinstance(g[n], NativeCoefficientGenBase)):
               g[n+"_coeff"] = CoefficientVariable(g[n], g)
               new_names.append(n+"_coeff")
               name_translation[n+"_coeff"] = n

           if (n in g and isinstance(g[n], Variable)):
               new_names.extend(g[n].dependency)
               new_names.append(n)
               name_translation[n] = n
           elif n in g:
               new_names.append(n)
               name_translation[n] = n

        for n in new_names:
           if (n in g and isinstance(g[n], Variable)):
               if not g[n] in self.knowns:
                  self.knowns[g[n]] = g[n].point_values(counts = self.counts,
                                        locs = self.locs,
                                        attrs = self.masked_attrs,
                                        elem_ids = self.elem_ids,
                                        mesh = self.mesh()[self.emesh_idx],
                                        int_points = self.int_points,
                                        g = g,
                                        knowns = self.knowns)

               #ll[n] = self.knowns[g[n]]
               ll_name.append(name_translation[n])
               ll_value.append(self.knowns[g[n]])
           elif (n in g):
               var_g2[n] = g[n]

        if len(ll_value) > 0:
            val = np.array([eval(code, var_g2, dict(zip(ll_name, v)))
                        for v in zip(*ll_value)])
        else:
            # if expr does not involve Varialbe, evaluate code once
            # and generate an array 
            val = np.array([eval(code, var_g2)]*len(self.locs))
        return val


        
    def eval(self, expr, solvars, phys):
        from petram.sol.bdr_nodal_evaluator import get_emesh_idx
        
        emesh_idx = get_emesh_idx(self, expr, solvars, phys)
        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"

        if len(emesh_idx) == 1:
            if self.emesh_idx != emesh_idx[0]:
                self.preprocess_geometry(self.attrs, emesh_idx=emesh_idx[0],
                                         pc_type = self.pc_type, 
                                         pc_param = self.pc_param)

        if self.counts == 0:
            return None, None, None
        
        val = self.eval_at_points(expr, solvars, phys)
        
        if val is None:
            return None, None, None

        shape = self.ans_shape[:-1]
        attrs = self.masked_attrs.reshape(shape)
        
        return self.ans_points,  val,  attrs

        
    

