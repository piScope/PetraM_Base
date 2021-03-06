import traceback
import numpy as np

from petram.model import Domain, Bdry, Point, Pair
from petram.phys.phys_model import Phys, PhysModule, VectorPhysCoefficient, PhysCoefficient

# define variable for this BC.
from petram.phys.vtable import VtableElement, Vtable

class InitValue(PhysCoefficient):
   def EvalValue(self, x):
       v = super(InitValue, self).EvalValue(x)
       if self.real:  val = v.real
       else: val =  v.imag
       return val
   
class InitValueV(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(Einit_p, self).EvalValue(x)
       v = np.array((v[0], v[2]))
       if self.real:  val = v.real
       else: val =  v.imag
       return val

class Coeff2D_common(object):   
    @property
    def vt3(self):
        names = self.get_root_phys().dep_vars_base
        names2 = [n + '_init' for n in names]
        data = []
        if hasattr(self, '_vt3'):
            vt3 = self._vt3
            if vt3.keys() == names2: return vt3
            
        data = [(n+'_init', VtableElement(n+'_init',
                                          type='float', guilabel = n+'(init)',
                                          default = 0.0, tip = "initial value",
                                          chkbox = True)) for n in names]
        self._vt3 = Vtable(data)
        self.update_attribute_set()
        return self._vt3
    
    def get_init_coeff(self, engine, real=True, kfes=0):
        names = self.get_root_phys().dep_vars_base
        
        if not getattr(self, 'use_'+names[kfes]+'_init'): return
        
        f_name = self.vt3.make_value_or_expression(self)
        if len(f_name) == 0:
            coeff = InitValue(f_name[kfes],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
        else:
            coeff = InitValueV(len(f_name), f_name,
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
            
        return self.restrict_coeff(coeff, engine)

class Coeff2D_Domain(Coeff2D_common, Domain, Phys):
    has_3rd_panel = True    
    def __init__(self, **kwargs):
        super(Coeff2D_Domain, self).__init__(**kwargs)
        Domain.__init__(self, **kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
    
        
class Coeff2D_Bdry(Coeff2D_common, Bdry, Phys):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(Coeff2D_Bdry, self).__init__(**kwargs)        
        Bdry.__init__(self, **kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

class Coeff2D_Point(Coeff2D_common, Point, Phys):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(Coeff2D_Point, self).__init__(**kwargs)        
        Point.__init__(self, **kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        Point.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
