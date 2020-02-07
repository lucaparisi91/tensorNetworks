import ten
import numpy as np
np_c=ten.np_c
charges=ten.charges

class mpo:
    def __init__(self,tensors,Q):
         self.tensors=[tensor for tensor in tensors]
         for tensor in tensors:
             tensor.iset_leg_labels(["vOL","pT","pB","vOR"])
         
         self.leftBond=np_c.Array([self.tensors[0].legs[0].conj()])
         self.leftBond[:]=np.zeros( (self.tensors[0].shape[0])) + 1
         
         self.leftBond.iset_leg_labels(["vOR"])

         
         self.rightBond=np_c.Array([self.tensors[-1].legs[-1].conj()],qtotal=[Q])
         self.rightBond.iset_leg_labels(["vOL"])
         
         
         self.rightBond[:]=np.zeros( (self.tensors[-1].shape[-1])) + 1
         
    def __getitem__(self,i):
        return self.tensors[i]
    def __setitem__(self,i,value):
        self.tensors[i]=value
        
    def nSites(self):
        return len(self.tensors)
