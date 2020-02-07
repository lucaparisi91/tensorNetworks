import ten
import numpy as np
np_c=ten.np_c
charges=ten.charges

# generates an mps for the ansatz wavefunction
class mps:
    
    def __init__(self,tensors,Q):
        self.tensors=[tensor for tensor in tensors]
        for tensor in tensors:
             tensor.iset_leg_labels(["vL","p","vR"])
         
        self.orthogonalized=False
        self.orthogonalizationCenter=None
        self.L=len(self.tensors)
        
    
        self.leftBond=np_c.Array([self.tensors[0].legs[0].conj()])
        self.leftBond[:]=np.zeros( (self.tensors[0].shape[0]))
        self.leftBond[0]=1
        self.leftBond.iset_leg_labels(["vR"])
        
        
        self.rightBond=np_c.Array([self.tensors[-1].legs[-1].conj()],qtotal=[Q])
        self.rightBond.iset_leg_labels(["vL"])
        self.rightBond[:]=np.zeros( (self.tensors[-1].shape[-1]))
        self.rightBond[-1]=1

        
        self.normalized=False
        
    def shiftOrthogonalizationCenter(self,direction):
        if direction=="left":
            ten.shiftOrthogonalizationCenter(self.tensors[self.orthogonalizationCenter-1],self.tensors[self.orthogonalizationCenter],direction="left")
            self.orthogonalizationCenter-=1
        elif direction=="right":
            ten.shiftOrthogonalizationCenter(self.tensors[self.orthogonalizationCenter],self.tensors[self.orthogonalizationCenter+1],direction="right")
            self.orthogonalizationCenter+=1
    def normalize(self):
        self.orthogonalizationCenter=self.L-1
        for i in range(self.L-1):
            self.shiftOrthogonalizationCenter("left")


    def __getitem__(self,i):
        return self.tensors[i]
    def __setitem__(self,i,value):
        self.tensors[i]=value
        
    def nSites(self):
        return len(self.tensors)



def splitMPS(A,direction="right",trunc_par={}):

    A=A.combine_legs(["vL","p1"],qconj=1)
    A=A.combine_legs(["p2","vR"],qconj=-1)
    U,S,V,err,rFactor=ten.truncation.svd_theta(A,trunc_par=trunc_par)
    
    S=np_c.diag(S,leg=U.legs[1].conj())
    if direction=="right":
        V=np_c.tensordot(S,V,axes=[1,0])
    elif direction=="left":
        U=np_c.tensordot(U,S,axes=[1,0])
    U=U.split_legs([0])
    V=V.split_legs([1])
    U.iset_leg_labels(["vL","p","vR"])
    V.iset_leg_labels(["vL","p","vR"])
    return (U,V,err)

def norm(state):
    B=state[-1]
    B=np_c.tensordot(B,state.rightBond,axes=["vR","vL"])
    
    B=np_c.tensordot(B,B.conj(),axes=["p","p*"])
    i=state.nSites()-2
    for i in range(state.nSites()-2,0,-1):
        B=np_c.tensordot(state[i],B,axes=["vR","vL"])
        B=np_c.tensordot(B,state[i].conj(),axes=["vL*","vR*"])
        B=np_c.trace(B,"p","p*")
        
    #B=np_c.tensordot(state.leftBond,B,axes=["vR","vL"])
    #B=np_c.tensordot(B,state.leftBond.conj(),axes=["vL*","vR*"])
    B=np_c.trace(B,"vL","vL*")
    return B
