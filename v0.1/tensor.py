import numpy as np
import matplotlib.pylab as plt
import time

class numpyBackend:
    def _updateData(self):
        self.shape=self.data.shape
        
    def __init__(self,shape=None,data=None):
        if data is None:
            self.data=np.zeros(shape)
        else:
            self.data=data
        self._updateData()
    
    def __getitem__(self,index):
        return self.data[index]
    def __setitem__(self,index,item):
        self.data[index]=item
    def reshape(self,newShape):
        self.data=np.reshape(self.data,newShape)
    def contract(self,tensor2,indexPair):
        data=np.tensordot(self.data,tensor2.data,axes=indexPair)
        return numpyBackend(data=data)
    def random(self):
        self.data=np.random.rand( self.data.size ).reshape(self.shape)
    def transpose(self):
        return numpyBackend(data=self.data.transpose())
    def svd(self,Ub,Sb,Vb,centerIndex,truncation=None):
        '''
        Saves the SVD decompositions in three different numpy backends Ub, Sb, Vb
        Truncate up to truncation states in bond dimension
        '''
        newShape=( np.prod(self.shape[0:centerIndex]), np.prod(self.shape[centerIndex:]) )
        data=np.reshape(self.data,newShape)
        U,S,V=np.linalg.svd(data,full_matrices=False)
        UShape=list(self.shape[0:centerIndex]) + [S.shape[0]]
        U=U.reshape(UShape)
        V=V.reshape( [S.shape[0]] +   list(self.shape[centerIndex:])) 
        Ub.data=U
        Sb.data=np.diag(S)
        Vb.data=V
        Ub._updateData()
        Vb._updateData()
        Sb._updateData()

class leg:
    def __init__(self,leftTensor,leftIndex,rightTensor=None,rightIndex=None):
        self.leftTensor=leftTensor
        self.leftIndex=leftIndex
        self.rightTensor=rightTensor
        self.rightIndex=rightIndex
        
    def __repr__(self):
        s="Left: " + repr(self.leftTensor) + ","
        s+="Right: " + repr(self.rightTensor)
        return s
    def connect(self,leg):
        self.rightTensor=leg.leftTensor
        self.rightIndex=leg.leftIndex
        self.rightTensor.legs[self.rightIndex]=self
    def isDangling(self):
        if self.rightTensor is None:
            return True
        else:
            return False
        
class tensor:
    def __init__(self,shape=None,backendTensor=None,name=None):
        self
        if backendTensor is not None:
            self.backendTensor=backendTensor
            shape=self.backendTensor.shape
        else:
            self.backendTensor=numpyBackend(shape)
        self._size=np.prod(shape)
        self.reshape(shape)
        self.legs=[leg(leftTensor=self,leftIndex=i) for i in range(self.rank())]
    def __len__(self):
        return self._size
    def reshape(self,shape):
        self._shape=shape
        self._rank=len(shape)
        self.backendTensor.reshape(shape)
    def rank(self):
        return self._rank
    def size(self):
        return self._size
    def shape(self):
        return self._shape
    def __repr__(self):
        s="[Shape=" + str(self.shape() ) + ",size=" + str(len(self))+ "]"
        return s
    def __getitem__(self,index):
        return self.backendTensor[index]
    def __setitem__(self,index,item):
        self.backendTensor[index]=item
    def __str__(self):
        return str(self.backendTensor.data)
    def random(self):
        self.backendTensor.random()
    def _update(self):
        self._shape=self.backendTensor.shape
        self._size=np.prod(self._shape)
        self._rank=len(self._shape)
    def transpose(self):
        return tensor(data=self.backendTensor.transpose())
    def plot(self):
        pass    

def contract(leg):
    leftTB=leg.leftTensor.backendTensor
    rightTB=leg.rightTensor.backendTensor
    indexPair=(leg.leftIndex,leg.rightIndex)
    data=leftTB.contract(rightTB,indexPair=indexPair)
    t=tensor(backendTensor=data)
    return t

def orthogonalizeRight(leg):
    A=leg.leftTensor.backendTensor
    B=leg.rightTensor.backendTensor
    S=numpyBackend()
    V=numpyBackend()
    A.svd(A,S,V,leg.leftIndex)
    #leg.rightTensor.backendTensor=V.contract(B,(1,leg.rightIndex))
    R=S.contract(V,(1,0))
    leg.rightTensor.backendTensor=R.contract(B,(1,leg.rightIndex))
    leg.leftTensor._update()
    leg.rightTensor._update()
    
def orthogonalizeLeft(leg):
    A=leg.leftTensor.backendTensor
    B=leg.rightTensor.backendTensor
    S=numpyBackend()
    U=numpyBackend()
    B.svd(U,S,B,leg.rightIndex+1)
    R=U.contract(S,(1,0))
    leg.leftTensor.backendTensor=A.contract(R,(leg.leftIndex,0))
    leg.leftTensor._update()
    leg.rightTensor._update()



