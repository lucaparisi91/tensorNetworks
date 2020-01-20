from tensor import *

class mps:
    '''
    Creates an mps state
    d: physical basis
    D: bond dimensions
    L : number of sites
    '''
    
    def __init__(self,d,L,D=100):
        self.d=d
        self.L=L
        
        self.tensors=[tensor((D,d,D)) for i in range(L-2)]
        self.tensors=[tensor((d,D))] + self.tensors + [tensor((D,d))]
        self.orthogonalized=False
        self.center=None
        
        for i in range(1,L-1):
            self.tensors[i].legs[2].connect(self.tensors[i+1].legs[0])
        self.tensors[0].legs[1].connect(self.tensors[1].legs[0])
    def __getitem__(self,index):
        return self.tensors[index]
    def orthogonalizeRight(self):
        orthogonalizeRight(self.tensors[0].legs[1])
        for i in range(1,self.L-1):
            orthogonalizeRight(self.tensors[i].legs[2])
        self.center=self.L-1
        self.orthogonalized=True
    def orthogonalizeLeft(self):
        for i in range(1,self.L):
            orthogonalizeLeft(self.tensors[i].legs[0])
        self.orthogonalized=True
        self.center=0
    def shiftOrthogonalizationCenter(self):
        
        
