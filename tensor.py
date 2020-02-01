import numpy as np
import matplotlib.pylab as plt
import time
import scipy
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
from math import *
import copy
import itertools
import bisect

class numpyTensor:
    def _updateData(self):
        self.shape=self.data.shape
        
    def __init__(self,shape=None,data=None,dtype=complex):
        if data is None:
            self.data=np.zeros(shape,dtype=complex)
        else:
            self.data=data
        self.dtype=dtype
    def rank(self):
        return len(self.data.shape)
    def size(self):
        return self.data.size
    def __getitem__(self,index):
        return self.data[index]
    def __setitem__(self,index,item):
        self.data[index]=item
    def reshape(self,newShape):
        self.data=np.reshape(self.data,newShape)
    def contract(self,tensor2,indexPair):
        data=np.tensordot(self.data,tensor2.data,axes=indexPair)
        return numpyTensor(data=data)
    def random(self):
        shape=self.data.shape
        self.data=np.random.rand( self.data.size ).reshape(shape).astype(self.dtype)
        if self.dtype is complex:
            self.data+=1j*np.random.rand( self.data.size ).reshape(shape).astype(self.dtype)
        
    def mergeIndicesBetween(self,i1,i2):
        shape=tuple(self.data.shape)
        newShape=tuple(shape[0:i1])
        newShape+=(int(np.prod(shape[i1:i2])),)
        if i2<len(shape):
            newShape+=shape[i2:]
        self.reshape(newShape)
    def splitIndex(self,i,dimensions):
        newShape=tuple(self.shape()[0:i])
        newShape+=tuple(dimensions)
        if i < len(self.shape()) - 1:
            newShape+=tuple(self.shape()[i+1:])
        self.reshape(newShape)
    def shape(self):
        return self.data.shape
    def transpose(self):
        return numpyBackend(data=self.data.transpose())
    def svd(self,centerIndex):
        '''
        Truncate up to truncation states in bond dimension
        Returns a tuple of numpyTensor (U,S,V^t) with A= U * S * V^t
        '''
        shape=self.data.shape
        newShape=( np.prod(shape[0:centerIndex]), np.prod(shape[centerIndex:]) )
        data=np.reshape(self.data,newShape)
        
        U,S,V=np.linalg.svd(data,full_matrices=False)
        UShape=list(shape[0:centerIndex]) + [S.shape[0]]
        U=U.reshape(UShape)
        V=V.reshape( [S.shape[0]] +   list(shape[centerIndex:])) 
        Ub=numpyTensor(data=U)
        Sb=numpyTensor(data=np.diag(S))
        Vb=numpyTensor(data=V)
        return Ub,Sb,Vb
    def trace(self,i1,i2):
        return numpyTensor(data=self.data.trace(axis1=i1, axis2=i2))
    def __repr__(self):
        return self.data.__repr__() 
    def flatten(self):
        self.mergeIndicesBetween(0,self.rank())
    def conj(self):
        return numpyTensor(data=np.conj(self.data),dtype=self.dtype)

    
class edge:
    def __init__(self,leftNode=None,leftIndex=None,rightNode=None,rightIndex=None):
        self.leftNode=leftNode
        self.leftIndex=leftIndex
        self.rightNode=rightNode
        self.rightIndex=rightIndex
    def __repr__(self):
        s="Left: " + repr(self.leftNode) + ","
        s+="Right: " + repr(self.rightNode)
        return s
    def isDangling(self):
        if (self.rightNode is None) or (self.leftNode is None)   :
            return True
        else:
            return False
class node:
    def __init__(self,shape=None,tensor=None,name=None,data=None):
        self.name=name
        if tensor is not None:
            self.tensor=tensor
        else:
            if data is not None:
                self.tensor=numpyTensor(data=data)
            else:
                self.tensor=numpyTensor(shape)
        self.edges=[None for  i in range(self.rank())]
        self.edgesIn=[True for  i in range(self.rank())]

        self.resetEdges()
    def __len__(self):
        return self.tensor.rank()
    def rank(self):
        return self.tensor.rank()
    def size(self):
        return self.tensor.size()
    def shape(self):
        return self.tensor.shape()
    def __repr__(self):
        s="[Shape=" + str(self.shape() ) + ",size=" + str(self.size())+ "]"
        return s
    def __getitem__(self,index):
        return self.tensor[index]
    def __setitem__(self,index,item):
        self.tensor[index]=item
    def random(self):
        self.tensor.random()
    def resetEdge(self,i):
        self.edges[i]=edge(leftNode=self,leftIndex=i)
        self.edgesIn[i]=True
    def resetEdges(self):
        for i in range(self.rank()):
            self.resetEdge(i)
    def copy(self):
        ''' Returns a new node without copying over the edges '''
        return node(tensor=self.tensor)
    def conj(self,inPlace=False):
        if not inPlace:
            return node(tensor=self.tensor.conj())
        else:
            self.tensor=self.tensor.conj()
        

def connect(leftNode,leftIndex,rightNode,rightIndex):
    '''
    Breaks previous edges and creates a new edge between the left and right node
    '''
    if leftNode is not None:
        leftNode.edges[leftIndex].rightNode=None
        leftNode.edges[leftIndex]=edge(leftNode=leftNode,leftIndex=leftIndex,rightIndex=rightIndex,rightNode=rightNode)
        leftNode.edgesIn[leftIndex]=True
    if rightNode is not None:
        rightNode.edges[rightIndex].leftNode=None
        if leftNode is not None:
            rightNode.edges[rightIndex]=leftNode.edges[leftIndex]
        rightNode.edgesIn[rightIndex]=False
        
    if leftNode is not None:
        return leftNode.edges[leftIndex]
    else:
         if rightNode is not None:
            return rightNode.edges[rightIndex]


def reconnect(oldNode,old_indices,newNode,new_indices):
    '''
    oldNode, old_indices,newNode,new_indices 
    Reboud old edges bound to oldNode on old_indices to newNode on new_indices
    '''
    
    for i,i2 in zip(old_indices,new_indices):
        if oldNode.edgesIn[i]:
            oldNode.edges[i].leftNode=newNode
            oldNode.edges[i].leftIndex=i2
            newNode.edges[i2]=oldNode.edges[i]
            newNode.edgesIn[i2]=True
        else:
            oldNode.edges[i].rightNode=newNode
            oldNode.edges[i].rightIndex=i2
            newNode.edges[i2]=oldNode.edges[i]
            newNode.edgesIn[i2]=False
            
def trace(A,index1,index2):
    i1=min(index1,index2)
    i2=max(index1,index2)
    
    node2=node(tensor=A.tensor.trace(i1,i2))
    oldIndices=list(range(0,i1)) + list(range(i1+1,i2)) + list(range(i2+1,len(A.shape())))
    newIndices=list(range(0,len(node2.shape())))
    
    reconnect(A,oldIndices,node2,newIndices)
    
    return node2
    
        
def contract(edge):
    # perform the contraction between the two tensors
    
    if edge.leftNode is edge.rightNode:
        return trace(edge.leftNode,edge.leftIndex,edge.rightIndex)
    
    leftTB=edge.leftNode.tensor
    rightTB=edge.rightNode.tensor
    indexPair=(edge.leftIndex,edge.rightIndex)
    data=leftTB.contract(rightTB,indexPair=indexPair)
    t=node(tensor=data)
    
    # reconnect all previous edges to the new tensor
    i2=0
    for i,edge2 in enumerate(edge.leftNode.edges):
        if edge2 is not edge:
            if not edge.leftNode.edgesIn[i]:
                edge2.rightNode=t
                edge2.rightIndex=i2
                t.edges[i2]=edge2
                t.edgesIn[i2]=False
            else:
                edge2.leftNode=t
                edge2.leftIndex=i2
                t.edges[i2]=edge2
                t.edgesIn[i2]=True
                
            i2+=1
            
    for i,edge2 in enumerate(edge.rightNode.edges):
        if edge2 is not edge:
            if not edge.rightNode.edgesIn[i]:
                edge2.rightNode=t
                edge2.rightIndex=i2
                t.edges[i2]=edge2
                t.edgesIn[i2]=False
            else:
                edge2.leftNode=t
                edge2.leftIndex=i2
                t.edges[i2]=edge2
                t.edgesIn[i2]=True
            i2+=1
    return t

def split(edge,direction="left"):
    
    if direction=="left":
        splitNode=edge.leftNode
        splitIndex=edge.leftIndex
    else:
        if direction=="right":
            splitNode=edge.rightNode
            splitIndex=edge.rightIndex + 1
        else:
            raise ValueError("Directions is neither left or right. ")
            
    if splitNode is None:
        raise ValueError("Split node is None.")
    
    t=splitNode.tensor
    U,S,V=t.svd(int(splitIndex))
    U=node(tensor=U)
    S=node(tensor=S)
    V=node(tensor=V)
    
    # reconnect left edges
    
    reconnect(splitNode,range(0,splitIndex),U,range(0,splitIndex))
    
    # reconnect right edges
    
    oldIndices=range(splitIndex, len(splitNode.edges)) # indices on old node
    newIndices=range(1, len(splitNode.edges) - splitIndex +1 )  # mapped indices on new node
    
    reconnect(splitNode,oldIndices,V,newIndices)
    
    # def connect U , S , V
    connect(U,splitIndex,S,0)
    connect(S,1,V,0)
    
    return (U,S,V)


def shiftOrthogonalityCenterRight(edge):
    U,S,V=split(edge,direction="left")
    T=contract(S.edges[1])
    T=contract(edge)
    return (U,T)
def shiftOrthogonalityCenterLeft(edge):
    U,S,V=split(edge,direction="right")
    T=contract(S.edges[0])
    T=contract(edge)
    return (T,V)

def shiftOrthogonalityCenter(edge,direction):
    if direction == "left":
        return shiftOrthogonalityCenterLeft(edge)
    else:
        if direction == "right":
            return shiftOrthogonalityCenterRight(edge)
        else:
            raise AttributeError("Unkown shifting direction " )


class net:
    def __init__(self):
        self.nodes=[]
    def add(self,node):
        self.nodes.append(node)
    def __getitem__(self,i):
        return self.nodes[i]
    def initRandom(self):
        for node in self.nodes:
            node.random()
        
class mps(net):
    def __init__(self,siteDimension,nSites,bondDimension=100):
        super().__init__()
        self.bondDimension=100
        self.nSites=nSites
        
        self.add(node(shape=(bondDimension)))
        for i in range(0,nSites):
            self.add(node(shape=(bondDimension,siteDimension,bondDimension)))
        self.add(node(shape=(bondDimension)))
        self.nodes[0][0]=1
        self.nodes[-1][-1]=1
        # connect width edges
        
        connect(self.nodes[0],0,self.nodes[1],0)
        for i in range(1,self.nSites+1):
            connect(self.nodes[i],2,self.nodes[i+1],0)
        self.initRandom()
        
    def initRandom(self):
        for i in range(1,self.nSites+1):
            self.nodes[i].random()
    def norm(self):
        
        if self.orthogonalized:
            A=self[self.orthogonalizationCenter].copy()
            B=A.conj()
            edge1=connect(A,1,B,1)
            edge2=connect(A,0,B,0)
            edge3=connect(A,2,B,2)
            
            T=contract(edge2)
            T=contract(edge3)

            T=contract(edge1)
            
            return np.abs(T.tensor.data)
    def normalize(self):
        self[self.orthogonalizationCenter].tensor.data/=np.sqrt(self.norm())
            
    
    def orthogonalize(self):
        # orthogonolize from left
        #for i in range(0,orthogonalizationCenter):
        #    self.nodes[i],self.nodes[i+1]=shiftOrthogonalityCenter(self.nodes[i+1].edges[0],direction="right")
        self.orthogonalizationCenter=self.nSites - 1
        while(self.orthogonalizationCenter>=1):
            self.shiftOrthogonalizationCenter(direction="left")
        
        self.orthogonalized=True
        
        
    def shiftOrthogonalizationCenter(self,direction):
        i=self.orthogonalizationCenter+1
        if direction=="right":
            self.nodes[i],self.nodes[i+1]=shiftOrthogonalityCenter(self.nodes[i+1].edges[0],direction="right")
            self.orthogonalizationCenter+=1
        else:
        
            if direction=="left":
                self.nodes[i-1],self.nodes[i]=shiftOrthogonalityCenter(self.nodes[i].edges[0],direction="left")
                self.orthogonalizationCenter-=1
            else:
                raise ValueError("Unkown direction.")
    def __getitem__(self,i):
        return self.nodes[i+1]
    def __len__(self):
        return self.nSites

class mpo(net):
    def __init__(self,siteDimension,nSites,bondDimension=100):
        super().__init__()
        self.bondDimension=bondDimension
        self.nSites=nSites
        
        self.add(node(shape=(bondDimension)))
        for i in range(0,nSites):
            self.add(node(shape=(bondDimension,siteDimension,bondDimension,siteDimension)))
        self.add(node(shape=(bondDimension)))

        self.nodes[0][0]=1
        self.nodes[-1][-1]=1
        # connect width edges
        
        connect(self.nodes[0],0,self.nodes[1],0)
        for i in range(1,self.nSites+1):
            connect(self.nodes[i],2,self.nodes[i+1],0)
            
    def initRandom(self):
        for i in range(1,self.nSites+1):
            self.nodes[i].random()
       
    def __getitem__(self,i):
        return self.nodes[i+1]
    def __len__(self):
        return self.nSites


def applyHeff(Hl,Hc,Hr,A):
    # create the connections
    edges=[ connect(Hl,0,A,0),connect(Hl,1,Hc,0) , connect(A,1,Hc,1),connect(A,2,Hr,0),connect(Hc,2,Hr,1) ]
    
    # perform the contractions
    
    for edge in edges:
        T=contract(edge)
    
    return T

class effectiveHamiltonian:
    def __init__(self,mpsState,mpoOp):
        self.mpsState=mpsState
        self.mpoOp=mpoOp
        self.currentIndex=1
        self.Hls=[None for i in range( len(self.mpsState))]
        self.Hrs=[None for i in range( len(self.mpsState))]
        
        self.initHl()
        self.initHrs()
        
    def initHl(self):
        D=self.mpsState[0].shape()[0]
        M=self.mpoOp[0].shape()[0]
        
        self.Hls[0]=node(shape=(D,M,D))
        self.Hls[0][0,0,0]=1
        
    def initHr(self):
        
        D=self.mpsState[len(self.mpsState)-1].shape()[0]
        M=self.mpoOp[len(self.mpoOp)-1].shape()[0]
        
        self.Hrs[-1]=node(shape=(D,M,D))
        self.Hrs[-1][D-1,M-1,D-1]=1
        self.currentIndex=len(self.mpsState)-1
        
    def initHrs(self):
        self.initHr()
        self.updateRightHamiltonian()
        for i in range(self.mpsState.nSites-1,0,-1):
            self.currentIndex=i
            self.updateRightHamiltonian()
        self.currentIndex=0
        self.updateLinearOperator()

        
    def updateLinearOperator(self):
        D1,d,D2=self.mpsState[self.currentIndex].shape()
        toOptShape=(D1,d,D2)
        toOptSize=D1*d*D2
        
        def mv(v):
            A=node(data=v.reshape(toOptShape))
            Hc=self.mpoOp[self.currentIndex].copy()
            Hl=self.Hls[self.currentIndex].copy()
            Hr=self.Hrs[self.currentIndex].copy()
            
            A2=applyHeff(Hl,Hc,Hr,A)
            return A2.tensor.data.flatten()
        
        self.L=scipy.sparse.linalg.LinearOperator(  (toOptSize,toOptSize) ,matvec=mv)
        
    def solveHeff(self):
        energies,eigenVectors=eigs(self.L,maxiter=10000,which="SR",k=6)
        return (energies,eigenVectors)
    
    def updateRightHamiltonian(self):
        # create the net to be contracted
        A=self.mpsState[self.currentIndex].copy()
        B=A.conj()
        Hc=self.mpoOp[self.currentIndex].copy()
        Hr=self.Hrs[self.currentIndex].copy()
        
        edges= [ connect(A,2,Hr,0) ,connect(Hc,2,Hr,1), connect(A,1,Hc,1) , connect(B,2,Hr,2) ,connect(Hc,3,B,1)]
        
        for edge in edges:
            Hr=contract(edge)
            
        self.Hrs[self.currentIndex-1]=Hr
        
        return Hr
    
    def updateLeftHamiltonian(self):
        A=self.mpsState[self.currentIndex].copy()
        B=A.conj()
        Hc=self.mpoOp[self.currentIndex].copy()
        Hl=self.Hls[self.currentIndex].copy()
        
        edges= [ connect(Hl,0,A,0)  ,connect(Hl,1,Hc,0),connect(A,1,Hc,1) ,connect(Hl,2,B,0) ,connect(Hc,3,B,1) ]
        
        for edge in edges:
           Hl=contract(edge)
        self.Hls[self.currentIndex+1]=Hl
        
    def update(self,direction):
        # solves the eigen value problem
        e,v=self.solveHeff()
        
        self.e=e
        self.v=v
        # update the mps state
        newShape=self.mpsState[self.currentIndex].shape()
        self.mpsState[self.currentIndex].tensor.data=v[:,0].reshape(newShape)
        #self.mpsState.normalize()
        
        self.mpsState.shiftOrthogonalizationCenter(direction=direction)
        
        # update the left hamiltonian
        if direction=="right":
            self.updateLeftHamiltonian()
            self.currentIndex+=1
        else:
            if direction=="left":
                self.updateRightHamiltonian()
                self.currentIndex-=1
        
        self.updateLinearOperator()    

    def sweep(self,direction,outSteps=1):
        es=[]
        for i in range(len(self.mpsState)-1):
            self.update(direction)
            if i%outSteps==0:
                print("Energy: " + str(self.e[0]))
           
            es.append(self.e[0].real)
        return es
    def optimize(self,n,outSteps=1):
        es=[]
        
        for i in range(n):
            es=es + self.sweep("right",outSteps=outSteps)
            
            es+=self.sweep("left",outSteps=outSteps)
            print("Energy: " + str(self.e[0]))
        return np.array(es)

class chargeShape:
    def __init__(self,chargesDict):
        self._chargeToRanges=chargesDict
        
    def __getitem__(self,charge):
        return self._chargeToRanges[charge]
    def __len__(self):
        return len(self._chargeToRanges)
    def size(self):
        return np.sum( (len(r) for r in self._chargeToRanges.values() ) )
    def charges(self):
        return self._chargeToRanges.keys()
    def chargeOfIndex(self,i):
        for q in self.charges():
            if i in self[q]:
                return q
        return None
    
class blockedTensor:
    
    def __init__(self,chargeShapes):
        self._chargeShapes=chargeShapes
        self._data=[]
        self.charges=[c.charges() for c in self._chargeShapes]
        self._sectors=list(itertools.product(*self.charges ))
        for sector in self._sectors:
            shape=[ len(self._chargeShapes[i][q])  for i,q in enumerate(sector) ]
            self._data.append(numpyTensor(shape=shape))
        #self._sortsectors()
            
    def rank(self):
        return len(self._chargeShapes)
    def shape(self):
        return tuple([c.size() for c in self._chargeShapes])
    
    def _sortsectors(self):
        self._sectors,self._data= ( list(t) for t in  zip(*sorted(zip( self._sectors, self._data))) )
    def _indexBlock(self, block   ):
        i=bisect.bisect_left(self._sectors,block)
        if i!= len(self._sectors) and self._sectors[i] ==block:
            return i
        else:
            return None
    def isBlockZero(self,block):
        if self._indexBlock(block) is not None:
            return False
        else:
            return True
    
    def random(self):
        for t in self._data:
            t.random()

    def block(self,index):
        return self._data[self._indexBlock(index)]
    
    def blockIterator(self, index=None,qns=None):
        '''
        index : int 
        qns : list, quantum numbers in index to keep
        '''
        charges=[c.charges() for c in self._chargeShapes]
        if index is not None:
            def f (x):
                return x in qns
        
            charges[index]= filter(f ,charges[index] )
        return itertools.product(*charges)
    
    def contract(self,t2,index1,index2):
        _data=[]
        
        for block1 in self.blockIterator():
            for  block2 in t2.blockIterator(index=index2,qns=self.charges[index1]):
                pass
                
                
                
                
        
        
        
        
        
        
