import ten
import numpy as np
import mpsTen
import mpoTen
import tenTools

np_c=ten.np_c
charges=ten.charges
mps=mpsTen.mps
mpo=mpoTen.mpo


class effectiveHamiltonian:
    
    def __init__(self,mpsState,mpoOp):
        self.state=mpsState
        self.state.normalize()
        
        self.mpoH=mpoOp
        
        
        self.Hls=[None for i in range(self.state.nSites())]
        self.Hrs=[None for i in range(self.state.nSites())]
        
        Hr=np_c.outer(self.state.rightBond,self.mpoH.rightBond)
        Hr=np_c.outer(Hr,self.state.rightBond.conj() )
        self.Hrs[self.state.nSites()-2]=Hr

        Hl=np_c.outer(self.state.leftBond,self.mpoH.leftBond)
        Hl=np_c.outer(Hl,self.state.leftBond.conj() )
        self.Hls[0]=Hl
        self.center=self.state.nSites() - 2

        
        for i in range(self.state.nSites() -2):
              self.updateRightHamiltonian()
        
        
    def updateRightHamiltonian(self):
        A=self.state[self.center+1]
        O=self.mpoH[self.center+1]
        B=A.conj()
        
        Hr=self.Hrs[self.center]
        tmp=np_c.tensordot(A,Hr,axes=["vR","vL"])
        tmp=np_c.tensordot(tmp,O,axes=["vOL","vOR"])
        tmp=np_c.trace(tmp,"pT","p")

        tmp=np_c.tensordot(tmp,B,axes=["vL*","vR*"])
        tmp=np_c.trace(tmp,"p*","pB")
        self.center-=1
        
        self.Hrs[self.center]=tmp

        return tmp
    
    def updateLeftHamiltonian(self):
        A=self.state[self.center]
        O=self.mpoH[self.center]
        B=A.conj()
        
        Hl=self.Hls[self.center]
        
        tmp=np_c.tensordot(Hl,A,axes=["vR","vL"])
        tmp=np_c.tensordot(tmp,O,axes=["vOR","vOL"])
        tmp=np_c.trace(tmp,"pT","p")
        
        tmp=np_c.tensordot(tmp,B,axes=["vR*","vL*"])
        tmp=np_c.trace(tmp,"p*","pB")
        self.center+=1
        
        self.Hls[self.center]=tmp

        return tmp
    

class twoSiteEffectiveHamiltonian(effectiveHamiltonian):
    
    def __init__(self,*args,**kwds):
        super().__init__(*args,**kwds)
        self.minOv=1
        self.energies=[]
        self.run_statistics={"lanczos_time":tenTools.timer(),"running_time":tenTools.timer()}
        
    def matvec(self,psi0):
        Hl=self.Hls[self.center]
        Hr=self.Hrs[self.center]
        OLeft=self.mpoH[self.center]
        ORight=self.mpoH[self.center+1].copy()
        ORight=ORight.iset_leg_labels(["vOL1","pT1","pB1","vOR1"])
        tmp=np_c.tensordot(Hl,psi0,axes=["vR","vL"])
        tmp=np_c.tensordot(tmp,OLeft,axes=["vOR","vOL"])
        tmp=np_c.trace(tmp,"p1","pT")
        tmp=np_c.tensordot(tmp,ORight,axes=["vOR","vOL1"])
        tmp=np_c.trace(tmp,"p2","pT1")
        tmp=np_c.tensordot(tmp,Hr,axes=["vR","vL"])
        tmp=np_c.trace(tmp,"vOR1","vOL")
        tmp.iset_leg_labels(["vL","p1","p2","vR"])
        
        
        return tmp
    
    def optimize(self,direction="right",trunc_par={}):
        '''
        Two site optimization at center and center + 1
        '''

        
        i=self.center
        j=self.center + 1
        
        # set initial vector as the tensor product of the two tensors
        psi0=np_c.tensordot(self.state[i],self.state[j],axes=["vR","vL"])
        #psi0=psi0.from_func(lambda shape: np.random.random(size=shape),legcharges=psi0.legs)
        psi0.iset_leg_labels(["vL","p1","p2","vR"])
        
        #finds the ground state
        L=ten.lanczos.LanczosGroundState(self,psi0,{})
        self.run_statistics["lanczos_time"].toogle()
        e,AB,iterations=L.run()
        self.run_statistics["lanczos_time"].toogle()
        
        self.energies.append(e)
        
        self.state[i],self.state[i+1],err=mpsTen.splitMPS(AB,direction=direction,trunc_par=trunc_par)

        if err.ov < self.minOv:
            self.minOv=err.ov
        
    def updateHamiltonian(self,direction):
        if direction=="right":
            self.updateLeftHamiltonian()
        elif direction=="left":
            self.updateRightHamiltonian()
            
    def sweep(self,direction="right"):

        self.optimize(direction=direction)
        for i in range(self.state.nSites() - 2 ):
            self.updateHamiltonian(direction)
            self.optimize(direction=direction,trunc_par=self.trunc_par)
        
    def run(self,nSweeps,trunc_par={}):
        self.trunc_par=trunc_par
        self.run_statistics["running_time"].start()
        for i in range(nSweeps):
            self.sweep(direction="right")
            self.sweep(direction="left")
            
            print("----------------")
            print ("Sweep: " + str(i))
            print ("Min. Overlap: " + str(self.minOv))
            print ("Energy: " + str(self.energies[-1]))
            print(self.run_statistics)
            self.minOv=1
        self.run_statistics["running_time"].stop()
        
