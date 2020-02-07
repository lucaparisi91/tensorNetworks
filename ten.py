import numpy as np
import tenpy
import tenpy.linalg.charges as charges_tenpy
import tenpy.linalg.lanczos as lanczos
charges=charges_tenpy
np_c=tenpy.linalg.np_conserved

truncation=tenpy.algorithms.truncation



def shiftOrthogonalizationCenterLeft(A,B):
    B=B.combine_legs([1,2])
    U,S,V=np_c.svd(B)
    S=np_c.diag(S,leg=U.legs[1].conj())
    U=np_c.tensordot(U,S,axes=[1,0])
    V=V.split_legs([1])
    A=np_c.tensordot(A,U,axes=[2,0])
    return A,V

def shiftOrthogonalizationCenterRight(A,B):
    A=B.combine_legs([0,1])
    U,S,V=np_c.svd(A)
    S=np_c.diag(S,leg=U.legs[1].conj())
    V=np_c.tensordot(S,V,axes=[1,0])
    U=U.split_legs([0])
    B=np_c.tensordot(V,B,axes=[1,0])
    return U,B

def shiftOrthogonalizationCenter(A,B,direction):
    if direction == "left":
        return shiftOrthogonalizationCenterLeft(A,B)
    elif direction == "right":
        return shiftOrthogonalizationCenterRight(A,B)
    else:
        raise ValueError("Unkown direction : " + str(direction))



