import numpy
import math

class Strassen:
    def __checkCorrectInputs(A, B):
        if(not isinstance(A, numpy.ndarray)): raise Exception("First Matrix Input NOT a Matrix")
        if(not isinstance(B, numpy.ndarray)): raise Exception("Second Matrix Input NOT a Matrix")
        if(A.ndim != 2 or B.ndim !=2): raise Exception("Not 2 Dimensional Matrices") # check if matrices
        
        Arow, Acol = A.shape
        Brow, Bcol = B.shape
        if(not (Arow == Acol == Brow == Bcol)): raise Exception("Matrices NOT square") # check if square
        if(not (math.log(Arow, 2).is_integer() )): raise Exception("dimensions not module of 2") #check if dimensions are of 2^n // should pad with zeros
    
    def __StrassenAlgoMatMul(A,B):
        dim, col = A.shape

        if(dim <= 2**4): return numpy.matmul(A,B)

        A11 = A[0:dim//2, 0:dim//2].view(dtype=A.dtype, type=numpy.matrix)
        A22 = A[dim//2:, dim//2:].view(dtype=A.dtype, type=numpy.matrix)
        A12 = A[0:dim//2, dim//2:].view(dtype=A.dtype, type=numpy.matrix)
        A21 = A[dim//2:, 0:dim//2].view(dtype=A.dtype, type=numpy.matrix)

        B11 = B[0:dim//2, 0:dim//2].view(dtype=B.dtype, type=numpy.matrix)
        B22 = B[dim//2:, dim//2:].view(dtype=B.dtype, type=numpy.matrix)
        B12 = B[0:dim//2, dim//2:].view(dtype=B.dtype, type=numpy.matrix)
        B21 = B[dim//2:, 0:dim//2].view(dtype=B.dtype, type=numpy.matrix)

        M1 = Strassen.__StrassenAlgoMatMul(numpy.add(A11,A22),numpy.add(B11,B22))
        M2 = Strassen.__StrassenAlgoMatMul(numpy.add(A21,A22),B11)
        M3 = Strassen.__StrassenAlgoMatMul(A11, numpy.subtract(B12,B22))
        M4 = Strassen.__StrassenAlgoMatMul(A22, numpy.subtract(B21,B11))
        M5 = Strassen.__StrassenAlgoMatMul(numpy.add(A11,A12),B22)
        M6 = Strassen.__StrassenAlgoMatMul(numpy.subtract(A21,A11),numpy.add(B11,B12))
        M7 = Strassen.__StrassenAlgoMatMul(numpy.subtract(A12,A22),numpy.add(B21,B22))
        
        return numpy.block([ [numpy.subtract(numpy.add(numpy.add(M1,M4),M7),M5), numpy.add(M3,M5)], 
                             [numpy.add(M2,M4), numpy.subtract(numpy.add(numpy.add(M1,M3),M6),M2)]
                            ])

        #return numpy.matmul(A,B)
        
    def matMul(A, B):
        Strassen.__checkCorrectInputs(A,B)
        return Strassen.__StrassenAlgoMatMul(A,B)
        
    