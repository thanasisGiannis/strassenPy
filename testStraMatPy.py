from randMatPy.randMatPy import randomNumPyMatrix
from Strassen.Strassen import Strassen

import numpy 
import math
import time
from decimal import Decimal


rows = 2**11
cols = rows
A = randomNumPyMatrix.createRandomNumPyMatrix(rows,cols)
B = randomNumPyMatrix.createRandomNumPyMatrix(rows,cols)

stS = time.time()
C2 = Strassen.matMul(A,B)
etS = time.time()

stN = time.time()
C1 = numpy.matmul(A,B)
etN = time.time()

err = str(numpy.linalg.norm(numpy.subtract(C1,C2)))
scientific_notation = '%.2E' % Decimal(err)
print("Error: " + scientific_notation)
print("-------------")
elapsed_time = etS - stS
scientific_notation = '%.2E' % Decimal(elapsed_time)
print('Strassen Execution time: ' + scientific_notation + ' seconds')
print("-------------")
elapsed_time = etN - stN
scientific_notation = '%.2E' % Decimal(elapsed_time)
print('Numpy Execution time: ' + scientific_notation + ' seconds')
print("-------------")
