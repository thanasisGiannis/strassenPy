import numpy
class randomNumPyMatrix :
    def createRandomNumPyMatrix(rows, cols):
        #return numpy.random.uniform(low=-1, high=1, size=(rows,cols)) 
        return numpy.random.randint(-100,100,size = (rows,cols))
