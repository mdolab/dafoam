
# distutils: language = c++
# distutils: sources = TestDAFoamIncompressible.C

"""
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Cython wrapper functions that call OpenFOAM libraries defined
        in the *.C and *.H files. The python naming convention is to
        add "py" before the C++ class name
"""

# for using Petsc
# from petsc4py.PETSc cimport Vec, PetscVec

# declear cpp functions
cdef extern from "TestDAFoamIncompressible.H" namespace "Foam":
    cppclass TestDAFoamIncompressible:
        TestDAFoamIncompressible(char *) except +
        int testDAUtility(object, object)
        int testDAOption(object)
        int testDAModel(object)
        int testDAStateInfo(object)
        int testDAObjFunc(object)
        int testDAField(object)

# create python wrappers that call cpp functions
cdef class pyTestDAFoamIncompressible:

    # define a class pointer for cpp functions
    cdef:
        TestDAFoamIncompressible * _thisptr

    # initialize this class pointer with NULL
    def __cinit__(self):
        self._thisptr = NULL

    # deallocate the class pointer, and
    # make sure we don't have memory leak
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    # point the class pointer to the cpp class constructor
    def __init__(self, argsAll):
        self._thisptr = new TestDAFoamIncompressible(argsAll)

    def testDAUtility(self, pyDict, pyStrList):
        testErrors = self._thisptr.testDAUtility(pyDict, pyStrList)
        return testErrors
    
    def testDAOption(self, pyDict):
        testErrors = self._thisptr.testDAOption(pyDict)
        return testErrors

    def testDAModel(self, pyDict):
        testErrors = self._thisptr.testDAModel(pyDict)
        return testErrors
    
    def testDAStateInfo(self, pyDict):
        testErrors = self._thisptr.testDAStateInfo(pyDict)
        return testErrors
    
    def testDAObjFunc(self, pyDict):
        testErrors = self._thisptr.testDAObjFunc(pyDict)
        return testErrors
    
    def testDAField(self, pyDict):
        testErrors = self._thisptr.testDAField(pyDict)
        return testErrors
