
# distutils: language = c++
# distutils: sources = TestDAFoamCompressible.C

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
cdef extern from "TestDAFoamCompressible.H" namespace "Foam":
    cppclass TestDAFoamCompressible:
        TestDAFoamCompressible(char *) except +
        int testDAStateInfo(object)

# create python wrappers that call cpp functions
cdef class pyTestDAFoamCompressible:

    # define a class pointer for cpp functions
    cdef:
        TestDAFoamCompressible * _thisptr

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
        self._thisptr = new TestDAFoamCompressible(argsAll)
    
    def testDAStateInfo(self, pyDict):
        testErrors = self._thisptr.testDAStateInfo(pyDict)
        return testErrors
