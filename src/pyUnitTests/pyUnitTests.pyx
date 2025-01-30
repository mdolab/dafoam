
# distutils: language = c++
# distutils: sources = UnitTests.C

"""
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Cython wrapper functions that call OpenFOAM libraries defined
        in the *.C and *.H files. The python naming convention is to
        add "py" before the C++ class name
"""

# for using Petsc


# declare cpp functions
cdef extern from "UnitTests.H" namespace "Foam":
    cppclass UnitTests:
        UnitTests() except +
        void runDAUtilityTest1(char *, object)
    
# create python wrappers that call cpp functions
cdef class pyUnitTests:

    # define a class pointer for cpp functions
    cdef:
        UnitTests * _thisptr

    # initialize this class pointer with NULL
    def __cinit__(self):
        self._thisptr = NULL

    # deallocate the class pointer, and
    # make sure we don't have memory leak
    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr

    # point the class pointer to the cpp class constructor
    def __init__(self):
        self._thisptr = new UnitTests()
    
    # wrap all the other member functions in the cpp class
    def runDAUtilityTest1(self, argsAll, pyOptions):
        self._thisptr.runDAUtilityTest1(argsAll, pyOptions)