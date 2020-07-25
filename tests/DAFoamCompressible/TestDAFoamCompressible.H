/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Test functions in libDAFoamCompressible.so

\*---------------------------------------------------------------------------*/

#ifndef TestDAFoamCompressible_H
#define TestDAFoamCompressible_H

#include <petscksp.h>
#include "fvCFD.H"
#include "Python.h"
#include "DAUtility.H"
#include "DAOption.H"
#include "DAStateInfo.H"
#include "DATurbulenceModel.H"
#include "DAModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class TestDAFoamCompressible Declaration
\*---------------------------------------------------------------------------*/

class TestDAFoamCompressible
{

private:
    /// Disallow default bitwise copy construct
    TestDAFoamCompressible(const TestDAFoamCompressible&);

    /// Disallow default bitwise assignment
    void operator=(const TestDAFoamCompressible&);

    /// all the arguments
    char* argsAll_;

public:
    // Constructors

    /// Construct from components
    TestDAFoamCompressible(char* argsAll);

    /// Destructor
    virtual ~TestDAFoamCompressible();

    /// test DAStateInfo
    label testDAStateInfo(PyObject* pyDict);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
