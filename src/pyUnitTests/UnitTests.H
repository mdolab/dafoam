/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Python wrapper library for UnitTests

\*---------------------------------------------------------------------------*/

#ifndef UnitTests_H
#define UnitTests_H

#include <petscksp.h>
#include "Python.h"
#include "fvCFD.H"
#include "fvMesh.H"
#include "runTimeSelectionTables.H"
#include "OFstream.H"
#include "functionObjectList.H"
#include "fvOptions.H"
#include "DAOption.H"
#include "DAUtility.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class UnitTests Declaration
\*---------------------------------------------------------------------------*/

class UnitTests
{

private:
    /// Disallow default bitwise copy construct
    UnitTests(const UnitTests&);

    /// Disallow default bitwise assignment
    void operator=(const UnitTests&);

    /// args pointer
    autoPtr<argList> argsPtr_;

public:
    // Constructors

    /// Construct from components
    UnitTests();

    /// Destructor
    virtual ~UnitTests();

    /// run the test
    void runDAUtilityTest1(
        char* argsAll,
        PyObject* pyOptions);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
