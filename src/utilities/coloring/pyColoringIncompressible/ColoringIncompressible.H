/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Python wrapper library for computing coloring

\*---------------------------------------------------------------------------*/

#ifndef ColoringIncompressible_H
#define ColoringIncompressible_H

#include <petscksp.h>
#include "Python.h"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "argList.H"
#include "DAUtility.H"
#include "DACheckMesh.H"
#include "DAOption.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DATurbulenceModel.H"
#include "DAIndex.H"
#include "DAObjFunc.H"
#include "DAJacCon.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class ColoringIncompressible Declaration
\*---------------------------------------------------------------------------*/

class ColoringIncompressible
{

private:
    /// Disallow default bitwise copy construct
    ColoringIncompressible(const ColoringIncompressible&);

    /// Disallow default bitwise assignment
    void operator=(const ColoringIncompressible&);

    /// all the arguments
    char* argsAll_;

    /// all options in DAFoam
    PyObject* pyOptions_;

    /// args pointer
    autoPtr<argList> argsPtr_;

    /// runTime pointer
    autoPtr<Time> runTimePtr_;

    /// fvMesh pointer
    autoPtr<fvMesh> meshPtr_;

public:
    // Constructors

    /// Construct from components
    ColoringIncompressible(
        char* argsAll,
        PyObject* pyOptions);

    /// Destructor
    virtual ~ColoringIncompressible();

    /// run
    void run();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
