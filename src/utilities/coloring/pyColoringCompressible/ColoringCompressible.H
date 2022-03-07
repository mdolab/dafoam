/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Python wrapper library for computing coloring

\*---------------------------------------------------------------------------*/

#ifndef ColoringCompressible_H
#define ColoringCompressible_H

#include <petscksp.h>
#include "Python.h"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
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
                       Class ColoringCompressible Declaration
\*---------------------------------------------------------------------------*/

class ColoringCompressible
{

private:
    /// Disallow default bitwise copy construct
    ColoringCompressible(const ColoringCompressible&);

    /// Disallow default bitwise assignment
    void operator=(const ColoringCompressible&);

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
    ColoringCompressible(
        char* argsAll,
        PyObject* pyOptions);

    /// Destructor
    virtual ~ColoringCompressible();

    /// run
    void run();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
