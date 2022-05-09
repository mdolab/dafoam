/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Python wrapper library for computing coloring

\*---------------------------------------------------------------------------*/

#ifndef ColoringSolid_H
#define ColoringSolid_H

#include <petscksp.h>
#include "Python.h"
#include "argList.H"
#include "DAUtility.H"
#include "DACheckMesh.H"
#include "DAOption.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAObjFunc.H"
#include "DAJacCon.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class ColoringSolid Declaration
\*---------------------------------------------------------------------------*/

class ColoringSolid
{

private:
    /// Disallow default bitwise copy construct
    ColoringSolid(const ColoringSolid&);

    /// Disallow default bitwise assignment
    void operator=(const ColoringSolid&);

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
    ColoringSolid(
        char* argsAll,
        PyObject* pyOptions);

    /// Destructor
    virtual ~ColoringSolid();

    /// run
    void run();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
