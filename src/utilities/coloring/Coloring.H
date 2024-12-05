/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Python wrapper library for computing coloring

\*---------------------------------------------------------------------------*/

#ifndef Coloring_H
#define Coloring_H

#include <petscksp.h>
#include "Python.h"
#include "argList.H"
#include "DAUtility.H"
#include "DACheckMesh.H"
#include "DAOption.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DATurbulenceModel.H"
#include "DAIndex.H"
#include "DAFunction.H"
#include "DAJacCon.H"
#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class Coloring Declaration
\*---------------------------------------------------------------------------*/

class Coloring
{

private:
    /// Disallow default bitwise copy construct
    Coloring(const Coloring&);

    /// Disallow default bitwise assignment
    void operator=(const Coloring&);

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
    Coloring(
        char* argsAll,
        PyObject* pyOptions);

    /// Destructor
    virtual ~Coloring();

    /// run
    void run();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif
