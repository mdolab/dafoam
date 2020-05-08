/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for coloringSolverIncompressible

\*---------------------------------------------------------------------------*/

#ifndef ColoringSolverIncompressible_H
#define ColoringSolverIncompressible_H

#include <petscksp.h>
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "nearWallDist.H"
#include "argList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class ColoringSolverIncompressible Declaration
\*---------------------------------------------------------------------------*/

class ColoringSolverIncompressible
{

private:

    /// Disallow default bitwise copy construct
    ColoringSolverIncompressible(const ColoringSolverIncompressible&);

    /// Disallow default bitwise assignment
    void operator=(const ColoringSolverIncompressible&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    ColoringSolverIncompressible(char* argsAll);

    /// Destructor
    virtual ~ColoringSolverIncompressible();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

