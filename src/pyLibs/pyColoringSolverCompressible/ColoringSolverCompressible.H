/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for coloringSolverCompressible

\*---------------------------------------------------------------------------*/

#ifndef ColoringSolverCompressible_H
#define ColoringSolverCompressible_H

#include <petscksp.h>
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "fvOptions.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointObjectiveFunction.H"
#include "nearWallDist.H"
#include "argList.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class ColoringSolverCompressible Declaration
\*---------------------------------------------------------------------------*/

class ColoringSolverCompressible
{

private:

    /// Disallow default bitwise copy construct
    ColoringSolverCompressible(const ColoringSolverCompressible&);

    /// Disallow default bitwise assignment
    void operator=(const ColoringSolverCompressible&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    ColoringSolverCompressible(char* argsAll);

    /// Destructor
    virtual ~ColoringSolverCompressible();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

