/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for rhoSimpleCFoam

\*---------------------------------------------------------------------------*/

#ifndef RhoSimpleCDAFoam_H
#define RhoSimpleCDAFoam_H

#include <petscksp.h>
#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "fvOptions.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointObjectiveFunction.H"
#include "AdjointDerivative.H"
#include "AdjointNewtonKrylov.H"
#include "nearWallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class RhoSimpleCDAFoam Declaration
\*---------------------------------------------------------------------------*/

class RhoSimpleCDAFoam
{

private:

    /// Disallow default bitwise copy construct
    RhoSimpleCDAFoam(const RhoSimpleCDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const RhoSimpleCDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    RhoSimpleCDAFoam(char* argsAll);

    /// Destructor
    virtual ~RhoSimpleCDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

