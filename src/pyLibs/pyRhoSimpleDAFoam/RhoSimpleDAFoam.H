/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for rhoSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef RhoSimpleDAFoam_H
#define RhoSimpleDAFoam_H

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
                       Class RhoSimpleDAFoam Declaration
\*---------------------------------------------------------------------------*/

class RhoSimpleDAFoam
{

private:

    /// Disallow default bitwise copy construct
    RhoSimpleDAFoam(const RhoSimpleDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const RhoSimpleDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    RhoSimpleDAFoam(char* argsAll);

    /// Destructor
    virtual ~RhoSimpleDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

