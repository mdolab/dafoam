/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for simpleTFoam

\*---------------------------------------------------------------------------*/

#ifndef SimpleTDAFoam_H
#define SimpleTDAFoam_H

#include <petscksp.h>
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
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
                       Class SimpleTDAFoam Declaration
\*---------------------------------------------------------------------------*/

class SimpleTDAFoam
{

private:

    /// Disallow default bitwise copy construct
    SimpleTDAFoam(const SimpleTDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const SimpleTDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    SimpleTDAFoam(char* argsAll);

    /// Destructor
    virtual ~SimpleTDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

