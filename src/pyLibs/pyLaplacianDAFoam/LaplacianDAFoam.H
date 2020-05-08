/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for laplacianFoam

\*---------------------------------------------------------------------------*/

#ifndef LaplacianDAFoam_H
#define LaplacianDAFoam_H

#include <petscksp.h>
#include "fvCFD.H"
#include "fvOptions.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointObjectiveFunction.H"
#include "AdjointDerivative.H"
#include "nearWallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class LaplacianDAFoam Declaration
\*---------------------------------------------------------------------------*/

class LaplacianDAFoam
{

private:

    /// Disallow default bitwise copy construct
    LaplacianDAFoam(const LaplacianDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const LaplacianDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    LaplacianDAFoam(char* argsAll);

    /// Destructor
    virtual ~LaplacianDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

