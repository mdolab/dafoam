/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for solidDisplacementDAFoam

\*---------------------------------------------------------------------------*/

#ifndef SolidDisplacementDAFoam_H
#define SolidDisplacementDAFoam_H

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
                       Class SolidDisplacementDAFoam Declaration
\*---------------------------------------------------------------------------*/

class SolidDisplacementDAFoam
{

private:

    /// Disallow default bitwise copy construct
    SolidDisplacementDAFoam(const SolidDisplacementDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const SolidDisplacementDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    SolidDisplacementDAFoam(char* argsAll);

    /// Destructor
    virtual ~SolidDisplacementDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

