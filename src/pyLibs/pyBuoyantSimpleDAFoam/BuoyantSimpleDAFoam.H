/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for buoyantSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef BuoyantSimpleDAFoam_H
#define BuoyantSimpleDAFoam_H

#include <petscksp.h>
#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "radiationModel.H"
#include "simpleControl.H"
#include "fvOptions.H"
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
                       Class BuoyantSimpleDAFoam Declaration
\*---------------------------------------------------------------------------*/

class BuoyantSimpleDAFoam
{

private:

    /// Disallow default bitwise copy construct
    BuoyantSimpleDAFoam(const BuoyantSimpleDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const BuoyantSimpleDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    BuoyantSimpleDAFoam(char* argsAll);

    /// Destructor
    virtual ~BuoyantSimpleDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

