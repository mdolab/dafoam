/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

    Description:
        Python wrapper library for buoyantBoussinesqSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef BuoyantBoussinesqSimpleDAFoam_H
#define BuoyantBoussinesqSimpleDAFoam_H

#include <petscksp.h>
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "radiationModel.H"
#include "fvOptions.H"
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
                       Class BuoyantBoussinesqSimpleDAFoam Declaration
\*---------------------------------------------------------------------------*/

class BuoyantBoussinesqSimpleDAFoam
{

private:

    /// Disallow default bitwise copy construct
    BuoyantBoussinesqSimpleDAFoam(const BuoyantBoussinesqSimpleDAFoam&);

    /// Disallow default bitwise assignment
    void operator=(const BuoyantBoussinesqSimpleDAFoam&);

    /// all the arguments
    char* argsAll_;

public:

    // Constructors

    /// Construct from components
    BuoyantBoussinesqSimpleDAFoam(char* argsAll);

    /// Destructor
    virtual ~BuoyantBoussinesqSimpleDAFoam();

    /// run simpleFoam
    void run();

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

