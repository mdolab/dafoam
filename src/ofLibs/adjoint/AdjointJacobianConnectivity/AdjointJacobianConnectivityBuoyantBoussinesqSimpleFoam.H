/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for buoyantBoussinesqSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam_H
#define AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
  Class AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("buoyantBoussinesqSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityBuoyantBoussinesqSimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
