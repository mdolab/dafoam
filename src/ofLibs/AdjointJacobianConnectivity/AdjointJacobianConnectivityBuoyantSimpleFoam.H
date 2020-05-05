/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for buoyantSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityBuoyantSimpleFoam_H
#define AdjointJacobianConnectivityBuoyantSimpleFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivityBuoyantSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityBuoyantSimpleFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("buoyantSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityBuoyantSimpleFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityBuoyantSimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
