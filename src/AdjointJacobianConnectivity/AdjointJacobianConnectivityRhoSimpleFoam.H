/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for rhoSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityRhoSimpleFoam_H
#define AdjointJacobianConnectivityRhoSimpleFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivityRhoSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityRhoSimpleFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("rhoSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityRhoSimpleFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityRhoSimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
