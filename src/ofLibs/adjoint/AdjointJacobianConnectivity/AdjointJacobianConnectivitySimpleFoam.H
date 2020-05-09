/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for simpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivitySimpleFoam_H
#define AdjointJacobianConnectivitySimpleFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivitySimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivitySimpleFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("simpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivitySimpleFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivitySimpleFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
