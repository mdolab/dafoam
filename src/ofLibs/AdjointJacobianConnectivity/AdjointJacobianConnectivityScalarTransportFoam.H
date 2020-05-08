/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for scalarTransportFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivityScalarTransportFoam_H
#define AdjointJacobianConnectivityScalarTransportFoam_H

#include "AdjointJacobianConnectivity.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                 Class AdjointJacobianConnectivityScalarTransportFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivityScalarTransportFoam
:
public AdjointJacobianConnectivity
{

private:


public:

    TypeName ("scalarTransportDAFoam");
    // Constructors

    //- Construct from components
    AdjointJacobianConnectivityScalarTransportFoam
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivityScalarTransportFoam()
    {}
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
