/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for laplacianFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivativeLaplacianFoam_H
#define AdjointDerivativeLaplacianFoam_H

#include "AdjointDerivative.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointDerivativeLaplacianFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivativeLaplacianFoam
:
public AdjointDerivative
{

private:

protected:

    /// \name These are state variables, reference state variables, flow residuals, and partial derivatives
    //@{

    volScalarField& T_;
    volScalarField TRef_;
    volScalarField TRes_;
    volScalarField TResRef_;
    volScalarField TResPartDeriv_;
    //@}

public:

    TypeName ("laplacianDAFoam");
    // Constructors

    //- Construct from components
    AdjointDerivativeLaplacianFoam
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj
    );

    //- Destructor
    virtual ~AdjointDerivativeLaplacianFoam()
    {}
    
    virtual void calcResiduals
    (
        const label isRef,
        const label isPC,
        const word fvMatrixName="None",
        const label updatePhi=0
    );
    
    virtual void updateIntermediateVariables();

    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
