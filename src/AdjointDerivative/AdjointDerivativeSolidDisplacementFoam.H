/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for solidDisplacementFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivativeSolidDisplacementFoam_H
#define AdjointDerivativeSolidDisplacementFoam_H

#include "AdjointDerivative.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
            Class AdjointDerivativeSolidDisplacementFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivativeSolidDisplacementFoam
:
public AdjointDerivative
{

private:

protected:

    /// \name These are state variables, reference state variables, flow residuals, and partial derivatives
    //@{
    volVectorField& D_;
    volVectorField DRef_;
    volVectorField DRes_;
    volVectorField DResRef_;
    volVectorField DResPartDeriv_;
    volTensorField& gradD_;
    volSymmTensorField& sigmaD_;
    volVectorField& divSigmaExp_;
    volScalarField& lambda_;
    volScalarField& mu_;
    volVectorField centriF_;
    //@}

    label isTractionDisplacementBC_;

public:

    TypeName ("solidDisplacementDAFoam");
    // Constructors

    //- Construct from components
    AdjointDerivativeSolidDisplacementFoam
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
    virtual ~AdjointDerivativeSolidDisplacementFoam()
    {}
    
    virtual void calcResiduals
    (
        const label isRef,
        const label isPC,
        const word fvMatrixName="None",
        const label updatePhi=0
    );
    
    virtual void updateIntermediateVariables();

    void updateDAndGradD();
    
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
