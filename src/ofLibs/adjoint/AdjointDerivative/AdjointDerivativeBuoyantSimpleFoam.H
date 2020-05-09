/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for buoyantSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivativeBuoyantSimpleFoam_H
#define AdjointDerivativeBuoyantSimpleFoam_H

#include "AdjointDerivative.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "physicoChemicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointDerivativeBuoyantSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivativeBuoyantSimpleFoam
:
public AdjointDerivative
{

private:

protected:

    /// \name These are state variables, reference state variables, flow residuals, and partial derivatives
    //@{
    volVectorField& U_;
    volVectorField URef_;
    volVectorField URes_;
    volVectorField UResRef_;
    volVectorField UResPartDeriv_;

    volScalarField& p_rgh_;
    volScalarField p_rghRef_;
    volScalarField p_rghRes_;
    volScalarField p_rghResRef_;
    volScalarField p_rghResPartDeriv_;
    
    volScalarField& T_;
    volScalarField TRef_;
    volScalarField TRes_;
    volScalarField TResRef_;
    volScalarField TResPartDeriv_;
    
    surfaceScalarField& phi_;  
    surfaceScalarField phiRef_;  
    surfaceScalarField phiRes_;
    surfaceScalarField phiResRef_;
    surfaceScalarField phiResPartDeriv_; 
    
    volScalarField& he_;
    volScalarField& rho_;
    volScalarField& alphat_;
    volScalarField& psi_;

    volScalarField& p_;
    surfaceScalarField& ghf_;
    volScalarField& gh_;
    uniformDimensionedVectorField& g_;
    uniformDimensionedScalarField& hRef_;

    simpleControl simple_;
    //@}


public:

    TypeName ("buoyantSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointDerivativeBuoyantSimpleFoam
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj,
        fluidThermo& thermo
    );

    //- Destructor
    virtual ~AdjointDerivativeBuoyantSimpleFoam()
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
