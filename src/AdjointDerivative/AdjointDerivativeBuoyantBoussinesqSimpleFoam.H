/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Child class for buoyantBoussinesqSimpleFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivativeBuoyantBoussinesqSimpleFoam_H
#define AdjointDerivativeBuoyantBoussinesqSimpleFoam_H

#include "AdjointDerivative.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "radiationModel.H"
#include "constants.H"
#include "physicoChemicalConstants.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
         Class AdjointDerivativeBuoyantBoussinesqSimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivativeBuoyantBoussinesqSimpleFoam
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

    volScalarField& G_;
    volScalarField GRef_;
    volScalarField GRes_;
    volScalarField GResRef_;
    volScalarField GResPartDeriv_;
    
    surfaceScalarField& phi_;  
    surfaceScalarField phiRef_;  
    surfaceScalarField phiRes_;
    surfaceScalarField phiResRef_;
    surfaceScalarField phiResPartDeriv_; 
    //@}

    /// \name These are intermediate variables
    //@{
    volScalarField& alphat_;
    volScalarField& rhok_;
    volScalarField& p_;
    uniformDimensionedVectorField g_;
    uniformDimensionedScalarField hRef_;
    volScalarField& gh_;
    surfaceScalarField& ghf_;
    dimensionedScalar rhoCpRef_;

    IOdictionary radiationProperties_;
    word radiationOn_;
    dimensionedScalar a_;
    dimensionedScalar e_;
    dimensionedScalar E_;
    volScalarField gamma_;
    //@}
    
    simpleControl simple_;

    //- Temperature source term
    tmp<fvScalarMatrix> ST
    (
        const dimensionedScalar& rhoCp,
        volScalarField& T
    ) const;

    //- Source term component (for power of T^4)
    tmp<volScalarField> Rp() const;

    //- Source term component (constant)
    tmp<volScalarField::Internal> Ru() const;


public:

    TypeName ("buoyantBoussinesqSimpleDAFoam");
    // Constructors

    //- Construct from components
    AdjointDerivativeBuoyantBoussinesqSimpleFoam
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
    virtual ~AdjointDerivativeBuoyantBoussinesqSimpleFoam()
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
