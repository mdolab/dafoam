/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Child class for rhoSimpleCFoam

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivativeRhoSimpleCFoam_H
#define AdjointDerivativeRhoSimpleCFoam_H

#include "AdjointDerivative.H"
#include "addToRunTimeSelectionTable.H"
#include "simpleControl.H"
#include "physicoChemicalConstants.H"
#include "pressureControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointDerivativeRhoSimpleCFoam Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivativeRhoSimpleCFoam
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

    volScalarField& p_;
    volScalarField pRef_;
    volScalarField pRes_;
    volScalarField pResRef_;
    volScalarField pResPartDeriv_;

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

    simpleControl simple_;

    pressureControl pressureControl_;
    //@}


public:

    TypeName ("rhoSimpleCDAFoam");
    // Constructors

    //- Construct from components
    AdjointDerivativeRhoSimpleCFoam
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
    virtual ~AdjointDerivativeRhoSimpleCFoam()
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
