/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeSimpleFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeSimpleFoam::AdjointDerivativeSimpleFoam
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj
)
    :
    AdjointDerivative(mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj),
    // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
    setResidualClassMemberVector(U,dimensionSet(0,1,-2,0,0,0,0)),
    setResidualClassMemberScalar(p,dimensionSet(0,0,-1,0,0,0,0)),
    setResidualClassMemberPhi(phi),
    // create simpleControl
    simple_(mesh) 
    
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeSimpleFoam::calcResiduals
(
    const label isRef,
    const label isPC,
    const word fvMatrixName,
    const label updatePhi
)
{
    // We dont support MRF and fvOptions so all the related lines are commented 
    // out for now
    
    word divUScheme="div(phi,U)";
    if(isPC) divUScheme="div(pc)";

    // ******** U Residuals **********
    // copied and modified from UEqn.H

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi_, U_,divUScheme)
      + this->MRF_.DDt(U_)
      + adjRAS_.divDevReff(U_)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    //UEqn.relax();

    // set fvMatrix for fast PC construction in NK solver
    setFvMatrix("U",UEqn);

    if (!updatePhi)
    {
        if(isRef) UResRef_  = (UEqn&U_) + fvc::grad(p_);
        else URes_  = (UEqn&U_) + fvc::grad(p_);
        normalizeResiduals(URes);
        scaleResiduals(URes);
    }

    // ******** p Residuals **********
    // copied and modified from pEqn.H
    // NOTE manually set pRefCell and pRefValue
    label pRefCell=0;
    scalar pRefValue=0.0;
    
    // Note: relax UEqn after the URes is calculated
    UEqn.relax();
    
    volScalarField rAU(1.0/UEqn.A());
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U_, p_));
    //***************** NOTE *******************
    // we should not use the constrainHbyA function above since it
    // will degrade the accuracy of shape derivatives. Basically, we should
    // not constrain any variable because it will create discontinuity
    volVectorField HbyA("HbyA", U_);
    HbyA = rAU*UEqn.H();

    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
    this->MRF_.makeRelative(phiHbyA);
    adjustPhi(phiHbyA, U_, p_);

    tmp<volScalarField> rAtU(rAU);

    if (simple_.consistent())
    {
        rAtU = 1.0/(1.0/rAU - UEqn.H1());
        phiHbyA += fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p_)*mesh_.magSf();
        HbyA -= (rAU - rAtU())*fvc::grad(p_);
    }

    tUEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_, U_, phiHbyA, rAtU(), this->MRF_);
    
    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtU(), p_) == fvc::div(phiHbyA)
    );
    pEqn.setReference(pRefCell, pRefValue);

    // set fvMatrix for fast PC construction in NK solver
    setFvMatrix("p",pEqn);

    if (!updatePhi)
    {
        if(isRef) pResRef_  = pEqn&p_;
        else pRes_  = pEqn&p_;
        normalizeResiduals(pRes);
        scaleResiduals(pRes);
    }

    if(updatePhi) phi_=phiHbyA - pEqn.flux();

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    if(isRef) phiResRef_ = phiHbyA - pEqn.flux() - phi_;
    else phiRes_ = phiHbyA - pEqn.flux() - phi_;
    // need to normalize phiRes
    normalizePhiResiduals(phiRes);
    scalePhiResiduals(phiRes);   
    
    return;

}

void AdjointDerivativeSimpleFoam::updateIntermediateVariables()
{
    // update velocity boundary based on MRF
    this->MRF_.correctBoundaryVelocity(U_);
}

} // End namespace Foam

// ************************************************************************* //
