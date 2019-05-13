/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeSimpleTFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeSimpleTFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeSimpleTFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeSimpleTFoam::AdjointDerivativeSimpleTFoam
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
    setResidualClassMemberScalar(T,dimensionSet(0,0,-1,1,0,0,0)),
    setResidualClassMemberPhi(phi),
    alphat_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("alphat")
        )
    ),
    // create simpleControl
    simple_(mesh) 
    
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeSimpleTFoam::calcResiduals
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
    word divTScheme="div(phi,T)";
    if(isPC) 
    {
        divUScheme="div(pc)"; 
        divTScheme="div(pc)";
    }

    // ******** U Residuals **********
    // copied and modified from UEqn.H
    
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi_, U_,divUScheme)
      + adjRAS_.divDevReff(U_)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    
    if (!updatePhi)
    {
        if(isRef) UResRef_  = (UEqn&U_) + fvc::grad(p_);
        else URes_  = (UEqn&U_) + fvc::grad(p_);
        // need to normalize Res. 
        normalizeResiduals(URes);
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
    adjustPhi(phiHbyA, U_, p_);

    tmp<volScalarField> rAtU(rAU);

    if (simple_.consistent())
    {
        rAtU = 1.0/(1.0/rAU - UEqn.H1());
        phiHbyA += fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p_)*mesh_.magSf();
        HbyA -= (rAU - rAtU())*fvc::grad(p_);
    }

    tUEqn.clear();

    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAtU(), p_) == fvc::div(phiHbyA)
    );
    pEqn.setReference(pRefCell, pRefValue);

    if (!updatePhi)
    {
        if(isRef) pResRef_  = pEqn&p_;
        else pRes_  = pEqn&p_;
        // need to normalize Res.
        normalizeResiduals(pRes);
    }

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    if (!updatePhi)
    {
        if(isRef) phiResRef_ = phiHbyA - pEqn.flux() - phi_;
        else phiRes_ = phiHbyA - pEqn.flux() - phi_;
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);
    }

    if(updatePhi) phi_=phiHbyA - pEqn.flux();

    volScalarField alphaEff= adjRAS_.alphaEff();

    fvScalarMatrix TEqn
    (
        fvm::div(phi_, T_, divTScheme)
      - fvm::laplacian(alphaEff, T_)
    );
    
    if (!updatePhi)
    {
        if(isRef) TResRef_  = TEqn&T_;
        else TRes_  = TEqn&T_;
        // need to normalize Res. 
        normalizeResiduals(TRes);
    }
    
    return;

}

void AdjointDerivativeSimpleTFoam::updateIntermediateVariables()
{
    dimensionedScalar Prt1
    (
        "Prt1",
        dimless,
        adjIO_.flowProperties["Prt"]
    );
    
    alphat_ = adjRAS_.getNut()/Prt1;
    alphat_.correctBoundaryConditions();
}


} // End namespace Foam

// ************************************************************************* //
