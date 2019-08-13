/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeTurboFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeTurboFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeTurboFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeTurboFoam::AdjointDerivativeTurboFoam
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj,
    fluidThermo& thermo
)
    :
    AdjointDerivative(mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj,thermo),
    // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
    setResidualClassMemberVector(U,dimensionSet(1,-2,-2,0,0,0,0)),
    setResidualClassMemberScalar(p,dimensionSet(1,-3,-1,0,0,0,0)),
    setResidualClassMemberScalar(T,dimensionSet(1,-1,-3,0,0,0,0)),
    setResidualClassMemberPhi(phi),
    // these are intermediate variables or objects
    URel_
    (
        const_cast<volVectorField&>
        (
            db_.lookupObject<volVectorField>("URel")
        )
    ),
    he_
    (
        thermo.he()
    ),
    rho_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("rho")
        )
    ),
    alphat_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("alphat")
        )
    ),
    psi_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("thermo:psi")
        )
    ),
    // create simpleControl
    simple_(mesh),
    pressureControl_(p_,rho_,simple_.dict())
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeTurboFoam::calcResiduals
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
    word divHEScheme="div(phi,e)";
    word divPhidPScheme="div(phid,p)";
    if (he_.name()=="h") divHEScheme="div(phi,h)";
    if(isPC)
    {
        divUScheme="div(pc)";
        divHEScheme="div(pc)";
        divPhidPScheme="div(pc)";
    }

    // ******** U Residuals **********
    // copied and modified from UEqn.H
    //this->MRF_.correctBoundaryVelocity(U_);

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi_, U_, divUScheme)
      + this->MRF_.DDt(rho_, U_)
      + adjRAS_.divDevRhoReff(U_)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    if(isRef) UResRef_  = (UEqn&U_) + fvc::grad(p_);
    else URes_  = (UEqn&U_) + fvc::grad(p_);
    // need to normalize residuals
    normalizeResiduals(URes);

    // ******** e Residuals **********
    // copied and modified from EEqn.H
    volSymmTensorField Teff = - adjRAS_.devRhoReff();
    volScalarField alphaEff("alphaEff", thermo_.alphaEff(alphat_) );

    URel_ == U_;
    this->MRF_.makeRelative(URel_);

    fvScalarMatrix EEqn
    (
        fvm::div(phi_, he_, divHEScheme)
      + (
            he_.name() == "e"
          ? fvc::div(phi_, volScalarField("Ekp", 0.5*magSqr(U_) + p_/rho_))
          : fvc::div(phi_, volScalarField("K", 0.5*magSqr(U_))) - fvc::div( Teff.T()&U_ ) + fvc::div( p_*(U_-URel_) )
        )
      - fvm::Sp(fvc::div(phi_), he_)
      - fvm::laplacian(alphaEff, he_)
    );

    EEqn.relax();

    if(isRef) TResRef_  = EEqn&he_;
    else TRes_  = EEqn&he_;
    // need to normalize residuals
    normalizeResiduals(TRes);
    
    // ******** p and phi Residuals **********
    // copied and modified from pEqn.H
    volScalarField AU(UEqn.A());
    volScalarField AtU(AU - UEqn.H1());
    // need to create a tmp field to store U_ because U_=UEqn.H()/AU changes U_!
    volVectorField UTmp("UTmp",U_);
    U_=UEqn.H()/AU;
    

    volScalarField rAU(1.0/UEqn.A());
    tUEqn.clear();

    if (simple_.transonic())
    {
        surfaceScalarField phid
        (
            "phid",
            fvc::interpolate(psi_)*(fvc::interpolate(U_) & mesh_.Sf())
        );
    
        this->MRF_.makeRelative(fvc::interpolate(psi_), phid);
    
        fvScalarMatrix pEqn
        (
            fvm::div(phid, p_, divPhidPScheme)
          - fvm::laplacian(rho_*rAU, p_)
        );

        // Relax the pressure equation to maintain diagonal dominance
        pEqn.relax();
    
        pEqn.setReference(pressureControl_.refCell(),pressureControl_.refValue());
        
        // normalize pRes
        if (!updatePhi)
        {
            if(isRef) pResRef_  = pEqn&p_;
            else pRes_  = pEqn&p_;
            // need to normalize Res. 
            normalizeResiduals(pRes);
        }
    
        if(updatePhi) phi_ == pEqn.flux();
    
        // ******** phi Residuals **********
        // copied and modified from pEqn.H
        if(isRef) phiResRef_ == pEqn.flux() - phi_;
        else phiRes_ == pEqn.flux() - phi_;
    
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);

    }
    else
    {
        surfaceScalarField phiTmp("phiTmp",phi_);
        
        phi_ = fvc::interpolate(rho_*U_) & mesh_.Sf();
    
        this->MRF_.makeRelative(fvc::interpolate(rho_), phi_);
    
        adjustPhi(phi_, U_, p_);
        phi_ += fvc::interpolate(rho_/AtU - rho_/AU)*fvc::snGrad(p_)*mesh_.magSf();

        fvScalarMatrix pEqn
        (
            fvc::div(phi_)
          - fvm::laplacian(rho_/AtU, p_)
        );
    
        pEqn.setReference(pressureControl_.refCell(),pressureControl_.refValue());
        
        // normalize pRes
        if (!updatePhi)
        {
            if(isRef) pResRef_  = pEqn&p_;
            else pRes_  = pEqn&p_;
            // need to normalize Res. 
            normalizeResiduals(pRes);
        }
    
        if(updatePhi) phi_ += pEqn.flux();
    
        // ******** phi Residuals **********
        // copied and modified from pEqn.H
        // TODO: the phiRes is not zero, need to fix
        if(isRef) phiResRef_ == pEqn.flux() ;
        else phiRes_ == pEqn.flux();
    
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);

        // assign phi_ back
        phi_==phiTmp;

    }
    
    // assign U_ back
    U_=UTmp;
    U_.correctBoundaryConditions();
    

    return;

}

void AdjointDerivativeTurboFoam::updateIntermediateVariables()
{
    // ********************** NOTE *****************
    // we assume hePsiThermo 
    // TODO: need to do this using built-in openfoam functions.
    
    // we need to:
    // 1, update psi based on T, psi=1/(R*T)
    // 2, update rho based on p and psi, rho=psi*p
    // 3, update E based on T, p and rho, E=Cp*T-p/rho
    // 4, update alphat
    // 5, update velocity boundary based on MRF

    scalar RR=Foam::constant::thermodynamic::RR; // 8314.4700665  gas constant in OpenFOAM
    dimensionedScalar R
    (
        "R",
        dimensionSet(0,2,-2,-1,0,0,0),
        RR/adjIO_.flowProperties["molWeight"]
    );
    psi_=1/T_/R;
    rho_=psi_*p_;
    
    scalar Cp=adjIO_.flowProperties["Cp"];
    forAll(he_,idxI)
    {
        if( he_.name() == "e") he_[idxI]=Cp*T_[idxI]-p_[idxI]/rho_[idxI];
        else he_[idxI]=Cp*T_[idxI]; 
    }
    he_.correctBoundaryConditions();
    
    // NOTE: for compressible flow, Prt is defined in RAS-turbulenceProperties
    // see EddyDiffusivity.C for reference
    dimensionedScalar Prt1
    (
        "Prt1",
        dimless,
        adjIO_.flowProperties["Prt"]
    );
    
    alphat_ = rho_*adjRAS_.getNut()/Prt1;
    alphat_.correctBoundaryConditions();

    this->MRF_.correctBoundaryVelocity(U_);

}


} // End namespace Foam

// ************************************************************************* //
