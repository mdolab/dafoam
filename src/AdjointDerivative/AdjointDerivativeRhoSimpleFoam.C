/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeRhoSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeRhoSimpleFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeRhoSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeRhoSimpleFoam::AdjointDerivativeRhoSimpleFoam
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
    simple_(mesh),
    pressureControl_(p_,rho_,simple_.dict())
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeRhoSimpleFoam::calcResiduals
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

    // set fvMatrix for fast PC construction in NK solver
    setFvMatrix("U",UEqn);

    if (!updatePhi)
    {
        if(isRef) UResRef_  = (UEqn&U_) + fvc::grad(p_);
        else URes_  = (UEqn&U_) + fvc::grad(p_);
        normalizeResiduals(URes);
        scaleResiduals(URes);
    }

    // ******** e Residuals **********
    // copied and modified from EEqn.H
    volScalarField alphaEff("alphaEff", thermo_.alphaEff(alphat_) );

    fvScalarMatrix EEqn
    (
        fvm::div(phi_, he_,  divHEScheme)
      + (
            he_.name() == "e"
          ? fvc::div(phi_, volScalarField("Ekp", 0.5*magSqr(U_) + p_/rho_))
          : fvc::div(phi_, volScalarField("K", 0.5*magSqr(U_)))
        )
      - fvm::laplacian(alphaEff, he_)
    );

    EEqn.relax();

    // set fvMatrix for fast PC construction in NK solver
    // here the stateName is T and the matrix name is EEqn instead of TEqn
    setFvMatrix("T",EEqn);

    if (!updatePhi)
    {
        if(isRef) TResRef_  = EEqn&he_;
        else TRes_  = EEqn&he_;
        normalizeResiduals(TRes);
        scaleResiduals(TRes);
    }


    // ******** p and phi Residuals **********
    // copied and modified from pEqn.H
    volScalarField rAU(1.0/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho_*rAU));
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U_, p_));
    //***************** NOTE *******************
    // we should not use the constrainHbyA function above since it
    // will degrade the accuracy of shape derivatives. Basically, we should
    // not constrain any variable because it will create discontinuity
    volVectorField HbyA("HbyA", U_);
    HbyA = rAU*UEqn.H();

    tUEqn.clear();
    
    surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho_)*fvc::flux(HbyA));
    this->MRF_.makeRelative(fvc::interpolate(rho_), phiHbyA);

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_, rho_, U_, phiHbyA, rhorAUf, this->MRF_);
    
    if (simple_.transonic())
    {
        surfaceScalarField phid
        (
            "phid",
            (fvc::interpolate(psi_)/fvc::interpolate(rho_))*phiHbyA
        );
    
        phiHbyA -= fvc::interpolate(psi_*p_)*phiHbyA/fvc::interpolate(rho_);
    
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          + fvm::div(phid, p_)
          - fvm::laplacian(rhorAUf, p_)
        );

        // Relax the pressure equation to ensure diagonal-dominance
        pEqn.relax();
    
        pEqn.setReference(pressureControl_.refCell(),pressureControl_.refValue());

        // set fvMatrix for fast PC construction in NK solver
        setFvMatrix("p",pEqn);

        // normalize pRes
        if (!updatePhi)
        {
            if(isRef) pResRef_  = pEqn&p_;
            else pRes_  = pEqn&p_;
            normalizeResiduals(pRes);
            scaleResiduals(pRes);
        }
    
        if(updatePhi) phi_=phiHbyA + pEqn.flux();
    
        // ******** phi Residuals **********
        // copied and modified from pEqn.H
        if(isRef) phiResRef_ = phiHbyA + pEqn.flux() - phi_;
        else phiRes_ = phiHbyA + pEqn.flux() - phi_;
    
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);
        scalePhiResiduals(phiRes);
    }
    else
    {
        adjustPhi(phiHbyA, U_, p_);
    
        fvScalarMatrix pEqn
        (
            fvc::div(phiHbyA)
          - fvm::laplacian(rhorAUf, p_)
        );
    
        pEqn.setReference(pressureControl_.refCell(),pressureControl_.refValue());

        // set fvMatrix for fast PC construction in NK solver
        setFvMatrix("p",pEqn);
    
        // normalize pRes
        if (!updatePhi)
        {
            if(isRef) pResRef_  = pEqn&p_;
            else pRes_  = pEqn&p_;
            normalizeResiduals(pRes);
            scaleResiduals(pRes);
        }
    
        if(updatePhi) phi_=phiHbyA + pEqn.flux();
    
        // ******** phi Residuals **********
        // copied and modified from pEqn.H
        if(isRef) phiResRef_ = phiHbyA + pEqn.flux() - phi_;
        else phiRes_ = phiHbyA + pEqn.flux() - phi_;
    
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);
        scalePhiResiduals(phiRes);  
    }
    
    return;

}

void AdjointDerivativeRhoSimpleFoam::updateIntermediateVariables()
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
