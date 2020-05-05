/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeBuoyantSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeBuoyantSimpleFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeBuoyantSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeBuoyantSimpleFoam::AdjointDerivativeBuoyantSimpleFoam
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
    setResidualClassMemberScalar(p_rgh,dimensionSet(1,-3,-1,0,0,0,0)),
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
    p_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("p")
        )
    ),
    ghf_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("ghf")
        )
    ),
    gh_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("gh")
        )
    ),
    g_
    (
        const_cast<uniformDimensionedVectorField&>
        (
            db_.lookupObject<uniformDimensionedVectorField>("g")
        )
    ),
    hRef_
    (
        const_cast<uniformDimensionedScalarField&>
        (
            db_.lookupObject<uniformDimensionedScalarField>("hRef")
        )
    ),
    // create simpleControl
    simple_(mesh)
{
    this->copyStates("Var2Ref"); // copy states to statesRef
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeBuoyantSimpleFoam::calcResiduals
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
    if (he_.name()=="h") divHEScheme="div(phi,h)";
    if(isPC)
    {
        divUScheme="div(pc)";
        divHEScheme="div(pc)";
    }

    // ******** U Residuals **********
    // copied and modified from UEqn.H

    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi_, U_,divUScheme)
      + adjRAS_.divDevRhoReff(U_)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    if (!updatePhi)
    {
        if(isRef)
        {
            UResRef_  = (UEqn&U_) - fvc::reconstruct
                                    (
                                        (
                                          - ghf_*fvc::snGrad(rho_)
                                          - fvc::snGrad(p_rgh_)
                                        )*mesh_.magSf()
                                    );
        }
        else
        {
            URes_  = (UEqn&U_) - fvc::reconstruct
                                 (
                                     (
                                       - ghf_*fvc::snGrad(rho_)
                                       - fvc::snGrad(p_rgh_)
                                     )*mesh_.magSf()
                                 );
        }
        // need to normalize Res. NOTE: URes is normalized by its cell volume by default!
        normalizeResiduals(URes);
    }

    // ******** T Residuals **********
    // copied and modified from EEqn.H

    volScalarField alphaEff("alphaEff", thermo_.alphaEff(alphat_) );

    fvScalarMatrix EEqn
    (
        fvm::div(phi_, he_, divHEScheme)
      + (
            he_.name() == "e"
          ? fvc::div(phi_, volScalarField("Ekp", 0.5*magSqr(U_) + p_rgh_/rho_))
          : fvc::div(phi_, volScalarField("K", 0.5*magSqr(U_)))
        )
      - fvm::laplacian(alphaEff, he_)
     ==
        rho_*(U_&g_)
    );

    EEqn.relax();
 
    if (!updatePhi)
    {
        if(isRef) TResRef_  = EEqn&he_;
        else TRes_  = EEqn&he_;
        // need to normalize Res. NOTE: eRes is normalized by its cell volume by default!
        normalizeResiduals(TRes);
    }

    // ******** p Residuals **********
    // copied and modified from pEqn.H
    // NOTE manually set pRefCell and pRefValue
    label pRefCell=0;

    volScalarField rAU("rAU",1.0/UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho_*rAU));
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U_, p_rgh_));
    // ***************** NOTE *******************
    // we should not use the constrainHbyA function above since it
    // will degrade the accuracy of shape derivatives. Basically, we should
    // not constrain any variable because it will create discontinuity
    volVectorField HbyA("HbyA", U_);
    HbyA = rAU*UEqn.H();

    tUEqn.clear();

    surfaceScalarField phig(-rhorAUf*ghf_*fvc::snGrad(rho_)*mesh_.magSf());

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::flux(rho_*HbyA)
    );
        
    phiHbyA += phig;
    
    fvScalarMatrix p_rghEqn
    (
        fvm::laplacian(rhorAUf, p_rgh_) == fvc::div(phiHbyA)
    );
    p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh_, pRefCell));

    // normalize pRes
    if (!updatePhi)
    {
        if(isRef) p_rghResRef_  = p_rghEqn&p_rgh_;
        else p_rghRes_  = p_rghEqn&p_rgh_;
        // need to normalize Res.
        normalizeResiduals(p_rghRes);
    }

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    if (!updatePhi)
    {
        if(isRef) phiResRef_ = phiHbyA - p_rghEqn.flux() - phi_;
        else phiRes_ = phiHbyA - p_rghEqn.flux() - phi_;
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);
    }

    if(updatePhi) phi_ = phiHbyA - p_rghEqn.flux();

    return;

}

void AdjointDerivativeBuoyantSimpleFoam::updateIntermediateVariables()
{
    // ********************** NOTE *****************
    // TODO: need to do this using built-in openfoam functions.
    
    // we need to:
    // 1, update psi based on T, psi=1/(R*T)
    // 2, update rho based on p and psi, rho=psi*p
    // 3, update he based on T, p and rho, E=Cp*T-p/rho, H = Cp*T
    // 4, update alphat

    scalar RR=Foam::constant::thermodynamic::RR; // 8314.4700665  gas constant in OpenFOAM
    dimensionedScalar R
    (
        "R",
        dimensionSet(0,2,-2,-1,0,0,0),
        RR/adjIO_.flowProperties["molWeight"]
    );
    dimensionedScalar ghRef
    (
        mag(g_.value()) > SMALL
      ? g_ & (cmptMag(g_.value())/mag(g_.value()))*hRef_
      : dimensionedScalar("ghRef", g_.dimensions()*dimLength, 0)
    );
    gh_ = (g_&mesh_.C()) - ghRef;
    ghf_ = (g_&mesh_.Cf()) - ghRef;
    p_ = p_rgh_ + rho_*gh_;
    psi_=1/T_/R;
    rho_=psi_*p_;
    
    scalar Cp=adjIO_.flowProperties["Cp"];
    forAll(he_,idxI)
    {
        if( he_.name() == "e") he_[idxI]=Cp*T_[idxI]-p_rgh_[idxI]/rho_[idxI];
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

}

} // End namespace Foam

// ************************************************************************* //
