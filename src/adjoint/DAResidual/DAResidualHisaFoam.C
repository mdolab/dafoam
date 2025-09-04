/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAResidualHisaFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualHisaFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualHisaFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualHisaFoam::DAResidualHisaFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberVector(U, dimensionSet(1, -2, -2, 0, 0, 0, 0)),
      setResidualClassMemberScalar(p, dimensionSet(1, -3, -1, 0, 0, 0, 0)),
      setResidualClassMemberScalar(T, dimensionSet(1, -1, -3, 0, 0, 0, 0)),
      phi_(mesh.lookupObjectRef<surfaceScalarField>("phi")),
      phiUp_(mesh.lookupObjectRef<surfaceVectorField>("phiUp")),
      phiEp_(mesh.lookupObjectRef<surfaceScalarField>("phiEp")),
      Up_(mesh.lookupObjectRef<surfaceVectorField>("Up")),
      rho_(mesh.lookupObjectRef<volScalarField>("rho")),
      rhoU_(mesh.lookupObjectRef<volVectorField>("rhoU")),
      rhoE_(mesh.lookupObjectRef<volScalarField>("rhoE")),
      psi_(mesh_.thisDb().lookupObjectRef<volScalarField>("thermo:psi")),
      fluxScheme_(mesh.lookupObjectRef<fluxScheme>("fluxScheme")),
      solverModule_(mesh.lookupObjectRef<solverModule>("hisaSolver")),
      thermo_(mesh.thisDb().lookupObjectRef<fluidThermo>("thermophysicalProperties")),
      he_(thermo_.he()),
      turbulence_(mesh.thisDb().lookupObjectRef<compressible::turbulenceModel>("turbulenceProperties"))
{
    // get molWeight and Cp from thermophysicalProperties
    const IOdictionary& thermoDict = mesh.thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
    dictionary mixSubDict = thermoDict.subDict("mixture");
    dictionary specieSubDict = mixSubDict.subDict("specie");
    molWeight_ = specieSubDict.getScalar("molWeight");
    dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
    Cp_ = thermodynamicsSubDict.getScalar("Cp");

    if (daOption_.getOption<label>("debug"))
    {
        Info << "molWeight " << molWeight_ << endl;
        Info << "Cp " << Cp_ << endl;
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualHisaFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    URes_.clear();
    pRes_.clear();
    TRes_.clear();
}

void DAResidualHisaFoam::calcResiduals(const dictionary& options)
{
    /*
    Description:
        This is the function to compute residuals.
    
    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,U), div(phi,e)

        p_, T_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        URes_, pRes_, TRes_, phiRes_: residual field variables
    */

    label isPC = options.getLabel("isPC");

    if (isPC)
    {
        // force to use the first order FluxLaxFriedrichs scheme for PC mat
        this->calcFluxLaxFriedrichs(phi_, phiUp_, phiEp_, Up_);
    }
    else
    {
        fluxScheme_.calcFlux(phi_, phiUp_, phiEp_, Up_);
    }

    pRes_ = -fvc::div(phi_);

    URes_ = -fvc::div(phiUp_);

    TRes_ = -fvc::div(phiEp_);

    if (!solverModule_.inviscid())
    {

        volScalarField muEff("muEff", turbulence_.muEff());
        volScalarField alphaEff("alphaEff", turbulence_.alphaEff());

        volTensorField tauMC = mesh_.lookupObjectRef<volTensorField>("tauMC");

        tauMC = muEff * dev2(Foam::T(fvc::grad(U_)));

        if (isPC)
        {
            this->removeOffDiagonals(tauMC);
        }

        URes_ += fvc::laplacian(muEff, U_);
        URes_ += fvc::div(tauMC);

        surfaceScalarField sigmaDotU(
            "sigmaDotU",
            (
                fvc::interpolate(muEff) * fvc::snGrad(U_)
                + (mesh_.Sf() / mesh_.magSf() & fvc::interpolate(tauMC)))
                & Up_);
        TRes_ += fvc::div(sigmaDotU);

        volScalarField eCalc("eCalc", rhoE_ / rho_ - 0.5 * magSqr(U_)); // prevent e inheriting BC from T in thermo
        TRes_ += fvc::laplacian(alphaEff, eCalc, "laplacian(alphaEff,e)");
    }

    normalizeResiduals(pRes);
    normalizeResiduals(URes);
    normalizeResiduals(TRes);

    scalar URef = daOption_.getSubDictOption<scalar>("normalizeStates", "U");
    scalar TRef = daOption_.getSubDictOption<scalar>("normalizeStates", "T");
    URes_ = URes_ / URef;
    TRes_ = TRes_ / TRef / Cp_;
}

void DAResidualHisaFoam::calcFluxLaxFriedrichs(
    surfaceScalarField& phi,
    surfaceVectorField& phiUp,
    surfaceScalarField& phiEp,
    surfaceVectorField& Up)
{
    // calculate the flux using laxFriedrichs scheme for PC mat

    const volScalarField& p = thermo_.p();
    const fvMesh& mesh = mesh_;

    phi = linearInterpolate(rhoU_) & mesh.Sf();
    phiUp = linearInterpolate(rhoU_ * U_ + p * tensor::I) & mesh.Sf();
    phiEp = linearInterpolate((rhoE_ + p) * U_) & mesh.Sf();

    phi.setOriented();
    phiUp.setOriented();
    phiEp.setOriented();

    // Wave speed: Lax-Friedrich flux approximation of left-hand side Jacobian
    volScalarField c(sqrt(thermo_.gamma() / thermo_.psi()));
    tmp<surfaceScalarField> lambdaConv;

    lambdaConv = (fvc::interpolate(c) + mag(fvc::interpolate(U_) & mesh_.Sf() / mesh_.magSf())) / mesh_.deltaCoeffs();

    lambdaConv->setOriented(false);

    phi -= 0.5 * lambdaConv() * fv::orthogonalSnGrad<scalar>(mesh).snGrad(rho_) * mesh.magSf();
    phiUp -= 0.5 * lambdaConv() * fv::orthogonalSnGrad<vector>(mesh).snGrad(rhoU_) * mesh.magSf();
    phiEp -= 0.5 * lambdaConv() * fv::orthogonalSnGrad<scalar>(mesh).snGrad(rhoE_) * mesh.magSf();

    // Face velocity for sigmaDotU (turbulence term)
    Up = linearInterpolate(U_) * mesh_.magSf();
}

void DAResidualHisaFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables

        ********************** NOTE *****************
        we assume hePsiThermo, pureMixture, perfectGas, hConst, and const transport
        TODO: need to do this using built-in openfoam functions.
    
        we need to:
        1, update psi based on T, psi=1/(R*T)
        2, update rho based on p and psi, rho=psi*p
        3, update E based on T, p and rho, E=Cp*T-p/rho
        4, update velocity boundary based on MRF
    */

    // 8314.4700665  gas constant in OpenFOAM
    // src/OpenFOAM/global/constants/thermodynamic/thermodynamicConstants.H
    scalar RR = Foam::constant::thermodynamic::RR;

    // R = RR/molWeight
    // Foam::specie::R() function in src/thermophysicalModels/specie/specie/specieI.H
    dimensionedScalar R(
        "R1",
        dimensionSet(0, 2, -2, -1, 0, 0, 0),
        RR / molWeight_);

    // psi = 1/T/R
    // see src/thermophysicalModels/specie/equationOfState/perfectGas/perfectGasI.H
    psi_ = 1.0 / T_ / R;

    // rho = psi*p
    // see src/thermophysicalModels/basic/psiThermo/psiThermo.C
    rho_ = psi_ * p_;

    dimensionedScalar Cp(
        "Cp1",
        dimensionSet(0, 2, -2, -1, 0, 0, 0),
        Cp_);

    // Hs = Cp*T
    // see Hs() in src/thermophysicalModels/specie/thermo/hConst/hConstThermoI.H
    // here the H departure EquationOfState::H(p, T) will be zero for perfectGas
    // Es = Hs - p/rho = Hs - T * R;
    // see Es() in src/thermophysicalModels/specie/thermo/thermo/thermoI.H
    // **************** NOTE ****************
    // See the comment from the rho.relax() call, if we write he_=Cp*T-p/rho, the
    // accuracy of he_ may be impact by the inaccurate rho. So here we want to
    // rewrite he_ as he_ = Cp * T_ - T_ * R instead, such that we dont include rho
    // **************** NOTE ****************
    if (he_.name() == "e")
    {
        he_ = Cp * T_ - T_ * R;
    }
    else
    {
        FatalErrorInFunction
            << "Only energy type internalEnergy supported."
            << nl << exit(FatalError);
    }
    he_.correctBoundaryConditions();

    rhoU_.ref() = rho_.internalField() * U_.internalField();

    rhoE_.ref() = rho_.internalField() * (he_.internalField() + 0.5 * magSqr(U_.internalField()));

    rho_.correctBoundaryConditions();
    rhoU_.boundaryFieldRef() = rho_.boundaryField() * U_.boundaryField();
    rhoE_.boundaryFieldRef() = rho_.boundaryField() * (he_.boundaryField() + 0.5 * magSqr(U_.boundaryField()));
}

void DAResidualHisaFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */
    U_.correctBoundaryConditions();
    p_.correctBoundaryConditions();
    T_.correctBoundaryConditions();
}

void DAResidualHisaFoam::removeOffDiagonals(volTensorField& tauMC)
{
    forAll(tauMC, cellI)
    {
        tauMC[cellI].xy() = 0.0;
        tauMC[cellI].xz() = 0.0;
        tauMC[cellI].yx() = 0.0;
        tauMC[cellI].yz() = 0.0;
        tauMC[cellI].zx() = 0.0;
        tauMC[cellI].zy() = 0.0;
    }
    forAll(tauMC.boundaryField(), patchI)
    {
        forAll(tauMC.boundaryField()[patchI], faceI)
        {
            tauMC.boundaryFieldRef()[patchI][faceI].xy() = 0.0;
            tauMC.boundaryFieldRef()[patchI][faceI].xz() = 0.0;
            tauMC.boundaryFieldRef()[patchI][faceI].yx() = 0.0;
            tauMC.boundaryFieldRef()[patchI][faceI].yz() = 0.0;
            tauMC.boundaryFieldRef()[patchI][faceI].zx() = 0.0;
            tauMC.boundaryFieldRef()[patchI][faceI].zy() = 0.0;
        }
    }
}

} // End namespace Foam

// ************************************************************************* //
