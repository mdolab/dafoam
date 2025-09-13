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

    usePCFlux_ = daOption.getAllOptions().subDict("adjEqnOption").lookupOrDefault<label>("hisaPCFlux", 1);
    forceJSTFlux_ = daOption.getAllOptions().subDict("adjEqnOption").lookupOrDefault<label>("hisaForceJSTFlux", 0);
    removeTauMCOff_ = daOption.getAllOptions().subDict("adjEqnOption").lookupOrDefault<label>("hisaRemoveTauMCOff", 1);

    const IOdictionary& fvSchemes = mesh_.thisDb().lookupObject<IOdictionary>("fvSchemes");
    jst_k2_ = fvSchemes.lookupOrDefault<scalar>("jst_k2", 0.5);
    jst_k4_ = fvSchemes.lookupOrDefault<scalar>("jst_k4", 0.02);
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

    // forceJSTFlux_: force to use JST flux for the adjoint residual
    // the adjoint for AUSM flux does not converge well, so to bypass this issue
    // we can still use AUSM for primal, but the flux in the residuals will always be JST
    // this will obviously downgrade the adjoint accuracy, but it will allow the adjoint
    // to converge at least
    if (forceJSTFlux_)
    {
        // no matter what scheme is defined in fvSchemes, always use JST flux
        this->calcFluxJST(phi_, phiUp_, phiEp_, Up_);
    }
    else
    {
        if (isPC && usePCFlux_)
        {
            // force to use the first order FluxLaxFriedrichs scheme for PC mat
            this->calcFluxLaxFriedrichs(phi_, phiUp_, phiEp_, Up_);
        }
        else
        {
            fluxScheme_.calcFlux(phi_, phiUp_, phiEp_, Up_);
        }
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

        if (isPC && removeTauMCOff_)
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

void DAResidualHisaFoam::calcFluxJST(
    surfaceScalarField& phi,
    surfaceVectorField& phiUp,
    surfaceScalarField& phiEp,
    surfaceVectorField& Up)
{
    // calculate the flux using the JST scheme

    const volScalarField& p = thermo_.p();
    const fvMesh& mesh = mesh_;

    // add the central term
    phi = linearInterpolate(rhoU_) & mesh.Sf();
    phiUp = linearInterpolate(rhoU_ * U_ + p * tensor::I) & mesh.Sf();
    phiEp = linearInterpolate((rhoE_ + p) * U_) & mesh.Sf();

    // force to use oriented fluxes with signs
    phi.setOriented();
    phiUp.setOriented();
    phiEp.setOriented();

    // local spectral radius at faces  c + |U| * n
    volScalarField c(sqrt(thermo_.gamma() / thermo_.psi()));
    surfaceScalarField specR = (fvc::interpolate(c) + mag(fvc::interpolate(U_) & mesh_.Sf() / mesh_.magSf()));
    // specR should have no signs
    specR.setOriented(false);

    // shock sensor field: it can be pressure or entropy
    volScalarField sField("sField", p);
    // calculate the sensor = |s_N - s_P | / |s_N + s_P + eps| we use the first order difference
    dimensionedScalar smallS(
        "smallS",
        sField.dimensions(),
        1e-16);
    // |s_N - s_P |
    surfaceScalarField sDiff = mag(fv::orthogonalSnGrad<scalar>(mesh).snGrad(sField)) / mesh.deltaCoeffs();
    // |s_N + s_P |
    surfaceScalarField sSum = 2.0 * linearInterpolate(sField);
    surfaceScalarField sensor = sDiff / (sSum + smallS);
    sensor.setOriented(false);
    // bound it between 0 and 1
    sensor = min(max(sensor, scalar(0)), scalar(1));

    // JST artificial dissipation coefficients
    surfaceScalarField eps2 = jst_k2_ * sensor;
    eps2.setOriented(false);
    surfaceScalarField eps4 = max(scalar(0.0), jst_k4_ - eps2);
    eps4.setOriented(false);

    // dPhi = snGrad * d. we have to force to use the ortho snGrad without any corrections
    surfaceScalarField dRho = fv::orthogonalSnGrad<scalar>(mesh).snGrad(rho_) / mesh.deltaCoeffs();
    surfaceVectorField dRhoU = fv::orthogonalSnGrad<vector>(mesh).snGrad(rhoU_) / mesh.deltaCoeffs();
    surfaceScalarField dRhoE = fv::orthogonalSnGrad<scalar>(mesh).snGrad(rhoE_) / mesh.deltaCoeffs();
    dRho.setOriented();
    dRhoU.setOriented();
    dRhoE.setOriented();

    // sum over all face to get d2Rho_dx for each cell
    // NOTE: the fvc::div operator sums over all the face (without multiplying the area) and then divide it by the volume
    // So we have to manually multiply the surface area
    volScalarField d2Rho_dx = fvc::div(dRho * mesh.magSf());
    volVectorField d2RhoU_dx = fvc::div(dRhoU * mesh.magSf());
    volScalarField d2RhoE_dx = fvc::div(dRhoE * mesh.magSf());

    // apply the snGrad on the d2Rho_dx to get 3rd order differences
    // Note we have to do two mesh.deltaCoeffs() here because the first one is for snGrad and the 2nd one
    // is for d2Rho_dx (we want difference d2Rho, not d2Rho/dx). This makes sure the unit for d3Rho is the same as rho
    surfaceScalarField d3Rho = fv::orthogonalSnGrad<scalar>(mesh).snGrad(d2Rho_dx) / mesh.deltaCoeffs() / mesh.deltaCoeffs();
    surfaceVectorField d3RhoU = fv::orthogonalSnGrad<vector>(mesh).snGrad(d2RhoU_dx) / mesh.deltaCoeffs() / mesh.deltaCoeffs();
    surfaceScalarField d3RhoE = fv::orthogonalSnGrad<scalar>(mesh).snGrad(d2RhoE_dx) / mesh.deltaCoeffs() / mesh.deltaCoeffs();
    d3Rho.setOriented();
    d3RhoU.setOriented();
    d3RhoE.setOriented();

    // add the artificial fluxes:
    // Note when integrating the dRho fluxes over the control volume, we get d2Rho/dx2 (2nd order dissipation)
    // same applies to d3Rho, integrating it will get d4Rho/dx4 (fourth order dissipation)
    phi -= (eps2 * dRho - eps4 * d3Rho) * mesh.magSf() * specR;
    phiUp -= (eps2 * dRhoU - eps4 * d3RhoU) * mesh.magSf() * specR;
    phiEp -= (eps2 * dRhoE - eps4 * d3RhoE) * mesh.magSf() * specR;

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
