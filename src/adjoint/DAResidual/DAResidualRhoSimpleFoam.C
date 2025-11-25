/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAResidualRhoSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualRhoSimpleFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualRhoSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualRhoSimpleFoam::DAResidualRhoSimpleFoam(
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
      setResidualClassMemberPhi(phi),
      fvSource_(const_cast<volVectorField&>(
          mesh_.thisDb().lookupObject<volVectorField>("fvSource"))),
      fvSourceEnergy_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("fvSourceEnergy"))),
      fvOptions_(fv::options::New(mesh)),
      thermo_(const_cast<fluidThermo&>(
          mesh_.thisDb().lookupObject<fluidThermo>("thermophysicalProperties"))),
      he_(thermo_.he()),
      rho_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("rho"))),
      alphat_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("alphat"))),
      psi_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("thermo:psi"))),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      // create simpleControl
      simple_(const_cast<fvMesh&>(mesh)),
      pressureControl_(p_, rho_, simple_.dict()),
      MRF_(const_cast<IOMRFZoneListDF&>(
          mesh_.thisDb().lookupObject<IOMRFZoneListDF>("MRFProperties")))
{

    // initialize fvSource
    const dictionary& allOptions = daOption.getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
    }

    // this is just a dummy call because we need to run the constrain once
    // to initialize fvOptions, before we can use it. Otherwise, we may get
    // a seg fault when we call fvOptions_.correct(U_) in updateIntermediateVars
    fvVectorMatrix UEqn(
        fvm::div(phi_, U_)
        - fvOptions_(rho_, U_));
    fvOptions_.constrain(UEqn);
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualRhoSimpleFoam::clear()
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
    phiRes_.clear();
}

void DAResidualRhoSimpleFoam::calcResiduals(const dictionary& options)
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

    word divUScheme = "div(phi,U)";
    word divHEScheme = "div(phi,e)";

    if (he_.name() == "h")
    {
        divHEScheme = "div(phi,h)";
    }

    if (isPC)
    {
        divUScheme = "div(pc)";
        divHEScheme = "div(pc)";
    }

    // ******** U Residuals **********
    // copied and modified from UEqn.H

    if (hasFvSource_)
    {
        DAFvSource& daFvSource(const_cast<DAFvSource&>(
            mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource")));
        daFvSource.calcFvSource(fvSource_);
    }

    tmp<fvVectorMatrix> tUEqn(
        fvm::div(phi_, U_, divUScheme)
        + MRF_.DDt(rho_, U_)
        + daTurb_.divDevRhoReff(U_)
        - fvSource_
        - fvOptions_(rho_, U_));
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions_.constrain(UEqn);

    URes_ = (UEqn & U_) + fvc::grad(p_);
    normalizeResiduals(URes);

    // ******** e Residuals **********
    // copied and modified from EEqn.H
    volScalarField alphaEff("alphaEff", thermo_.alphaEff(alphat_));

    fvSourceEnergy_ = fvSource_ & U_;

    fvScalarMatrix EEqn(
        fvm::div(phi_, he_, divHEScheme)
        + (he_.name() == "e"
               ? fvc::div(phi_, volScalarField("Ekp", 0.5 * magSqr(U_) + p_ / rho_))
               : fvc::div(phi_, volScalarField("K", 0.5 * magSqr(U_))))
        - fvm::laplacian(alphaEff, he_)
        - fvSourceEnergy_
        - fvOptions_(rho_, he_));

    EEqn.relax();

    fvOptions_.constrain(EEqn);

    TRes_ = EEqn & he_;
    normalizeResiduals(TRes);

    // ******** p and phi Residuals **********
    // copied and modified from pEqn.H
    volScalarField rAU(1.0 / UEqn.A());
    surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho_ * rAU));
    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, it may degrade the accuracy of derivatives
    // because constraining variables will create discontinuity. Here we have a option to use the old
    // implementation in OpenFOAM-3.0+ and before (no constraint for HbyA)
    autoPtr<volVectorField> HbyAPtr = nullptr;
    label useConstrainHbyA = daOption_.getOption<label>("useConstrainHbyA");
    if (useConstrainHbyA)
    {
        HbyAPtr.reset(new volVectorField(constrainHbyA(rAU * UEqn.H(), U_, p_)));
    }
    else
    {
        HbyAPtr.reset(new volVectorField("HbyA", U_));
        HbyAPtr() = rAU * UEqn.H();
    }
    volVectorField& HbyA = HbyAPtr();

    tUEqn.clear();

    surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho_) * fvc::flux(HbyA));

    MRF_.makeRelative(fvc::interpolate(rho_), phiHbyA);

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_, rho_, U_, phiHbyA, rhorAUf, MRF_);

    // NOTE: we don't support transonic = true

    adjustPhi(phiHbyA, U_, p_);

    fvScalarMatrix pEqn(
        fvc::div(phiHbyA)
        - fvm::laplacian(rhorAUf, p_)
        - fvOptions_(psi_, p_, rho_.name()));

    pEqn.setReference(pressureControl_.refCell(), pressureControl_.refValue());

    pRes_ = pEqn & p_;
    normalizeResiduals(pRes);

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    phiRes_ = phiHbyA + pEqn.flux() - phi_;
    normalizePhiResiduals(phiRes);
}

void DAResidualRhoSimpleFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
        we need to:
        1, update psi, rho, he, and optionally mu and alpha by calling updateThermoVars
        2, update velocity boundary based on MRF
        3, update fvOptions
    */

    this->updateThermoVars();

    MRF_.correctBoundaryVelocity(U_);
    fvOptions_.correct(U_);
}

void DAResidualRhoSimpleFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    U_.correctBoundaryConditions();
    p_.correctBoundaryConditions();
    T_.correctBoundaryConditions();
}

} // End namespace Foam

// ************************************************************************* //
