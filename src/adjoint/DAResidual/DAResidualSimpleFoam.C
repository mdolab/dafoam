/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAResidualSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualSimpleFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualSimpleFoam::DAResidualSimpleFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberVector(U, dimensionSet(0, 1, -2, 0, 0, 0, 0)),
      setResidualClassMemberScalar(p, dimensionSet(0, 0, -1, 0, 0, 0, 0)),
      setResidualClassMemberPhi(phi),
      alphaPorosity_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("alphaPorosity"))),
      fvSource_(const_cast<volVectorField&>(
          mesh_.thisDb().lookupObject<volVectorField>("fvSource"))),
      fvOptions_(fv::options::New(mesh)),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      // create simpleControl
      simple_(const_cast<fvMesh&>(mesh)),
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
        - fvOptions_(U_));
    fvOptions_.constrain(UEqn);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualSimpleFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    URes_.clear();
    pRes_.clear();
    phiRes_.clear();
}

void DAResidualSimpleFoam::calcResiduals(const dictionary& options)
{
    /*
    Description:
        This is the function to compute residuals.
    
    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,U)

        p_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        URes_, pRes_, phiRes_: residual field variables
    */

    // ******** U Residuals **********
    // copied and modified from UEqn.H

    word divUScheme = "div(phi,U)";

    label isPC = options.getLabel("isPC");

    if (isPC)
    {
        divUScheme = "div(pc)";
    }

    if (hasFvSource_)
    {
        DAFvSource& daFvSource(const_cast<DAFvSource&>(
            mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource")));
        daFvSource.calcFvSource(fvSource_);
    }

    tmp<fvVectorMatrix> tUEqn(
        fvm::div(phi_, U_, divUScheme)
        + fvm::Sp(alphaPorosity_, U_)
        + MRF_.DDt(U_)
        + daTurb_.divDevReff(U_)
        - fvSource_
        - fvOptions_(U_));
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvOptions_.constrain(UEqn);

    URes_ = (UEqn & U_) + fvc::grad(p_);
    normalizeResiduals(URes);

    // ******** p Residuals **********
    // copied and modified from pEqn.H
    // NOTE manually set pRefCell and pRefValue
    label pRefCell = 0;
    scalar pRefValue = 0.0;

    volScalarField rAU(1.0 / UEqn.A());
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U_, p_));
    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, We do NOT use the constrainHbyA
    // function in DAFoam because we found it significantly degrades the accuracy of shape derivatives.
    // Basically, we should not constrain any variable because it will create discontinuity.
    // Instead, we use the old implementation used in OpenFOAM-3.0+ and before
    volVectorField HbyA("HbyA", U_);
    HbyA = rAU * UEqn.H();

    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

    MRF_.makeRelative(phiHbyA);

    adjustPhi(phiHbyA, U_, p_);

    tmp<volScalarField> rAtU(rAU);

    if (simple_.consistent())
    {
        rAtU = 1.0 / (1.0 / rAU - UEqn.H1());
        phiHbyA += fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p_) * mesh_.magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(p_);
    }

    tUEqn.clear();

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p_, U_, phiHbyA, rAtU(), MRF_);

    fvScalarMatrix pEqn(
        fvm::laplacian(rAtU(), p_)
        == fvc::div(phiHbyA));
    pEqn.setReference(pRefCell, pRefValue);

    pRes_ = pEqn & p_;
    normalizeResiduals(pRes);

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    phiRes_ = phiHbyA - pEqn.flux() - phi_;
    // need to normalize phiRes
    normalizePhiResiduals(phiRes);
}

void DAResidualSimpleFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
    */

    MRF_.correctBoundaryVelocity(U_);
    fvOptions_.correct(U_);
}

void DAResidualSimpleFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    U_.correctBoundaryConditions();
    p_.correctBoundaryConditions();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
