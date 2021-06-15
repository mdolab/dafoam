/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAResidualPimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualPimpleFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualPimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualPimpleFoam::DAResidualPimpleFoam(
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
      fvSource_(const_cast<volVectorField&>(
          mesh_.thisDb().lookupObject<volVectorField>("fvSource"))),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      // create simpleControl
      pimple_(const_cast<fvMesh&>(mesh))
{
    // initialize fvSource
    const dictionary& allOptions = daOption.getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
    }

    mode_ = daOption.getSubDictOption<word>("unsteadyAdjoint", "mode");
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualPimpleFoam::clear()
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

void DAResidualPimpleFoam::calcResiduals(const dictionary& options)
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

    // We dont support MRF and fvOptions so all the related lines are commented
    // out for now

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
        // update the actuator source term
        daFvSource.calcFvSource(fvSource_);
    }

    fvVectorMatrix UEqn(
        fvm::ddt(U_)
        + fvm::div(phi_, U_, divUScheme)
        + daTurb_.divDevReff(U_)
        - fvSource_);

    if (mode_ == "hybridAdjoint")
    {
        UEqn -= fvm::ddt(U_);
    }

    UEqn.relax();

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

    surfaceScalarField phiHbyA(
        "phiHbyA",
        fvc::flux(HbyA)
            + fvc::interpolate(rAU) * fvc::ddtCorr(U_, phi_));

    if (mode_ == "hybridAdjoint")
    {
        phiHbyA -= fvc::interpolate(rAU) * fvc::ddtCorr(U_, phi_);
    }

    adjustPhi(phiHbyA, U_, p_);

    tmp<volScalarField> rAtU(rAU);

    if (pimple_.consistent())
    {
        rAtU = 1.0 / max(1.0 / rAU - UEqn.H1(), 0.1 / rAU);
        phiHbyA +=
            fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p_) * mesh_.magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(p_);
    }

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

void DAResidualPimpleFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
    */

    // nothing to update for DAPimpleFoam
}

void DAResidualPimpleFoam::correctBoundaryConditions()
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
