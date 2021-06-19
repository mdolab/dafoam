/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAResidualSolidDisplacementFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualSolidDisplacementFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualSolidDisplacementFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualSolidDisplacementFoam::DAResidualSolidDisplacementFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberVector(D, dimensionSet(0, 1, -2, 0, 0, 0, 0)),
      // these are intermediate variables or objects
      gradD_(const_cast<volTensorField&>(
          mesh.thisDb().lookupObject<volTensorField>("gradD"))),
      sigmaD_(const_cast<volSymmTensorField&>(
          mesh.thisDb().lookupObject<volSymmTensorField>("sigmaD"))),
      divSigmaExp_(const_cast<volVectorField&>(
          mesh.thisDb().lookupObject<volVectorField>("divSigmaExp"))),
      lambda_(const_cast<volScalarField&>(
          mesh.thisDb().lookupObject<volScalarField>("solid:lambda"))),
      mu_(const_cast<volScalarField&>(
          mesh.thisDb().lookupObject<volScalarField>("solid:mu")))
{

    IOdictionary thermalProperties(
        IOobject(
            "thermalProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    Switch thermalStress(thermalProperties.lookup("thermalStress"));
    if (thermalStress)
    {
        FatalErrorIn("") << "thermalStress=true not supported" << abort(FatalError);
    }

    const dictionary& stressControl = mesh.solutionDict().subDict("stressAnalysis");

    Switch compactNormalStress(stressControl.lookup("compactNormalStress"));

    if (!compactNormalStress)
    {
        FatalErrorIn("") << "compactNormalStress=false not supported" << abort(FatalError);
    }

    isTractionDisplacementBC_ = 0;
    forAll(D_.boundaryField(), patchI)
    {
        if (D_.boundaryField()[patchI].type() == "tractionDisplacement")
        {
            isTractionDisplacementBC_ = 1;
        }
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualSolidDisplacementFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    DRes_.clear();
}

void DAResidualSolidDisplacementFoam::calcResiduals(const dictionary& options)
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

    fvVectorMatrix DEqn(
        fvm::d2dt2(D_)
        == fvm::laplacian(2 * mu_ + lambda_, D_, "laplacian(DD,D)")
            + divSigmaExp_);

    DRes_ = DEqn & D_;
    normalizeResiduals(DRes);
}

void DAResidualSolidDisplacementFoam::updateDAndGradD()
{
    /*
    Description:
        Update D and gradD.

        NOTE: we need to update D boundary conditions iteratively if tractionDisplacement BC 
        is used this is because tractionDisplacement BC is dependent on gradD, while gradD
        is dependent on the D bc values.
    */

    // this will be called after doing perturbStates
    if (isTractionDisplacementBC_)
    {
        for (label i = 0; i < daOption_.getOption<label>("maxTractionBCIters"); i++)
        {
            D_.correctBoundaryConditions();
            gradD_ = fvc::grad(D_);
        }
    }
    else
    {
        D_.correctBoundaryConditions();
        gradD_ = fvc::grad(D_);
    }
}

void DAResidualSolidDisplacementFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
    */

    this->updateDAndGradD();

    sigmaD_ = mu_ * twoSymm(gradD_) + (lambda_ * I) * tr(gradD_);

    divSigmaExp_ = fvc::div(sigmaD_ - (2 * mu_ + lambda_) * gradD_, "div(sigmaD)");
}

void DAResidualSolidDisplacementFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    D_.correctBoundaryConditions();
}

} // End namespace Foam

// ************************************************************************* //
