/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAResidualHeatTransferFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualHeatTransferFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualHeatTransferFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualHeatTransferFoam::DAResidualHeatTransferFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberScalar(T, dimPower / dimLength / dimLength / dimLength),
      fvSource_(const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>("fvSource"))),
      kappa_(const_cast<volScalarField&>(mesh_.thisDb().lookupObject<volScalarField>("kappa")))

{
    IOdictionary solidProperties(
        IOobject(
            "solidProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    if (solidProperties.found("kappa"))
    {
        kappaCoeffs_ = List<scalar>(1, solidProperties.getScalar("kappa"));
    }
    else if (solidProperties.found("kappaCoeffs"))
    {
        kappaCoeffs_ = solidProperties.lookup("kappaCoeffs");
    }
    else
    {
        FatalErrorInFunction
            << "Neither 'kappa' nor 'kappaCoeffs' found in dictionary: "
            << solidProperties.name() << exit(FatalError);
    }

    const dictionary& allOptions = daOption.getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualHeatTransferFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    TRes_.clear();
}

void DAResidualHeatTransferFoam::calcResiduals(const dictionary& options)
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

    if (hasFvSource_)
    {
        DAFvSource& daFvSource(const_cast<DAFvSource&>(
            mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource")));
        daFvSource.calcFvSource(fvSource_);
    }

    fvScalarMatrix TEqn(
        fvm::laplacian(kappa_, T_)
        + fvSource_);

    TRes_ = TEqn & T_;
    normalizeResiduals(TRes);
}

void DAResidualHeatTransferFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
    */
    // update k
    forAll(kappa_, cellI)
    {
        kappa_[cellI] = 0.0;
        forAll(kappaCoeffs_, order)
        {
            kappa_[cellI] += kappaCoeffs_[order] * pow(T_[cellI], order);
        }
    }
    // update boundary
    forAll(kappa_.boundaryField(), patchI)
    {
        forAll(kappa_.boundaryField()[patchI], faceI)
        {
            kappa_.boundaryFieldRef()[patchI][faceI] = 0;
            forAll(kappaCoeffs_, order)
            {
                kappa_.boundaryFieldRef()[patchI][faceI] += kappaCoeffs_[order] * pow(T_.boundaryField()[patchI][faceI], order);
            }
        }
    }
}

void DAResidualHeatTransferFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    T_.correctBoundaryConditions();
}

} // End namespace Foam

// ************************************************************************* //
