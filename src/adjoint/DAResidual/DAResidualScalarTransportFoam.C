/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAResidualScalarTransportFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualScalarTransportFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualScalarTransportFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualScalarTransportFoam::DAResidualScalarTransportFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberScalar(T, dimensionSet(0, 0, -1, 1, 0, 0, 0)),
      phi_(const_cast<surfaceScalarField&>(
          mesh_.thisDb().lookupObject<surfaceScalarField>("phi")))

{
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    dimensionedScalar DT("DT", dimViscosity, transportProperties);

    DT_ = DT.value();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualScalarTransportFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    TRes_.clear();
}

void DAResidualScalarTransportFoam::calcResiduals(const dictionary& options)
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

    dimensionedScalar DT("DT", dimViscosity, DT_);

    fvScalarMatrix TEqn(
        fvm::ddt(T_)
        + fvm::div(phi_, T_)
        - fvm::laplacian(DT, T_));

    TEqn.relax();

    TRes_ = TEqn & T_;
    normalizeResiduals(TRes);
}

void DAResidualScalarTransportFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
    */
    // do nothing
}

void DAResidualScalarTransportFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    T_.correctBoundaryConditions();
}

} // End namespace Foam

// ************************************************************************* //
