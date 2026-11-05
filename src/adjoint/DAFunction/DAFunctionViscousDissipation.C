#include "DAFunctionViscousDissipation.H"

namespace Foam
{

defineTypeNameAndDebug(DAFunctionViscousDissipation, 0);
addToRunTimeSelectionTable(
    DAFunction,
    DAFunctionViscousDissipation,
    dictionary);

DAFunctionViscousDissipation::DAFunctionViscousDissipation(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName),
      daTurb_(daModel.getDATurbulenceModel())
{
}

scalar DAFunctionViscousDissipation::calcFunction()
{
    /*
    Description:
        Calculate the total power dissipation (Reference: https://doi.org/10.1002/nme.1468)

            J = \int [ 2*muEff*D:D + alphaPorosity*(U·U) ] dV

        where

            D = symm(grad(U))

        The first term represents viscous dissipation while the second
        represents Brinkman (Darcy) dissipation.

    Output:
        functionValue
    */

    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();

    //-------------------------------------------------------------
    // Flow variables
    //-------------------------------------------------------------

    const volVectorField& U =
        db.lookupObject<volVectorField>("U");

    const volScalarField& alpha =
        db.lookupObject<volScalarField>("alpha");

    //-------------------------------------------------------------
    // Velocity gradient
    //-------------------------------------------------------------

    tmp<volTensorField> tGradU = fvc::grad(U);

    //-------------------------------------------------------------
    // Symmetric strain-rate tensor
    //-------------------------------------------------------------

    volSymmTensorField D(
        symm(tGradU()));

    //-------------------------------------------------------------
    // Effective viscosity
    //-------------------------------------------------------------

    tmp<volScalarField> tNuEff = daTurb_.nuEff();

    //-------------------------------------------------------------
    // Integrate objective
    //-------------------------------------------------------------

    forAll(cellSources_, idxI)
    {
        const label cellI = cellSources_[idxI];

        scalar viscousTerm =
            2.0
            * tNuEff()[cellI]
            * (D[cellI] && D[cellI]);

        scalar brinkmanTerm =
            alpha[cellI]
            * magSqr(U[cellI]);

        functionValue +=
            scale_
            * (viscousTerm + brinkmanTerm)
            * mesh_.V()[cellI];
    }

    //-------------------------------------------------------------
    // Parallel reduction
    //-------------------------------------------------------------

    reduce(functionValue, sumOp<scalar>());

    this->calcRefVar(functionValue);

    return functionValue;
}

}
