/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionVonMisesStressKS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionVonMisesStressKS, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionVonMisesStressKS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionVonMisesStressKS::DAFunctionVonMisesStressKS(
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
        functionName)
{

    functionDict_.readEntry<scalar>("coeffKS", coeffKS_);
}

/// calculate the value of objective function
scalar DAFunctionVonMisesStressKS::calcFunction()
{
    /*
    Description:
        Calculate the maximal von Mises stress aggregated using the KS function
        von Mises stress = KS( mu*twoSymm(fvc::grad(D)) + lambda*(I*tr(fvc::grad(D))) )
        where the KS function KS(x) = 1/coeffKS * ln( sum[exp(coeffKS*x_i)] ) 
    */

    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volTensorField& gradD = db.lookupObject<volTensorField>("gradD");
    const volScalarField& lambda = db.lookupObject<volScalarField>("solid:lambda");
    const volScalarField& mu = db.lookupObject<volScalarField>("solid:mu");
    const volScalarField& rho = db.lookupObject<volScalarField>("solid:rho");

    // volTensorField gradD(fvc::grad(D));
    volSymmTensorField sigma = rho * (mu * twoSymm(gradD) + (lambda * I) * tr(gradD));

    // NOTE: vonMises stress is scaled by scale_ provided in the function dict
    volScalarField vonMises = scale_ * sqrt((3.0 / 2.0) * magSqr(dev(sigma)));

    scalar objValTmp = 0.0;
    forAll(cellSources_, idxI)
    {
        const label& cellI = cellSources_[idxI];

        objValTmp += exp(coeffKS_ * vonMises[cellI]);

        if (objValTmp > 1e200)
        {
            FatalErrorIn(" ") << "KS function summation term too large! "
                              << "Reduce coeffKS! " << abort(FatalError);
        }
    }

    // need to reduce the sum of force across all processors
    reduce(objValTmp, sumOp<scalar>());

    functionValue = log(objValTmp) / coeffKS_;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
