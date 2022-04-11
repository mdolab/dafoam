/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncVonMisesStressKS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncVonMisesStressKS, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncVonMisesStressKS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncVonMisesStressKS::DAObjFuncVonMisesStressKS(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
    : DAObjFunc(
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        objFuncName,
        objFuncPart,
        objFuncDict)
{
    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    // setup the connectivity for heat flux, this is needed in Foam::DAJacCondFdW
    objFuncConInfo_ = {
        {"D"}, // level 0
        {"D"}}; // level 1

    objFuncDict_.readEntry<scalar>("scale", scale_);

    objFuncDict_.readEntry<scalar>("coeffKS", coeffKS_);
}

/// calculate the value of objective function
void DAObjFuncVonMisesStressKS::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the maximal von Mises stress aggregated using the KS function
        von Mises stress = KS( mu*twoSymm(fvc::grad(D)) + lambda*(I*tr(fvc::grad(D))) )
        where the KS function KS(x) = 1/coeffKS * ln( sum[exp(coeffKS*x_i)] ) 

    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective

    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // initialize faceValues to zero
    forAll(objFuncCellValues, idxI)
    {
        objFuncCellValues[idxI] = 0.0;
    }

    objFuncValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    //const volVectorField& D = db.lookupObject<volVectorField>("D");
    const volScalarField& lambda = db.lookupObject<volScalarField>("solid:lambda");
    const volScalarField& mu = db.lookupObject<volScalarField>("solid:mu");
    const volScalarField& rho = db.lookupObject<volScalarField>("solid:rho");
    const volTensorField& gradD = db.lookupObject<volTensorField>("gradD");

    volSymmTensorField sigma = rho * (mu * twoSymm(gradD) + lambda * (I * tr(gradD)));

    // NOTE: vonMises stress is scaled by scale_ provided in the objFunc dict
    volScalarField vonMises = scale_* sqrt((3.0 / 2.0) * magSqr(dev(sigma)));

    scalar objValTmp = 0.0;
    forAll(objFuncCellSources, idxI)
    {
        const label& cellI = objFuncCellSources[idxI];

        objFuncCellValues[idxI] = exp(coeffKS_ * vonMises[cellI]);

        objValTmp += objFuncCellValues[idxI];

        if (objValTmp > 1e200)
        {
            FatalErrorIn(" ") << "KS function summation term too large! "
                              << "Reduce coeffKS! " << abort(FatalError);
        }
    }

    // need to reduce the sum of force across all processors
    reduce(objValTmp, sumOp<scalar>());

    // expSumKS stores sum[exp(coeffKS*x_i)], it will be used to scale dFdW
    expSumKS = objValTmp;

    objFuncValue = log(objValTmp) / coeffKS_;

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
