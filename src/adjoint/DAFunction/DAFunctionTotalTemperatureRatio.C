/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionTotalTemperatureRatio.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionTotalTemperatureRatio, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionTotalTemperatureRatio, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionTotalTemperatureRatio::DAFunctionTotalTemperatureRatio(
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

    functionDict_.readEntry<wordList>("inletPatches", inletPatches_);
    functionDict_.readEntry<wordList>("outletPatches", outletPatches_);

    // get Cp from thermophysicalProperties
    const IOdictionary& thermoDict = mesh_.thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
    dictionary mixSubDict = thermoDict.subDict("mixture");
    dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
    Cp_ = thermodynamicsSubDict.getScalar("Cp");
    gamma_ = thermodynamicsSubDict.getScalar("gamma");

    if (daOption_.getOption<label>("debug"))
    {
        Info << "Cp " << Cp_ << endl;
        Info << "gamma " << gamma_ << endl;
    }
}

/// calculate the value of objective function
scalar DAFunctionTotalTemperatureRatio::calcFunction()
{
    /*
    Description:
        Calculate total temperature ratio,  TTOut/TTIn
    */

    // always calculate the area of all the inlet/outletPatches
    areaSumInlet_ = 0.0;
    areaSumOutlet_ = 0.0;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];
        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (inletPatches_.found(patchName))
        {
            areaSumInlet_ += mesh_.magSf().boundaryField()[patchI][faceI];
        }
        else if (outletPatches_.found(patchName))
        {
            areaSumOutlet_ += mesh_.magSf().boundaryField()[patchI][faceI];
        }
        else
        {
            FatalErrorIn(" ") << "inlet/outletPatches names are not in patches" << abort(FatalError);
        }
    }
    reduce(areaSumInlet_, sumOp<scalar>());
    reduce(areaSumOutlet_, sumOp<scalar>());

    // initialize objFunValue
    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    const scalar R = Cp_ - Cp_ / gamma_;

    const volScalarField::Boundary& TBf = T.boundaryField();
    const volVectorField::Boundary& UBf = U.boundaryField();

    // we first compute the averaged inlet and outlet total pressure they will
    // be used later for normalization
    scalar TTIn = 0.0;
    scalar TTOut = 0.0;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar TS = TBf[patchI][faceI];
        scalar UMag = mag(UBf[patchI][faceI]);
        scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
        scalar TT = TS * (1.0 + 0.5 * (gamma_ - 1.0) * Ma2);

        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (inletPatches_.found(patchName))
        {
            TTIn += TT * SfX / areaSumInlet_;
        }
        else if (outletPatches_.found(patchName))
        {
            TTOut += TT * SfX / areaSumOutlet_;
        }
        else
        {
            FatalErrorIn(" ") << "inlet/outletPatches names are not in patches" << abort(FatalError);
        }
    }
    reduce(TTIn, sumOp<scalar>());
    reduce(TTOut, sumOp<scalar>());

    functionValue = TTOut / TTIn;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
