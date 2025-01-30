/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionTotalPressureRatio.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionTotalPressureRatio, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionTotalPressureRatio, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionTotalPressureRatio::DAFunctionTotalPressureRatio(
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
scalar DAFunctionTotalPressureRatio::calcFunction()
{
    /*
    Description:
        Calculate total pressure ratio,  TP_outlet/TP_inlet
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
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    const scalar R = Cp_ - Cp_ / gamma_;
    const scalar expCoeff = gamma_ / (gamma_ - 1.0);

    const volScalarField::Boundary& pBf = p.boundaryField();
    const volScalarField::Boundary& TBf = T.boundaryField();
    const volVectorField::Boundary& UBf = U.boundaryField();

    // we first compute the averaged inlet and outlet total pressure they will
    // be used later for normalization
    scalar TPIn = 0.0;
    scalar TPOut = 0.0;
    forAll(faceSources_, idxI)
    {
        const label& functionFaceI = faceSources_[idxI];
        label bFaceI = functionFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar pS = pBf[patchI][faceI];
        scalar TS = TBf[patchI][faceI];
        scalar UMag = mag(UBf[patchI][faceI]);
        scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
        scalar pT = pS * pow(1.0 + 0.5 * (gamma_ - 1.0) * Ma2, expCoeff);

        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (inletPatches_.found(patchName))
        {
            TPIn += pT * SfX / areaSumInlet_;
        }
        else if (outletPatches_.found(patchName))
        {
            TPOut += pT * SfX / areaSumOutlet_;
        }
        else
        {
            FatalErrorIn(" ") << "inlet/outletPatches names are not in patches" << abort(FatalError);
        }
    }
    reduce(TPIn, sumOp<scalar>());
    reduce(TPOut, sumOp<scalar>());

    functionValue = TPOut / TPIn;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
