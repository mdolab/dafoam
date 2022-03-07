/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncTotalTemperatureRatio.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncTotalTemperatureRatio, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncTotalTemperatureRatio, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncTotalTemperatureRatio::DAObjFuncTotalTemperatureRatio(
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

    // setup the connectivity for total pressure, this is needed in Foam::DAJacCondFdW
    // for pressureInlet velocity, U depends on phi
    objFuncConInfo_ = {{"U", "T", "phi"}};

    objFuncDict_.readEntry<scalar>("scale", scale_);

    objFuncDict_.readEntry<wordList>("inletPatches", inletPatches_);
    objFuncDict_.readEntry<wordList>("outletPatches", outletPatches_);

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
void DAObjFuncTotalTemperatureRatio::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate total temperature ratio,  TTIn/TTOut
        NOTE: to enable coloring, we need to separate TTIn and TTOut for each face, we do:
        d(TTInAvg/TTOutAvg)/dw = -TTOutAvg/TTInAvg^2 * dTTInAvg/dw + 1/TTInAvg * dTTOutAvg/dw
        so here objDiscreteVal = -TTOutAvg/TTInAvg^2 * dTTIn_i*ds_i/dSIn for inlet patches
        and objDiscreteVal = 1/TTInAvg * TTOut_i*ds_i/dSOut for outlet patches

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

    // calculate the area of all the inlet/outletPatches
    if (areaSumInlet_ < 0.0 || areaSumOutlet_ < 0.0)
    {
        areaSumInlet_ = 0.0;
        areaSumOutlet_ = 0.0;
        forAll(objFuncFaceSources, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            word patchName = mesh_.boundaryMesh()[patchI].name();
            if (DAUtility::isInList<word>(patchName, inletPatches_))
            {
                areaSumInlet_ += mesh_.magSf().boundaryField()[patchI][faceI];
            }
            else if (DAUtility::isInList<word>(patchName, outletPatches_))
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
    }

    // initialize faceValues to zero
    forAll(objFuncFaceValues, idxI)
    {
        objFuncFaceValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

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
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar TS = TBf[patchI][faceI];
        scalar UMag = mag(UBf[patchI][faceI]);
        scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
        scalar TT = TS * (1.0 + 0.5 * (gamma_ - 1.0) * Ma2);

        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (DAUtility::isInList<word>(patchName, inletPatches_))
        {
            TTIn += TT * SfX / areaSumInlet_;
        }
        else if (DAUtility::isInList<word>(patchName, outletPatches_))
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

    // set Reference values
    if (calcRefCoeffs)
    {
        TTInRef_ = TTIn;
        TTOutRef_ = TTOut;
    }

    objFuncValue = TTOut / TTIn;

    // pTotal ratio: Note we need special treatment to enable coloring
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar TS = TBf[patchI][faceI];
        scalar UMag = mag(UBf[patchI][faceI]);
        scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
        scalar TT = TS * (1.0 + 0.5 * (gamma_ - 1.0) * Ma2);

        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (DAUtility::isInList<word>(patchName, inletPatches_))
        {
            objFuncFaceValues[idxI] = -TT * SfX / areaSumInlet_ * TTOutRef_ / TTInRef_ / TTInRef_;
        }
        else if (DAUtility::isInList<word>(patchName, outletPatches_))
        {
            objFuncFaceValues[idxI] = TT * SfX / areaSumOutlet_ / TTInRef_;
        }
        else
        {
            FatalErrorIn(" ") << "inlet/outletPatches names are not in patches" << abort(FatalError);
        }
    }

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
