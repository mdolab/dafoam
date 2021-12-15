/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAObjFuncTotalPressureRatio.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncTotalPressureRatio, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncTotalPressureRatio, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncTotalPressureRatio::DAObjFuncTotalPressureRatio(
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
    word pName = "p";
    if (mesh_.thisDb().foundObject<volScalarField>("p_rgh"))
    {
        pName = "p_rgh";
    }
    // for pressureInlet velocity, U depends on phi
    objFuncConInfo_ = {{"U", "T", pName, "phi"}};

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
void DAObjFuncTotalPressureRatio::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate total pressure ratio,  TP_outlet/TP_inlet
        NOTE: to enable coloring, we need to separate TPIn and TPOut for each face, so we do:
        d(TPInAvg/TPOutAvg)/dw = -TPOutAvg/TPInAvg^2 * dTPInAvg/dw + 1/TPInAvg * dTPOutAvg/dw
        so here objDiscreteVal = -TPOutAvg/TPInAvg^2 * dTPIn_i*ds_i/dSIn for inlet patches
        and objDiscreteVal = 1/TPInAvg * TPOut_i*ds_i/dSOut for outlet patches

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
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar pS = pBf[patchI][faceI];
        scalar TS = TBf[patchI][faceI];
        scalar UMag = mag(UBf[patchI][faceI]);
        scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
        scalar pT = pS * pow(1.0 + 0.5 * (gamma_ - 1.0) * Ma2, expCoeff);

        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (DAUtility::isInList<word>(patchName, inletPatches_))
        {
            TPIn += pT * SfX / areaSumInlet_;
        }
        else if (DAUtility::isInList<word>(patchName, outletPatches_))
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

    // set Reference values
    if (calcRefCoeffs)
    {
        TPInRef_ = TPIn;
        TPOutRef_ = TPOut;
    }

    objFuncValue = TPOut / TPIn;

    // pTotal ratio: Note we need special treatment to enable coloring
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar pS = pBf[patchI][faceI];
        scalar TS = TBf[patchI][faceI];
        scalar UMag = mag(UBf[patchI][faceI]);
        scalar SfX = mesh_.magSf().boundaryField()[patchI][faceI];
        scalar Ma2 = sqr(UMag / sqrt(gamma_ * R * TS));
        scalar pT = pS * pow(1.0 + 0.5 * (gamma_ - 1.0) * Ma2, expCoeff);

        word patchName = mesh_.boundaryMesh()[patchI].name();
        if (DAUtility::isInList<word>(patchName, inletPatches_))
        {
            objFuncFaceValues[idxI] = -pT * SfX / areaSumInlet_ * TPOutRef_ / TPInRef_ / TPInRef_;
        }
        else if (DAUtility::isInList<word>(patchName, outletPatches_))
        {
            objFuncFaceValues[idxI] = pT * SfX / areaSumOutlet_ / TPInRef_;
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
