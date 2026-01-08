/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAFunctionResidualNorm.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionResidualNorm, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionResidualNorm, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionResidualNorm::DAFunctionResidualNorm(
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
    // initialize stateInfo_
    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();

    dictionary resWeightDict = daOption.getAllOptions().subDict("function").subDict(functionName).subDict("resWeight");

    forAll(resWeightDict.toc(), idxI)
    {
        word resName = resWeightDict.toc()[idxI];
        scalar weight = resWeightDict.getScalar(resName);
        resWeight_.set(resName, weight);
    }

    resMode_ = daOption.getAllOptions().subDict("function").subDict(functionName).lookupOrDefault<word>("resMode", "L2Norm");
    Info << "residual weights for DAFunctionResidualNorm " << resWeight_;
    Info << "residual mode " << resMode_;
}

/// calculate the value of objective function
scalar DAFunctionResidualNorm::calcFunction()
{
    /*
    Description:
        Calculate the L2 norm of all residuals
    */

    // initialize objFunValue
    scalar functionValue = 0.0;

    const objectRegistry& db = mesh_.thisDb();
    DAResidual& daResidual = const_cast<DAResidual&>(db.lookupObject<DAResidual>("DAResidual"));
    DAModel& daModel = const_cast<DAModel&>(daModel_);

    dictionary options;
    options.set("isPC", 0);
    daResidual.calcResiduals(options);
    daModel.calcResiduals(options);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word resName = stateName + "Res";
        const volVectorField& stateRes = mesh_.thisDb().lookupObject<volVectorField>(resName);
        scalar weight2 = resWeight_[resName] * resWeight_[resName];

        forAll(stateRes, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                if (resMode_ == "L2Norm")
                {
                    functionValue += weight2 * stateRes[cellI][i] * stateRes[cellI][i];
                }
                else if (resMode_ == "L1Norm")
                {
                    functionValue += sqrt(weight2 * stateRes[cellI][i] * stateRes[cellI][i]);
                }
                else
                {
                    FatalErrorIn(" ") << "resMode not supported! options are L2Norm or L2Norm "
                                      << abort(FatalError);
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word resName = stateName + "Res";
        const volScalarField& stateRes = mesh_.thisDb().lookupObject<volScalarField>(resName);
        scalar weight2 = resWeight_[resName] * resWeight_[resName];

        forAll(stateRes, cellI)
        {
            if (resMode_ == "L2Norm")
            {
                functionValue += weight2 * stateRes[cellI] * stateRes[cellI];
            }
            else if (resMode_ == "L1Norm")
            {
                functionValue += sqrt(weight2 * stateRes[cellI] * stateRes[cellI]);
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word resName = stateName + "Res";
        const volScalarField& stateRes = mesh_.thisDb().lookupObject<volScalarField>(resName);
        scalar weight2 = resWeight_[resName] * resWeight_[resName];

        forAll(stateRes, cellI)
        {
            if (resMode_ == "L2Norm")
            {
                functionValue += weight2 * stateRes[cellI] * stateRes[cellI];
            }
            else if (resMode_ == "L1Norm")
            {
                functionValue += sqrt(weight2 * stateRes[cellI] * stateRes[cellI]);
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word resName = stateName + "Res";
        const surfaceScalarField& stateRes = mesh_.thisDb().lookupObject<surfaceScalarField>(resName);
        scalar weight2 = resWeight_[resName] * resWeight_[resName];

        forAll(stateRes, faceI)
        {
            functionValue += weight2 * stateRes[faceI] * stateRes[faceI];
        }
        forAll(stateRes.boundaryField(), patchI)
        {
            forAll(stateRes.boundaryField()[patchI], faceI)
            {
                scalar bPhiRes = stateRes.boundaryField()[patchI][faceI];
                if (resMode_ == "L2Norm")
                {
                    functionValue += weight2 * bPhiRes * bPhiRes;
                }
                else if (resMode_ == "L1Norm")
                {
                    functionValue += sqrt(weight2 * bPhiRes * bPhiRes);
                }
            }
        }
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    if (resMode_ == "L2Norm")
    {
        functionValue = sqrt(functionValue);
    }

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
