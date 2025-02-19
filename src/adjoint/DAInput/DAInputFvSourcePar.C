/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAInputFvSourcePar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAInputFvSourcePar, 0);
addToRunTimeSelectionTable(DAInput, DAInputFvSourcePar, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAInputFvSourcePar::DAInputFvSourcePar(
    const word inputName,
    const word inputType,
    fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAInput(
        inputName,
        inputType,
        mesh,
        daOption,
        daModel,
        daIndex)
{
    fvSourceName_ = daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).getWord("fvSourceName");
    fvSourceType_ = daOption_.getAllOptions().subDict("fvSource").subDict(fvSourceName_).getWord("type");

    // if users set the indices, read it from inputInto, otherwise, use all indices
    if (daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).found("indices"))
    {
        daOption_.getAllOptions().subDict("inputInfo").subDict(inputName_).readEntry<labelList>("indices", indices_);
    }
    else
    {
        label size = 0;
        if (fvSourceType_ == "actuatorDisk")
        {
            size = 13;
        }
        else if (fvSourceType_ == "actuatorLine")
        {
            size = 18;
        }
        else if (fvSourceType_ == "actuatorPoint")
        {
            size = 13;
        }
        else if (fvSourceType_ == "heatSource")
        {
            size = 9;
        }
        else
        {
            FatalErrorIn("DAInputFvSourcePar") << "fvSourceType not supported "
                                               << abort(FatalError);
        }

        indices_.setSize(size);
        forAll(indices_, idxI)
        {
            indices_[idxI] = idxI;
        }
    }
}

void DAInputFvSourcePar::run(const scalarList& input)
{
    /*
    Description:
        Assign the input to OF fields
    */

#ifndef CODI_ADR
    Info << "DAInputFvSourcePar. " << endl;
    Info << "Setting fvSource indices " << indices_ << " with " << input << endl;
#endif

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));

    DAFvSource& daFvSource =
        const_cast<DAFvSource&>(mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource"));

    if (fvSourceType_ == "actuatorDisk")
    {

        forAll(indices_, idxI)
        {
            label selectedIndex = indices_[idxI];
            globalVar.actuatorDiskPars[fvSourceName_][selectedIndex] = input[idxI];
        }
    }
    else if (fvSourceType_ == "actuatorLine")
    {
        forAll(indices_, idxI)
        {
            label selectedIndex = indices_[idxI];
            globalVar.actuatorLinePars[fvSourceName_][selectedIndex] = input[idxI];
        }
    }
    else if (fvSourceType_ == "actuatorPoint")
    {
        forAll(indices_, idxI)
        {
            label selectedIndex = indices_[idxI];
            globalVar.actuatorPointPars[fvSourceName_][selectedIndex] = input[idxI];
        }
    }
    else if (fvSourceType_ == "heatSource")
    {
        forAll(indices_, idxI)
        {
            label selectedIndex = indices_[idxI];
            globalVar.heatSourcePars[fvSourceName_][selectedIndex] = input[idxI];
        }
    }
    else
    {
        FatalErrorIn("DAInputFvSourcePar") << "fvSourceType not supported "
                                           << abort(FatalError);
    }

    daFvSource.updateFvSource();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
