/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAOutput.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAOutput, 0);
defineRunTimeSelectionTable(DAOutput, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAOutput::DAOutput(
    const word outputName,
    const word outputType,
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
    : outputName_(outputName),
      outputType_(outputType),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daResidual_(daResidual),
      daFunctionList_(daFunctionList)
{
    // initialize stateInfo_
    word solverName = daOption_.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAOutput> DAOutput::New(
    const word outputName,
    const word outputType,
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting output: " << outputType << " for DAOutput." << endl;
    }

    auto* ctorPtr = dictionaryConstructorTable(outputType);

    if (!ctorPtr)
    {
        FatalErrorInLookup(
            "DAOutput",
            outputType,
            *dictionaryConstructorTablePtr_)
            << exit(FatalError);
    }

    return autoPtr<DAOutput>(
        ctorPtr(outputName,
                outputType,
                mesh,
                daOption,
                daModel,
                daIndex,
                daResidual,
                daFunctionList));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
