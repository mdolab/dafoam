/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(outputType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAOutput::New"
            "("
            "    const word,"
            "    fvMesh&,"
            "    const DAOption&,"
            "    DAModel&,"
            "    const DAIndex&,"
            "    DAResidual&,"
            "    UPtrList<DAFunction>&"
            ")")
            << "Unknown DAOutput type "
            << outputType << nl << nl
            << "Valid DAOutput types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAOutput>(
        cstrIter()(outputName,
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
