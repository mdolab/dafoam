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
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
    : outputName_(outputName),
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
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
{
    // standard setup for runtime selectable classes

    // A word expression for any outputName that has "_", if yes, it is a composite outputName
    // for example, outputName = "function_CD", then we need to use "function" as the typeName
    word typeName;
    wordRe compositeOutputNameExp = {".*_.*", wordRe::REGEX};
    if (compositeOutputNameExp.match(outputName))
    {
        // there is a "_" in the outputName, we need to parse the info and use the 
        // words before the first "_" as the typeName
        const auto splitNames = stringOps::split(outputName, "_");
        typeName = splitNames[0].str();
    }
    else
    {
        // there is no "_", so outputName is typeName
        typeName = outputName;
    }

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting output: " << typeName << " for DAOutput." << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(typeName);

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
            << typeName << nl << nl
            << "Valid DAOutput types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAOutput>(
        cstrIter()(typeName,
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
