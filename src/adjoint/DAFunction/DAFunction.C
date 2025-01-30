/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunction.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAFunction, 0);
defineRunTimeSelectionTable(DAFunction, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunction::DAFunction(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      functionName_(functionName)
{
    functionDict_ = daOption.getAllOptions().subDict("function").subDict(functionName);

    // Assign type and scale, this is common for all objectives
    functionDict_.readEntry<word>("type", functionType_);
    functionDict_.readEntry<scalar>("scale", scale_);
    timeOp_ = functionDict_.lookupOrDefault<word>("timeOp", "final");

    // calcualte the face and cell indices that are associated with this objective
    this->calcFunctionSources();

    // initialize calcRefVar related stuff
    calcRefVar_ = functionDict_.lookupOrDefault<label>("calcRefVar", 0);
    if (calcRefVar_)
    {
        functionDict_.readEntry<scalarList>("ref", ref_);
    }
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAFunction> DAFunction::New(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
{
    // standard setup for runtime selectable classes

    dictionary functionDict = daOption.getAllOptions().subDict("function").subDict(functionName);

    // look up the solver name
    word modelType;
    functionDict.readEntry<word>("type", modelType);

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting type: " << modelType << " for DAFunction. Name: " << functionName << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAFunction::New"
            "("
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&,"
            "    const word,"
            ")")
            << "Unknown DAFunction type "
            << modelType << nl << nl
            << "Valid DAFunction types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAFunction>(
        cstrIter()(mesh,
                   daOption,
                   daModel,
                   daIndex,
                   functionName));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAFunction::calcFunctionSources()
{
    /*
    Description:
        Compute the face and cell sources for the objective function.

    Output:
        faceSources, cellSources: The face and cell indices that 
        are associated with this objective function

    Example:
        A typical function dictionary reads:
    
        {
            "type": "force",
            "source": "patchToFace",
            "patches": ["walls", "wallsbump"],
            "scale": 0.5,
        }

        This information is obtained from DAFunction::functionDict_

    */

    // all avaiable source type are in src/meshTools/sets/cellSources
    // Example of IO parameters os in applications/utilities/mesh/manipulation/topoSet

    word functionSource;
    functionDict_.readEntry("source", functionSource);
    if (functionSource == "patchToFace")
    {
        // create a topoSet
        autoPtr<topoSet> currentSet(
            topoSet::New(
                "faceSet",
                mesh_,
                "set0",
                IOobject::NO_READ));
        // create the source
        autoPtr<topoSetSource> sourceSet(
            topoSetSource::New(functionSource, mesh_, functionDict_));

        // add the sourceSet to topoSet
        sourceSet().applyToSet(topoSetSource::NEW, currentSet());
        // get the face index from currentSet, we need to use
        // this special for loop
        for (const label i : currentSet())
        {
            faceSources_.append(i);
        }
    }
    else if (functionSource == "boxToCell")
    {
        // create a topoSet
        autoPtr<topoSet> currentSet(
            topoSet::New(
                "cellSet",
                mesh_,
                "set0",
                IOobject::NO_READ));
        // we need to change the min and max because they need to
        // be of type point; however, we can't parse point type
        // in pyDict, we need to change them here.
        dictionary functionTmp = functionDict_;
        scalarList boxMin;
        scalarList boxMax;
        functionDict_.readEntry("min", boxMin);
        functionDict_.readEntry("max", boxMax);

        point boxMin1;
        point boxMax1;
        boxMin1[0] = boxMin[0];
        boxMin1[1] = boxMin[1];
        boxMin1[2] = boxMin[2];
        boxMax1[0] = boxMax[0];
        boxMax1[1] = boxMax[1];
        boxMax1[2] = boxMax[2];

        functionTmp.set("min", boxMin1);
        functionTmp.set("max", boxMax1);

        // create the source
        autoPtr<topoSetSource> sourceSet(
            topoSetSource::New(functionSource, mesh_, functionTmp));

        // add the sourceSet to topoSet
        sourceSet().applyToSet(topoSetSource::NEW, currentSet());
        // get the face index from currentSet, we need to use
        // this special for loop
        for (const label i : currentSet())
        {
            cellSources_.append(i);
        }
    }
    else if (functionSource == "allCells")
    {
        forAll(mesh_.cells(), cellI)
        {
            cellSources_.append(cellI);
        }
    }
    else
    {
        FatalErrorIn("calcFunctionSources") << "source: " << functionSource << " not supported!"
                                            << "Options are: allCells, patchToFace, or boxToCell!"
                                            << abort(FatalError);
    }
}

void DAFunction::calcRefVar(scalar& functionValue)
{
    /*
    Description:
        Call the variable difference with respect to a given reference and take a square of it.
        This can be used in FIML. This function is for calcRefVar == 1
    */

    if (calcRefVar_)
    {
        if (ref_.size() == 1)
        {
            functionValue = (functionValue - ref_[0]) * (functionValue - ref_[0]);
        }
        else
        {
            label idxI = mesh_.time().timeIndex() - 1;
            functionValue = (functionValue - ref_[idxI]) * (functionValue - ref_[idxI]);
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
