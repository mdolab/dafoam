/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFvSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAFvSource, 0);
defineRunTimeSelectionTable(DAFvSource, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFvSource::DAFvSource(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : regIOobject(
        IOobject(
            "DAFvSource", // always use DAFvSource for the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      modelType_(modelType),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAFvSource> DAFvSource::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAFvSource" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAFvSource::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&"
            ")")
            << "Unknown DAFvSource type "
            << modelType << nl << nl
            << "Valid DAFvSource types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAFvSource>(
        cstrIter()(modelType, mesh, daOption, daModel, daIndex));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAFvSource::calcFvSource(volVectorField& fvSource)
{
    /*
    Description:
        Calculate the fvSource term
        NOTE: this need to be implemented in the child class, if not,
        print an error!
    */
    FatalErrorIn("") << "calcFvSource not implemented " << endl
                     << " in the child class for " << modelType_
                     << abort(FatalError);
}

void DAFvSource::calcFvSource(volScalarField& fvSource)
{
    /*
    Description:
        Calculate the fvSource term
        NOTE: this need to be implemented in the child class, if not,
        print an error!
    */
    FatalErrorIn("") << "calcFvSource not implemented " << endl
                     << " in the child class for " << modelType_
                     << abort(FatalError);
}

bool DAFvSource::writeData(Ostream& os) const
{
    /*
    Description:
        This is a virtual function for regIOobject
    */
    // do nothing
    return true;
}

void DAFvSource::initFvSourcePars()
{
    /*
    Description:
        Initialize the values for all types of fvSource in DAGlobalVar, including 
        actuatorDiskPars, heatSourcePars, etc
        NOTE: this need to be implemented in the child class, if not,
        print an error!
    */
    FatalErrorIn("") << "initFvSourcePars not implemented " << endl
                     << " in the child class for " << modelType_
                     << abort(FatalError);
}

void DAFvSource::findGlobalSnappedCenter(
    label snappedCenterCellI,
    vector& center)
{
    scalar centerX = 0.0;
    scalar centerY = 0.0;
    scalar centerZ = 0.0;

    if (snappedCenterCellI >= 0)
    {
        centerX = mesh_.C()[snappedCenterCellI][0];
        centerY = mesh_.C()[snappedCenterCellI][1];
        centerZ = mesh_.C()[snappedCenterCellI][2];
    }
    reduce(centerX, sumOp<scalar>());
    reduce(centerY, sumOp<scalar>());
    reduce(centerZ, sumOp<scalar>());

    center[0] = centerX;
    center[1] = centerY;
    center[2] = centerZ;
}

void DAFvSource::updateFvSource()
{
    // calculate fvSource based on the latest parameters defined in DAGlobalVar
    if (mesh_.thisDb().foundObject<volVectorField>("fvSource"))
    {
        volVectorField& fvSource = const_cast<volVectorField&>(
            mesh_.thisDb().lookupObject<volVectorField>("fvSource"));

        this->calcFvSource(fvSource);
    }
    else if (mesh_.thisDb().foundObject<volScalarField>("fvSource"))
    {
        volScalarField& fvSource = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("fvSource"));

        this->calcFvSource(fvSource);
    }
    else
    {
        FatalErrorIn("") << "fvSource not found! "
                         << abort(FatalError);
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
