/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

bool DAFvSource::writeData(Ostream& os) const
{
    /*
    Description:
        This is a virtual function for regIOobject
    */
    // do nothing
    return true;
}

void DAFvSource::syncDAOptionToActuatorDVs()
{
    /*
    Description:
        Synchronize the values in DAOption and actuatorDiskDVs_. 
        We need to synchronize the values defined in fvSource from DAOption to actuatorDiskDVs_
        NOTE: we need to call this function whenever we change the actuator design variables 
        during optimization. This is needed because we need to use actuatorDiskDVs_ in AD 
    */

    // now we need to initialize actuatorDiskDVs_
    dictionary fvSourceSubDict = daOption_.getAllOptions().subDict("fvSource");
    word diskName0 = fvSourceSubDict.toc()[0];

    word type0 = fvSourceSubDict.subDict(diskName0).getWord("type");

    if (type0 == "actuatorDisk")
    {
        word source0 = fvSourceSubDict.subDict(diskName0).getWord("source");

        if (source0 == "cylinderAnnulusSmooth")
        {
            forAll(fvSourceSubDict.toc(), idxI)
            {
                word diskName = fvSourceSubDict.toc()[idxI];

                // sub dictionary with all parameters for this disk
                dictionary diskSubDict = fvSourceSubDict.subDict(diskName);

                // now read in all parameters for this actuator disk
                scalarList centerList;
                diskSubDict.readEntry<scalarList>("center", centerList);

                // we have 9 design variables for each disk
                scalarList dvList(9);
                dvList[0] = centerList[0];
                dvList[1] = centerList[1];
                dvList[2] = centerList[2];
                dvList[3] = diskSubDict.getScalar("innerRadius");
                dvList[4] = diskSubDict.getScalar("outerRadius");
                dvList[5] = diskSubDict.getScalar("scale");
                dvList[6] = diskSubDict.getScalar("POD");
                dvList[7] = diskSubDict.getScalar("expM");
                dvList[8] = diskSubDict.getScalar("expN");

                // set actuatorDiskDVs_
                actuatorDiskDVs_.set(diskName, dvList);
            }
        }
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
