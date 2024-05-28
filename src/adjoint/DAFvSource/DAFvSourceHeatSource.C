/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAFvSourceHeatSource.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFvSourceHeatSource, 0);
addToRunTimeSelectionTable(DAFvSource, DAFvSourceHeatSource, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFvSourceHeatSource::DAFvSourceHeatSource(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAFvSource(modelType, mesh, daOption, daModel, daIndex)
{
    this->calcFvSourceCellIndices(fvSourceCellIndices_);
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAFvSourceHeatSource::calcFvSource(volScalarField& fvSource)
{
    /*
    Description:
        Calculate the heat source and assign them to fvSource

    Example:
        An example of the fvSource in pyOptions in pyDAFoam can be
        defOptions = 
        {
            "fvSource"
            {
                "source1"
                {
                    "type": "heatSource",
                    "source": "cylinderAnnulusToCell",
                    "p1": [0.5, 0.3, 0.1], # p1 and p2 define the axis and width
                    "p2": [0.5, 0.3, 0.5], # p2-p1 should be the axis of the cylinder
                    "innerRadius": 0.1,
                    "outerRadius": 0.8,
                    "power": 101.0,  # here we should prescribe the power in W
                },
                "source2"
                {
                    "type": "heatSource",
                    "source": "cylinderAnnulusToCell",
                    "p1": [0.5, 0.3, 0.1], # p1 and p2 define the axis and width
                    "p2": [0.5, 0.3, 0.5], # p2-p1 should be the axis of the cylinder
                    "innerRadius": 0.0,
                    "outerRadius": 1.5,
                    "power": 10.5, # here we should prescribe the power in W
                }
            }
        }
    */

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = 0.0;
    }

    const dictionary& allOptions = daOption_.getAllOptions();

    dictionary fvSourceSubDict = allOptions.subDict("fvSource");

    word sourceName0 = fvSourceSubDict.toc()[0];
    word source0 = fvSourceSubDict.subDict(sourceName0).getWord("source");

    if (source0 == "cylinderAnnulusToCell")
    {

        // loop over all the cell indices for all heat sources
        forAll(fvSourceCellIndices_.toc(), idxI)
        {

            // name of this heat source
            word sourceName = fvSourceCellIndices_.toc()[idxI];

            // sub dictionary with all parameters for this heat source
            dictionary sourceSubDict = fvSourceSubDict.subDict(sourceName);
            // NOTE: here power should be in W. We will evenly divide the power by the 
            // total volume of the source 
            scalar power = sourceSubDict.getScalar("power");

            // loop over all cell indices for this source and assign the source term
            // ----- first loop, calculate the total volume
            scalar totalV = 0.0;
            forAll(fvSourceCellIndices_[sourceName], idxJ)
            {
                label cellI = fvSourceCellIndices_[sourceName][idxJ];
                scalar cellV = mesh_.V()[cellI];
                totalV += cellV;
            }
            reduce(totalV, sumOp<scalar>());

            // ------- second loop, assign power
            scalar sourceTotal = 0.0;
            forAll(fvSourceCellIndices_[sourceName], idxJ)
            {
                // cell index
                label cellI = fvSourceCellIndices_[sourceName][idxJ];

                fvSource[cellI] = power / totalV;
                sourceTotal += fvSource[cellI] * mesh_.V()[cellI];
            }

            reduce(sourceTotal, sumOp<scalar>());

            Info << "Total volume for " << sourceName << ": " << totalV << " m^3" << endl;
            Info << "Total heat source for " << sourceName << ": " << sourceTotal << " W."<< endl;
        }
    }
    else
    {
        FatalErrorIn("calcFvSourceCells") << "source: " << source0 << " not supported!"
                                          << "Options are: cylinderAnnulusToCell!"
                                          << abort(FatalError);
    }

    fvSource.correctBoundaryConditions();
}

void DAFvSourceHeatSource::calcFvSourceCellIndices(HashTable<labelList>& fvSourceCellIndices)
{
    /*
    Description:
        Calculate the lists of cell indices that are within the source space
        NOTE: we support multiple heat sources.
    
    Output:
        fvSourceCellIndices: Hash table that contains the lists of cell indices that 
        are within the source space. The hash key is the name of the source. We support
        multiple sources. An example of fvSourceCellIndices can be:

        fvSourceCellIndices=
        {
            "source1": {1,2,3,4,5},
            "source2": {6,7,8,9,10,11,12}
        }
    */

    const dictionary& allOptions = daOption_.getAllOptions();

    dictionary fvSourceSubDict = allOptions.subDict("fvSource");

    if (fvSourceSubDict.toc().size() == 0)
    {
        FatalErrorIn("calcSourceCells") << "heatSource is selected as fvSource "
                                        << " but the options are empty!"
                                        << abort(FatalError);
    }

    forAll(fvSourceSubDict.toc(), idxI)
    {
        word sourceName = fvSourceSubDict.toc()[idxI];

        fvSourceCellIndices.set(sourceName, {});

        dictionary sourceSubDict = fvSourceSubDict.subDict(sourceName);
        word sourceType = sourceSubDict.getWord("source");
        // all available source type are in src/meshTools/sets/cellSources
        // Example of IO parameters os in applications/utilities/mesh/manipulation/topoSet
        if (sourceType == "cylinderAnnulusToCell")
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

            scalarList point1;
            scalarList point2;
            sourceSubDict.readEntry<scalarList>("p1", point1);
            sourceSubDict.readEntry<scalarList>("p2", point2);

            point p1;
            point p2;
            p1[0] = point1[0];
            p1[1] = point1[1];
            p1[2] = point1[2];
            p2[0] = point2[0];
            p2[1] = point2[1];
            p2[2] = point2[2];

            scalar outerRadius = sourceSubDict.getScalar("outerRadius");
            scalar innerRadius = sourceSubDict.getScalar("innerRadius");

            dictionary tmpDict;
            tmpDict.set("p1", p1);
            tmpDict.set("p2", p2);
            tmpDict.set("innerRadius", innerRadius);
            tmpDict.set("outerRadius", outerRadius);

            // create the source
            autoPtr<topoSetSource> sourceSet(
                topoSetSource::New(sourceType, mesh_, tmpDict));

            // add the sourceSet to topoSet
            sourceSet().applyToSet(topoSetSource::NEW, currentSet());
            // get the face index from currentSet, we need to use
            // this special for loop
            for (const label i : currentSet())
            {
                fvSourceCellIndices[sourceName].append(i);
            }
        }
        else
        {
            FatalErrorIn("calcFvSourceCells") << "source: " << sourceType << " not supported!"
                                              << "Options are: cylinderAnnulusToCell!"
                                              << abort(FatalError);
        }
    }

    if (daOption_.getOption<label>("debug"))
    {
        Info << "fvSourceCellIndices " << fvSourceCellIndices << endl;
    }
}

} // End namespace Foam

// ************************************************************************* //
