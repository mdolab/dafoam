/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
    this->initFvSourcePars();

    printInterval_ = daOption.getOption<label>("printInterval");

    const dictionary& allOptions = daOption_.getAllOptions();

    dictionary fvSourceSubDict = allOptions.subDict("fvSource");

    forAll(fvSourceSubDict.toc(), idxI)
    {
        word sourceName = fvSourceSubDict.toc()[idxI];
        dictionary sourceSubDict = fvSourceSubDict.subDict(sourceName);
        word sourceType = sourceSubDict.getWord("source");

        if (sourceType == "cylinderToCell")
        {
            scalarList p1List;
            sourceSubDict.readEntry<scalarList>("p1", p1List);
            cylinderP1_.set(sourceName, p1List);
            scalarList p2List;
            sourceSubDict.readEntry<scalarList>("p2", p2List);
            cylinderP2_.set(sourceName, p2List);

            cylinderRadius_.set(sourceName, sourceSubDict.getScalar("radius"));
            power_.set(sourceName, sourceSubDict.getScalar("power"));

            fvSourceCellIndices_.set(sourceName, {});
            // all available source type are in src/meshTools/sets/cellSources
            // Example of IO parameters os in applications/utilities/mesh/manipulation/topoSet

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

            point p1;
            point p2;
            p1[0] = cylinderP1_[sourceName][0];
            p1[1] = cylinderP1_[sourceName][1];
            p1[2] = cylinderP1_[sourceName][2];
            p2[0] = cylinderP2_[sourceName][0];
            p2[1] = cylinderP2_[sourceName][1];
            p2[2] = cylinderP2_[sourceName][2];

            dictionary tmpDict;
            tmpDict.set("p1", p1);
            tmpDict.set("p2", p2);
            tmpDict.set("radius", cylinderRadius_[sourceName]);

            // create the source
            autoPtr<topoSetSource> sourceSet(
                topoSetSource::New("cylinderToCell", mesh_, tmpDict));

            // add the sourceSet to topoSet
            sourceSet().applyToSet(topoSetSource::NEW, currentSet());
            // get the face index from currentSet, we need to use
            // this special for loop
            for (const label i : currentSet())
            {
                fvSourceCellIndices_[sourceName].append(i);
            }

            if (daOption_.getOption<label>("debug"))
            {
                Info << "fvSourceCellIndices " << fvSourceCellIndices_ << endl;
            }
        }
        else if (sourceType == "cylinderSmooth")
        {
            DAGlobalVar& globalVar =
                const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
            HashTable<List<scalar>>& heatSourcePars = globalVar.heatSourcePars;

            // eps is a smoothing parameter, it should be the local mesh cell size in meters
            // near the cylinder region
            cylinderEps_.set(sourceName, sourceSubDict.getScalar("eps"));

            snapCenter2Cell_.set(sourceName, sourceSubDict.lookupOrDefault<label>("snapCenter2Cell", 0));
            if (snapCenter2Cell_[sourceName])
            {
                point centerPoint = {heatSourcePars[sourceName][0], heatSourcePars[sourceName][1], heatSourcePars[sourceName][2]};

                // NOTE: we need to call a self-defined findCell func to make it work correctly in ADR
                label myCellI = DAUtility::myFindCell(mesh_, centerPoint);

                snappedCenterCellI_.set(sourceName, myCellI);
                label foundCellI = 0;
                if (snappedCenterCellI_[sourceName] >= 0)
                {
                    foundCellI = 1;
                    //Pout << "snap source " << sourceName << " to center " << mesh_.C()[snappedCenterCellI_[sourceName]] << endl;
                }
                reduce(foundCellI, sumOp<label>());
                if (foundCellI != 1)
                {
                    FatalErrorIn(" ") << "There should be only one cell found globally while "
                                      << foundCellI << " was returned."
                                      << " Please adjust center such that it is located completely"
                                      << " within a cell in the mesh domain. The center should not "
                                      << " be outside of the mesh domain or on a mesh face "
                                      << abort(FatalError);
                }

                vector snappedCenter = vector::zero;
                this->findGlobalSnappedCenter(snappedCenterCellI_[sourceName], snappedCenter);
                Info << "heat source " << sourceName << " snap to center " << snappedCenter << endl;
            }
        }
        else
        {
            FatalErrorIn("DAFvSourceHeatSource") << "source: " << sourceType << " not supported!"
                                                 << "Options are: cylinderAnnulusToCell and cylinderSmooth!"
                                                 << abort(FatalError);
        }
    }
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
                    "source": "cylinderToCell",
                    "p1": [0.5, 0.3, 0.1], # p1 and p2 define the axis and width
                    "p2": [0.5, 0.3, 0.5], # p2-p1 should be the axis of the cylinder
                    "radius": 0.8,
                    "power": 101.0,  # here we should prescribe the power in W
                },
                "source2"
                {
                    "type": "heatSource",
                    "source": "cylinder",
                    "p1": [0.5, 0.3, 0.1], # p1 and p2 define the axis and width
                    "p2": [0.5, 0.3, 0.5], # p2-p1 should be the axis of the cylinder
                    "radius": 1.5,
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

    forAll(fvSourceSubDict.toc(), idxI)
    {
        word sourceName = fvSourceSubDict.toc()[idxI];
        dictionary sourceSubDict = fvSourceSubDict.subDict(sourceName);
        word sourceType = sourceSubDict.getWord("source");
        // NOTE: here power should be in W. We will evenly divide the power by the
        // total volume of the source
        if (sourceType == "cylinderToCell")
        {

            scalar power = power_[sourceName];

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

                fvSource[cellI] += power / totalV;
                sourceTotal += fvSource[cellI] * mesh_.V()[cellI];
            }

            reduce(sourceTotal, sumOp<scalar>());

#ifndef CODI_ADR
            if (mesh_.time().timeIndex() % printInterval_ == 0 || mesh_.time().timeIndex() == 1)
            {
                Info << "Total volume for " << sourceName << ": " << totalV << " m^3" << endl;
                Info << "Total heat source for " << sourceName << ": " << sourceTotal << " W." << endl;
            }
#endif
        }
        else if (sourceType == "cylinderSmooth")
        {
            DAGlobalVar& globalVar =
                const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
            HashTable<List<scalar>>& heatSourcePars = globalVar.heatSourcePars;

            vector cylinderCenter =
                {heatSourcePars[sourceName][0], heatSourcePars[sourceName][1], heatSourcePars[sourceName][2]};

            if (snapCenter2Cell_[sourceName])
            {
                this->findGlobalSnappedCenter(snappedCenterCellI_[sourceName], cylinderCenter);
            }
            vector cylinderDir =
                {heatSourcePars[sourceName][3], heatSourcePars[sourceName][4], heatSourcePars[sourceName][5]};
            scalar radius = heatSourcePars[sourceName][6];
            scalar cylinderLen = heatSourcePars[sourceName][7];
            scalar power = heatSourcePars[sourceName][8];
            vector cylinderDirNorm = cylinderDir / mag(cylinderDir);
            scalar eps = cylinderEps_[sourceName];

            // first loop, calculate the total volume
            scalar volumeTotal = 0.0;
            forAll(mesh_.cells(), cellI)
            {
                // the cell center coordinates of this cellI
                vector cellC = mesh_.C()[cellI];
                // cell center to disk center vector
                vector cellC2AVec = cellC - cylinderCenter;
                // tmp tensor for calculating the axial/radial components of cellC2AVec
                tensor cellC2AVecE(tensor::zero);
                cellC2AVecE.xx() = cellC2AVec.x();
                cellC2AVecE.yy() = cellC2AVec.y();
                cellC2AVecE.zz() = cellC2AVec.z();

                // now we need to decompose cellC2AVec into axial and radial components
                // the axial component of cellC2AVec vector
                vector cellC2AVecA = cellC2AVecE & cylinderDirNorm;
                // the radial component of cellC2AVec vector
                vector cellC2AVecR = cellC2AVec - cellC2AVecA;

                // the magnitude of radial component of cellC2AVecR
                scalar cellC2AVecRLen = mag(cellC2AVecR);
                // the magnitude of axial component of cellC2AVecR
                scalar cellC2AVecALen = mag(cellC2AVecA);

                scalar axialSmoothC = 1.0;
                scalar radialSmoothC = 1.0;

                scalar halfL = cylinderLen / 2.0;

                if (cellC2AVecALen > halfL)
                {
                    scalar d2 = (cellC2AVecALen - halfL) * (cellC2AVecALen - halfL);
                    axialSmoothC = exp(-d2 / eps / eps);
                }

                if (cellC2AVecRLen > radius)
                {
                    scalar d2 = (cellC2AVecRLen - radius) * (cellC2AVecRLen - radius);
                    radialSmoothC = exp(-d2 / eps / eps);
                }
                volumeTotal += axialSmoothC * radialSmoothC * mesh_.V()[cellI];
            }
            reduce(volumeTotal, sumOp<scalar>());

            // second loop, calculate fvSource
            scalar sourceTotal = 0.0;
            forAll(mesh_.cells(), cellI)
            {
                // the cell center coordinates of this cellI
                vector cellC = mesh_.C()[cellI];
                // cell center to disk center vector
                vector cellC2AVec = cellC - cylinderCenter;
                // tmp tensor for calculating the axial/radial components of cellC2AVec
                tensor cellC2AVecE(tensor::zero);
                cellC2AVecE.xx() = cellC2AVec.x();
                cellC2AVecE.yy() = cellC2AVec.y();
                cellC2AVecE.zz() = cellC2AVec.z();

                // now we need to decompose cellC2AVec into axial and radial components
                // the axial component of cellC2AVec vector
                vector cellC2AVecA = cellC2AVecE & cylinderDirNorm;
                // the radial component of cellC2AVec vector
                vector cellC2AVecR = cellC2AVec - cellC2AVecA;

                // the magnitude of radial component of cellC2AVecR
                scalar cellC2AVecRLen = mag(cellC2AVecR);
                // the magnitude of axial component of cellC2AVecR
                scalar cellC2AVecALen = mag(cellC2AVecA);

                scalar axialSmoothC = 1.0;
                scalar radialSmoothC = 1.0;

                scalar halfL = cylinderLen / 2.0;

                if (cellC2AVecALen > halfL)
                {
                    scalar d2 = (cellC2AVecALen - halfL) * (cellC2AVecALen - halfL);
                    axialSmoothC = exp(-d2 / eps / eps);
                }

                if (cellC2AVecRLen > radius)
                {
                    scalar d2 = (cellC2AVecRLen - radius) * (cellC2AVecRLen - radius);
                    radialSmoothC = exp(-d2 / eps / eps);
                }

                fvSource[cellI] += power / volumeTotal * axialSmoothC * radialSmoothC;
                sourceTotal += power / volumeTotal * axialSmoothC * radialSmoothC * mesh_.V()[cellI];
            }

            reduce(sourceTotal, sumOp<scalar>());

#ifndef CODI_ADR
            if (mesh_.time().timeIndex() % printInterval_ == 0 || mesh_.time().timeIndex() == 1)
            {
                Info << "Total volume for " << sourceName << ": " << volumeTotal << " m^3" << endl;
                Info << "Total heat source for " << sourceName << ": " << sourceTotal << " W." << endl;
            }
#endif
        }
        else
        {
            FatalErrorIn("DAFvSourceHeatSource") << "source: " << sourceType << " not supported!"
                                                 << "Options are: cylinderAnnulusToCell and cylinderSmooth!"
                                                 << abort(FatalError);
        }
    }

    fvSource.correctBoundaryConditions();
}

void DAFvSourceHeatSource::initFvSourcePars()
{
    /*
    Description:
        Initialize the values for all types of fvSource in DAGlobalVar, including 
        actuatorDiskPars, heatSourcePars, etc
    */

    // now we need to initialize actuatorDiskPars
    dictionary fvSourceSubDict = daOption_.getAllOptions().subDict("fvSource");

    forAll(fvSourceSubDict.toc(), idxI)
    {
        word diskName = fvSourceSubDict.toc()[idxI];
        // sub dictionary with all parameters for this disk
        dictionary diskSubDict = fvSourceSubDict.subDict(diskName);
        word type = diskSubDict.getWord("type");
        word source = diskSubDict.getWord("source");

        if (type == "heatSource" && source == "cylinderSmooth")
        {
            // now read in all parameters for this actuator disk
            scalarList centerList;
            diskSubDict.readEntry<scalarList>("center", centerList);

            scalarList axisList;
            diskSubDict.readEntry<scalarList>("axis", axisList);

            // we have 13 design variables for each disk
            scalarList dvList(9);
            dvList[0] = centerList[0];
            dvList[1] = centerList[1];
            dvList[2] = centerList[2];
            dvList[3] = axisList[0];
            dvList[4] = axisList[1];
            dvList[5] = axisList[2];
            dvList[6] = diskSubDict.getScalar("radius");
            dvList[7] = diskSubDict.getScalar("length");
            dvList[8] = diskSubDict.getScalar("power");

            // set actuatorDiskPars
            DAGlobalVar& globalVar =
                const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
            globalVar.heatSourcePars.set(diskName, dvList);
        }
    }
}

} // End namespace Foam

// ************************************************************************* //
