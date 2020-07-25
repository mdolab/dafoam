/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAFvSourceActuatorDisk.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFvSourceActuatorDisk, 0);
addToRunTimeSelectionTable(DAFvSource, DAFvSourceActuatorDisk, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFvSourceActuatorDisk::DAFvSourceActuatorDisk(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAFvSource(modelType, mesh, daOption, daModel, daIndex)
{
    const dictionary& allOptions = daOption_.getAllOptions();
    fvSourceSubDict_ = allOptions.subDict("fvSource");
    this->calcFvSourceCellIndices(fvSourceCellIndices_);
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAFvSourceActuatorDisk::calcFvSource(volVectorField& fvSource)
{
    /*
    Description:
        Compute the actuator disk source term.
        We follow: Hoekstra, A RANS-based analysis tool for ducted propeller systems 
        in open water condition, International Shipbuilding Progress
        source = rStar * Foam::sqrt(1.0 - rStar) * scale
        where rStar is the normalized radial location and scale is used to 
        make sure the integral force equals the desired total thrust
    
    Example:
        An example of the fvSource in pyOptions in pyDAFoam can be
        defOptions = 
        {
            "fvSource"
            {
                "disk1"
                {
                    "type": "actuatorDisk",
                    "source": cylinderAnnulusToCell,
                    "p1": [0.5, 0.3, 0.1], # p1 and p2 define the axis and width
                    "p2": [0.5, 0.3, 0.5], # p2-p1 should be streamwise
                    "innerRadius": 0.1,
                    "outerRadius": 0.8,
                    "rotDir": "left",
                    "scale": 1.0,
                    "POD": 0.7
                },
                "disk2"
                {
                    "type": "actuatorDisk",
                    "source": cylinderAnnulusToCell,
                    "p1": [0.0, 0.0, 0.1],
                    "p2": [0.0, 0.0, 0.5],
                    "innerRadius": 0.1,
                    "outerRadius": 0.8,
                    "rotDir": "right",
                    "scale": 0.1,
                    "POD": 1.0
                }
            }
        }
    */

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = vector::zero;
    }

    // loop over all the cell indices for all actuator disks
    forAll(fvSourceCellIndices_.toc(), idxI)
    {

        // name of this disk
        word diskName = fvSourceCellIndices_.toc()[idxI];

        // sub dictionary with all parameters for this disk
        dictionary diskSubDict = fvSourceSubDict_.subDict(diskName);

        // now read in all parameters for this actuator disk
        scalarList point1;
        scalarList point2;
        diskSubDict.readEntry<scalarList>("p1", point1);
        diskSubDict.readEntry<scalarList>("p2", point2);
        vector p1 = {point1[0], point1[1], point1[2]};
        vector p2 = {point2[0], point2[1], point2[2]};
        vector diskCenter = (p1 + p2) / 2.0;
        vector diskDir = p2 - p1; // NOTE: p2 - p1 should be streamwise
        vector diskDirNorm = diskDir / mag(diskDir);
        scalar outerRadius = diskSubDict.getScalar("outerRadius");
        scalar innerRadius = diskSubDict.getScalar("innerRadius");
        word rotDir = diskSubDict.getWord("rotDir");
        scalar scale = diskSubDict.getScalar("scale");
        scalar POD = diskSubDict.getScalar("POD");

        // loop over all cell indices for this disk and computer the source term
        scalar thrustSourceSum = 0.0;
        scalar torqueSourceSum = 0.0;
        forAll(fvSourceCellIndices_[diskName], idxJ)
        {
            // cell index
            label cellI = fvSourceCellIndices_[diskName][idxJ];

            // the cell center coordinates of this cellI
            vector cellC = mesh_.C()[cellI];
            // cell center to disk center vector
            vector cellC2AVec = cellC - diskCenter;
            // tmp tensor for calculating the axial/radial components of cellC2AVec
            tensor cellC2AVecE(tensor::zero);
            cellC2AVecE.xx() = cellC2AVec.x();
            cellC2AVecE.yy() = cellC2AVec.y();
            cellC2AVecE.zz() = cellC2AVec.z();

            // now we need to decompose cellC2AVec into axial and radial components
            // the axial component of cellC2AVec vector
            vector cellC2AVecA = cellC2AVecE & diskDirNorm;
            // the radial component of cellC2AVec vector
            vector cellC2AVecR = cellC2AVec - cellC2AVecA;

            // now we can use the cross product to compute the tangential
            // (circ) direction of cellI
            vector cellC2AVecC(vector::zero);
            if (rotDir == "left")
            {
                // this assumes right hand rotation of propellers
                cellC2AVecC = cellC2AVecR ^ cellC2AVecA; // circ
            }
            else if (rotDir == "right")
            {
                // this assumes left hand rotation of propellers
                cellC2AVecC = cellC2AVecA ^ cellC2AVecR; // circ
            }
            else
            {
                FatalErrorIn(" ") << "rotDir not valid" << abort(FatalError);
            }

            // the magnigude of radial compoent of cellC2AVecR
            scalar cellC2AVecRLen = mag(cellC2AVecR);
            // the magnigude of tangential compoent of cellC2AVecR
            scalar cellC2AVecCLen = mag(cellC2AVecC);
            // the normalized cellC2AVecC (tangential) vector
            vector cellC2AVecCNorm = cellC2AVecC / cellC2AVecCLen;

            // now we can use Hoekstra's formulation to compute source
            scalar rPrime = cellC2AVecRLen / outerRadius;
            scalar rPrimeHub = innerRadius / outerRadius;
            // rStar is normalized radial location
            scalar rStar = (rPrime - rPrimeHub) / (1.0 - rPrimeHub);
            // axial force, NOTE: user need to prescribe "scale" such that the integraged
            // axial force matches the desired thrust
            scalar fAxial = rStar * Foam::sqrt(1.0 - rStar) * scale;
            // we use Hoekstra's method to calculate the fCirc based on fAxial
            scalar fCirc = fAxial * POD / constant::mathematical::pi / rPrime;
            vector sourceVec = (fAxial * diskDirNorm + fCirc * cellC2AVecCNorm);
            // the source is the force normalized by the cell volume
            fvSource[cellI] += sourceVec;
            thrustSourceSum += fAxial * mesh_.V()[cellI];
            torqueSourceSum += fCirc * mesh_.V()[cellI];
        }

        reduce(thrustSourceSum, sumOp<scalar>());
        reduce(torqueSourceSum, sumOp<scalar>());

        Info << "ThrustCoeff Source Term for " << diskName << ": " << thrustSourceSum << endl;
        Info << "TorqueCoeff Source Term for " << diskName << ": " << torqueSourceSum << endl;
    }
}

void DAFvSourceActuatorDisk::calcFvSourceCellIndices(HashTable<labelList>& fvSourceCellIndices)
{
    /*
    Description:
        Calculate the lists of cell indices that are within the actuator disk space
        NOTE: we support multiple actuator disks.
    
    Output:
        fvSourceCellIndices: Hash table that contains the lists of cell indices that 
        are within the actuator disk space. The hash key is the name of the disk. We support
        multiple disks. An example of fvSourceCellIndices can be:

        fvSourceCellIndices=
        {
            "disk1": {1,2,3,4,5},
            "disk2": {6,7,8,9,10,11,12}
        }
    
    Example:
        An example of the fvSource in pyOptions in pyDAFoam can be
        defOptions = 
        {
            "fvSource"
            {
                "disk1"
                {
                    "type": "actuatorDisk",
                    "source": cylinderAnnulusToCell,
                    "p1": [0.5, 0.3, 0.1], # p1 and p2 define the axis and width
                    "p2": [0.5, 0.3, 0.5],
                    "innerRadius": 0.1,
                    "outerRadius": 0.8,
                    "rotDir": "left",
                    "scale": 1.0,
                    "POD": 0.7
                },
                "disk2"
                {
                    "type": "actuatorDisk",
                    "source": cylinderAnnulusToCell,
                    "p1": [0.0, 0.0, 0.1],
                    "p2": [0.0, 0.0, 0.5],
                    "innerRadius": 0.1,
                    "outerRadius": 0.8,
                    "rotDir": "right",
                    "scale": 0.1,
                    "POD": 1.0
                }
            }
        }
    */

    if (fvSourceSubDict_.toc().size() == 0)
    {
        FatalErrorIn("calcSourceCells") << "actuatorDisk is selected as fvSource "
                                        << " but the options are empty!"
                                        << abort(FatalError);
    }

    forAll(fvSourceSubDict_.toc(), idxI)
    {
        word diskName = fvSourceSubDict_.toc()[idxI];

        fvSourceCellIndices.set(diskName, {});

        dictionary diskSubDict = fvSourceSubDict_.subDict(diskName);
        word sourceType = diskSubDict.getWord("source");
        // all avaiable source type are in src/meshTools/sets/cellSources
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
            diskSubDict.readEntry<scalarList>("p1", point1);
            diskSubDict.readEntry<scalarList>("p2", point2);

            point p1;
            point p2;
            p1[0] = point1[0];
            p1[1] = point1[1];
            p1[2] = point1[2];
            p2[0] = point2[0];
            p2[1] = point2[1];
            p2[2] = point2[2];

            scalar outerRadius = diskSubDict.getScalar("outerRadius");
            scalar innerRadius = diskSubDict.getScalar("innerRadius");

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
                fvSourceCellIndices[diskName].append(i);
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
