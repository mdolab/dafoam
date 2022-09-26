/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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
    this->calcFvSourceCellIndices(fvSourceCellIndices_);

    printInterval_ = daOption.getOption<label>("printInterval");

    // now we need to initialize actuatorDiskDVs_ by synchronizing the values
    // defined in fvSource from DAOption to actuatorDiskDVs_
    // NOTE: we need to call this function whenever we change the actuator
    // design variables during optimization
    this->syncDAOptionToActuatorDVs();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAFvSourceActuatorDisk::calcFvSource(volVectorField& fvSource)
{
    /*
    Description:
        Compute the actuator disk source term.
        We follow: Hoekstra, A RANS-based analysis tool for ducted propeller systems 
        in open water condition, International Shipbuilding Progress
        source = rStar * sqrt(1.0 - rStar) * scale
        where rStar is the normalized radial location and scale is used to 
        make sure the integral force equals the desired total thrust
        
        NOTE: rotDir = right means propeller rotates clockwise viewed from 
        the tail of the aircraft looking forward

        There are two options to assign the source term:
        1. cylinderAnnulusToCell. Users prescribe a cylinderAnnulus and the fvSource will be
        added to all the cells inside this cylinderAnnulus
        2. cylinderAnnulusSmooth. Users prescribe the cylinderAnnulus and the Gaussian function
        will be used to smoothly assign fvSource term. This allows us to use the actuatorDisk 
        parameters as design variables and move the actuator during optimization

    Example:
        An example of the fvSource in pyOptions in pyDAFoam can be
        defOptions = 
        {
            "fvSource"
            {
                "disk1"
                {
                    "type": "actuatorDisk",
                    "source": "cylinderAnnulusToCell",
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
                    "source": "cylinderAnnulusSmooth",
                    "center": [0.0, 0.0, 0.0],
                    "direction": [1.0, 0.0, 0.0],
                    "innerRadius": 0.1,
                    "outerRadius": 0.8,
                    "rotDir": "right",
                    "scale": 0.1,
                    "POD": 1.0,
                    "eps": 0.05  # eps should be of cell size
                    "expM": 1.0,
                    "expN": 0.5,
                }
            }
        }
    */

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = vector::zero;
    }

    const dictionary& allOptions = daOption_.getAllOptions();

    dictionary fvSourceSubDict = allOptions.subDict("fvSource");

    word diskName0 = fvSourceSubDict.toc()[0];
    word source0 = fvSourceSubDict.subDict(diskName0).getWord("source");

    if (source0 == "cylinderAnnulusToCell")
    {

        // loop over all the cell indices for all actuator disks
        forAll(fvSourceCellIndices_.toc(), idxI)
        {

            // name of this disk
            word diskName = fvSourceCellIndices_.toc()[idxI];

            // sub dictionary with all parameters for this disk
            dictionary diskSubDict = fvSourceSubDict.subDict(diskName);

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
                    // propeller rotates counter-clockwise viewed from the tail of the aircraft looking forward
                    cellC2AVecC = cellC2AVecR ^ diskDirNorm; // circ
                }
                else if (rotDir == "right")
                {
                    // propeller rotates clockwise viewed from the tail of the aircraft looking forward
                    cellC2AVecC = diskDirNorm ^ cellC2AVecR; // circ
                }
                else
                {
                    FatalErrorIn(" ") << "rotDir not valid" << abort(FatalError);
                }

                // the magnitude of radial component of cellC2AVecR
                scalar cellC2AVecRLen = mag(cellC2AVecR);
                // the magnitude of tangential component of cellC2AVecR
                scalar cellC2AVecCLen = mag(cellC2AVecC);
                // the normalized cellC2AVecC (tangential) vector
                vector cellC2AVecCNorm = cellC2AVecC / cellC2AVecCLen;

                // now we can use Hoekstra's formulation to compute source
                scalar rPrime = cellC2AVecRLen / outerRadius;
                scalar rPrimeHub = innerRadius / outerRadius;
                // rStar is normalized radial location
                scalar rStar = (rPrime - rPrimeHub) / (1.0 - rPrimeHub);
                // axial force, NOTE: user need to prescribe "scale" such that the integrated
                // axial force matches the desired thrust
                scalar fAxial = rStar * sqrt(1.0 - rStar) * scale;
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

            if (daOption_.getOption<word>("runStatus") == "solvePrimal")
            {
                if (mesh_.time().timeIndex() % printInterval_ == 0 || mesh_.time().timeIndex() == 1)
                {
                    Info << "ThrustCoeff Source Term for " << diskName << ": " << thrustSourceSum << endl;
                    Info << "TorqueCoeff Source Term for " << diskName << ": " << torqueSourceSum << endl;
                }
            }
        }
    }
    else if (source0 == "cylinderAnnulusSmooth")
    {

        forAll(fvSourceSubDict.toc(), idxI)
        {
            word diskName = fvSourceSubDict.toc()[idxI];
            dictionary diskSubDict = fvSourceSubDict.subDict(diskName);

            scalarList direction;
            diskSubDict.readEntry<scalarList>("direction", direction);
            vector dirNorm = {direction[0], direction[1], direction[2]};
            dirNorm = dirNorm / mag(dirNorm);
            vector center = {
                actuatorDiskDVs_[diskName][0], actuatorDiskDVs_[diskName][1], actuatorDiskDVs_[diskName][2]};
            scalar innerRadius = actuatorDiskDVs_[diskName][3];
            scalar outerRadius = actuatorDiskDVs_[diskName][4];
            word rotDir = diskSubDict.getWord("rotDir");
            // we will calculate or read scale later
            scalar scale;
            scalar POD = actuatorDiskDVs_[diskName][6];
            scalar eps = diskSubDict.getScalar("eps");
            scalar expM = actuatorDiskDVs_[diskName][7];
            scalar expN = actuatorDiskDVs_[diskName][8];
            // Now we need to compute normalized eps in the radial direction, i.e. epsRStar this is because
            // we need to smooth the radial distribution of the thrust, here the radial location is
            // normalized as rStar = (r - rInner) / (rOuter - rInner), so to make epsRStar consistent with this
            // we need to normalize eps with the demoninator of rStar, i.e. Outer - rInner
            scalar epsRStar = eps / (outerRadius - innerRadius);
            scalar rStarMin = epsRStar;
            scalar rStarMax = 1.0 - epsRStar;
            scalar fRMin = pow(rStarMin, expM) * pow(1.0 - rStarMin, expN);
            scalar fRMax = pow(rStarMax, expM) * pow(1.0 - rStarMax, expN);

            label adjustThrust = diskSubDict.getLabel("adjustThrust");
            // if adjustThrust = False, we just read "scale" from daOption
            // if we want to adjust thrust, we calculate scale, instead of reading from daOption
            // to calculate the scale, we just compute the fAxial with scale = 1, then we find
            // the correct scale = targetThrust / thrust_with_scale_1
            if (adjustThrust)
            {
                scale = 1.0;
                scalar tmpThrustSumAll = 0.0;
                forAll(mesh_.cells(), cellI)
                {
                    // the cell center coordinates of this cellI
                    vector cellC = mesh_.C()[cellI];
                    // cell center to disk center vector
                    vector cellC2AVec = cellC - center;
                    // tmp tensor for calculating the axial/radial components of cellC2AVec
                    tensor cellC2AVecE(tensor::zero);
                    cellC2AVecE.xx() = cellC2AVec.x();
                    cellC2AVecE.yy() = cellC2AVec.y();
                    cellC2AVecE.zz() = cellC2AVec.z();

                    // now we need to decompose cellC2AVec into axial and radial components
                    // the axial component of cellC2AVec vector
                    vector cellC2AVecA = cellC2AVecE & dirNorm;
                    // the radial component of cellC2AVec vector
                    vector cellC2AVecR = cellC2AVec - cellC2AVecA;

                    // the magnitude of radial component of cellC2AVecR
                    scalar cellC2AVecRLen = mag(cellC2AVecR);
                    // the magnitude of axial component of cellC2AVecR
                    scalar cellC2AVecALen = mag(cellC2AVecA);

                    // now we can use the smoothed formulation to compute source
                    scalar rPrime = cellC2AVecRLen / outerRadius;
                    scalar rPrimeHub = innerRadius / outerRadius;
                    // rStar is normalized radial location
                    scalar rStar = (rPrime - rPrimeHub) / (1.0 - rPrimeHub);

                    scalar fAxial = 0.0;

                    scalar dA2 = cellC2AVecALen * cellC2AVecALen;

                    if (rStar < rStarMin)
                    {
                        scalar dR2 = (rStar - rStarMin) * (rStar - rStarMin);
                        scalar fR = fRMin * exp(-dR2 / epsRStar / epsRStar) * scale;
                        fAxial = fR * exp(-dA2 / eps / eps);
                    }
                    else if (rStar >= rStarMin && rStar <= rStarMax)
                    {
                        scalar fR = pow(rStar, expM) * pow(1.0 - rStar, expN) * scale;
                        fAxial = fR * exp(-dA2 / eps / eps);
                    }
                    else
                    {
                        scalar dR2 = (rStar - rStarMax) * (rStar - rStarMax);
                        scalar fR = fRMax * exp(-dR2 / epsRStar / epsRStar) * scale;
                        fAxial = fR * exp(-dA2 / eps / eps);
                    }

                    tmpThrustSumAll += fAxial * mesh_.V()[cellI];
                }
                reduce(tmpThrustSumAll, sumOp<scalar>());
                scalar targetThrust = actuatorDiskDVs_[diskName][9];
                scale = targetThrust / tmpThrustSumAll;
            }
            else
            {
                scale = actuatorDiskDVs_[diskName][5];
            }

            // now we have the correct scale, repeat the loop to assign fvSource
            scalar thrustSourceSum = 0.0;
            scalar torqueSourceSum = 0.0;
            forAll(mesh_.cells(), cellI)
            {
                // the cell center coordinates of this cellI
                vector cellC = mesh_.C()[cellI];
                // cell center to disk center vector
                vector cellC2AVec = cellC - center;
                // tmp tensor for calculating the axial/radial components of cellC2AVec
                tensor cellC2AVecE(tensor::zero);
                cellC2AVecE.xx() = cellC2AVec.x();
                cellC2AVecE.yy() = cellC2AVec.y();
                cellC2AVecE.zz() = cellC2AVec.z();

                // now we need to decompose cellC2AVec into axial and radial components
                // the axial component of cellC2AVec vector
                vector cellC2AVecA = cellC2AVecE & dirNorm;
                // the radial component of cellC2AVec vector
                vector cellC2AVecR = cellC2AVec - cellC2AVecA;

                // now we can use the cross product to compute the tangential
                // (circ) direction of cellI
                vector cellC2AVecC(vector::zero);
                if (rotDir == "left")
                {
                    // propeller rotates counter-clockwise viewed from the tail of the aircraft looking forward
                    cellC2AVecC = cellC2AVecR ^ dirNorm; // circ
                }
                else if (rotDir == "right")
                {
                    // propeller rotates clockwise viewed from the tail of the aircraft looking forward
                    cellC2AVecC = dirNorm ^ cellC2AVecR; // circ
                }
                else
                {
                    FatalErrorIn(" ") << "rotDir not valid" << abort(FatalError);
                }

                // the magnitude of radial component of cellC2AVecR
                scalar cellC2AVecRLen = mag(cellC2AVecR);
                // the magnitude of tangential component of cellC2AVecR
                scalar cellC2AVecCLen = mag(cellC2AVecC);
                // the magnitude of axial component of cellC2AVecR
                scalar cellC2AVecALen = mag(cellC2AVecA);
                // the normalized cellC2AVecC (tangential) vector
                vector cellC2AVecCNorm = cellC2AVecC / cellC2AVecCLen;

                // now we can use the smoothed formulation to compute source
                scalar rPrime = cellC2AVecRLen / outerRadius;
                scalar rPrimeHub = innerRadius / outerRadius;
                // rStar is normalized radial location
                scalar rStar = (rPrime - rPrimeHub) / (1.0 - rPrimeHub);

                scalar fAxial = 0.0;
                scalar dA2 = cellC2AVecALen * cellC2AVecALen;

                if (rStar < rStarMin)
                {
                    scalar dR2 = (rStar - rStarMin) * (rStar - rStarMin);
                    scalar fR = fRMin * exp(-dR2 / epsRStar / epsRStar) * scale;
                    fAxial = fR * exp(-dA2 / eps / eps);
                }
                else if (rStar >= rStarMin && rStar <= rStarMax)
                {
                    scalar fR = pow(rStar, expM) * pow(1.0 - rStar, expN) * scale;
                    fAxial = fR * exp(-dA2 / eps / eps);
                }
                else
                {
                    scalar dR2 = (rStar - rStarMax) * (rStar - rStarMax);
                    scalar fR = fRMax * exp(-dR2 / epsRStar / epsRStar) * scale;
                    fAxial = fR * exp(-dA2 / eps / eps);
                }
                // we use Hoekstra's method to calculate the fCirc based on fAxial
                // here we add 0.01*eps/outerRadius to avoid diving a zero rPrime
                // this might happen if a cell center is very close to actuator center
                scalar fCirc = fAxial * POD / constant::mathematical::pi / (rPrime + 0.01 * eps / outerRadius);

                vector sourceVec = (fAxial * dirNorm + fCirc * cellC2AVecCNorm);
                // the source is the force normalized by the cell volume
                fvSource[cellI] += sourceVec;
                thrustSourceSum += fAxial * mesh_.V()[cellI];
                torqueSourceSum += fCirc * mesh_.V()[cellI];
            }

            reduce(thrustSourceSum, sumOp<scalar>());
            reduce(torqueSourceSum, sumOp<scalar>());

            if (daOption_.getOption<word>("runStatus") == "solvePrimal")
            {
                if (mesh_.time().timeIndex() % printInterval_ == 0 || mesh_.time().timeIndex() == 1)
                {
                    Info << "ThrustCoeff Source Term for " << diskName << ": " << thrustSourceSum << endl;
                    Info << "TorqueCoeff Source Term for " << diskName << ": " << torqueSourceSum << endl;
                    if (adjustThrust)
                    {
                        Info << "Dynamically adjusted scale for " << diskName << ": " << scale << endl;
                    }
                    if (daOption_.getOption<label>("debug"))
                    {
                        Info << "adjustThrust for " << diskName << ": " << adjustThrust << endl;
                        Info << "center for " << diskName << ": " << center << endl;
                        Info << "innerRadius for " << diskName << ": " << innerRadius << endl;
                        Info << "outerRadius for " << diskName << ": " << outerRadius << endl;
                        Info << "scale for " << diskName << ": " << scale << endl;
                        Info << "POD for " << diskName << ": " << POD << endl;
                        Info << "eps for " << diskName << ": " << eps << endl;
                        Info << "expM for " << diskName << ": " << expM << endl;
                        Info << "expN for " << diskName << ": " << expN << endl;
                        Info << "epsRStar for " << diskName << ": " << epsRStar << endl;
                        Info << "rStarMin for " << diskName << ": " << rStarMin << endl;
                        Info << "rStarMax for " << diskName << ": " << rStarMax << endl;
                    }
                }
            }
        }
    }
    else
    {
        FatalErrorIn("calcFvSourceCells") << "source: " << source0 << " not supported!"
                                          << "Options are: cylinderAnnulusToCell and cylinderAnnulusSmooth!"
                                          << abort(FatalError);
    }

    fvSource.correctBoundaryConditions();
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

    const dictionary& allOptions = daOption_.getAllOptions();

    dictionary fvSourceSubDict = allOptions.subDict("fvSource");

    if (fvSourceSubDict.toc().size() == 0)
    {
        FatalErrorIn("calcSourceCells") << "actuatorDisk is selected as fvSource "
                                        << " but the options are empty!"
                                        << abort(FatalError);
    }

    forAll(fvSourceSubDict.toc(), idxI)
    {
        word diskName = fvSourceSubDict.toc()[idxI];

        fvSourceCellIndices.set(diskName, {});

        dictionary diskSubDict = fvSourceSubDict.subDict(diskName);
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
        else if (sourceType == "cylinderAnnulusSmooth")
        {
            // do nothing, no need to compute the cell indices since
            // we are using Gaussian function to compute a smooth
            // distribution of fvSource term
        }
        else
        {
            FatalErrorIn("calcFvSourceCells") << "source: " << sourceType << " not supported!"
                                              << "Options are: cylinderAnnulusToCell and cylinderAnnulusSmooth!"
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
