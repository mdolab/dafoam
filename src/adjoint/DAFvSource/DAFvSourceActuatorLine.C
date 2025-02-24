/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFvSourceActuatorLine.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFvSourceActuatorLine, 0);
addToRunTimeSelectionTable(DAFvSource, DAFvSourceActuatorLine, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFvSourceActuatorLine::DAFvSourceActuatorLine(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAFvSource(modelType, mesh, daOption, daModel, daIndex)
{
    printIntervalUnsteady_ = daOption.getOption<label>("printIntervalUnsteady");
    this->initFvSourcePars();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAFvSourceActuatorLine::calcFvSource(volVectorField& fvSource)
{
    /*
    Description:
        Compute the actuator line source term. 
        Reference:  Stokkermans et al. Validation and comparison of RANS propeller 
        modeling methods for tip-mounted applications
        NOTE: rotDir = right means propeller rotates clockwise viewed from 
        the tail of the aircraft looking forward
    */

    const scalar& pi = Foam::constant::mathematical::pi;

    scalar t = mesh_.time().timeOutputValue();

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = vector::zero;
    }

    dictionary fvSourceSubDict = daOption_.getAllOptions().subDict("fvSource");

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
    HashTable<List<scalar>>& actuatorLinePars = globalVar.actuatorLinePars;

    // loop over all the cell indices for all actuator lines
    forAll(fvSourceSubDict.toc(), idxI)
    {
        // name of this line model
        word lineName = fvSourceSubDict.toc()[idxI];

        // sub dictionary with all parameters for this disk
        dictionary lineSubDict = fvSourceSubDict.subDict(lineName);
        // center of the actuator line
        vector center = {
            actuatorLinePars[lineName][0],
            actuatorLinePars[lineName][1],
            actuatorLinePars[lineName][2]};
        // thrust direction
        vector direction = {
            actuatorLinePars[lineName][3],
            actuatorLinePars[lineName][4],
            actuatorLinePars[lineName][5]};
        direction = direction / mag(direction);
        // initial vector for the actuator line
        vector initial = {
            actuatorLinePars[lineName][6],
            actuatorLinePars[lineName][7],
            actuatorLinePars[lineName][8]};
        initial = initial / mag(initial);
        if (fabs(direction & initial) > 1.0e-10)
        {
            FatalErrorIn(" ") << "direction and initial need to be orthogonal!" << abort(FatalError);
        }
        // rotation direction, can be either right or left
        word rotDir = lineSubDict.getWord("rotDir");
        // inner and outer radius of the lines
        scalar innerRadius = actuatorLinePars[lineName][9];
        scalar outerRadius = actuatorLinePars[lineName][10];
        // rotation speed in rpm
        scalar rpm = actuatorLinePars[lineName][15];
        scalar radPerS = rpm / 60.0 * 2.0 * pi;
        // smooth factor which should be of grid size
        scalar eps = actuatorLinePars[lineName][17];
        // scaling factor to ensure a desired integrated thrust
        scalar scale = actuatorLinePars[lineName][11];
        // phase (rad) of the rotation positive value means rotates ahead of time for phase rad
        scalar phase = actuatorLinePars[lineName][16];
        // number of blades for this line model
        label nBlades = lineSubDict.get<label>("nBlades");
        scalar POD = actuatorLinePars[lineName][12];
        scalar expM = actuatorLinePars[lineName][13];
        scalar expN = actuatorLinePars[lineName][14];
        // Now we need to compute normalized eps in the radial direction, i.e. epsRStar this is because
        // we need to smooth the radial distribution of the thrust, here the radial location is
        // normalized as rStar = (r - rInner) / (rOuter - rInner), so to make epsRStar consistent with this
        // we need to normalize eps with the demoninator of rStar, i.e. Outer - rInner
        scalar epsRStar = eps / (outerRadius - innerRadius);
        scalar rStarMin = epsRStar;
        scalar rStarMax = 1.0 - epsRStar;
        scalar fRMin = pow(rStarMin, expM) * pow(1.0 - rStarMin, expN);
        scalar fRMax = pow(rStarMax, expM) * pow(1.0 - rStarMax, expN);

        scalar thrustTotal = 0.0;
        scalar torqueTotal = 0.0;
        forAll(mesh_.C(), cellI)
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
            vector cellC2AVecA = cellC2AVecE & direction;
            // the radial component of cellC2AVec vector
            vector cellC2AVecR = cellC2AVec - cellC2AVecA;
            // now we can use the cross product to compute the tangential
            // (circ) direction of cellI
            vector cellC2AVecC(vector::zero);
            if (rotDir == "left")
            {
                // propeller rotates counter-clockwise viewed from the tail of the aircraft looking forward
                cellC2AVecC = cellC2AVecR ^ direction; // circ
            }
            else if (rotDir == "right")
            {
                // propeller rotates clockwise viewed from the tail of the aircraft looking forward
                cellC2AVecC = direction ^ cellC2AVecR; // circ
            }
            else
            {
                FatalErrorIn(" ") << "rotDir not valid" << abort(FatalError);
            }
            // the magnitude of radial component of cellC2AVecR
            scalar cellC2AVecRLen = mag(cellC2AVecR);
            // the magnitude of tangential component of cellC2AVecR
            scalar cellC2AVecCLen = mag(cellC2AVecC);
            // the magnitude of axial component of cellC2AVecA
            scalar cellC2AVecALen = mag(cellC2AVecA);
            // the normalized cellC2AVecC (tangential) vector
            vector cellC2AVecCNorm = cellC2AVecC / (cellC2AVecCLen + SMALL);
            // smooth coefficient in the axial direction
            scalar etaAxial = exp(-sqr(cellC2AVecALen / eps));

            scalar etaTheta = 0.0;
            for (label bb = 0; bb < nBlades; bb++)
            {
                scalar thetaBlade = bb * 2.0 * pi / nBlades + radPerS * t + phase;
#ifdef CODI_NO_AD
                if (cellI == 0)
                {
                    if (mesh_.time().timeIndex() % printIntervalUnsteady_ == 0
                        || mesh_.time().timeIndex() == 1)
                    {
                        scalar twoPi = 2.0 * pi;
                        Info << "blade " << bb << " theta: "
                             //<< fmod(thetaBlade.getValue(), twoPi.getValue()) * 180.0 / pi.getValue()
                             << fmod(thetaBlade, twoPi) * 180.0 / pi
                             << " deg" << endl;
                    }
                }
#endif
                // compute the rotated vector of initial by thetaBlade degree
                // We use a simplified version of Rodrigues rotation formulation
                vector rotatedVec = vector::zero;
                if (rotDir == "right")
                {
                    rotatedVec = initial * cos(thetaBlade)
                        + (direction ^ initial) * sin(thetaBlade);
                }
                else if (rotDir == "left")
                {
                    rotatedVec = initial * cos(thetaBlade)
                        + (initial ^ direction) * sin(thetaBlade);
                }
                else
                {
                    FatalErrorIn(" ") << "rotDir not valid" << abort(FatalError);
                }
                // scale the rotated vector to have the same length as cellC2AVecR
                rotatedVec *= cellC2AVecRLen;
                // now we can compute the distance between the cellC2AVecR and the rotatedVec
                scalar dS_Theta = mag(cellC2AVecR - rotatedVec);
                // smooth coefficient in the theta direction
                etaTheta += exp(-sqr(dS_Theta / eps));
            }

            // now we can use Hoekstra's formulation to compute radial thrust distribution
            scalar rPrime = cellC2AVecRLen / outerRadius;
            scalar rPrimeHub = innerRadius / outerRadius;
            // rStar is normalized radial location
            scalar rStar = (rPrime - rPrimeHub) / (1.0 - rPrimeHub);

            scalar fAxial = 0.0;
            if (rStar < rStarMin)
            {
                scalar dR2 = (rStar - rStarMin) * (rStar - rStarMin);
                fAxial = fRMin * exp(-dR2 / epsRStar / epsRStar);
            }
            else if (rStar >= rStarMin && rStar <= rStarMax)
            {
                fAxial = pow(rStar, expM) * pow(1.0 - rStar, expN);
            }
            else
            {
                scalar dR2 = (rStar - rStarMax) * (rStar - rStarMax);
                fAxial = fRMax * exp(-dR2 / epsRStar / epsRStar);
            }
            // we use Hoekstra's method to calculate the fCirc based on fAxial
            // here we add 0.01*eps/outerRadius to avoid diving a zero rPrime
            // this might happen if a cell center is very close to actuator center
            scalar fCirc = fAxial * POD / pi / (rPrime + 0.01 * eps / outerRadius);

            vector sourceVec = (fAxial * direction + fCirc * cellC2AVecCNorm);

            fvSource[cellI] += scale * etaAxial * etaTheta * sourceVec;

            thrustTotal += scale * etaAxial * etaTheta * fAxial * mesh_.V()[cellI];
            torqueTotal += scale * etaAxial * etaTheta * fCirc * mesh_.V()[cellI];
        }
        reduce(thrustTotal, sumOp<scalar>());
        reduce(torqueTotal, sumOp<scalar>());

#ifdef CODI_NO_AD
        if (mesh_.time().timeIndex() % printIntervalUnsteady_ == 0 || mesh_.time().timeIndex() == 1)
        {
            Info << "Actuator line source: " << lineName << endl;
            Info << "Total thrust source: " << thrustTotal << endl;
            Info << "Total torque source: " << torqueTotal << endl;
        }
#endif
    }

    fvSource.correctBoundaryConditions();
}

void DAFvSourceActuatorLine::initFvSourcePars()
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
        dictionary lineSubDict = fvSourceSubDict.subDict(diskName);
        word type = lineSubDict.getWord("type");

        if (type == "actuatorLine")
        {

            // now read in all parameters for this actuator disk
            scalarList centerList;
            lineSubDict.readEntry<scalarList>("center", centerList);

            scalarList dirList;
            lineSubDict.readEntry<scalarList>("direction", dirList);

            scalarList initList;
            lineSubDict.readEntry<scalarList>("initial", initList);

            // we have 18 design variables for each line
            scalarList dvList(18);
            dvList[0] = centerList[0];
            dvList[1] = centerList[1];
            dvList[2] = centerList[2];
            dvList[3] = dirList[0];
            dvList[4] = dirList[1];
            dvList[5] = dirList[2];
            dvList[6] = initList[0];
            dvList[7] = initList[1];
            dvList[8] = initList[2];
            dvList[9] = lineSubDict.getScalar("innerRadius");
            dvList[10] = lineSubDict.getScalar("outerRadius");
            dvList[11] = lineSubDict.getScalar("scale");
            dvList[12] = lineSubDict.getScalar("POD");
            dvList[13] = lineSubDict.getScalar("expM");
            dvList[14] = lineSubDict.getScalar("expN");
            dvList[15] = lineSubDict.getScalar("rpm");
            dvList[16] = lineSubDict.getScalar("phase");
            dvList[17] = lineSubDict.getScalar("eps");

            // set actuatorDiskPars
            DAGlobalVar& globalVar =
                const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
            globalVar.actuatorLinePars.set(diskName, dvList);
        }
    }
}

} // End namespace Foam

// ************************************************************************* //
