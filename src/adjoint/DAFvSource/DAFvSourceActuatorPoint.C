/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFvSourceActuatorPoint.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFvSourceActuatorPoint, 0);
addToRunTimeSelectionTable(DAFvSource, DAFvSourceActuatorPoint, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFvSourceActuatorPoint::DAFvSourceActuatorPoint(
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

void DAFvSourceActuatorPoint::calcFvSource(volVectorField& fvSource)
{
    /*
    Description:
        Compute the actuator point source term. We support two types of smooth functions:

        1. Hyperbolic:
            thrust = xT*yT*zT where xT is
            xT = tanh( eps*(meshCX-actCX+0.5*actSX) ) -  tanh( eps*(meshCX-actCX-0.5*actSX) )
            if the mesh cell center is within the actuator box, xT~2 other wise xT~0
            eps is a parameter to smooth the distribution

        2. Gaussian: 
            We use a 2D Gaussian function for thrust source distribution:
            thrust = 1/(2*pi*eps^2)*e^{-d^2/(2*eps^2)} where d is the
            distance between the cell center and actuator source center

            NOTE: this option has some issues for U field... it has many spikes
    
    Example:
        An example of the fvSource in pyOptions in pyDAFoam can be
        defOptions = 
        {
            "fvSource"
            {
                "line1"
                {
                    "type": "actuatorPoint",
                    "smoothFunction": "hyperbolic",
                    "center": [0.0, 0.0, 0.0], # center and size define a rectangular
                    "size": [0.5, 0.3, 0.5],
                    "amplitude": [0.0, 1.0, 0.0],
                    "phase": 0.0,
                    "thrustDirIdx": 0,
                    "periodicity": 1.0,
                    "eps": 1.0,
                    "scale": 1.0  # scale the source such the integral equals desired thrust
                },
                "line2"
                {
                    "type": "actuatorDisk",
                    "smoothFunction": "gaussian",
                    "center": [0.0, 0.0, 0.0],
                    "amplitude": [0.0, 1.0, 0.0],
                    "phase": 1.0,
                    "thrustDirIdx": 0,
                    "periodicity": 1.0,
                    "eps": 1.0,
                    "scale": 1.0  # scale the source such the integral equals desired thrust
                }
            }
        }
    */

    forAll(fvSource, idxI)
    {
        fvSource[idxI] = vector::zero;
    }

    dictionary fvSourceSubDict = daOption_.getAllOptions().subDict("fvSource");

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
    HashTable<List<scalar>>& actuatorPointPars = globalVar.actuatorPointPars;

    // loop over all the cell indices for all actuator points
    forAll(fvSourceSubDict.toc(), idxI)
    {

        // name of this point model
        word pointName = fvSourceSubDict.toc()[idxI];

        // sub dictionary with all parameters for this disk
        dictionary pointSubDict = fvSourceSubDict.subDict(pointName);

        word smoothFunction = pointSubDict.get<word>("smoothFunction");

        if (smoothFunction == "hyperbolic")
        {
            // now read in all parameters for this actuator point
            vector center = {
                actuatorPointPars[pointName][0],
                actuatorPointPars[pointName][1],
                actuatorPointPars[pointName][2]};
            vector size = {
                actuatorPointPars[pointName][3],
                actuatorPointPars[pointName][4],
                actuatorPointPars[pointName][5]};
            vector amp = {
                actuatorPointPars[pointName][6],
                actuatorPointPars[pointName][7],
                actuatorPointPars[pointName][8]};
            scalar period = actuatorPointPars[pointName][10];
            scalar eps = actuatorPointPars[pointName][12];
            scalar scale = actuatorPointPars[pointName][9];
            label thrustDirIdx = pointSubDict.get<label>("thrustDirIdx");
            scalar phase = actuatorPointPars[pointName][11];

            scalar t = mesh_.time().timeOutputValue();
            center += amp * sin(constant::mathematical::twoPi * t / period + phase);

            scalar xTerm, yTerm, zTerm, s;
            scalar thrustTotal = 0.0;
            forAll(mesh_.C(), cellI)
            {
                const vector& meshC = mesh_.C()[cellI];
                xTerm = (tanh(eps * (meshC[0] + 0.5 * size[0] - center[0])) - tanh(eps * (meshC[0] - 0.5 * size[0] - center[0])));
                yTerm = (tanh(eps * (meshC[1] + 0.5 * size[1] - center[1])) - tanh(eps * (meshC[1] - 0.5 * size[1] - center[1])));
                zTerm = (tanh(eps * (meshC[2] + 0.5 * size[2] - center[2])) - tanh(eps * (meshC[2] - 0.5 * size[2] - center[2])));

                s = xTerm * yTerm * zTerm;
                // here we need to use += for multiple actuator points
                fvSource[cellI][thrustDirIdx] += s * scale;

                thrustTotal += s * scale * mesh_.V()[cellI];
            }
            reduce(thrustTotal, sumOp<scalar>());

#ifdef CODI_NO_AD
            if (mesh_.time().timeIndex() % printIntervalUnsteady_ == 0 || mesh_.time().timeIndex() == 0)
            {
                Info << "Actuator point source: " << pointName << endl;
                Info << "Total thrust source: " << thrustTotal << endl;
            }
#endif
        }
        else if (smoothFunction == "gaussian")
        {
            // now read in all parameters for this actuator point
            vector center = {
                actuatorPointPars[pointName][0],
                actuatorPointPars[pointName][1],
                actuatorPointPars[pointName][2]};
            // NOTE: the size variable is not used in the gaussian method, but we set it
            // anyway. So the gaussian case will have empty DV keys for indices from 3 to 5
            vector size = {
                actuatorPointPars[pointName][3],
                actuatorPointPars[pointName][4],
                actuatorPointPars[pointName][5]};
            vector amp = {
                actuatorPointPars[pointName][6],
                actuatorPointPars[pointName][7],
                actuatorPointPars[pointName][8]};
            scalar period = actuatorPointPars[pointName][10];
            scalar eps = actuatorPointPars[pointName][12];
            scalar scale = actuatorPointPars[pointName][9];
            label thrustDirIdx = pointSubDict.get<label>("thrustDirIdx");
            scalar phase = actuatorPointPars[pointName][11];

            scalar t = mesh_.time().timeOutputValue();
            center += amp * sin(constant::mathematical::twoPi * t / period + phase);

            scalar thrustTotal = 0.0;
            scalar coeff = 1.0 / constant::mathematical::twoPi / eps / eps;
            forAll(mesh_.C(), cellI)
            {
                const vector& meshC = mesh_.C()[cellI];
                scalar d = mag(meshC - center);
                scalar s = coeff * exp(-d * d / 2.0 / eps / eps);
                // here we need to use += for multiple actuator points
                fvSource[cellI][thrustDirIdx] += s * scale;
                thrustTotal += s * scale * mesh_.V()[cellI];
            }
            reduce(thrustTotal, sumOp<scalar>());

#ifdef CODI_NO_AD
            if (mesh_.time().timeIndex() % printIntervalUnsteady_ == 0 || mesh_.time().timeIndex() == 1)
            {
                Info << "Actuator point source: " << pointName << endl;
                Info << "Total thrust source: " << thrustTotal << endl;
            }
#endif
        }
        else
        {
            FatalErrorIn("") << "smoothFunction should be either hyperbolic or gaussian" << abort(FatalError);
        }
    }

    fvSource.correctBoundaryConditions();
}

void DAFvSourceActuatorPoint::initFvSourcePars()
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
        word pointName = fvSourceSubDict.toc()[idxI];
        // sub dictionary with all parameters for this disk
        dictionary pointSubDict = fvSourceSubDict.subDict(pointName);
        word type = pointSubDict.getWord("type");

        if (type == "actuatorPoint")
        {

            // now read in all parameters for this actuator disk
            scalarList centerList;
            pointSubDict.readEntry<scalarList>("center", centerList);

            scalarList sizeList;
            word smoothFunction = pointSubDict.getWord("smoothFunction");
            // read the size list for hyperbolic
            if (smoothFunction == "hyperbolic")
            {
                pointSubDict.readEntry<scalarList>("size", sizeList);
            }
            else if (smoothFunction == "gaussian")
            {
                sizeList = {0.0, 0.0, 0.0};
            }

            scalarList ampList;
            pointSubDict.readEntry<scalarList>("amplitude", ampList);

            // we have 13 design variables for each line
            scalarList dvList(13);
            dvList[0] = centerList[0];
            dvList[1] = centerList[1];
            dvList[2] = centerList[2];
            dvList[3] = sizeList[0];
            dvList[4] = sizeList[1];
            dvList[5] = sizeList[2];
            dvList[6] = ampList[0];
            dvList[7] = ampList[1];
            dvList[8] = ampList[2];
            dvList[9] = pointSubDict.getScalar("scale");
            dvList[10] = pointSubDict.getScalar("periodicity");
            dvList[11] = pointSubDict.getScalar("phase");
            dvList[12] = pointSubDict.getScalar("eps");

            // set actuatorDiskPars
            DAGlobalVar& globalVar =
                const_cast<DAGlobalVar&>(mesh_.thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));
            globalVar.actuatorPointPars.set(pointName, dvList);
        }
    }
}

} // End namespace Foam

// ************************************************************************* //
