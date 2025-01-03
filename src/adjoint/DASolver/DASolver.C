/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// initialize the static variable, which will be used in forward mode AD
// computation for AOA and BC derivatives
scalar Foam::DAUtility::angleOfAttackRadForwardAD = -9999.0;

// initialize the python call back function static pointers
void* Foam::DAUtility::pyCalcBeta = NULL;
pyComputeInterface Foam::DAUtility::pyCalcBetaInterface = NULL;

void* Foam::DAUtility::pyCalcBetaJacVecProd = NULL;
pyJacVecProdInterface Foam::DAUtility::pyCalcBetaJacVecProdInterface = NULL;

void* Foam::DAUtility::pySetModelName = NULL;
pySetCharInterface Foam::DAUtility::pySetModelNameInterface = NULL;

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DASolver, 0);
defineRunTimeSelectionTable(DASolver, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DASolver::DASolver(
    char* argsAll,
    PyObject* pyOptions)
    : argsAll_(argsAll),
      pyOptions_(pyOptions),
      argsPtr_(nullptr),
      runTimePtr_(nullptr),
      meshPtr_(nullptr),
      daOptionPtr_(nullptr),
      daModelPtr_(nullptr),
      daIndexPtr_(nullptr),
      daFieldPtr_(nullptr),
      daCheckMeshPtr_(nullptr),
      daLinearEqnPtr_(nullptr),
      daResidualPtr_(nullptr),
      daRegressionPtr_(nullptr),
      daGlobalVarPtr_(nullptr)
#ifdef CODI_ADR
      ,
      globalADTape_(codi::RealReverse::getTape())
#endif
{
    // initialize fvMesh and Time object pointer
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTimePython.H"
#include "createMeshPython.H"
    Info << "Initializing mesh and runtime for DASolver" << endl;

    daOptionPtr_.reset(new DAOption(meshPtr_(), pyOptions_));

    primalMinResTol_ = daOptionPtr_->getOption<scalar>("primalMinResTol");
    primalMinIters_ = daOptionPtr_->getOption<label>("primalMinIters");
    printInterval_ = daOptionPtr_->getOption<label>("printInterval");

    // initialize the objStd variables.
    this->initObjStd();

    Info << "DAOpton initialized " << endl;
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * d* * * * //

autoPtr<DASolver> DASolver::New(
    char* argsAll,
    PyObject* pyOptions)
{
    // standard setup for runtime selectable classes

    // look up the solver name defined in pyOptions
    dictionary allOptions;
    DAUtility::pyDict2OFDict(pyOptions, allOptions);
    word modelType;
    allOptions.readEntry<word>("solverName", modelType);

    if (allOptions.lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DASolver" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DASolver::New"
            "("
            "    char*,"
            "    PyObject*"
            ")")
            << "Unknown DASolver type "
            << modelType << nl << nl
            << "Valid DASolver types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DASolver>(
        cstrIter()(argsAll, pyOptions));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

label DASolver::loop(Time& runTime)
{
    /*
    Description:
        The loop method to increment the runtime. The reason we implement this is
        because the runTime.loop() and simple.loop() give us seg fault...
    */

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    scalar t = runTime.timeOutputValue();

    // execute functionObjectList, e.g., field averaging, sampling
    functionObjectList& funcObj = const_cast<functionObjectList&>(runTime.functionObjects());
    if (runTime.timeIndex() == runTime.startTimeIndex())
    {
        funcObj.start();
    }
    else
    {
        funcObj.execute();
    }

    // calculate the objective function standard deviation. It will be used in determining if the primal converges
    this->calcObjStd(runTime);

    // check exit condition, we need to satisfy both the residual and function std condition
    if (primalMaxRes_ < primalMinResTol_ && runTime.timeIndex() > primalMinIters_ && primalObjStd_ < primalObjStdTol_)
    {
        Info << "Time = " << t << endl;

        Info << "Minimal residual " << primalMaxRes_ << " satisfied the prescribed tolerance " << primalMinResTol_ << endl
             << endl;

        if (primalObjStdActive_)
        {
            Info << "Function standard deviation " << primalObjStd_ << " satisfied the prescribed tolerance " << primalObjStdTol_ << endl
                 << endl;
        }

        this->printAllFunctions();
        runTime.writeNow();
        prevPrimalSolTime_ = t;
        funcObj.end();
        daRegressionPtr_->writeFeatures();
        return 0;
    }
    else if (t > endTime - 0.5 * deltaT)
    {
        prevPrimalSolTime_ = t;
        funcObj.end();
        daRegressionPtr_->writeFeatures();
        return 0;
    }
    else
    {
        ++runTime;
        // initialize primalMaxRes_ with a small value for this iteration
        primalMaxRes_ = -1e10;
        printToScreen_ = this->isPrintTime(runTime, printInterval_);
        return 1;
    }
}

void DASolver::initObjStd()
{
    /*
    Description:
        Initialize the objStd variables.
    */

    // check if the objective function std is used in determining the primal convergence
    primalObjStdActive_ = daOptionPtr_->getSubDictOption<label>("primalObjStdTol", "active");
    if (primalObjStdActive_)
    {
        // if it is active, read in the tolerance and set a large value for the initial std
        primalObjStdTol_ = daOptionPtr_->getSubDictOption<scalar>("primalObjStdTol", "tol");
        primalObjStd_ = 999.0;

        label steps = daOptionPtr_->getSubDictOption<label>("primalObjStdTol", "steps");
        primalObjSeries_.setSize(steps, 0.0);

        word functionNameWanted = daOptionPtr_->getSubDictOption<word>("primalObjStdTol", "functionName");

        label functionNameFound = 0;
        const dictionary& functionDict = daOptionPtr_->getAllOptions().subDict("function");
        forAll(functionDict.toc(), idxI)
        {
            word functionName = functionDict.toc()[idxI];
            if (functionName == functionNameWanted)
            {
                functionNameFound = 1;
            }
        }
        if (functionNameFound == 0)
        {
            FatalErrorIn("initObjStd") << "objStd->functionName not found! "
                                       << abort(FatalError);
        }
    }
    else
    {
        // if it is not active, set primalObjStdTol_ > primalObjStd_, such that it will
        // always pass the condition in DASolver::loop (ignore primalObjStd)
        primalObjStdTol_ = 1e-5;
        primalObjStd_ = 0.0;
    }
}

void DASolver::calcObjStd(Time& runTime)
{
    /*
    Description:
        calculate the objective function's std, this will be used to stop the primal simulation and also 
        evaluate whether the primal converges. We will start calculating the objStd when primalObjSeries_
        is filled at least once, i.e., runTime.timeIndex() >= steps
    */

    if (!primalObjStdActive_)
    {
        return;
    }
    else if (runTime.timeIndex() < 1)
    {
        // if primalObjStd is active and timeIndex = 0, we need to reset primalObjStd_ to a large value
        // NOTE: we need to reset primalObjStd_ for each primal call!
        // Because timeIndex == 0, we don't need to compute the objStd, so we can return
        // we will start computing the ojbStd for timeIndex>=1
        primalObjStd_ = 999.0;
        return;
    }

    label steps = daOptionPtr_->getSubDictOption<label>("primalObjStdTol", "steps");

    word functionNameWanted = daOptionPtr_->getSubDictOption<word>("primalObjStdTol", "functionName");

    scalar functionSum = 0.0;
    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName = daFunction.getFunctionName();
        if (functionName == functionNameWanted)
        {
            functionSum += daFunction.getFunctionValue();
        }
    }
    label seriesI = (runTime.timeIndex() - 1) % steps;
    primalObjSeries_[seriesI] = functionSum;

    if (runTime.timeIndex() >= steps)
    {
        scalar mean = 0;
        forAll(primalObjSeries_, idxI)
        {
            mean += primalObjSeries_[idxI];
        }
        mean /= steps;
        primalObjStd_ = 0.0;
        forAll(primalObjSeries_, idxI)
        {
            primalObjStd_ += (primalObjSeries_[idxI] - mean) * (primalObjSeries_[idxI] - mean);
        }
        primalObjStd_ = sqrt(primalObjStd_ / steps);
    }
}

void DASolver::calcUnsteadyFunctions()
{
    /*
    Description:
        Calculate unsteady objective function, e.g., the time average obj func
    */

    if (daFunctionPtrList_.size() == 0)
    {
        FatalErrorIn("calcUnsteadyFunctions") << "daFunctionPtrList_.size() ==0... "
                                              << "Forgot to call setDAFunctionList?"
                                              << abort(FatalError);
    }

    word timeOperator = daOptionPtr_->getSubDictOption<word>("unsteadyAdjoint", "functionTimeOperator");

    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName = daFunction.getFunctionName();
        word uKey = functionName;
        if (timeOperator == "None")
        {
            FatalErrorIn("") << "calcUnsteadyFunctions is called but the timeOperator is not set!!! Options are: average or sum"
                             << abort(FatalError);
        }
        else if (timeOperator == "sum")
        {
            label startTimeIndex = this->getUnsteadyFunctionStartTimeIndex();
            label endTimeIndex = this->getUnsteadyFunctionEndTimeIndex();
            label timeIndex = runTimePtr_->timeIndex();
            if (timeIndex >= startTimeIndex && timeIndex <= endTimeIndex)
            {
                unsteadyFunctions_[uKey] += daFunction.getFunctionValue();
            }
        }
        else if (timeOperator == "average")
        {
            // calculate the average on the fly, i.e., moving average
            label startTimeIndex = this->getUnsteadyFunctionStartTimeIndex();
            label endTimeIndex = this->getUnsteadyFunctionEndTimeIndex();
            label timeIndex = runTimePtr_->timeIndex();
            if (timeIndex >= startTimeIndex && timeIndex <= endTimeIndex)
            {
                label n = timeIndex - startTimeIndex + 1;
                scalar functionVal = daFunction.getFunctionValue();
                unsteadyFunctions_[uKey] = (unsteadyFunctions_[uKey] * (n - 1) + functionVal) / n;
            }
        }
        else
        {
            FatalErrorIn("") << "calcUnsteadyFunctions is called but the timeOperator is not set!!! Options are: average or sum"
                             << abort(FatalError);
        }
    }
}

void DASolver::printAllFunctions()
{
    /*
    Description:
        Calculate the values of all objective functions and print them to screen
        NOTE: we need to call DASolver::setDAFunctionList before calling this function!
    */

    if (daFunctionPtrList_.size() == 0)
    {
        FatalErrorIn("printAllFunctions") << "daFunctionPtrList_.size() ==0... "
                                          << "Forgot to call setDAFunctionList?"
                                          << abort(FatalError);
    }

    word timeOperator = daOptionPtr_->getSubDictOption<word>("unsteadyAdjoint", "functionTimeOperator");

    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName = daFunction.getFunctionName();
        word uKey = functionName;
        scalar functionVal = daFunction.getFunctionValue();
        Info << functionName
             << "-" << daFunction.getFunctionType()
             << ": " << functionVal;
        if (primalObjStdActive_)
        {
            word functionNameWanted = daOptionPtr_->getSubDictOption<word>("primalObjStdTol", "functionName");
            if (functionNameWanted == functionName)
            {
                Info << " Std " << primalObjStd_;
            }
        }
        if (timeOperator == "average" || timeOperator == "sum")
        {
            Info << " Unsteady " << timeOperator << " " << unsteadyFunctions_[uKey];
        }
#ifdef CODI_ADF

        // if the forwardModeAD is active,, we need to get the total derivatives here
        if (daOptionPtr_->getAllOptions().subDict("useAD").getWord("mode") == "forward")
        {
            Info << " ForwardAD Deriv: " << functionVal.getGradient();

            // assign the forward mode AD derivative to forwardADDerivVal_
            // such that we can get this value later
            forwardADDerivVal_.set(functionName, functionVal.getGradient());

            if (daOptionPtr_->getSubDictOption<word>("unsteadyAdjoint", "mode") == "timeAccurate")
            {
                Info << " Unsteady Deriv: " << unsteadyFunctions_[uKey].getGradient();
            }
        }
#endif
        Info << endl;
    }
}

scalar DASolver::getFunctionValueUnsteady(const word functionName)
{
    /*
    Description:
        Return the value of the objective function for unsteady cases
        NOTE: we will sum up all the parts in functionName

    Input:
        functionName: the name of the objective function

    Output:
        functionValue: the value of the objective
    */
    if (daFunctionPtrList_.size() == 0)
    {
        FatalErrorIn("printAllFunctions") << "daFunctionPtrList_.size() ==0... "
                                          << "Forgot to call setDAFunctionList?"
                                          << abort(FatalError);
    }

    scalar functionValue = 0.0;

    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionNameI = daFunction.getFunctionName();
        word uKey = functionNameI;

        if (functionNameI == functionName)
        {
            functionValue += unsteadyFunctions_[uKey];
        }
    }

    return functionValue;
}

scalar DASolver::getFunctionValue(const word functionName)
{
    /*
    Description:
        Return the value of the objective function.
        NOTE: we will sum up all the parts in functionName

    Input:
        functionName: the name of the objective function

    Output:
        functionValue: the value of the objective
    */

    if (daFunctionPtrList_.size() == 0)
    {
        FatalErrorIn("printAllFunctions") << "daFunctionPtrList_.size() ==0... "
                                          << "Forgot to call setDAFunctionList?"
                                          << abort(FatalError);
    }

    scalar functionValue = 0.0;

    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        if (daFunction.getFunctionName() == functionName)
        {
            functionValue += daFunction.getFunctionValue();
        }
    }

    return functionValue;
}

void DASolver::setDAFunctionList()
{
    /*
    Description:
        Set up the objective function list such that we can call printAllFunctions and getFunctionValue
        NOTE: this function needs to be called before calculating any objective functions

    Example:
        A typical function dictionary looks like this:
    
        "function": 
        {
            "func0": 
            {
                "functionName": "force",
                "source": "patchToFace",
                "patches": ["walls", "wallsbump"],
                "scale": 0.5,
            },
            "func1":
            {
                "functionName": "force",
                "source": "patchToFace",
                "patches": ["wallsbump", "frontandback"],
                "scale": 0.5,
            },
            "func2": 
            {
                "functionName": "force",
                "source": "patchToFace",
                "patches": ["walls", "wallsbump", "frontandback"],
                "scale": 1.0,
            },
        }
    */

    const dictionary& allOptions = daOptionPtr_->getAllOptions();

    const dictionary& functionDict = allOptions.subDict("function");

    // loop over all functions and parts and calc the number of
    // DAFunction instances we need
    label nFunctionInstances = 0;
    forAll(functionDict.toc(), idxI)
    {
        nFunctionInstances++;
    }

    daFunctionPtrList_.setSize(nFunctionInstances);

    // we need to repeat the loop to initialize the
    // DAFunction instances
    forAll(functionDict.toc(), idxI)
    {
        word functionName = functionDict.toc()[idxI];
        daFunctionPtrList_.set(
            idxI,
            DAFunction::New(
                meshPtr_(),
                daOptionPtr_(),
                daModelPtr_(),
                daIndexPtr_(),
                functionName)
                .ptr());
    }

    // here we also initialize the unsteadyFunctions hashtable
    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName = daFunction.getFunctionName();
        word uKey = functionName;
        unsteadyFunctions_.set(uKey, 0.0);

        word timeOperator = daOptionPtr_->getSubDictOption<word>("unsteadyAdjoint", "functionTimeOperator");
        if (timeOperator == "None" || timeOperator == "sum")
        {
            unsteadyFunctionsScaling_ = 1.0;
        }
        else if (timeOperator == "average")
        {
            label startTimeIndex = this->getUnsteadyFunctionStartTimeIndex();
            label endTimeIndex = this->getUnsteadyFunctionEndTimeIndex();
            label nInstances = endTimeIndex - startTimeIndex + 1;
            unsteadyFunctionsScaling_ = 1.0 / nInstances;
        }
        else
        {
            FatalErrorIn("setDAFunctionList") << "timeOperator not valid! Options are None, average, or sum"
                                              << abort(FatalError);
        }
    }
}

label DASolver::getNCouplingFaces()
{
    /*
    Description:
        Get the number of faces for the MDO coupling patches
    */
    wordList patchList;
    this->getCouplingPatchList(patchList);
    // get the total number of points and faces for the patchList
    label nPoints, nFaces;
    this->getPatchInfo(nPoints, nFaces, patchList);

    return nFaces;
}

label DASolver::getNCouplingPoints()
{
    /*
    Description:
        Get the number of points for the MDO coupling patches
    */
    wordList patchList;
    this->getCouplingPatchList(patchList);
    // get the total number of points and faces for the patchList
    label nPoints, nFaces;
    this->getPatchInfo(nPoints, nFaces, patchList);

    return nPoints;
}

void DASolver::calcCouplingFaceCoords(
    const scalar* volCoords,
    scalar* surfCoords)
{
    /*
    Description:
        Calculate a list of face center coordinates for the MDO coupling patches, given 
        the volume mesh point coordinates

    Input:
        volCoords: volume mesh point coordinates
    
    Output:
        surfCoords: face center coordinates for coupling patches
    */
    this->updateOFMesh(volCoords);

    wordList patchList;
    this->getCouplingPatchList(patchList);
    // get the total number of points and faces for the patchList
    label nPoints, nFaces;
    this->getPatchInfo(nPoints, nFaces, patchList);

    // ******** first loop
    label counterFaceI = 0;
    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            for (label i = 0; i < 3; i++)
            {
                surfCoords[counterFaceI] = meshPtr_->Cf().boundaryField()[patchI][faceI][i];
                counterFaceI++;
            }
        }
    }

    // ******** second loop
    // NOTE. Since we create two duplicated surface point coordinates for transferring two variables
    // we need to translate the 2nd one by 1000, so the meld component will find the correct
    // coordinates for interpolation. If these two sets of coords are overlapped, we will have
    // wrong interpolations from meld.
    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            for (label i = 0; i < 3; i++)
            {
                surfCoords[counterFaceI] = meshPtr_->Cf().boundaryField()[patchI][faceI][i] + 1000.0;
                counterFaceI++;
            }
        }
    }
}

void DASolver::calcCouplingFaceCoordsAD(
    const double* volCoords,
    const double* seeds,
    double* product)
{
#ifdef CODI_ADR

    label nCouplingFaces = this->getNCouplingFaces();

    scalar* volCoordsArray = new scalar[daIndexPtr_->nLocalXv];
    for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
    {
        volCoordsArray[i] = volCoords[i];
    }

    scalar* surfCoordsArray = new scalar[nCouplingFaces * 6];

    // update the OpenFOAM variables and reset their seeds (gradient part) to zeros
    this->resetOFSeeds();
    // reset the AD tape
    this->globalADTape_.reset();
    // start recording
    this->globalADTape_.setActive();

    // register inputs
    for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
    {
        this->globalADTape_.registerInput(volCoordsArray[i]);
    }

    // calculate outputs
    this->calcCouplingFaceCoords(volCoordsArray, surfCoordsArray);

    // register outputs
    for (label i = 0; i < nCouplingFaces * 6; i++)
    {
        this->globalADTape_.registerOutput(surfCoordsArray[i]);
    }

    // stop recording
    this->globalADTape_.setPassive();

    // set seeds to the outputs
    for (label i = 0; i < nCouplingFaces * 6; i++)
    {
        surfCoordsArray[i].setGradient(seeds[i]);
    }

    // now calculate the reverse matrix-vector product
    this->globalADTape_.evaluate();

    // get the matrix-vector product from the inputs
    for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
    {
        product[i] = volCoordsArray[i].getGradient();
    }

    // clean up AD
    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************
    for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
    {
        this->globalADTape_.deactivateValue(volCoordsArray[i]);
    }
    this->calcCouplingFaceCoords(volCoordsArray, surfCoordsArray);

    delete[] volCoordsArray;
    delete[] surfCoordsArray;

#endif
}

void DASolver::getCouplingPatchList(
    wordList& patchList,
    word groupName)
{
    /*
    Description:
        Return the couplingPatchList for MDO
        We loop over all the scenario in couplingInfo and find which scenario is active.
        If yes, we return the patchList defined in that scenario.
        If none of the scenario is active, we return the designSurface as the patchList
        NOTE: we always sort the returned patchList
    
    Input:
        groupName (optional): a specific group name one wants to extract the patchList from.
        The default is NONE and we will extract the first group. 

    Output:
        patchList: a word list of the coupling Patch for MDO or just the design surface patches
    */

    // first, we read couplingInfo
    dictionary couplingInfo = daOptionPtr_->getAllOptions().subDict("couplingInfo");

    // loop over all the keys to check active one
    forAll(couplingInfo.toc(), idxI)
    {
        word scenario = couplingInfo.toc()[idxI];
        label active = couplingInfo.subDict(scenario).getLabel("active");
        if (active)
        {
            dictionary couplingGroups = couplingInfo.subDict(scenario).subDict("couplingSurfaceGroups");
            // we support only one couplingSurfaceGroups
            if (groupName == "NONE")
            {
                groupName = couplingGroups.toc()[0];
                couplingGroups.readEntry<wordList>(groupName, patchList);
                sort(patchList);
                return;
            }
            else
            {
                couplingGroups.readEntry<wordList>(groupName, patchList);
                sort(patchList);
                return;
            }
        }
    }

    // if none of the scenarios are active, we return the design surface patches and print a warning
    Info << "************************  WARNING ************************" << endl;
    Info << "getCouplingPatchList is called but none of " << endl;
    Info << "scenario is active in couplingInfo dict! " << endl;
    Info << "return designSurfaces as patchList.." << endl;
    Info << "************************  WARNING ************************" << endl;

    daOptionPtr_->getAllOptions().readEntry<wordList>("designSurfaces", patchList);

    sort(patchList);

    return;
}

void DASolver::setThermal(scalar* thermal)
{
    /*
    Description:
        Assign the thermal BC values to all of the faces on the conjugate heat 
        transfer patches. 

        We assume the conjugate coupling patches are of mixed type, so we need to assign
        refValue = neighbour near wall temperature
        refGrad = 0
        valueFraction = neighKDeltaCoeffs / ( neighKDeltaCoeffs + myKDeltaCoeffs)

        NOTE: we have two separate variables saved in the thermal array. 
        One is the neighbour near wall temperature and the other is the neighbour kappa/d. 
        So the size of thermal array is 2 * nCouplingFaces
        NOTE: this function can be called by either fluid or solid domain!

        This conjugate heat transfer coupling uses the OpenFOAM's implementation in
        turbulentTemperatureCoupledBaffleMixed.C

    Inputs:
        thermal: the thermal BC values on the conjugate heat transfer patch
    
    Outputs:
        The T field in OpenFOAM
    */

    List<word> patchList;
    this->getCouplingPatchList(patchList);

    volScalarField& T =
        const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("T"));

    // ********* first loop, set the refValue
    label localFaceI = 0;
    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);

        mixedFvPatchField<scalar>& mixedPatch =
            refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

        forAll(mixedPatch.refValue(), faceI)
        {
            mixedPatch.refValue()[faceI] = thermal[localFaceI];
            mixedPatch.refGrad()[faceI] = 0;
            localFaceI++;
        }
    }

    // ********* second loop, set the valueFraction:
    // neighKDeltaCoeffs / ( neighKDeltaCoeffs + myKDeltaCoeffs)

#ifdef IncompressibleFlow
    // for incompressible flow  Q = Cp * alphaEff * dT/dz, so kappa = Cp * alphaEff
    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
    volScalarField alphaEff = daTurb.alphaEff();

    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            meshPtr_->time().constant(),
            meshPtr_(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    scalar Cp = readScalar(transportProperties.lookup("Cp"));

    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);

        mixedFvPatchField<scalar>& mixedPatch =
            refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // deltaCoeffs = 1 / d
            scalar deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
            scalar alphaEffBf = alphaEff.boundaryField()[patchI][faceI];
            scalar myKDeltaCoeffs = Cp * alphaEffBf * deltaCoeffs;
            scalar neighKDeltaCoeffs = thermal[localFaceI];
            mixedPatch.valueFraction()[faceI] = neighKDeltaCoeffs / (myKDeltaCoeffs + neighKDeltaCoeffs);
            localFaceI++;
        }
    }
#endif

#ifdef CompressibleFlow
    // for compressible flow Q = alphaEff * dHE/dz, so if enthalpy is used, kappa = Cp * alphaEff
    // if the internalEnergy is used, kappa = (Cp - R) * alphaEff

    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
    volScalarField alphaEff = daTurb.alphaEff();
    // compressible flow, H = alphaEff * dHE/dz
    const fluidThermo& thermo = meshPtr_->thisDb().lookupObject<fluidThermo>("thermophysicalProperties");
    const volScalarField& he = thermo.he();

    const IOdictionary& thermoDict = meshPtr_->thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
    dictionary mixSubDict = thermoDict.subDict("mixture");
    dictionary specieSubDict = mixSubDict.subDict("specie");
    scalar molWeight = specieSubDict.getScalar("molWeight");
    dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
    scalar Cp = thermodynamicsSubDict.getScalar("Cp");

    // 8314.4700665  gas constant in OpenFOAM
    // src/OpenFOAM/global/constants/thermodynamic/thermodynamicConstants.H
    scalar RR = Foam::constant::thermodynamic::RR;

    // R = RR/molWeight
    // Foam::specie::R() function in src/thermophysicalModels/specie/specie/specieI.H
    scalar R = RR / molWeight;

    scalar tmpVal = 0;
    // e = (Cp - R) * T, so Q = alphaEff * (Cp-R) * dT/dz
    if (he.name() == "e")
    {
        tmpVal = Cp - R;
    }
    // h = Cp * T, so Q = alphaEff * Cp * dT/dz
    else
    {
        tmpVal = Cp;
    }

    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);

        mixedFvPatchField<scalar>& mixedPatch =
            refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // deltaCoeffs = 1 / d
            scalar deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
            scalar alphaEffBf = alphaEff.boundaryField()[patchI][faceI];
            scalar myKDeltaCoeffs = tmpVal * alphaEffBf * deltaCoeffs;
            scalar neighKDeltaCoeffs = thermal[localFaceI];
            mixedPatch.valueFraction()[faceI] = neighKDeltaCoeffs / (myKDeltaCoeffs + neighKDeltaCoeffs);
            localFaceI++;
        }
    }
#endif

#ifdef SolidDASolver
    // for solid solvers Q = k * dT/dz, so kappa = k
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            meshPtr_->time().constant(),
            meshPtr_(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    scalar k = readScalar(transportProperties.lookup("k"));

    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);

        mixedFvPatchField<scalar>& mixedPatch =
            refCast<mixedFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // deltaCoeffs = 1 / d
            scalar deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
            scalar myKDeltaCoeffs = k * deltaCoeffs;
            scalar neighKDeltaCoeffs = thermal[localFaceI];
            mixedPatch.valueFraction()[faceI] = neighKDeltaCoeffs / (myKDeltaCoeffs + neighKDeltaCoeffs);
            localFaceI++;
        }
    }
#endif
}

void DASolver::getThermal(
    const scalar* volCoords,
    const scalar* states,
    scalar* thermal)
{
    /*
    Description:
        Compute the thermal variables for all of the faces on the conjugate heat 
        transfer patches.

        NOTE: we have two separate variables to assign to the thermal array. 
        One is the near wall temperature and the other is kappa/d. 
        So the size of thermal array is 2 * nCouplingFaces

        NOTE: this function can be called by either fluid or solid domain!

        This conjugate heat transfer coupling uses the OpenFOAM's implementation in
        turbulentTemperatureCoupledBaffleMixed.C

    Inputs:

        volCoords: volume coordinates

        states: state variables

    Output:
        thermal: the thermal variables on the conjugate heat transfer patch
    */

    this->updateOFMesh(volCoords);
    this->updateOFField(states);

    List<word> patchList;
    this->getCouplingPatchList(patchList);

    const objectRegistry& db = meshPtr_->thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");

    // ************ first loop, get the near wall cell temperature
    label localFaceI = 0;
    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            label faceCellI = meshPtr_->boundaryMesh()[patchI].faceCells()[faceI];
            thermal[localFaceI] = T[faceCellI];
            localFaceI++;
        }
    }

    // ********* second loop, get the (kappa / d) coefficient

#ifdef IncompressibleFlow

    // for incompressible flow  Q = Cp * alphaEff * dT/dz, so kappa = Cp * alphaEff

    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
    volScalarField alphaEff = daTurb.alphaEff();

    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            meshPtr_->time().constant(),
            meshPtr_(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    scalar Cp = readScalar(transportProperties.lookup("Cp"));

    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // deltaCoeffs = 1 / d
            scalar deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
            scalar alphaEffBf = alphaEff.boundaryField()[patchI][faceI];
            thermal[localFaceI] = Cp * alphaEffBf * deltaCoeffs;
            localFaceI++;
        }
    }
#endif

#ifdef CompressibleFlow
    // for compressible flow Q = alphaEff * dHE/dz, so if enthalpy is used, kappa = Cp * alphaEff
    // if the internalEnergy is used, kappa = (Cp - R) * alphaEff

    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
    volScalarField alphaEff = daTurb.alphaEff();
    // compressible flow, H = alphaEff * dHE/dz
    const fluidThermo& thermo = meshPtr_->thisDb().lookupObject<fluidThermo>("thermophysicalProperties");
    const volScalarField& he = thermo.he();

    const IOdictionary& thermoDict = meshPtr_->thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
    dictionary mixSubDict = thermoDict.subDict("mixture");
    dictionary specieSubDict = mixSubDict.subDict("specie");
    scalar molWeight = specieSubDict.getScalar("molWeight");
    dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
    scalar Cp = thermodynamicsSubDict.getScalar("Cp");

    // 8314.4700665  gas constant in OpenFOAM
    // src/OpenFOAM/global/constants/thermodynamic/thermodynamicConstants.H
    scalar RR = Foam::constant::thermodynamic::RR;

    // R = RR/molWeight
    // Foam::specie::R() function in src/thermophysicalModels/specie/specie/specieI.H
    scalar R = RR / molWeight;

    scalar tmpVal = 0;
    // e = (Cp - R) * T, so Q = alphaEff * (Cp-R) * dT/dz
    if (he.name() == "e")
    {
        tmpVal = Cp - R;
    }
    // h = Cp * T, so Q = alphaEff * Cp * dT/dz
    else
    {
        tmpVal = Cp;
    }

    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // deltaCoeffs = 1 / d
            scalar deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
            scalar alphaEffBf = alphaEff.boundaryField()[patchI][faceI];
            thermal[localFaceI] = tmpVal * alphaEffBf * deltaCoeffs;
            localFaceI++;
        }
    }
#endif

#ifdef SolidDASolver
    // for solid solvers Q = k * dT/dz, so kappa = k
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            meshPtr_->time().constant(),
            meshPtr_(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    scalar k = readScalar(transportProperties.lookup("k"));

    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // deltaCoeffs = 1 / d
            scalar deltaCoeffs = T.boundaryField()[patchI].patch().deltaCoeffs()[faceI];
            thermal[localFaceI] = k * deltaCoeffs;
            localFaceI++;
        }
    }
#endif
}

void DASolver::getThermalAD(
    const word inputName,
    const double* volCoords,
    const double* states,
    const double* seeds,
    double* product)
{
    /*
    Description:
        Calculate dThermaldStates or dThermaldXv using reverse-mode AD. Mode can be either temperature or heatFlux
    
    Input:
        inputName: either volCoords or states

        volCoords: the volume mesh coordinate 

        states: the state variable 

        seeds: the derivative seed 
    
    Output:
        product: [dTemperature/dW]^T * psi, [dTemperature/dXv]^T * psi, [dHeatFlux/dW]^T * psi, or [dHeatFlux/dXv]^T * psi
    */

#ifdef CODI_ADR

    label nCouplingFaces = this->getNCouplingFaces();

    scalar* volCoordsArray = new scalar[daIndexPtr_->nLocalXv];
    for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
    {
        volCoordsArray[i] = volCoords[i];
    }

    scalar* statesArray = new scalar[daIndexPtr_->nLocalAdjointStates];
    for (label i = 0; i < daIndexPtr_->nLocalAdjointStates; i++)
    {
        statesArray[i] = states[i];
    }

    scalar* thermalArray = new scalar[nCouplingFaces * 2];

    // update the OpenFOAM variables and reset their seeds (gradient part) to zeros
    this->resetOFSeeds();
    // reset the AD tape
    this->globalADTape_.reset();
    // start recording
    this->globalADTape_.setActive();

    // register inputs
    if (inputName == "volCoords")
    {
        for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
        {
            this->globalADTape_.registerInput(volCoordsArray[i]);
        }
    }
    else if (inputName == "states")
    {
        for (label i = 0; i < daIndexPtr_->nLocalAdjointStates; i++)
        {
            this->globalADTape_.registerInput(statesArray[i]);
        }
    }
    else
    {
        FatalErrorIn("getThermalAD") << " inputName not valid. "
                                     << abort(FatalError);
    }

    // calculate outputs
    this->getThermal(volCoordsArray, statesArray, thermalArray);

    // register outputs
    for (label i = 0; i < nCouplingFaces * 2; i++)
    {
        this->globalADTape_.registerOutput(thermalArray[i]);
    }

    // stop recording
    this->globalADTape_.setPassive();

    // set seeds to the outputs
    for (label i = 0; i < nCouplingFaces * 2; i++)
    {
        thermalArray[i].setGradient(seeds[i]);
    }

    // now calculate the reverse matrix-vector product
    this->globalADTape_.evaluate();

    // get the matrix-vector product from the inputs
    if (inputName == "volCoords")
    {
        for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
        {
            product[i] = volCoordsArray[i].getGradient();
        }
    }
    else if (inputName == "states")
    {
        for (label i = 0; i < daIndexPtr_->nLocalAdjointStates; i++)
        {
            product[i] = statesArray[i].getGradient();
        }
    }
    else
    {
        FatalErrorIn("getThermalAD") << " inputName not valid. "
                                     << abort(FatalError);
    }

    // clean up AD
    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************

    for (label i = 0; i < daIndexPtr_->nLocalXv; i++)
    {
        this->globalADTape_.deactivateValue(volCoordsArray[i]);
    }
    for (label i = 0; i < daIndexPtr_->nLocalAdjointStates; i++)
    {
        this->globalADTape_.deactivateValue(statesArray[i]);
    }
    this->getThermal(volCoordsArray, statesArray, thermalArray);

    delete[] volCoordsArray;
    delete[] statesArray;
    delete[] thermalArray;

#endif
}

void DASolver::getOFField(
    const word fieldName,
    const word fieldType,
    Vec field) const
{
    /*
    Description:
        assign a OpenFoam layer field variable in mesh.Db() to field
    */

    PetscScalar* vecArray;
    VecGetArray(field, &vecArray);

    if (fieldType == "scalar")
    {
        const volScalarField& field = meshPtr_->thisDb().lookupObject<volScalarField>(fieldName);
        forAll(field, cellI)
        {
            assignValueCheckAD(vecArray[cellI], field[cellI]);
        }
    }
    else if (fieldType == "vector")
    {
        const volVectorField& field = meshPtr_->thisDb().lookupObject<volVectorField>(fieldName);
        label localIdx = 0;
        forAll(field, cellI)
        {
            for (label comp = 0; comp < 3; comp++)
            {
                assignValueCheckAD(vecArray[localIdx], field[cellI][comp]);
                localIdx++;
            }
        }
    }
    else
    {
        FatalErrorIn("getField") << " fieldType not valid. Options: scalar or vector"
                                 << abort(FatalError);
    }

    VecRestoreArray(field, &vecArray);
}

void DASolver::getForces(Vec fX, Vec fY, Vec fZ)
{
    /*
    Description:
        Compute the nodal forces for all of the nodes on the fluid-structure-interaction
        patches. This routine is a wrapper that exposes the actual force computation
        routine to the Python layer using PETSc vectors. For the actual force computation
        routine view the getForcesInternal() function.

    Inputs:
        fX: Vector of X-component of forces

        fY: Vector of Y-component of forces

        fZ: Vector of Z-component of forces

    Output:
        fX, fY, fZ, and pointList are modified / set in place.
    */
#ifndef SolidDASolver
    // Get Data
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);

    // Allocate arrays
    List<scalar> fXTemp(nPoints);
    List<scalar> fYTemp(nPoints);
    List<scalar> fZTemp(nPoints);

    // Compute forces
    this->getForcesInternal(fXTemp, fYTemp, fZTemp, patchList);

    // Zero PETSc Arrays
    VecZeroEntries(fX);
    VecZeroEntries(fY);
    VecZeroEntries(fZ);

    // Get PETSc arrays
    PetscScalar* vecArrayFX;
    VecGetArray(fX, &vecArrayFX);
    PetscScalar* vecArrayFY;
    VecGetArray(fY, &vecArrayFY);
    PetscScalar* vecArrayFZ;
    VecGetArray(fZ, &vecArrayFZ);

    // Transfer to PETSc Array
    label pointCounter = 0;
    forAll(fXTemp, cI)
    {
        // Get Values
        PetscScalar val1, val2, val3;
        assignValueCheckAD(val1, fXTemp[pointCounter]);
        assignValueCheckAD(val2, fYTemp[pointCounter]);
        assignValueCheckAD(val3, fZTemp[pointCounter]);

        // Set Values
        vecArrayFX[pointCounter] = val1;
        vecArrayFY[pointCounter] = val2;
        vecArrayFZ[pointCounter] = val3;

        // Increment counter
        pointCounter += 1;
    }
    VecRestoreArray(fX, &vecArrayFX);
    VecRestoreArray(fY, &vecArrayFY);
    VecRestoreArray(fZ, &vecArrayFZ);
#endif
    return;
}

void DASolver::getAcousticData(Vec x, Vec y, Vec z, Vec nX, Vec nY, Vec nZ, Vec a, Vec fX, Vec fY, Vec fZ, word groupName)
{
    /*
    Description:
        Compute the nodal forces for all of the nodes on the fluid-structure-interaction
        patches. This routine is a wrapper that exposes the actual force computation
        routine to the Python layer using PETSc vectors. For the actual force computation
        routine view the getForcesInternal() function.

    Inputs:
        x: Vector of X-component coordinates

        y: Vector of Y-component coordinates

        z: Vector of Z-component coordinates

        nX: Vector of X-component of normal vectors

        nY: Vector of Y-component of normal vectors

        nZ: Vector of Z-component of normal vectors

        a: Vector of areas

        fX: Vector of X-component of forces

        fY: Vector of Y-component of forces

        fZ: Vector of Z-component of forces

        groupName: Name of acoustic group

    Output:
        x, y, z, nX, nY, nZ, a, fX, fY, fZ, and patchList are modified / set in place.
    */
#ifndef SolidDASolver
    // Get Data
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList, groupName);
    this->getPatchInfo(nPoints, nFaces, patchList);

    // Allocate arrays
    List<scalar> xTemp(nFaces);
    List<scalar> yTemp(nFaces);
    List<scalar> zTemp(nFaces);
    List<scalar> nXTemp(nFaces);
    List<scalar> nYTemp(nFaces);
    List<scalar> nZTemp(nFaces);
    List<scalar> aTemp(nFaces);
    List<scalar> fXTemp(nFaces);
    List<scalar> fYTemp(nFaces);
    List<scalar> fZTemp(nFaces);

    // Compute forces
    this->getAcousticDataInternal(xTemp, yTemp, zTemp, nXTemp, nYTemp, nZTemp, aTemp, fXTemp, fYTemp, fZTemp, patchList);

    // Zero PETSc Arrays
    VecZeroEntries(x);
    VecZeroEntries(y);
    VecZeroEntries(z);
    VecZeroEntries(nX);
    VecZeroEntries(nY);
    VecZeroEntries(nZ);
    VecZeroEntries(a);
    VecZeroEntries(fX);
    VecZeroEntries(fY);
    VecZeroEntries(fZ);

    // Get PETSc arrays
    PetscScalar* vecArrayX;
    VecGetArray(x, &vecArrayX);
    PetscScalar* vecArrayY;
    VecGetArray(y, &vecArrayY);
    PetscScalar* vecArrayZ;
    VecGetArray(z, &vecArrayZ);
    PetscScalar* vecArrayNX;
    VecGetArray(nX, &vecArrayNX);
    PetscScalar* vecArrayNY;
    VecGetArray(nY, &vecArrayNY);
    PetscScalar* vecArrayNZ;
    VecGetArray(nZ, &vecArrayNZ);
    PetscScalar* vecArrayA;
    VecGetArray(a, &vecArrayA);
    PetscScalar* vecArrayFX;
    VecGetArray(fX, &vecArrayFX);
    PetscScalar* vecArrayFY;
    VecGetArray(fY, &vecArrayFY);
    PetscScalar* vecArrayFZ;
    VecGetArray(fZ, &vecArrayFZ);

    // Transfer to PETSc Array
    label pointCounter = 0;
    forAll(xTemp, cI)
    {
        // Get Values
        PetscScalar val1, val2, val3, val4, val5, val6, val7, val8, val9, val10;
        assignValueCheckAD(val1, xTemp[pointCounter]);
        assignValueCheckAD(val2, yTemp[pointCounter]);
        assignValueCheckAD(val3, zTemp[pointCounter]);
        assignValueCheckAD(val4, nXTemp[pointCounter]);
        assignValueCheckAD(val5, nYTemp[pointCounter]);
        assignValueCheckAD(val6, nZTemp[pointCounter]);
        assignValueCheckAD(val7, aTemp[pointCounter]);
        assignValueCheckAD(val8, fXTemp[pointCounter]);
        assignValueCheckAD(val9, fYTemp[pointCounter]);
        assignValueCheckAD(val10, fZTemp[pointCounter]);

        // Set Values
        vecArrayX[pointCounter] = val1;
        vecArrayY[pointCounter] = val2;
        vecArrayZ[pointCounter] = val3;
        vecArrayNX[pointCounter] = val4;
        vecArrayNY[pointCounter] = val5;
        vecArrayNZ[pointCounter] = val6;
        vecArrayA[pointCounter] = val7;
        vecArrayFX[pointCounter] = val8;
        vecArrayFY[pointCounter] = val9;
        vecArrayFZ[pointCounter] = val10;

        // Increment counter
        pointCounter += 1;
    }
    VecRestoreArray(x, &vecArrayX);
    VecRestoreArray(y, &vecArrayY);
    VecRestoreArray(z, &vecArrayZ);
    VecRestoreArray(nX, &vecArrayNX);
    VecRestoreArray(nY, &vecArrayNY);
    VecRestoreArray(nZ, &vecArrayNZ);
    VecRestoreArray(a, &vecArrayA);
    VecRestoreArray(fX, &vecArrayFX);
    VecRestoreArray(fY, &vecArrayFY);
    VecRestoreArray(fZ, &vecArrayFZ);
#endif
    return;
}

void DASolver::getPatchInfo(
    label& nPoints,
    label& nFaces,
    List<word>& patchList)
{
    /*
    Description:
        Compute information needed to compute surface forces and other vars on walls.
        This includes total number of nodes and faces and a list of patches to
        include in the computation.

    Inputs:
        nPoints: Number of nodes included in the force computation

        nFaces: number of faces

        patchList: Patches included in the force computation

    Outputs:
        nPoints and patchList are modified / set in-place
    */
    // Generate patches, point mesh, and point boundary mesh
    const pointMesh& pMesh = pointMesh::New(meshPtr_());
    const pointBoundaryMesh& boundaryMesh = pMesh.boundary();

    // compute size of point and connectivity arrays
    nPoints = 0;
    nFaces = 0;
    forAll(patchList, cI)
    {
        // Get number of points in patch
        label patchIPoints = boundaryMesh.findPatchID(patchList[cI]);
        nPoints += boundaryMesh[patchIPoints].size();

        // get number of faces in patch
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        nFaces += meshPtr_->boundaryMesh()[patchI].size();
    }
    return;
}

void DASolver::getForcesInternal(
    List<scalar>& fX,
    List<scalar>& fY,
    List<scalar>& fZ,
    List<word>& patchList)
{
    /*
    Description:
        Wrapped version force computation routine to perform match to compute forces specified patches.

    Inputs:
        fX: Vector of X-component of forces

        fY: Vector of Y-component of forces

        fZ: Vector of Z-component of forces

    Output:
        fX, fY, fZ, and patchList are modified / set in place.
    */
#ifndef SolidDASolver
    // Get reference pressure
    dictionary couplingInfo = daOptionPtr_->getAllOptions().subDict("couplingInfo");
    scalar pRef = couplingInfo.subDict("aerostructural").getScalar("pRef");

    SortableList<word> patchListSort(patchList);

    // Initialize surface field for face-centered forces
    volVectorField volumeForceField(
        IOobject(
            "volumeForceField",
            meshPtr_->time().timeName(),
            meshPtr_(),
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        meshPtr_(),
        dimensionedVector("surfaceForce", dimensionSet(1, 1, -2, 0, 0, 0, 0), vector::zero),
        fixedValueFvPatchScalarField::typeName);

    // this code is pulled from:
    // src/functionObjects/forces/forces.C
    // modified slightly
    vector force(vector::zero);

    const objectRegistry& db = meshPtr_->thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    const surfaceVectorField::Boundary& Sfb = meshPtr_->Sf().boundaryField();

    const DATurbulenceModel& daTurb = daModelPtr_->getDATurbulenceModel();
    tmp<volSymmTensorField> tdevRhoReff = daTurb.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    const pointMesh& pMesh = pointMesh::New(meshPtr_());
    const pointBoundaryMesh& boundaryMesh = pMesh.boundary();

    // iterate over patches and extract boundary surface forces
    forAll(patchListSort, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchListSort[cI]);
        // create a shorter handle for the boundary patch
        const fvPatch& patch = meshPtr_->boundary()[patchI];
        // normal force
        vectorField fN(Sfb[patchI] * (p.boundaryField()[patchI] - pRef));
        // tangential force
        vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);
        // sum them up
        forAll(patch, faceI)
        {
            force.x() = fN[faceI].x() + fT[faceI].x();
            force.y() = fN[faceI].y() + fT[faceI].y();
            force.z() = fN[faceI].z() + fT[faceI].z();
            volumeForceField.boundaryFieldRef()[patchI][faceI] = force;
        }
    }
    volumeForceField.write();

    // The above volumeForceField is face-centered, we need to interpolate it to point-centered
    pointField meshPoints = meshPtr_->points();

    vector nodeForce(vector::zero);

    label patchStart = 0;
    forAll(patchListSort, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchListSort[cI]);
        label patchIPoints = boundaryMesh.findPatchID(patchListSort[cI]);

        label nPointsPatch = boundaryMesh[patchIPoints].size();
        List<scalar> fXTemp(nPointsPatch);
        List<scalar> fYTemp(nPointsPatch);
        List<scalar> fZTemp(nPointsPatch);
        List<label> pointListTemp(nPointsPatch);
        pointListTemp = -1;

        label pointCounter = 0;
        // Loop over Faces
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // Get number of points
            const label nPoints = meshPtr_->boundaryMesh()[patchI][faceI].size();

            // Divide force to nodes
            nodeForce = volumeForceField.boundaryFieldRef()[patchI][faceI] / double(nPoints);

            forAll(meshPtr_->boundaryMesh()[patchI][faceI], pointI)
            {
                // this is the index that corresponds to meshPoints, which contains both volume and surface points
                // so we can't directly reuse this index because we want to have only surface points
                label faceIPointIndexI = meshPtr_->boundaryMesh()[patchI][faceI][pointI];

                // Loop over pointListTemp array to check if this node is already included in this patch
                bool found = false;
                label iPoint = -1;
                for (label i = 0; i < pointCounter; i++)
                {
                    if (faceIPointIndexI == pointListTemp[i])
                    {
                        found = true;
                        iPoint = i;
                        break;
                    }
                }

                // If node is already included, add value to its entry
                if (found)
                {
                    // Add Force
                    fXTemp[iPoint] += nodeForce[0];
                    fYTemp[iPoint] += nodeForce[1];
                    fZTemp[iPoint] += nodeForce[2];
                }
                // If node is not already included, add it as the newest point and add global index mapping
                else
                {
                    // Add Force
                    fXTemp[pointCounter] = nodeForce[0];
                    fYTemp[pointCounter] = nodeForce[1];
                    fZTemp[pointCounter] = nodeForce[2];

                    // Add to Node Order Array
                    pointListTemp[pointCounter] = faceIPointIndexI;

                    // Increment counter
                    pointCounter += 1;
                }
            }
        }

        // Sort Patch Indices and Insert into Global Arrays
        SortableList<label> pointListSort(pointListTemp);
        forAll(pointListSort.indices(), indexI)
        {
            fX[patchStart + indexI] = fXTemp[pointListSort.indices()[indexI]];
            fY[patchStart + indexI] = fYTemp[pointListSort.indices()[indexI]];
            fZ[patchStart + indexI] = fZTemp[pointListSort.indices()[indexI]];
        }

        // Increment Patch Start Index
        patchStart += nPointsPatch;
    }
#endif
    return;
}

void DASolver::getAcousticDataInternal(
    List<scalar>& x,
    List<scalar>& y,
    List<scalar>& z,
    List<scalar>& nX,
    List<scalar>& nY,
    List<scalar>& nZ,
    List<scalar>& a,
    List<scalar>& fX,
    List<scalar>& fY,
    List<scalar>& fZ,
    List<word>& patchList)
{
    /*
    Description:
        Wrapped version force computation routine to perform match to compute forces specified patches.

    Inputs:
        x: Vector of X-component coordinates

        y: Vector of Y-component coordinates

        z: Vector of Z-component coordinates

        nX: Vector of X-component of normal vectors

        nY: Vector of Y-component of normal vectors

        nZ: Vector of Z-component of normal vectors

        a: Vector of areas

        fX: Vector of X-component of forces

        fY: Vector of Y-component of forces

        fZ: Vector of Z-component of forces

        patchList: Lift of patches to use for computation

    Output:
        x, y, z, nX, nY, nZ, a, fX, fY, fZ, and patchList are modified / set in place.
    */
#ifndef SolidDASolver
    // Get reference pressure
    scalar pRef;
    daOptionPtr_->getAllOptions().subDict("couplingInfo").subDict("aeroacoustic").readEntry<scalar>("pRef", pRef);

    SortableList<word> patchListSort(patchList);

    // ========================================================================
    // Compute Values
    // ========================================================================
    // this code is pulled from:
    // src/functionObjects/forcces/forces.C
    // modified slightly
    const objectRegistry& db = meshPtr_->thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    const surfaceVectorField::Boundary& Sfb = meshPtr_->Sf().boundaryField();

    const DATurbulenceModel& daTurb = daModelPtr_->getDATurbulenceModel();
    tmp<volSymmTensorField> tdevRhoReff = daTurb.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    // iterate over patches and extract boundary surface forces
    label iFace = 0;
    forAll(patchListSort, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchListSort[cI]);
        // create a shorter handle for the boundary patch
        const fvPatch& patch = meshPtr_->boundary()[patchI];

        // normal force
        vectorField fN(Sfb[patchI] * (p.boundaryField()[patchI] - pRef));
        // tangential force
        vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);

        // Store Values of Faces
        forAll(patch, faceI)
        {
            // Position
            x[iFace] = patch.Cf()[faceI].x();
            y[iFace] = patch.Cf()[faceI].y();
            z[iFace] = patch.Cf()[faceI].z();

            // Normal
            nX[iFace] = -patch.Sf()[faceI].x() / patch.magSf()[faceI];
            nY[iFace] = -patch.Sf()[faceI].y() / patch.magSf()[faceI];
            nZ[iFace] = -patch.Sf()[faceI].z() / patch.magSf()[faceI];

            // Area
            a[iFace] = patch.magSf()[faceI];

            // Force
            fX[iFace] = -(fN[faceI].x() + fT[faceI].x());
            fY[iFace] = -(fN[faceI].y() + fT[faceI].y());
            fZ[iFace] = -(fN[faceI].z() + fT[faceI].z());

            iFace++;
        }
    }
#endif
    return;
}

void DASolver::calcForceProfile(
    const word propName,
    Vec aForce,
    Vec tForce,
    Vec rDist,
    Vec integralForce)
{
    /*
    Description:
        Calculate the radial profile of forces on the propeller surface
        We need to call this function from the propeller component

    Input:
        State variables

    Output:
        xForce, the radial profile of force in the x direction
    */

    // Get Data
    // label nPoints = daOptionPtr_->getSubDictOption<scalar>("wingProp", "nForceSections");
    const dictionary& propSubDict = daOptionPtr_->getAllOptions().subDict("wingProp").subDict(propName);
    label nPoints = propSubDict.getLabel("nForceSections");

    // Allocate Arrays
    Field<scalar> aForceTemp(nPoints);
    Field<scalar> tForceTemp(nPoints);
    List<scalar> rDistTemp(nPoints);
    List<scalar> integralForceTemp(2);

    // Get PETSc Arrays

    // Set Values

    // Compute force profiles
    this->calcForceProfileInternal(propName, aForceTemp, tForceTemp, rDistTemp, integralForceTemp);

    VecZeroEntries(aForce);
    PetscScalar* vecArrayAForce;
    VecGetArray(aForce, &vecArrayAForce);
    VecZeroEntries(tForce);
    PetscScalar* vecArrayTForce;
    VecGetArray(tForce, &vecArrayTForce);
    VecZeroEntries(rDist);
    PetscScalar* vecArrayRDist;
    VecGetArray(rDist, &vecArrayRDist);
    VecZeroEntries(integralForce);
    PetscScalar* vecArrayIntegralForce;
    VecGetArray(integralForce, &vecArrayIntegralForce);

    // Tranfer to PETSc Array for force profiles and radius
    forAll(aForceTemp, cI)
    {
        // Get Values
        PetscScalar val1, val2, val3;
        assignValueCheckAD(val1, aForceTemp[cI]);
        assignValueCheckAD(val2, tForceTemp[cI]);
        assignValueCheckAD(val3, rDistTemp[cI]);

        // Set Values
        vecArrayAForce[cI] = val1;
        vecArrayTForce[cI] = val2;
        vecArrayRDist[cI] = val3;
    }
    PetscScalar val1, val2;
    assignValueCheckAD(val1, integralForceTemp[0]);
    assignValueCheckAD(val2, integralForceTemp[1]);
    vecArrayIntegralForce[0] = val1;
    vecArrayIntegralForce[1] = val2;

    VecRestoreArray(aForce, &vecArrayAForce);
    VecRestoreArray(tForce, &vecArrayTForce);
    VecRestoreArray(rDist, &vecArrayRDist);
    VecRestoreArray(integralForce, &vecArrayIntegralForce);

    return;
}

void DASolver::calcForceProfileInternal(
    const word propName,
    scalarList& aForce,
    scalarList& tForce,
    scalarList& rDist,
    scalarList& integralForce)
{
    /*
    Description:
        Same as calcForceProfile but for internal AD
    */

#ifndef SolidDASolver

    const dictionary& propSubDict = daOptionPtr_->getAllOptions().subDict("wingProp").subDict(propName);
    label sections = propSubDict.getLabel("nForceSections");
    word bladePatchName = propSubDict.getWord("bladeName");

    fvMesh& mesh = meshPtr_();

    scalarList axisDummy;
    scalarList rotationCenterDummy;
    propSubDict.readEntry<scalarList>("axis", axisDummy);
    propSubDict.readEntry<scalarList>("rotationCenter", rotationCenterDummy);
    vector axis;
    axis[0] = axisDummy[0];
    axis[1] = axisDummy[1];
    axis[2] = axisDummy[2];
    vector rotationCenter;
    rotationCenter[0] = rotationCenterDummy[0];
    rotationCenter[1] = rotationCenterDummy[1];
    rotationCenter[2] = rotationCenterDummy[2];

    // Ensure that axis is a unit vector
    axis = axis / sqrt(sqr(axis[0]) + sqr(axis[1]) + sqr(axis[2]));

    int quot;
    vector cellDir, projP, cellCen;
    scalar length, axialDist;

    // get the pressure in the memory
    const volScalarField& p = mesh.thisDb().lookupObject<volScalarField>("p");

    // find the patch ID of the blade surface
    label bladePatchI = mesh.boundaryMesh().findPatchID(bladePatchName);

    // radiiCell initialization, cell radii will be stored in it
    scalarList radiiCell(p.boundaryField()[bladePatchI].size());

    // meshTanDir initialization, mesh tangential direction will be stored in it
    vectorField meshTanDir = mesh.Cf().boundaryField()[bladePatchI];
    meshTanDir = meshTanDir * 0.;

    // radius limits initialization
    scalar minRadius = 1000000;
    scalar maxRadius = -1000000;

    forAll(p.boundaryField()[bladePatchI], faceI)
    {
        // directional vector from the propeller center to the cell center & dictance between them
        cellCen = mesh.Cf().boundaryField()[bladePatchI][faceI];
        cellDir = cellCen - rotationCenter;
        length = sqrt(sqr(cellDir[0]) + sqr(cellDir[1]) + sqr(cellDir[2]));

        // unit vector conversion
        cellDir = cellDir / length;

        // axial distance between the propeller center and the cell center
        axialDist = (cellDir & axis) * length;

        // projected point of the cell center on the axis
        projP = {rotationCenter[0] + axis[0] * axialDist, rotationCenter[1] + axis[1] * axialDist, rotationCenter[2] + axis[2] * axialDist};

        // radius of the cell center
        radiiCell[faceI] = sqrt(sqr(cellCen[0] - projP[0]) + sqr(cellCen[1] - projP[1]) + sqr(cellCen[2] - projP[2]));

        if (radiiCell[faceI] < minRadius)
        {
            minRadius = radiiCell[faceI];
            //minRadiIndx = faceI;
        }
        if (radiiCell[faceI] > maxRadius)
        {
            maxRadius = radiiCell[faceI];
            //maxRadiIndx = faceI;
        }

        // storing tangential vector as a unit vector
        meshTanDir[faceI] = axis ^ cellDir;
        length = sqrt(sqr(meshTanDir[faceI][0]) + sqr(meshTanDir[faceI][1]) + sqr(meshTanDir[faceI][2]));
        meshTanDir[faceI] = meshTanDir[faceI] / length;
    }

    reduce(maxRadius, maxOp<scalar>());
    reduce(minRadius, minOp<scalar>());

    // generating empty lists
    scalarList axialForce(sections);
    forAll(axialForce, Index)
    {
        axialForce[Index] = 0;
    }
    scalarList tangtForce = axialForce;
    scalarList radialDist = axialForce;
    scalarList intForce(2);
    intForce[0] = 0.0;
    intForce[1] = 0.0;

    // sectional radius computation
    scalar sectRad = (maxRadius - minRadius) / sections;
    for (int Index = 0; Index < sections; Index++)
    {
        radialDist[Index] = minRadius + sectRad * (Index + 0.5);
    }

    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
    tmp<volSymmTensorField> tdevRhoReff = daTurb.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    // computation of forces
    forAll(p.boundaryField()[bladePatchI], faceI)
    {
        // finding the section of the cell
        quot = floor((radiiCell[faceI] - minRadius) / sectRad);
        if (quot == sections)
        {
            quot = quot - 1;
        }

        // pressure direction is opposite of the surface normal
        axialForce[quot] = axialForce[quot] + (mesh.Sf().boundaryField()[bladePatchI][faceI] & axis) * p.boundaryField()[bladePatchI][faceI];
        tangtForce[quot] = tangtForce[quot] + (mesh.Sf().boundaryField()[bladePatchI][faceI] & meshTanDir[faceI]) * p.boundaryField()[bladePatchI][faceI];

        vector fT(mesh.Sf().boundaryField()[bladePatchI][faceI] & devRhoReffb[bladePatchI][faceI]);
        axialForce[quot] = axialForce[quot] + (fT & axis);
        tangtForce[quot] = tangtForce[quot] + (fT & meshTanDir[faceI]);
    }

    forAll(axialForce, index)
    {
        intForce[0] = intForce[0] + axialForce[index];
        intForce[1] = intForce[1] + tangtForce[index];
        axialForce[index] = axialForce[index] / sectRad;
        tangtForce[index] = tangtForce[index] / sectRad;
    }

    forAll(aForce, index)
    {
        reduce(axialForce[index], sumOp<scalar>());
        reduce(tangtForce[index], sumOp<scalar>());
    }
    reduce(intForce[0], sumOp<scalar>());
    reduce(intForce[1], sumOp<scalar>());

    aForce = axialForce;
    tForce = tangtForce;
    rDist = radialDist;
    integralForce = intForce;

    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "integral force " << integralForce << endl;
    }

#endif
}

void DASolver::calcdForceProfiledXvWAD(
    const word propName,
    const word outputMode,
    const word inputMode,
    const Vec xvVec,
    const Vec wVec,
    Vec psi,
    Vec dForcedXvW)
{
    /*
    Description:
        Calculate the matrix-vector product for [dForceProfile/dParamteres]^T * psi
    */

#ifdef CODI_ADR

    Info << "Calculating [dForceProfile/dInputs]^T*Psi using reverse-mode AD. PropName: "
         << propName << " inputMode: " << inputMode << " ouputMode: " << outputMode << endl;

    VecZeroEntries(dForcedXvW);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    const dictionary& propSubDict = daOptionPtr_->getAllOptions().subDict("wingProp").subDict(propName);
    label nPoints = propSubDict.getLabel("nForceSections");

    List<scalar> aForce(nPoints);
    List<scalar> tForce(nPoints);
    List<scalar> rDist(nPoints);
    List<scalar> integralForce(2);
    pointField meshPoints = meshPtr_->points();

    const PetscScalar* vecArrayPsi;

    // Step 1
    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    // Step 2
    if (inputMode == "mesh")
    {
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                this->globalADTape_.registerInput(meshPoints[i][j]);
            }
        }
        meshPtr_->movePoints(meshPoints);
        meshPtr_->moving(false);
    }
    else if (inputMode == "state")
    {
        this->registerStateVariableInput4AD();
    }
    else
    {
        FatalErrorIn("calcdFvSourcedInputsTPsiAD") << "inputMode not valid"
                                                   << abort(FatalError);
    }
    this->updateStateBoundaryConditions();

    // Step 3
    this->calcForceProfileInternal(propName, aForce, tForce, rDist, integralForce);

    // Step 4
    if (outputMode == "aForce")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.registerOutput(aForce[i]);
        }
    }
    else if (outputMode == "tForce")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.registerOutput(tForce[i]);
        }
    }
    else if (outputMode == "rDist")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.registerOutput(rDist[i]);
        }
    }
    else if (outputMode == "integralForce")
    {
        this->globalADTape_.registerOutput(integralForce[0]);
        this->globalADTape_.registerOutput(integralForce[1]);
    }
    else
    {
        FatalErrorIn("calcdFvSourcedInputsTPsiAD") << "outputMode not valid"
                                                   << abort(FatalError);
    }

    // Step 5
    this->globalADTape_.setPassive();

    // Step 6
    VecGetArrayRead(psi, &vecArrayPsi);
    if (outputMode == "aForce")
    {
        forAll(aForce, i)
        {
            aForce[i].setGradient(vecArrayPsi[i]);
        }
    }
    else if (outputMode == "tForce")
    {
        forAll(tForce, i)
        {
            tForce[i].setGradient(vecArrayPsi[i]);
        }
    }
    else if (outputMode == "rDist")
    {
        forAll(rDist, i)
        {
            rDist[i].setGradient(vecArrayPsi[i]);
        }
    }
    else if (outputMode == "integralForce")
    {
        integralForce[0].setGradient(vecArrayPsi[0]);
        integralForce[1].setGradient(vecArrayPsi[1]);
    }
    VecRestoreArrayRead(psi, &vecArrayPsi);

    // Step 7
    this->globalADTape_.evaluate();

    // Step 8
    if (inputMode == "mesh")
    {
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
                PetscScalar val = meshPoints[i][j].getGradient();
                VecSetValue(dForcedXvW, rowI, val, INSERT_VALUES);
            }
        }
        VecAssemblyBegin(dForcedXvW);
        VecAssemblyEnd(dForcedXvW);
    }
    else if (inputMode == "state")
    {
        this->assignStateGradient2Vec(dForcedXvW);
        VecAssemblyBegin(dForcedXvW);
        VecAssemblyEnd(dForcedXvW);
    }

    // Step 9
    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************

    if (inputMode == "mesh")
    {
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                this->globalADTape_.deactivateValue(meshPoints[i][j]);
            }
        }
        meshPtr_->movePoints(meshPoints);
        meshPtr_->moving(false);
    }
    else if (inputMode == "state")
    {
        this->deactivateStateVariableInput4AD();
    }
    this->updateStateBoundaryConditions();
    this->calcForceProfileInternal(propName, aForce, tForce, rDist, integralForce);

#endif
}

void DASolver::calcdForcedStateTPsiAD(
    const word mode,
    Vec xvVec,
    Vec stateVec,
    Vec psiVec,
    Vec prodVec)
{
}

void DASolver::calcFvSourceInternal(
    const word propName,
    const scalarField& aForce,
    const scalarField& tForce,
    const scalarField& rDist,
    const scalarList& targetForce,
    const vector& center,
    volVectorField& fvSource)
{
    /*
    Description:
        Smoothing the force distribution on propeller blade on the entire mesh to ensure that it will not diverge during optimization.
        Forces are smoothed using 4th degree polynomial distribution for inner radius, normal distribution for outer radius, and Gaussiam distribution for axial direction.
    Inputs:
        aForce: Axis force didtribution on propeller blade
        tForce: Tangential force distribution on propeller blade
        rDist: Force distribution locations and radii of propeller (first element is inner radius, last element is outer radius)
    Output:
        fvSource: Smoothed forces in each mesh cell
    */

    vector axis;
    const dictionary& propSubDict = daOptionPtr_->getAllOptions().subDict("wingProp").subDict(propName);
    scalar actEps = propSubDict.getScalar("actEps");
    word rotDir = propSubDict.getWord("rotDir");
    word interpScheme = propSubDict.getWord("interpScheme");
    scalarList axisDummy;
    propSubDict.readEntry<scalarList>("axis", axisDummy);
    axis[0] = axisDummy[0];
    axis[1] = axisDummy[1];
    axis[2] = axisDummy[2];

    if (interpScheme == "poly4Gauss")
    {
        // Fit a 4th order polynomial for the inner and Gaussian function for the outer

        scalar rotDirCon = 0.0;
        if (rotDir == "right")
        {
            rotDirCon = -1.0;
        }
        else if (rotDir == "left")
        {
            rotDirCon = 1.0;
        }
        else
        {
            FatalErrorIn("calcFvSourceInternal") << "Rotation direction must be either right of left"
                                                 << abort(FatalError);
        }

        // meshC is the cell center coordinates & meshV is the cell volume
        const volVectorField& meshC = fvSource.mesh().C();
        const scalarField& meshV = fvSource.mesh().V();

        // dummy vector field for storing the tangential vector of each cell
        volVectorField meshTanDir = meshC * 0;

        // Normalization of the blade radius distribution.
        scalar rOuter = (3 * rDist[rDist.size() - 1] - rDist[rDist.size() - 2]) / 2;
        scalarField rNorm = rDist; // normalized blade radius distribution
        forAll(rDist, index)
        {
            rNorm[index] = rDist[index] / rOuter;
        }

        // Inner and outer radius distribution limits
        scalar rStarMin = rNorm[0];
        scalar rStarMax = rNorm[rNorm.size() - 1];

        // Polynomial (inner) and Normal (outer) distribution  parameters' initialization
        scalar f1 = aForce[aForce.size() - 2];
        scalar f2 = aForce[aForce.size() - 1];
        scalar f3 = aForce[0];
        scalar f4 = aForce[1];
        scalar g1 = tForce[tForce.size() - 2];
        scalar g2 = tForce[tForce.size() - 1];
        scalar g3 = tForce[0];
        scalar g4 = tForce[1];
        scalar r1 = rNorm[rNorm.size() - 2];
        scalar r2 = rNorm[rNorm.size() - 1];
        scalar r3 = rNorm[0];
        scalar r4 = rNorm[1];
        scalar df3 = (f4 - f3) / (r4 - r3);
        scalar dg3 = (g4 - g3) / (r4 - r3);

        // Polynomial (inner) and Normal (outer) distribution  parameters' computation
        // Axial Outer
        scalar mu = 2 * r1 - r2;
        scalar maxI = 100;
        scalar sigmaS = 0;
        scalar i = 0;
        for (i = 0; i < maxI; i++)
        {
            sigmaS = ((r2 - mu) * (r2 - mu) - (r1 - mu) * (r1 - mu)) / (2 * log(f1 / f2));
            mu = r1 - sqrt(-2 * sigmaS * log(f1 * sqrt(2 * degToRad(180) * sigmaS)));
            if (mu > r1)
            {
                mu = 2 * r1 - mu;
            }
        }
        scalar sigmaAxialOut = sqrt(sigmaS);
        scalar muAxialOut = mu;

        // Tangential Outer
        mu = 2 * r1 - r2;
        for (i = 0; i < maxI; i++)
        {
            sigmaS = ((r2 - mu) * (r2 - mu) - (r1 - mu) * (r1 - mu)) / (2 * log(g1 / g2));
            mu = r1 - sqrt(-2 * sigmaS * log(g1 * sqrt(2 * degToRad(180) * sigmaS)));
            if (mu > r1)
            {
                mu = 2 * r1 - mu;
            }
        }
        scalar sigmaTangentialOut = sqrt(sigmaS);
        scalar muTangentialOut = mu;

        // Axial Inner
        scalar coefAAxialIn = (df3 * r3 - 2 * f3) / (2 * pow(r3, 4));
        scalar coefBAxialIn = (f3 - coefAAxialIn * pow(r3, 4)) / (r3 * r3);

        // Tangential Inner
        scalar coefATangentialIn = (dg3 * r3 - 2 * g3) / (2 * pow(r3, 4));
        scalar coefBTangentialIn = (g3 - coefATangentialIn * pow(r3, 4)) / (r3 * r3);

        // Cell 3D force computation loop
        forAll(meshC, cellI)
        {
            // Finding directional vector from mesh cell to the actuator center
            vector cellDir = meshC[cellI] - center;
            scalar length = sqrt(sqr(cellDir[0]) + sqr(cellDir[1]) + sqr(cellDir[2]));
            cellDir = cellDir / length;

            // Finding axial distance from mesh cell to the actuator center & projected point of mesh cell on the axis
            scalar meshDist = (axis & cellDir) * length;
            vector projP = {center[0] - axis[0] * meshDist, center[1] - axis[1] * meshDist, center[2] - axis[2] * meshDist};
            meshDist = mag(meshDist);

            // Finding the radius of the point
            scalar meshR = sqrt(sqr(meshC[cellI][0] - projP[0]) + sqr(meshC[cellI][1] - projP[1]) + sqr(meshC[cellI][2] - projP[2]));

            // Tangential component of the radius vector of the cell center
            vector cellAxDir = cellDir ^ axis;

            // Storing the tangential component
            meshTanDir[cellI] = cellAxDir;

            scalar rStar = meshR / rOuter;

            if (rStar < rStarMin)
            {
                fvSource[cellI] = ((coefAAxialIn * pow(rStar, 4) + coefBAxialIn * pow(rStar, 2)) * axis + (coefATangentialIn * pow(rStar, 4) + coefBTangentialIn * pow(rStar, 2)) * cellAxDir * rotDirCon) * exp(-sqr(meshDist / actEps));
            }
            else if (rStar > rStarMax)
            {
                fvSource[cellI] = (1 / (sigmaAxialOut * sqrt(2 * degToRad(180)))) * exp(-0.5 * sqr((rStar - muAxialOut) / sigmaAxialOut)) * axis;
                fvSource[cellI] = fvSource[cellI] + (1 / (sigmaTangentialOut * sqrt(2 * degToRad(180)))) * exp(-0.5 * sqr((rStar - muTangentialOut) / sigmaTangentialOut)) * cellAxDir * rotDirCon;
                fvSource[cellI] = fvSource[cellI] * exp(-sqr(meshDist / actEps));
            }
            else
            {
                fvSource[cellI] = (interpolateSplineXY(rStar, rNorm, aForce) * axis + interpolateSplineXY(rStar, rNorm, tForce) * cellAxDir * rotDirCon) * exp(-sqr(meshDist / actEps));
            }
        }

        // Scale factor computation loop
        scalar scaleAxial = 0;
        scalar scaleTangential = 0;
        forAll(meshV, cellI)
        {
            scaleAxial = scaleAxial + (fvSource[cellI] & axis) * meshV[cellI];
            scaleTangential = scaleTangential + (fvSource[cellI] & meshTanDir[cellI]) * meshV[cellI];
        }
        reduce(scaleAxial, sumOp<scalar>());
        reduce(scaleTangential, sumOp<scalar>());
        scaleAxial = targetForce[0] / scaleAxial;
        scaleTangential = targetForce[1] / scaleTangential * rotDirCon;

        // Cell 3D force scaling loop
        forAll(meshV, cellI)
        {
            fvSource[cellI][0] = fvSource[cellI][0] * mag(axis[0]) * scaleAxial + fvSource[cellI][0] * mag(meshTanDir[cellI][0]) * scaleTangential;
            fvSource[cellI][1] = fvSource[cellI][1] * mag(axis[1]) * scaleAxial + fvSource[cellI][1] * mag(meshTanDir[cellI][1]) * scaleTangential;
            fvSource[cellI][2] = fvSource[cellI][2] * mag(axis[2]) * scaleAxial + fvSource[cellI][2] * mag(meshTanDir[cellI][2]) * scaleTangential;
        }
    }
    else if (interpScheme == "gauss")
    {
        fvMesh& mesh = meshPtr_();

        scalar rInner = rDist[0];
        scalar rOuter = rDist[rDist.size() - 1];
        scalar fAxialInner = aForce[0];
        scalar fAxialOuter = aForce[aForce.size() - 1];
        scalar fTanInner = tForce[0];
        scalar fTanOuter = tForce[tForce.size() - 1];

        vector dirNorm = {axis[0], axis[1], axis[2]};
        dirNorm /= mag(axis);

        // first loop, we calculate the integral force and then compute the scaling factor
        scalar axialForceSum = 0.0;
        scalar tangentialForceSum = 0.0;
        forAll(mesh.cells(), cellI)
        {
            // the cell center coordinates of this cellI
            vector cellC = mesh.C()[cellI];
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

            scalar fAxial = 0.0;
            scalar fTan = 0.0;

            scalar dA2_Eps2 = (cellC2AVecALen * cellC2AVecALen) / actEps / actEps;

            // we need to smooth the force in the radial and axial directions if r is outside of [rInner:rOuter]
            // if r is inside, we just interpolate the prescribed aForce and tForce and smooth the
            // force in the axial direction only
            if (cellC2AVecRLen < rInner)
            {
                scalar dR2_Eps2 = (cellC2AVecRLen - rInner) * (cellC2AVecRLen - rInner) / actEps / actEps;
                fAxial = fAxialInner * exp(-dR2_Eps2) * exp(-dA2_Eps2);
                fTan = fTanInner * exp(-dR2_Eps2) * exp(-dA2_Eps2);
            }
            else if (cellC2AVecRLen >= rInner && cellC2AVecRLen <= rOuter)
            {
                fAxial = interpolateSplineXY(cellC2AVecRLen, rDist, aForce) * exp(-dA2_Eps2);
                fTan = interpolateSplineXY(cellC2AVecRLen, rDist, tForce) * exp(-dA2_Eps2);
            }
            else
            {
                scalar dR2_Eps2 = (cellC2AVecRLen - rOuter) * (cellC2AVecRLen - rOuter) / actEps / actEps;
                fAxial = fAxialOuter * exp(-dR2_Eps2) * exp(-dA2_Eps2);
                fTan = fTanOuter * exp(-dR2_Eps2) * exp(-dA2_Eps2);
            }

            axialForceSum += fAxial * mesh.V()[cellI];
            tangentialForceSum += fTan * mesh.V()[cellI];
        }
        reduce(axialForceSum, sumOp<scalar>());
        reduce(tangentialForceSum, sumOp<scalar>());

        scalar aForceScale = targetForce[0] / axialForceSum;
        scalar tForceScale = targetForce[1] / tangentialForceSum;

        // loop again with the correct scale so that the integral forces matches the prescribed ones
        axialForceSum = 0;
        tangentialForceSum = 0;
        forAll(mesh.cells(), cellI)
        {
            // the cell center coordinates of this cellI
            vector cellC = mesh.C()[cellI];
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

            scalar fAxial = 0.0;
            scalar fTan = 0.0;

            scalar dA2_Eps2 = (cellC2AVecALen * cellC2AVecALen) / actEps / actEps;

            // we need to smooth the force in the radial and axial directions if r is outside of [rInner:rOuter]
            // if r is inside, we just interpolate the prescribed aForce and tForce and smooth the
            // force in the axial direction only
            if (cellC2AVecRLen < rInner)
            {
                scalar dR2_Eps2 = (cellC2AVecRLen - rInner) * (cellC2AVecRLen - rInner) / actEps / actEps;
                fAxial = aForceScale * fAxialInner * exp(-dR2_Eps2) * exp(-dA2_Eps2);
                fTan = tForceScale * fTanInner * exp(-dR2_Eps2) * exp(-dA2_Eps2);
            }
            else if (cellC2AVecRLen >= rInner && cellC2AVecRLen <= rOuter)
            {
                fAxial = aForceScale * interpolateSplineXY(cellC2AVecRLen, rDist, aForce) * exp(-dA2_Eps2);
                fTan = tForceScale * interpolateSplineXY(cellC2AVecRLen, rDist, tForce) * exp(-dA2_Eps2);
            }
            else
            {
                scalar dR2_Eps2 = (cellC2AVecRLen - rOuter) * (cellC2AVecRLen - rOuter) / actEps / actEps;
                fAxial = aForceScale * fAxialOuter * exp(-dR2_Eps2) * exp(-dA2_Eps2);
                fTan = tForceScale * fTanOuter * exp(-dR2_Eps2) * exp(-dA2_Eps2);
            }

            vector sourceVec = (fAxial * dirNorm + fTan * cellC2AVecCNorm);

            axialForceSum += fAxial * mesh.V()[cellI];
            tangentialForceSum += fTan * mesh.V()[cellI];

            fvSource[cellI] += sourceVec;
        }

        reduce(axialForceSum, sumOp<scalar>());
        reduce(tangentialForceSum, sumOp<scalar>());

        if (daOptionPtr_->getOption<word>("runStatus") == "solvePrimal")
        {
            Info << "Integrated Axial Force for " << propName << ": " << axialForceSum << endl;
            Info << "Integrated Tangential Force for " << propName << ": " << tangentialForceSum << endl;
        }
    }
    else
    {
        FatalErrorIn("") << "interpScheme not valid! Options: poly4Gauss or gauss"
                         << abort(FatalError);
    }
}

void DASolver::calcFvSource(
    const word propName,
    Vec aForce,
    Vec tForce,
    Vec rDist,
    Vec targetForce,
    Vec center,
    Vec xvVec,
    Vec fvSource)
{
    /*
    Description:
        Calculate the fvSource based on the radial force profile and the propeller parameters
        We need to call this function from the wing component

    Input:
        parameters: propeller parameters, i.e., center_x, center_y, center_z, r_inner, r_outer

        force: the radial force profiles (fx1, fy1, fz1, fx2, fy2, fz2, ... )

    Output:
        fvSource: a volVectorField variable that will be added to the momentum eqn
    */

    // Get Data
    const dictionary& propSubDict = daOptionPtr_->getAllOptions().subDict("wingProp").subDict(propName);
    label nPoints = propSubDict.getLabel("nForceSections");
    // label meshSize = meshPtr_->nCells();

    // Allocate Arrays
    Field<scalar> aForceTemp(nPoints);
    Field<scalar> tForceTemp(nPoints);
    Field<scalar> rDistTemp(nPoints);
    List<scalar> targetForceTemp(2);
    Vector<scalar> centerTemp;
    volVectorField fvSourceTemp(
        IOobject(
            "fvSourceTemp",
            meshPtr_->time().timeName(),
            meshPtr_(),
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        meshPtr_(),
        dimensionedVector("surfaceForce", dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::zero),
        fixedValueFvPatchScalarField::typeName);

    // Get PETSc Arrays
    PetscScalar* vecArrayAForce;
    VecGetArray(aForce, &vecArrayAForce);
    PetscScalar* vecArrayTForce;
    VecGetArray(tForce, &vecArrayTForce);
    PetscScalar* vecArrayRDist;
    VecGetArray(rDist, &vecArrayRDist);
    PetscScalar* vecArrayTargetForce;
    VecGetArray(targetForce, &vecArrayTargetForce);
    PetscScalar* vecArrayCenter;
    VecGetArray(center, &vecArrayCenter);

    // Set Values
    forAll(aForceTemp, cI)
    {
        aForceTemp[cI] = vecArrayAForce[cI];
        tForceTemp[cI] = vecArrayTForce[cI];
        rDistTemp[cI] = vecArrayRDist[cI];
    }
    targetForceTemp[0] = vecArrayTargetForce[0];
    targetForceTemp[1] = vecArrayTargetForce[1];
    centerTemp[0] = vecArrayCenter[0];
    centerTemp[1] = vecArrayCenter[1];
    centerTemp[2] = vecArrayCenter[2];

    this->updateOFMesh(xvVec);

    // Compute fvSource
    this->calcFvSourceInternal(propName, aForceTemp, tForceTemp, rDistTemp, targetForceTemp, centerTemp, fvSourceTemp);

    VecZeroEntries(fvSource);
    PetscScalar* vecArrayFvSource;
    VecGetArray(fvSource, &vecArrayFvSource);

    // Tranfer to PETSc Array for fvSource
    forAll(fvSourceTemp, cI)
    {
        // Get Values
        PetscScalar val1, val2, val3;
        assignValueCheckAD(val1, fvSourceTemp[cI][0]);
        assignValueCheckAD(val2, fvSourceTemp[cI][1]);
        assignValueCheckAD(val3, fvSourceTemp[cI][2]);

        // Set Values
        vecArrayFvSource[3 * cI] = val1;
        vecArrayFvSource[3 * cI + 1] = val2;
        vecArrayFvSource[3 * cI + 2] = val3;
    }

    VecRestoreArray(aForce, &vecArrayAForce);
    VecRestoreArray(tForce, &vecArrayTForce);
    VecRestoreArray(rDist, &vecArrayRDist);
    VecRestoreArray(targetForce, &vecArrayTargetForce);
    VecRestoreArray(center, &vecArrayCenter);
    VecRestoreArray(fvSource, &vecArrayFvSource);

    return;
}

void DASolver::calcdFvSourcedInputsTPsiAD(
    const word propName,
    const word mode,
    Vec aForce,
    Vec tForce,
    Vec rDist,
    Vec targetForce,
    Vec center,
    Vec xvVec,
    Vec psi,
    Vec dFvSource)
{
    /*
    Description:
        Calculate the matrix-vector product for either [dFvSource/dParameters]^T * psi, or [dFvSource/dForce]^T * psi
    */

#ifdef CODI_ADR

    Info << "Calculating [dFvSource/dInputs]^T*Psi using reverse-mode AD. PropName: "
         << propName << " mode: " << mode << endl;

    VecZeroEntries(dFvSource);

    this->updateOFMesh(xvVec);

    const dictionary& propSubDict = daOptionPtr_->getAllOptions().subDict("wingProp").subDict(propName);
    label nPoints = propSubDict.getLabel("nForceSections");

    Field<scalar> aForceField(nPoints);
    Field<scalar> tForceField(nPoints);
    Field<scalar> rDistField(nPoints);
    List<scalar> targetForceList(2);
    vector centerVector = vector::zero;
    pointField meshPoints = meshPtr_->points();

    volVectorField fvSourceVField(
        IOobject(
            "fvSourceVField",
            meshPtr_->time().timeName(),
            meshPtr_(),
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        meshPtr_(),
        dimensionedVector("surfaceForce", dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::zero),
        fixedValueFvPatchScalarField::typeName);

    PetscScalar* vecArrayAForce;
    PetscScalar* vecArrayTForce;
    PetscScalar* vecArrayRDist;
    PetscScalar* vecArrayTargetForce;
    PetscScalar* vecArrayCenter;
    const PetscScalar* vecArrayPsi;

    VecGetArray(aForce, &vecArrayAForce);
    for (label i = 0; i < nPoints; i++)
    {
        aForceField[i] = vecArrayAForce[i];
    }
    VecRestoreArray(aForce, &vecArrayAForce);

    VecGetArray(tForce, &vecArrayTForce);
    for (label i = 0; i < nPoints; i++)
    {
        tForceField[i] = vecArrayTForce[i];
    }
    VecRestoreArray(tForce, &vecArrayTForce);

    VecGetArray(rDist, &vecArrayRDist);
    for (label i = 0; i < nPoints; i++)
    {
        rDistField[i] = vecArrayRDist[i];
    }
    VecRestoreArray(rDist, &vecArrayRDist);

    VecGetArray(targetForce, &vecArrayTargetForce);
    targetForceList[0] = vecArrayTargetForce[0];
    targetForceList[1] = vecArrayTargetForce[1];
    VecRestoreArray(targetForce, &vecArrayTargetForce);

    VecGetArray(center, &vecArrayCenter);
    centerVector[0] = vecArrayCenter[0];
    centerVector[1] = vecArrayCenter[1];
    centerVector[2] = vecArrayCenter[2];
    VecRestoreArray(center, &vecArrayCenter);

    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    if (mode == "aForce")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.registerInput(aForceField[i]);
        }
    }
    else if (mode == "tForce")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.registerInput(tForceField[i]);
        }
    }
    else if (mode == "rDist")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.registerInput(rDistField[i]);
        }
    }
    else if (mode == "targetForce")
    {
        for (label i = 0; i < 2; i++)
        {
            this->globalADTape_.registerInput(targetForceList[i]);
        }
    }
    else if (mode == "center")
    {
        for (label i = 0; i < 3; i++)
        {
            this->globalADTape_.registerInput(centerVector[i]);
        }
    }
    else if (mode == "mesh")
    {
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                this->globalADTape_.registerInput(meshPoints[i][j]);
            }
        }
        meshPtr_->movePoints(meshPoints);
        meshPtr_->moving(false);
    }
    else
    {
        FatalErrorIn("calcdFvSourcedInputsTPsiAD") << "mode not valid"
                                                   << abort(FatalError);
    }

    // Step 3
    this->calcFvSourceInternal(propName, aForceField, tForceField, rDistField, targetForceList, centerVector, fvSourceVField);

    // Step 4
    forAll(fvSourceVField, i)
    {
        this->globalADTape_.registerOutput(fvSourceVField[i][0]);
        this->globalADTape_.registerOutput(fvSourceVField[i][1]);
        this->globalADTape_.registerOutput(fvSourceVField[i][2]);
    }

    this->globalADTape_.setPassive();

    // Step 6
    VecGetArrayRead(psi, &vecArrayPsi);
    forAll(fvSourceVField, i)
    {
        // Set seeds
        fvSourceVField[i][0].setGradient(vecArrayPsi[i * 3]);
        fvSourceVField[i][1].setGradient(vecArrayPsi[i * 3 + 1]);
        fvSourceVField[i][2].setGradient(vecArrayPsi[i * 3 + 2]);
    }
    VecRestoreArrayRead(psi, &vecArrayPsi);

    this->globalADTape_.evaluate();

    PetscScalar* vecArrayProd;
    VecGetArray(dFvSource, &vecArrayProd);
    if (mode == "aForce")
    {
        forAll(aForceField, i)
        {
            vecArrayProd[i] = aForceField[i].getGradient();
        }
    }
    else if (mode == "tForce")
    {
        forAll(tForceField, i)
        {
            vecArrayProd[i] = tForceField[i].getGradient();
        }
    }
    else if (mode == "rDist")
    {
        forAll(rDistField, i)
        {
            vecArrayProd[i] = rDistField[i].getGradient();
        }
    }
    else if (mode == "targetForce")
    {
        forAll(targetForceList, i)
        {
            vecArrayProd[i] = targetForceList[i].getGradient();
        }
    }
    else if (mode == "center")
    {
        forAll(centerVector, i)
        {
            vecArrayProd[i] = centerVector[i].getGradient();
        }
    }
    else if (mode == "mesh")
    {
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
                PetscScalar val = meshPoints[i][j].getGradient();
                VecSetValue(dFvSource, rowI, val, INSERT_VALUES);
            }
        }
        VecAssemblyBegin(dFvSource);
        VecAssemblyEnd(dFvSource);
    }

    VecRestoreArray(dFvSource, &vecArrayProd);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************

    if (mode == "aForce")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.deactivateValue(aForceField[i]);
        }
    }
    else if (mode == "tForce")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.deactivateValue(tForceField[i]);
        }
    }
    else if (mode == "rDist")
    {
        for (label i = 0; i < nPoints; i++)
        {
            this->globalADTape_.deactivateValue(rDistField[i]);
        }
    }
    else if (mode == "targetForce")
    {
        for (label i = 0; i < 2; i++)
        {
            this->globalADTape_.deactivateValue(targetForceList[i]);
        }
    }
    else if (mode == "center")
    {
        for (label i = 0; i < 3; i++)
        {
            this->globalADTape_.deactivateValue(centerVector[i]);
        }
    }
    else if (mode == "mesh")
    {
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                this->globalADTape_.deactivateValue(meshPoints[i][j]);
            }
        }
        meshPtr_->movePoints(meshPoints);
        meshPtr_->moving(false);
    }

    this->calcFvSourceInternal(propName, aForceField, tForceField, rDistField, targetForceList, centerVector, fvSourceVField);
#endif
}

void DASolver::reduceStateResConLevel(
    const dictionary& maxResConLv4JacPCMat,
    HashTable<List<List<word>>>& stateResConInfo) const
{
    /*
    Description:
        Reduce the connectivity levels for stateResConInfo
        based on maxResConLv4JacPCMat specified in DAOption

    Input:
        maxResConLv4JacPCMat: the maximal levels of connectivity for each
    state variable residual

    Output:
        stateResConInfo: reduced connectivity level.

    Example:

        If the original stateResConInfo reads:
    
        stateResConInfo
        {
            "URes":
            {
                {"U", "p", "phi"}, // level 0
                {"U", "p"},        // level 1
                {"U"}              // level 2
            }
        }
        And maxResConLv4JacPCMat in DAOption reads:
    
        maxResConLv4JacPCMat
        {
            "URes": 1
        }
        
        Then, calling reduceStateResConLevel will give:
    
        stateResConInfo
        {
            "URes":
            {
                {"U", "p", "phi"}, // level 0
                {"U", "p"},        // level 1
            }
        }
    
        Note that the level 2 of the connectivity in URes is removed becasue
        "URes"=1 in maxResConLv4JacPCMat

    */

    // if no maxResConLv4JacPCMat is specified, just return;
    if (maxResConLv4JacPCMat.toc().size() == 0)
    {
        Info << "maxResConLv4JacPCMat is empty, just return" << endl;
        return;
    }

    // now check if maxResConLv4JacPCMat has all the maxRes level defined
    // and these max levels are <= stateResConInfo.size()
    forAll(stateResConInfo.toc(), idxJ)
    {
        word key1 = stateResConInfo.toc()[idxJ];
        bool keyFound = false;
        forAll(maxResConLv4JacPCMat.toc(), idxI)
        {
            word key = maxResConLv4JacPCMat.toc()[idxI];
            if (key == key1)
            {
                keyFound = true;
                label maxLv = maxResConLv4JacPCMat.getLabel(key);
                label maxLv1 = stateResConInfo[key1].size() - 1;
                if (maxLv > maxLv1)
                {
                    FatalErrorIn("") << "maxResConLv4JacPCMat maxLevel"
                                     << maxLv << " for " << key
                                     << " larger than stateResConInfo maxLevel "
                                     << maxLv1 << " for " << key1
                                     << abort(FatalError);
                }
            }
        }
        if (!keyFound)
        {
            FatalErrorIn("") << key1 << " not found in maxResConLv4JacPCMat"
                             << abort(FatalError);
        }
    }

    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Reducing max connectivity level of Jacobian PC Mat to : ";
        Info << maxResConLv4JacPCMat << endl;
    }

    // assign stateResConInfo to stateResConInfoBK
    HashTable<List<List<word>>> stateResConInfoBK;
    forAll(stateResConInfo.toc(), idxI)
    {
        word key = stateResConInfo.toc()[idxI];
        stateResConInfoBK.set(key, stateResConInfo[key]);
    }

    // now we can erase stateResConInfo
    stateResConInfo.clearStorage();

    // get the reduced stateResConInfo
    forAll(stateResConInfoBK.toc(), idxI)
    {
        word key = stateResConInfoBK.toc()[idxI];
        label maxConLevel = maxResConLv4JacPCMat.getLabel(key);
        label conSize = stateResConInfoBK[key].size();
        if (conSize > maxConLevel + 1)
        {
            List<List<word>> conList;
            conList.setSize(maxConLevel + 1);
            for (label i = 0; i <= maxConLevel; i++) // NOTE: it is <=
            {
                conList[i] = stateResConInfoBK[key][i];
            }
            stateResConInfo.set(key, conList);
        }
        else
        {
            stateResConInfo.set(key, stateResConInfoBK[key]);
        }
    }
    //Info<<stateResConInfo<<endl;
}

/// run the coloring solver
void DASolver::runColoring()
{
    /*
    Description:
        Run the coloring for dRdW and save them as dRdWColoring_n.bin where n is the number
        of processors
    */

    DAJacCon daJacCon("dRdW", meshPtr_(), daOptionPtr_(), daModelPtr_(), daIndexPtr_());

    if (!daJacCon.coloringExists())
    {
        dictionary options;
        const HashTable<List<List<word>>>& stateResConInfo = daStateInfoPtr_->getStateResConInfo();
        options.set("stateResConInfo", stateResConInfo);

        // need to first setup preallocation vectors for the dRdWCon matrix
        // because directly initializing the dRdWCon matrix will use too much memory
        daJacCon.setupJacConPreallocation(options);

        // now we can initilaize dRdWCon
        daJacCon.initializeJacCon(options);

        // setup dRdWCon
        daJacCon.setupJacCon(options);
        Info << "dRdWCon Created. " << meshPtr_->time().elapsedClockTime() << " s" << endl;

        // compute the coloring
        Info << "Calculating dRdW Coloring... " << meshPtr_->time().elapsedClockTime() << " s" << endl;
        daJacCon.calcJacConColoring();
        Info << "Calculating dRdW Coloring... Completed! " << meshPtr_->time().elapsedClockTime() << " s" << endl;

        // clean up
        daJacCon.clear();
    }
}

void DASolver::calcPrimalResidualStatistics(
    const word mode,
    const label writeRes)
{
    /*
    Description:
        Calculate the mean, max, and norm2 for all residuals and print it to screen
    */

    if (mode == "print")
    {
        // print the primal residuals to screen
        Info << "Printing Primal Residual Statistics." << endl;
    }
    else if (mode == "calc")
    {
        // we will just calculate but not printting anything
    }
    else
    {
        FatalErrorIn("") << "mode not valid" << abort(FatalError);
    }

    this->calcResiduals();

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word resName = stateName + "Res";
        const volVectorField& stateRes = meshPtr_->thisDb().lookupObject<volVectorField>(resName);

        vector vecResMax(0, 0, 0);
        vector vecResNorm2(0, 0, 0);
        vector vecResMean(0, 0, 0);
        forAll(stateRes, cellI)
        {
            vecResNorm2.x() += pow(stateRes[cellI].x(), 2.0);
            vecResNorm2.y() += pow(stateRes[cellI].y(), 2.0);
            vecResNorm2.z() += pow(stateRes[cellI].z(), 2.0);
            vecResMean.x() += fabs(stateRes[cellI].x());
            vecResMean.y() += fabs(stateRes[cellI].y());
            vecResMean.z() += fabs(stateRes[cellI].z());
            if (fabs(stateRes[cellI].x()) > vecResMax.x())
            {
                vecResMax.x() = fabs(stateRes[cellI].x());
            }
            if (fabs(stateRes[cellI].y()) > vecResMax.y())
            {
                vecResMax.y() = fabs(stateRes[cellI].y());
            }
            if (fabs(stateRes[cellI].z()) > vecResMax.z())
            {
                vecResMax.z() = fabs(stateRes[cellI].z());
            }
        }
        vecResMean = vecResMean / stateRes.size();
        reduce(vecResMean, sumOp<vector>());
        vecResMean = vecResMean / Pstream::nProcs();
        reduce(vecResNorm2, sumOp<vector>());
        reduce(vecResMax, maxOp<vector>());
        vecResNorm2.x() = pow(vecResNorm2.x(), 0.5);
        vecResNorm2.y() = pow(vecResNorm2.y(), 0.5);
        vecResNorm2.z() = pow(vecResNorm2.z(), 0.5);
        if (mode == "print")
        {
            Info << stateName << " Residual Norm2: " << vecResNorm2 << endl;
            Info << stateName << " Residual Mean: " << vecResMean << endl;
            Info << stateName << " Residual Max: " << vecResMax << endl;
        }

        if (writeRes)
        {
            stateRes.write();
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word resName = stateName + "Res";
        const volScalarField& stateRes = meshPtr_->thisDb().lookupObject<volScalarField>(resName);

        scalar scalarResMax = 0, scalarResNorm2 = 0, scalarResMean = 0;
        forAll(stateRes, cellI)
        {
            scalarResNorm2 += pow(stateRes[cellI], 2.0);
            scalarResMean += fabs(stateRes[cellI]);
            if (fabs(stateRes[cellI]) > scalarResMax)
                scalarResMax = fabs(stateRes[cellI]);
        }
        scalarResMean = scalarResMean / stateRes.size();
        reduce(scalarResMean, sumOp<scalar>());
        scalarResMean = scalarResMean / Pstream::nProcs();
        reduce(scalarResNorm2, sumOp<scalar>());
        reduce(scalarResMax, maxOp<scalar>());
        scalarResNorm2 = pow(scalarResNorm2, 0.5);
        if (mode == "print")
        {
            Info << stateName << " Residual Norm2: " << scalarResNorm2 << endl;
            Info << stateName << " Residual Mean: " << scalarResMean << endl;
            Info << stateName << " Residual Max: " << scalarResMax << endl;
        }

        if (writeRes)
        {
            stateRes.write();
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word resName = stateName + "Res";
        const volScalarField& stateRes = meshPtr_->thisDb().lookupObject<volScalarField>(resName);

        scalar scalarResMax = 0, scalarResNorm2 = 0, scalarResMean = 0;
        forAll(stateRes, cellI)
        {
            scalarResNorm2 += pow(stateRes[cellI], 2.0);
            scalarResMean += fabs(stateRes[cellI]);
            if (fabs(stateRes[cellI]) > scalarResMax)
                scalarResMax = fabs(stateRes[cellI]);
        }
        scalarResMean = scalarResMean / stateRes.size();
        reduce(scalarResMean, sumOp<scalar>());
        scalarResMean = scalarResMean / Pstream::nProcs();
        reduce(scalarResNorm2, sumOp<scalar>());
        reduce(scalarResMax, maxOp<scalar>());
        scalarResNorm2 = pow(scalarResNorm2, 0.5);
        if (mode == "print")
        {
            Info << stateName << " Residual Norm2: " << scalarResNorm2 << endl;
            Info << stateName << " Residual Mean: " << scalarResMean << endl;
            Info << stateName << " Residual Max: " << scalarResMax << endl;
        }

        if (writeRes)
        {
            stateRes.write();
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word resName = stateName + "Res";
        const surfaceScalarField& stateRes = meshPtr_->thisDb().lookupObject<surfaceScalarField>(resName);

        scalar phiResMax = 0, phiResNorm2 = 0, phiResMean = 0;
        forAll(stateRes, faceI)
        {
            phiResNorm2 += pow(stateRes[faceI], 2.0);
            phiResMean += fabs(stateRes[faceI]);
            if (fabs(stateRes[faceI]) > phiResMax)
                phiResMax = fabs(stateRes[faceI]);
        }
        forAll(stateRes.boundaryField(), patchI)
        {
            forAll(stateRes.boundaryField()[patchI], faceI)
            {
                scalar bPhiRes = stateRes.boundaryField()[patchI][faceI];
                phiResNorm2 += pow(bPhiRes, 2.0);
                phiResMean += fabs(bPhiRes);
                if (fabs(bPhiRes) > phiResMax)
                    phiResMax = fabs(bPhiRes);
            }
        }
        phiResMean = phiResMean / meshPtr_->nFaces();
        reduce(phiResMean, sumOp<scalar>());
        phiResMean = phiResMean / Pstream::nProcs();
        reduce(phiResNorm2, sumOp<scalar>());
        reduce(phiResMax, maxOp<scalar>());
        phiResNorm2 = pow(phiResNorm2, 0.5);
        if (mode == "print")
        {
            Info << stateName << " Residual Norm2: " << phiResNorm2 << endl;
            Info << stateName << " Residual Mean: " << phiResMean << endl;
            Info << stateName << " Residual Max: " << phiResMax << endl;
        }

        if (writeRes)
        {
            stateRes.write();
        }
    }

    return;
}

void DASolver::calcdRdWT(
    const label isPC,
    Mat dRdWT)
{
    /*
    Description:
        This function computes partials derivatives dRdWT or dRdWTPC.
        PC means preconditioner matrix
    
    Input:

        isPC: isPC=1 computes dRdWTPC, isPC=0 computes dRdWT
    
    Output:
        dRdWT: the partial derivative matrix [dR/dW]^T
        NOTE: You need to call MatCreate for the dRdWT matrix before calling this function.
        No need to call MatSetSize etc because they will be done in this function
    */

    // create the state and volCoord vecs from the OF fields
    Vec wVec, xvVec;
    VecCreate(PETSC_COMM_WORLD, &wVec);
    VecSetSizes(wVec, daIndexPtr_->nLocalAdjointStates, PETSC_DECIDE);
    VecSetFromOptions(wVec);

    label nXvs = daIndexPtr_->nLocalPoints * 3;
    VecCreate(PETSC_COMM_WORLD, &xvVec);
    VecSetSizes(xvVec, nXvs, PETSC_DECIDE);
    VecSetFromOptions(xvVec);

    daFieldPtr_->ofField2StateVec(wVec);
    daFieldPtr_->ofMesh2PointVec(xvVec);

    word matName;
    if (isPC == 0)
    {
        matName = "dRdWT";
    }
    else if (isPC == 1)
    {
        matName = "dRdWTPC";
    }
    else
    {
        FatalErrorIn("") << "isPC " << isPC << " not supported! "
                         << "Options are: 0 (for dRdWT) and 1 (for dRdWTPC)." << abort(FatalError);
    }

    if (daOptionPtr_->getOption<label>("debug"))
    {
        this->calcPrimalResidualStatistics("print");
    }

    Info << "Computing " << matName << " " << runTimePtr_->elapsedClockTime() << " s" << endl;
    Info << "Initializing dRdWCon. " << runTimePtr_->elapsedClockTime() << " s" << endl;

    // initialize DAJacCon object
    word modelType = "dRdW";
    DAJacCon daJacCon(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_());

    dictionary options;
    const HashTable<List<List<word>>>& stateResConInfo = daStateInfoPtr_->getStateResConInfo();

    if (isPC == 1)
    {
        // need to reduce the JacCon for PC to reduce memory usage
        HashTable<List<List<word>>> stateResConInfoReduced = stateResConInfo;

        dictionary maxResConLv4JacPCMat = daOptionPtr_->getAllOptions().subDict("maxResConLv4JacPCMat");

        this->reduceStateResConLevel(maxResConLv4JacPCMat, stateResConInfoReduced);
        options.set("stateResConInfo", stateResConInfoReduced);
    }
    else
    {
        options.set("stateResConInfo", stateResConInfo);
    }

    // need to first setup preallocation vectors for the dRdWCon matrix
    // because directly initializing the dRdWCon matrix will use too much memory
    daJacCon.setupJacConPreallocation(options);

    // now we can initialize dRdWCon
    daJacCon.initializeJacCon(options);

    // setup dRdWCon
    daJacCon.setupJacCon(options);
    Info << "dRdWCon Created. " << runTimePtr_->elapsedClockTime() << " s" << endl;

    // read the coloring
    daJacCon.readJacConColoring();

    // initialize partDeriv object
    DAPartDeriv daPartDeriv(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_(),
        daJacCon,
        daResidualPtr_());

    // we want transposed dRdW
    dictionary options1;
    options1.set("transposed", 1);
    options1.set("isPC", isPC);
    // we can set lower bounds for the Jacobians to save memory
    if (isPC == 1)
    {
        options1.set("lowerBound", daOptionPtr_->getSubDictOption<scalar>("jacLowerBounds", "dRdWPC"));
    }
    else
    {
        options1.set("lowerBound", daOptionPtr_->getSubDictOption<scalar>("jacLowerBounds", "dRdW"));
    }

    // initialize dRdWT matrix
    daPartDeriv.initializePartDerivMat(options1, dRdWT);

    // calculate dRdWT
    daPartDeriv.calcPartDerivMat(options1, xvVec, wVec, dRdWT);

    if (daOptionPtr_->getOption<label>("debug"))
    {
        this->calcPrimalResidualStatistics("print");
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdWT") || writeJacobians.found("all"))
    {
        DAUtility::writeMatrixBinary(dRdWT, matName);
    }

    // clear up
    daJacCon.clear();
}

void DASolver::setBCToOFVars(
    const dictionary& dvSubDict,
    const scalar& BC)
{
    /*
    Description:
        Assign the boundary condition (BC) value to OF variables
    
    Input:

        dvSubDict: the design variable subDict that contains the inform for setting BCs

        BC: the BC value
    */

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the variable for changing the boundary condition
    word varName = dvSubDict.getWord("variable");
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the component of a vector variable, ignore when it is a scalar
    label comp = dvSubDict.getLabel("comp");

    // set the BC
    forAll(patches, idxI)
    {
        word patchName = patches[idxI];
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);
        if (meshPtr_->thisDb().foundObject<volVectorField>(varName))
        {
            volVectorField& state(const_cast<volVectorField&>(
                meshPtr_->thisDb().lookupObject<volVectorField>(varName)));
            // for decomposed domain, don't set BC if the patch is empty
            if (meshPtr_->boundaryMesh()[patchI].size() > 0)
            {
                if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                {
                    forAll(state.boundaryFieldRef()[patchI], faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI][comp] = BC;
                    }
                }
                else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                         || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                {
                    mixedFvPatchField<vector>& inletOutletPatch =
                        refCast<mixedFvPatchField<vector>>(state.boundaryFieldRef()[patchI]);
                    vector val = inletOutletPatch.refValue()[0];
                    val[comp] = BC;
                    inletOutletPatch.refValue() = val;
                }
            }
        }
        else if (meshPtr_->thisDb().foundObject<volScalarField>(varName))
        {
            volScalarField& state(const_cast<volScalarField&>(
                meshPtr_->thisDb().lookupObject<volScalarField>(varName)));
            // for decomposed domain, don't set BC if the patch is empty
            if (meshPtr_->boundaryMesh()[patchI].size() > 0)
            {
                if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                {
                    forAll(state.boundaryFieldRef()[patchI], faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI] = BC;
                    }
                }
                else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                         || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                {
                    mixedFvPatchField<scalar>& inletOutletPatch =
                        refCast<mixedFvPatchField<scalar>>(state.boundaryFieldRef()[patchI]);
                    inletOutletPatch.refValue() = BC;
                }
            }
        }
    }
}

void DASolver::getBCFromOFVars(
    const dictionary& dvSubDict,
    scalar& BC)
{
    /*
    Description:
        Get the boundary condition (BC) value from OF variables
    
    Input:

        dvSubDict: the design variable subDict that contains the inform for setting BCs

    Output:
        BC: the BC value
    */

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the variable for changing the boundary condition
    word varName = dvSubDict.getWord("variable");
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the component of a vector variable, ignore when it is a scalar
    label comp = dvSubDict.getLabel("comp");

    // Now get the BC value
    forAll(patches, idxI)
    {
        word patchName = patches[idxI];
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);
        if (meshPtr_->thisDb().foundObject<volVectorField>(varName))
        {
            volVectorField& state(const_cast<volVectorField&>(
                meshPtr_->thisDb().lookupObject<volVectorField>(varName)));
            // for decomposed domain, don't set BC if the patch is empty
            if (meshPtr_->boundaryMesh()[patchI].size() > 0)
            {
                if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                {
                    BC = state.boundaryFieldRef()[patchI][0][comp];
                }
                else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                         || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                {
                    mixedFvPatchField<vector>& inletOutletPatch =
                        refCast<mixedFvPatchField<vector>>(state.boundaryFieldRef()[patchI]);
                    BC = inletOutletPatch.refValue()[0][comp];
                }
            }
        }
        else if (meshPtr_->thisDb().foundObject<volScalarField>(varName))
        {
            volScalarField& state(const_cast<volScalarField&>(
                meshPtr_->thisDb().lookupObject<volScalarField>(varName)));
            // for decomposed domain, don't set BC if the patch is empty
            if (meshPtr_->boundaryMesh()[patchI].size() > 0)
            {
                if (state.boundaryFieldRef()[patchI].type() == "fixedValue")
                {
                    BC = state.boundaryFieldRef()[patchI][0];
                }
                else if (state.boundaryFieldRef()[patchI].type() == "inletOutlet"
                         || state.boundaryFieldRef()[patchI].type() == "outletInlet")
                {
                    mixedFvPatchField<scalar>& inletOutletPatch =
                        refCast<mixedFvPatchField<scalar>>(state.boundaryFieldRef()[patchI]);
                    BC = inletOutletPatch.refValue()[0];
                }
            }
        }
    }
}

void DASolver::createMLRKSP(
    const Mat jacMat,
    const Mat jacPCMat,
    KSP ksp)
{
    /*
    Description:
        Call createMLRKSP from DALinearEqn
        This is the main function we need to call to initialize the KSP and set
        up parameters for solving the linear equations
    */

    daLinearEqnPtr_->createMLRKSP(jacMat, jacPCMat, ksp);
}

void DASolver::updateKSPPCMat(
    Mat PCMat,
    KSP ksp)
{
    /*
    Description:
        Update the preconditioner matrix for the ksp object
    */
    KSPSetOperators(ksp, dRdWTMF_, PCMat);
}

void DASolver::createMLRKSPMatrixFree(
    const Mat jacPCMat,
    KSP ksp)
{
#ifdef CODI_ADR
    /*
    Description:
        Call createMLRKSP from DALinearEqn
        This is the main function we need to call to initialize the KSP and set
        up parameters for solving the linear equations
        NOTE: this is the matrix-free version of the createMLRKSP function.
        We dont need to input the jacMat because we will use dRdWTMF_: the
        matrix-free state Jacobian matrix
    */

    daLinearEqnPtr_->createMLRKSP(dRdWTMF_, jacPCMat, ksp);
#endif
}

label DASolver::solveLinearEqn(
    const KSP ksp,
    const Vec rhsVec,
    Vec solVec)
{
    /*
    Description:
        Call solveLinearEqn from DALinearEqn to solve a linear equation.
    
    Input:
        ksp: the KSP object, obtained from calling Foam::createMLRKSP

        rhsVec: the right-hand-side petsc vector

    Output:
        solVec: the solution vector

        Return 0 if the linear equation solution finished successfully otherwise return 1
    */

    label error = daLinearEqnPtr_->solveLinearEqn(ksp, rhsVec, solVec);

    // need to reset globalADTapeInitialized to 0 because every matrix-free
    // adjoint solution need to re-initialize the AD tape
    globalADTape4dRdWTInitialized = 0;

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************
    this->deactivateStateVariableInput4AD();
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    return error;
}

void DASolver::resetOFSeeds()
{
    /*
    Description:
        RESET the seeds to all state variables and vol coordinates to zeros
        This is done by passing a double array to the OpenFOAM's scalar field
        and setting all the gradient part to zero.
        In CODIPack, if we pass a double value to a scalar value, it will assign
        an zero to the scalar variable's gradient part (seeds).
        NOTE. this is important because CODIPack's tape.reset does not clean
        the seeds in the OpenFOAM's variables. So we need to manually reset them
        Not doing this will cause inaccurate AD values.
    
    Outputs:
        
        The OpenFOAM variables's seed values
    */

    this->setPrimalBoundaryConditions(0);
    daFieldPtr_->resetOFSeeds();

    this->updateStateBoundaryConditions();
}

void DASolver::updateOFField(const Vec wVec)
{
    /*
    Description:
        Update the OpenFOAM field values (including both internal
        and boundary fields) based on the state vector wVec

    Input:
        wVec: state variable vector

    Output:
        OpenFoam flow fields (internal and boundary)
    */

    label printInfo = 0;
    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Updating the OpenFOAM field..." << endl;
        printInfo = 1;
    }
    this->setPrimalBoundaryConditions(printInfo);
    daFieldPtr_->stateVec2OFField(wVec);

    // if we have regression models, we also need to update them because they will update the fields
    this->regressionModelCompute();

    // We need to call correctBC multiple times to reproduce
    // the exact residual, this is needed for some boundary conditions
    // and intermediate variables (e.g., U for inletOutlet, nut with wall functions)
    label maxCorrectBCCalls = daOptionPtr_->getOption<label>("maxCorrectBCCalls");
    for (label i = 0; i < maxCorrectBCCalls; i++)
    {
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
    }
}

void DASolver::updateOFField(const scalar* states)
{
    label printInfo = 0;
    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Updating the OpenFOAM field..." << endl;
        printInfo = 1;
    }
    this->setPrimalBoundaryConditions(printInfo);
    daFieldPtr_->state2OFField(states);

    // if we have regression models, we also need to update them because they will update the fields
    this->regressionModelCompute();

    // We need to call correctBC multiple times to reproduce
    // the exact residual, this is needed for some boundary conditions
    // and intermediate variables (e.g., U for inletOutlet, nut with wall functions)
    label maxCorrectBCCalls = daOptionPtr_->getOption<label>("maxCorrectBCCalls");
    for (label i = 0; i < maxCorrectBCCalls; i++)
    {
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
    }
}

void DASolver::updateOFMesh(const Vec xvVec)
{
    /*
    Description:
        Update the OpenFOAM mesh based on the point vector xvVec

    Input:
        xvVec: point coordinate vector

    Output:
        OpenFoam flow fields (internal and boundary)
    */
    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Updating the OpenFOAM mesh..." << endl;
    }
    daFieldPtr_->pointVec2OFMesh(xvVec);
}

void DASolver::updateOFMesh(const scalar* volCoords)
{
    /*
    Description:
        Update the OpenFOAM mesh based on the volume coordinates point volCoords

    Input:
        volCoords: point coordinate array

    Output:
        OpenFoam flow fields (internal and boundary)
    */
    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Updating the OpenFOAM mesh..." << endl;
    }
    daFieldPtr_->point2OFMesh(volCoords);
}

void DASolver::initializedRdWTMatrixFree()
{
#ifdef CODI_ADR
    /*
    Description:
        This function initialize the matrix-free dRdWT, which will be
        used later in the adjoint solution
    */

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    //this->updateOFField(wVec);
    //this->updateOFMesh(xvVec);

    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "In initializedRdWTMatrixFree" << endl;
        this->calcPrimalResidualStatistics("print");
    }

    // No need to set the size, instead, we need to provide a function to compute
    // matrix-vector product, i.e., the dRdWTMatVecMultFunction function
    label localSize = daIndexPtr_->nLocalAdjointStates;
    MatCreateShell(PETSC_COMM_WORLD, localSize, localSize, PETSC_DETERMINE, PETSC_DETERMINE, this, &dRdWTMF_);
    MatShellSetOperation(dRdWTMF_, MATOP_MULT, (void (*)(void))dRdWTMatVecMultFunction);
    MatSetUp(dRdWTMF_);
    Info << "dRdWT Jacobian Free created!" << endl;

#endif
}

void DASolver::destroydRdWTMatrixFree()
{
#ifdef CODI_ADR
    /*
    Description:
        Destroy dRdWTMF_
    */
    MatDestroy(&dRdWTMF_);
#endif
}

PetscErrorCode DASolver::dRdWTMatVecMultFunction(Mat dRdWTMF, Vec vecX, Vec vecY)
{
#ifdef CODI_ADR
    /*
    Description:
        This function implements a way to compute matrix-vector products
        associated with dRdWTMF matrix. 
        Here we need to return vecY = dRdWTMF * vecX.
        We use the reverse-mode AD to compute vecY in a matrix-free manner
    */
    DASolver* ctx;
    MatShellGetContext(dRdWTMF, (void**)&ctx);

    // Need to re-initialize the tape, setup inputs and outputs,
    // and run the forward computation and save the intermediate
    // variables in the tape, such that we don't re-compute them
    // for each GMRES iteration. This initialization needs to
    // happen for each adjoint solution. We will reset
    // globalADTape4dRdWTInitialized = 0 in DASolver::solveLinearEqn function
    if (!ctx->globalADTape4dRdWTInitialized)
    {
        ctx->initializeGlobalADTape4dRdWT();
        ctx->globalADTape4dRdWTInitialized = 1;
    }

    // assign the variable in vecX as the residual gradient for reverse AD
    ctx->assignVec2ResidualGradient(vecX);
    // do the backward computation to propagate the derivatives to the states
    ctx->globalADTape_.evaluate();
    // assign the derivatives stored in the states to the vecY vector
    ctx->assignStateGradient2Vec(vecY);
    // NOTE: we need to normalize the vecY vector.
    ctx->normalizeGradientVec(vecY);
    // clear the adjoint to prepare the next matrix-free GMRES iteration
    ctx->globalADTape_.clearAdjoints();

#endif

    return 0;
}

void DASolver::initializeGlobalADTape4dRdWT()
{
#ifdef CODI_ADR
    /*
    Description:
        Initialize the global tape for computing dRdWT*psi
        using revere-mode AD. Here we need to register inputs
        and outputs, compute the residuals, and record all the
        intermediate variables in the tape. Then in the 
        dRdWTMatVecMultFunction function, we can assign gradients
        and call tape.evaluate multiple times 
    */

    // always reset the tape before recording
    this->globalADTape_.reset();
    // set the tape to active and start recording intermediate variables
    this->globalADTape_.setActive();
    // register state variables as the inputs
    this->registerStateVariableInput4AD();
    // need to correct BC and update all intermediate variables
    this->updateStateBoundaryConditions();
    // Now we can compute the residuals
    this->calcResiduals();
    // Set the residual as the output
    this->registerResidualOutput4AD();
    // All done, set the tape to passive
    this->globalADTape_.setPassive();

    // Now the tape is ready to use in the matrix-free GMRES solution
#endif
}

void DASolver::normalizeJacTVecProduct(
    const word inputName,
    double* product)
{

#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Normalize the jacobian vector product that has states as the input such as dFdW and dRdW
    
    Input/Output:

        inputName: 
        name of the input for the Jacobian, we normalize the product only if inputName=stateVar

        product: 
        jacobian vector product to be normalized. vecY = vecY * scalingFactor
        the scalingFactor depends on states.
        This is needed for the matrix-vector products in matrix-free adjoint

    */

    if (inputName == "stateVar")
    {

        dictionary normStateDict = daOptionPtr_->getAllOptions().subDict("normalizeStates");

        forAll(stateInfo_["volVectorStates"], idxI)
        {
            const word stateName = stateInfo_["volVectorStates"][idxI];
            // if normalized state not defined, skip
            if (normStateDict.found(stateName))
            {
                scalar scalingFactor = normStateDict.getScalar(stateName);

                forAll(meshPtr_->cells(), cellI)
                {
                    for (label i = 0; i < 3; i++)
                    {
                        label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                        product[localIdx] *= scalingFactor.getValue();
                    }
                }
            }
        }

        forAll(stateInfo_["volScalarStates"], idxI)
        {
            const word stateName = stateInfo_["volScalarStates"][idxI];
            // if normalized state not defined, skip
            if (normStateDict.found(stateName))
            {
                scalar scalingFactor = normStateDict.getScalar(stateName);

                forAll(meshPtr_->cells(), cellI)
                {
                    label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
                    product[localIdx] *= scalingFactor.getValue();
                }
            }
        }

        forAll(stateInfo_["modelStates"], idxI)
        {
            const word stateName = stateInfo_["modelStates"][idxI];
            // if normalized state not defined, skip
            if (normStateDict.found(stateName))
            {

                scalar scalingFactor = normStateDict.getScalar(stateName);

                forAll(meshPtr_->cells(), cellI)
                {
                    label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
                    product[localIdx] *= scalingFactor.getValue();
                }
            }
        }

        forAll(stateInfo_["surfaceScalarStates"], idxI)
        {
            const word stateName = stateInfo_["surfaceScalarStates"][idxI];
            // if normalized state not defined, skip
            if (normStateDict.found(stateName))
            {
                scalar scalingFactor = normStateDict.getScalar(stateName);

                forAll(meshPtr_->faces(), faceI)
                {
                    label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

                    if (faceI < daIndexPtr_->nLocalInternalFaces)
                    {
                        scalar meshSf = meshPtr_->magSf()[faceI];
                        product[localIdx] *= scalingFactor.getValue() * meshSf.getValue();
                    }
                    else
                    {
                        label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                        label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                        label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                        scalar meshSf = meshPtr_->magSf().boundaryField()[patchIdx][faceIdx];
                        product[localIdx] *= scalingFactor.getValue() * meshSf.getValue();
                    }
                }
            }
        }
    }

#endif
}

void DASolver::setSolverInput(
    const word inputName,
    const word inputType,
    const int inputSize,
    const double* input,
    const double* seed)
{
    /*
    Description:
        Set seeds for forward mode AD using the DAInput class
    */

    // initialize the input and output objects
    autoPtr<DAInput> daInput(
        DAInput::New(
            inputName,
            inputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_()));

    scalarList inputList(inputSize, 0.0);

    // assign the input array to the input list and the seed to its gradient()
    // Note: we need to use scalarList for AD
    forAll(inputList, idxI)
    {
        inputList[idxI] = input[idxI];
#ifdef CODI_ADF
        inputList[idxI].gradient() = seed[idxI];
#endif
    }

    // call daInput->run to assign inputList to OF variables
    daInput->run(inputList);
}

label DASolver::getInputSize(
    const word inputName,
    const word inputType)
{
    autoPtr<DAInput> daInput(
        DAInput::New(
            inputName,
            inputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_()));

    return daInput->size();
}

label DASolver::getOutputSize(
    const word outputName,
    const word outputType)
{
    autoPtr<DAOutput> daOutput(
        DAOutput::New(
            outputName,
            outputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            daFunctionPtrList_));

    return daOutput->size();
}

label DASolver::getInputDistributed(
    const word inputName,
    const word inputType)
{
    autoPtr<DAInput> daInput(
        DAInput::New(
            inputName,
            inputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_()));

    return daInput->distributed();
}

label DASolver::getOutputDistributed(
    const word outputName,
    const word outputType)
{
    autoPtr<DAOutput> daOutput(
        DAOutput::New(
            outputName,
            outputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            daFunctionPtrList_));

    return daOutput->distributed();
}

void DASolver::calcJacTVecProduct(
    const word inputName,
    const word inputType,
    const int inputSize,
    const double* input,
    const word outputName,
    const word outputType,
    const int outputSize,
    const double* seed,
    double* product)
{
#ifdef CODI_ADR
    /*
    Description:
        Calculate the Jacobian-matrix-transposed and vector product for [dOutput/dInput]^T * psi
    
    Input:
        inputName: name of the input. This is usually defined in solverInputs

        inputType: type of the input. This should be consistent with the child class type in DAInput

        inputSize: size of the input array

        input: the actual value of the input array

        outputName: name of the output.

        outputType: type of the output. This should be consistent with the child class type in DAOutput

        outputSize: size of the output array

        seed: the seed array
    
    Output:
        product: the mat-vec product array
    */

    Info << "Computing d[" << outputName << "]/d[" << inputName << "]^T * psi" << endl;

    // initialize the input and output objects
    autoPtr<DAInput> daInput(
        DAInput::New(
            inputName,
            inputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_()));

    autoPtr<DAOutput> daOutput(
        DAOutput::New(
            outputName,
            outputType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            daFunctionPtrList_));

    // create input and output lists
    scalarList inputList(inputSize, 0.0);
    scalarList outputList(outputSize, 0.0);

    // assign the input array to the input list.
    // Note: we need to use scalarList for AD
    forAll(inputList, idxI)
    {
        inputList[idxI] = input[idxI];
    }

    // reset tape
    this->globalADTape_.reset();
    // activate tape, start recording
    this->globalADTape_.setActive();
    // register input
    forAll(inputList, idxI)
    {
        this->globalADTape_.registerInput(inputList[idxI]);
    }
    // call daInput->run to assign inputList to OF variables
    daInput->run(inputList);
    // update all intermediate variables and boundary conditions
    this->updateStateBoundaryConditions();
    // call daOutput->run to compute OF output variables and assign them to outputList
    daOutput->run(outputList);
    // register output
    forAll(outputList, idxI)
    {
        this->globalADTape_.registerOutput(outputList[idxI]);
    }
    // stop recording
    this->globalADTape_.setPassive();
    // assign the seed to the outputList's gradient
    forAll(outputList, idxI)
    {
        // if the output is in serial (e.g., function), we need to assign the seed to
        // only the master processor. This is because the serial output already called
        // a reduce in the daOutput->run function.
        if (daOutput().distributed())
        {
            // output is distributed, assign seed to all procs
            outputList[idxI].setGradient(seed[idxI]);
        }
        else
        {
            // output is in serial, assign seed to the master proc only
            if (Pstream::master())
            {
                outputList[idxI].setGradient(seed[idxI]);
            }
        }
    }
    // evaluate tape to compute derivative
    this->globalADTape_.evaluate();
    // get the matrix-vector product=[dOutput/dInput]^T*seed from the inputList
    // and assign it to the product array
    forAll(inputList, idxI)
    {
        product[idxI] = inputList[idxI].getGradient();
        // if the input is in serial (e.g., angle of attack), we need to reduce the product and
        // make sure the product is consistent among all processors
        if (!daInput().distributed())
        {
            reduce(product[idxI], sumOp<double>());
        }
    }

    // we need to normalize the jacobian vector product if inputType == stateVar
    this->normalizeJacTVecProduct(inputType, product);

    // need to clear adjoint and tape after the computation is done!
    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // clean up OF vars's AD seeds by deactivating the inputs (set its gradients to zeros)
    // and calculate the output one more time. This will propagate the zero seeds
    // to all the intermediate variables and reset their gradient to zeros
    // NOTE: cleaning up the seeds is critical; otherwise, it will create AD conflict
    forAll(inputList, idxI)
    {
        this->globalADTape_.deactivateValue(inputList[idxI]);
    }
    daInput->run(inputList);
    this->updateStateBoundaryConditions();
    daOutput->run(outputList);

#endif
}

void DASolver::calcdRdWTPsiAD(
    const label isInit,
    const Vec psi,
    Vec dRdWTPsi)
{
#ifdef CODI_ADR
    /*
    Description:
        Compute the matrix-vector products dRdW^T*Psi using reverse-mode AD
        Note that this function does not assign wVec and xVec to OF fields
    
    Input:

        mode: either "init" or "run"

        psi: the vector to multiply dRdW0^T
    
    Output:
        dRdWTPsi: the matrix-vector products dRdW^T * Psi
    */

    // this function is not used and commented out for now

    /*
    Info << "Calculating [dRdW]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdWTPsi);

    if (isInit)
    {
        this->globalADTape_.reset();
        this->globalADTape_.setActive();

        this->registerStateVariableInput4AD();

        // compute residuals
        this->updateStateBoundaryConditions();
        this->calcResiduals();

        this->registerResidualOutput4AD();
        this->globalADTape_.setPassive();
    }

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    // get the deriv values
    this->assignStateGradient2Vec(dRdWTPsi);

    VecAssemblyBegin(dRdWTPsi);
    VecAssemblyEnd(dRdWTPsi);

    this->globalADTape_.clearAdjoints();
    */

#endif
}

void DASolver::calcdRdWTPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    Vec dRdWTPsi)
{
#ifdef CODI_ADR
    /*
    Description:
        Compute the matrix-vector products dRdW^T*Psi using reverse-mode AD
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdW0^T
    
    Output:
        dRdWTPsi: the matrix-vector products dRdW^T * Psi
    */

    Info << "Calculating [dRdW]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdWTPsi);

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    this->registerStateVariableInput4AD();

    // compute residuals
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    // get the deriv values
    this->assignStateGradient2Vec(dRdWTPsi);

    // NOTE: we need to normalize dRdWTPsi!
    this->normalizeGradientVec(dRdWTPsi);

    VecAssemblyBegin(dRdWTPsi);
    VecAssemblyEnd(dRdWTPsi);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************
    this->deactivateStateVariableInput4AD();
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdWTPsi") || writeJacobians.found("all"))
    {
        word outputName = "dRdWTPsi";
        DAUtility::writeVectorBinary(dRdWTPsi, outputName);
        DAUtility::writeVectorASCII(dRdWTPsi, outputName);
    }
#endif
}

void DASolver::calcdRdWOldTPsiAD(
    const label oldTimeLevel,
    const Vec psi,
    Vec dRdWOldTPsi)
{
#ifdef CODI_ADR
    /*
    Description:
        Compute the matrix-vector products dRdWOld^T*Psi using reverse-mode AD
        Here WOld means the state variable from previous time step,
        if oldTimeLevel = 1, WOld means W0, if oldTimeLevel=2, WOld means W00
        R is always the residuals for the current time step
        NOTE: if the oldTimeLevel is greater than the max nOldTimes a variable 
        has, the derivative will be zero. This is done by registering input
        only for variables that have enough oldTimes in registerStateVariableInput4AD
    
    Input:

        oldTimeLevel: 1-dRdW0^T  2-dRdW00^T

        psi: the vector to multiply dRdW0^T
    
    Output:
        dRdWOldTPsi: the matrix-vector products dRdWOld^T * Psi
    */

    Info << "Calculating [dRdWOld]^T * Psi using reverse-mode AD with level " << oldTimeLevel << endl;

    VecZeroEntries(dRdWOldTPsi);

    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    this->registerStateVariableInput4AD(oldTimeLevel);

    // compute residuals
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    // get the deriv values
    this->assignStateGradient2Vec(dRdWOldTPsi, oldTimeLevel);

    // NOTE: we need to normalize dRdWOldTPsi!
    this->normalizeGradientVec(dRdWOldTPsi);

    VecAssemblyBegin(dRdWOldTPsi);
    VecAssemblyEnd(dRdWOldTPsi);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************
    this->deactivateStateVariableInput4AD(oldTimeLevel);
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdWOldTPsi") || writeJacobians.found("all"))
    {
        word outputName = "dRdWOldTPsi";
        DAUtility::writeVectorBinary(dRdWOldTPsi, outputName);
        DAUtility::writeVectorASCII(dRdWOldTPsi, outputName);
    }
#endif
}

void DASolver::registerStateVariableInput4AD(const label oldTimeLevel)
{
#ifdef CODI_ADR
    /*
    Description:
        Register all state variables as the input for reverse-mode AD
    
    Input:
        oldTimeLevel: which time level to register, the default value
        is 0, meaning it will register the state itself. If its 
        value is 1, it will register state.oldTime(), if its value
        is 2, it will register state.oldTime().oldTime(). For
        steady-state adjoint oldTimeLevel = 0
    */

    if (oldTimeLevel < 0 || oldTimeLevel > 2)
    {
        FatalErrorIn("") << "oldTimeLevel not valid. Options: 0, 1, or 2"
                         << abort(FatalError);
    }

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, cellI)
            {
                for (label i = 0; i < 3; i++)
                {
                    if (oldTimeLevel == 0)
                    {
                        this->globalADTape_.registerInput(state[cellI][i]);
                    }
                    else if (oldTimeLevel == 1)
                    {
                        this->globalADTape_.registerInput(state.oldTime()[cellI][i]);
                    }
                    else if (oldTimeLevel == 2)
                    {
                        this->globalADTape_.registerInput(state.oldTime().oldTime()[cellI][i]);
                    }
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, cellI)
            {
                if (oldTimeLevel == 0)
                {
                    this->globalADTape_.registerInput(state[cellI]);
                }
                else if (oldTimeLevel == 1)
                {
                    this->globalADTape_.registerInput(state.oldTime()[cellI]);
                }
                else if (oldTimeLevel == 2)
                {
                    this->globalADTape_.registerInput(state.oldTime().oldTime()[cellI]);
                }
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, cellI)
            {
                if (oldTimeLevel == 0)
                {
                    this->globalADTape_.registerInput(state[cellI]);
                }
                else if (oldTimeLevel == 1)
                {
                    this->globalADTape_.registerInput(state.oldTime()[cellI]);
                }
                else if (oldTimeLevel == 2)
                {
                    this->globalADTape_.registerInput(state.oldTime().oldTime()[cellI]);
                }
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, faceI)
            {
                if (oldTimeLevel == 0)
                {
                    this->globalADTape_.registerInput(state[faceI]);
                }
                else if (oldTimeLevel == 1)
                {
                    this->globalADTape_.registerInput(state.oldTime()[faceI]);
                }
                else if (oldTimeLevel == 2)
                {
                    this->globalADTape_.registerInput(state.oldTime().oldTime()[faceI]);
                }
            }
            forAll(state.boundaryField(), patchI)
            {
                forAll(state.boundaryField()[patchI], faceI)
                {
                    if (oldTimeLevel == 0)
                    {
                        this->globalADTape_.registerInput(state.boundaryFieldRef()[patchI][faceI]);
                    }
                    else if (oldTimeLevel == 1)
                    {
                        this->globalADTape_.registerInput(state.oldTime().boundaryFieldRef()[patchI][faceI]);
                    }
                    else if (oldTimeLevel == 2)
                    {
                        this->globalADTape_.registerInput(state.oldTime().oldTime().boundaryFieldRef()[patchI][faceI]);
                    }
                }
            }
        }
    }

#endif
}

void DASolver::deactivateStateVariableInput4AD(const label oldTimeLevel)
{
#ifdef CODI_ADR
    /*
    Description:
        Deactivate all state variables as the input for reverse-mode AD
    Input:
        oldTimeLevel: which time level to register, the default value
        is 0, meaning it will register the state itself. If its 
        value is 1, it will register state.oldTime(), if its value
        is 2, it will register state.oldTime().oldTime(). For
        steady-state adjoint oldTimeLevel = 0
    */

    if (oldTimeLevel < 0 || oldTimeLevel > 2)
    {
        FatalErrorIn("") << "oldTimeLevel not valid. Options: 0, 1, or 2"
                         << abort(FatalError);
    }

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, cellI)
            {
                for (label i = 0; i < 3; i++)
                {
                    if (oldTimeLevel == 0)
                    {
                        this->globalADTape_.deactivateValue(state[cellI][i]);
                    }
                    else if (oldTimeLevel == 1)
                    {
                        this->globalADTape_.deactivateValue(state.oldTime()[cellI][i]);
                    }
                    else if (oldTimeLevel == 2)
                    {
                        this->globalADTape_.deactivateValue(state.oldTime().oldTime()[cellI][i]);
                    }
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, cellI)
            {
                if (oldTimeLevel == 0)
                {
                    this->globalADTape_.deactivateValue(state[cellI]);
                }
                else if (oldTimeLevel == 1)
                {
                    this->globalADTape_.deactivateValue(state.oldTime()[cellI]);
                }
                else if (oldTimeLevel == 2)
                {
                    this->globalADTape_.deactivateValue(state.oldTime().oldTime()[cellI]);
                }
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, cellI)
            {
                if (oldTimeLevel == 0)
                {
                    this->globalADTape_.deactivateValue(state[cellI]);
                }
                else if (oldTimeLevel == 1)
                {
                    this->globalADTape_.deactivateValue(state.oldTime()[cellI]);
                }
                else if (oldTimeLevel == 2)
                {
                    this->globalADTape_.deactivateValue(state.oldTime().oldTime()[cellI]);
                }
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(state, faceI)
            {
                if (oldTimeLevel == 0)
                {
                    this->globalADTape_.deactivateValue(state[faceI]);
                }
                else if (oldTimeLevel == 1)
                {
                    this->globalADTape_.deactivateValue(state.oldTime()[faceI]);
                }
                else if (oldTimeLevel == 2)
                {
                    this->globalADTape_.deactivateValue(state.oldTime().oldTime()[faceI]);
                }
            }
            forAll(state.boundaryField(), patchI)
            {
                forAll(state.boundaryField()[patchI], faceI)
                {
                    if (oldTimeLevel == 0)
                    {
                        this->globalADTape_.deactivateValue(state.boundaryFieldRef()[patchI][faceI]);
                    }
                    else if (oldTimeLevel == 1)
                    {
                        this->globalADTape_.deactivateValue(state.oldTime().boundaryFieldRef()[patchI][faceI]);
                    }
                    else if (oldTimeLevel == 2)
                    {
                        this->globalADTape_.deactivateValue(state.oldTime().oldTime().boundaryFieldRef()[patchI][faceI]);
                    }
                }
            }
        }
    }

#endif
}

void DASolver::registerFieldVariableInput4AD(
    const word fieldName,
    const word fieldType)
{
#ifdef CODI_ADR
    /*
    Description:
        Register field variables as the input for reverse-mode AD
    
    Input:
        fieldName: the name of the flow field to register

        fieldType: can be either scalar or vector
    */

    if (fieldType == "scalar")
    {
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));

        forAll(state, cellI)
        {
            this->globalADTape_.registerInput(state[cellI]);
        }
    }
    else if (fieldType == "vector")
    {
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));

        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                this->globalADTape_.registerInput(state[cellI][i]);
            }
        }
    }
    else
    {
        FatalErrorIn("") << "fieldType not valid. Options: scalar or vector"
                         << abort(FatalError);
    }

#endif
}

void DASolver::deactivateFieldVariableInput4AD(
    const word fieldName,
    const word fieldType)
{
#ifdef CODI_ADR
    /*
    Description:
        Deacitvate field variables as the input for reverse-mode AD
    
    Input:
        fieldName: the name of the flow field to register

        fieldType: can be either scalar or vector
    */

    if (fieldType == "scalar")
    {
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));

        forAll(state, cellI)
        {
            this->globalADTape_.deactivateValue(state[cellI]);
        }
    }
    else if (fieldType == "vector")
    {
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));

        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                this->globalADTape_.deactivateValue(state[cellI][i]);
            }
        }
    }
    else
    {
        FatalErrorIn("") << "fieldType not valid. Options: scalar or vector"
                         << abort(FatalError);
    }

#endif
}

void DASolver::registerResidualOutput4AD()
{
#ifdef CODI_ADR
    /*
    Description:
        Register all residuals as the output for reverse-mode AD
    */

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word stateResName = stateName + "Res";
        volVectorField& stateRes = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(stateResName));

        forAll(stateRes, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                this->globalADTape_.registerOutput(stateRes[cellI][i]);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word stateResName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateResName));

        forAll(stateRes, cellI)
        {
            this->globalADTape_.registerOutput(stateRes[cellI]);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word stateResName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateResName));

        forAll(stateRes, cellI)
        {
            this->globalADTape_.registerOutput(stateRes[cellI]);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word stateResName = stateName + "Res";
        surfaceScalarField& stateRes = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateResName));

        forAll(stateRes, faceI)
        {
            this->globalADTape_.registerOutput(stateRes[faceI]);
        }
        forAll(stateRes.boundaryField(), patchI)
        {
            forAll(stateRes.boundaryField()[patchI], faceI)
            {
                this->globalADTape_.registerOutput(stateRes.boundaryFieldRef()[patchI][faceI]);
            }
        }
    }
#endif
}

void DASolver::registerForceOutput4AD(
    List<scalar>& fX,
    List<scalar>& fY,
    List<scalar>& fZ)
{
#if defined(CODI_ADR)
    /*
    Description:
        Register all force components as the output for reverse-mode AD

    Inputs:
        fX: Vector of X-component of forces

        fY: Vector of Y-component of forces

        fZ: Vector of Z-component of forces
    */
    forAll(fX, cI)
    {
        // Set seeds
        this->globalADTape_.registerOutput(fX[cI]);
        this->globalADTape_.registerOutput(fY[cI]);
        this->globalADTape_.registerOutput(fZ[cI]);
    }
#endif
}

void DASolver::registerAcousticOutput4AD(
    List<scalar>& a)
{
#if defined(CODI_ADR)
    /*
    Description:
        Register all acoustic components as the output for reverse-mode AD

    Inputs:
        a: Vector of scalar entries
    */
    forAll(a, cI)
    {
        // Set seeds
        this->globalADTape_.registerOutput(a[cI]);
    }
#endif
}

void DASolver::normalizeGradientVec(Vec vecY)
{
#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Normalize the reverse-mode AD derivatives stored in vecY
    
    Input/Output:
        vecY: vector to be normalized. vecY = vecY * scalingFactor
        the scalingFactor depends on states.
        This is needed for the matrix-vector products in matrix-free adjoint

    */

    dictionary normStateDict = daOptionPtr_->getAllOptions().subDict("normalizeStates");

    PetscScalar* vecArray;
    VecGetArray(vecY, &vecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        // if normalized state not defined, skip
        if (normStateDict.found(stateName))
        {

            scalar scalingFactor = normStateDict.getScalar(stateName);

            forAll(meshPtr_->cells(), cellI)
            {
                for (label i = 0; i < 3; i++)
                {
                    label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                    vecArray[localIdx] *= scalingFactor.getValue();
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        // if normalized state not defined, skip
        if (normStateDict.found(stateName))
        {
            scalar scalingFactor = normStateDict.getScalar(stateName);

            forAll(meshPtr_->cells(), cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
                vecArray[localIdx] *= scalingFactor.getValue();
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        // if normalized state not defined, skip
        if (normStateDict.found(stateName))
        {

            scalar scalingFactor = normStateDict.getScalar(stateName);

            forAll(meshPtr_->cells(), cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
                vecArray[localIdx] *= scalingFactor.getValue();
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        // if normalized state not defined, skip
        if (normStateDict.found(stateName))
        {
            scalar scalingFactor = normStateDict.getScalar(stateName);

            forAll(meshPtr_->faces(), faceI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

                if (faceI < daIndexPtr_->nLocalInternalFaces)
                {
                    scalar meshSf = meshPtr_->magSf()[faceI];
                    vecArray[localIdx] *= scalingFactor.getValue() * meshSf.getValue();
                }
                else
                {
                    label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                    label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                    label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                    scalar meshSf = meshPtr_->magSf().boundaryField()[patchIdx][faceIdx];
                    vecArray[localIdx] *= scalingFactor.getValue() * meshSf.getValue();
                }
            }
        }
    }

    VecRestoreArray(vecY, &vecArray);

#endif
}

void DASolver::assignSeeds2ResidualGradient(const double* seeds)
{

/*
    Description:
        Assign the reverse-mode AD input seeds from vecX to the residuals in OpenFOAM
    
    Input:
        vecX: vector storing the input seeds
    
    Output:
        All residual variables in OpenFOAM will be set: stateRes[cellI].setGradient(vecX[localIdx])
    */
#if defined(CODI_ADF) || defined(CODI_ADR)

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word resName = stateName + "Res";
        volVectorField& stateRes = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(resName));

        forAll(meshPtr_->cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                stateRes[cellI][i].setGradient(seeds[localIdx]);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word resName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(resName));

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            stateRes[cellI].setGradient(seeds[localIdx]);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word resName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(resName));

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            stateRes[cellI].setGradient(seeds[localIdx]);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word resName = stateName + "Res";
        surfaceScalarField& stateRes = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(resName));

        forAll(meshPtr_->faces(), faceI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                stateRes[faceI].setGradient(seeds[localIdx]);
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                stateRes.boundaryFieldRef()[patchIdx][faceIdx].setGradient(seeds[localIdx]);
            }
        }
    }
#endif
}

void DASolver::assignVec2ResidualGradient(Vec vecX)
{
#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Assign the reverse-mode AD input seeds from vecX to the residuals in OpenFOAM
    
    Input:
        vecX: vector storing the input seeds
    
    Output:
        All residual variables in OpenFOAM will be set: stateRes[cellI].setGradient(vecX[localIdx])
    */

    const PetscScalar* vecArray;
    VecGetArrayRead(vecX, &vecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word resName = stateName + "Res";
        volVectorField& stateRes = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(resName));

        forAll(meshPtr_->cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                stateRes[cellI][i].setGradient(vecArray[localIdx]);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word resName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(resName));

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            stateRes[cellI].setGradient(vecArray[localIdx]);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word resName = stateName + "Res";
        volScalarField& stateRes = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(resName));

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            stateRes[cellI].setGradient(vecArray[localIdx]);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word resName = stateName + "Res";
        surfaceScalarField& stateRes = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(resName));

        forAll(meshPtr_->faces(), faceI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                stateRes[faceI].setGradient(vecArray[localIdx]);
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                stateRes.boundaryFieldRef()[patchIdx][faceIdx].setGradient(vecArray[localIdx]);
            }
        }
    }

    VecRestoreArrayRead(vecX, &vecArray);
#endif
}

void DASolver::assignVec2ForceGradient(
    Vec fBarVec,
    List<scalar>& fX,
    List<scalar>& fY,
    List<scalar>& fZ)
{
#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Assign the reverse-mode AD input seeds from fBarVec to the force vectors
    
    Inputs:
        fBarVec: vector storing the input seeds

        fX: Vector of X-component of forces

        fY: Vector of Y-component of forces

        fZ: Vector of Z-component of forces
    
    Outputs:
        All force variables (for computing surface forces) will be set
    */
    PetscScalar* vecArrayFBarVec;
    VecGetArray(fBarVec, &vecArrayFBarVec);

    label i = 0;
    forAll(fX, cI)
    {
        // Set seeds
        fX[cI].setGradient(vecArrayFBarVec[i]);
        fY[cI].setGradient(vecArrayFBarVec[i + 1]);
        fZ[cI].setGradient(vecArrayFBarVec[i + 2]);

        // Increment counter
        i += 3;
    }

    VecRestoreArray(fBarVec, &vecArrayFBarVec);
#endif
}

void DASolver::assignVec2AcousticGradient(
    Vec fBarVec,
    List<scalar>& a,
    label offset,
    label step)
{
#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Assign the reverse-mode AD input seeds from fBarVec to the force vectors
    
    Inputs:
        fBarVec: vector storing the input seeds

        a: Vector of entries 1
    
    Outputs:
        All acoustic variables will be set
    */
    PetscScalar* vecArrayFBarVec;
    VecGetArray(fBarVec, &vecArrayFBarVec);

    label i = 0;
    forAll(a, cI)
    {
        // Set seeds
        a[cI].setGradient(vecArrayFBarVec[i + offset]);

        // Increment counter
        i += step;
    }

    VecRestoreArray(fBarVec, &vecArrayFBarVec);
#endif
}

void DASolver::assignStateGradient2Vec(
    Vec vecY,
    const label oldTimeLevel)
{
#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Set the reverse-mode AD derivatives from the state variables in OpenFOAM to vecY
    
    Input:
        OpenFOAM state variables that contain the reverse-mode derivative 

        oldTimeLevel: which time level to register, the default value
        is 0, meaning it will register the state itself. If its 
        value is 1, it will register state.oldTime(), if its value
        is 2, it will register state.oldTime().oldTime(). For
        steady-state adjoint oldTimeLevel = 0
    
    Output:
        vecY: a vector to store the derivatives. The order of this vector is 
        the same as the state variable vector
    */

    if (oldTimeLevel < 0 || oldTimeLevel > 2)
    {
        FatalErrorIn("") << "oldTimeLevel not valid. Options: 0, 1, or 2"
                         << abort(FatalError);
    }

    PetscScalar* vecArray;
    VecGetArray(vecY, &vecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(meshPtr_->cells(), cellI)
            {
                for (label i = 0; i < 3; i++)
                {
                    label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                    if (oldTimeLevel == 0)
                    {
                        vecArray[localIdx] = state[cellI][i].getGradient();
                    }
                    else if (oldTimeLevel == 1)
                    {
                        vecArray[localIdx] = state.oldTime()[cellI][i].getGradient();
                    }
                    else if (oldTimeLevel == 2)
                    {
                        vecArray[localIdx] = state.oldTime().oldTime()[cellI][i].getGradient();
                    }
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(meshPtr_->cells(), cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
                if (oldTimeLevel == 0)
                {
                    vecArray[localIdx] = state[cellI].getGradient();
                }
                else if (oldTimeLevel == 1)
                {
                    vecArray[localIdx] = state.oldTime()[cellI].getGradient();
                }
                else if (oldTimeLevel == 2)
                {
                    vecArray[localIdx] = state.oldTime().oldTime()[cellI].getGradient();
                }
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(meshPtr_->cells(), cellI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
                if (oldTimeLevel == 0)
                {
                    vecArray[localIdx] = state[cellI].getGradient();
                }
                else if (oldTimeLevel == 1)
                {
                    vecArray[localIdx] = state.oldTime()[cellI].getGradient();
                }
                else if (oldTimeLevel == 2)
                {
                    vecArray[localIdx] = state.oldTime().oldTime()[cellI].getGradient();
                }
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            forAll(meshPtr_->faces(), faceI)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

                if (faceI < daIndexPtr_->nLocalInternalFaces)
                {
                    if (oldTimeLevel == 0)
                    {
                        vecArray[localIdx] = state[faceI].getGradient();
                    }
                    else if (oldTimeLevel == 1)
                    {
                        vecArray[localIdx] = state.oldTime()[faceI].getGradient();
                    }
                    else if (oldTimeLevel == 2)
                    {
                        vecArray[localIdx] = state.oldTime().oldTime()[faceI].getGradient();
                    }
                }
                else
                {
                    label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                    label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                    label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];

                    if (oldTimeLevel == 0)
                    {
                        vecArray[localIdx] =
                            state.boundaryField()[patchIdx][faceIdx].getGradient();
                    }
                    else if (oldTimeLevel == 1)
                    {
                        vecArray[localIdx] =
                            state.oldTime().boundaryField()[patchIdx][faceIdx].getGradient();
                    }
                    else if (oldTimeLevel == 2)
                    {
                        vecArray[localIdx] =
                            state.oldTime().oldTime().boundaryField()[patchIdx][faceIdx].getGradient();
                    }
                }
            }
        }
    }

    VecRestoreArray(vecY, &vecArray);

#endif
}

void DASolver::assignFieldGradient2Vec(
    const word fieldName,
    const word fieldType,
    Vec vecY)
{
#if defined(CODI_ADF) || defined(CODI_ADR)
    /*
    Description:
        Set the reverse-mode AD derivatives from the field variables in OpenFOAM to vecY
    
    Input:
        OpenFOAM field variables that contain the reverse-mode derivative 
    
    Output:
        vecY: a vector to store the derivatives. The order of this vector is 
        the same as the field variable vector
    */

    PetscScalar* vecArray;
    VecGetArray(vecY, &vecArray);

    if (fieldType == "scalar")
    {
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));

        forAll(state, cellI)
        {
            vecArray[cellI] = state[cellI].getGradient();
        }
    }
    else if (fieldType == "vector")
    {
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));

        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = cellI * 3 + i;
                vecArray[localIdx] = state[cellI][i].getGradient();
            }
        }
    }
    else
    {
        FatalErrorIn("") << "fieldType not valid. Options: scalar or vector"
                         << abort(FatalError);
    }

    VecRestoreArray(vecY, &vecArray);

#endif
}

void DASolver::convertMPIVec2SeqVec(
    const Vec mpiVec,
    Vec seqVec)
{
    /*
    Description: 
        Convert a MPI vec to a seq vec by using VecScatter
    
    Input:
        mpiVec: the MPI vector in parallel

    Output:
        seqVec: the seq vector in serial
    */
    label vecSize;
    VecGetSize(mpiVec, &vecSize);

    // scatter colors to local array for all procs
    Vec vout;
    VecScatter ctx;
    VecScatterCreateToAll(mpiVec, &ctx, &vout);
    VecScatterBegin(ctx, mpiVec, vout, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, mpiVec, vout, INSERT_VALUES, SCATTER_FORWARD);

    PetscScalar* voutArray;
    VecGetArray(vout, &voutArray);

    PetscScalar* seqVecArray;
    VecGetArray(seqVec, &seqVecArray);

    for (label i = 0; i < vecSize; i++)
    {
        seqVecArray[i] = voutArray[i];
    }
    VecRestoreArray(vout, &voutArray);
    VecRestoreArray(seqVec, &seqVecArray);
    VecScatterDestroy(&ctx);
    VecDestroy(&vout);
}

void DASolver::setdXvdFFDMat(const Mat dXvdFFDMat)
{
    /*
    Description:
        Set the value for dXvdFFDMat_. Basically we use MatConvert
    */
    MatConvert(dXvdFFDMat, MATSAME, MAT_INITIAL_MATRIX, &dXvdFFDMat_);
    //MatDuplicate(dXvdFFDMat, MAT_COPY_VALUES, &dXvdFFDMat_);
    MatAssemblyBegin(dXvdFFDMat_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dXvdFFDMat_, MAT_FINAL_ASSEMBLY);
}

void DASolver::setFFD2XvSeedVec(Vec vecIn)
{
    /*
    Description:
        Set the value for FFD2XvSeedVec_
    
    Input:
        vecIn: this vector will be copied to FFD2XvSeedVec_
    */
    VecDuplicate(vecIn, &FFD2XvSeedVec_);
    VecCopy(vecIn, FFD2XvSeedVec_);
}

label DASolver::checkResidualTol(const scalar& primalMaxRes)
{
    /*
    Description:
        Check whether the min residual in primal satisfy the prescribed tolerance
        If yes, return 0 else return 1
    */

    // when checking the tolerance, we relax the criteria by tolMax

    scalar tolMax = daOptionPtr_->getOption<scalar>("primalMinResTolDiff");
    scalar stdTolMax = daOptionPtr_->getSubDictOption<scalar>("primalObjStdTol", "tolDiff");
    if (primalMaxRes / primalMinResTol_ > tolMax)
    {
        Info << "********************************************" << endl;
        Info << "Primal min residual " << primalMaxRes << endl
             << "did not satisfy the prescribed tolerance "
             << primalMinResTol_ << endl;
        Info << "Primal solution failed!" << endl;
        Info << "********************************************" << endl;
        return 1;
    }
    else if (primalObjStd_ / primalObjStdTol_ > stdTolMax)
    {
        Info << "********************************************" << endl;
        Info << "Primal function standard deviation " << primalObjStd_ << endl
             << "did not satisfy the prescribed tolerance "
             << primalObjStdTol_ << endl;
        Info << "Primal solution failed!" << endl;
        Info << "********************************************" << endl;
        return 1;
    }
    else
    {
        return 0;
    }

    return 1;
}

label DASolver::isPrintTime(
    const Time& runTime,
    const label printInterval) const
{
    /*
    Description:
        Check if it is print time
    */
    if (runTime.timeIndex() % printInterval == 0 || runTime.timeIndex() == 1)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void DASolver::writeAssociatedFields()
{
    /*
    Description:
        Write associated fields such as relative velocity
    */

    IOobject MRFIO(
        "MRFProperties",
        runTimePtr_->constant(),
        meshPtr_(),
        IOobject::MUST_READ,
        IOobject::NO_WRITE,
        false); // do not register

    if (MRFIO.typeHeaderOk<IOdictionary>(true))
    {
        IOdictionary MRFProperties(MRFIO);

        bool activeMRF(MRFProperties.subDict("MRF").lookupOrDefault("active", true));

        if (activeMRF)
        {
            const volVectorField& U = meshPtr_->thisDb().lookupObject<volVectorField>("U");

            volVectorField URel("URel", U);
            IOMRFZoneList MRF(meshPtr_());
            MRF.makeRelative(URel);
            URel.write();
        }
    }
}

void DASolver::setFieldValue4LocalCellI(
    const word fieldName,
    const scalar val,
    const label localCellI,
    const label compI)
{
    /*
    Description:
        Set the field value based on the local cellI. 
    
    Input:
        fieldName: the name of the field to set

        val: the value to set

        localCellI: the local cell index

        compI: which component to set (only for vectors such as U)
    */

    if (meshPtr_->thisDb().foundObject<volVectorField>(fieldName))
    {
        volVectorField& field =
            const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));
        field[localCellI][compI] = val;
    }
    else if (meshPtr_->thisDb().foundObject<volScalarField>(fieldName))
    {
        volScalarField& field =
            const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));
        field[localCellI] = val;
    }
    else
    {
        FatalErrorIn("") << fieldName << " not found in volScalar and volVector Fields "
                         << abort(FatalError);
    }
}

void DASolver::setFieldValue4GlobalCellI(
    const word fieldName,
    const scalar val,
    const label globalCellI,
    const label compI)
{
    /*
    Description:
        Set the field value based on the global cellI. This is usually
        used if the state variables are design variables, e.g., betaSA
        The reason to use global cell index, instead of local one, is 
        because this index is usually provided by the optimizer. Optimizer
        uses global cell index as the design variable
    
    Input:
        fieldName: the name of the field to set

        val: the value to set

        globalCellI: the global cell index

        compI: which component to set (only for vectors such as U)
    */

    if (meshPtr_->thisDb().foundObject<volVectorField>(fieldName))
    {
        if (daIndexPtr_->globalCellNumbering.isLocal(globalCellI))
        {
            volVectorField& field =
                const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));
            label localCellI = daIndexPtr_->globalCellNumbering.toLocal(globalCellI);
            field[localCellI][compI] = val;
        }
    }
    else if (meshPtr_->thisDb().foundObject<volScalarField>(fieldName))
    {
        if (daIndexPtr_->globalCellNumbering.isLocal(globalCellI))
        {
            volScalarField& field =
                const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));
            label localCellI = daIndexPtr_->globalCellNumbering.toLocal(globalCellI);
            field[localCellI] = val;
        }
    }
    else
    {
        FatalErrorIn("") << fieldName << " not found in volScalar and volVector Fields "
                         << abort(FatalError);
    }
}

void DASolver::calcResidualVec(Vec resVec)
{
    /*
    Description:
        Calculate the residual and assign it to resVec
    
    Input/Output:
        resVec: residual vector
    */

    // compute residuals
    this->updateStateBoundaryConditions();
    this->calcResiduals();

    PetscScalar* vecArray;
    VecGetArray(resVec, &vecArray);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const word resName = stateName + "Res";
        const volVectorField& stateRes = meshPtr_->thisDb().lookupObject<volVectorField>(resName);

        forAll(meshPtr_->cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                assignValueCheckAD(vecArray[localIdx], stateRes[cellI][i]);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const word resName = stateName + "Res";
        const volScalarField& stateRes = meshPtr_->thisDb().lookupObject<volScalarField>(resName);

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            assignValueCheckAD(vecArray[localIdx], stateRes[cellI]);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word resName = stateName + "Res";
        const volScalarField& stateRes = meshPtr_->thisDb().lookupObject<volScalarField>(resName);

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            assignValueCheckAD(vecArray[localIdx], stateRes[cellI]);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const word resName = stateName + "Res";
        const surfaceScalarField& stateRes = meshPtr_->thisDb().lookupObject<surfaceScalarField>(resName);

        forAll(meshPtr_->faces(), faceI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                assignValueCheckAD(vecArray[localIdx], stateRes[faceI]);
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                assignValueCheckAD(vecArray[localIdx], stateRes.boundaryField()[patchIdx][faceIdx]);
            }
        }
    }

    VecRestoreArray(resVec, &vecArray);
}

void DASolver::updateBoundaryConditions(
    const word fieldName,
    const word fieldType)
{
    /*
    Description:
        Update the boundary condition for a field
    
    Input:
        fieldName: the name of the field to update

        fieldType: either scalar or vector
    */

    if (fieldType == "scalar")
    {
        volScalarField& field =
            const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));
        field.correctBoundaryConditions();
    }
    else if (fieldType == "vector")
    {
        volVectorField& field =
            const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));
        field.correctBoundaryConditions();
    }
    else
    {
        FatalErrorIn("") << fieldType << " not support. Options are: vector or scalar "
                         << abort(FatalError);
    }
}

void DASolver::calcResiduals(label isPC)
{
    /*
    Description:
        Calculate the residuals and assign values to the residual OF variables in the DAResidual object, such as URes_, pRes_
    
    Inputs:
        isPC: whether the residual calculate is for preconditioner, default false
    */

    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);
}

void DASolver::updateStateBoundaryConditions()
{
    /*
    Description:
        Update the boundary condition and intermediate variables for all state variables
    */

    // if we have regression models, we also need to update them because they will update the fields
    this->regressionModelCompute();

    label nBCCalls = 1;
    if (daOptionPtr_->getOption<label>("hasIterativeBC"))
    {
        nBCCalls = daOptionPtr_->getOption<label>("maxCorrectBCCalls");
    }

    for (label i = 0; i < nBCCalls; i++)
    {
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
    }
}

void DASolver::saveTimeInstanceFieldHybrid(label& timeInstanceI)
{
    /*
    Description:
        Save primal variable to time instance list for unsteady adjoint
        Here we save the last nTimeInstances snapshots
    */

    scalar endTime = runTimePtr_->endTime().value();
    scalar t = runTimePtr_->timeOutputValue();
    scalar instanceStart =
        endTime - periodicity_ / nTimeInstances_ * (nTimeInstances_ - 1 - timeInstanceI);

    // the 2nd condition is for t=9.999999999999 scenario)
    if (t > instanceStart || fabs(t - endTime) < 1e-8)
    {
        Info << "Saving time instance " << timeInstanceI << " at Time = " << t << endl;

        // save fields
        daFieldPtr_->ofField2List(
            stateAllInstances_[timeInstanceI],
            stateBoundaryAllInstances_[timeInstanceI]);

        // save objective functions
        forAll(daOptionPtr_->getAllOptions().subDict("function").toc(), idxI)
        {
            word functionName = daOptionPtr_->getAllOptions().subDict("function").toc()[idxI];
            scalar functionVal = this->getFunctionValue(functionName);
            functionsAllInstances_[timeInstanceI].set(functionName, functionVal);
        }

        // save runTime
        runTimeAllInstances_[timeInstanceI] = t;
        runTimeIndexAllInstances_[timeInstanceI] = runTimePtr_->timeIndex();

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        timeInstanceI++;
    }
    return;
}

void DASolver::saveTimeInstanceFieldTimeAccurate(label& timeInstanceI)
{
    /*
    Description:
        Save primal variable to time instance list for unsteady adjoint
        Here we save every time step
    */
    // save fields
    daFieldPtr_->ofField2List(
        stateAllInstances_[timeInstanceI],
        stateBoundaryAllInstances_[timeInstanceI]);

    // save objective functions
    forAll(daOptionPtr_->getAllOptions().subDict("function").toc(), idxI)
    {
        word functionName = daOptionPtr_->getAllOptions().subDict("function").toc()[idxI];
        scalar functionVal = this->getFunctionValue(functionName);
        functionsAllInstances_[timeInstanceI].set(functionName, functionVal);
    }

    // save runTime
    scalar t = runTimePtr_->timeOutputValue();
    runTimeAllInstances_[timeInstanceI] = t;
    runTimeIndexAllInstances_[timeInstanceI] = runTimePtr_->timeIndex();

    timeInstanceI++;
}

void DASolver::setTimeInstanceField(const label instanceI)
{
    /*
    Description:
        Assign primal variables based on the current time instance
        If unsteady adjoint solvers are used, this virtual function should be 
        implemented in a child class, otherwise, return error if called
    */

    Info << "Setting fields for time instance " << instanceI << endl;

    label idxI = -9999;

    // set run time
    // NOTE: we need to call setTime before updating the oldTime fields, this is because
    // the setTime call will assign field to field.oldTime()
    runTimePtr_->setTime(runTimeAllInstances_[instanceI], runTimeIndexAllInstances_[instanceI]);

    word mode = daOptionPtr_->getSubDictOption<word>("unsteadyAdjoint", "mode");

    // set fields
    label oldTimeLevel = 0;
    daFieldPtr_->list2OFField(
        stateAllInstances_[instanceI],
        stateBoundaryAllInstances_[instanceI],
        oldTimeLevel);

    // for time accurate adjoint, in addition to assign current fields,
    // we need to assign oldTime fields.
    if (mode == "timeAccurate")
    {
        // assign U.oldTime()
        oldTimeLevel = 1;
        // if instanceI - 1 < 0, we just assign idxI = 0. This is essentially
        // assigning U.oldTime() = U0
        idxI = max(instanceI - 1, 0);
        daFieldPtr_->list2OFField(
            stateAllInstances_[idxI],
            stateBoundaryAllInstances_[idxI],
            oldTimeLevel);

        // assign U.oldTime().oldTime()
        oldTimeLevel = 2;
        // if instanceI - 2 < 0, we just assign idxI = 0, This is essentially
        // assigning U.oldTime().oldTime() = U0
        idxI = max(instanceI - 2, 0);
        daFieldPtr_->list2OFField(
            stateAllInstances_[idxI],
            stateBoundaryAllInstances_[idxI],
            oldTimeLevel);
    }

    // if we have regression models, we also need to update them because they will update the fields
    this->regressionModelCompute();

    // We need to call correctBC multiple times to reproduce
    // the exact residual for mulitpoint, this is needed for some boundary conditions
    // and intermediate variables (e.g., U for inletOutlet, nut with wall functions)
    for (label i = 0; i < 10; i++)
    {
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
    }
}

void DASolver::calcPCMatWithFvMatrix(Mat PCMat)
{
    /*
    Description:
        calculate the PC mat using fvMatrix. Here we only calculate the block diagonal components, 
        e.g., dR_U/dU, dR_p/dp, etc.
    */

#ifndef SolidDASolver
    //DAUtility::writeMatrixASCII(PCMat, "MatOrig");

    // MatZeroEntries(PCMat);

    // non turbulence variables
    daResidualPtr_->calcPCMatWithFvMatrix(PCMat);

    // turbulence variables
    DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
    const labelUList& owner = meshPtr_->owner();
    const labelUList& neighbour = meshPtr_->neighbour();

    PetscScalar val;

    dictionary normStateDict = daOptionPtr_->getAllOptions().subDict("normalizeStates");
    wordList normResDict = daOptionPtr_->getOption<wordList>("normalizeResiduals");
    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const word resName = stateName + "Res";
        label nCells = meshPtr_->nCells();
        label nInternalFaces = daIndexPtr_->nLocalInternalFaces;
        scalarField D(nCells, 0.0);
        scalarField upper(nInternalFaces, 0.0);
        scalarField lower(nInternalFaces, 0.0);
        daTurb.getFvMatrixFields(stateName, D, upper, lower);

        scalar stateScaling = 1.0;
        if (normStateDict.found(stateName))
        {
            stateScaling = normStateDict.getScalar(stateName);
        }
        scalar resScaling = 1.0;
        // set diag
        forAll(meshPtr_->cells(), cellI)
        {
            if (normResDict.found(resName))
            {
                resScaling = meshPtr_->V()[cellI];
            }

            PetscInt rowI = daIndexPtr_->getGlobalAdjointStateIndex(stateName, cellI);
            PetscInt colI = rowI;
            scalar val1 = D[cellI] * stateScaling / resScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &rowI, 1, &colI, &val, INSERT_VALUES);
        }

        // set lower/owner
        for (label faceI = 0; faceI < daIndexPtr_->nLocalInternalFaces; faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (normResDict.found(resName))
            {
                resScaling = meshPtr_->V()[neighbourCellI];
            }

            PetscInt rowI = daIndexPtr_->getGlobalAdjointStateIndex(stateName, neighbourCellI);
            PetscInt colI = daIndexPtr_->getGlobalAdjointStateIndex(stateName, ownerCellI);
            scalar val1 = lower[faceI] * stateScaling / resScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
        }

        // set upper/neighbour
        for (label faceI = 0; faceI < daIndexPtr_->nLocalInternalFaces; faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (normResDict.found(resName))
            {
                resScaling = meshPtr_->V()[ownerCellI];
            }

            PetscInt rowI = daIndexPtr_->getGlobalAdjointStateIndex(stateName, ownerCellI);
            PetscInt colI = daIndexPtr_->getGlobalAdjointStateIndex(stateName, neighbourCellI);
            scalar val1 = upper[faceI] * stateScaling / resScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(PCMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(PCMat, MAT_FINAL_ASSEMBLY);

    //DAUtility::writeMatrixASCII(PCMat, "MatNew");
#endif
}

/*
void DASolver::disableStateAutoWrite(const wordList& noWriteVars)
{


    forAll(noWriteVars, idxI)
    {
        word varName = noWriteVars[idxI];

        if (varName == "None")
        {
            continue;
        }

        if (meshPtr_->thisDb().foundObject<volScalarField>(varName))
        {
            volScalarField& var =
                const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(varName));

            var.writeOpt() = IOobject::NO_WRITE;
        }
        else if (meshPtr_->thisDb().foundObject<volVectorField>(varName))
        {
            volVectorField& var =
                const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(varName));

            var.writeOpt() = IOobject::NO_WRITE;
        }
        else if (meshPtr_->thisDb().foundObject<surfaceScalarField>(varName))
        {
            surfaceScalarField& var =
                const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(varName));

            var.writeOpt() = IOobject::NO_WRITE;
        }
        else
        {
            Info << "Warning! The prescribed reduceIOVars " << varName << " not found in the db!" << endl;
        }
    }
}
*/

void DASolver::writeAdjStates(
    const label writeMesh,
    const wordList& additionalOutput)
{
    /*
    Description:
        Write only the adjoint states
    */

    if (runTimePtr_->writeTime())
    {

        // volVector states
        forAll(stateInfo_["volVectorStates"], idxI)
        {
            const word stateName = stateInfo_["volVectorStates"][idxI];
            volVectorField& state =
                const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(stateName));

            state.write();
        }

        // volScalar states
        forAll(stateInfo_["volScalarStates"], idxI)
        {
            const word stateName = stateInfo_["volScalarStates"][idxI];
            volScalarField& state =
                const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

            state.write();
        }

        // model states
        forAll(stateInfo_["modelStates"], idxI)
        {
            const word stateName = stateInfo_["modelStates"][idxI];
            volScalarField& state =
                const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

            state.write();
        }

        // surfaceScalar states
        forAll(stateInfo_["surfaceScalarStates"], idxI)
        {
            const word stateName = stateInfo_["surfaceScalarStates"][idxI];
            surfaceScalarField& state =
                const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));

            state.write();
        }

        // also write additional states
        forAll(additionalOutput, idxI)
        {
            word varName = additionalOutput[idxI];

            if (varName == "None")
            {
                continue;
            }

            if (meshPtr_->thisDb().foundObject<volScalarField>(varName))
            {
                volScalarField& var =
                    const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(varName));

                var.write();
            }
            else if (meshPtr_->thisDb().foundObject<volVectorField>(varName))
            {
                volVectorField& var =
                    const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(varName));

                var.write();
            }
            else if (meshPtr_->thisDb().foundObject<surfaceScalarField>(varName))
            {
                surfaceScalarField& var =
                    const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(varName));

                var.write();
            }
            else
            {
                Info << "Warning! The prescribed additionalOutput " << varName << " not found in the db! Ignoring it.." << endl;
            }
        }

        if (writeMesh)
        {
            pointIOField points = meshPtr_->thisDb().lookupObject<pointIOField>("points");
            points.write();
        }
    }
}

void DASolver::readStateVars(
    scalar timeVal,
    label oldTimeLevel)
{
    /*
    Description:
        Read the state variables from the disk and assign the value to the prescribe time level.
        NOTE: we use == to assign both internal and boundary fields!
        We always read oldTimes for volStates, no matter if the oldTimes are actually needed.
        This is not the case for phi. We only read phi oldTime if needed.
        This is to save memory because most of the time, we don't need phi.oldTime(); we do not
        include the ddtCorr term.
    
    Inputs:
        
        timeName: Which time to read, i.e., time.timeName()

        oldTimeLevel: 
            0: read the states and assign to the current time level
            1: read the states and assign to the previous time level (oldTime())
            2: read the states and assign to the 2 previous time level (oldTime().oldTime())
        
    */

    // we can't read negatiev time, so if the timeName is negative, we just read the vars from the 0 folder
    word timeName = Foam::name(timeVal);
    if (timeVal < 0)
    {
        timeName = "0";
    }

    fvMesh& mesh = meshPtr_();

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state =
            const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(stateName));

        volVectorField stateRead(
            IOobject(
                stateName,
                timeName,
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh);

        if (oldTimeLevel == 0)
        {
            state == stateRead;
        }
        else if (oldTimeLevel == 1)
        {
            state.oldTime() == stateRead;
        }
        else if (oldTimeLevel == 2)
        {
            if (timeVal < 0)
            {
                volVectorField state0Read(
                    IOobject(
                        stateName + "_0",
                        timeName,
                        mesh,
                        IOobject::READ_IF_PRESENT,
                        IOobject::NO_WRITE),
                    stateRead);
                state.oldTime().oldTime() == state0Read;
            }
            else
            {
                state.oldTime().oldTime() == stateRead;
            }
        }
        else
        {
            FatalErrorIn("") << "oldTimeLevel can only be 0, 1, and 2!" << abort(FatalError);
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state =
            const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        volScalarField stateRead(
            IOobject(
                stateName,
                timeName,
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh);

        if (oldTimeLevel == 0)
        {
            state == stateRead;
        }
        else if (oldTimeLevel == 1)
        {
            state.oldTime() == stateRead;
        }
        else if (oldTimeLevel == 2)
        {
            if (timeVal < 0)
            {
                volScalarField state0Read(
                    IOobject(
                        stateName + "_0",
                        timeName,
                        mesh,
                        IOobject::READ_IF_PRESENT,
                        IOobject::NO_WRITE),
                    stateRead);
                state.oldTime().oldTime() == state0Read;
            }
            else
            {
                state.oldTime().oldTime() == stateRead;
            }
        }
        else
        {
            FatalErrorIn("") << "oldTimeLevel can only be 0, 1, and 2!" << abort(FatalError);
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state =
            const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        volScalarField stateRead(
            IOobject(
                stateName,
                timeName,
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh);

        if (oldTimeLevel == 0)
        {
            state == stateRead;
        }
        else if (oldTimeLevel == 1)
        {
            state.oldTime() == stateRead;
        }
        else if (oldTimeLevel == 2)
        {
            if (timeVal < 0)
            {
                volScalarField state0Read(
                    IOobject(
                        stateName + "_0",
                        timeName,
                        mesh,
                        IOobject::READ_IF_PRESENT,
                        IOobject::NO_WRITE),
                    stateRead);
                state.oldTime().oldTime() == state0Read;
            }
            else
            {
                state.oldTime().oldTime() == stateRead;
            }
        }
        else
        {
            FatalErrorIn("") << "oldTimeLevel can only be 0, 1, and 2!" << abort(FatalError);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state =
            const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));

        label maxOldTimes = state.nOldTimes();

        if (maxOldTimes >= oldTimeLevel)
        {
            surfaceScalarField stateRead(
                IOobject(
                    stateName,
                    timeName,
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE),
                mesh);

            if (oldTimeLevel == 0)
            {
                state == stateRead;
            }
            else if (oldTimeLevel == 1)
            {
                state.oldTime() == stateRead;
            }
            else if (oldTimeLevel == 2)
            {
                if (timeVal < 0)
                {
                    surfaceScalarField state0Read(
                        IOobject(
                            stateName + "_0",
                            timeName,
                            mesh,
                            IOobject::READ_IF_PRESENT,
                            IOobject::NO_WRITE),
                        stateRead);
                    state.oldTime().oldTime() == state0Read;
                }
                else
                {
                    state.oldTime().oldTime() == stateRead;
                }
            }
            else
            {
                FatalErrorIn("") << "oldTimeLevel can only be 0, 1, and 2!" << abort(FatalError);
            }
        }
    }

    // update the BC and intermediate variables. This is important, e.g., for turbulent cases
    this->updateStateBoundaryConditions();
}

void DASolver::setTimeInstanceVar(
    const word mode,
    Mat stateMat,
    Mat stateBCMat,
    Vec timeVec,
    Vec timeIdxVec)
{
    PetscInt Istart, Iend;
    MatGetOwnershipRange(stateMat, &Istart, &Iend);

    PetscInt IstartBC, IendBC;
    MatGetOwnershipRange(stateBCMat, &IstartBC, &IendBC);

    for (label n = 0; n < nTimeInstances_; n++)
    {
        for (label i = Istart; i < Iend; i++)
        {
            label relIdx = i - Istart;
            PetscScalar val;
            if (mode == "mat2List")
            {
                MatGetValues(stateMat, 1, &i, 1, &n, &val);
                stateAllInstances_[n][relIdx] = val;
            }
            else if (mode == "list2Mat")
            {
                assignValueCheckAD(val, stateAllInstances_[n][relIdx]);
                MatSetValue(stateMat, i, n, val, INSERT_VALUES);
            }
            else
            {
                FatalErrorIn("") << "mode not valid!" << abort(FatalError);
            }
        }

        for (label i = IstartBC; i < IendBC; i++)
        {
            label relIdx = i - IstartBC;
            PetscScalar val;
            if (mode == "mat2List")
            {
                MatGetValues(stateBCMat, 1, &i, 1, &n, &val);
                stateBoundaryAllInstances_[n][relIdx] = val;
            }
            else if (mode == "list2Mat")
            {
                assignValueCheckAD(val, stateBoundaryAllInstances_[n][relIdx]);
                MatSetValue(stateBCMat, i, n, val, INSERT_VALUES);
            }
            else
            {
                FatalErrorIn("") << "mode not valid!" << abort(FatalError);
            }
        }
    }

    if (mode == "list2Mat")
    {
        MatAssemblyBegin(stateMat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(stateMat, MAT_FINAL_ASSEMBLY);
        MatAssemblyBegin(stateBCMat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(stateBCMat, MAT_FINAL_ASSEMBLY);
    }

    PetscScalar* timeVecArray;
    PetscScalar* timeIdxVecArray;
    VecGetArray(timeVec, &timeVecArray);
    VecGetArray(timeIdxVec, &timeIdxVecArray);

    for (label n = 0; n < nTimeInstances_; n++)
    {
        if (mode == "mat2List")
        {
            runTimeAllInstances_[n] = timeVecArray[n];
            runTimeIndexAllInstances_[n] = round(timeIdxVecArray[n]);
        }
        else if (mode == "list2Mat")
        {
            assignValueCheckAD(timeVecArray[n], runTimeAllInstances_[n]);
            timeIdxVecArray[n] = runTimeIndexAllInstances_[n];
        }
        else
        {
            FatalErrorIn("") << "mode not valid!" << abort(FatalError);
        }
    }

    VecRestoreArray(timeVec, &timeVecArray);
    VecRestoreArray(timeIdxVec, &timeIdxVecArray);
}

void DASolver::writeFailedMesh()
{
    /*
    Description:
        If the mesh fails, we set the time to 10000 and write the results to the disk.
        This way, the results will be renamed to 0.00000x during optimization, so that we 
        can visualize them in Paraview to debug which part of the mesh is failing.
    */
    if (daOptionPtr_->getOption<label>("writeMinorIterations"))
    {
        runTimePtr_->setTime(10000.0, 10000);
        runTimePtr_->writeNow();
    }
}

void DASolver::initMeanStates()
{
    /*
    Description:
        Initialize the mean states for DASolver::calcMeanStates
    */

    useMeanStates_ = daOptionPtr_->getSubDictOption<label>("useMeanStates", "active");

    if (useMeanStates_)
    {
        meanStateStart_ = daOptionPtr_->getSubDictOption<scalar>("useMeanStates", "start");
    }
    else
    {
        return;
    }

    Info << "useMeanStates activated. Initializing the meanStates...." << endl;

    // set the sizes
    meanVolScalarStates_.setSize(stateInfo_["volScalarStates"].size());
    meanVolVectorStates_.setSize(stateInfo_["volVectorStates"].size());
    meanModelStates_.setSize(stateInfo_["modelStates"].size());
    meanSurfaceScalarStates_.setSize(stateInfo_["surfaceScalarStates"].size());

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const volVectorField& state = meshPtr_->thisDb().lookupObject<volVectorField>(stateName);

        meanVolVectorStates_.set(
            idxI,
            new volVectorField(
                IOobject(
                    stateName,
                    runTimePtr_->timeName(),
                    meshPtr_(),
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE),
                state));

        meanVolVectorStates_[idxI].rename(stateName + "Mean");
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);

        meanVolScalarStates_.set(
            idxI,
            new volScalarField(
                IOobject(
                    stateName,
                    runTimePtr_->timeName(),
                    meshPtr_(),
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE),
                state));

        meanVolScalarStates_[idxI].rename(stateName + "Mean");
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);

        meanModelStates_.set(
            idxI,
            new volScalarField(
                IOobject(
                    stateName,
                    runTimePtr_->timeName(),
                    meshPtr_(),
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE),
                state));

        meanModelStates_[idxI].rename(stateName + "Mean");
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const surfaceScalarField& state = meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName);

        meanSurfaceScalarStates_.set(
            idxI,
            new surfaceScalarField(
                IOobject(
                    stateName,
                    runTimePtr_->timeName(),
                    meshPtr_(),
                    IOobject::READ_IF_PRESENT,
                    IOobject::NO_WRITE),
                state));

        meanSurfaceScalarStates_[idxI].rename(stateName + "Mean");
    }
}

void DASolver::zeroMeanStates()
{
    /*
    Description:
        Set all the mean states to zeros
    */

    if (!useMeanStates_)
    {
        return;
    }

    Info << "Zeroing the meanStates...." << endl;

    forAll(meanVolVectorStates_, idxI)
    {
        forAll(meanVolVectorStates_[idxI], cellI)
        {
            meanVolVectorStates_[idxI][cellI] = vector::zero;
        }
    }

    forAll(meanVolScalarStates_, idxI)
    {
        forAll(meanVolScalarStates_[idxI], cellI)
        {
            meanVolScalarStates_[idxI][cellI] = 0.0;
        }
    }

    forAll(meanModelStates_, idxI)
    {
        forAll(meanModelStates_[idxI], cellI)
        {
            meanModelStates_[idxI][cellI] = 0.0;
        }
    }

    forAll(meanSurfaceScalarStates_, idxI)
    {
        forAll(meanSurfaceScalarStates_[idxI], faceI)
        {
            meanSurfaceScalarStates_[idxI][faceI] = 0.0;
        }

        forAll(meanSurfaceScalarStates_[idxI].boundaryField(), patchI)
        {
            forAll(meanSurfaceScalarStates_[idxI].boundaryField()[patchI], faceI)
            {
                meanSurfaceScalarStates_[idxI].boundaryFieldRef()[patchI][faceI] = 0.0;
            }
        }
    }
}

void DASolver::assignMeanStatesToStates()
{
    /*
    Description:
        Assigned the calculated meanStates to the primal states and update intermediate vars 
        NOTE: if meanStatesCalculated_ == 0, we will not assignMeanStatesToStates at the end of the primal.
        meanStatesCalculated_ is assigned to 1 if timeIndex >= startTimeIndex in DASolver::calcMeanStates
    */

    if (!useMeanStates_ || !meanStatesCalculated_)
    {
        return;
    }

    Info << "Assigning the meanStates to states...." << endl;

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(stateName));

        state = meanVolVectorStates_[idxI];
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        state = meanVolScalarStates_[idxI];
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));

        state = meanModelStates_[idxI];
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));

        state == meanSurfaceScalarStates_[idxI];
    }

    // update state BC and intermedate vars
    this->updateStateBoundaryConditions();

    // after the meanStates is assigned to states, reset meanStatesCalculated_ for the next primal solution.
    meanStatesCalculated_ = 0;
}

void DASolver::calcMeanStates()
{
    /*
    Description:
        Calculate the mean states
        This is useful for cases when steady-state solvers do not converge very well, e.g., flow
        separation. In these cases, the flow field and the objective function will oscillate and
        to get a better flow field and obj func value, we can use step-averaged (mean) states
    */

    if (!useMeanStates_)
    {
        return;
    }

    // Info << "Calculating the meanStates...." << endl;

    // calculate the average on the fly, i.e., moving average
    scalar endTime = runTimePtr_->endTime().value();
    scalar deltaT = runTimePtr_->deltaT().value();
    label nSteps = round(endTime / deltaT);
    label startTimeIndex = round(nSteps * meanStateStart_);
    label timeIndex = runTimePtr_->timeIndex();
    if (timeIndex >= startTimeIndex)
    {
        label n = timeIndex - startTimeIndex + 1;
        forAll(stateInfo_["volVectorStates"], idxI)
        {
            const word stateName = stateInfo_["volVectorStates"][idxI];
            const volVectorField& state = meshPtr_->thisDb().lookupObject<volVectorField>(stateName);
            forAll(meanVolVectorStates_[idxI], cellI)
            {
                meanVolVectorStates_[idxI][cellI] = (meanVolVectorStates_[idxI][cellI] * (n - 1) + state[cellI]) / n;
            }
        }

        forAll(stateInfo_["volScalarStates"], idxI)
        {
            const word stateName = stateInfo_["volScalarStates"][idxI];
            const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);

            forAll(meanVolScalarStates_[idxI], cellI)
            {
                meanVolScalarStates_[idxI][cellI] = (meanVolScalarStates_[idxI][cellI] * (n - 1) + state[cellI]) / n;
            }
        }

        forAll(stateInfo_["modelStates"], idxI)
        {
            const word stateName = stateInfo_["modelStates"][idxI];
            const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);

            forAll(meanModelStates_[idxI], cellI)
            {
                meanModelStates_[idxI][cellI] = (meanModelStates_[idxI][cellI] * (n - 1) + state[cellI]) / n;
            }
        }

        forAll(stateInfo_["surfaceScalarStates"], idxI)
        {
            const word stateName = stateInfo_["surfaceScalarStates"][idxI];
            const surfaceScalarField& state = meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName);

            forAll(meanSurfaceScalarStates_[idxI], faceI)
            {
                meanSurfaceScalarStates_[idxI][faceI] = (meanSurfaceScalarStates_[idxI][faceI] * (n - 1) + state[faceI]) / n;
            }

            forAll(meanSurfaceScalarStates_[idxI].boundaryField(), patchI)
            {
                forAll(meanSurfaceScalarStates_[idxI].boundaryField()[patchI], faceI)
                {
                    scalar val = meanSurfaceScalarStates_[idxI].boundaryField()[patchI][faceI];
                    scalar val1 = state.boundaryField()[patchI][faceI];
                    meanSurfaceScalarStates_[idxI].boundaryFieldRef()[patchI][faceI] = (val * (n - 1) + val1) / n;
                }
            }
        }

        // if we have caluclate mean states, i.e., timeIndex >= startTimeIndex, set meanStatesCalculated_ = 1
        // this is to avoid setting a large startTime but the flow somehow converge before the startTime is
        // triggered. In this case, the meanStates is never calculated and will return the wrong results
        // if meanStatesCalculated_ == 0, we will not assignMeanStatesToStates at the end of the primal
        // check DASolver::assignMeanStatesToStates
        meanStatesCalculated_ = 1;
    }
}

scalar DASolver::getTimeInstanceFunction(
    const label instanceI,
    const word functionName)
{
    /*
    Description:
        Return the value of objective function at the given time instance and name
    */

    return functionsAllInstances_[instanceI].getScalar(functionName);
}

void DASolver::setPrimalBoundaryConditions(const label printInfo)
{
    /*
    Description:
        Update the state boundary conditions based on the ones defined in primalBC
    */

    // first check if we need to change the boundary conditions based on
    // the primalBC dict in DAOption. NOTE: this will overwrite whatever
    // boundary conditions defined in the "0" folder
    dictionary bcDict = daOptionPtr_->getAllOptions().subDict("primalBC");
    if (bcDict.toc().size() != 0)
    {
        if (printInfo)
        {
            Info << "Setting up primal boundary conditions based on pyOptions: " << endl;
        }
        daFieldPtr_->setPrimalBoundaryConditions(printInfo);
    }
}

label DASolver::runFPAdj(
    const Vec xvVec,
    const Vec wVec,
    Vec dFdW,
    Vec psi)
{
    /*
    Description:
        Solve the adjoint using the fixed-point iteration approach
    */

    FatalErrorIn("DASolver::runFPAdj")
        << "Child class not implemented!"
        << abort(FatalError);

    return 1;
}

void DASolver::getInitStateVals(HashTable<scalar>& initState)
{
    /*
    Description:
        Get the initial state values from the field's 1st index
    */

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const volVectorField& state = meshPtr_->thisDb().lookupObject<volVectorField>(stateName);
        for (label i = 0; i < 3; i++)
        {
            initState.set(stateName + Foam::name(i), state[0][i]);
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        initState.set(stateName, state[0]);
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        initState.set(stateName, state[0]);
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const surfaceScalarField& state = meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName);
        initState.set(stateName, state[0]);
    }

    Info << "initState: " << initState << endl;
}

void DASolver::resetStateVals()
{
    /*
    Description:
        Reset the initial state values using DASolver::initStateVals_
    */

    Info << "Resetting state to its initial values" << endl;

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(stateName));
        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                state[cellI][i] = initStateVals_[stateName + Foam::name(i)];
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));
        forAll(state, cellI)
        {
            state[cellI] = initStateVals_[stateName];
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));
        forAll(state, cellI)
        {
            state[cellI] = initStateVals_[stateName];
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));
        forAll(state, faceI)
        {
            state[faceI] = initStateVals_[stateName];
        }
    }
}

label DASolver::validateStates()
{
    /*
    Description:
        check if the state variables have valid values, if yes, return 1
    */

    label fail = 0;

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const volVectorField& state = meshPtr_->thisDb().lookupObject<volVectorField>(stateName);
        fail += this->validateVectorField(state);
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        fail += this->validateField(state);
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        fail += this->validateField(state);
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const surfaceScalarField& state = meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName);
        fail += this->validateField(state);
    }

    reduce(fail, sumOp<label>());

    if (fail > 0)
    {
        Info << "*************************************** Warning! ***************************************" << endl;
        Info << "Invalid values found. Return primal failure and reset the states to their initial values" << endl;
        Info << "*************************************** Warning! ***************************************" << endl;
        this->resetStateVals();
        return 1;
    }
    else
    {
        return 0;
    }
}

void DASolver::writeSensMapSurface(
    const word name,
    const double* dFdXs,
    const double* Xs,
    const label size,
    const double timeName)
{
    /*
    Description:
        write the sensitivity map for all wall surfaces
    
    Inputs:
        name: the name of the sens map written tot the disk

        dFdXs: the derivative of an objective function wrt the surface point coordinates (flatten)

        Xs: flatten surface point coordinates

        size: the size of the dFdXs array

        timeName: the name of time folder to write sens
    */

    volVectorField sens(
        IOobject(
            name,
            meshPtr_->time().timeName(),
            meshPtr_(),
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        meshPtr_(),
        dimensionedVector(name, dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::zero),
        "fixedValue");

    forAll(sens.boundaryField(), patchI)
    {
        if (meshPtr_->boundaryMesh()[patchI].type() == "wall")
        {
            forAll(sens.boundaryField()[patchI], faceI)
            {
                sens.boundaryFieldRef()[patchI][faceI] = vector::zero;
            }
        }
    }

    label nSurfPoints = round(size / 3);

    List<point> surfCoords(nSurfPoints);
    List<vector> surfdFdXs(nSurfPoints);
    label counterI = 0;
    forAll(surfCoords, pointI)
    {
        for (label i = 0; i < 3; i++)
        {
            surfCoords[pointI][i] = Xs[counterI];
            surfdFdXs[pointI][i] = dFdXs[counterI];
            counterI++;
        }
    }

    pointField meshPoints = meshPtr_->points();

    scalar distance;
    scalar minDistanceNorm = 0.0;
    // loop over all boundary points
    forAll(meshPtr_->boundaryMesh(), patchI)
    {
        if (meshPtr_->boundaryMesh()[patchI].type() == "wall")
        {
            forAll(meshPtr_->boundaryMesh()[patchI], faceI)
            {
                forAll(meshPtr_->boundaryMesh()[patchI][faceI], pointI)
                {
                    // for each point on the boundary, search for the closest point from surfCoords
                    label faceIPointIndexI = meshPtr_->boundaryMesh()[patchI][faceI][pointI];
                    scalar minDistance = 9999999;
                    label minPointJ = -1;
                    forAll(surfCoords, pointJ)
                    {
                        distance = mag(surfCoords[pointJ] - meshPoints[faceIPointIndexI]);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            minPointJ = pointJ;
                        }
                    }

                    minDistanceNorm += minDistance * minDistance;

                    // add the sensitivty to the corresponding faces
                    sens.boundaryFieldRef()[patchI][faceI] += surfdFdXs[minPointJ];
                }
            }
        }
    }

    minDistanceNorm = sqrt(minDistanceNorm);

    // finally, we need to divide the sens by the number of points for each face
    forAll(sens.boundaryField(), patchI)
    {
        if (meshPtr_->boundaryMesh()[patchI].type() == "wall")
        {
            forAll(sens.boundaryField()[patchI], faceI)
            {
                sens.boundaryFieldRef()[patchI][faceI] /= sens.boundaryFieldRef()[patchI][faceI].size();
            }
        }
    }

    Info << "Writing sensitivty map for " << name << endl;
    Info << "minDistance Norm: " << minDistanceNorm << endl;

    // switch to timeName and write sens, then reset the time
    scalar t = runTimePtr_->timeOutputValue();
    label timeIndex = runTimePtr_->timeIndex();

    runTimePtr_->setTime(timeName, timeIndex);
    sens.write();
    runTimePtr_->setTime(t, timeIndex);
}

void DASolver::writeSensMapField(
    const word name,
    const double* dFdField,
    const word fieldType,
    const double timeName)
{
    /*
    Description:
        write the sensitivity map for the entire field
    
    Inputs:
        name: the name of the sens map written tot the disk

        dFdField: the derivative of an objective function wrt the field variable

        fieldType: scalar or vector for the field variable

        timeName: the name of time folder to write sens
    */

    if (fieldType == "scalar")
    {
        volScalarField sens(
            IOobject(
                name,
                meshPtr_->time().timeName(),
                meshPtr_(),
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            meshPtr_(),
            dimensionedScalar(name, dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
            "fixedValue");

        forAll(sens, cellI)
        {
            sens[cellI] = dFdField[cellI];
        }

        Info << "Writing sensitivty map for " << name << endl;

        // switch to timeName and write sens, then reset the time
        scalar t = runTimePtr_->timeOutputValue();
        label timeIndex = runTimePtr_->timeIndex();

        runTimePtr_->setTime(timeName, timeIndex);
        sens.correctBoundaryConditions();
        sens.write();

        runTimePtr_->setTime(t, timeIndex);
    }
    else if (fieldType == "vector")
    {
        volVectorField sens(
            IOobject(
                name,
                meshPtr_->time().timeName(),
                meshPtr_(),
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            meshPtr_(),
            dimensionedVector(name, dimensionSet(0, 0, 0, 0, 0, 0, 0), vector::zero),
            "fixedValue");

        label counterI = 0;
        forAll(sens, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                sens[cellI][i] = dFdField[counterI];
                counterI++;
            }
        }

        Info << "Writing sensitivty map for " << name << endl;

        // switch to timeName and write sens, then reset the time
        scalar t = runTimePtr_->timeOutputValue();
        label timeIndex = runTimePtr_->timeIndex();

        runTimePtr_->setTime(timeName, timeIndex);
        sens.correctBoundaryConditions();
        sens.write();

        runTimePtr_->setTime(t, timeIndex);
    }
    else
    {
        FatalErrorIn("DASolver::writeSensMapField")
            << "fieldType not supported!"
            << abort(FatalError);
    }
}

void DASolver::writeAdjointFields(
    const word function,
    const double writeTime,
    const double* psi)
{
    /*
    Description:
        write the adjoint variables to the disk as OpenFOAM variables so they can be viewed
        in ParaView
    
    Inputs:
        writeTime: solution time the fields will be saved to
        psi: the adjoint vector array, computed in the Python layer
    */

    runTimePtr_->setTime(writeTime, 0);

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const volVectorField& state = meshPtr_->thisDb().lookupObject<volVectorField>(stateName);
        word varName = "adjoint_" + function + "_" + stateName;
        volVectorField adjointVar(varName, state);
        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI, i);
                adjointVar[cellI][i] = psi[localIdx];
            }
        }
        adjointVar.correctBoundaryConditions();
        adjointVar.write();
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        word varName = "adjoint_" + function + "_" + stateName;
        volScalarField adjointVar(varName, state);
        forAll(state, cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            adjointVar[cellI] = psi[localIdx];
        }
        adjointVar.correctBoundaryConditions();
        adjointVar.write();
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        word varName = "adjoint_" + function + "_" + stateName;
        volScalarField adjointVar(varName, state);
        forAll(state, cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            adjointVar[cellI] = psi[localIdx];
        }
        adjointVar.correctBoundaryConditions();
        adjointVar.write();
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        const surfaceScalarField& state = meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName);
        word varName = "adjoint_" + function + "_" + stateName;
        surfaceScalarField adjointVar(varName, state);

        forAll(meshPtr_->faces(), faceI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, faceI);

            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                adjointVar[faceI] = psi[localIdx];
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                adjointVar.boundaryFieldRef()[patchIdx][faceIdx] = psi[localIdx];
            }
        }
        adjointVar.write();
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
