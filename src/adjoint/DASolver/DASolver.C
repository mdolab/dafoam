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
      daGlobalVarPtr_(nullptr),
      points0Ptr_(nullptr)
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

    // if the dynamic mesh is used, set moving to true here
    dictionary allOptions = daOptionPtr_->getAllOptions();
    if (allOptions.subDict("dynamicMesh").getLabel("active"))
    {
        meshPtr_->moving(true);
        // if we have volCoord as the input, there is no need to save the initial
        // mesh because OpenMDAO will assign a new coordinates (set_solver_input) to OF
        // before each primal run. however, if we do not have the volCoord as the input,
        // we need to save the initial mesh, such that we can reset the OF's fvMesh
        // info for each dynamicMesh primal run.
        if (!this->hasVolCoordInput())
        {
            points0Ptr_.reset(new pointField(meshPtr_->points()));
        }
        // we need to initialize the dynamic mesh by calling move points to create V0, V00 etc
        // this is usually done automatically in primal solution, but for the AD version,
        // we never cal primal so we need to manually initialize V0 etc here
        this->initDynamicMesh();
    }
    else
    {
        meshPtr_->moving(false);
    }

    primalMinResTol_ = daOptionPtr_->getOption<scalar>("primalMinResTol");
    primalMinIters_ = daOptionPtr_->getOption<label>("primalMinIters");
    printInterval_ = daOptionPtr_->getOption<label>("printInterval");
    printIntervalUnsteady_ = daOptionPtr_->getOption<label>("printIntervalUnsteady");

    // if inputInto has unsteadyField, we need to initial GlobalVar::inputFieldUnsteady here
    this->initInputFieldUnsteady();

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

    // check exit condition, we need to satisfy both the residual and function std condition
    if (daGlobalVarPtr_->primalMaxRes < primalMinResTol_ && runTime.timeIndex() > primalMinIters_)
    {
        Info << "Time = " << t << endl;

        Info << "Minimal residual " << daGlobalVarPtr_->primalMaxRes << " satisfied the prescribed tolerance " << primalMinResTol_ << endl
             << endl;

        this->calcAllFunctions(1);
        runTime.writeNow();
        prevPrimalSolTime_ = t;
        funcObj.end();
        daRegressionPtr_->writeFeatures();
        primalFinalTimeIndex_ = runTime.timeIndex();
        return 0;
    }
    else if (t > endTime - 0.5 * deltaT)
    {
        prevPrimalSolTime_ = t;
        funcObj.end();
        daRegressionPtr_->writeFeatures();
        primalFinalTimeIndex_ = runTime.timeIndex();
        return 0;
    }
    else
    {
        ++runTime;
        // initialize primalMaxRes with a small value for this iteration
        daGlobalVarPtr_->primalMaxRes = -1e10;
        printToScreen_ = this->isPrintTime(runTime, printInterval_);
        return 1;
    }
}

void DASolver::calcAllFunctions(label print)
{
    /*
    Description:
        Calculate the values of all objective functions and print them to screen
        NOTE: we need to call DASolver::setDAFunctionList before calling this function!
    */

    if (daFunctionPtrList_.size() == 0)
    {
        // if users do not set function, we can just skip this call
        if (daOptionPtr_->getAllOptions().subDict("function").toc().size() != 0)
        {
            FatalErrorIn("printAllFunctions") << "daFunctionPtrList_.size() ==0... "
                                              << "Forgot to call setDAFunctionList?"
                                              << abort(FatalError);
        }
        else
        {
            return;
        }
    }

    label timeIndex = runTimePtr_->timeIndex();
    label listIndex = timeIndex - 1;

    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName = daFunction.getFunctionName();
        word timeOpType = daFunction.getFunctionTimeOp();
        scalar functionVal = daFunction.calcFunction();
        functionTimeSteps_[idxI][listIndex] = functionVal;

        if (print)
        {
            const dictionary& functionDict = daOptionPtr_->getAllOptions().subDict("function").subDict(functionName);
            label timeOpStartIndex = functionDict.lookupOrDefault<label>("timeOpStartIndex", 0);
            scalar timeOpVal = 0.0;
            // if timeOpStartIndex > listIndex, timeOpVal is zero
            if (timeOpStartIndex <= listIndex)
            {
                timeOpVal = daTimeOpPtrList_[idxI].compute(
                    functionTimeSteps_[idxI], timeOpStartIndex, listIndex);
            }

            Info << functionName
                 << ": " << functionVal
                 << " " << timeOpType << ": " << timeOpVal;
#ifdef CODI_ADF
            Info << " ADF-Deriv: " << timeOpVal.getGradient();
#endif
            Info << endl;
        }
    }
}

double DASolver::getTimeOpFuncVal(const word functionName)
{
    // return the function value based on timeOp
    label listFinalIndex = primalFinalTimeIndex_ - 1;
    scalar funcVal = 0.0;
    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName1 = daFunction.getFunctionName();

        if (functionName1 == functionName)
        {
            const dictionary& functionDict = daOptionPtr_->getAllOptions().subDict("function").subDict(functionName);
            label timeOpStartIndex = functionDict.lookupOrDefault<label>("timeOpStartIndex", 0);
            // if timeOpStartIndex should not be larger than listFinalIndex
            if (timeOpStartIndex <= listFinalIndex)
            {
                funcVal = daTimeOpPtrList_[idxI].compute(
                    functionTimeSteps_[idxI], timeOpStartIndex, listFinalIndex);
            }
            else
            {
                FatalErrorIn("") << "timeOpStartIndex can not be larger than listFinalIndex!"
                                 << abort(FatalError);
            }
        }
    }
#ifdef CODI_ADF
    return funcVal.getGradient();
#endif

#ifdef CODI_ADR
    return funcVal.getValue();
#endif

#ifdef CODI_NO_AD
    return funcVal;
#endif
}

/// get the scaling factor for dF/d? derivative computation
scalar DASolver::getdFScaling(
    const word functionName,
    const label timeIdx)
{
    scalar scaling = 0.0;
    label listFinalIndex = primalFinalTimeIndex_ - 1;
    forAll(daFunctionPtrList_, idxI)
    {
        DAFunction& daFunction = daFunctionPtrList_[idxI];
        word functionName1 = daFunction.getFunctionName();
        if (functionName1 == functionName)
        {
            const dictionary& functionDict = daOptionPtr_->getAllOptions().subDict("function").subDict(functionName);
            label timeOpStartIndex = functionDict.lookupOrDefault<label>("timeOpStartIndex", 0);
            // if timeIdx is outside of [timeOpStartIndex, listFinalIndex], dFScaling is zero
            if (timeIdx >= timeOpStartIndex && timeIdx <= listFinalIndex)
            {
                scaling = daTimeOpPtrList_[idxI].dFScaling(
                    functionTimeSteps_[idxI], timeOpStartIndex, listFinalIndex, timeIdx);
            }
            return scaling;
        }
    }
    FatalErrorIn("getdFScaling") << "functionName not found! "
                                 << abort(FatalError);
    return scaling;
}

void DASolver::setDAFunctionList()
{
    /*
    Description:
        Set up the objective function list such that we can call calcAllFunctions and calcFunction
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
                "timeOp": "final"
            },
            "func1":
            {
                "functionName": "force",
                "source": "patchToFace",
                "patches": ["wallsbump", "frontandback"],
                "scale": 0.5,
                "timeOp": "average"
            },
            "func2": 
            {
                "functionName": "force",
                "source": "patchToFace",
                "patches": ["walls", "wallsbump", "frontandback"],
                "scale": 1.0,
                "timeOp": "variance"
            },
        }
    */

    const dictionary& allOptions = daOptionPtr_->getAllOptions();

    const dictionary& functionDict = allOptions.subDict("function");

    // loop over all functions and calc the number of
    // DAFunction instances we need
    label nFunctions = 0;
    forAll(functionDict.toc(), idxI)
    {
        nFunctions++;
    }

    daFunctionPtrList_.setSize(nFunctions);
    daTimeOpPtrList_.setSize(nFunctions);

    // we need to repeat the loop to initialize the
    // DAFunction instances
    forAll(functionDict.toc(), idxI)
    {
        word functionName = functionDict.toc()[idxI];
        dictionary funcSubDict = functionDict.subDict(functionName);
        daFunctionPtrList_.set(
            idxI,
            DAFunction::New(
                meshPtr_(),
                daOptionPtr_(),
                daModelPtr_(),
                daIndexPtr_(),
                functionName)
                .ptr());

        // initialize DATimeOp pointer list
        word timeOp = daFunctionPtrList_[idxI].getFunctionTimeOp();
        daTimeOpPtrList_.set(
            idxI,
            DATimeOp::New(timeOp, funcSubDict).ptr());
    }

    // here we also initialize the functionTimeSteps lists
    scalar endTime = runTimePtr_->endTime().value();
    scalar deltaT = runTimePtr_->deltaT().value();
    label nSteps = round(endTime / deltaT);
    functionTimeSteps_.setSize(nFunctions);
    forAll(daFunctionPtrList_, idxI)
    {
        functionTimeSteps_[idxI].setSize(nSteps);
        forAll(functionTimeSteps_[idxI], idxJ)
        {
            functionTimeSteps_[idxI][idxJ] = 0.0;
        }
    }
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
        Info << "dRdWCon Created. " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

        // compute the coloring
        Info << "Calculating dRdW Coloring... " << meshPtr_->time().elapsedCpuTime() << " s" << endl;
        daJacCon.calcJacConColoring();
        Info << "Calculating dRdW Coloring... Completed! " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

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

    Info << " " << endl;

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

    Info << "Computing " << matName << " " << runTimePtr_->elapsedCpuTime() << " s" << endl;
    Info << "Initializing dRdWCon. " << runTimePtr_->elapsedCpuTime() << " s" << endl;

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
    Info << "dRdWCon Created. " << runTimePtr_->elapsedCpuTime() << " s" << endl;

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

void DASolver::getOFMeshPoints(double* points)
{
    // get the flatten mesh points coordinates
    pointField meshPoints(meshPtr_->points());
    label counterI = 0;
    forAll(meshPoints, pointI)
    {
        for (label i = 0; i < 3; i++)
        {
            assignValueCheckAD(points[counterI], meshPoints[pointI][i]);
            counterI++;
        }
    }
}

void DASolver::getOFField(
    const word fieldName,
    const word fieldType,
    double* fieldArray)
{
    /*
    Description:
        assign a OpenFoam layer field variable in mesh.Db() to field
    */

    if (fieldType == "scalar")
    {
        const volScalarField& field = meshPtr_->thisDb().lookupObject<volScalarField>(fieldName);
        forAll(field, cellI)
        {
            assignValueCheckAD(fieldArray[cellI], field[cellI]);
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
                assignValueCheckAD(fieldArray[localIdx], field[cellI][comp]);
                localIdx++;
            }
        }
    }
    else
    {
        FatalErrorIn("getField") << " fieldType not valid. Options: scalar or vector"
                                 << abort(FatalError);
    }
}

void DASolver::updateOFFields(const scalar* states)
{
    label printInfo = 0;
    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Updating the OpenFOAM field..." << endl;
        printInfo = 1;
    }
    daFieldPtr_->state2OFField(states);

    this->updateStateBoundaryConditions();
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

    PetscScalar* vecArray;
    const PetscScalar* vecArrayRead;
    // assign the variable in vecX as the residual gradient for reverse AD
    VecGetArrayRead(vecX, &vecArrayRead);
    ctx->assignVec2ResidualGradient(vecArrayRead);
    VecRestoreArrayRead(vecX, &vecArrayRead);
    // do the backward computation to propagate the derivatives to the states
    ctx->globalADTape_.evaluate();
    // assign the derivatives stored in the states to the vecY vector
    VecGetArray(vecY, &vecArray);
    ctx->assignStateGradient2Vec(vecArray);
    // NOTE: we need to normalize the vecY vector.
    ctx->normalizeGradientVec(vecArray);
    VecRestoreArray(vecY, &vecArray);
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

/// get whether the output is distributed among processors
void DASolver::calcOutput(
    const word outputName,
    const word outputType,
    double* output)
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

    label outputSize = daOutput->size();

    scalarList outputList(outputSize, 0.0);

    daOutput->run(outputList);

    forAll(outputList, idxI)
    {
        assignValueCheckAD(output[idxI], outputList[idxI]);
    }
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
    const double* input,
    const word outputName,
    const word outputType,
    const double* seed,
    double* product)
{
#ifdef CODI_ADR
    /*
    Description:
        Calculate the Jacobian-matrix-transposed and vector product for [dOutput/dInput]^T * psi
    
    Input:
        inputName: name of the input. This is usually defined in inputInfo

        inputType: type of the input. This should be consistent with the child class type in DAInput

        input: the actual value of the input array

        outputName: name of the output.

        outputType: type of the output. This should be consistent with the child class type in DAOutput

        seed: the seed array
    
    Output:
        product: the mat-vec product array
    */

    Info << "Computing d[" << outputName << "]/d[" << inputName << "]^T * psi " << runTimePtr_->elapsedCpuTime() << " s" << endl;

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

    label inputSize = daInput->size();
    label outputSize = daOutput->size();

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

    wordList patches;
    wordList components;
    dictionary outputInfo = daOptionPtr_->getAllOptions().subDict("outputInfo");
    forAll(outputInfo.toc(), idxI)
    {
        word outputName = outputInfo.toc()[idxI];
        outputInfo.subDict(outputName).readEntry("components", components);
        if (components.found("thermalCoupling"))
        {
            outputInfo.subDict(outputName).readEntry("patches", patches);
            break;
        }
    }
    // NOTE: always sort the patch because the order of the patch element matters in CHT coupling
    sort(patches);

    // ******** first loop
    label counterFaceI = 0;
    forAll(patches, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patches[cI]);
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
    forAll(patches, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patches[cI]);
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

void DASolver::calcdRdWOldTPsiAD(
    const label oldTimeLevel,
    const double* psi,
    double* dRdWOldTPsi)
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

        psi: the array to multiply dRdW0^T
    
    Output:
        dRdWOldTPsi: the matrix-vector products dRdWOld^T * Psi
    */

    Info << "Computing [dRdWOld]^T * psi: level " << oldTimeLevel << ". " << runTimePtr_->elapsedCpuTime() << " s" << endl;

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

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    // **********************************************************************************************
    // clean up OF vars's AD seeds by deactivating the inputs and call the forward func one more time
    // **********************************************************************************************
    this->deactivateStateVariableInput4AD(oldTimeLevel);
    this->updateStateBoundaryConditions();
    this->calcResiduals();
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

void DASolver::normalizeGradientVec(double* vecArray)
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

#endif
}

void DASolver::assignVec2ResidualGradient(const double* vecArray)
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

#endif
}

void DASolver::assignStateGradient2Vec(
    double* vecArray,
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

#endif
}

label DASolver::checkPrimalFailure()
{
    /*
    Description:
        Check whether the primal solution fails. Yes: return 1. No: return 0
        - Check whether the min residual in primal satisfy the prescribed tolerance
        - Check whether the regression model computation fails
    */

    // when checking the tolerance, we relax the criteria by tolMax

    if (regModelFail_ != 0)
    {
        Info << "Regression model computation has invalid values. Primal solution failed!" << endl;
        return 1;
    }

    scalar tolMax = daOptionPtr_->getOption<scalar>("primalMinResTolDiff");
    if (daGlobalVarPtr_->primalMaxRes / primalMinResTol_ > tolMax)
    {
        Info << "********************************************" << endl;
        Info << "Primal min residual " << daGlobalVarPtr_->primalMaxRes << endl
             << "did not satisfy the prescribed tolerance "
             << primalMinResTol_ << endl;
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

    label nBCCalls = daOptionPtr_->getOption<label>("maxCorrectBCCalls");

    for (label i = 0; i < nBCCalls; i++)
    {
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
    }

    // if we have regression models, we also need to update them because they will update the fields
    // NOTE we should have done it in DAInput, no need to call it again.
    this->regressionModelCompute();

    // we also need to update DAGlobaVar::inputUnsteadyField if unsteadyField is used in inputInfo
    this->updateInputFieldUnsteady();
}

void DASolver::calcPCMatWithFvMatrix(Mat PCMat, const label turbOnly)
{
    /*
    Description:
        calculate the PC mat using fvMatrix. Here we only calculate the block diagonal components, 
        e.g., dR_U/dU, dR_p/dp, etc.
    */

    //DAUtility::writeMatrixASCII(PCMat, "MatOrig");

    // MatZeroEntries(PCMat);

    // non turbulence variables
    if (!turbOnly)
    {
        daResidualPtr_->calcPCMatWithFvMatrix(PCMat);
    }

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

void DASolver::readMeshPoints(const scalar timeVal)
{
    /*
    Description:
        read the mesh points from the disk and run movePoints to deform the mesh
    
    Inputs:
        
        timeVal: Which time to read, i.e., time.timeName()
    */

    pointIOField readPoints(
        IOobject(
            "points",
            Foam::name(timeVal),
            "polyMesh",
            meshPtr_(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE));

    meshPtr_->movePoints(readPoints);
}

void DASolver::writeMeshPoints(const double* points, const scalar timeVal)
{
    /*
    Description:
        write the mesh points to the disk for the given timeVal
    
    Inputs:
        
        timeVal: Which time to read, i.e., time.timeName()
    */

    pointIOField writePoints(
        IOobject(
            "points",
            Foam::name(timeVal),
            "polyMesh",
            runTimePtr_(),
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false),
        meshPtr_->points());

    //pointIOField writePoints = meshPtr_->points();

    label counterI = 0;
    forAll(writePoints, pointI)
    {
        for (label i = 0; i < 3; i++)
        {
            writePoints[pointI][i] = points[counterI];
            counterI++;
        }
    }
    // time index is not important here. Users need to reset the time after
    // calling this function
    runTimePtr_->setTime(timeVal, 0);
    //Info << "writing points to " << writePoints.path() << endl;
    writePoints.write();
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

label DASolver::solveAdjointFP(
    Vec dFdW,
    Vec psi)
{
    /*
    Description:
        Solve the adjoint using the fixed-point iteration approach
    */

    FatalErrorIn("DASolver::solveAdjointFP")
        << "Child class not implemented!"
        << abort(FatalError);

    return 1;
}

void DASolver::getInitStateVals(HashTable<scalar>& initState)
{
    /*
    Description:
        Get the initial state values from the field's average value
    */

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        const volVectorField& state = meshPtr_->thisDb().lookupObject<volVectorField>(stateName);
        scalarList avgState(3, 0.0);
        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                avgState[i] += state[cellI][i] / daIndexPtr_->nGlobalCells;
            }
        }
        reduce(avgState[0], sumOp<scalar>());
        reduce(avgState[1], sumOp<scalar>());
        reduce(avgState[2], sumOp<scalar>());

        for (label i = 0; i < 3; i++)
        {
            initState.set(stateName + Foam::name(i), avgState[i]);
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        scalar avgState = 0.0;
        forAll(state, cellI)
        {
            avgState += state[cellI];
        }
        avgState /= daIndexPtr_->nGlobalCells;
        reduce(avgState, sumOp<scalar>());

        initState.set(stateName, avgState);
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        const volScalarField& state = meshPtr_->thisDb().lookupObject<volScalarField>(stateName);
        scalar avgState = 0.0;
        forAll(state, cellI)
        {
            avgState += state[cellI];
        }
        avgState /= daIndexPtr_->nGlobalCells;
        reduce(avgState, sumOp<scalar>());

        initState.set(stateName, avgState);
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        // const surfaceScalarField& state = meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName);
        // we can reset the flux to zeros
        initState.set(stateName, 0.0);
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
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));
        forAll(state, cellI)
        {
            state[cellI] = initStateVals_[stateName];
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(stateName));
        forAll(state, cellI)
        {
            state[cellI] = initStateVals_[stateName];
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));
        forAll(state, faceI)
        {
            state[faceI] = initStateVals_[stateName];
        }
        forAll(state.boundaryField(), patchI)
        {
            forAll(state.boundaryField()[patchI], faceI)
            {
                state.boundaryFieldRef()[patchI][faceI] = initStateVals_[stateName];
            }
        }
        // if this is a phi var, we inerpolate U to get phi
        if (stateName == "phi")
        {
            const volVectorField& U = meshPtr_->thisDb().lookupObject<volVectorField>("U");
            state = linearInterpolate(U) & meshPtr_->Sf();
        }
    }

    // we need to also update the BC and update all the intermediate variables
    this->updateStateBoundaryConditions();
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

    Info << "Writting adjoint fields " << endl;

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

label DASolver::hasVolCoordInput()
{
    // whether the volCoord input is defined
    dictionary allOptions = daOptionPtr_->getAllOptions();
    label hasVolCoordInput = 0;
    forAll(allOptions.subDict("inputInfo").toc(), idxI)
    {
        word inputName = allOptions.subDict("inputInfo").toc()[idxI];
        word inputType = allOptions.subDict("inputInfo").subDict(inputName).getWord("type");
        if (inputType == "volCoord")
        {
            hasVolCoordInput = 1;
        }
    }
    return hasVolCoordInput;
}

void DASolver::initDynamicMesh()
{
    /*
    Description:
        Resetting internal info in fvMesh, which is needed for multiple primal runs
        For example, after one primal run, the mesh points is the final time t = N*dt
        If we directly start the primal again, the meshPhi from t=dt will use the mesh
        info from t=N*dt as the previous time step, this is wrong. Similary things happen
        for V0, V00 calculation. Therefore, to fix this problem, we need to call 
        movePoints multiple times to rest all the internal info in fvMesh to t=0
    */
    label ddtSchemeOrder = this->getDdtSchemeOrder();
    label hasVolCoordInput = this->hasVolCoordInput();

    if (hasVolCoordInput)
    {
        // if we have volCoord as the input, the fvMesh is assigned by external components
        // such as IDWarap, and it is should be at t=0 already, so
        // we just call movePoints multiple times (depending on ddtSchemeOrder)
        // NOTE: we just use the i index starting from 100 to avoid conflict with
        // the starting timeIndex=1 because if the timeIndex equals the fvMesh's
        // internal timeIndex counter, it will not reset the internal mesh info
        pointField points0 = meshPtr_->points();
        for (label i = 100; i <= 100 + ddtSchemeOrder; i++)
        {
            runTimePtr_->setTime(0.0, i);
            meshPtr_->movePoints(points0);
        }
    }
    else
    {
        // if there is no volCoord as the input, the fvMesh is at t=N*dt
        // in this case, we need to use the initial mesh save in points0Ptr_
        for (label i = 100; i <= 100 + ddtSchemeOrder; i++)
        {
            runTimePtr_->setTime(0.0, i);
            meshPtr_->movePoints(points0Ptr_());
        }
    }

    // finally, reset the time to 0
    runTimePtr_->setTime(0.0, 0);
}

void DASolver::meanStatesToStates()
{
    // assign the mean states values to states
    forAll(stateInfo_["volVectorStates"], idxI)
    {
        const word stateName = stateInfo_["volVectorStates"][idxI];
        volVectorField& state = const_cast<volVectorField&>(
            meshPtr_->thisDb().lookupObject<volVectorField>(stateName));
        word meanStateName = stateName + "Mean";
        const volVectorField& stateMean =
            meshPtr_->thisDb().lookupObject<volVectorField>(meanStateName);
        forAll(state, cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                state[cellI][i] = stateMean[cellI][i];
            }
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));
        word meanStateName = stateName + "Mean";
        const volScalarField& stateMean =
            meshPtr_->thisDb().lookupObject<volScalarField>(meanStateName);
        forAll(state, cellI)
        {
            state[cellI] = stateMean[cellI];
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        volScalarField& state = const_cast<volScalarField&>(
            meshPtr_->thisDb().lookupObject<volScalarField>(stateName));
        word meanStateName = stateName + "Mean";
        const volScalarField& stateMean =
            meshPtr_->thisDb().lookupObject<volScalarField>(meanStateName);
        forAll(state, cellI)
        {
            state[cellI] = stateMean[cellI];
        }
        state.correctBoundaryConditions();
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        surfaceScalarField& state = const_cast<surfaceScalarField&>(
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(stateName));
        word meanStateName = stateName + "Mean";
        const surfaceScalarField& stateMean =
            meshPtr_->thisDb().lookupObject<surfaceScalarField>(meanStateName);

        forAll(meshPtr_->faces(), faceI)
        {
            if (faceI < daIndexPtr_->nLocalInternalFaces)
            {
                state[faceI] = stateMean[faceI];
            }
            else
            {
                label relIdx = faceI - daIndexPtr_->nLocalInternalFaces;
                label patchIdx = daIndexPtr_->bFacePatchI[relIdx];
                label faceIdx = daIndexPtr_->bFaceFaceI[relIdx];
                state.boundaryFieldRef()[patchIdx][faceIdx] = stateMean.boundaryField()[patchIdx][faceIdx];
            }
        }
    }

    // we need to also update the BC for other variables
    this->updateStateBoundaryConditions();
}

void DASolver::initInputFieldUnsteady()
{
    /*
    Description:
        initialize inputFieldUnsteady from the GlobalVar class
    */

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(meshPtr_->thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));

    dictionary inputInfoDict = daOptionPtr_->getAllOptions().subDict("inputInfo");
    forAll(inputInfoDict.toc(), idxI)
    {
        word inputName = inputInfoDict.toc()[idxI];
        word inputType = inputInfoDict.subDict(inputName).getWord("type");
        if (inputType == "fieldUnsteady")
        {
            label stepInterval = inputInfoDict.subDict(inputName).getLabel("stepInterval");
            scalar endTime = meshPtr_->time().endTime().value();
            scalar deltaT = meshPtr_->time().deltaT().value();
            label nSteps = round(endTime / deltaT);
            word interpMethod = inputInfoDict.subDict(inputName).getWord("interpolationMethod");
            label nParameters = -1;
            if (interpMethod == "linear")
            {
                nParameters = nSteps / stepInterval + 1;
            }
            else if (interpMethod == "rbf")
            {
                nParameters = 2 * (nSteps / stepInterval + 1);
            }

            // NOTE: inputFieldUnsteady is alway local!
            scalarList initVal(nParameters * meshPtr_->nCells(), 0.0);
            globalVar.inputFieldUnsteady.set(inputName, initVal);
        }
    }
}

void DASolver::updateInputFieldUnsteady()
{
    /*
    Description:
        Assign the inputFieldUnsteady values to the OF field vars

        For linear interpolation, the filed u are saved in this format

        ------ t = 0 ------|---- t=1interval ---|---- t=2interval ---
        u1, u2, u3, ... un | u1, u2, u3, ... un | u1, u2, u3, ... un

        For rbf interpolation, the data are saved in this format, here w and s are the parameters for rbf

       ------ t = 0 ------|---- t=1interval ---|---- t=2interval ---|------ t = 0 ------|---- t=1interval ---|---- t=2interval --
       w1, w2, w3, ... wn | w1, w2, w3, ... wn | w1, w2, w3, ... wn |s1, s2, s3, ... sn |s1, s2, s3, ... sn  | s1, s2, s3, ... sn|

    */

    DAGlobalVar& globalVar =
        const_cast<DAGlobalVar&>(meshPtr_->thisDb().lookupObject<DAGlobalVar>("DAGlobalVar"));

    if (globalVar.inputFieldUnsteady.size() == 0)
    {
        return;
    }

    forAll(globalVar.inputFieldUnsteady.toc(), idxI)
    {
        word inputName = globalVar.inputFieldUnsteady.toc()[idxI];
        const dictionary& subDict = daOptionPtr_->getAllOptions().subDict("inputInfo").subDict(inputName);
        word fieldName = subDict.getWord("fieldName");
        word fieldType = subDict.getWord("fieldType");
        label stepInterval = subDict.getLabel("stepInterval");
        word interpMethod = subDict.getWord("interpolationMethod");

        label timeIndex = runTimePtr_->timeIndex();

        if (fieldType == "scalar")
        {
            volScalarField& field =
                const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>(fieldName));

            // linear interpolation
            if (interpMethod == "linear")
            {
                label timeR = timeIndex % stepInterval;
                label timeI = timeIndex / stepInterval;
                // set the initial index for the counter
                label counterI = timeI * meshPtr_->nCells();
                label deltaI = meshPtr_->nCells();
                forAll(field, cellI)
                {
                    scalar val1 = globalVar.inputFieldUnsteady[inputName][counterI];
                    if (timeR == 0)
                    {
                        // this should be the anchor field per stepInterval, no need to interpolate.
                        field[cellI] = val1;
                    }
                    else
                    {
                        // we interpolate using counterI and counterI+deltaI
                        label counterINextField = counterI + deltaI;
                        scalar val2 = globalVar.inputFieldUnsteady[inputName][counterINextField];
                        field[cellI] = val1 + (val2 - val1) * timeR / stepInterval;
                    }
                    counterI++;
                }
            }
            else if (interpMethod == "rbf")
            {
                scalar offset = subDict.getScalar("offset");
                scalar endTime = meshPtr_->time().endTime().value();
                scalar deltaT = meshPtr_->time().deltaT().value();
                label nSteps = round(endTime / deltaT);
                label nFields = nSteps / stepInterval + 1;

                forAll(field, cellI)
                {
                    field[cellI] = offset;
                }
                // rbf interpolation y = f(t)
                // y = sum_i ( w_i * exp(-s_i^2 * (t-c)^2 ) )
                // here c is the interpolation point from 0 to T with an interval of stepInterval
                label halfSize = globalVar.inputFieldUnsteady[inputName].size() / 2;
                label deltaI = nFields * meshPtr_->nCells();
                for (label i = 0; i < halfSize; i++)
                {
                    label cellI = i % meshPtr_->nCells();
                    scalar w = globalVar.inputFieldUnsteady[inputName][i];
                    scalar s = globalVar.inputFieldUnsteady[inputName][i + deltaI];
                    label interpTimeIndex = i / meshPtr_->nCells() * stepInterval;
                    scalar d = (timeIndex - interpTimeIndex);
                    field[cellI] += w * exp(-s * s * d * d);
                }
            }
            field.correctBoundaryConditions();
        }
        else
        {
            FatalErrorIn("") << "fieldType not valid" << exit(FatalError);
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
