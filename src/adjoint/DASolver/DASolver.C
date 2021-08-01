/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// initialize the static variable, which will be used in forward mode AD
// computation for AOA derivatives
scalar Foam::DAUtility::angleOfAttackRadForwardAD = -9999.0;

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
      daResidualPtr_(nullptr)
#ifdef CODI_AD_REVERSE
      ,
      globalADTape_(codi::RealReverse::getGlobalTape())
#endif
{
    // initialize fvMesh and Time object pointer
#include "setArgs.H"
#include "setRootCasePython.H"
#include "createTimePython.H"
#include "createMeshPython.H"
    Info << "Initializing mesh and runtime for DASolver" << endl;
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

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
        The loop method to increment the runtime. The reason we implent this is
        because the runTime.loop() and simple.loop() give us seg fault...
    */

    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaT().value();
    scalar t = runTime.timeOutputValue();
    scalar tol = daOptionPtr_->getOption<scalar>("primalMinResTol");

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

    // check exit condition
    if (primalMinRes_ < tol)
    {
        Info << "Time = " << t << endl;
        Info << "Minimal residual " << primalMinRes_ << " satisfied the prescribed tolerance " << tol << endl
             << endl;
        this->printAllObjFuncs();
        runTime.writeNow();
        prevPrimalSolTime_ = t;
        funcObj.end();
        return 0;
    }
    else if (t > endTime - 0.5 * deltaT)
    {
        prevPrimalSolTime_ = t;
        funcObj.end();
        return 0;
    }
    else
    {
        ++runTime;
        return 1;
    }
}

void DASolver::printAllObjFuncs()
{
    /*
    Description:
        Calculate the values of all objective functions and print them to screen
        NOTE: we need to call DASolver::setDAObjFuncList before calling this function!
    */

    if (daObjFuncPtrList_.size() == 0)
    {
        FatalErrorIn("printAllObjFuncs") << "daObjFuncPtrList_.size() ==0... "
                                         << "Forgot to call setDAObjFuncList?"
                                         << abort(FatalError);
    }

    forAll(daObjFuncPtrList_, idxI)
    {
        DAObjFunc& daObjFunc = daObjFuncPtrList_[idxI];
        word objFuncName = daObjFunc.getObjFuncName();
        scalar objFuncVal = daObjFunc.getObjFuncValue();
        Info << objFuncName
             << "-" << daObjFunc.getObjFuncPart()
             << "-" << daObjFunc.getObjFuncType()
             << ": " << objFuncVal;
#ifdef CODI_AD_FORWARD

        // if the forwardModeAD is active,, we need to get the total derivatives here
        if (daOptionPtr_->getAllOptions().subDict("useAD").getWord("mode") == "forward")
        {
            Info << " ForwardAD Deriv: " << objFuncVal.getGradient();

            // assign the forward mode AD derivative to forwardADDerivVal_
            // such that we can get this value later
            forwardADDerivVal_.set(objFuncName, objFuncVal.getGradient());
        }
#endif
        Info << endl;
    }
}

scalar DASolver::getObjFuncValue(const word objFuncName)
{
    /*
    Description:
        Return the value of the objective function.
        NOTE: we will sum up all the parts in objFuncName

    Input:
        objFuncName: the name of the objective function

    Output:
        objFuncValue: the value of the objective
    */

    if (daObjFuncPtrList_.size() == 0)
    {
        FatalErrorIn("printAllObjFuncs") << "daObjFuncPtrList_.size() ==0... "
                                         << "Forgot to call setDAObjFuncList?"
                                         << abort(FatalError);
    }

    scalar objFuncValue = 0.0;

    forAll(daObjFuncPtrList_, idxI)
    {
        DAObjFunc& daObjFunc = daObjFuncPtrList_[idxI];
        if (daObjFunc.getObjFuncName() == objFuncName)
        {
            objFuncValue += daObjFunc.getObjFuncValue();
        }
    }

    return objFuncValue;
}

void DASolver::setDAObjFuncList()
{
    /*
    Description:
        Set up the objective function list such that we can call printAllObjFuncs and getObjFuncValue
        NOTE: this function needs to be called before calculating any objective functions

    Example:
        A typical objFunc dictionary looks like this:
    
        "objFunc": 
        {
            "func1": 
            {
                "part1": 
                {
                    "objFuncName": "force",
                    "source": "patchToFace",
                    "patches": ["walls", "wallsbump"],
                    "scale": 0.5,
                    "addToAdjoint": True,
                },
                "part2": 
                {
                    "objFuncName": "force",
                    "source": "patchToFace",
                    "patches": ["wallsbump", "frontandback"],
                    "scale": 0.5,
                    "addToAdjoint": True,
                },
            },
            "func2": 
            {
                "part1": 
                {
                    "objFuncName": "force",
                    "source": "patchToFace",
                    "patches": ["walls", "wallsbump", "frontandback"],
                    "scale": 1.0,
                    "addToAdjoint": False,
                }
            },
        }
    */

    const dictionary& allOptions = daOptionPtr_->getAllOptions();

    dictionary objFuncDict = allOptions.subDict("objFunc");

    // loop over all objFuncs and parts and calc the number of
    // DAObjFunc instances we need
    label nObjFuncInstances = 0;
    forAll(objFuncDict.toc(), idxI)
    {
        word objFunI = objFuncDict.toc()[idxI];
        dictionary objFuncSubDict = objFuncDict.subDict(objFunI);
        forAll(objFuncSubDict.toc(), idxJ)
        {
            nObjFuncInstances++;
        }
    }

    daObjFuncPtrList_.setSize(nObjFuncInstances);

    // we need to repeat the loop to initialize the
    // DAObjFunc instances
    label objFuncInstanceI = 0;
    forAll(objFuncDict.toc(), idxI)
    {
        word objFunI = objFuncDict.toc()[idxI];
        dictionary objFuncSubDict = objFuncDict.subDict(objFunI);
        forAll(objFuncSubDict.toc(), idxJ)
        {

            word objPart = objFuncSubDict.toc()[idxJ];
            dictionary objFuncSubDictPart = objFuncSubDict.subDict(objPart);

            fvMesh& mesh = meshPtr_();

            daObjFuncPtrList_.set(
                objFuncInstanceI,
                DAObjFunc::New(
                    mesh,
                    daOptionPtr_(),
                    daModelPtr_(),
                    daIndexPtr_(),
                    daResidualPtr_(),
                    objFunI,
                    objPart,
                    objFuncSubDictPart)
                    .ptr());

            objFuncInstanceI++;
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

    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

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
    const Vec xvVec,
    const Vec wVec,
    const label isPC,
    Mat dRdWT)
{
    /*
    Description:
        This function computes partials derivatives dRdWT or dRdWTPC.
        PC means preconditioner matrix
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        isPC: isPC=1 computes dRdWTPC, isPC=0 computes dRdWT
    
    Output:
        dRdWT: the partial derivative matrix [dR/dW]^T
        NOTE: You need to call MatCreate for the dRdWT matrix before calling this function.
        No need to call MatSetSize etc because they will be done in this function
    */

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
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

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
    daJacCon->setupJacConPreallocation(options);

    // now we can initialize dRdWCon
    daJacCon->initializeJacCon(options);

    // setup dRdWCon
    daJacCon->setupJacCon(options);
    Info << "dRdWCon Created. " << runTimePtr_->elapsedClockTime() << " s" << endl;

    // read the coloring
    daJacCon->readJacConColoring();

    // initialize partDeriv object
    autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_(),
        daJacCon(),
        daResidualPtr_()));

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
    daPartDeriv->initializePartDerivMat(options1, dRdWT);

    // calculate dRdWT
    daPartDeriv->calcPartDerivMat(options1, xvVec, wVec, dRdWT);

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
    daJacCon->clear();
}

void DASolver::calcdFdW(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    Vec dFdW)
{
    /*
    Description:
        This function computes partials derivatives dFdW
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F
    
    Output:
        dFdW: the partial derivative vector dF/dW
        NOTE: You need to fully initialize the dFdW vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    VecZeroEntries(dFdW);

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts for this objFuncName
    forAll(objFuncSubDict.toc(), idxJ)
    {
        // get the subDict for this part
        word objFuncPart = objFuncSubDict.toc()[idxJ];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        // NOTE: dFdW is a matrix here and it has nObjFuncCellSources+nObjFuncFaceSources rows
        Mat dFdWMat;
        MatCreate(PETSC_COMM_WORLD, &dFdWMat);

        // initialize DAJacCon object
        word modelType = "dFdW";
        autoPtr<DAJacCon> daJacCon(DAJacCon::New(
            modelType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_()));

        // initialize objFunc to get objFuncCellSources and objFuncFaceSources
        autoPtr<DAObjFunc> daObjFunc(DAObjFunc::New(
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            objFuncName,
            objFuncPart,
            objFuncSubDictPart));

        // setup options for daJacCondFdW computation
        dictionary options;
        const List<List<word>>& objFuncConInfo = daObjFunc->getObjFuncConInfo();
        const labelList& objFuncFaceSources = daObjFunc->getObjFuncFaceSources();
        const labelList& objFuncCellSources = daObjFunc->getObjFuncCellSources();
        options.set("objFuncConInfo", objFuncConInfo);
        options.set("objFuncFaceSources", objFuncFaceSources);
        options.set("objFuncCellSources", objFuncCellSources);
        options.set("objFuncName", objFuncName);
        options.set("objFuncPart", objFuncPart);
        options.set("objFuncSubDictPart", objFuncSubDictPart);

        // now we can initilaize dFdWCon
        daJacCon->initializeJacCon(options);

        // setup dFdWCon
        daJacCon->setupJacCon(options);
        Info << "dFdWCon Created. " << meshPtr_->time().elapsedClockTime() << " s" << endl;

        // read the coloring
        word postFix = "_" + objFuncName + "_" + objFuncPart;
        daJacCon->readJacConColoring(postFix);

        // initialize DAPartDeriv to computing dFdW
        autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
            modelType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daJacCon(),
            daResidualPtr_()));

        // initialize dFdWMat
        daPartDeriv->initializePartDerivMat(options, dFdWMat);

        // compute it
        daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dFdWMat);

        // now we need to add all the rows of dFdW together to get dFdWPart
        // NOTE: dFdW is a matrix with nObjFuncCellSources+nObjFuncFaceSources rows
        // and nLocalAdjStates columns. So we can do dFdWPart = oneVec*dFdW
        Vec dFdWPart, oneVec;
        label objGeoSize = objFuncFaceSources.size() + objFuncCellSources.size();
        VecCreate(PETSC_COMM_WORLD, &oneVec);
        VecSetSizes(oneVec, objGeoSize, PETSC_DETERMINE);
        VecSetFromOptions(oneVec);
        // assign one to all elements
        VecSet(oneVec, 1.0);
        VecDuplicate(wVec, &dFdWPart);
        VecZeroEntries(dFdWPart);
        // dFdWPart = oneVec*dFdW
        MatMultTranspose(dFdWMat, oneVec, dFdWPart);

        // we need to add dFdWPart to dFdW because we want to sum all dFdWPart
        // for all parts of this objFuncName. When solving the adjoint equation, we use
        // dFdW
        VecAXPY(dFdW, 1.0, dFdWPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        MatDestroy(&dFdWMat);
        VecDestroy(&dFdWPart);
        VecDestroy(&oneVec);

        // clear up
        daJacCon->clear();
        daObjFunc->clear();
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdW") || writeJacobians.found("all"))
    {
        word outputName = "dFdW_" + objFuncName;
        DAUtility::writeVectorBinary(dFdW, outputName);
        DAUtility::writeVectorASCII(dFdW, outputName);
    }
}

void DASolver::calcdRdBC(
    const Vec xvVec,
    const Vec wVec,
    const word designVarName,
    Mat dRdBC)
{
    /*
    Description:
        This function computes partials derivatives dRdBC
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        designVarName: the name of the design variable
    
    Output:
        dRdBC: the partial derivative matrix dR/dBC
        NOTE: You need to call MatCreate for the dRdBC matrix before calling this function.
        No need to call MatSetSize etc because they will be done in this function
    */

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the variable for changing the boundary condition
    word varName = dvSubDict.getWord("variable");
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the compoent of a vector variable, ignore when it is a scalar
    label comp = dvSubDict.getLabel("comp");

    // no coloring is need for BC, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // ********************** compute dRdBC **********************

    // create DAPartDeriv object
    word modelType = "dRdBC";
    autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_(),
        daJacCon(),
        daResidualPtr_()));

    // setup options to compute dRdBC
    dictionary options;
    options.set("variable", varName);
    options.set("patches", patches);
    options.set("comp", comp);
    options.set("isPC", 0);

    // initialize the dRdBC matrix
    daPartDeriv->initializePartDerivMat(options, dRdBC);

    // compute it using brute force finite-difference
    daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dRdBC);

    if (daOptionPtr_->getOption<label>("debug"))
    {
        this->calcPrimalResidualStatistics("print");
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdBC") || writeJacobians.found("all"))
    {
        word outputName = "dRdBC_" + designVarName;
        DAUtility::writeMatrixBinary(dRdBC, outputName);
        DAUtility::writeMatrixASCII(dRdBC, outputName);
    }
}

void DASolver::calcdFdBC(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdBC)
{
    /*
    Description:
        This function computes partials derivatives dFdW
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F

        designVarName: the name of the design variable
    
    Output:
        dFdBC: the partial derivative vector dF/dBC
        NOTE: You need to fully initialize the dFdBC vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    VecZeroEntries(dFdBC);

    // no coloring is need for BC, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // get the subDict for this dvName
    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the variable for changing the boundary condition
    word varName = dvSubDict.getWord("variable");
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the compoent of a vector variable, ignore when it is a scalar
    label comp = dvSubDict.getLabel("comp");

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts of this objFuncName
    forAll(objFuncSubDict.toc(), idxK)
    {
        word objFuncPart = objFuncSubDict.toc()[idxK];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        Mat dFdBCMat;
        MatCreate(PETSC_COMM_WORLD, &dFdBCMat);

        // initialize DAPartDeriv for dFdBC
        word modelType = "dFdBC";
        autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
            modelType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daJacCon(),
            daResidualPtr_()));

        // initialize options
        dictionary options;
        options.set("objFuncName", objFuncName);
        options.set("objFuncPart", objFuncPart);
        options.set("objFuncSubDictPart", objFuncSubDictPart);
        options.set("variable", varName);
        options.set("patches", patches);
        options.set("comp", comp);

        // initialize dFdBC
        daPartDeriv->initializePartDerivMat(options, dFdBCMat);

        // calculate it
        daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dFdBCMat);

        // now we need to add all the rows of dFdBCMat together to get dFdBC
        // NOTE: dFdBCMat is a 1 by 1 matrix but we just do a matrix-vector product
        // to convert dFdBCMat from a matrix to a vector
        Vec dFdBCPart, oneVec;
        VecDuplicate(dFdBC, &oneVec);
        VecSet(oneVec, 1.0);
        VecDuplicate(dFdBC, &dFdBCPart);
        VecZeroEntries(dFdBCPart);
        // dFdBCPart = dFdBCMat * oneVec
        MatMult(dFdBCMat, oneVec, dFdBCPart);

        // we need to add dFdBCPart to dFdBC because we want to sum
        // all parts of this objFuncName.
        VecAXPY(dFdBC, 1.0, dFdBCPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        // clear up
        MatDestroy(&dFdBCMat);
        VecDestroy(&dFdBCPart);
        VecDestroy(&oneVec);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdBC") || writeJacobians.found("all"))
    {
        word outputName = "dFdBC_" + designVarName;
        DAUtility::writeVectorBinary(dFdBC, outputName);
        DAUtility::writeVectorASCII(dFdBC, outputName);
    }
}

void DASolver::calcdRdAOA(
    const Vec xvVec,
    const Vec wVec,
    const word designVarName,
    Mat dRdAOA)
{
    /*
    Description:
        This function computes partials derivatives dRdAOA
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        designVarName: the name of the design variable
    
    Output:
        dRdAOA: the partial derivative matrix dR/dAOA
        NOTE: You need to call MatCreate for the dRdAOA matrix before calling this function.
        No need to call MatSetSize etc because they will be done in this function
    */

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the streamwise axis of aoa, aoa = tan( U_normal/U_flow )
    word flowAxis = dvSubDict.getWord("flowAxis");
    word normalAxis = dvSubDict.getWord("normalAxis");

    // no coloring is need for BC, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // ********************** compute dRdAOA **********************

    // create DAPartDeriv object
    word modelType = "dRdAOA";
    autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_(),
        daJacCon(),
        daResidualPtr_()));

    // setup options to compute dRdAOA
    dictionary options;
    options.set("patches", patches);
    options.set("flowAxis", flowAxis);
    options.set("normalAxis", normalAxis);
    options.set("isPC", 0);

    // initialize the dRdAOA matrix
    daPartDeriv->initializePartDerivMat(options, dRdAOA);

    // compute it using brute force finite-difference
    daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dRdAOA);

    if (daOptionPtr_->getOption<label>("debug"))
    {
        this->calcPrimalResidualStatistics("print");
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdAOA") || writeJacobians.found("all"))
    {
        word outputName = "dRdAOA_" + designVarName;
        DAUtility::writeMatrixBinary(dRdAOA, outputName);
        DAUtility::writeMatrixASCII(dRdAOA, outputName);
    }
}

void DASolver::calcdRdBCTPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    const word designVarName,
    Vec dRdBCTPsi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute the matrix-vector products dRdBC^T*Psi using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdBC

        designVarName: name of the design variable
    
    Output:
        dRdBCTPsi: the matrix-vector products dRdBC^T * Psi
    */

    Info << "Calculating [dRdBC]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdBCTPsi);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the variable for changing the boundary condition
    word varName = dvSubDict.getWord("variable");
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the component of a vector variable, ignore when it is a scalar
    label comp = dvSubDict.getLabel("comp");

    // Now get the BC value
    scalar BC = -1e16;
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
    // need to reduce the BC value across all processors, this is because some of
    // the processors might not own the prescribed patches so their BC value will be still -1e16, but
    // when calling the following reduce function, they will get the correct BC from other processors
    reduce(BC, maxOp<scalar>());

    this->globalADTape_.reset();
    this->globalADTape_.setActive();
    // register BC as the input
    this->globalADTape_.registerInput(BC);
    // ******* now set BC ******
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
    // ******* now set BC done******
    // compute residuals
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    PetscScalar derivVal = BC.getGradient();
    // we need to do ADD_VALUES to get contribution from all procs
    VecSetValue(dRdBCTPsi, 0, derivVal, ADD_VALUES);

    VecAssemblyBegin(dRdBCTPsi);
    VecAssemblyEnd(dRdBCTPsi);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdBCTPsi") || writeJacobians.found("all"))
    {
        word outputName = "dRdBCTPsi_" + designVarName;
        DAUtility::writeVectorBinary(dRdBCTPsi, outputName);
        DAUtility::writeVectorASCII(dRdBCTPsi, outputName);
    }
#endif
}

void DASolver::calcdFdAOA(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdAOA)
{
    /*
    Description:
        This function computes partials derivatives dFdAOA
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F

        designVarName: the name of the design variable
    
    Output:
        dFdAOA: the partial derivative vector dF/dAOA
        NOTE: You need to fully initialize the dFdAOA vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    VecZeroEntries(dFdAOA);

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the streamwise axis of aoa, aoa = tan( U_normal/U_flow )
    word flowAxis = dvSubDict.getWord("flowAxis");
    word normalAxis = dvSubDict.getWord("normalAxis");

    // no coloring is need for BC, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts of this objFuncName
    forAll(objFuncSubDict.toc(), idxK)
    {
        word objFuncPart = objFuncSubDict.toc()[idxK];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        Mat dFdAOAMat;
        MatCreate(PETSC_COMM_WORLD, &dFdAOAMat);

        // initialize DAPartDeriv for dFdAOAMat
        word modelType = "dFdAOA";
        autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
            modelType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daJacCon(),
            daResidualPtr_()));

        // initialize options
        dictionary options;
        options.set("objFuncName", objFuncName);
        options.set("objFuncPart", objFuncPart);
        options.set("objFuncSubDictPart", objFuncSubDictPart);
        options.set("patches", patches);
        options.set("flowAxis", flowAxis);
        options.set("normalAxis", normalAxis);

        // initialize dFdAOA
        daPartDeriv->initializePartDerivMat(options, dFdAOAMat);

        // calculate it
        daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dFdAOAMat);

        // NOTE: dFdAOAMat is a 1 by 1 matrix but we just do a matrix-vector product
        // to convert dFdAOAMat from a matrix to a vector
        Vec dFdAOAPart, oneVec;
        VecDuplicate(dFdAOA, &oneVec);
        VecSet(oneVec, 1.0);
        VecDuplicate(dFdAOA, &dFdAOAPart);
        VecZeroEntries(dFdAOAPart);
        // dFdAOAPart = dFdAOAMat * oneVec
        MatMult(dFdAOAMat, oneVec, dFdAOAPart);

        // we need to add dFdAOAVec to dFdAOAVecAllParts because we want to sum
        // all dFdAOAVec for all parts of this objFuncName.
        VecAXPY(dFdAOA, 1.0, dFdAOAPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        // clear up
        MatDestroy(&dFdAOAMat);
        VecDestroy(&dFdAOAPart);
        VecDestroy(&oneVec);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdAOA") || writeJacobians.found("all"))
    {
        word outputName = "dFdAOA_" + designVarName;
        DAUtility::writeVectorBinary(dFdAOA, outputName);
        DAUtility::writeVectorASCII(dFdAOA, outputName);
    }
}

void DASolver::calcdRdAOATPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    const word designVarName,
    Vec dRdAOATPsi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute the matrix-vector products dRdAOA^T*Psi using reverse-mode AD
        Similar to dF/dAlpha, here
        dR/dAlpha = dR/dTan(Alpha) * dTan(Alpha)/dAlpha 
                  = dR/d(Uy/Ux) / Cos(Alpha)^2 
                  = dR/dUy * Ux / Cos(Alpha)^2
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdAOA

        designVarName: name of the design variable
    
    Output:
        dRdAOATPsi: the matrix-vector products dRdAOA^T * Psi
    */

    Info << "Calculating [dRdAOA]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdAOATPsi);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);

    // get info from dvSubDict. This needs to be defined in the pyDAFoam
    // name of the boundary patch
    wordList patches;
    dvSubDict.readEntry<wordList>("patches", patches);
    // the streamwise axis of aoa, aoa = tan( U_normal/U_flow )
    word flowAxis = dvSubDict.getWord("flowAxis");
    word normalAxis = dvSubDict.getWord("normalAxis");

    HashTable<label> axisIndices;
    axisIndices.set("x", 0);
    axisIndices.set("y", 1);
    axisIndices.set("z", 2);
    label flowAxisIndex = axisIndices[flowAxis];
    label normalAxisIndex = axisIndices[normalAxis];

    volVectorField& U = const_cast<volVectorField&>(
        meshPtr_->thisDb().lookupObject<volVectorField>("U"));

    // now we need to get the Ux, Uy values from the inout patches
    scalar Ux0 = -1e16, Uy0 = -1e16;
    forAll(patches, idxI)
    {
        word patchName = patches[idxI];
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);
        if (meshPtr_->boundaryMesh()[patchI].size() > 0)
        {
            if (U.boundaryField()[patchI].type() == "fixedValue")
            {
                Uy0 = U.boundaryField()[patchI][0][normalAxisIndex];
                Ux0 = U.boundaryField()[patchI][0][flowAxisIndex];
                break;
            }
            else if (U.boundaryField()[patchI].type() == "inletOutlet")
            {
                mixedFvPatchField<vector>& inletOutletPatch =
                    refCast<mixedFvPatchField<vector>>(U.boundaryFieldRef()[patchI]);
                Uy0 = inletOutletPatch.refValue()[0][normalAxisIndex];
                Ux0 = inletOutletPatch.refValue()[0][flowAxisIndex];
                break;
            }
            else
            {
                FatalErrorIn("") << "boundaryType: " << U.boundaryFieldRef()[patchI].type()
                                 << " not supported!"
                                 << "Avaiable options are: fixedValue, inletOutlet"
                                 << abort(FatalError);
            }
        }
    }
    // need to reduce the U value across all processors, this is because some of
    // the processors might not own the prescribed patches so their U value will be still -1e16, but
    // when calling the following reduce function, they will get the correct U from other processors
    reduce(Ux0, maxOp<scalar>());
    reduce(Uy0, maxOp<scalar>());
    scalar aoa = atan(Uy0 / Ux0);

    this->globalADTape_.reset();
    this->globalADTape_.setActive();
    // register aoa as the input
    this->globalADTape_.registerInput(aoa);
    // set far field U
    forAll(patches, idxI)
    {
        word patchName = patches[idxI];
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchName);

        if (meshPtr_->boundaryMesh()[patchI].size() > 0)
        {
            scalar UMag = sqrt(Ux0 * Ux0 + Uy0 * Uy0);
            scalar UxNew = UMag * cos(aoa);
            scalar UyNew = UMag * sin(aoa);

            if (U.boundaryField()[patchI].type() == "fixedValue")
            {
                forAll(U.boundaryField()[patchI], faceI)
                {
                    U.boundaryFieldRef()[patchI][faceI][flowAxisIndex] = UxNew;
                    U.boundaryFieldRef()[patchI][faceI][normalAxisIndex] = UyNew;
                }
            }
            else if (U.boundaryField()[patchI].type() == "inletOutlet")
            {
                mixedFvPatchField<vector>& inletOutletPatch =
                    refCast<mixedFvPatchField<vector>>(U.boundaryFieldRef()[patchI]);

                forAll(U.boundaryField()[patchI], faceI)
                {
                    inletOutletPatch.refValue()[faceI][flowAxisIndex] = UxNew;
                    inletOutletPatch.refValue()[faceI][normalAxisIndex] = UyNew;
                }
            }
        }
    }
    // compute residuals
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    // need to convert dFdAOA from radian to degree
    PetscScalar derivVal = aoa.getGradient() * constant::mathematical::pi.getValue() / 180.0;
    // we need to do ADD_VALUES to get contribution from all procs
    VecSetValue(dRdAOATPsi, 0, derivVal, ADD_VALUES);

    VecAssemblyBegin(dRdAOATPsi);
    VecAssemblyEnd(dRdAOATPsi);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdAOATPsi") || writeJacobians.found("all"))
    {
        word outputName = "dRdAOATPsi_" + designVarName;
        DAUtility::writeVectorBinary(dRdAOATPsi, outputName);
        DAUtility::writeVectorASCII(dRdAOATPsi, outputName);
    }
#endif
}

void DASolver::calcdRdFFD(
    const Vec xvVec,
    const Vec wVec,
    const word designVarName,
    Mat dRdFFD)
{
    /*
    Description:
        This function computes partials derivatives dRdFFD
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        designVarName: the name of the design variable
    
    Output:
        dRdFFD: the partial derivative matrix dR/dFFD
        NOTE: You need to call MatCreate for the dRdFFD matrix before calling this function.
        No need to call MatSetSize etc because they will be done in this function
    */

    // get the size of dXvdFFDMat_, nCols will be the number of FFD points
    // for this design variable
    // NOTE: dXvdFFDMat_ needs to be assigned by calling DASolver::setdXvdFFDMat in
    // the python layer
    label nDesignVars = -9999;
    MatGetSize(dXvdFFDMat_, NULL, &nDesignVars);

    // no coloring is need for FFD, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // create DAPartDeriv object
    word modelType = "dRdFFD";
    autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_(),
        daJacCon(),
        daResidualPtr_()));

    // setup options
    dictionary options;
    options.set("nDesignVars", nDesignVars);
    options.set("isPC", 0);

    // for FFD, we need to first assign dXvdFFDMat to daPartDeriv
    daPartDeriv->setdXvdFFDMat(dXvdFFDMat_);

    // initialize the dRdFFD matrix
    daPartDeriv->initializePartDerivMat(options, dRdFFD);

    // compute it using brute force finite-difference
    daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dRdFFD);

    if (daOptionPtr_->getOption<label>("debug"))
    {
        this->calcPrimalResidualStatistics("print");
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdFFD") || writeJacobians.found("all"))
    {
        word outputName = "dRdFFD_" + designVarName;
        DAUtility::writeMatrixBinary(dRdFFD, outputName);
        DAUtility::writeMatrixASCII(dRdFFD, outputName);
    }

    // clear up dXvdFFD Mat in daPartDeriv
    daPartDeriv->clear();
}

void DASolver::calcdFdFFD(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdFFD)
{
    /*
    Description:
        This function computes partials derivatives dFdFFD
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F

        designVarName: the name of the design variable
    
    Output:
        dFdFFD: the partial derivative vector dF/dFFD
        NOTE: You need to fully initialize the dFdFFD vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    VecZeroEntries(dFdFFD);

    // get the size of dXvdFFDMat_, nCols will be the number of FFD points
    // for this design variable
    // NOTE: dXvdFFDMat_ needs to be assigned by calling DASolver::setdXvdFFDMat in
    // the python layer
    label nDesignVars = -9999;
    MatGetSize(dXvdFFDMat_, NULL, &nDesignVars);

    // no coloring is need for FFD, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts of this objFuncName
    forAll(objFuncSubDict.toc(), idxK)
    {
        word objFuncPart = objFuncSubDict.toc()[idxK];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        Mat dFdFFDMat;
        MatCreate(PETSC_COMM_WORLD, &dFdFFDMat);

        // initialize DAPartDeriv for dFdFFD
        word modelType = "dFdFFD";
        autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
            modelType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daJacCon(),
            daResidualPtr_()));

        // initialize options
        dictionary options;
        options.set("objFuncName", objFuncName);
        options.set("objFuncPart", objFuncPart);
        options.set("objFuncSubDictPart", objFuncSubDictPart);
        options.set("nDesignVars", nDesignVars);

        // for FFD, we need to first assign dXvdFFDMat to daPartDeriv
        daPartDeriv->setdXvdFFDMat(dXvdFFDMat_);

        // initialize dFdFFD
        daPartDeriv->initializePartDerivMat(options, dFdFFDMat);

        // calculate it
        daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dFdFFDMat);

        // now we need to convert the dFdFFD mat to dFdFFDPart
        // NOTE: dFdFFDMat is a 1 by nDesignVars matrix but dFdFFDPart is
        // a nDesignVars by 1 vector, we need to do
        // dFdFFDPart = (dFdFFDMat)^T * oneVec
        Vec dFdFFDPart, oneVec;
        VecCreate(PETSC_COMM_WORLD, &oneVec);
        VecSetSizes(oneVec, PETSC_DETERMINE, 1);
        VecSetFromOptions(oneVec);
        VecSet(oneVec, 1.0);
        VecDuplicate(dFdFFD, &dFdFFDPart);
        VecZeroEntries(dFdFFDPart);
        // dFdFFDVec = oneVec*dFdFFD
        MatMultTranspose(dFdFFDMat, oneVec, dFdFFDPart);

        // we need to add dFdFFDPart to dFdFFD because we want to sum
        // all dFdFFDPart for all parts of this objFuncName.
        VecAXPY(dFdFFD, 1.0, dFdFFDPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        MatDestroy(&dFdFFDMat);
        VecDestroy(&dFdFFDPart);
        VecDestroy(&oneVec);

        // clear up dXvdFFD Mat in daPartDeriv
        daPartDeriv->clear();
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdFFD") || writeJacobians.found("all"))
    {
        word outputName = "dFdFFD_" + designVarName;
        DAUtility::writeVectorBinary(dFdFFD, outputName);
        DAUtility::writeVectorASCII(dFdFFD, outputName);
    }
}

void DASolver::calcdRdACT(
    const Vec xvVec,
    const Vec wVec,
    const word designVarName,
    const word designVarType,
    Mat dRdACT)
{
    /*
    Description:
        This function computes partials derivatives dRdACT
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        designVarName: the name of the design variable

        designVarType: the type of the design variable: ACTP, ACTL, ACTD
    
    Output:
        dRdACT: the partial derivative matrix dR/dACT
        NOTE: You need to call MatCreate for the dRdACT matrix before calling this function.
        No need to call MatSetSize etc because they will be done in this function
    */

    // get the subDict for this dvName
    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);
    word actuatorName = dvSubDict.getWord("actuatorName");

    // no coloring is need for actuator, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // create DAPartDeriv object
    word modelType = "dRd" + designVarType;
    autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
        modelType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_(),
        daJacCon(),
        daResidualPtr_()));

    // setup options to compute dRdACT*
    dictionary options;
    options.set("actuatorName", actuatorName);
    options.set("isPC", 0);

    // initialize the dRdACT* matrix
    daPartDeriv->initializePartDerivMat(options, dRdACT);

    // compute it using brute force finite-difference
    daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dRdACT);

    if (daOptionPtr_->getOption<label>("debug"))
    {
        this->calcPrimalResidualStatistics("print");
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdACT") || writeJacobians.found("all"))
    {
        word outputName = "dRd" + designVarType + "_" + designVarName;
        DAUtility::writeMatrixBinary(dRdACT, outputName);
        DAUtility::writeMatrixASCII(dRdACT, outputName);
    }
}

void DASolver::calcdFdACT(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    const word designVarType,
    Vec dFdACT)
{
    /*
    Description:
        This function computes partials derivatives dFdW
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F

        designVarName: the name of the design variable
    
    Output:
        dFdACT: the partial derivative vector dF/dACT
        NOTE: You need to fully initialize the dFdACT vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    VecZeroEntries(dFdACT);

    if (designVarType != "ACTD")
    {
        return;
    }

    // no coloring is need for ACT, so we create a dummy DAJacCon
    word dummyType = "dummy";
    autoPtr<DAJacCon> daJacCon(DAJacCon::New(
        dummyType,
        meshPtr_(),
        daOptionPtr_(),
        daModelPtr_(),
        daIndexPtr_()));

    // get the subDict for this dvName
    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);
    word actuatorName = dvSubDict.getWord("actuatorName");

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts of this objFuncName
    forAll(objFuncSubDict.toc(), idxK)
    {
        word objFuncPart = objFuncSubDict.toc()[idxK];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        Mat dFdACTMat;
        MatCreate(PETSC_COMM_WORLD, &dFdACTMat);

        // initialize DAPartDeriv for dFdACT
        word modelType = "dFd" + designVarType;
        autoPtr<DAPartDeriv> daPartDeriv(DAPartDeriv::New(
            modelType,
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daJacCon(),
            daResidualPtr_()));

        // initialize options
        dictionary options;
        options.set("objFuncName", objFuncName);
        options.set("objFuncPart", objFuncPart);
        options.set("objFuncSubDictPart", objFuncSubDictPart);
        options.set("actuatorName", actuatorName);

        // initialize dFdACT
        daPartDeriv->initializePartDerivMat(options, dFdACTMat);

        // calculate it
        daPartDeriv->calcPartDerivMat(options, xvVec, wVec, dFdACTMat);

        // now we need to extract the dFdACT from dFdACTMatrix
        // NOTE: dFdACTMat is a nACTDVs by 1 matrix
        Vec dFdACTPart;
        VecDuplicate(dFdACT, &dFdACTPart);
        VecZeroEntries(dFdACTPart);
        MatGetColumnVector(dFdACTMat, dFdACTPart, 0);

        // we need to add dFdACTPart to dFdACT because we want to sum
        // all parts of this objFuncName.
        VecAXPY(dFdACT, 1.0, dFdACTPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
        }

        // clear up
        MatDestroy(&dFdACTMat);
        VecDestroy(&dFdACTPart);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdACT") || writeJacobians.found("all"))
    {
        word outputName = "dFdACT_" + designVarName;
        DAUtility::writeVectorBinary(dFdACT, outputName);
        DAUtility::writeVectorASCII(dFdACT, outputName);
    }
}

void DASolver::calcdFdFieldAD(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdField)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        This function computes partials derivatives dFdField
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F

        designVarName: the name of the design variable
    
    Output:
        dFdField: the partial derivative vector dF/dField
        NOTE: You need to fully initialize the dF vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    Info << "Calculating dFdField using reverse-mode AD for " << designVarName << endl;

    VecZeroEntries(dFdField);

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);

    word fieldName = dvSubDict.getWord("fieldName");
    word fieldType = dvSubDict.getWord("fieldType");

    // loop over all parts of this objFuncName
    forAll(objFuncSubDict.toc(), idxK)
    {
        word objFuncPart = objFuncSubDict.toc()[idxK];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        // initialize objFunc to get objFuncCellSources and objFuncFaceSources
        autoPtr<DAObjFunc> daObjFunc(DAObjFunc::New(
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            objFuncName,
            objFuncPart,
            objFuncSubDictPart));

        // reset tape
        this->globalADTape_.reset();
        // activate tape, start recording
        this->globalADTape_.setActive();
        // register states as the input
        this->registerFieldVariableInput4AD(fieldName, fieldType);
        this->updateBoundaryConditions(fieldName, fieldType);
        // update all intermediate variables and boundary conditions
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
        // compute the objective function
        scalar fRef = daObjFunc->getObjFuncValue();
        // register f as the output
        this->globalADTape_.registerOutput(fRef);
        // stop recording
        this->globalADTape_.setPassive();

        // Note: since we used reduced objFunc, we only need to
        // assign the seed for master proc
        if (Pstream::master())
        {
            fRef.setGradient(1.0);
        }
        // evaluate tape to compute derivative
        this->globalADTape_.evaluate();

        // assign the computed derivatives from the OpenFOAM variable to dFdFieldPart
        Vec dFdFieldPart;
        VecDuplicate(dFdField, &dFdFieldPart);
        VecZeroEntries(dFdFieldPart);
        this->assignFieldGradient2Vec(fieldName, fieldType, dFdFieldPart);

        // need to clear adjoint and tape after the computation is done!
        this->globalADTape_.clearAdjoints();
        this->globalADTape_.reset();

        // we need to add dFdFieldPart to dFdField because we want to sum
        // all dFdFieldPart for all parts of this objFuncName.
        VecAXPY(dFdField, 1.0, dFdFieldPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            Info << "In calcdFdFieldAD" << endl;
            this->calcPrimalResidualStatistics("print");
            Info << objFuncName << ": " << fRef << endl;
        }

        VecDestroy(&dFdFieldPart);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdField") || writeJacobians.found("all"))
    {
        word outputName = "dFdField_" + designVarName;
        DAUtility::writeVectorBinary(dFdField, outputName);
        DAUtility::writeVectorASCII(dFdField, outputName);
    }
#endif
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

void DASolver::createMLRKSPMatrixFree(
    const Mat jacPCMat,
    KSP ksp)
{
#ifdef CODI_AD_REVERSE
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

    return error;
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
    Info << "Updating the OpenFOAM field..." << endl;
    //Info << "Setting up primal boundary conditions based on pyOptions: " << endl;
    daFieldPtr_->setPrimalBoundaryConditions(0);
    daFieldPtr_->stateVec2OFField(wVec);
    // We need to call correctBC multiple times to reproduce
    // the exact residual, this is needed for some boundary conditions
    // and intermediate variables (e.g., U for inletOutlet, nut with wall functions)
    for (label i = 0; i < 10; i++)
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
    Info << "Updating the OpenFOAM mesh..." << endl;
    daFieldPtr_->pointVec2OFMesh(xvVec);
}

void DASolver::initializedRdWTMatrixFree(
    const Vec xvVec,
    const Vec wVec)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        This function initialize the matrix-free dRdWT, which will be
        used later in the adjoint solution
    */

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

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
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Destroy dRdWTMF_
    */
    MatDestroy(&dRdWTMF_);
#endif
}

PetscErrorCode DASolver::dRdWTMatVecMultFunction(Mat dRdWTMF, Vec vecX, Vec vecY)
{
#ifdef CODI_AD_REVERSE
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
#ifdef CODI_AD_REVERSE
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
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    // Now we can compute the residuals
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);
    // Set the residual as the output
    this->registerResidualOutput4AD();
    // All done, set the tape to passive
    this->globalADTape_.setPassive();

    // Now the tape is ready to use in the matrix-free GMRES solution
#endif
}

void DASolver::calcdFdWAD(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    Vec dFdW)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        This function computes partials derivatives dFdW using AD
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: name of the objective function F
    
    Output:
        dFdW: the partial derivative vector dF/dW
        NOTE: You need to fully initialize the dFdW vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    Info << "Calculating dFdW using reverse-mode AD" << endl;

    VecZeroEntries(dFdW);

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts for this objFuncName
    forAll(objFuncSubDict.toc(), idxJ)
    {
        // get the subDict for this part
        word objFuncPart = objFuncSubDict.toc()[idxJ];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        // initialize objFunc to get objFuncCellSources and objFuncFaceSources
        autoPtr<DAObjFunc> daObjFunc(DAObjFunc::New(
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            objFuncName,
            objFuncPart,
            objFuncSubDictPart));

        // reset tape
        this->globalADTape_.reset();
        // activate tape, start recording
        this->globalADTape_.setActive();
        // register states as the input
        this->registerStateVariableInput4AD();
        // update all intermediate variables and boundary conditions
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
        // compute the objective function
        scalar fRef = daObjFunc->getObjFuncValue();
        // register f as the output
        this->globalADTape_.registerOutput(fRef);
        // stop recording
        this->globalADTape_.setPassive();

        // Note: since we used reduced objFunc, we only need to
        // assign the seed for master proc
        if (Pstream::master())
        {
            fRef.setGradient(1.0);
        }
        // evaluate tape to compute derivative
        this->globalADTape_.evaluate();

        // assign the computed derivatives from the OpenFOAM variable to dFdWPart
        Vec dFdWPart;
        VecDuplicate(dFdW, &dFdWPart);
        VecZeroEntries(dFdWPart);
        this->assignStateGradient2Vec(dFdWPart);

        // need to clear adjoint and tape after the computation is done!
        this->globalADTape_.clearAdjoints();
        this->globalADTape_.reset();

        // we need to add dFdWPart to dFdW because we want to sum
        // all dFdWPart for all parts of this objFuncName.
        VecAXPY(dFdW, 1.0, dFdWPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            Info << "In calcdFdWAD" << endl;
            this->calcPrimalResidualStatistics("print");
            Info << objFuncName << ": " << fRef << endl;
        }

        VecDestroy(&dFdWPart);
    }

    // NOTE: we need to normalize dFdW!
    this->normalizeGradientVec(dFdW);

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdW") || writeJacobians.found("all"))
    {
        word outputName = "dFdW_" + objFuncName;
        DAUtility::writeVectorBinary(dFdW, outputName);
        DAUtility::writeVectorASCII(dFdW, outputName);
    }

#endif
}

void DASolver::calcdFdXvAD(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdXv)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute dFdXv using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: the name of the objective function

        designVarName: name of the design variable
    
    Output:
        dFdXv: dF/dXv
    */

    Info << "Calculating dFdXv using reverse-mode AD" << endl;

    VecZeroEntries(dFdXv);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

    // loop over all parts for this objFuncName
    forAll(objFuncSubDict.toc(), idxJ)
    {
        // get the subDict for this part
        word objFuncPart = objFuncSubDict.toc()[idxJ];
        dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

        // initialize objFunc to get objFuncCellSources and objFuncFaceSources
        autoPtr<DAObjFunc> daObjFunc(DAObjFunc::New(
            meshPtr_(),
            daOptionPtr_(),
            daModelPtr_(),
            daIndexPtr_(),
            daResidualPtr_(),
            objFuncName,
            objFuncPart,
            objFuncSubDictPart));

        pointField meshPoints = meshPtr_->points();

        // reset tape
        this->globalADTape_.reset();
        // activate tape, start recording
        this->globalADTape_.setActive();
        // register points as the input
        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                this->globalADTape_.registerInput(meshPoints[i][j]);
            }
        }
        meshPtr_->movePoints(meshPoints);
        meshPtr_->moving(false);
        // update all intermediate variables and boundary conditions
        daResidualPtr_->correctBoundaryConditions();
        daResidualPtr_->updateIntermediateVariables();
        daModelPtr_->correctBoundaryConditions();
        daModelPtr_->updateIntermediateVariables();
        // compute the objective function
        scalar fRef = daObjFunc->getObjFuncValue();
        // register f as the output
        this->globalADTape_.registerOutput(fRef);
        // stop recording
        this->globalADTape_.setPassive();

        // Note: since we used reduced objFunc, we only need to
        // assign the seed for master proc
        if (Pstream::master())
        {
            fRef.setGradient(1.0);
        }
        // evaluate tape to compute derivative
        this->globalADTape_.evaluate();

        // assign the computed derivatives from the OpenFOAM variable to dFd*Part
        Vec dFdXvPart;
        VecDuplicate(dFdXv, &dFdXvPart);
        VecZeroEntries(dFdXvPart);

        forAll(meshPoints, i)
        {
            for (label j = 0; j < 3; j++)
            {
                label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
                PetscScalar val = meshPoints[i][j].getGradient();
                VecSetValue(dFdXvPart, rowI, val, INSERT_VALUES);
            }
        }
        VecAssemblyBegin(dFdXvPart);
        VecAssemblyEnd(dFdXvPart);

        // need to clear adjoint and tape after the computation is done!
        this->globalADTape_.clearAdjoints();
        this->globalADTape_.reset();

        // we need to add dFd*Part to dFd* because we want to sum
        // all dFd*Part for all parts of this objFuncName.
        VecAXPY(dFdXv, 1.0, dFdXvPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            this->calcPrimalResidualStatistics("print");
            Info << objFuncName << ": " << fRef << endl;
        }

        VecDestroy(&dFdXvPart);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdXv") || writeJacobians.found("all"))
    {
        word outputName = "dFdXv_" + objFuncName + "_" + designVarName;
        DAUtility::writeVectorBinary(dFdXv, outputName);
        DAUtility::writeVectorASCII(dFdXv, outputName);
    }
#endif
}

void DASolver::calcdRdXvTPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    Vec dRdXvTPsi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute the matrix-vector products dRdXv^T*Psi using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdXv
    
    Output:
        dRdXvTPsi: the matrix-vector products dRdXv^T * Psi
    */

    Info << "Calculating [dRdXv]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdXvTPsi);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    pointField meshPoints = meshPtr_->points();
    this->globalADTape_.reset();
    this->globalADTape_.setActive();
    forAll(meshPoints, i)
    {
        for (label j = 0; j < 3; j++)
        {
            this->globalADTape_.registerInput(meshPoints[i][j]);
        }
    }
    meshPtr_->movePoints(meshPoints);
    meshPtr_->moving(false);
    // compute residuals
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    forAll(meshPoints, i)
    {
        for (label j = 0; j < 3; j++)
        {
            label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
            PetscScalar val = meshPoints[i][j].getGradient();
            VecSetValue(dRdXvTPsi, rowI, val, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(dRdXvTPsi);
    VecAssemblyEnd(dRdXvTPsi);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::calcdRdFieldTPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    const word designVarName,
    Vec dRdFieldTPsi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute the matrix-vector products dRdField^T*Psi using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdField

        designVarName: name of the design variable
    
    Output:
        dRdFieldTPsi: the matrix-vector products dRdField^T * Psi
    */

    Info << "Calculating [dRdField]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdFieldTPsi);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);

    word fieldName = dvSubDict.getWord("fieldName");
    word fieldType = dvSubDict.getWord("fieldType");

    this->globalADTape_.reset();
    this->globalADTape_.setActive();
    this->registerFieldVariableInput4AD(fieldName, fieldType);
    this->updateBoundaryConditions(fieldName, fieldType);
    // compute residuals
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

    this->registerResidualOutput4AD();
    this->globalADTape_.setPassive();

    this->assignVec2ResidualGradient(psi);
    this->globalADTape_.evaluate();

    this->assignFieldGradient2Vec(fieldName, fieldType, dRdFieldTPsi);

    VecAssemblyBegin(dRdFieldTPsi);
    VecAssemblyEnd(dRdFieldTPsi);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdFieldTPsi") || writeJacobians.found("all"))
    {
        word outputName = "dRdFieldTPsi_" + designVarName;
        DAUtility::writeVectorBinary(dRdFieldTPsi, outputName);
        DAUtility::writeVectorASCII(dRdFieldTPsi, outputName);
    }
#endif
}

void DASolver::calcdFdACTAD(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdACT)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute dFdACT using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        objFuncName: the name of the objective function

        designVarName: name of the design variable
    
    Output:
        dFdACT: dF/dACT
    */

    Info << "Calculating dFdACT using reverse-mode AD" << endl;

    VecZeroEntries(dFdACT);

    // first check if the input is valid
    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);
    word designVarType = dvSubDict.getWord("designVarType");
    if (designVarType == "ACTD")
    {
        DAFvSource& fvSource = const_cast<DAFvSource&>(
            meshPtr_->thisDb().lookupObject<DAFvSource>("DAFvSource"));

        word diskName = dvSubDict.getWord("actuatorName");
        dictionary fvSourceSubDict = daOptionPtr_->getAllOptions().subDict("fvSource");
        word source = fvSourceSubDict.subDict(diskName).getWord("source");
        if (source == "cylinderAnnulusSmooth")
        {
            this->updateOFField(wVec);
            this->updateOFMesh(xvVec);

            // get the subDict for this objective function
            dictionary objFuncSubDict =
                daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);

            // loop over all parts for this objFuncName
            forAll(objFuncSubDict.toc(), idxJ)
            {
                // get the subDict for this part
                word objFuncPart = objFuncSubDict.toc()[idxJ];
                dictionary objFuncSubDictPart = objFuncSubDict.subDict(objFuncPart);

                // initialize objFunc to get objFuncCellSources and objFuncFaceSources
                autoPtr<DAObjFunc> daObjFunc(DAObjFunc::New(
                    meshPtr_(),
                    daOptionPtr_(),
                    daModelPtr_(),
                    daIndexPtr_(),
                    daResidualPtr_(),
                    objFuncName,
                    objFuncPart,
                    objFuncSubDictPart));

                // get the design variable vals
                scalarList actDVList(9);
                for (label i = 0; i < 9; i++)
                {
                    actDVList[i] = fvSource.getActuatorDVs(diskName, i);
                }

                // reset tape
                this->globalADTape_.reset();
                // activate tape, start recording
                this->globalADTape_.setActive();
                // register  the input
                for (label i = 0; i < 9; i++)
                {
                    this->globalADTape_.registerInput(actDVList[i]);
                }
                // set dv values to fvSource obj for all procs
                for (label i = 0; i < 9; i++)
                {
                    fvSource.setActuatorDVs(diskName, i, actDVList[i]);
                }
                // the actuatorDVs are updated, now we need to recompute fvSource
                // this is not needed for the residual partials because fvSource
                // will be automatically calculated in the UEqn, but for the
                // obj partials, we need to manually recompute fvSource
                fvSource.updateFvSource();

                // update all intermediate variables and boundary conditions
                daResidualPtr_->correctBoundaryConditions();
                daResidualPtr_->updateIntermediateVariables();
                daModelPtr_->correctBoundaryConditions();
                daModelPtr_->updateIntermediateVariables();
                // compute the objective function
                scalar fRef = daObjFunc->getObjFuncValue();
                // register f as the output
                this->globalADTape_.registerOutput(fRef);
                // stop recording
                this->globalADTape_.setPassive();

                // Note: since we used reduced objFunc, we only need to
                // assign the seed for master proc
                if (Pstream::master())
                {
                    fRef.setGradient(1.0);
                }
                // evaluate tape to compute derivative
                this->globalADTape_.evaluate();

                // assign the computed derivatives from the OpenFOAM variable to dFd*Part
                Vec dFdACTPart;
                VecDuplicate(dFdACT, &dFdACTPart);
                VecZeroEntries(dFdACTPart);

                for (label i = 0; i < 9; i++)
                {
                    PetscScalar valIn = actDVList[i].getGradient();
                    // we need to do ADD_VALUES to get contribution from all procs
                    VecSetValue(dFdACTPart, i, valIn, ADD_VALUES);
                }

                VecAssemblyBegin(dFdACTPart);
                VecAssemblyEnd(dFdACTPart);

                // need to clear adjoint and tape after the computation is done!
                this->globalADTape_.clearAdjoints();
                this->globalADTape_.reset();

                // we need to add dFd*Part to dFd* because we want to sum
                // all dFd*Part for all parts of this objFuncName.
                VecAXPY(dFdACT, 1.0, dFdACTPart);

                if (daOptionPtr_->getOption<label>("debug"))
                {
                    this->calcPrimalResidualStatistics("print");
                    Info << objFuncName << ": " << fRef << endl;
                }

                VecDestroy(&dFdACTPart);
            }
        }
        else
        {
            FatalErrorIn("") << "source not supported. Options: cylinderAnnulusSmooth"
                             << abort(FatalError);
        }
    }
    else
    {
        FatalErrorIn("") << "designVarType not supported. Options: ACTD"
                         << abort(FatalError);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdACT") || writeJacobians.found("all"))
    {
        word outputName = "dFdACT_" + objFuncName + "_" + designVarName;
        DAUtility::writeVectorBinary(dFdACT, outputName);
        DAUtility::writeVectorASCII(dFdACT, outputName);
    }
#endif
}

void DASolver::calcdRdActTPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    const word designVarName,
    Vec dRdActTPsi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute the matrix-vector products dRdAct^T*Psi using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdAct

        designVarName: name of the design variable
    
    Output:
        dRdActTPsi: the matrix-vector products dRdAct^T * Psi
    */

    Info << "Calculating [dRdAct]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(dRdActTPsi);

    dictionary dvSubDict = daOptionPtr_->getAllOptions().subDict("designVar").subDict(designVarName);
    word designVarType = dvSubDict.getWord("designVarType");
    if (designVarType == "ACTD")
    {

        DAFvSource& fvSource = const_cast<DAFvSource&>(
            meshPtr_->thisDb().lookupObject<DAFvSource>("DAFvSource"));

        word diskName = dvSubDict.getWord("actuatorName");

        dictionary fvSourceSubDict = daOptionPtr_->getAllOptions().subDict("fvSource");
        word source = fvSourceSubDict.subDict(diskName).getWord("source");
        if (source == "cylinderAnnulusSmooth")
        {

            this->updateOFField(wVec);
            this->updateOFMesh(xvVec);

            scalarList actDVList(9);
            for (label i = 0; i < 9; i++)
            {
                actDVList[i] = fvSource.getActuatorDVs(diskName, i);
            }

            this->globalADTape_.reset();
            this->globalADTape_.setActive();

            for (label i = 0; i < 9; i++)
            {
                this->globalADTape_.registerInput(actDVList[i]);
            }

            // set dv values to fvSource obj for all procs
            for (label i = 0; i < 9; i++)
            {
                fvSource.setActuatorDVs(diskName, i, actDVList[i]);
            }

            // compute residuals
            daResidualPtr_->correctBoundaryConditions();
            daResidualPtr_->updateIntermediateVariables();
            daModelPtr_->correctBoundaryConditions();
            daModelPtr_->updateIntermediateVariables();
            label isPC = 0;
            dictionary options;
            options.set("isPC", isPC);
            daResidualPtr_->calcResiduals(options);
            daModelPtr_->calcResiduals(options);

            this->registerResidualOutput4AD();
            this->globalADTape_.setPassive();

            this->assignVec2ResidualGradient(psi);
            this->globalADTape_.evaluate();

            for (label i = 0; i < 9; i++)
            {
                PetscScalar valIn = actDVList[i].getGradient();
                // we need to do ADD_VALUES to get contribution from all procs
                VecSetValue(dRdActTPsi, i, valIn, ADD_VALUES);
            }

            VecAssemblyBegin(dRdActTPsi);
            VecAssemblyEnd(dRdActTPsi);

            this->globalADTape_.clearAdjoints();
            this->globalADTape_.reset();
        }
        else
        {
            FatalErrorIn("") << "source not supported. Options: cylinderAnnulusSmooth"
                             << abort(FatalError);
        }
    }
    else
    {
        FatalErrorIn("") << "designVarType not supported. Options: ACTD"
                         << abort(FatalError);
    }
    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dRdActTPsi") || writeJacobians.found("all"))
    {
        word outputName = "dRdActTPsi_" + designVarName;
        DAUtility::writeVectorBinary(dRdActTPsi, outputName);
        DAUtility::writeVectorASCII(dRdActTPsi, outputName);
    }
#endif
}

void DASolver::calcdRdWTPsiAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    Vec dRdWTPsi)
{
#ifdef CODI_AD_REVERSE
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
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

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
#ifdef CODI_AD_REVERSE
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
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

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
#ifdef CODI_AD_REVERSE
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

void DASolver::registerFieldVariableInput4AD(
    const word fieldName,
    const word fieldType)
{
#ifdef CODI_AD_REVERSE
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

void DASolver::registerResidualOutput4AD()
{
#ifdef CODI_AD_REVERSE
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

void DASolver::normalizeGradientVec(Vec vecY)
{
#if defined(CODI_AD_FORWARD) || defined(CODI_AD_REVERSE)
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

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        scalar scalingFactor = normStateDict.getScalar(stateName);

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            vecArray[localIdx] *= scalingFactor.getValue();
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        scalar scalingFactor = normStateDict.getScalar(stateName);

        forAll(meshPtr_->cells(), cellI)
        {
            label localIdx = daIndexPtr_->getLocalAdjointStateIndex(stateName, cellI);
            vecArray[localIdx] *= scalingFactor.getValue();
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
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

    VecRestoreArray(vecY, &vecArray);

#endif
}

void DASolver::assignVec2ResidualGradient(Vec vecX)
{
#if defined(CODI_AD_FORWARD) || defined(CODI_AD_REVERSE)
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

void DASolver::assignStateGradient2Vec(
    Vec vecY,
    const label oldTimeLevel)
{
#if defined(CODI_AD_FORWARD) || defined(CODI_AD_REVERSE)
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
#if defined(CODI_AD_FORWARD) || defined(CODI_AD_REVERSE)
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

label DASolver::checkResidualTol()
{
    /*
    Description:
        Check whether the min residual in primal satisfy the prescribed tolerance
        If yes, return 0 else return 1
    */

    scalar tol = daOptionPtr_->getOption<scalar>("primalMinResTol");
    scalar tolMax = daOptionPtr_->getOption<scalar>("primalMinResTolDiff");
    if (primalMinRes_ / tol > tolMax)
    {
        Info << "********************************************" << endl;
        Info << "Primal min residual " << primalMinRes_ << endl
             << "did not satisfy the prescribed tolerance "
             << tol << endl;
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

void DASolver::setRotingWallVelocity()
{
    /*
    Description:
        If MRF active, set velocity boundary condition for rotating walls
        This function should be called once for each primal solution.
        It should be called AFTER the mesh points are updated
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
            volVectorField& U = const_cast<volVectorField&>(
                meshPtr_->thisDb().lookupObject<volVectorField>("U"));

            wordList nonRotatingPatches;
            MRFProperties.subDict("MRF").readEntry<wordList>("nonRotatingPatches", nonRotatingPatches);

            vector origin;
            MRFProperties.subDict("MRF").readEntry<vector>("origin", origin);
            vector axis;
            MRFProperties.subDict("MRF").readEntry<vector>("axis", axis);
            scalar omega = MRFProperties.subDict("MRF").getScalar("omega");

            forAll(meshPtr_->boundaryMesh(), patchI)
            {
                word bcName = meshPtr_->boundaryMesh()[patchI].name();
                word bcType = meshPtr_->boundaryMesh()[patchI].type();
                if (!DAUtility::isInList<word>(bcName, nonRotatingPatches) && bcType != "processor")
                {
                    Info << "Setting rotating wall velocity for " << bcName << endl;
                    if (U.boundaryField()[patchI].size() > 0)
                    {
                        forAll(U.boundaryField()[patchI], faceI)
                        {
                            vector patchCf = meshPtr_->Cf().boundaryField()[patchI][faceI];
                            U.boundaryFieldRef()[patchI][faceI] =
                                -omega * ((patchCf - origin) ^ (axis / mag(axis)));
                        }
                    }
                }
            }
        }
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
        if (daIndexPtr_->globalCellVectorNumbering.isLocal(globalCellI))
        {
            volVectorField& field =
                const_cast<volVectorField&>(meshPtr_->thisDb().lookupObject<volVectorField>(fieldName));
            label localCellI = daIndexPtr_->globalCellVectorNumbering.toLocal(globalCellI);
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
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();
    label isPC = 0;
    dictionary options;
    options.set("isPC", isPC);
    daResidualPtr_->calcResiduals(options);
    daModelPtr_->calcResiduals(options);

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
        forAll(daOptionPtr_->getAllOptions().subDict("objFunc").toc(), idxI)
        {
            word objFuncName = daOptionPtr_->getAllOptions().subDict("objFunc").toc()[idxI];
            scalar objFuncVal = this->getObjFuncValue(objFuncName);
            objFuncsAllInstances_[timeInstanceI].set(objFuncName, objFuncVal);
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
    forAll(daOptionPtr_->getAllOptions().subDict("objFunc").toc(), idxI)
    {
        word objFuncName = daOptionPtr_->getAllOptions().subDict("objFunc").toc()[idxI];
        scalar objFuncVal = this->getObjFuncValue(objFuncName);
        objFuncsAllInstances_[timeInstanceI].set(objFuncName, objFuncVal);
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
    if (mode == "timeAccurateAdjoint")
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

scalar DASolver::getTimeInstanceObjFunc(
    const label instanceI,
    const word objFuncName)
{
    /*
    Description:
        Return the value of objective function at the given time instance and name
    */

    return objFuncsAllInstances_[instanceI].getScalar(objFuncName);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
