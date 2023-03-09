/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// initialize the static variable, which will be used in forward mode AD
// computation for AOA and BC derivatives
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

    const dictionary& objFuncDict = allOptions.subDict("objFunc");

    // loop over all objFuncs and parts and calc the number of
    // DAObjFunc instances we need
    label nObjFuncInstances = 0;
    forAll(objFuncDict.toc(), idxI)
    {
        word objFunI = objFuncDict.toc()[idxI];
        const dictionary& objFuncSubDict = objFuncDict.subDict(objFunI);
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
        const dictionary& objFuncSubDict = objFuncDict.subDict(objFunI);
        forAll(objFuncSubDict.toc(), idxJ)
        {

            word objPart = objFuncSubDict.toc()[idxJ];
            const dictionary& objFuncSubDictPart = objFuncSubDict.subDict(objPart);

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

void DASolver::calcdXvdXsTPsiAD(
    Vec xvVec,
    Vec psi,
    Vec prod)
{
#ifdef CODI_AD_REVERSE
    this->updateOFMesh(xvVec);

    wordList patchList;
    this->getCouplingPatchList(patchList);

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

    // get the total number of points and faces for the patchList
    label nPoints, nFaces;
    this->getPatchInfo(nPoints, nFaces, patchList);

    scalarList xsList(nFaces * 3);

    label counterFaceI = 0;
    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // Divide force to nodes
            for (label i = 0; i < 3; i++)
            {
                xsList[counterFaceI] = meshPtr_->boundary()[patchI].Cf()[faceI][i];
                counterFaceI++;
            }
        }
    }
    forAll(xsList, idxI)
    {
        this->globalADTape_.registerOutput(xsList[idxI]);
    }

    this->globalADTape_.setPassive();

    PetscScalar* vecArray;
    VecGetArray(psi, &vecArray);
    forAll(xsList, idxI)
    {
        xsList[idxI].setGradient(vecArray[idxI]);
    }
    VecRestoreArray(psi, &vecArray);

    this->globalADTape_.evaluate();

    VecGetArray(prod, &vecArray);
    label counterI = 0;
    forAll(meshPoints, i)
    {
        for (label j = 0; j < 3; j++)
        {
            vecArray[counterI] = meshPoints[i][j].getGradient();
            counterI++;
        }
    }
    VecRestoreArray(prod, &vecArray);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::getFaceCoords(
    Vec xvVec,
    Vec xsVec)
{
    /*
    Description:
        Calculate a list of face center coordinates (xsVec) for the MDO coupling patches, given 
        the volume mesh point coordinates xvVec

    Input:
        xvVec: volume mesh point coordinates
    
    Output:
        xsVec: face center coordinates for coupling patches
    */
    this->updateOFMesh(xvVec);

    wordList patchList;
    this->getCouplingPatchList(patchList);
    // get the total number of points and faces for the patchList
    label nPoints, nFaces;
    this->getPatchInfo(nPoints, nFaces, patchList);

    VecSetSizes(xsVec, nFaces * 3, PETSC_DETERMINE);
    VecSetFromOptions(xsVec);
    VecZeroEntries(xsVec);

    PetscScalar* vecArray;
    VecGetArray(xsVec, &vecArray);

    label counterFaceI = 0;
    PetscScalar val;
    forAll(patchList, cI)
    {
        // get the patch id label
        label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
        forAll(meshPtr_->boundaryMesh()[patchI], faceI)
        {
            // Divide force to nodes
            for (label i = 0; i < 3; i++)
            {
                assignValueCheckAD(val, meshPtr_->boundary()[patchI].Cf()[faceI][i]);
                vecArray[counterFaceI] = val;
                counterFaceI++;
            }
        }
    }

    VecRestoreArray(xsVec, &vecArray);
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

void DASolver::setThermal(
    word varName,
    scalar* thermal)
{
    /*
    Description:
        Assign the temperature or heat flux BC to all of the faces on the conjugate heat 
        transfer patches. 
        NOTE: this function can be called by either fluid or solid domain!

    Inputs:
        varName: either temperature or heatFlux
        thermal: the temperature or heatFlux var on the conjugate heat transfer patch
    
    Outputs:
        The T field in OpenFOAM
    */

    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);

    if (varName == "temperature")
    {
        // here we receive set the temperature from the solid domain (varVec) and will assign fixedValue
        // BC to the fluid domain

        volScalarField& T =
            const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("T"));

        label localFaceI = 0;
        forAll(patchList, cI)
        {
            // get the patch id label
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
            const fvPatch& patch = meshPtr_->boundary()[patchI];
            forAll(patch, faceI)
            {
                T.boundaryFieldRef()[patchI][faceI] = thermal[localFaceI];
                localFaceI++;
            }
        }
    }
    else if (varName == "heatFlux")
    {
        // here we receive heatFlux from the fluid domain (varVec) and will assign the fixedGradient
        // boundary condition to the solid domain

        IOdictionary transportProperties(
            IOobject(
                "transportProperties",
                meshPtr_->time().constant(),
                meshPtr_(),
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false));
        // for incompressible flow, we need to read Cp and rho from transportProperties
        scalar Cp = readScalar(transportProperties.lookup("Cp"));
        scalar rho = readScalar(transportProperties.lookup("rho"));
        scalar DT = readScalar(transportProperties.lookup("DT"));
        scalar coeff = DT * Cp * rho;

        volScalarField& T =
            const_cast<volScalarField&>(meshPtr_->thisDb().lookupObject<volScalarField>("T"));

        label localFaceI = 0;
        forAll(patchList, cI)
        {
            // get the patch id label
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
            fixedGradientFvPatchField<scalar>& patchBC =
                refCast<fixedGradientFvPatchField<scalar>>(T.boundaryFieldRef()[patchI]);

            scalarField& grad = const_cast<scalarField&>(patchBC.gradient());
            forAll(grad, faceI)
            {
                grad[faceI] = thermal[localFaceI] / coeff;
                localFaceI++;
            }
        }
    }
    else
    {
        FatalErrorIn("") << varName << " not valid! "
                         << abort(FatalError);
    }
}

void DASolver::getThermal(
    word varName,
    Vec thermalVec)
{
    /*
    Description:
        Compute the temperature or heat flux for all of the faces on the conjugate heat 
        transfer patches. This routine is a wrapper that exposes the actual computation
        routine to the Python layer using PETSc vectors. For the actual computation
        routine view the getThermalInternal() function.
        NOTE: this function can be called by either fluid or solid domain!

    Inputs:
        varName: either temperature or heatFlux

    Output:
        thermalVec: the temperature or heatFlux vector on the conjugate heat transfer patch
    */

    List<scalar> thermalList;

    this->getThermalInternal(varName, thermalList);

    // Zero PETSc Arrays
    VecZeroEntries(thermalVec);

    // Get PETSc arrays
    PetscScalar* vecArray;
    VecGetArray(thermalVec, &vecArray);

    // Transfer to PETSc Array
    PetscScalar val;
    forAll(thermalList, cI)
    {
        // Get Values
        assignValueCheckAD(val, thermalList[cI]);
        // Set Values
        vecArray[cI] = val;
    }
    VecRestoreArray(thermalVec, &vecArray);
}

void DASolver::getThermalInternal(
    word varName,
    scalarList& thermalList)
{
    /*
    Description:
        Same as getThermal, except that this function can be used in AD

    Inputs:
        varName: either temperature or heatFlux

    Output:
        thermalList: the temperature or heatFlux list on the conjugate heat transfer patch
    */

    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);

    thermalList.setSize(nFaces);

    if (varName == "temperature")
    {
        volScalarField temperatureField(
            IOobject(
                "temperatureField",
                meshPtr_->time().timeName(),
                meshPtr_(),
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            meshPtr_(),
            dimensionedScalar("temperatureField", dimensionSet(0, 0, 0, 0, 0, 0, 0), 0.0),
            fixedValueFvPatchScalarField::typeName);

        const objectRegistry& db = meshPtr_->thisDb();
        const volScalarField& T = db.lookupObject<volScalarField>("T");

        label localFaceI = 0;
        forAll(patchList, cI)
        {
            // get the patch id label
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
            const fvPatch& patch = meshPtr_->boundary()[patchI];
            forAll(patch, faceI)
            {
                temperatureField.boundaryFieldRef()[patchI][faceI] = T.boundaryField()[patchI][faceI];
                thermalList[localFaceI] = T.boundaryField()[patchI][faceI];
                localFaceI++;
            }
        }

        // this is for debugging
        temperatureField.write();
    }
    else if (varName == "heatFlux")
    {

#ifdef IncompressibleFlow

        volScalarField heatFluxField(
            IOobject(
                "heatFluxField",
                meshPtr_->time().timeName(),
                meshPtr_(),
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            meshPtr_(),
            dimensionedScalar("heatFluxField", dimensionSet(1, -2, 1, 1, 0, 0, 0), 0.0),
            "calculated");

        DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
        volScalarField alphaEff = daTurb.alphaEff();
        // incompressible flow does not have he, so we do H = Cp * alphaEff * dT/dz
        // initialize the Prandtl number from transportProperties

        IOdictionary transportProperties(
            IOobject(
                "transportProperties",
                meshPtr_->time().constant(),
                meshPtr_(),
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false));
        // for incompressible flow, we need to read Cp from transportProperties
        scalar Cp = readScalar(transportProperties.lookup("Cp"));

        const objectRegistry& db = meshPtr_->thisDb();
        const volScalarField& T = db.lookupObject<volScalarField>("T");

        const volScalarField::Boundary& TBf = T.boundaryField();
        const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();

        label localFaceI = 0;
        forAll(patchList, cI)
        {
            // get the patch id label
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
            heatFluxField.boundaryFieldRef()[patchI] = Cp * alphaEffBf[patchI] * TBf[patchI].snGrad();

            forAll(meshPtr_->boundaryMesh()[patchI], faceI)
            {
                thermalList[localFaceI] = heatFluxField.boundaryField()[patchI][faceI];
                localFaceI++;
            }
        }

        // for debugging
        heatFluxField.write();

#endif

#ifdef CompressibleFlow

        volScalarField heatFluxField(
            IOobject(
                "heatFluxField",
                meshPtr_->time().timeName(),
                meshPtr_(),
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            meshPtr_(),
            dimensionedScalar("heatFluxField", dimensionSet(1, 0, -3, 0, 0, 0, 0), 0.0),
            "calculated");

        DATurbulenceModel& daTurb = const_cast<DATurbulenceModel&>(daModelPtr_->getDATurbulenceModel());
        volScalarField alphaEff = daTurb.alphaEff();
        // compressible flow, H = alphaEff * dHE/dz
        fluidThermo& thermo = const_cast<fluidThermo&>(daModelPtr_->getThermo());
        volScalarField& he = thermo.he();
        const volScalarField::Boundary& heBf = he.boundaryField();
        const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();

        label localFaceI = 0;
        forAll(patchList, cI)
        {
            // get the patch id label
            label patchI = meshPtr_->boundaryMesh().findPatchID(patchList[cI]);
            heatFluxField.boundaryFieldRef()[patchI] = alphaEffBf[patchI] * heBf[patchI].snGrad();

            forAll(meshPtr_->boundaryMesh()[patchI], faceI)
            {
                thermalList[localFaceI] = heatFluxField.boundaryField()[patchI][faceI];
                localFaceI++;
            }
        }

        // for debugging
        heatFluxField.write();
#endif
    }
    else
    {
        FatalErrorIn("getPatchVarInternal") << " varName not valid. "
                                            << abort(FatalError);
    }
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
    Vec center,
    Vec aForceL,
    Vec tForceL,
    Vec rDistL)
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

    /*

    // Get Data
    label nPoints = daOptionPtr_->getSubDictOption<scalar>("wingProp", "nForceSections");
    fvMesh& mesh = meshPtr_();

    // Allocate Arrays
    Vector<scalar> centerTemp;
    Field<scalar> aForceTemp(nPoints);
    Field<scalar> tForceTemp(nPoints);
    List<scalar> rDistLTemp(nPoints+2);

    // Get PETSc Arrays
    PetscScalar* vecArrayCenter;
    VecGetArray(center, &vecArrayCenter);

    // Set Values
    centerTemp[0] = vecArrayCenter[0];
    centerTemp[1] = vecArrayCenter[1];
    centerTemp[2] = vecArrayCenter[2];

    // Compute force profiles
    this->calcForceProfileInternal(mesh, centerTemp, aForceTemp, tForceTemp, rDistLTemp);

    VecZeroEntries(aForceL);
    VecZeroEntries(tForceL);
    VecZeroEntries(rDistL);
    PetscScalar* vecArrayAForceL;
    VecGetArray(aForceL, &vecArrayAForceL);
    PetscScalar* vecArrayTForceL;
    VecGetArray(tForceL, &vecArrayTForceL);
    PetscScalar* vecArrayRDistL;
    VecGetArray(rDistL, &vecArrayRDistL);

    // Tranfer to PETSc Array for force profiles and radius
    forAll(aForceTemp, cI)
    {
        // Get Values
        PetscScalar val1, val2;
        assignValueCheckAD(val1, aForceTemp[cI]);
        assignValueCheckAD(val2, tForceTemp[cI]);

        // Set Values
        vecArrayAForceL[cI] = val1;
        vecArrayTForceL[cI] = val2;
    }
    forAll(aForceTemp, cI)
    {
        // Get Values
        PetscScalar val1;
        assignValueCheckAD(val1, rDistLTemp[cI]);

        // Set Values
        vecArrayRDistL[cI] = val1;
    }

    VecRestoreArray(center, &vecArrayCenter);
    VecRestoreArray(aForceL, &vecArrayAForceL);
    VecRestoreArray(tForceL, &vecArrayTForceL);
    VecRestoreArray(rDistL, &vecArrayRDistL);

    return;
    */
}

void DASolver::calcForceProfileInternal(
    fvMesh& mesh,
    const vector& center,
    scalarList& aForceL,
    scalarList& tForceL,
    scalarList& rDistL)
{
    /*
    Description:
        Same as calcForceProfile but for internal AD
    */

    /*

    // fvMesh& mesh = meshPtr_();
    scalar sections = daOptionPtr_->getSubDictOption<scalar>("wingProp", "nForceSections");
    scalarList axisDummy = daOptionPtr_->getSubDictOption<scalarList>("wingProp", "axis");
    vector axis;
    axis[0] = axisDummy[0];
    axis[1] = axisDummy[1];
    axis[2] = axisDummy[2];

    int quot;
    vector cellDir, projP, cellCen;
    scalar length, axialDist;

    // get the pressure in the memory
    const volScalarField& p = mesh.thisDb().lookupObject<volScalarField>("p");

    // name of the blade
    word bladePatchName = "propeller";

    // find the patch ID of the blade surface
    label bladePatchI = mesh.boundaryMesh().findPatchID(bladePatchName);

    // radiiCell initialization, cell radii will be stored in it
    scalarList radiiCell(p.boundaryField()[bladePatchI].size());

    // meshTanDir initialization, mesh tangential direction will be stored in it
    vectorField meshTanDir = mesh.Cf().boundaryField()[bladePatchI] * 0;

    // radius limits initialization
    scalar minRadius = 1000000;
    scalar minRadiIndx = -1;
    scalar maxRadius = -1000000;
    scalar maxRadiIndx = -1;

    forAll(p.boundaryField()[bladePatchI], faceI)
    {
        // directional vector from the propeller center to the cell center & dictance between them
        cellCen = mesh.Cf().boundaryField()[bladePatchI][faceI];
        cellDir = cellCen - center;
        length = Foam:: sqrt(sqr(cellDir[0])+sqr(cellDir[1])+sqr(cellDir[2]));

        // unit vector conversion
        cellDir = cellDir / length;

        // axial distance between the propeller center and the cell center
        axialDist = (cellDir & axis) * length;

        // projected point of the cell center on the axis
        projP = {center[0] + axis[0] * axialDist, center[1] + axis[1] * axialDist, center[2] + axis[2] * axialDist};

        // radius of the cell center
        radiiCell[faceI] = Foam::sqrt(sqr(cellCen[0] - projP[0]) + sqr(cellCen[1] - projP[1]) + sqr(cellCen[2] - projP[2]));

        if(radiiCell[faceI] < minRadius)
        {
            minRadius = radiiCell[faceI];
            minRadiIndx = faceI;
        }
        if(radiiCell[faceI] > maxRadius)
        {
            maxRadius = radiiCell[faceI];
            maxRadiIndx = faceI;
        }

        // storing tangential vector as a unit vector
        meshTanDir[faceI] = axis ^ cellDir;
        length = Foam:: sqrt(sqr(meshTanDir[faceI][0])+sqr(meshTanDir[faceI][1])+sqr(meshTanDir[faceI][2]));
        meshTanDir[faceI] = meshTanDir[faceI] / length;
    }

    // generating empty lists
    scalarList axialForce(sections);
    forAll(axialForce, Index){axialForce[Index] = 0;}
    scalarList tangtForce = axialForce;
    scalarList radialDist(sections + 2);

    // sectional radius computation
    scalar sectRad = (maxRadius - minRadius) / sections;
    radialDist[0] = minRadius;
    radialDist[sections + 1] = maxRadius;
    for(int Index = 1; Index < sections + 1; Index++){radialDist[Index] = minRadius + sectRad * (Index - 0.5);}
    scalarList counter = axialForce;

    // computation of forces
    forAll(p.boundaryField()[bladePatchI], faceI)
    {
        // finding the section of the cell
        quot = (radiiCell[faceI] - minRadius) / sectRad;
        if(quot == sections){quot = quot - 1;}

        // pressure direction is opposite of the surface normal
        axialForce[quot] = axialForce[quot] - (mesh.Sf().boundaryField()[bladePatchI][faceI] & axis) * p.boundaryField()[bladePatchI][faceI] / sectRad;
        tangtForce[quot] = tangtForce[quot] - (mesh.Sf().boundaryField()[bladePatchI][faceI] & meshTanDir[faceI]) * p.boundaryField()[bladePatchI][faceI] / sectRad;
        counter[quot] = counter[quot] + 1;
    }
    */
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
    const scalarField& aForce,
    const scalarField& tForce,
    const scalarList& rDistExt,
    const scalarList& targetForce,
    const vector& center,
    volVectorField& fvSource)
{
    /*
    Description:
        Smoothing the force distribution on propeller blade on the entire mesh to ensure that it will not diverge during optimization.
        Forces are smoothed using polynomial distribution for inner radius, normal distribution for outer radius, and Gaussiam distribution for axial direction.
    Inputs:
        aForceL: Axis force didtribution on propeller blade
        tForceL: Tangential force distribution on propeller blade
        rDist: Force distribution locations and radii of propeller (first element is inner radius, last element is outer radius)
    Output:
        fvSource: Smoothed forces in each mesh cell
    */

    vector axis;
    scalar actEps = daOptionPtr_->getSubDictOption<scalar>("wingProp", "actEps");
    word rotDir = daOptionPtr_->getSubDictOption<word>("wingProp", "rotDir");
    scalarList axisDummy = daOptionPtr_->getSubDictOption<scalarList>("wingProp", "axis");
    axis[0] = axisDummy[0];
    axis[1] = axisDummy[1];
    axis[2] = axisDummy[2];

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

    // Extraction of inner and outer radii, and resizing of the blade radius distribution.
    // scalar rInner = rDistExt[0];
    scalar rOuter = rDistExt[rDistExt.size() - 1];
    scalarField rDist = aForce * 0.0; // real blade radius distribution
    scalarField rNorm = rDist; // normalized blade radius distribution
    forAll(aForce, index)
    {
        rDist[index] = rDistExt[index + 1];
    }
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
    scalar coefAAxialIn = (f3 * r4 - f4 * r3) / (r3 * r4 * (r3 - r4));
    scalar coefBAxialIn = (f3 - coefAAxialIn * r3 * r3) / r3;

    // Tangential Inner
    scalar coefATangentialIn = (g3 * r4 - g4 * r3) / (r3 * r4 * (r3 - r4));
    scalar coefBTangentialIn = (g3 - coefATangentialIn * r3 * r3) / r3;

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
            fvSource[cellI] = ((coefAAxialIn * rStar * rStar + coefBAxialIn * rStar) * axis + (coefATangentialIn * rStar * rStar + coefBTangentialIn * rStar) * cellAxDir * rotDirCon) * exp(-sqr(meshDist / actEps));
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

void DASolver::calcFvSource(
    Vec aForce,
    Vec tForce,
    Vec rDistExt,
    Vec targetForce,
    Vec center,
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
    label nPoints = daOptionPtr_->getSubDictOption<label>("wingProp", "nForceSections");
    // label meshSize = meshPtr_->nCells();

    // Allocate Arrays
    Field<scalar> aForceTemp(nPoints);
    Field<scalar> tForceTemp(nPoints);
    List<scalar> rDistExtTemp(nPoints + 2);
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
    PetscScalar* vecArrayRDistExt;
    VecGetArray(rDistExt, &vecArrayRDistExt);
    PetscScalar* vecArrayTargetForce;
    VecGetArray(targetForce, &vecArrayTargetForce);
    PetscScalar* vecArrayCenter;
    VecGetArray(center, &vecArrayCenter);

    // Set Values
    forAll(aForceTemp, cI)
    {
        aForceTemp[cI] = vecArrayAForce[cI];
        tForceTemp[cI] = vecArrayTForce[cI];
    }

    forAll(rDistExtTemp, cI)
    {
        rDistExtTemp[cI] = vecArrayRDistExt[cI];
    }
    targetForceTemp[0] = vecArrayTargetForce[0];
    targetForceTemp[1] = vecArrayTargetForce[1];
    centerTemp[0] = vecArrayCenter[0];
    centerTemp[1] = vecArrayCenter[1];
    centerTemp[2] = vecArrayCenter[2];

    // Compute fvSource
    this->calcFvSourceInternal(aForceTemp, tForceTemp, rDistExtTemp, targetForceTemp, centerTemp, fvSourceTemp);

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
    VecRestoreArray(rDistExt, &vecArrayRDistExt);
    VecRestoreArray(targetForce, &vecArrayTargetForce);
    VecRestoreArray(center, &vecArrayCenter);
    VecRestoreArray(fvSource, &vecArrayFvSource);

    return;
}

void DASolver::calcdFvSourcedInputsTPsiAD(
    const word mode,
    Vec centerVec,
    Vec radiusVec,
    Vec forceVec,
    Vec psiVec,
    Vec prodVec)
{
    /*
    Description:
        Calculate the matrix-vector product for either [dFvSource/dParameters]^T * psi, or [dFvSource/dForce]^T * psi
    */
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

    // create a common variable BC, it will be the input of the AD
    scalar BC = -1e16;

    if (designVarName == "MRF")
    {
        const IOMRFZoneListDF& MRF = meshPtr_->thisDb().lookupObject<IOMRFZoneListDF>("MRFProperties");

        // first, we get the current value of omega and assign it to BC
        scalar& omega = const_cast<scalar&>(MRF.getOmegaRef());
        BC = omega;

        this->globalADTape_.reset();
        this->globalADTape_.setActive();
        // register BC as the input
        this->globalADTape_.registerInput(BC);
        // ******* now set BC ******
        omega = BC;
    }
    else
    {

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

void DASolver::calcdFdBCAD(
    const Vec xvVec,
    const Vec wVec,
    const word objFuncName,
    const word designVarName,
    Vec dFdBC)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        This function computes partials derivatives dFdBC
    
    Input:
        xvVec: the volume mesh coordinate vector
        wVec: the state variable vector
        objFuncName: name of the objective function F
        designVarName: the name of the design variable
    
    Output:
        dFdBC: the partial derivative vector dF/dBC
        NOTE: You need to fully initialize the dF vec before calliing this function,
        i.e., VecCreate, VecSetSize, VecSetFromOptions etc. Or call VeDuplicate
    */

    Info << "Calculating dFdBC using reverse-mode AD for " << designVarName << endl;

    VecZeroEntries(dFdBC);

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    dictionary designVarDict = daOptionPtr_->getAllOptions().subDict("designVar");

    // get the subDict for this dvName
    dictionary dvSubDict = designVarDict.subDict(designVarName);

    scalar BC = -1e16;

    // now we need to get the BC value
    if (designVarName == "MRF")
    {
        const IOMRFZoneListDF& MRF = meshPtr_->thisDb().lookupObject<IOMRFZoneListDF>("MRFProperties");
        // first, we get the current value of omega and assign it to BC
        scalar& omega = const_cast<scalar&>(MRF.getOmegaRef());
        BC = omega;
    }
    else
    {
        // get info from dvSubDict. This needs to be defined in the pyDAFoam
        // name of the variable for changing the boundary condition
        word varName = dvSubDict.getWord("variable");
        // name of the boundary patch
        wordList patches;
        dvSubDict.readEntry<wordList>("patches", patches);
        // the compoent of a vector variable, ignore when it is a scalar
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
        // need to reduce the BC value across all processors, this is because some of
        // the processors might not own the prescribed patches so their BC value will be still -1e16, but
        // when calling the following reduce function, they will get the correct BC from other processors
        reduce(BC, maxOp<scalar>());
    }

    // get the subDict for this objective function
    dictionary objFuncSubDict =
        daOptionPtr_->getAllOptions().subDict("objFunc").subDict(objFuncName);
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
        // register BC as the input
        this->globalADTape_.registerInput(BC);
        // now set BC
        if (designVarName == "MRF")
        {
            const IOMRFZoneListDF& MRF = meshPtr_->thisDb().lookupObject<IOMRFZoneListDF>("MRFProperties");
            // first, we get the current value of omega and assign it to BC
            scalar& omega = const_cast<scalar&>(MRF.getOmegaRef());
            omega = BC;
        }
        else
        {
            // get info from dvSubDict. This needs to be defined in the pyDAFoam
            // name of the variable for changing the boundary condition
            word varName = dvSubDict.getWord("variable");
            // name of the boundary patch
            wordList patches;
            dvSubDict.readEntry<wordList>("patches", patches);
            // the compoent of a vector variable, ignore when it is a scalar
            label comp = dvSubDict.getLabel("comp");
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
        Vec dFdBCPart;
        VecDuplicate(dFdBC, &dFdBCPart);
        VecZeroEntries(dFdBCPart);
        PetscScalar derivVal = BC.getGradient();
        // we need to do ADD_VALUES to get contribution from all procs
        VecSetValue(dFdBCPart, 0, derivVal, ADD_VALUES);
        VecAssemblyBegin(dFdBCPart);
        VecAssemblyEnd(dFdBCPart);

        // need to clear adjoint and tape after the computation is done!
        this->globalADTape_.clearAdjoints();
        this->globalADTape_.reset();

        // we need to add dFdBCPart to dFdBC because we want to sum
        // all dFdBCPart for all parts of this objFuncName.
        VecAXPY(dFdBC, 1.0, dFdBCPart);

        if (daOptionPtr_->getOption<label>("debug"))
        {
            Info << "In calcdFdBCAD" << endl;
            this->calcPrimalResidualStatistics("print");
            Info << objFuncName << ": " << fRef << endl;
        }

        VecDestroy(&dFdBCPart);
    }

    wordList writeJacobians;
    daOptionPtr_->getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdBC") || writeJacobians.found("all"))
    {
        word outputName = "dFdBC_" + designVarName + "_" + objFuncName;
        DAUtility::writeVectorBinary(dFdBC, outputName);
        DAUtility::writeVectorASCII(dFdBC, outputName);
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

    label printInfo = 0;
    if (daOptionPtr_->getOption<label>("debug"))
    {
        Info << "Updating the OpenFOAM field..." << endl;
        printInfo = 1;
    }
    this->setPrimalBoundaryConditions(printInfo);
    daFieldPtr_->stateVec2OFField(wVec);
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

void DASolver::calcdRdThermalTPsiAD(
    const word varName,
    const Vec xvVec,
    const Vec wVec,
    const Vec psi,
    const Vec thermalVec,
    Vec prodVec)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Compute the matrix-vector products dRdThermal^T*Psi using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psi: the vector to multiply dRdXv
    
    Output:
        prodVec: the matrix-vector products dRdThermal^T * Psi
    */

    Info << "Calculating [dRdThermal]^T * Psi using reverse-mode AD" << endl;

    VecZeroEntries(prodVec);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);

    const PetscScalar* thermalArray;
    VecGetArrayRead(thermalVec, &thermalArray);
    scalar* thermal = new scalar[nFaces];
    for (label i = 0; i < nFaces; i++)
    {
        thermal[i] = thermalArray[i];
    }
    VecRestoreArrayRead(thermalVec, &thermalArray);

    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    for (label i = 0; i < nFaces; i++)
    {
        this->globalADTape_.registerInput(thermal[i]);
    }

    this->setThermal(varName, thermal);

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

    PetscScalar* vecArray;
    VecGetArray(prodVec, &vecArray);
    for (label i = 0; i < nFaces; i++)
    {
        vecArray[i] = thermal[i].getGradient();
    }

    VecRestoreArray(prodVec, &vecArray);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
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

void DASolver::calcdForcedXvAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec fBarVec,
    Vec dForcedXv)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Calculate dForcedXv using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        fBarVec: the derivative seed vector
    
    Output:
        dForcedXv: dForce/dXv
    */

    Info << "Calculating dForcedXvAD using reverse-mode AD" << endl;

    VecZeroEntries(dForcedXv);

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

    // Allocate arrays
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);
    List<scalar> fX(nPoints);
    List<scalar> fY(nPoints);
    List<scalar> fZ(nPoints);

    this->getForcesInternal(fX, fY, fZ, patchList);
    this->registerForceOutput4AD(fX, fY, fZ);
    this->globalADTape_.setPassive();

    this->assignVec2ForceGradient(fBarVec, fX, fY, fZ);
    this->globalADTape_.evaluate();

    forAll(meshPoints, i)
    {
        for (label j = 0; j < 3; j++)
        {
            label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
            PetscScalar val = meshPoints[i][j].getGradient();
            VecSetValue(dForcedXv, rowI, val, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(dForcedXv);
    VecAssemblyEnd(dForcedXv);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::calcdAcousticsdXvAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec fBarVec,
    Vec dAcoudXv,
    const word varName,
    const word groupName)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Calculate dAcoudXv using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        fBarVec: the derivative seed vector
    
    Output:
        dAcouXv: dAcou/dXv
    */

    Info << "Calculating dAcoudXvAD using reverse-mode AD" << endl;

    VecZeroEntries(dAcoudXv);

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

    // Allocate arrays
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList, groupName);
    this->getPatchInfo(nPoints, nFaces, patchList);
    List<scalar> x(nFaces);
    List<scalar> y(nFaces);
    List<scalar> z(nFaces);
    List<scalar> nX(nFaces);
    List<scalar> nY(nFaces);
    List<scalar> nZ(nFaces);
    List<scalar> a(nFaces);
    List<scalar> fX(nFaces);
    List<scalar> fY(nFaces);
    List<scalar> fZ(nFaces);

    this->getAcousticDataInternal(x, y, z, nX, nY, nZ, a, fX, fY, fZ, patchList);

    if (varName == "xAcou")
    {
        this->registerAcousticOutput4AD(x);
        this->registerAcousticOutput4AD(y);
        this->registerAcousticOutput4AD(z);
    }
    else if (varName == "nAcou")
    {
        this->registerAcousticOutput4AD(nX);
        this->registerAcousticOutput4AD(nY);
        this->registerAcousticOutput4AD(nZ);
    }
    else if (varName == "aAcou")
    {
        this->registerAcousticOutput4AD(a);
    }
    else if (varName == "fAcou")
    {
        this->registerAcousticOutput4AD(fX);
        this->registerAcousticOutput4AD(fY);
        this->registerAcousticOutput4AD(fZ);
    }
    this->globalADTape_.setPassive();

    if (varName == "xAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, x, 0, 3);
        this->assignVec2AcousticGradient(fBarVec, y, 1, 3);
        this->assignVec2AcousticGradient(fBarVec, z, 2, 3);
    }
    else if (varName == "nAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, nX, 0, 3);
        this->assignVec2AcousticGradient(fBarVec, nY, 1, 3);
        this->assignVec2AcousticGradient(fBarVec, nZ, 2, 3);
    }
    else if (varName == "aAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, a, 0, 1);
    }
    else if (varName == "fAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, fX, 0, 3);
        this->assignVec2AcousticGradient(fBarVec, fY, 1, 3);
        this->assignVec2AcousticGradient(fBarVec, fZ, 2, 3);
    }
    this->globalADTape_.evaluate();

    forAll(meshPoints, i)
    {
        for (label j = 0; j < 3; j++)
        {
            label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
            PetscScalar val = meshPoints[i][j].getGradient();
            VecSetValue(dAcoudXv, rowI, val, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(dAcoudXv);
    VecAssemblyEnd(dAcoudXv);

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
                scalarList actDVList(10);
                for (label i = 0; i < 10; i++)
                {
                    actDVList[i] = fvSource.getActuatorDVs(diskName, i);
                }

                // reset tape
                this->globalADTape_.reset();
                // activate tape, start recording
                this->globalADTape_.setActive();
                // register  the input
                for (label i = 0; i < 10; i++)
                {
                    this->globalADTape_.registerInput(actDVList[i]);
                }
                // set dv values to fvSource obj for all procs
                for (label i = 0; i < 10; i++)
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

                for (label i = 0; i < 10; i++)
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

            scalarList actDVList(10);
            for (label i = 0; i < 10; i++)
            {
                actDVList[i] = fvSource.getActuatorDVs(diskName, i);
            }

            this->globalADTape_.reset();
            this->globalADTape_.setActive();

            for (label i = 0; i < 10; i++)
            {
                this->globalADTape_.registerInput(actDVList[i]);
            }

            // set dv values to fvSource obj for all procs
            for (label i = 0; i < 10; i++)
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

            for (label i = 0; i < 10; i++)
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

void DASolver::calcdThermaldWTPsiAD(
    const word mode,
    const Vec xvVec,
    const Vec wVec,
    const Vec psiVec,
    Vec prodVec)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Calculate dThermaldW using reverse-mode AD. Mode can be either temperature or heatFlux
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psiVec: the derivative seed vector
    
    Output:
        prodVec: Either dTemperature/dW or dHeatFlux/dW
    */

    Info << "Calculating dThermaldW using reverse-mode AD" << endl;

    VecZeroEntries(prodVec);

    // this is needed because the self.solverAD object in the Python layer
    // never run the primal solution, so the wVec and xvVec is not always
    // update to date
    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    // Allocate arrays
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);

    scalarList thermalList(nFaces);

    this->globalADTape_.reset();
    this->globalADTape_.setActive();

    this->registerStateVariableInput4AD();

    // compute residuals
    daResidualPtr_->correctBoundaryConditions();
    daResidualPtr_->updateIntermediateVariables();
    daModelPtr_->correctBoundaryConditions();
    daModelPtr_->updateIntermediateVariables();

    this->getThermalInternal(mode, thermalList);
    forAll(thermalList, idxI)
    {
        this->globalADTape_.registerOutput(thermalList[idxI]);
    }
    this->globalADTape_.setPassive();

    PetscScalar* vecArray;
    VecGetArray(psiVec, &vecArray);

    forAll(thermalList, idxI)
    {
        thermalList[idxI].setGradient(vecArray[idxI]);
    }
    VecRestoreArray(psiVec, &vecArray);
    this->globalADTape_.evaluate();

    // get the deriv values
    this->assignStateGradient2Vec(prodVec);

    // NOTE: we need to normalize dForcedW!
    this->normalizeGradientVec(prodVec);

    VecAssemblyBegin(prodVec);
    VecAssemblyEnd(prodVec);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::calcdThermaldXvTPsiAD(
    const word mode,
    const Vec xvVec,
    const Vec wVec,
    const Vec psiVec,
    Vec prodVec)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Calculate [dThermal/dXv]^T * Psi using reverse-mode AD
    
    Input:

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        psiVec: the derivative seed vector
    
    Output:
        prodVec: [dThermal/dXv]^T * Psi
    */

    Info << "Calculating dThermaldXvAD using reverse-mode AD" << endl;

    VecZeroEntries(prodVec);

    this->updateOFField(wVec);
    this->updateOFMesh(xvVec);

    // Allocate arrays
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);

    scalarList thermalList(nFaces);

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

    this->getThermalInternal(mode, thermalList);

    forAll(thermalList, idxI)
    {
        this->globalADTape_.registerOutput(thermalList[idxI]);
    }
    this->globalADTape_.setPassive();

    PetscScalar* vecArray;
    VecGetArray(psiVec, &vecArray);

    forAll(thermalList, idxI)
    {
        thermalList[idxI].setGradient(vecArray[idxI]);
    }
    VecRestoreArray(psiVec, &vecArray);
    this->globalADTape_.evaluate();

    forAll(meshPoints, i)
    {
        for (label j = 0; j < 3; j++)
        {
            label rowI = daIndexPtr_->getGlobalXvIndex(i, j);
            PetscScalar val = meshPoints[i][j].getGradient();
            VecSetValue(prodVec, rowI, val, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(prodVec);
    VecAssemblyEnd(prodVec);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::calcdForcedWAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec fBarVec,
    Vec dForcedW)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Calculate dForcedW using reverse-mode AD
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        fBarVec: the derivative seed vector
    
    Output:
        dForcedW: dForce/dW
    */

    Info << "Calculating dForcesdW using reverse-mode AD" << endl;

    VecZeroEntries(dForcedW);

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

    // Allocate arrays
    label nPoints, nFaces;
    List<word> patchList;
   this->getCouplingPatchList(patchList);
    this->getPatchInfo(nPoints, nFaces, patchList);
    List<scalar> fX(nPoints);
    List<scalar> fY(nPoints);
    List<scalar> fZ(nPoints);

    this->getForcesInternal(fX, fY, fZ, patchList);
    this->registerForceOutput4AD(fX, fY, fZ);
    this->globalADTape_.setPassive();

    this->assignVec2ForceGradient(fBarVec, fX, fY, fZ);
    this->globalADTape_.evaluate();

    // get the deriv values
    this->assignStateGradient2Vec(dForcedW);

    // NOTE: we need to normalize dForcedW!
    this->normalizeGradientVec(dForcedW);

    VecAssemblyBegin(dForcedW);
    VecAssemblyEnd(dForcedW);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::calcdAcousticsdWAD(
    const Vec xvVec,
    const Vec wVec,
    const Vec fBarVec,
    Vec dAcoudW,
    word varName,
    word groupName)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Calculate dForcedW using reverse-mode AD
    
    Input:
        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector

        fBarVec: the derivative seed vector
    
    Output:
        dAcoudW: dAcou/dW
    */

    Info << "Calculating dAcoudW using reverse-mode AD" << endl;

    VecZeroEntries(dAcoudW);

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

    // Allocate arrays
    label nPoints, nFaces;
    List<word> patchList;
    this->getCouplingPatchList(patchList, groupName);
    this->getPatchInfo(nPoints, nFaces, patchList);
    List<scalar> x(nFaces);
    List<scalar> y(nFaces);
    List<scalar> z(nFaces);
    List<scalar> nX(nFaces);
    List<scalar> nY(nFaces);
    List<scalar> nZ(nFaces);
    List<scalar> a(nFaces);
    List<scalar> fX(nFaces);
    List<scalar> fY(nFaces);
    List<scalar> fZ(nFaces);

    this->getAcousticDataInternal(x, y, z, nX, nY, nZ, a, fX, fY, fZ, patchList);

    if (varName == "xAcou")
    {
        this->registerAcousticOutput4AD(x);
        this->registerAcousticOutput4AD(y);
        this->registerAcousticOutput4AD(z);
    }
    else if (varName == "nAcou")
    {
        this->registerAcousticOutput4AD(nX);
        this->registerAcousticOutput4AD(nY);
        this->registerAcousticOutput4AD(nZ);
    }
    else if (varName == "aAcou")
    {
        this->registerAcousticOutput4AD(a);
    }
    else if (varName == "fAcou")
    {
        this->registerAcousticOutput4AD(fX);
        this->registerAcousticOutput4AD(fY);
        this->registerAcousticOutput4AD(fZ);
    }
    this->globalADTape_.setPassive();

    if (varName == "xAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, x, 0, 3);
        this->assignVec2AcousticGradient(fBarVec, y, 1, 3);
        this->assignVec2AcousticGradient(fBarVec, z, 2, 3);
    }
    else if (varName == "nAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, nX, 0, 3);
        this->assignVec2AcousticGradient(fBarVec, nY, 1, 3);
        this->assignVec2AcousticGradient(fBarVec, nZ, 2, 3);
    }
    else if (varName == "aAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, a, 0, 1);
    }
    else if (varName == "fAcou")
    {
        this->assignVec2AcousticGradient(fBarVec, fX, 0, 3);
        this->assignVec2AcousticGradient(fBarVec, fY, 1, 3);
        this->assignVec2AcousticGradient(fBarVec, fZ, 2, 3);
    }
    this->globalADTape_.evaluate();

    // get the deriv values
    this->assignStateGradient2Vec(dAcoudW);

    // NOTE: we need to normalize dAcoudW!
    this->normalizeGradientVec(dAcoudW);

    VecAssemblyBegin(dAcoudW);
    VecAssemblyEnd(dAcoudW);

    this->globalADTape_.clearAdjoints();
    this->globalADTape_.reset();
#endif
}

void DASolver::calcdRdWTPsiAD(
    const label isInit,
    const Vec psi,
    Vec dRdWTPsi)
{
#ifdef CODI_AD_REVERSE
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

void DASolver::registerForceOutput4AD(
    List<scalar>& fX,
    List<scalar>& fY,
    List<scalar>& fZ)
{
#if defined(CODI_AD_REVERSE)
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
#if defined(CODI_AD_REVERSE)
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

void DASolver::assignVec2ForceGradient(
    Vec fBarVec,
    List<scalar>& fX,
    List<scalar>& fY,
    List<scalar>& fZ)
{
#if defined(CODI_AD_FORWARD) || defined(CODI_AD_REVERSE)
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
#if defined(CODI_AD_FORWARD) || defined(CODI_AD_REVERSE)
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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
