/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointDerivative.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivative, 0);
defineRunTimeSelectionTable(AdjointDerivative, dictionary);

#ifdef CompressibleFlow

// Constructors for compressible flow
AdjointDerivative::AdjointDerivative
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj,
    fluidThermo& thermo
)
    :
    mesh_(mesh),
    adjIO_(adjIO),
    adjReg_(adjReg),
    adjRAS_(adjRAS),
    adjIdx_(adjIdx),
    adjCon_(adjCon),
    adjObj_(adjObj),
    thermo_(thermo),
    db_(mesh.thisDb()),
    pointsRef_(mesh.points()),
    MRF_(mesh_)

{

}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<AdjointDerivative> AdjointDerivative::New
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj,
    fluidThermo& thermo
)
{
    // get model name, but do not register the dictionary
    // otherwise it is registered in the database twice
    const word modelType
    (
        IOdictionary
        (
            IOobject
            (
                "controlDict",
                mesh.time().system(),
                mesh,
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE,
                false
            )
        ).lookup("application")
    );
    
    Info<< "Selecting " << modelType<<" for AdjointDerivative" << endl;

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn
        (
            "AdjointDerivative::New"
            "("
            "    fvMesh& mesh,"
            "    const AdjointIO& adjIO,"
            "    const AdjointSolverRegistry& adjReg,"
            "    AdjointRASModel& adjRAS,"
            "    AdjointIndexing& adjIdx,"
            "    AdjointJacobianConnectivity& adjCon,"
            "    AdjointObjectiveFunction& adjObj"
            ")"
        )   << "Unknown AdjointDerivative type "
            << modelType << nl << nl
            << "Valid AdjointDerivative types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<AdjointDerivative>
           (
               cstrIter()(mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj,thermo)
           );
}

#else

// Constructors for incompressible flow
AdjointDerivative::AdjointDerivative
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj
)
    :
    mesh_(mesh),
    adjIO_(adjIO),
    adjReg_(adjReg),
    adjRAS_(adjRAS),
    adjIdx_(adjIdx),
    adjCon_(adjCon),
    adjObj_(adjObj),
    db_(mesh.thisDb()),
    pointsRef_(mesh.points()),
    MRF_(mesh_)
    
{

}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<AdjointDerivative> AdjointDerivative::New
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj
)
{
    // get model name, but do not register the dictionary
    // otherwise it is registered in the database twice
    const word modelType
    (
        IOdictionary
        (
            IOobject
            (
                "controlDict",
                mesh.time().system(),
                mesh,
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE,
                false
            )
        ).lookup("application")
    );
    
    Info<< "Selecting " << modelType<<" for AdjointDerivative" << endl;

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn
        (
            "AdjointDerivative::New"
            "("
            "    fvMesh& mesh,"
            "    const AdjointIO& adjIO,"
            "    const AdjointSolverRegistry& adjReg,"
            "    AdjointRASModel& adjRAS,"
            "    AdjointIndexing& adjIdx,"
            "    AdjointJacobianConnectivity& adjCon,"
            "    AdjointObjectiveFunction& adjObj"
            ")"
        )   << "Unknown AdjointDerivative type "
            << modelType << nl << nl
            << "Valid AdjointDerivative types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<AdjointDerivative>
           (
               cstrIter()(mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj)
           );
}

#endif

void AdjointDerivative::solve()
{

    // save the unperturb residual statistics as reference
    // we will verify against this reference to ensure consistent residuals after perturbing
    // and resetting states
    this->calcFlowResidualStatistics("set");

    if (adjIO_.writeMatrices) adjIdx_.writeAdjointIndexing();

    this->assembledRdW4Adjoint();
    
    // create the linear ksp object
    Info<<"Setting up Adjoint KSP"<< endl;
    // add options
    dictionary adjOptions;
    adjOptions.add("GMRESRestart",adjIO_.adjGMRESRestart);
    adjOptions.add("GlobalPCIters",adjIO_.adjGlobalPCIters);
    adjOptions.add("ASMOverlap",adjIO_.adjASMOverlap);
    adjOptions.add("LocalPCIters",adjIO_.adjLocalPCIters);
    adjOptions.add("JacMatReOrdering",adjIO_.adjJacMatReOrdering);
    adjOptions.add("PCFillLevel",adjIO_.adjPCFillLevel);
    adjOptions.add("GMRESMaxIters",adjIO_.adjGMRESMaxIters);
    adjOptions.add("GMRESRelTol",adjIO_.adjGMRESRelTol);
    adjOptions.add("GMRESAbsTol",adjIO_.adjGMRESAbsTol);
    adjOptions.add("printInfo",1);
    if(adjIO_.calcPCMat) this->createMLRKSP(&ksp_,dRdWT_,dRdWTPC_,adjOptions);
    else this->createMLRKSP(&ksp_,dRdWT_,dRdWT_,adjOptions);
    
    // now calculate dFdW for each objFunc and assign them to dFdWAll
    this->initializedFdW();
    forAll(adjIO_.objFuncs,idxI)
    {
    
        word objFunc = adjIO_.objFuncs[idxI];

        if(adjIO_.useColoring) 
        {
            adjCon_.initializedFdWCon(objFunc);
            adjCon_.setupObjFuncCon(objFunc,"dFdW");
            adjCon_.readdFdWColoring(objFunc);
        }

        this->calcdFdW(objFunc);

        if(adjIO_.useColoring) adjCon_.deletedFdWCon();

    }

    MatAssemblyBegin(dFdWAll_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dFdWAll_,MAT_FINAL_ASSEMBLY);

    // before calling solveAdjoint, clear all vars to reduce peak memory usage
    // This will clear URef, URes, UResRef, UResPartDeriv
    this->clearVars4AdjSol();

    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];

        Info<<"Solving Adjoint for "+objFunc<<" "<<this->getRunTime()<<" s"<<endl;

        this->dFdWAll2dFdW(objFunc);
        
        if(adjIO_.adjSegregated)
        {
            this->solveAdjointSegregated(objFunc);
        }
        else
        {
            this->solveAdjoint(objFunc);
        }
    } 
    
    // Destroy stuff to save memory
    KSPDestroy(&ksp_);
    MatDestroy(&dRdWT_);
    if(adjIO_.calcPCMat) MatDestroy(&dRdWTPC_);

    this->calcTotalDeriv();
    
    return;

}

void AdjointDerivative::clearVars4AdjSol()
{
    // Caution: this function should be called right before solveAdjoint()
    Info<<"Deleting class vars to reduce peak memory..."<<endl;
    
    this->clearVars();
    adjRAS_.clearTurbVars();
}


void AdjointDerivative::initializedRdW(Mat* jacMat,const label transposed)
{

    // now initialize the memory for the jacobian itself
    label localSize = adjIdx_.nLocalAdjointStates;
    
    // create dRdWTPC
    if(adjIO_.calcPCMat)
    {
        MatCreate(PETSC_COMM_WORLD,jacMat);
        MatSetSizes(*jacMat,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE);
        MatSetFromOptions(*jacMat);
        if(adjIO_.useColoring && !adjIO_.readMatrices)
        {
            adjCon_.preallocatedRdW(*jacMat,transposed);
            //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        }
        else
        {
            MatMPIAIJSetPreallocation(*jacMat,5000,NULL,5000,NULL);
            MatSeqAIJSetPreallocation(*jacMat,5000,NULL);
            MatSetOption(*jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        }
        MatSetUp(*jacMat);// create dRdWTPC
        Info<<"State Jacobian Created. "<<this->getRunTime()<<" s"<<endl;
    }

}

void AdjointDerivative::initializedFdW()
{

    label localSize = adjIdx_.nLocalAdjointStates;
    VecCreate(PETSC_COMM_WORLD,&dFdW_);
    VecSetSizes(dFdW_,localSize,PETSC_DETERMINE);
    VecSetFromOptions(dFdW_);
    Info<<"dFdW Created!"<<endl;
    
    VecDuplicate(dFdW_,&psi_);

    label nObjFuncs=-1000;
    nObjFuncs = adjIO_.objFuncs.size();
    MatCreate(PETSC_COMM_WORLD,&dFdWAll_);
    MatSetSizes(dFdWAll_,localSize,PETSC_DECIDE,PETSC_DETERMINE,nObjFuncs);
    MatSetFromOptions(dFdWAll_);
    MatMPIAIJSetPreallocation(dFdWAll_,nObjFuncs,NULL,nObjFuncs,NULL);
    MatSeqAIJSetPreallocation(dFdWAll_,nObjFuncs,NULL);
    MatSetOption(dFdWAll_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dFdWAll_);
    Info<<"dFdWAll Created. "<<this->getRunTime()<<" s"<<endl;
    MatZeroEntries(dFdWAll_);

    return;
}


scalar AdjointDerivative::getStateScaling(const word stateName,label faceI)
{
    // read the stateScaling dict from adjointDict and return the scaling values
    // for surfaceScalarState, return the face area
    if (adjIO_.isInList<word>(stateName,adjIO_.normalizeStates))
    {
        word stateScalingName = stateName+"Scaling";
        if(adjIdx_.adjStateType[stateName]!="surfaceScalarState")
        {
            return adjIO_.stateScaling[stateScalingName];
        }
        else
        {
            if (faceI==-1) 
            {
                FatalErrorIn("")<<"faceI needs to be set!"<<abort(FatalError);
                return -9999.0;
            }
            else
            {
                if(faceI<adjIdx_.nLocalInternalFaces)
                {
                    return mesh_.magSf()[faceI] * adjIO_.stateScaling[stateScalingName];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    return mesh_.magSf().boundaryField()[patchIdx][faceIdx] *
                           adjIO_.stateScaling[stateScalingName];
                }
            }
        }
    }
    else
    {
        return 1.0;
    }
    
}


void AdjointDerivative::initializedRdUIn()
{
    
    // now initialize the memory for the jacobian itself
    label localSize = adjIdx_.nLocalAdjointStates;
    
    // create dRdUIn
    MatCreate(PETSC_COMM_WORLD,&dRdUIn_);
    MatSetSizes(dRdUIn_,localSize,PETSC_DECIDE,PETSC_DETERMINE,3);
    MatSetFromOptions(dRdUIn_);
    MatMPIAIJSetPreallocation(dRdUIn_,3,NULL,3,NULL);
    MatSeqAIJSetPreallocation(dRdUIn_,3,NULL);
    MatSetOption(dRdUIn_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dRdUIn_);// create dRdWTPC
    Info<<"dRdUIn Created!"<<endl;
    return;     

}

void AdjointDerivative::initializedFdUIn()
{

    VecCreate(PETSC_COMM_WORLD,&dFdUIn_);
    VecSetSizes(dFdUIn_,PETSC_DETERMINE,3);
    VecSetFromOptions(dFdUIn_);
    Info<<"dFdUIn Created!"<<endl;

    return;     

}

void AdjointDerivative::initializedRdVis()
{
    
    // now initialize the memory for the jacobian itself
    label localSize = adjIdx_.nLocalAdjointStates;
    
    // create dRdVis
    MatCreate(PETSC_COMM_WORLD,&dRdVis_);
    MatSetSizes(dRdVis_,localSize,PETSC_DECIDE,PETSC_DETERMINE,1);
    MatSetFromOptions(dRdVis_);
    MatMPIAIJSetPreallocation(dRdVis_,1,NULL,1,NULL);
    MatSeqAIJSetPreallocation(dRdVis_,1,NULL);
    MatSetOption(dRdVis_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dRdVis_);// create dRdWTPC
    Info<<"dRdVis Created!"<<endl;
    return;     

}

void AdjointDerivative::initializedFdVis()
{

    VecCreate(PETSC_COMM_WORLD,&dFdVis_);
    VecSetSizes(dFdVis_,PETSC_DETERMINE,1);
    VecSetFromOptions(dFdVis_);
    Info<<"dFdVis Created!"<<endl;

    return;     

}



void AdjointDerivative::initializedRdFFD()
{
    label nFFD = adjIO_.nFFDPoints;
    label localSize = adjIdx_.nLocalAdjointStates;
    
    MatCreate(PETSC_COMM_WORLD,&dRdFFD_);
    MatSetSizes(dRdFFD_,localSize,PETSC_DECIDE,PETSC_DETERMINE,nFFD);
    MatSetFromOptions(dRdFFD_);
    MatMPIAIJSetPreallocation(dRdFFD_,nFFD,NULL,nFFD,NULL);
    MatSeqAIJSetPreallocation(dRdFFD_,nFFD,NULL);
    //MatSetOption(dRdFFD_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dRdFFD_);
    
    return;
}

void AdjointDerivative::initializedFdFFD()
{
    label nFFD = adjIO_.nFFDPoints;
    
    VecCreate(PETSC_COMM_WORLD,&dFdFFD_);
    VecSetSizes(dFdFFD_,PETSC_DETERMINE,nFFD);
    VecSetFromOptions(dFdFFD_);

    return;
}

void AdjointDerivative::initializedRdXv()
{
    
    MatCreate(PETSC_COMM_WORLD,&dRdXv_);
    MatSetSizes(dRdXv_,adjIdx_.nLocalAdjointStates,adjIdx_.nLocalXv,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(dRdXv_);
    if(adjIO_.useColoring)
    {
        adjCon_.preallocatedRdXv(dRdXv_);
    }
    else
    {
        MatMPIAIJSetPreallocation(dRdXv_,5000,NULL,5000,NULL);
        MatSeqAIJSetPreallocation(dRdXv_,5000,NULL);
        MatSetOption(dRdXv_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    }
    MatSetUp(dRdXv_);
    Info<<"dRdXv Created. "<<this->getRunTime()<<" s"<<endl;
    
    return;
}

void AdjointDerivative::initializedFdXv()
{

    label nLocalPoints = adjIdx_.nLocalPoints;
    VecCreate(PETSC_COMM_WORLD,&dFdXv_);
    VecSetSizes(dFdXv_,nLocalPoints*3,PETSC_DETERMINE);
    VecSetFromOptions(dFdXv_);

    VecCreate(PETSC_COMM_WORLD,&dFdXvSerial_);
    VecSetSizes(dFdXvSerial_,PETSC_DETERMINE,adjIdx_.nUndecomposedPoints*3);
    VecSetFromOptions(dFdXvSerial_);

    return;
}

void AdjointDerivative::perturbFFD
(
    Mat deltaVolPointMat,
    PetscInt iDV,
    pointField& newPoints
)
{
    PetscInt Istart,Iend;
    MatGetOwnershipRange(deltaVolPointMat,&Istart,&Iend);
    // set the perturbed coords based on deltaVolPointMat
    for(PetscInt j=Istart;j<Iend;j++)
    {
        PetscScalar deltaVal;
        MatGetValues(deltaVolPointMat,1,&j,1,&iDV,&deltaVal);
        PetscInt idxRel= (j-Istart)/3;
        PetscInt coordI = (j-Istart)%3;
        if (coordI ==0)
            newPoints[idxRel].x() = pointsRef_[idxRel].x() + deltaVal;
        else if (coordI ==1)
            newPoints[idxRel].y() = pointsRef_[idxRel].y() + deltaVal;
        else if (coordI ==2)
            newPoints[idxRel].z() = pointsRef_[idxRel].z() + deltaVal;
        else
            FatalErrorIn("coordI not valid")<< abort(FatalError);
    }

    // Sync point coordinates at inter-processor boundaries
    //this->syncBoundaryPointCoords(newPoints); // no need to do this?

    // now update the metrics
    tmp<scalarField> tsweptVols = mesh_.movePoints(newPoints);

    // *************** NOTE ************* TODO
    // this gives much better accuracy for buoyancy 
    // but it messes up the residual checks!
    // some intermediate variables may depend on geometric metrics, e.g., ghf
    this->updateStateVariableBCs();
    this->updateIntermediateVariables();
    
    // correct wall distance
    if( adjIO_.correctWallDist ) adjRAS_.correctWallDist();

    return;

}

void AdjointDerivative::perturbXv
(
    label colorI,
    const word mode,
    pointField& newPoints
)
{
    for(label i=0;i<adjIdx_.nLocalPoints;i++)
    {
        for(label j=0;j<3;j++)
        {
            label color = adjCon_.getXvColor(mode,i,j);   
            if (color==colorI)
            {
                newPoints[i][j] = pointsRef_[i][j] + adjIO_.epsDerivXv;
            }
        }
    }

    // Sync point coordinates at inter-processor boundaries
    //this->syncBoundaryPointCoords(newPoints); // no need to do this?

    // now update the metrics
    tmp<scalarField> tsweptVols = mesh_.movePoints(newPoints);
    
    // correct wall distance
    if( adjIO_.correctWallDist ) adjRAS_.correctWallDist();

    return;
}

void AdjointDerivative::syncBoundaryPointCoords(pointField &newPoints)
{
    // NOTE: this function is depricated.
    // We have dupicated points at the inter-proc boundaries. However, we treat these
    // points as if they are completely independent. That being said, we don't need to
    // sync these points when perturbing Xv (even their physical locations are not consistent) 
    // When we pass the volume sensitivy to mesh warping, the sensitivity at these duplicated
    // points will be sum up so we can get right values
/*
    if (Pstream::parRun())
    {
        // Get the list of master points
        PackedBoolList isMasterPoint(syncTools::getMasterPoints(mesh_));

        // Loop over all points and zero all of the slave points
        forAll(newPoints,pointI)
        {
            if(not isMasterPoint[pointI])
            {
                forAll( newPoints[pointI],iDim)
                {
                    newPoints[pointI][iDim]=0;
                }
            }
        }

        // Sync the points. Since the slaves are all 0, the max magnitude will be the master
        syncTools::syncPointPositions
        (
            mesh_,
            newPoints,
            //maxEqOp<point>(),//
            maxMagSqrEqOp<point>(),
            point(GREAT, GREAT, GREAT)
        );

    }
*/
    return;
}


void AdjointDerivative::perturbStates
(
    const label colorI,
    const word mode
)
{
   
    // perturb volVectorStates
    forAll(adjReg_.volVectorStates,idxI)                                           
    {        
        // create state and stateRef
        makeState(volVectorStates,volVectorField,adjReg_); 
        makeStateRef(volVectorStates,volVectorField,adjReg_);                 
                                                                               
        forAll(mesh_.cells(),cellI)                                            
        {                                                                      
            // check if this state's color = coloI
            for(label i=0;i<3;i++)                                         
            {                                                                  
                label color = adjCon_.getStateColor(mode,stateName,cellI,i);        
                if (color == colorI)                                           
                {
                    scalar normScaling = this->getStateScaling(stateName);                                                              
                    state[cellI][i] = stateRef[cellI][i] + adjIO_.epsDeriv*normScaling;   
                }                                                              
            }                                                                  
        } 
        // correct BC
        state.correctBoundaryConditions();                                                                    
    }

    // perturb volScalarStates
    forAll(adjReg_.volScalarStates,idxI)
    {
        // create state and stateRef
        makeState(volScalarStates,volScalarField,adjReg_);
        makeStateRef(volScalarStates,volScalarField,adjReg_);
        
        forAll(mesh_.cells(),cellI)
        {         
            // check if this state's color = coloI
            label color = adjCon_.getStateColor(mode,stateName,cellI);
            if (color == colorI)
            {
                scalar normScaling = this->getStateScaling(stateName); 
                state[cellI] = stateRef[cellI] + adjIO_.epsDeriv*normScaling;
            }
        }
        
        // correct BC
        state.correctBoundaryConditions();  
    }

    // perturb turbStates
    forAll(adjRAS_.turbStates,idxI)
    {
        // create state and stateRef
        makeState(turbStates,volScalarField,adjRAS_);
        makeStateRef(turbStates,volScalarField,adjRAS_);
        
        forAll(mesh_.cells(),cellI)
        {         
            // check if this state's color = coloI
            label color = adjCon_.getStateColor(mode,stateName,cellI);
            if (color == colorI)
            {
                scalar normScaling = this->getStateScaling(stateName); 
                state[cellI] = stateRef[cellI] + adjIO_.epsDeriv*normScaling;
            }
        }

    }
    // BC for turbStates are implemented in the AdjRAS class
    adjRAS_.correctTurbBoundaryConditions();

    // perturb surfaceScalarStates
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        // create state and stateRef
        makeState(surfaceScalarStates,surfaceScalarField,adjReg_);
        makeStateRef(surfaceScalarStates,surfaceScalarField,adjReg_);
        
        forAll(mesh_.faces(),faceI)
        {
            // check if this state's color = coloI
            label color = adjCon_.getStateColor(mode,stateName,faceI);
            if (color == colorI)
            {
                scalar normScaling = this->getStateScaling(stateName,faceI); 
                if (faceI < adjIdx_.nLocalInternalFaces)
                {
                    state[faceI] = stateRef[faceI] + adjIO_.epsDeriv*normScaling;
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    state.boundaryFieldRef()[patchIdx][faceIdx] = 
                        stateRef.boundaryField()[patchIdx][faceIdx] + adjIO_.epsDeriv*normScaling;        
                }  
            }          
            
        }
        
    }

    // NOTE: we also need to update states that are related to the adjoint states but not
    // perturbed here. For example, in buoyantBoussinesqFoam, p is related to p_rgh;
    // however, we only perturb p_rgh in this function. To calculate the perturbed
    // p due to the p_rgh perturbation for calculating force, we need to do p=p_rgh+rhok*gh
    // Similar treatment is needed for rhok and alphat. Basically, any variables apprear in flow residual
    // calculation or objection function calculation that are not state variables need to be updated.
    // This function is implemented in child class
    this->updateIntermediateVariables();
    
    return;
}

void AdjointDerivative::perturbUIn(scalar eps)
{
    word stateName = word(adjIO_.derivUInInfo["stateName"]);
    label comp = readLabel( adjIO_.derivUInInfo["component"] );
    List<word> patchNames = adjIO_.derivUInInfo["patchNames"];
    word type = word(adjIO_.derivUInInfo["type"]);

    if(type=="fixedValue")
    {
        const word& stateType = adjIdx_.adjStateType[stateName];
        forAll(patchNames,idxI)
        {
            word patchName = patchNames[idxI];
    
            if(stateType=="volVectorState")
            {
                volVectorField& state
                (
                    const_cast<volVectorField&>
                    (
                        db_.lookupObject<volVectorField>(stateName) 
                    ) 
                ); 
        
                label patchI = mesh_.boundaryMesh().findPatchID( patchName );
                if(mesh_.boundaryMesh()[patchI].size()>0)
                {
                    forAll(state.boundaryField()[patchI],faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI][comp] += eps;
                    }
                } 
            }
            else if (stateType=="volScalarState" || stateType=="turbState")
            {
                volScalarField& state
                (
                    const_cast<volScalarField&>
                    (
                        db_.lookupObject<volScalarField>(stateName) 
                    ) 
                ); 
        
                label patchI = mesh_.boundaryMesh().findPatchID( patchName );
                if(mesh_.boundaryMesh()[patchI].size()>0)
                {
                    forAll(state.boundaryField()[patchI],faceI)
                    {
                        state.boundaryFieldRef()[patchI][faceI] += eps;
                    }
                } 
            }
            else
            {
                FatalErrorIn("perturbInlet")<< abort(FatalError);
            }
        }
        
        // we may need to update nut when perturbing U since nut may depend on U (e.g., for the SST model) 
        adjRAS_.updateNut();
        this->updateIntermediateVariables();

    }
    else if (type=="fixedGradient")
    {
        const word& stateType = adjIdx_.adjStateType[stateName];
        forAll(patchNames,idxI)
        {
            word patchName = patchNames[idxI];
    
            if(stateType=="volVectorState")
            {
                volVectorField& state
                (
                    const_cast<volVectorField&>
                    (
                        db_.lookupObject<volVectorField>(stateName) 
                    ) 
                ); 
        
                label patchI = mesh_.boundaryMesh().findPatchID( patchName );

                fixedGradientFvPatchVectorField& patchBC =
                    refCast< fixedGradientFvPatchVectorField > (state.boundaryFieldRef()[patchI]);
                vectorField& grad = const_cast<vectorField&>(patchBC.gradient());

                forAll(grad,idxI)
                {
                    grad[idxI][comp] += eps;
                }

                // actually update the boundary values
                state.correctBoundaryConditions();

            }
            else if (stateType=="volScalarState" || stateType=="turbState")
            {
                volScalarField& state
                (
                    const_cast<volScalarField&>
                    (
                        db_.lookupObject<volScalarField>(stateName) 
                    ) 
                ); 
        
                label patchI = mesh_.boundaryMesh().findPatchID( patchName );

                fixedGradientFvPatchScalarField& patchBC =
                    refCast< fixedGradientFvPatchScalarField > (state.boundaryFieldRef()[patchI]);
                scalarField& grad = const_cast<scalarField&>(patchBC.gradient());

                forAll(grad,idxI)
                {
                    grad[idxI] += eps;
                }

                // actually update the boundary values
                state.correctBoundaryConditions();
                
            }
            else
            {
                FatalErrorIn("perturbInlet")<< abort(FatalError);
            }
        }
        
        // we may need to update nut when perturbing U since nut may depend on U (e.g., for the SST model) 
        adjRAS_.updateNut();
        this->updateIntermediateVariables();
        
    }
    else if (type=="tractionDisplacement")
    {
        volVectorField& state
        (
            const_cast<volVectorField&>
            (
                db_.lookupObject<volVectorField>(stateName) 
            ) 
        ); 

        forAll(patchNames,idxI)
        {
            word patchName = patchNames[idxI];
    
            label patchI = mesh_.boundaryMesh().findPatchID( patchName );

            tractionDisplacementFvPatchVectorField& patchBC = 
                refCast< tractionDisplacementFvPatchVectorField > (state.boundaryFieldRef()[patchI]);

            vectorField& traction = const_cast<vectorField&>(patchBC.traction());
            scalarField& pressure = const_cast<scalarField&>(patchBC.pressure());

            if(comp==-1) // perturb pressure
            {
                forAll(pressure,idxI)
                {
                    pressure[idxI] += eps;
                }

            }
            else // perturb traction
            {
                forAll(traction,idxI)
                {
                    traction[idxI][comp] += eps;
                }
            }

        }
        // actually update the boundary values
        state.correctBoundaryConditions();
        this->updateIntermediateVariables();
    }
    else if (type=="rotRad")
    {
        vector& rotRad = const_cast<vector&>(adjIO_.rotRad); 
        rotRad[comp] += eps;
        this->updateIntermediateVariables();
    }
    else
    {
        FatalErrorIn("")<<"type not support"<< abort(FatalError);
    }
    
    return;
}

void AdjointDerivative::calcResPartDeriv(const scalar eps, const label isPC)
{  

    label isRef=0;
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);
    
    forAll(adjReg_.volVectorStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeResAll(volVectorStates,volVectorField,adjReg_);
        // calculate partial derivative
        stateResPartDeriv = ( stateRes - stateResRef ) / eps;
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeResAll(volScalarStates,volScalarField,adjReg_);
        // calculate partial derivative
        stateResPartDeriv = ( stateRes - stateResRef ) / eps;
    }

    forAll(adjRAS_.turbStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeResAll(turbStates,volScalarField,adjRAS_);
        // calculate partial derivative
        stateResPartDeriv = ( stateRes - stateResRef ) / eps;
    }

    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeResAll(surfaceScalarStates,surfaceScalarField,adjReg_);
        // calculate partial derivative
        stateResPartDeriv = ( stateRes - stateResRef ) / eps;
    }

    return;
}

void AdjointDerivative::setNormalizeStatesScaling2Vec(Vec vecY)
{

    PetscScalar *vecYArray;
    VecGetArray(vecY,&vecYArray);
    
    forAll(adjReg_.volVectorStates,idxI)                                           
    {        
        const word stateName = adjReg_.volVectorStates[idxI];    
        scalar scalingFactor = getStateScaling(stateName); 
        if(scalingFactor==1.0) continue;

        forAll(mesh_.cells(),cellI)                                            
        {                                                                      
            for(label i=0;i<3;i++)                                         
            {                                                                  
                label localIdx = adjIdx_.getLocalAdjointStateIndex(stateName,cellI,i);
                PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor; 
                vecYArray[localIdx] = scaledVal;        
            }                                                                  
        }                                                                   
    }
    
    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI]; 
        scalar scalingFactor = getStateScaling(stateName);
        if(scalingFactor==1.0) continue;

        forAll(mesh_.cells(),cellI)
        {         
            label localIdx = adjIdx_.getLocalAdjointStateIndex(stateName,cellI);    
            PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor; 
            vecYArray[localIdx] = scaledVal;          
        }
         
    }
    
    forAll(adjRAS_.turbStates,idxI)
    {
        const word stateName = adjRAS_.turbStates[idxI]; 
        scalar scalingFactor = getStateScaling(stateName);
        if(scalingFactor==1.0) continue;

        forAll(mesh_.cells(),cellI)
        {         
            label localIdx = adjIdx_.getLocalAdjointStateIndex(stateName,cellI);   
            PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor;
            vecYArray[localIdx] = scaledVal;        
        }
    
    }
    
    
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        const word stateName = adjReg_.surfaceScalarStates[idxI]; 
    
        forAll(mesh_.faces(),faceI)
        {
            label localIdx = adjIdx_.getLocalAdjointStateIndex(stateName,faceI); 
            scalar scalingFactor = getStateScaling(stateName,faceI);
            if(scalingFactor!=1.0)
            {
                PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor;
                vecYArray[localIdx] = scaledVal;
            }     
        }
        
    }
    
    VecRestoreArray(vecY,&vecYArray);

    return;

}

void AdjointDerivative::updateStateVariableBCs()
{
    forAll(adjReg_.volVectorStates,idxI)                                           
    {        
        makeState(volVectorStates,volVectorField,adjReg_);
        state.correctBoundaryConditions();
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        makeState(volScalarStates,volScalarField,adjReg_);
        state.correctBoundaryConditions();
    }

    adjRAS_.correctTurbBoundaryConditions();

    this->updateIntermediateVariables();

    return;
}


void AdjointDerivative::copyStates(const word option)
{
    // "Ref2Var", assign ref states to states
    // "Var2Ref", assign states to ref states
    if(option=="Ref2Var")
    {
    
        forAll(adjReg_.volVectorStates,idxI)
        {
            makeState(volVectorStates,volVectorField,adjReg_);
            makeStateRef(volVectorStates,volVectorField,adjReg_);
            state = stateRef;
            state.correctBoundaryConditions();
        }
        
        forAll(adjReg_.volScalarStates,idxI)
        {
            makeState(volScalarStates,volScalarField,adjReg_);
            makeStateRef(volScalarStates,volScalarField,adjReg_);
            state = stateRef;
            state.correctBoundaryConditions();
        }
        
        adjRAS_.copyTurbStates(option);
        
        forAll(adjReg_.surfaceScalarStates,idxI)
        {
            makeState(surfaceScalarStates,surfaceScalarField,adjReg_);
            makeStateRef(surfaceScalarStates,surfaceScalarField,adjReg_);
            state = stateRef;
            // manually do the boundary field since we do not correct boundary conditions for phi
            forAll(state.boundaryField(),patchI)
            {
                forAll(state.boundaryField()[patchI],faceI)
                {
                    state.boundaryFieldRef()[patchI][faceI]=stateRef.boundaryField()[patchI][faceI];
                }
            }
        }

        // NOTE: we also need to update intermediate variables that are related to the adjoint states
        // This is to make sure all intermediate states are properly set
        this->updateIntermediateVariables();

    }
    else if (option=="Var2Ref")
    {
    
        forAll(adjReg_.volVectorStates,idxI)
        {
            makeState(volVectorStates,volVectorField,adjReg_);
            makeStateRef(volVectorStates,volVectorField,adjReg_);
            stateRef = state;
        }
        
        forAll(adjReg_.volScalarStates,idxI)
        {
            makeState(volScalarStates,volScalarField,adjReg_);
            makeStateRef(volScalarStates,volScalarField,adjReg_);
            stateRef = state;
        }
        
        adjRAS_.copyTurbStates(option);
        
        forAll(adjReg_.surfaceScalarStates,idxI)
        {
            makeState(surfaceScalarStates,surfaceScalarField,adjReg_);
            makeStateRef(surfaceScalarStates,surfaceScalarField,adjReg_);
            stateRef = state;
            // manually do the boundary field
            forAll(state.boundaryField(),patchI)
            {
                forAll(state.boundaryField()[patchI],faceI)
                {
                    stateRef.boundaryFieldRef()[patchI][faceI]=state.boundaryField()[patchI][faceI];
                }
            }
        }
        
    }
    else
    {
        FatalErrorIn("option not valid! Should be either Var2Ref or Ref2Var")
        << abort(FatalError);
    }
    
    return;
}


scalar AdjointDerivative::adjStateLocalIdx2PartDerivVal(const label localIdx)
{

    word stateName = adjIdx_.adjStateName4LocalAdjIdx[localIdx];
    word stateResPartDerivName = stateName+"ResPartDeriv";
    const word& stateType = adjIdx_.adjStateType[stateName];
    
    scalar cellIFaceI = adjIdx_.cellIFaceI4LocalAdjIdx[localIdx];
    
    if(stateType == "volVectorState")
    {
        const volVectorField& stateResPartDeriv = db_.lookupObject<volVectorField>(stateResPartDerivName);
        label cellI,comp;
        cellI = round(cellIFaceI);
        comp = round(10*(cellIFaceI-cellI));

        return stateResPartDeriv[cellI][comp];
    }
    else if(stateType == "volScalarState" or stateType == "turbState")
    {
        const volScalarField& stateResPartDeriv = db_.lookupObject<volScalarField>(stateResPartDerivName);
        label cellI;
        cellI = round(cellIFaceI);
        return stateResPartDeriv[cellI];
    }
    else if(stateType == "surfaceScalarState")
    {
        const surfaceScalarField& stateResPartDeriv = db_.lookupObject<surfaceScalarField>(stateResPartDerivName);
        label faceI;
        faceI = round(cellIFaceI);

        if(faceI<adjIdx_.nLocalInternalFaces)
        {
            return stateResPartDeriv[faceI];
        }
        else
        {
            label relIdx=faceI-adjIdx_.nLocalInternalFaces;
            label patchIdx=adjIdx_.bFacePatchI[relIdx];
            label faceIdx=adjIdx_.bFaceFaceI[relIdx];
            return stateResPartDeriv.boundaryField()[patchIdx][faceIdx];  
        }
    }
    else
    {
        FatalErrorIn("")<<"stateType not known"<< abort(FatalError);
    }
    
    FatalErrorIn("")<<"localIdx not valid"<< abort(FatalError);
    return -9999.0;
}

void AdjointDerivative::setJacobianMatColored
(
    Mat matIn,
    Vec coloredColumns,
    const label transposed,
    const label isPC
)
{
    // read the min tolerance for dRdWTPC , set it to a relatively
    // large value (compared with minTolPC) can reduce the size of the PC matrix.
    scalar minTol,maxTol;
    if (isPC) 
    {
        minTol = adjIO_.minTolPC;
        maxTol = adjIO_.maxTolPC;
    }
    else 
    {
        minTol = adjIO_.minTolJac;
        maxTol = adjIO_.maxTolJac;
    }
    
    PetscInt    idx, Istart, Iend;
    // get the local ownership range
    VecGetOwnershipRange(coloredColumns,&Istart,&Iend);

    PetscScalar *coloredCol;
    VecGetArray(coloredColumns,&coloredCol);
    
    // Loop over the owned values of this row and set the corresponding
    // Jacobian entries
    for(PetscInt j=Istart; j<Iend; j++)
    {
        // The value in the vector is the global index that the
        // corresponding state derivative belongs to
        label idxColorLocal = j-Istart;
        idx = round(coloredCol[idxColorLocal]);
        // This color affects this state, so set the corresponding Jacobian entry
        if(idx>=0)
        {
            // Translate the row into a local index then find the correct
            // residual for that row
            label localIdx = adjIdx_.globalAdjointStateNumbering.toLocal(j);
            
            PetscScalar resVal;
            resVal = this->adjStateLocalIdx2PartDerivVal(localIdx);

            if ( (fabs(resVal)>minTol && fabs(resVal) < maxTol) || idx==j )
            {
                if(transposed)
                {
                    MatSetValues(matIn,1,&idx,1,&j,&resVal,INSERT_VALUES);
                }
                else
                {
                    MatSetValues(matIn,1,&j,1,&idx,&resVal,INSERT_VALUES);
                }
            }
        }
    }
    
    VecRestoreArray(coloredColumns,&coloredCol);        

    return;
}


void AdjointDerivative::setJacobianMat
(
    Mat matIn,
    const label colorI,
    const label transposed
)
{

    forAll(adjReg_.volVectorStates,idxI)
    {
        
        const word stateName = adjReg_.volVectorStates[idxI];
        makeResPartDeriv(volVectorStates,volVectorField,adjReg_);
        
        forAll(mesh_.cells(), cellI)
        {
            for(label i=0;i<3;i++)
            {
                PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,i);
                PetscInt colI = colorI;
                PetscScalar val;
                val = stateResPartDeriv[cellI][i];

                if ( fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
                {
                    if(transposed)
                    {
                        MatSetValues(matIn,1,&colI,1,&rowI,&val,INSERT_VALUES);
                    }
                    else
                    {
                        MatSetValues(matIn,1,&rowI,1,&colI,&val,INSERT_VALUES);
                    }
                }
            }
        }
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI];
        makeResPartDeriv(volScalarStates,volScalarField,adjReg_);
        
        forAll(mesh_.cells(), cellI)
        {
            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI);
            PetscInt colI = colorI;
            PetscScalar val;
            val = stateResPartDeriv[cellI];

            if ( fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
            {
                if(transposed)
                {
                    MatSetValues(matIn,1,&colI,1,&rowI,&val,INSERT_VALUES);
                }
                else
                {
                    MatSetValues(matIn,1,&rowI,1,&colI,&val,INSERT_VALUES);
                }
            }
        }
    }

    forAll(adjRAS_.turbStates,idxI)
    {
        const word stateName = adjRAS_.turbStates[idxI];
        makeResPartDeriv(turbStates,volScalarField,adjRAS_);
        
        forAll(mesh_.cells(), cellI)
        {
            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI);
            PetscInt colI = colorI;
            PetscScalar val;
            val = stateResPartDeriv[cellI];

            if ( fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
            {
                if(transposed)
                {
                    MatSetValues(matIn,1,&colI,1,&rowI,&val,INSERT_VALUES);
                }
                else
                {
                    MatSetValues(matIn,1,&rowI,1,&colI,&val,INSERT_VALUES);
                }
            }
        }
    }
    
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        const word stateName = adjReg_.surfaceScalarStates[idxI];
        makeResPartDeriv(surfaceScalarStates,surfaceScalarField,adjReg_);
        
        forAll(mesh_.faces(), faceI)
        {
            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,faceI);
            PetscInt colI = colorI;
            PetscScalar val;
            if (faceI < adjIdx_.nLocalInternalFaces)
            {
                val = stateResPartDeriv[faceI];
            }
            else
            {
                label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                label patchIdx=adjIdx_.bFacePatchI[relIdx];
                label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                val = stateResPartDeriv.boundaryField()[patchIdx][faceIdx];
            } 
            if ( fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
            {
                if(transposed)
                {
                    MatSetValues(matIn,1,&colI,1,&rowI,&val,INSERT_VALUES);
                }
                else
                {
                    MatSetValues(matIn,1,&rowI,1,&colI,&val,INSERT_VALUES);
                }
            }
  
        }
    }

    return;
}

void AdjointDerivative::setdFVec
(
    Vec dFVec,
    const word objFunc,
    const label colorI
)
{
    
    PetscScalar val;
    val = adjObj_.getObjFuncPartDeriv(objFunc);
    PetscInt rowI = colorI;
    if (fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
    {
        VecSetValue(dFVec,rowI,val,INSERT_VALUES);
    }

    return;
}

void AdjointDerivative::setdFVecColored
(
    Vec dFVec,
    Vec coloredColumns,
    const word objFunc
)
{

    label Istart, Iend;
    PetscScalar valIn=0.0;
    // get the local ownership range
    VecGetOwnershipRange(coloredColumns,&Istart,&Iend);

    PetscScalar *coloredCol;
    VecGetArray(coloredColumns,&coloredCol);
       
    List<word> objFuncGeoInfo = adjIdx_.getObjFuncGeoInfo(objFunc);
    
    label objElementI = 0;
    forAll(objFuncGeoInfo,idxI)
    {
        // for internal patch
        if( adjIdx_.isUserDefinedPatch(objFuncGeoInfo[idxI]) )
        {
            labelList userDefinedPatchFaces = adjIdx_.faceIdx4UserDefinedPatches[objFuncGeoInfo[idxI]];
            forAll(userDefinedPatchFaces,faceI)
            {
                PetscInt idxJ = round( coloredCol[objElementI] );
                if(idxJ>=0)
                {
                    valIn = adjObj_.getObjFuncDiscretePartDeriv(objFunc,objElementI);
                    if( fabs(valIn)>adjIO_.minTolJac && fabs(valIn) < adjIO_.maxTolJac)
                    {
                        VecSetValue(dFVec,idxJ,valIn,ADD_VALUES);
                    }
                }
                objElementI+=1;
            }
        }
        else if (adjIdx_.isUserDefinedVolume(objFuncGeoInfo[idxI]))
        {
            // get info
            word geoName = objFuncGeoInfo[idxI];

            forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],cellI)
            {
                PetscInt idxJ = round( coloredCol[objElementI] );
                if(idxJ>=0)
                {
                    valIn = adjObj_.getObjFuncDiscretePartDeriv(objFunc,objElementI);
                    if( fabs(valIn)>adjIO_.minTolJac && fabs(valIn) < adjIO_.maxTolJac)
                    {
                        VecSetValue(dFVec,idxJ,valIn,ADD_VALUES);
                    }
                }
                objElementI+=1;
            }
        }
        else if (objFunc=="VMS")
        {
            forAll(mesh_.cells(),cellI)
            {
                PetscInt idxJ = round( coloredCol[cellI] );
                if(idxJ>=0)
                {
                    valIn = adjObj_.getObjFuncDiscretePartDeriv(objFunc,cellI);
                    if( fabs(valIn)>adjIO_.minTolJac && fabs(valIn) < adjIO_.maxTolJac)
                    {
                        VecSetValue(dFVec,idxJ,valIn,ADD_VALUES);
                    }
                }
            }   
        }
        else
        {
            // get the patch id label
            label patchI = mesh_.boundaryMesh().findPatchID( objFuncGeoInfo[idxI] );
            // create a shorter handle for the boundary patch
            const fvPatch& patch = mesh_.boundary()[patchI];
            forAll(patch,faceI)
            {
                PetscInt idxJ = round( coloredCol[objElementI] );
                if(idxJ>=0)
                {
                    valIn = adjObj_.getObjFuncDiscretePartDeriv(objFunc,objElementI);
                    if( fabs(valIn)>adjIO_.minTolJac && fabs(valIn) < adjIO_.maxTolJac)
                    {
                        VecSetValue(dFVec,idxJ,valIn,ADD_VALUES);
                    }
                }
                objElementI+=1;
            }
        }
    }
    
    VecRestoreArray(coloredColumns,&coloredCol);
    
    return;
}

void AdjointDerivative::assembledRdW4Adjoint()
{
    Info<<"Initializing State Jacobians "<<this->getRunTime()<<" s"<<endl;

    if(adjIO_.useColoring and !adjIO_.readMatrices) 
    {
        // compute preallocation vecs for dRdW 
        adjCon_.setupdRdWCon(1);
        adjCon_.initializedRdWCon();
        adjCon_.setupdRdWCon(0);
        // Read in the precomputed coloring
        adjCon_.readdRdWColoring();
    }
    
    label transposed=1;
    this->initializedRdW(&dRdWT_,transposed);
    if(adjIO_.calcPCMat) this->initializedRdW(&dRdWTPC_,transposed);

    Info<<"Calculating State Jacobians "<<this->getRunTime()<<" s"<<endl;
    label isPC;

    // calc dRdWT
    isPC=0;
    if(adjIO_.useColoring)
    {
        // if reducedConJac is ON, we initialize dRdWConPC instead
        // and we will use this reduced connectivity to assign dRdWT
        // in this case, we need to first delete dRdWCon and initialize
        // dRdWConPC
        if(adjIO_.reduceResCon4JacMat) 
        {
            adjCon_.deletedRdWCon();
            adjCon_.initializedRdWConPC();
            adjCon_.setupdRdWCon(0,1);
        }
    }
    this->calcdRdW(dRdWT_,transposed,isPC);

    // calc dRdWTPC
    isPC=1;
    if(adjIO_.calcPCMat)
    {
        if(adjIO_.useColoring && !adjIO_.fastPCMat)
        {
            // if reducedConJac is OFF, we have not deleted dRdWCon when
            // calculating dRdWT, so we need to delete here and initialize
            // dRdWConPC for dRdWTPC computation
            if(!adjIO_.reduceResCon4JacMat) 
            {
                adjCon_.deletedRdWCon();
                adjCon_.initializedRdWConPC();
                adjCon_.setupdRdWCon(0,1);
            }
        }
        if(adjIO_.fastPCMat) this->calcdRdWPCFast(dRdWTPC_,transposed);
        else this->calcdRdW(dRdWTPC_,transposed,isPC);
    }

    
    if(adjIO_.useColoring) 
    {
        adjCon_.deletedRdWCon();
        adjCon_.deletedRdWConPC();
    }

    return;
}

void AdjointDerivative::calcdRdWPCFast
(
    Mat jacMat,
    const label transposed,
    const label printStatistics
)
{

    label isRef=0,isPC=1;
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();

    forAll(adjReg_.volVectorStates,idxI)
    {
        
        const word stateName = adjReg_.volVectorStates[idxI];
        const word resName = stateName+"Res";
        const word fvMatrixName = stateName+"Eqn";
        scalar resScaling=1.0;
        scalar stateScaling= this->getStateScaling(stateName);
        
        this->calcResiduals(isRef,isPC,fvMatrixName);

        // set diag
        for(label cellI=0;cellI<adjIdx_.nLocalCells;cellI++)
        {
            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[cellI];
            }    

            for(label i=0;i<3;i++)
            {
                PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,i);
                PetscInt colI = rowI;
                PetscScalar val=fvMatrixDiag[cellI]/resScaling*stateScaling;
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }

        // set lower/owner
        for(label faceI=0;faceI<adjIdx_.nLocalInternalFaces;faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[neighbourCellI];
            } 

            for(label i=0;i<3;i++)
            {
                PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,neighbourCellI,i);
                PetscInt colI = adjIdx_.getGlobalAdjointStateIndex(stateName,ownerCellI,i);
                PetscScalar val=fvMatrixLower[faceI]/resScaling*stateScaling;
                if(transposed)
                {
                    MatSetValues(jacMat,1,&colI,1,&rowI,&val,INSERT_VALUES);
                }
                else
                {
                    MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
                }
            }
        }

        // set upper/neighbour
        for(label faceI=0;faceI<adjIdx_.nLocalInternalFaces;faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[ownerCellI];
            } 

            for(label i=0;i<3;i++)
            {
                PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,ownerCellI,i);
                PetscInt colI = adjIdx_.getGlobalAdjointStateIndex(stateName,neighbourCellI,i);
                PetscScalar val=fvMatrixUpper[faceI]/resScaling*stateScaling;
                if(transposed)
                {
                    MatSetValues(jacMat,1,&colI,1,&rowI,&val,INSERT_VALUES);
                }
                else
                {
                    MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
                }
            }
        }
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI];
        const word resName = stateName+"Res";
        const word fvMatrixName = stateName+"Eqn";
        scalar resScaling=1.0;
        scalar stateScaling= this->getStateScaling(stateName);
        
        this->calcResiduals(isRef,isPC,fvMatrixName);

        // set diag
        for(label cellI=0;cellI<adjIdx_.nLocalCells;cellI++)
        {
            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[cellI];
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI);
            PetscInt colI = rowI;
            PetscScalar val=fvMatrixDiag[cellI]/resScaling*stateScaling;
            MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
        }

        // set lower/owner
        for(label faceI=0;faceI<adjIdx_.nLocalInternalFaces;faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[neighbourCellI];
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,neighbourCellI);
            PetscInt colI = adjIdx_.getGlobalAdjointStateIndex(stateName,ownerCellI);
            PetscScalar val=fvMatrixLower[faceI]/resScaling*stateScaling;
            if(transposed)
            {
                MatSetValues(jacMat,1,&colI,1,&rowI,&val,INSERT_VALUES);
            }
            else
            {
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }

        // set upper/neighbour
        for(label faceI=0;faceI<adjIdx_.nLocalInternalFaces;faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[ownerCellI];
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,ownerCellI);
            PetscInt colI = adjIdx_.getGlobalAdjointStateIndex(stateName,neighbourCellI);
            PetscScalar val=fvMatrixUpper[faceI]/resScaling*stateScaling;
            if(transposed)
            {
                MatSetValues(jacMat,1,&colI,1,&rowI,&val,INSERT_VALUES);
            }
            else
            {
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }
    }

    forAll(adjRAS_.turbStates,idxI)
    {
        const word stateName = adjRAS_.turbStates[idxI];
        const word resName = stateName+"Res";
        const word fvMatrixName = stateName+"Eqn";
        scalar resScaling=1.0;
        scalar stateScaling= this->getStateScaling(stateName);
        
        adjRAS_.calcTurbResiduals(isRef,isPC,fvMatrixName);

        // set diag
        for(label cellI=0;cellI<adjIdx_.nLocalCells;cellI++)
        {

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[cellI];
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI);
            PetscInt colI = rowI;
            PetscScalar val=adjRAS_.fvMatrixDiag[cellI]/resScaling*stateScaling;
            MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
        }

        // set lower/owner
        for(label faceI=0;faceI<adjIdx_.nLocalInternalFaces;faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[neighbourCellI];
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,neighbourCellI);
            PetscInt colI = adjIdx_.getGlobalAdjointStateIndex(stateName,ownerCellI);
            PetscScalar val=adjRAS_.fvMatrixLower[faceI]/resScaling*stateScaling;
            if(transposed)
            {
                MatSetValues(jacMat,1,&colI,1,&rowI,&val,INSERT_VALUES);
            }
            else
            {
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }

        // set upper/neighbour
        for(label faceI=0;faceI<adjIdx_.nLocalInternalFaces;faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[ownerCellI];
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,ownerCellI);
            PetscInt colI = adjIdx_.getGlobalAdjointStateIndex(stateName,neighbourCellI);
            PetscScalar val=adjRAS_.fvMatrixUpper[faceI]/resScaling*stateScaling;
            if(transposed)
            {
                MatSetValues(jacMat,1,&colI,1,&rowI,&val,INSERT_VALUES);
            }
            else
            {
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }
    }
    
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        const word stateName = adjReg_.surfaceScalarStates[idxI];
        const word resName = stateName+"Res";
        scalar resScaling=1.0,stateScaling=1.0;

        // for phi, we keep only the diagonal component
        for(label faceI=0;faceI<adjIdx_.nLocalFaces;faceI++)
        {
            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                if (faceI<adjIdx_.nLocalInternalFaces)
                {
                    resScaling = mesh_.magSf()[faceI];
                }
                else
                {
                    label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                    label patchIdx=adjIdx_.bFacePatchI[relIdx];
                    label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    resScaling = mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                }
            } 

            if (adjIO_.isInList<word>(stateName,adjIO_.normalizeStates)) 
            {
                stateScaling = this->getStateScaling(stateName,faceI);
            } 

            PetscInt rowI = adjIdx_.getGlobalAdjointStateIndex(stateName,faceI);
            PetscInt colI = rowI;
            PetscScalar val=-1.0/resScaling*stateScaling;
            MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
        }
    }

    MatAssemblyBegin(jacMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat,MAT_FINAL_ASSEMBLY);

    if(printStatistics) adjIdx_.printMatChars(jacMat);
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(jacMat,"fastPCMat");
    }
}

void AdjointDerivative::calcdRdW(Mat jacMat,const label transposed, const label isPC)
{

    word matName;
    if(isPC==0 and transposed==1) matName = "dRdWT";
    else if (isPC==1 and transposed==1) matName = "dRdWTPC";
    else if (isPC==0 and transposed==0) matName = "dRdW";
    else if (isPC==1 and transposed==0) matName = "dRdWPC";
    else FatalErrorIn("")<<"transposed and isPC not valid"<< abort(FatalError);

    // if we need to read matrix, read them and return
    if(adjIO_.readMatrices && transposed==1)
    {
        adjIO_.readMatrixBinary(jacMat,matName);
        return;
    }

    // zero all the matrices
    MatZeroEntries(jacMat);

    label nColors = adjCon_.getNdRdWColors();

    // main loop to assemble dRdWT,dRdWTPC   
    // we need to calculate the reference Res first
    label isRef=1;
    this->copyStates("Ref2Var");
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);
    
    for(label color = 0; color<nColors; color++)
    {
        label eTime = this->getRunTime();
        // print progress
        if (color%100==0 or color==nColors-1)
        {
            Info<<matName<<" "<<color<<" of "<<nColors<<", ExecutionTime: "<<eTime<<" s"<<endl;
        }
    
        // perturb states 
        this->perturbStates(color,"dRdW");
        // calculate partials using 1st order forward difference
        scalar eps=adjIO_.epsDeriv;
        this->calcResPartDeriv(eps,isPC);
        // reset perturbation
        this->copyStates("Ref2Var");
        // set JMatrix entries by coloring scheme
        if(adjIO_.useColoring)
        {
            // NOTE: if adjIO_.reduceResCon4JacMat is on, we will always use dRdWConPC
            if(adjIO_.reduceResCon4JacMat) adjCon_.calcdRdWColoredColumns(color,1);
            else adjCon_.calcdRdWColoredColumns(color,isPC);
            this->setJacobianMatColored(jacMat,adjCon_.getdRdWColoredColumns(),transposed,isPC);
        }
        else
        {
            this->setJacobianMat(jacMat,color,transposed);
        }
    }
    
    MatAssemblyBegin(jacMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat,MAT_FINAL_ASSEMBLY);
    adjIdx_.printMatChars(jacMat);
    this->calcFlowResidualStatistics("verify");
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(jacMat,matName);
    }
    
    Info<<matName<<" Completed! "<<this->getRunTime()<<" s"<<endl;

    return;
}


void AdjointDerivative::calcdRdVis()
{

    Info<<"Calculating dRdVis"<<endl;

    // read nuInDict
    IOdictionary& transDict
    (
        const_cast<IOdictionary&> 
        ( 
            db_.lookupObject<IOdictionary>("transportProperties") 
        )
    );
    scalar nuInDict = readScalar(transDict.lookup("nu"));

    MatZeroEntries(dRdVis_);
    
    // we need to calculate the reference Res first
    label isRef=1,isPC=0,transposed=0;
    this->copyStates("Ref2Var");
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);
    
    volScalarField& nu                                          
    (                                                                      
        const_cast<volScalarField&>                                             
        (                                                                  
            db_.lookupObject<volScalarField>("nu")                         
        )                                                                  
    ); 
    const volScalarField nuRef=nu; 

    // pertur nu
    forAll(nu,idxI)
    {
        nu[idxI] = nuInDict+adjIO_.epsDerivVis;
    }
    forAll(nu.boundaryField(),patchI)
    {
        forAll(nu.boundaryField()[patchI],faceI)
        {
            nu.boundaryFieldRef()[patchI][faceI] = nuInDict+adjIO_.epsDerivVis;
        }
    }
    // perturb nu key in the dict, this is important since other functions may read values from here
    scalar valPerturbed=nuInDict+adjIO_.epsDerivVis;
    transDict.set("nu",valPerturbed);

    // calculate partials using 1st order forward difference
    scalar eps=adjIO_.epsDerivVis;
    this->calcResPartDeriv(eps,isPC);

    // reset nu
    nu=nuRef;
    transDict.set("nu",nuInDict);

    this->setJacobianMat(dRdVis_,0,transposed);

    MatAssemblyBegin(dRdVis_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dRdVis_,MAT_FINAL_ASSEMBLY);
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(dRdVis_,"dRdVis");
    }

    this->calcFlowResidualStatistics("verify");
}


void AdjointDerivative::calcdRdUIn()
{

    Info<<"Calculating dRdUIn"<<endl;

    MatZeroEntries(dRdUIn_);
    
    // we need to calculate the reference Res first
    label isRef=1,isPC=0,transposed=0;
    this->copyStates("Ref2Var");
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);
    
    for (label i=0; i<3; i++)
    {
        scalar eps=adjIO_.epsDerivUIn;
        // perturb states 
        this->perturbUIn(eps);

        // calculate partials using 1st order forward difference
        this->calcResPartDeriv(eps,isPC);

        // reset perturbation
        this->perturbUIn(-eps);

        this->setJacobianMat(dRdUIn_,i,transposed);
        
    }
    
    MatAssemblyBegin(dRdUIn_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dRdUIn_,MAT_FINAL_ASSEMBLY);
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(dRdUIn_,"dRdUIn");
    }

    this->calcFlowResidualStatistics("verify");
}

void AdjointDerivative::calcdRdFFD(Mat dRdFFD)
{
    
    MatZeroEntries(dRdFFD);

    // read deltaVolPointMat
    label nProcs = Pstream::nProcs();
    
    std::ostringstream np("");
    np<<nProcs;
    std::string fName1="deltaVolPointMatPlusEps_"+np.str()+".bin";

    Info<<"Reading deltaVolPointMat"<<endl;
    
    Mat deltaVolPointMatPlusEps;
    PetscViewer    viewer; 
    PetscInt nFFD = adjIO_.nFFDPoints;
    // read plus mat
    MatCreate(PETSC_COMM_WORLD,&deltaVolPointMatPlusEps);
    MatSetSizes(deltaVolPointMatPlusEps,adjIdx_.nLocalPoints*3,PETSC_DECIDE,PETSC_DETERMINE,nFFD);
    MatSetUp(deltaVolPointMatPlusEps);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,fName1.c_str(),FILE_MODE_READ,&viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT);
    MatLoad(deltaVolPointMatPlusEps,viewer);
    PetscViewerDestroy(&viewer);
    
    Info<< "Calculating dRdFFD..." << endl;
    pointField newPoints(pointsRef_);
    
    // we need to calculate the reference Res first
    label isRef=1,isPC=0,transposed=0;
    this->copyStates("Ref2Var");
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);

    // Loop through each perturbed FFD DV and calculate the perturbed residuals,
    for (label i=0; i<nFFD; i++)
    {
        // print progress
        if (i%100==0 or i==nFFD-1)
        {
            label elapsedClockTime = this->getRunTime();
            Info<<"dRdFFD: "<<i<<" of "<<nFFD<<", ExecutionTime: "<<elapsedClockTime<<" s"<<endl;
        }
        
        // +epsFFD
        this->perturbFFD(deltaVolPointMatPlusEps,i,newPoints);
        
        // calculate partials using 1st order forward difference
        scalar eps=adjIO_.epsDerivFFD;
        this->calcResPartDeriv(eps,isPC);
        
        // reset perturbation
        newPoints=pointsRef_;

        // Index will be based on the iFFD
        this->setJacobianMat(dRdFFD,i,transposed);

    }

    // now update all the x metrics again
    tmp<scalarField> tsweptVols = mesh_.movePoints(pointsRef_);
    
    adjRAS_.correctWallDist();

    MatAssemblyBegin(dRdFFD,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dRdFFD,MAT_FINAL_ASSEMBLY);
    
    MatDestroy(&deltaVolPointMatPlusEps);

    //output the matrix to a file
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(dRdFFD,"dRdFFD");
    }

    this->calcFlowResidualStatistics("verify");
    
    return;
}


void AdjointDerivative::calcdRdXv()
{
    
    MatZeroEntries(dRdXv_);
    
    Info<< "Calculating dRdXv..." << endl;
    pointField newPoints(pointsRef_);
    
    // we need to calculate the reference Res first
    label isRef=1,isPC=0,transposed=0;
    this->copyStates("Ref2Var");
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);

    label nColors = adjCon_.getNdRdXvColors();
    // Loop through each perturbed point and calculate the perturbed residuals,
    for (label color=0; color<nColors; color++)
    {
        // print progress
        if (color%100==0 or color==nColors-1)
        {
            label elapsedClockTime = this->getRunTime();
            Info<<"dRdXv: "<<color<<" of "<<nColors<<", ExecutionTime: "<<elapsedClockTime<<" s"<<endl;
        }
        
        // +epsXv
        this->perturbXv(color,"dRdXv",newPoints);

        // calculate partials using 1st order forward difference
        scalar eps=adjIO_.epsDerivXv;
        this->calcResPartDeriv(eps,isPC);

        // reset perturbation
        newPoints=pointsRef_;

        // Index will be based on the iFFD
        if(adjIO_.useColoring)
        {
            adjCon_.calcdRdXvColoredColumns(color);
            this->setJacobianMatColored(dRdXv_,adjCon_.getdRdXvColoredColumns(),transposed,isPC);
        }
        else
        {
            this->setJacobianMat(dRdXv_,color,transposed);
        }

    }

    // now update all the x metrics again
    tmp<scalarField> tsweptVols = mesh_.movePoints(pointsRef_);
    
    adjRAS_.correctWallDist();

    MatAssemblyBegin(dRdXv_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(dRdXv_,MAT_FINAL_ASSEMBLY);

    //output the matrix to a file
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(dRdXv_,"dRdXv");
    }

    this->calcFlowResidualStatistics("verify");
    
    return;
}

void AdjointDerivative::calcdFdVis(const word objFunc)
{

    Info<<"Calculating dFdVis for "<<objFunc<<endl;

    VecZeroEntries(dFdVis_);

    // read nuInDict
    IOdictionary& transDict
    (
        const_cast<IOdictionary&> 
        ( 
            db_.lookupObject<IOdictionary>("transportProperties") 
        )
    );
    scalar nuInDict = readScalar(transDict.lookup("nu"));
 
    // we need to calculate the reference Res first
    label isRef=1;
    this->copyStates("Ref2Var");
    adjObj_.calcObjFuncs(objFunc,isRef);

    volScalarField& nu                                          
    (                                                                      
        const_cast<volScalarField&>                                             
        (                                                                  
            db_.lookupObject<volScalarField>("nu")                         
        )                                                                  
    );
    const volScalarField nuRef=nu; 

    // perturb nuField
    forAll(nu,idxI)
    {
        nu[idxI] = nuInDict+adjIO_.epsDerivVis;
    }
    forAll(nu.boundaryField(),patchI)
    {
        forAll(nu.boundaryField()[patchI],faceI)
        {
            nu.boundaryFieldRef()[patchI][faceI] = nuInDict+adjIO_.epsDerivVis;
        }
    }
    // perturb nu key in the dict, this is important since other functions may read values from here
    scalar valPerturbed=nuInDict+adjIO_.epsDerivVis;
    transDict.set("nu",valPerturbed);

    // we need to update nut since nut may depend on nu, e.g., in wall functions
    adjRAS_.updateNut();
    this->updateStateVariableBCs();

    // calculate partials using 1st order forward difference
    adjObj_.calcObjFuncPartDerivs(adjIO_.epsDerivVis,objFunc);

    // reset nu
    nu=nuRef;
    transDict.set("nu",nuInDict);

    PetscScalar val;
    val = adjObj_.getObjFuncPartDeriv(objFunc);

    if (fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
    {
        VecSetValue(dFdVis_,0,val,INSERT_VALUES);
    }

    this->calcFlowResidualStatistics("verify");

    VecAssemblyBegin(dFdVis_);
    VecAssemblyEnd(dFdVis_);
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeVectorASCII(dFdVis_,"dFdVis_"+objFunc);
    }

    return;

}

void AdjointDerivative::calcdFdUIn(const word objFunc)
{

    Info<<"Calculating dFdUIn for "<<objFunc<<endl;

    VecZeroEntries(dFdUIn_);

    if(adjIO_.inletPatches.size()>1) 
        FatalErrorIn("")<<"Inletpatch size >1"<< abort(FatalError);
 
    // we need to calculate the reference Res first
    label isRef=1;
    this->copyStates("Ref2Var");
    adjObj_.calcObjFuncs(objFunc,isRef);
    
    for (PetscInt i=0; i<3; i++)
    {
        scalar eps=adjIO_.epsDerivUIn;
        
        // perturb states 
        this->perturbUIn(eps);

        // calculate partials using 1st order forward difference
        adjObj_.calcObjFuncPartDerivs(eps,objFunc);

        // reset perturbation
        this->perturbUIn(-eps);

        PetscScalar val;

        val = adjObj_.getObjFuncPartDeriv(objFunc);

        if (fabs(val) > adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
        {
            PetscInt rowI = i;
            VecSetValue(dFdUIn_,rowI,val,INSERT_VALUES);
        }
        
    }

    this->calcFlowResidualStatistics("verify");

    VecAssemblyBegin(dFdUIn_);
    VecAssemblyEnd(dFdUIn_);
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeVectorASCII(dFdUIn_,"dFdUIn_"+objFunc);
    }

    return;

}

void AdjointDerivative::calcdFdFFD(const word objFunc)
{
    VecZeroEntries(dFdFFD_);
    
    // read deltaVolPointMat
    label nProcs = Pstream::nProcs();
    
    std::ostringstream np("");
    np<<nProcs;
    std::string fName1="deltaVolPointMatPlusEps_"+np.str()+".bin";

    Info<<"Reading deltaVolPointMat"<<endl;
    
    Mat deltaVolPointMatPlusEps;
    PetscViewer    viewer; 
    PetscInt nFFD = adjIO_.nFFDPoints;
    // read plus mat
    MatCreate(PETSC_COMM_WORLD,&deltaVolPointMatPlusEps);
    MatSetSizes(deltaVolPointMatPlusEps,adjIdx_.nLocalPoints*3,PETSC_DECIDE,PETSC_DETERMINE,nFFD);
    MatSetUp(deltaVolPointMatPlusEps);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,fName1.c_str(),FILE_MODE_READ,&viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_DEFAULT);
    MatLoad(deltaVolPointMatPlusEps,viewer);
    PetscViewerDestroy(&viewer);

    // calc dFdFFD
    Info<< "Calculating dFdFFD for "<<objFunc << endl;
    
    // we need to calculate the reference Res first
    label isRef=1;
    this->copyStates("Ref2Var");
    adjObj_.calcObjFuncs(objFunc,isRef);

    pointField newPoints(pointsRef_);
    
    // Loop through each perturbed FFD DV and calculate the perturbed residuals,
    for (label i=0; i<nFFD; i++)
    {
        // print progress
        if (i%100==0 or i==nFFD-1)
        {
            label elapsedClockTime = this->getRunTime();
            Info<<"dFdFFD: "<<i<<" of "<<nFFD<<", ExecutionTime: "<<elapsedClockTime<<" s"<<endl;
        }

        // +epsFFD
        this->perturbFFD(deltaVolPointMatPlusEps,i,newPoints);
        
        // calculate partials using 1st order forward difference
        scalar eps=adjIO_.epsDerivFFD;
        adjObj_.calcObjFuncPartDerivs(eps,objFunc);
        
        // reset perturbation
        newPoints=pointsRef_;
        
        PetscScalar val;
        val = adjObj_.getObjFuncPartDeriv(objFunc);       

        if (fabs(val)>adjIO_.minTolJac && fabs(val) < adjIO_.maxTolJac)
        {
            VecSetValue(dFdFFD_,i,val,INSERT_VALUES);
        }
        
    }

    // now update all the x metrics again
    tmp<scalarField> tsweptVols = mesh_.movePoints(pointsRef_);
    
    adjRAS_.correctWallDist();

    VecAssemblyBegin(dFdFFD_);
    VecAssemblyEnd(dFdFFD_);
    
    MatDestroy(&deltaVolPointMatPlusEps);

    //output the matrix to a file
    if(adjIO_.writeMatrices)
    {
        //adjIO_.writeVectorBinary(dFdFFD_,"dFdFFD_"+objFunc);
        adjIO_.writeVectorASCII(dFdFFD_,"dFdFFD_"+objFunc);
    }

    this->calcFlowResidualStatistics("verify");

}

void AdjointDerivative::calcdFdXv(const word objFunc)
{

    // calc dFdXv
    Info<< "Calculating dFdXv for "<<objFunc << endl;

    VecZeroEntries(dFdXv_);

    label nColors = adjCon_.getNdFdXvColors();

    // we need to calculate the reference Res first
    label isRef=1;
    this->copyStates("Ref2Var");
    adjObj_.calcObjFuncs(objFunc,isRef);

    pointField newPoints(pointsRef_);
    
    // Loop through each perturbed point and calculate the perturbed residuals,
    for (label color=0; color<nColors; color++)
    {
        // print progress
        if (color%100==0 or color==nColors-1)
        {
            label elapsedClockTime = this->getRunTime();
            Info<<"dFdXv: "<<color<<" of "<<nColors<<", ExecutionTime: "<<elapsedClockTime<<" s"<<endl;
        }

        // +epsFFD
        this->perturbXv(color,"dFdXv",newPoints);
        
        // calculate partials using 1st order forward difference
        scalar eps=adjIO_.epsDerivXv;
        adjObj_.calcObjFuncPartDerivs(eps,objFunc);
        
        // reset perturbation
        newPoints=pointsRef_;
        
        if(adjIO_.useColoring) 
        {
            adjCon_.calcdFdXvColoredColumns(color);
            this->setdFVecColored(dFdXv_,adjCon_.getdFdXvColoredColumns(),objFunc);
        }
        else 
        {
            this->setdFVec(dFdXv_,objFunc,color);
        }
        
    }

    // now update all the x metrics again
    tmp<scalarField> tsweptVols = mesh_.movePoints(pointsRef_);
    
    adjRAS_.correctWallDist();

    this->calcFlowResidualStatistics("verify");

    VecAssemblyBegin(dFdXv_);
    VecAssemblyEnd(dFdXv_);
    
    //output the matrix to a file
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeVectorBinary(dFdXv_,"dFdXv_"+objFunc);
        adjIO_.writeVectorASCII(dFdXv_,"dFdXv_"+objFunc);
        // compute the undecomposed dFdXvSerial and save to file for debugging
        this->setObjFuncsSensSerial(dFdXv_,dFdXvSerial_,objFunc,"partialDeriv");
    }

    return;

}

void AdjointDerivative::calcdFdW(const word objFunc)
{
    
    Info<< "Calculating dFdW for "<<objFunc<<" "<<this->getRunTime()<<" s"<<endl;

    VecZeroEntries(dFdW_);
        
    label nColors = adjCon_.getNdFdWColors();
    
    // we need to calculate the reference Res first
    label isRef=1;
    this->copyStates("Ref2Var");
    adjObj_.calcObjFuncs(objFunc,isRef);

    for(label color = 0; color<nColors; color++)
    {
        label eTime = this->getRunTime();
        // print progress
        if (color%100==0 or color==nColors-1)
        {
            Info<<"dFdW: "<<color<<" of "<<nColors<<", ExecutionTime: "<<eTime<<" s"<<endl;
        }
        
        // perturb states 
        this->perturbStates(color,"dFdW");

        // calculate partials using 1st order forward difference
        scalar eps=adjIO_.epsDeriv;
        adjObj_.calcObjFuncPartDerivs(eps,objFunc);

        // reset perturbation
        this->copyStates("Ref2Var");

        if(adjIO_.useColoring) 
        {
            adjCon_.calcdFdWColoredColumns(color);
            this->setdFVecColored(dFdW_,adjCon_.getdFdWColoredColumns(),objFunc);
        }
        else 
        {
            this->setdFVec(dFdW_,objFunc,color);
        }
    }
    
    VecAssemblyBegin(dFdW_);
    VecAssemblyEnd(dFdW_);

    // NOTE: for VMS, we need to scale dFdW_, see AdjointObjectiveFunction::calcVMS for reference 
    if (objFunc=="VMS" && adjIO_.useColoring) 
    {
        const scalar& KSCoeff =adjIO_.referenceValues["KSCoeff"];
        scalar scaleFactor = 1.0/KSCoeff/adjObj_.KSExpSumRef;
        VecScale(dFdW_,scaleFactor);
    }

    this->calcFlowResidualStatistics("verify");

    // assign dFdW to dFdWAll
    this->dFdW2dFdWAll(objFunc);

    if(adjIO_.writeMatrices)
    {
        adjIO_.writeVectorASCII(dFdW_,"dFdW_"+objFunc);
        adjIO_.writeVectorBinary(dFdW_,"dFdW_"+objFunc);
    }

    Info<< "Calculating dFdW for "<<objFunc<<" completed. "<<this->getRunTime()<<" s"<<endl;

    return;
}

void AdjointDerivative::dFdW2dFdWAll(const word objFunc)
{
    PetscInt Istart, Iend;
    PetscScalar val;
    PetscInt colI=-1000;

    label colSet=0;
    forAll(adjIO_.objFuncs,idxI)
    {
        if (objFunc==adjIO_.objFuncs[idxI])
        {
            colI=idxI;
            colSet+=1;
        }
    }
    if (colSet!=1) FatalErrorIn("")<<"col not valid!!"<< abort(FatalError);

    VecGetOwnershipRange(dFdW_,&Istart,&Iend);
    for (PetscInt i=Istart;i<Iend;i++)
    {
        VecGetValues(dFdW_,1,&i,&val);
        if ( fabs(val)>adjIO_.minTolJac and fabs(val)<adjIO_.maxTolJac)
        {
            MatSetValue(dFdWAll_,i,colI,val,INSERT_VALUES);
        }
    }

    return;
    
}

void AdjointDerivative::dFdWAll2dFdW(const word objFunc)
{
    VecZeroEntries(dFdW_);

    PetscInt Istart, Iend;
    PetscScalar val;
    PetscInt colI=-1000;

    label colSet=0;
    forAll(adjIO_.objFuncs,idxI)
    {
        if (objFunc==adjIO_.objFuncs[idxI])
        {
            colI=idxI;
            colSet+=1;
        }
    }
    if (colSet!=1) FatalErrorIn("")<<"col not valid!!"<< abort(FatalError);

    MatGetOwnershipRange(dFdWAll_,&Istart,&Iend);
    for (PetscInt i=Istart;i<Iend;i++)
    {
        MatGetValues(dFdWAll_,1,&i,1,&colI,&val);
        if ( fabs(val)>adjIO_.minTolJac and fabs(val)<adjIO_.maxTolJac)
        {
            VecSetValue(dFdW_,i,val,INSERT_VALUES);
        }
    }

    VecAssemblyBegin(dFdW_);
    VecAssemblyEnd(dFdW_);

    return;
    
}

void AdjointDerivative::calcTotalDerivVis()
{
    Info<<"Calculating Vis Total Derivatives. "<<this->getRunTime()<<" s"<<endl;
        
    this->initializedRdVis();
    this->calcdRdVis();
    
    this->initializedFdVis();
    
    VecCreate(PETSC_COMM_WORLD,&dFdVisTotal_);
    VecSetSizes(dFdVisTotal_,PETSC_DETERMINE,1);
    VecSetFromOptions(dFdVisTotal_);
    
    Vec psi;
    VecDuplicate(psi_,&psi);
    
    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];
        word psiName = "psi_"+objFunc;
        word sensName = "objFuncsSens_d"+objFunc+"dVis.bin";
        word sensNameASCII = "objFuncsSens_d"+objFunc+"dVis.dat";
        
        this->calcdFdVis(objFunc);

        VecZeroEntries(psi);
        adjIO_.readVectorBinary(psi,psiName.c_str());
        
        VecZeroEntries(dFdVisTotal_);
        MatMultTranspose(dRdVis_,psi,dFdVisTotal_);
        VecAXPY(dFdVisTotal_,-1.0,dFdVis_);
        VecScale(dFdVisTotal_,-1.0);
        
        PetscViewer    viewer; 
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,sensName.c_str(),FILE_MODE_WRITE,&viewer);
        VecView(dFdVisTotal_,viewer);
        PetscViewerDestroy(&viewer);
        
        PetscViewer    viewer1; 
        PetscViewerASCIIOpen(PETSC_COMM_WORLD,sensNameASCII.c_str(),&viewer1);
        PetscViewerPushFormat(viewer1,PETSC_VIEWER_ASCII_MATLAB); // write all the digits
        VecView(dFdVisTotal_,viewer1);
        PetscViewerDestroy(&viewer1);
    }

    Info<<"Calculating Vis Total Derivatives. Completed. "<<this->getRunTime()<<" s"<<endl;

    return;

}


void AdjointDerivative::calcTotalDerivUIn()
{
    Info<<"Calculating UIn Total Derivatives. "<<this->getRunTime()<<" s"<<endl;
        
    this->initializedRdUIn();
    this->calcdRdUIn();
  
    this->initializedFdUIn();
    
    VecCreate(PETSC_COMM_WORLD,&dFdUInTotal_);
    VecSetSizes(dFdUInTotal_,PETSC_DETERMINE,3);
    VecSetFromOptions(dFdUInTotal_);
    
    Vec psi;
    VecDuplicate(psi_,&psi);
    
    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];
        word psiName = "psi_"+objFunc;
        word sensName = "objFuncsSens_d"+objFunc+"dUIn.bin";
        word sensNameASCII = "objFuncsSens_d"+objFunc+"dUIn.dat";
        
        this->calcdFdUIn(objFunc);
        VecZeroEntries(psi);
        adjIO_.readVectorBinary(psi,psiName.c_str());
        
        VecZeroEntries(dFdUInTotal_);
        MatMultTranspose(dRdUIn_,psi,dFdUInTotal_);
        VecAXPY(dFdUInTotal_,-1.0,dFdUIn_);
        VecScale(dFdUInTotal_,-1.0);
        
        PetscViewer    viewer; 
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,sensName.c_str(),FILE_MODE_WRITE,&viewer);
        VecView(dFdUInTotal_,viewer);
        PetscViewerDestroy(&viewer);
        
        PetscViewer    viewer1; 
        PetscViewerASCIIOpen(PETSC_COMM_WORLD,sensNameASCII.c_str(),&viewer1);
        PetscViewerPushFormat(viewer1,PETSC_VIEWER_ASCII_MATLAB); // write all the digits
        VecView(dFdUInTotal_,viewer1);
        PetscViewerDestroy(&viewer1);
    }

    Info<<"Calculating UIn Total Derivatives. Completed. "<<this->getRunTime()<<" s"<<endl;

    return;

}

void AdjointDerivative::calcTotalDerivFFD()
{
    Info<<"Calculating FFD Total Derivatives. "<<this->getRunTime()<<" s"<<endl;

    this->initializedRdFFD();

    this->calcdRdFFD(dRdFFD_);

    this->initializedFdFFD();

    VecCreate(PETSC_COMM_WORLD,&dFdFFDTotal_);
    VecSetSizes(dFdFFDTotal_,PETSC_DETERMINE,adjIO_.nFFDPoints);
    VecSetFromOptions(dFdFFDTotal_);
    
    Vec psi;
    VecDuplicate(psi_,&psi);
    
    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];
        word psiName = "psi_"+objFunc;
        word sensName = "objFuncsSens_d"+objFunc+"dFFD.bin";
        word sensNameASCII = "objFuncsSens_d"+objFunc+"dFFD.dat";
        
        this->calcdFdFFD(objFunc);

        VecZeroEntries(psi);
        adjIO_.readVectorBinary(psi,psiName.c_str());
        
        VecZeroEntries(dFdFFDTotal_);
        MatMultTranspose(dRdFFD_,psi,dFdFFDTotal_);
        VecAXPY(dFdFFDTotal_,-1.0,dFdFFD_);
        VecScale(dFdFFDTotal_,-1.0);
        
        PetscViewer    viewer; 
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,sensName.c_str(),FILE_MODE_WRITE,&viewer);
        VecView(dFdFFDTotal_,viewer);
        PetscViewerDestroy(&viewer);
        
        PetscViewer    viewer1; 
        PetscViewerASCIIOpen(PETSC_COMM_WORLD,sensNameASCII.c_str(),&viewer1);
        PetscViewerPushFormat(viewer1,PETSC_VIEWER_ASCII_MATLAB); // write all the digits
        VecView(dFdFFDTotal_,viewer1);
        PetscViewerDestroy(&viewer1);
    }  

    Info<<"Calculating FFD Total Derivatives. Completed. "<<this->getRunTime()<<" s"<<endl;

    return; 

}


void AdjointDerivative::calcTotalDerivXv()
{
    Info<<"Calculating Xv Total Derivatives. "<<this->getRunTime()<<" s"<<endl;

    if(adjIO_.useColoring) 
    {
        // compute preallocation vecs for dRdXv
        adjCon_.setupdRdXvCon(1);
        adjCon_.initializedRdXvCon();
        adjCon_.setupdRdXvCon(0);
        // Read in the precomputed coloring
        adjCon_.readdRdXvColoring();
    }

    this->initializedRdXv();

    this->calcdRdXv();

    this->initializedFdXv();

    VecCreate(PETSC_COMM_WORLD,&dFdXvTotal_);
    VecSetSizes(dFdXvTotal_,adjIdx_.nLocalXv,PETSC_DETERMINE);
    VecSetFromOptions(dFdXvTotal_);

    // sensitivity for undecomposed domain
    VecCreate(PETSC_COMM_WORLD,&dFdXvTotalSerial_);
    VecSetSizes(dFdXvTotalSerial_,PETSC_DETERMINE,adjIdx_.nUndecomposedPoints*3);
    VecSetFromOptions(dFdXvTotalSerial_);
    
    Vec psi;
    VecDuplicate(psi_,&psi);
    
    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];
        word psiName = "psi_"+objFunc;
        word sensName = "objFuncsSens_d"+objFunc+"dXv.bin";
        word sensNameASCII = "objFuncsSens_d"+objFunc+"dXv.dat";
        
        if(adjIO_.useColoring) 
        {
            adjCon_.initializedFdXvCon(objFunc);
            adjCon_.setupObjFuncCon(objFunc,"dFdXv");
            adjCon_.readdFdXvColoring(objFunc);
        }
        this->calcdFdXv(objFunc);
        if(adjIO_.useColoring) adjCon_.deletedFdXvCon();

        VecZeroEntries(psi);
        adjIO_.readVectorBinary(psi,psiName.c_str());
        
        VecZeroEntries(dFdXvTotal_);

        MatMultTranspose(dRdXv_,psi,dFdXvTotal_);
        VecAXPY(dFdXvTotal_,-1.0,dFdXv_);
        VecScale(dFdXvTotal_,-1.0);

        PetscViewer    viewer; 
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,sensName.c_str(),FILE_MODE_WRITE,&viewer);
        VecView(dFdXvTotal_,viewer);
        PetscViewerDestroy(&viewer);
        
        PetscViewer    viewer1; 
        PetscViewerASCIIOpen(PETSC_COMM_WORLD,sensNameASCII.c_str(),&viewer1);
        PetscViewerPushFormat(viewer1,PETSC_VIEWER_ASCII_MATLAB); // write all the digits
        VecView(dFdXvTotal_,viewer1);
        PetscViewerDestroy(&viewer1);

        // now set sensitivity for undecomposed domain based on dFdXvTotal_ and save them to files
        this->setObjFuncsSensSerial(dFdXvTotal_,dFdXvTotalSerial_,objFunc,"totalDeriv");
    }

    Info<<"Calculating Xv Total Derivatives. Completed. "<<this->getRunTime()<<" s"<<endl;

}


void AdjointDerivative::setObjFuncsSensSerial(Vec decompVec, Vec undecompVec, word objFunc,word mode)
{
    // set the undecomposed sens vec based on the decomposed sens vec

    VecZeroEntries(undecompVec);

    PetscInt  undecompPointIdx, rowI;
    PetscScalar valIn;

    PetscScalar *decompVecArray;
    VecGetArray(decompVec,&decompVecArray);
    
    for(label i=0; i<adjIdx_.nLocalPoints; i++)
    {
        for (label j=0;j<3;j++)
        {
            label localXvIdx = adjIdx_.getLocalXvIndex(i,j);
            valIn = decompVecArray[localXvIdx];
    
            undecompPointIdx = adjIdx_.pointProcAddressing[i];
            rowI = undecompPointIdx*3+j; // this is essentially getLocalXvIndex(undecompPointIdx,j)
            // we need to use ADD values since a undecomposed point may correspond to multiple decomposed points
            VecSetValue(undecompVec,rowI,valIn,ADD_VALUES);  
        }
    }

    VecRestoreArray(decompVec,&decompVecArray);

    VecAssemblyBegin(undecompVec);
    VecAssemblyEnd(undecompVec);

    word sensNameSerial,sensNameSerialASCII;
    if (mode=="partialDeriv")
    {
        sensNameSerial = "dFdXv_"+objFunc+"_Serial.bin";
        sensNameSerialASCII = "dFdXv_"+objFunc+"_Serial.dat";
    }
    else if (mode=="totalDeriv")
    {
        sensNameSerial = "objFuncsSens_d"+objFunc+"dXv_Serial.bin";
        sensNameSerialASCII = "objFuncsSens_d"+objFunc+"dXv_Serial.dat";
    }
    else
    {
        FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    }

    PetscViewer    viewer; 
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,sensNameSerial.c_str(),FILE_MODE_WRITE,&viewer);
    VecView(undecompVec,viewer);
    PetscViewerDestroy(&viewer);
    
    PetscViewer    viewer1; 
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,sensNameSerialASCII.c_str(),&viewer1);
    PetscViewerPushFormat(viewer1,PETSC_VIEWER_ASCII_MATLAB); // write all the digits
    VecView(undecompVec,viewer1);
    PetscViewerDestroy(&viewer1);

    return;

}

void AdjointDerivative::calcTotalDeriv()
{
    Info<<"Calculating Total Derivatives..."<<this->getRunTime()<<" s"<<endl;

    if (adjIO_.isInList<word>("Xv",adjIO_.adjDVTypes))
    {
        this->calcTotalDerivXv();
    }

    if( adjIO_.isInList<word>("UIn",adjIO_.adjDVTypes)) 
    {
        this->calcTotalDerivUIn();   
    }

    if (adjIO_.isInList<word>("FFD",adjIO_.adjDVTypes))
    {
        this->calcTotalDerivFFD();
    }

    if (adjIO_.isInList<word>("Vis",adjIO_.adjDVTypes))
    {
        this->calcTotalDerivVis();
    }

}

void AdjointDerivative::solveAdjoint(const word objFunc)
{

    Info<<"Solving Adjoint... "<<this->getRunTime()<<" s"<<endl;
                    
    //Solve adjoint
    VecZeroEntries(psi_);

    KSPSolve(ksp_,dFdW_,psi_);

    // check if we need to extract the computed eigenvalue and save to files
    if(adjIO_.adjGMRESCalcEigen)
    {
        word prefix = "PreconditionedJacobianEigenValues_"+objFunc+".dat";
        OFstream fOut(prefix);
        PetscInt nn=adjIO_.adjGMRESRestart;
        PetscReal realEigen[nn],complexEigen[nn];
        PetscInt nEigen;
        KSPComputeEigenvalues(ksp_,nn,realEigen,complexEigen,&nEigen);
    
        for(label i=0;i<nEigen;i++)
        {
            fOut<<realEigen[i]<<" "<<complexEigen[i]<<endl;
        }
    }
    
    //Print convergence information
    label its;
    KSPGetIterationNumber(ksp_,&its);
    PetscScalar finalResNorm;
    KSPGetResidualNorm(ksp_,&finalResNorm);
    PetscPrintf
    (
        PETSC_COMM_WORLD,
        "Main iteration %D KSP Residual norm %14.12e %d s \n",
        its,
        finalResNorm,
        this->getRunTime()
    );
    PetscPrintf(PETSC_COMM_WORLD,"Total iterations %D\n",its);

    VecAssemblyBegin(psi_);
    VecAssemblyEnd(psi_);

    // write the psi vectors for each objFunc
    word prefix = "psi_"+objFunc;
    adjIO_.writeVectorBinary(psi_,prefix);

    if(adjIO_.writeMatrices) adjIO_.writeVectorASCII(psi_,prefix);

    Info<<"Solving Adjoint... Completed! "<<this->getRunTime()<<" s"<<endl;

    return;
}

void AdjointDerivative::solveAdjointSegregated(const word objFunc)
{
    Info<<"Segregated Adjoint"<<endl;

    Mat dRdUT, dRdPT, dRdPhiT, dRdNuTildaT;
    Mat dRdUTPC, dRdPTPC, dRdPhiTPC, dRdNuTildaTPC;
    Vec dFdU, dFdP, dFdPhi, dFdNuTilda;
    Vec dFdU0, dFdP0, dFdPhi0, dFdNuTilda0;
    KSP ksp_dRdUT, ksp_dRdPT, ksp_dRdPhiT, ksp_dRdNuTildaT;
    Vec psiU, psiP, psiNuTilda, psiPhi;
    Vec psiU0, psiP0, psiNuTilda0, psiPhi0;
    Vec psiCoupled, psiCoupledTmp, rTotal;

    scalar rTotalNorm2=1, rTotalNorm2Initial=1;

    scalar alphaU = adjIO_.segAdjParameters["alphaU"];
    scalar alphaP = adjIO_.segAdjParameters["alphaP"];
    scalar alphaNuTilda = adjIO_.segAdjParameters["alphaNuTilda"];
    scalar alphaPhi = adjIO_.segAdjParameters["alphaPhi"];

    scalar maxIters = adjIO_.segAdjParameters["maxIters"];
    scalar relTol = adjIO_.segAdjParameters["relTol"];
    scalar absTol = adjIO_.segAdjParameters["absTol"];

    label its;
    PetscScalar finalResNorm;

    label nCells = adjIdx_.nLocalCells;
    label nFaces = adjIdx_.nLocalFaces;

    this->initializeSegregatedMat(&dRdUT,"U");
    this->initializeSegregatedMat(&dRdPT,"p");
    this->initializeSegregatedMat(&dRdNuTildaT,"nuTilda");
    this->initializeSegregatedMat(&dRdPhiT,"phi");

    this->initializeSegregatedMat(&dRdUTPC,"U");
    this->initializeSegregatedMat(&dRdPTPC,"p");
    this->initializeSegregatedMat(&dRdNuTildaTPC,"nuTilda");
    this->initializeSegregatedMat(&dRdPhiTPC,"phi");

    VecCreate(PETSC_COMM_WORLD,&dFdU);
    VecSetSizes(dFdU,nCells*3,PETSC_DETERMINE);
    VecSetFromOptions(dFdU);
    VecDuplicate(dFdU,&dFdU0);
    VecDuplicate(dFdU,&psiU);
    VecDuplicate(dFdU,&psiU0);

    VecCreate(PETSC_COMM_WORLD,&dFdPhi);
    VecSetSizes(dFdPhi,nFaces,PETSC_DETERMINE);
    VecSetFromOptions(dFdPhi);
    VecDuplicate(dFdPhi,&dFdPhi0);
    VecDuplicate(dFdPhi,&psiPhi);
    VecDuplicate(dFdPhi,&psiPhi0);

    VecCreate(PETSC_COMM_WORLD,&dFdP);
    VecSetSizes(dFdP,nCells,PETSC_DETERMINE);
    VecSetFromOptions(dFdP);
    VecDuplicate(dFdP,&dFdP0);
    VecDuplicate(dFdP,&psiP);
    VecDuplicate(dFdP,&psiP0);

    VecDuplicate(dFdP,&dFdNuTilda);
    VecDuplicate(dFdP,&dFdNuTilda0);
    VecDuplicate(dFdP,&psiNuTilda);
    VecDuplicate(dFdP,&psiNuTilda0);

    VecDuplicate(psi_,&psiCoupled);
    VecDuplicate(psi_,&psiCoupledTmp);
    VecDuplicate(psi_,&rTotal);

    VecZeroEntries(psiU);
    VecZeroEntries(psiP);
    VecZeroEntries(psiPhi);
    VecZeroEntries(psiNuTilda);

    // extract the segregated
    this->extractSegregatedAdjointMat(dRdWT_,dRdUT,"U");
    this->extractSegregatedAdjointMat(dRdWTPC_,dRdUTPC,"U");
    this->extractSegregatedAdjointVec(dFdW_,dFdU,"U");
    VecCopy(dFdU,dFdU0);

    this->extractSegregatedAdjointMat(dRdWT_,dRdPT,"p");
    this->extractSegregatedAdjointMat(dRdWTPC_,dRdPTPC,"p");
    this->extractSegregatedAdjointVec(dFdW_,dFdP,"p");
    VecCopy(dFdP,dFdP0);

    this->extractSegregatedAdjointMat(dRdWT_,dRdNuTildaT,"nuTilda");
    this->extractSegregatedAdjointMat(dRdWTPC_,dRdNuTildaTPC,"nuTilda");
    this->extractSegregatedAdjointVec(dFdW_,dFdNuTilda,"nuTilda");
    VecCopy(dFdNuTilda,dFdNuTilda0);

    this->extractSegregatedAdjointMat(dRdWT_,dRdPhiT,"phi");
    this->extractSegregatedAdjointMat(dRdWTPC_,dRdPhiTPC,"phi");
    this->extractSegregatedAdjointVec(dFdW_,dFdPhi,"phi");
    VecCopy(dFdPhi,dFdPhi0);

    // we no longer need dRdWTPC
    MatDestroy(&dRdWTPC_);

    // create the linear ksp object
    Info<<"Setting up Segregated Adjoint KSP"<< endl;
    // add options
    dictionary adjOptions;
    adjOptions.add("GMRESRestart",adjIO_.adjGMRESRestart);
    adjOptions.add("GlobalPCIters",adjIO_.adjGlobalPCIters);
    adjOptions.add("ASMOverlap",adjIO_.adjASMOverlap);
    adjOptions.add("LocalPCIters",adjIO_.adjLocalPCIters);
    adjOptions.add("JacMatReOrdering",adjIO_.adjJacMatReOrdering);
    adjOptions.add("PCFillLevel",adjIO_.adjPCFillLevel);
    adjOptions.add("GMRESMaxIters",adjIO_.adjGMRESMaxIters);
    adjOptions.add("GMRESRelTol",adjIO_.adjGMRESRelTol);
    adjOptions.add("GMRESAbsTol",adjIO_.adjGMRESAbsTol);
    adjOptions.add("printInfo",1);

    // crete KSP
    this->createMLRKSP(&ksp_dRdUT,dRdUT,dRdUTPC,adjOptions);
    this->createMLRKSP(&ksp_dRdPT,dRdPT,dRdPTPC,adjOptions);
    this->createMLRKSP(&ksp_dRdNuTildaT,dRdNuTildaTPC,dRdNuTildaT,adjOptions);
    this->createMLRKSP(&ksp_dRdPhiT,dRdPhiT,dRdPhiTPC,adjOptions);

    KSPSetInitialGuessNonzero(ksp_dRdUT,PETSC_TRUE);
    KSPSetInitialGuessNonzero(ksp_dRdPT,PETSC_TRUE);
    KSPSetInitialGuessNonzero(ksp_dRdNuTildaT,PETSC_TRUE);
    KSPSetInitialGuessNonzero(ksp_dRdPhiT,PETSC_TRUE);

    VecZeroEntries(psiCoupled);
    for(label n=0;n<round(maxIters);n++)
    {

        Info<<"Step: "<<n<<endl;
        VecZeroEntries(rTotal);
        MatMult(dRdWT_,psiCoupled,rTotal);
        VecAXPY(rTotal,-1.0,dFdW_);
        VecNorm(rTotal,NORM_2,&rTotalNorm2);
        Info<<"Total Residual: "<<rTotalNorm2<<endl;
        if (n==0) rTotalNorm2Initial= rTotalNorm2;

        if(rTotalNorm2 < absTol) break;
        if(rTotalNorm2/rTotalNorm2Initial < relTol) break;

        // U
        VecZeroEntries(psiU0);
        this->setSegregatedVecs(psiCoupled,psiU0,"U",1.0,"segregated2Coupled"); // zero out
        MatMult(dRdWT_,psiCoupled,psiCoupledTmp);
        VecCopy(dFdU0,dFdU);
        this->setSegregatedVecs(psiCoupledTmp,dFdU,"U",-1.0,"coupledAdd2Segregated"); // subtract to rhs

        VecCopy(psiU,psiU0);
        KSPSolve(ksp_dRdUT,dFdU,psiU);
        VecAssemblyBegin(psiU);
        VecAssemblyEnd(psiU);
        this->underRelaxVec(psiU,psiU0,alphaU);
        this->setSegregatedVecs(psiCoupled,psiU,"U",1.0,"segregated2Coupled"); // update psiCoupled
    
        //Print convergence information
        KSPGetIterationNumber(ksp_dRdUT,&its);
        KSPGetResidualNorm(ksp_dRdUT,&finalResNorm);
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %d s UFinal\n",
            its,
            finalResNorm,
            this->getRunTime()
        );

        // P
        VecZeroEntries(psiP0);
        this->setSegregatedVecs(psiCoupled,psiP0,"p",1.0,"segregated2Coupled"); // zero out
        MatMult(dRdWT_,psiCoupled,psiCoupledTmp);
        VecCopy(dFdP0,dFdP);
        this->setSegregatedVecs(psiCoupledTmp,dFdP,"p",-1.0,"coupledAdd2Segregated"); // subtract to rhs

        VecCopy(psiP,psiP0);
        KSPSolve(ksp_dRdPT,dFdP,psiP);
        VecAssemblyBegin(psiP);
        VecAssemblyEnd(psiP);
        this->underRelaxVec(psiP,psiP0,alphaP);
        this->setSegregatedVecs(psiCoupled,psiP,"p",1.0,"segregated2Coupled"); // update psiCoupled
    
        //Print convergence information
        KSPGetIterationNumber(ksp_dRdPT,&its);
        KSPGetResidualNorm(ksp_dRdPT,&finalResNorm);
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %d s PFinal\n",
            its,
            finalResNorm,
            this->getRunTime()
        );


        // phi
        VecZeroEntries(psiPhi0);
        this->setSegregatedVecs(psiCoupled,psiPhi0,"phi",1.0,"segregated2Coupled"); // zero out
        MatMult(dRdWT_,psiCoupled,psiCoupledTmp);
        VecCopy(dFdPhi0,dFdPhi);
        this->setSegregatedVecs(psiCoupledTmp,dFdPhi,"phi",-1.0,"coupledAdd2Segregated"); // subtract to rhs

        VecCopy(psiPhi,psiPhi0);
        KSPSolve(ksp_dRdPhiT,dFdPhi,psiPhi);
        VecAssemblyBegin(psiPhi);
        VecAssemblyEnd(psiPhi);
        this->underRelaxVec(psiPhi,psiPhi0,alphaPhi);
        this->setSegregatedVecs(psiCoupled,psiPhi,"phi",1.0,"segregated2Coupled"); // update psiCoupled
    
        //Print convergence information
        KSPGetIterationNumber(ksp_dRdPhiT,&its);
        KSPGetResidualNorm(ksp_dRdPhiT,&finalResNorm);
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %d s PhiFinal\n",
            its,
            finalResNorm,
            this->getRunTime()
        );

        
        // NuTilda
        VecZeroEntries(psiNuTilda0);
        this->setSegregatedVecs(psiCoupled,psiNuTilda0,"nuTilda",1.0,"segregated2Coupled"); // zero out
        MatMult(dRdWT_,psiCoupled,psiCoupledTmp);
        VecCopy(dFdNuTilda0,dFdNuTilda);
        this->setSegregatedVecs(psiCoupledTmp,dFdNuTilda,"nuTilda",-1.0,"coupledAdd2Segregated"); // subtract to rhs

        VecCopy(psiNuTilda,psiNuTilda0);
        KSPSolve(ksp_dRdNuTildaT,dFdNuTilda,psiNuTilda);
        VecAssemblyBegin(psiNuTilda);
        VecAssemblyEnd(psiNuTilda);
        this->underRelaxVec(psiNuTilda,psiNuTilda0,alphaNuTilda);
        this->setSegregatedVecs(psiCoupled,psiNuTilda,"nuTilda",1.0,"segregated2Coupled"); // update psiCoupled
    
        //Print convergence information
        KSPGetIterationNumber(ksp_dRdNuTildaT,&its);
        KSPGetResidualNorm(ksp_dRdNuTildaT,&finalResNorm);
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %d s NuTildaFinal\n",
            its,
            finalResNorm,
            this->getRunTime()
        );

        Info<<endl;
    
    }

    VecZeroEntries(psi_);
    VecCopy(psiCoupled,psi_);
    // write the psi vectors for each objFunc
    word prefix = "psi_"+objFunc;
    adjIO_.writeVectorBinary(psi_,prefix);

    if(adjIO_.writeMatrices) adjIO_.writeVectorASCII(psi_,prefix);

    Info<<"Solving Adjoint... Completed! "<<this->getRunTime()<<" s"<<endl;


    return;
}

void AdjointDerivative::underRelaxVec
(
    Vec vecNew,
    Vec vecOld,
    scalar alpha
)
{
    PetscScalar *vecNewArray;
    VecGetArray(vecNew,&vecNewArray);

    const PetscScalar *vecOldArray;
    VecGetArrayRead(vecOld,&vecOldArray);

    PetscInt Istart, Iend;
    VecGetOwnershipRange(vecNew,&Istart,&Iend);

    for(label i=Istart; i<Iend; i++)
    {
        label localIdx = i-Istart;
        scalar deltaVal = alpha*(vecNewArray[localIdx]-vecOldArray[localIdx]);
        vecNewArray[localIdx] = vecOldArray[localIdx] + deltaVal;
    }
    return;
}


void AdjointDerivative::setSegregatedVecs
(
    Vec psi,
    Vec psiS, 
    const word stateName,
    const scalar scale,
    const word mode
)
{
    label rowCellI, rowComp, rowLocalAdjIdx;
    label localSegregatedRowI;

    PetscInt Istart, Iend;

    VecGetOwnershipRange(psi,&Istart,&Iend);
    PetscScalar *psiArray;
    VecGetArray(psi,&psiArray);

    PetscScalar *psiSArray;
    VecGetArray(psiS,&psiSArray);

    for(label i=Istart; i<Iend; i++)
    {
        rowLocalAdjIdx = i-Istart;
        const word& rowStateName = adjIdx_.adjStateName4LocalAdjIdx[rowLocalAdjIdx];
        if (rowStateName == stateName)
        {
            const scalar& rowCellIFaceI = adjIdx_.cellIFaceI4LocalAdjIdx[rowLocalAdjIdx];
            rowCellI = round( rowCellIFaceI );
            rowComp = round( 10*(rowCellIFaceI-rowCellI) );
            localSegregatedRowI = adjIdx_.getLocalSegregatedAdjointStateIndex(rowStateName,rowCellI,rowComp);

            if(mode == "coupled2Segregated")
            {
                psiSArray[localSegregatedRowI] = scale * psiArray[rowLocalAdjIdx];
            }
            if(mode == "coupledAdd2Segregated")
            {
                scalar tmp = psiSArray[localSegregatedRowI];
                psiSArray[localSegregatedRowI] = tmp + scale * psiArray[rowLocalAdjIdx];
            }
            else if (mode == "segregated2Coupled")
            {
                psiArray[rowLocalAdjIdx] = scale * psiSArray[localSegregatedRowI];
            }
            else
            {
                FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
            }
        }
    }
    VecRestoreArray(psi,&psiArray);
    VecRestoreArray(psiS,&psiSArray);

    if(adjIO_.writeMatrices)
    {
        adjIO_.writeVectorASCII(psi,"psiSet"+stateName);
    }
}


void AdjointDerivative::initializeSegregatedMat
(
    Mat* matIn,
    const word stateName
)
{
    label nCells = adjIdx_.nLocalCells;
    label nFaces = adjIdx_.nLocalFaces;

    MatCreate(PETSC_COMM_WORLD,matIn);
    if(adjIdx_.adjStateType[stateName]=="volVectorState")
    {
        MatSetSizes(*matIn,nCells*3,nCells*3,PETSC_DETERMINE,PETSC_DETERMINE);
    }
    else if (adjIdx_.adjStateType[stateName]=="surfaceScalarState")
    {
        MatSetSizes(*matIn,nFaces,nFaces,PETSC_DETERMINE,PETSC_DETERMINE);
    } 
    else
    {
        MatSetSizes(*matIn,nCells,nCells,PETSC_DETERMINE,PETSC_DETERMINE);
    }
    MatSetFromOptions(*matIn);
    MatMPIAIJSetPreallocation(*matIn,200,NULL,200,NULL);
    MatSeqAIJSetPreallocation(*matIn,200,NULL);
    MatSetOption(*matIn, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*matIn);
}

void AdjointDerivative::extractSegregatedAdjointMat
(
    Mat coupledMat, 
    Mat segregatedMat, 
    const word stateName
)
{
    MatZeroEntries(segregatedMat);

    label rowCellI, rowComp, colCellI, colComp, rowLocalAdjIdx, colLocalAdjIdx;
    label globalRowI,globalColI;

    PetscInt nCols, Istart, Iend;
    const PetscInt    *cols;
    const PetscScalar *vals;

    scalarList cellIFaceIGlobal;
    labelList stateIDGlobal;
    cellIFaceIGlobal.setSize(adjIdx_.nGlobalAdjointStates);
    stateIDGlobal.setSize(adjIdx_.nGlobalAdjointStates);
    adjIdx_.calcAdjStateID4GlobalAdjIdx(stateIDGlobal);
    adjIdx_.calcCellIFaceI4GlobalAdjIdx(cellIFaceIGlobal);

    MatGetOwnershipRange(coupledMat,&Istart,&Iend);
    
    for(label i=Istart; i<Iend; i++)
    {
        // get the global row index for the segregated matrix
        rowLocalAdjIdx = i-Istart;
        const word& rowStateName = adjIdx_.adjStateName4LocalAdjIdx[rowLocalAdjIdx];
        if (rowStateName == stateName)
        {
            const scalar& rowCellIFaceI = adjIdx_.cellIFaceI4LocalAdjIdx[rowLocalAdjIdx];
            rowCellI = round( rowCellIFaceI );
            rowComp = round( 10*(rowCellIFaceI-rowCellI) );
            globalRowI = adjIdx_.getGlobalSegregatedAdjointStateIndex(rowStateName,rowCellI,rowComp);
    
            // get the global col indices, here we can't use local indexing
            MatGetRow(coupledMat,i,&nCols,&cols,&vals);
            for(label j=0;j<nCols;j++)
            {
                const label& adjGlobalColI = cols[j];

                if(adjGlobalColI >= Istart && adjGlobalColI < Iend)
                {
                    colLocalAdjIdx = adjGlobalColI - Istart;
                    const word& colStateName = adjIdx_.adjStateName4LocalAdjIdx[colLocalAdjIdx];
                    if (colStateName == stateName)
                    {
                        const scalar& colCellIFaceI = adjIdx_.cellIFaceI4LocalAdjIdx[colLocalAdjIdx];
                        colCellI = round( colCellIFaceI );
                        colComp = round( 10*(colCellIFaceI-colCellI) );
                        globalColI = adjIdx_.getGlobalSegregatedAdjointStateIndex(stateName,colCellI,colComp);
                        scalar val = vals[j];
                        MatSetValues(segregatedMat,1,&globalRowI,1,&globalColI,&val,INSERT_VALUES);
                    }
                }
            }
            MatRestoreRow(coupledMat,i,&nCols,&cols,&vals);
        }
    }

    MatAssemblyBegin(segregatedMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(segregatedMat,MAT_FINAL_ASSEMBLY);

    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(segregatedMat,"dRd"+stateName+"T");
    }

}

/*
void AdjointDerivative::extractSegregatedAdjointMat
(
    Mat coupledMat, 
    Mat segregatedMat, 
    const word stateName
)
{
    MatZeroEntries(segregatedMat);

    label rowCellI, rowComp, colCellI, colComp, rowLocalAdjIdx;
    label globalRowI,globalColI;

    PetscInt nCols, Istart, Iend;
    const PetscInt    *cols;
    const PetscScalar *vals;

    scalarList cellIFaceIGlobal;
    labelList stateIDGlobal;
    cellIFaceIGlobal.setSize(adjIdx_.nGlobalAdjointStates);
    stateIDGlobal.setSize(adjIdx_.nGlobalAdjointStates);
    adjIdx_.calcAdjStateID4GlobalAdjIdx(stateIDGlobal);
    adjIdx_.calcCellIFaceI4GlobalAdjIdx(cellIFaceIGlobal);

    MatGetOwnershipRange(coupledMat,&Istart,&Iend);
    
    for(label i=Istart; i<Iend; i++)
    {
        // get the global row index for the segregated matrix
        rowLocalAdjIdx = i-Istart;
        const word& rowStateName = adjIdx_.adjStateName4LocalAdjIdx[rowLocalAdjIdx];
        if (rowStateName == stateName)
        {
            const scalar& rowCellIFaceI = adjIdx_.cellIFaceI4LocalAdjIdx[rowLocalAdjIdx];
            rowCellI = round( rowCellIFaceI );
            rowComp = round( 10*(rowCellIFaceI-rowCellI) );
            globalRowI = adjIdx_.getGlobalSegregatedAdjointStateIndex(rowStateName,rowCellI,rowComp);
    
            // get the global col indices, here we can't use local indexing
            MatGetRow(coupledMat,i,&nCols,&cols,&vals);
            for(label j=0;j<nCols;j++)
            {
                const label& adjGlobalColI = cols[j];
                label stateID = stateIDGlobal[adjGlobalColI];
                if (stateID == adjIdx_.adjStateID[stateName])
                {
                    const scalar& colCellIFaceI = cellIFaceIGlobal[adjGlobalColI];
                    colCellI = round( colCellIFaceI );
                    colComp = round( 10*(colCellIFaceI-colCellI) );
                    globalColI = adjIdx_.getGlobalSegregatedAdjointStateIndex(stateName,colCellI,colComp);
                    scalar val = vals[j];
                    MatSetValues(segregatedMat,1,&globalRowI,1,&globalColI,&val,INSERT_VALUES);
                }
            }
            MatRestoreRow(coupledMat,i,&nCols,&cols,&vals);
        }
    }

    MatAssemblyBegin(segregatedMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(segregatedMat,MAT_FINAL_ASSEMBLY);

    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(segregatedMat,"dRd"+stateName+"T");
    }

}
*/

void AdjointDerivative::extractSegregatedAdjointVec
(
    Vec coupledVec, 
    Vec segregatedVec, 
    const word stateName
)
{

    VecZeroEntries(segregatedVec);

    label rowCellI, rowComp, rowLocalAdjIdx;
    label globalRowI;

    PetscInt Istart, Iend;

    VecGetOwnershipRange(coupledVec,&Istart,&Iend);
    const PetscScalar *coupledVecArray;
    VecGetArrayRead(coupledVec,&coupledVecArray);

    for(label i=Istart; i<Iend; i++)
    {
        // get the global row index for the segregated matrix
        rowLocalAdjIdx = i-Istart;
        const word& rowStateName = adjIdx_.adjStateName4LocalAdjIdx[rowLocalAdjIdx];
        if (rowStateName == stateName)
        {
            const scalar& rowCellIFaceI = adjIdx_.cellIFaceI4LocalAdjIdx[rowLocalAdjIdx];
            rowCellI = round( rowCellIFaceI );
            rowComp = round( 10*(rowCellIFaceI-rowCellI) );
            globalRowI = adjIdx_.getGlobalSegregatedAdjointStateIndex(rowStateName,rowCellI,rowComp);

            const scalar& val = coupledVecArray[rowLocalAdjIdx];
            VecSetValue(segregatedVec,globalRowI,val,INSERT_VALUES);
            
        }
    }
    VecRestoreArrayRead(coupledVec,&coupledVecArray);

    VecAssemblyBegin(segregatedVec);
    VecAssemblyEnd(segregatedVec);

    if(adjIO_.writeMatrices)
    {
        adjIO_.writeVectorASCII(segregatedVec,"dFd"+stateName);
    }
}

void AdjointDerivative::createMLRKSP
(
    KSP *genksp,
    Mat jac, 
    Mat jacPC,
    dictionary options
)
{

    PC  MLRMasterPC, MLRGlobalPC;
    PC  MLRsubpc;
    KSP MLRMasterPCKSP;
    KSP * MLRsubksp;
    // ASM Preconditioner variables
    PetscInt  MLRoverlap;        // width of subdomain overlap
    PetscInt  MLRnlocal,MLRfirst;   // number of local subblocks, first local subblock 
    
    // Create linear solver context
    KSPCreate(PETSC_COMM_WORLD,genksp);

    // Set operators. Here the matrix that defines the linear
    // system also serves as the preconditioning matrix.
    KSPSetOperators(*genksp,jac,jacPC);

    // This code sets up the supplied kspObject in the following
    // specific fashion.
    //
    // The hierarchy of the setup is:
    //  kspObject --> Supplied KSP object
    //  |
    //  --> master_PC --> Preconditioner type set to KSP
    //      |
    //      --> master_PC_KSP --> KSP type set to Richardson with 'globalPreConIts'
    //          |
    //           --> globalPC --> PC type set to 'globalPCType'
    //               |            Usually Additive Schwartz and overlap is set
    //               |            with 'ASMOverlap'. Use 0 to get BlockJacobi
    //               |
    //               --> subKSP --> KSP type set to Richardon with 'LocalPreConIts'
    //                   |
    //                   --> subPC -->  PC type set to 'localPCType'.
    //                                  Usually ILU. 'localFillLevel' is
    //                                  set and 'localMatrixOrder' is used.
    //
    // Note that if globalPreConIts=1 then maser_PC_KSP is NOT created and master_PC=globalPC
    // and if localPreConIts=1 then subKSP is set to preOnly.

    // First, KSPSetFromOptions MUST be called
    KSPSetFromOptions(*genksp);

    // Set GMRES
    // Set the type of solver to GMRES
    KSPType kspObjectType=KSPGMRES;

    KSPSetType(*genksp, kspObjectType);
    // Set the gmres restart
    PetscInt restartGMRES = readLabel( options.lookup("GMRESRestart") );
    
    KSPGMRESSetRestart(*genksp,restartGMRES);
    // Set the GMRES refinement type
    KSPGMRESSetCGSRefinementType(*genksp,KSP_GMRES_CGS_REFINE_IFNEEDED);

    // Set the preconditioner side
    KSPSetPCSide(*genksp, PC_RIGHT);

    // Set global and local PC iters
    PetscInt globalPreConIts = readLabel( options.lookup("GlobalPCIters") );
    
    // Since there is an extraneous matMult required when using the
    // richardson precondtiter with only 1 iteration, only use it when we need
    // to do more than 1 iteration.
    if (globalPreConIts > 1)
    {
        // Extract preconditioning context for main KSP solver: (MLRMasterPC)
        KSPGetPC(*genksp, &MLRMasterPC);

        // Set the type of MLRMasterPC to ksp. This lets us do multiple
        // iterations of preconditioner application
        PCSetType(MLRMasterPC, PCKSP);

        // Get the ksp context from MLRMasterPC which is the actual preconditioner:
        PCKSPGetKSP(MLRMasterPC, &MLRMasterPCKSP);

        // MLRMasterPCKSP type will always be of type richardson. If the
        // number  of iterations is set to 1, this ksp object is transparent.
        KSPSetType(MLRMasterPCKSP, KSPRICHARDSON);

        // Important to set the norm-type to None for efficiency.
        KSPSetNormType(MLRMasterPCKSP, KSP_NORM_NONE);

        // Do one iteration of the outer ksp preconditioners. Note the
        // tolerances are unsued since we have set KSP_NORM_NONE
        KSPSetTolerances(MLRMasterPCKSP, PETSC_DEFAULT, PETSC_DEFAULT,
                         PETSC_DEFAULT, globalPreConIts);

        // Get the 'preconditioner for MLRMasterPCKSP, called 'MLRGlobalPC'. This
        // preconditioner is potentially run multiple times.
        KSPGetPC(MLRMasterPCKSP, &MLRGlobalPC);
    }
    else
    {
        // Just pull out the pc-object if we are not using kspRichardson
        KSPGetPC(*genksp, &MLRGlobalPC);
    }

    // Set the type of 'MLRGlobalPC'. This will almost always be additive schwartz
    PCSetType(MLRGlobalPC, PCASM );

    // Set the overlap required
    MLRoverlap = readLabel( options.lookup("ASMOverlap") );
    PCASMSetOverlap(MLRGlobalPC, MLRoverlap);

    if(adjIO_.adjGMRESCalcEigen) KSPSetComputeEigenvalues(*genksp,PETSC_TRUE);

    //Setup the main ksp context before extracting the subdomains
    KSPSetUp(*genksp);

    // Extract the ksp objects for each subdomain
    PCASMGetSubKSP(MLRGlobalPC, &MLRnlocal, &MLRfirst, &MLRsubksp);

    //Loop over the local blocks, setting various KSP options
    //for each block.
    PetscInt localPreConIts = readLabel( options.lookup("LocalPCIters") ); 
    word matOrdering= word(options.lookup("JacMatReOrdering"));
    PetscInt localFillLevel=readLabel( options.lookup("PCFillLevel") );
    for (PetscInt i=0; i<MLRnlocal; i++)
    {
        // Since there is an extraneous matMult required when using the
        // richardson precondtiter with only 1 iteration, only use it we need
        // to do more than 1 iteration.
        if (localPreConIts > 1)
        {
            // This 'subksp' object will ALSO be of type richardson so we can do
            // multiple iterations on the sub-domains
            KSPSetType(MLRsubksp[i], KSPRICHARDSON);

            // Set the number of iterations to do on local blocks. Tolerances are ignored.
            KSPSetTolerances(MLRsubksp[i], PETSC_DEFAULT, PETSC_DEFAULT,
                             PETSC_DEFAULT, localPreConIts);

            // Again, norm_type is NONE since we don't want to check error
            KSPSetNormType(MLRsubksp[i], KSP_NORM_NONE);
        }
        else
        {
            KSPSetType(MLRsubksp[i], KSPPREONLY);
        }

        // Extract the preconditioner for subksp object.
        KSPGetPC(MLRsubksp[i], &MLRsubpc);

        // The subpc type will almost always be ILU
        PCType localPCType = PCILU;
        PCSetType(MLRsubpc, localPCType);

        // Set PC factor
        PCFactorSetPivotInBlocks(MLRsubpc, PETSC_TRUE);
        PCFactorSetShiftType(MLRsubpc, MAT_SHIFT_NONZERO);
        PCFactorSetShiftAmount(MLRsubpc, PETSC_DECIDE);

        // Setup the matrix ordering for the subpc object:
        // 'natural':'natural',
        // 'rcm':'rcm',
        // 'nested dissection':'nd' (default),
        // 'one way dissection':'1wd',
        // 'quotient minimum degree':'qmd',
        MatOrderingType localMatrixOrdering;
        if(matOrdering=="natural")
        {
            localMatrixOrdering = MATORDERINGNATURAL;
        }
        else if(matOrdering=="nd")
        {
            localMatrixOrdering = MATORDERINGND;
        }
        else if(matOrdering=="rcm")
        {
            localMatrixOrdering = MATORDERINGRCM;
        }
        else if(matOrdering=="1wd")
        {
            localMatrixOrdering = MATORDERING1WD;
        }
        else if(matOrdering=="qmd")
        {
            localMatrixOrdering = MATORDERINGQMD;
        }
        else
        {
            Info<<"matOrdering not known. Using default: nested dissection"<<endl;
            localMatrixOrdering = MATORDERINGND;
        }
        PCFactorSetMatOrderingType(MLRsubpc, localMatrixOrdering);

        // Set the ILU parameters
        PCFactorSetLevels(MLRsubpc, localFillLevel);
    }

    // Set the norm to unpreconditioned
    KSPSetNormType(*genksp,KSP_NORM_UNPRECONDITIONED);
    // Setup monitor if necessary:
    if(readLabel( options.lookup("printInfo") )) KSPMonitorSet(*genksp,myKSPMonitor,this,0);

    PetscInt maxIts =  readLabel( options.lookup("GMRESMaxIters") );
    PetscScalar rtol,atol;
    rtol =readScalar( options.lookup("GMRESRelTol") );
    atol =readScalar( options.lookup("GMRESAbsTol") );
    KSPSetTolerances(*genksp,rtol,atol,PETSC_DEFAULT,maxIts);

    if(readLabel( options.lookup("printInfo") ))
    {
        Info<<"Solver Type: "<<kspObjectType<<endl;
        Info<<"GMRES Restart: "<<restartGMRES<<endl;
        Info<<"ASM Overlap: "<<MLRoverlap<<endl;
        Info<<"Global PC Iters: "<<globalPreConIts<<endl;
        Info<<"Local PC Iters: "<<localPreConIts<<endl;
        Info<<"Mat ReOrdering: "<<matOrdering<<endl;
        Info<<"ILU PC Fill Level: "<<localFillLevel<<endl;
        Info<<"GMRES Max Iterations: "<<maxIts<<endl;
        Info<<"GMRES Relative Tolerance: "<<rtol<<endl;
        Info<<"GMRES Absolute Tolerance: "<<atol<<endl;
    }

}

PetscErrorCode AdjointDerivative::myKSPMonitor(KSP ksp,PetscInt n,PetscReal rnorm,void *ctx)
{

    /*
      Write the solution vector and residual norm to stdout.
      - PetscPrintf() handles output for multiprocessor jobs
      by printing from only one processor in the communicator.
      - The parallel viewer PETSC_VIEWER_STDOUT_WORLD handles
      data from multiple processors so that the output
      is not jumbled.
    */

    AdjointDerivative *adjDev= (AdjointDerivative*) ctx;

    PetscInt printFrequency = 10; // residual print frequency
    if (n%printFrequency==0)
    {
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %d s\n",
            n,
            rnorm,
            adjDev->getRunTime()
        );
    }
    return 0;
}

label AdjointDerivative::getRunTime()
{
    return mesh_.time().elapsedClockTime();
}

void AdjointDerivative::calcFlowResidualStatistics
(
    const word mode,
    const label writeRes
)
{
    if(mode=="print")
    {
        // print the flow residuals to screen
        Info<<"Printing Flow Residual Statistics."<<endl;
    }
    else if(mode=="set")
    {
        // print the flow residuals to screen
        Info<<"Setting Flow Residual Statistics."<<endl;
    }
    else if(mode=="verify")
    {
        // print the flow residuals to screen
        Info<<"Verifying Flow Residual Statistics."<<endl;
    }
    else
    {
        FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    }

    scalar verifyRelTol=adjIO_.stateResetTol;

    // calculate the residuals
    label isRef=0,isPC=0;
    this->copyStates("Ref2Var");
    this->calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);

    forAll(adjReg_.volVectorStates,idxI)
    {
        const word stateName = adjReg_.volVectorStates[idxI];
        const word resName = stateName+"Res";                   
        const volVectorField& stateRes = db_.lookupObject<volVectorField>(resName); 

        if (stateRes.size()==0)
        {
            Pout<<"Warning!!!!! Number of cells equals zero"<<endl;
        }
        
        vector vecResMax(0,0,0);
        vector vecResNorm2(0,0,0);
        vector vecResMean(0,0,0);
        forAll(stateRes,cellI)
        {
            vecResNorm2.x()+=Foam::pow(stateRes[cellI].x(),2.0);
            vecResNorm2.y()+=Foam::pow(stateRes[cellI].y(),2.0);
            vecResNorm2.z()+=Foam::pow(stateRes[cellI].z(),2.0);
            vecResMean.x()+=fabs(stateRes[cellI].x());
            vecResMean.y()+=fabs(stateRes[cellI].y());
            vecResMean.z()+=fabs(stateRes[cellI].z());
            if(fabs(stateRes[cellI].x()) > vecResMax.x()) vecResMax.x()=fabs(stateRes[cellI].x());
            if(fabs(stateRes[cellI].y()) > vecResMax.y()) vecResMax.y()=fabs(stateRes[cellI].y());
            if(fabs(stateRes[cellI].z()) > vecResMax.z()) vecResMax.z()=fabs(stateRes[cellI].z());
        }
        vecResMean=vecResMean/(stateRes.size()+1.0e-8);
        reduce(vecResMean,sumOp<vector>());
        vecResMean=vecResMean/Pstream::nProcs();
        reduce(vecResNorm2,sumOp<vector>());
        reduce(vecResMax, maxOp<vector>());
        vecResNorm2.x()=Foam::pow(vecResNorm2.x(),0.5);
        vecResNorm2.y()=Foam::pow(vecResNorm2.y(),0.5);
        vecResNorm2.z()=Foam::pow(vecResNorm2.z(),0.5);
        if(mode=="print")
        {
            Info<<stateName<<" Residual Norm2: "<<vecResNorm2<<endl;
            Info<<stateName<<" Residual Mean: "<<vecResMean<<endl;
            Info<<stateName<<" Residual Max: "<<vecResMax<<endl;
        }
        else if(mode=="set")
        {
            refResL2Norm_.set(resName,mag(vecResNorm2));
        }
        else if (mode=="verify")
        {
            scalar relError = fabs(refResL2Norm_[resName]-mag(vecResNorm2)) / fabs(refResL2Norm_[resName]+1e-16);
            if( relError>verifyRelTol )
            {
                FatalErrorIn("")<<resName<<" "<<vecResNorm2
                                <<" L2 norm mag "<<mag(vecResNorm2)
                                <<" not equal to ref mag"<<refResL2Norm_[resName]
                                <<abort(FatalError);
            }
        }
        
        if (writeRes) stateRes.write();

    }
    
    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI];
        const word resName = stateName+"Res";                   
        const volScalarField& stateRes = db_.lookupObject<volScalarField>(resName); 

        if (stateRes.size()==0)
        {
            Pout<<"Warning!!!!! Number of cells equals zero"<<endl;
        }
        
        scalar scalarResMax=0, scalarResNorm2=0, scalarResMean=0;
        forAll(stateRes,cellI)
        {
            scalarResNorm2+=Foam::pow(stateRes[cellI],2.0);
            scalarResMean+=fabs(stateRes[cellI]);
            if(fabs(stateRes[cellI]) > scalarResMax) scalarResMax=fabs(stateRes[cellI]);
        }
        scalarResMean=scalarResMean/(stateRes.size()+1.0e-8);
        reduce(scalarResMean,sumOp<scalar>());
        scalarResMean=scalarResMean/Pstream::nProcs();
        reduce(scalarResNorm2,sumOp<scalar>());
        reduce(scalarResMax, maxOp<scalar>());
        scalarResNorm2=Foam::pow(scalarResNorm2,0.5);
        if(mode=="print")
        {
            Info<<stateName<<" Residual Norm2: "<<scalarResNorm2<<endl;
            Info<<stateName<<" Residual Mean: "<<scalarResMean<<endl;
            Info<<stateName<<" Residual Max: "<<scalarResMax<<endl;
        }
        else if(mode=="set")
        {
            refResL2Norm_.set(resName,scalarResNorm2);
        }
        else if (mode=="verify")
        {
            scalar relError = fabs(refResL2Norm_[resName]-scalarResNorm2) / fabs(refResL2Norm_[resName]+1e-16);
            if( relError>verifyRelTol )
            {
                FatalErrorIn("")<<resName<<" L2 norm "<<scalarResNorm2
                                <<" not equal to ref "<<refResL2Norm_[resName]
                                <<abort(FatalError);
            }
        }
        
        if (writeRes) stateRes.write();
    }
    
    forAll(adjRAS_.turbStates,idxI)
    {
        const word stateName = adjRAS_.turbStates[idxI];
        const word resName = stateName+"Res";  
        const volScalarField& stateRes = db_.lookupObject<volScalarField>(resName); 

        if (stateRes.size()==0)
        {
            Pout<<"Warning!!!!! Number of cells equals zero"<<endl;
        }
        
        scalar scalarResMax=0, scalarResNorm2=0, scalarResMean=0;
        forAll(stateRes,cellI)
        {
            scalarResNorm2+=Foam::pow(stateRes[cellI],2.0);
            scalarResMean+=fabs(stateRes[cellI]);
            if(fabs(stateRes[cellI]) > scalarResMax) scalarResMax=fabs(stateRes[cellI]);
        }
        scalarResMean=scalarResMean/(stateRes.size()+1.0e-8);
        reduce(scalarResMean,sumOp<scalar>());
        scalarResMean=scalarResMean/Pstream::nProcs();
        reduce(scalarResNorm2,sumOp<scalar>());
        reduce(scalarResMax, maxOp<scalar>());
        scalarResNorm2=Foam::pow(scalarResNorm2,0.5);
        if(mode=="print")
        {
            Info<<stateName<<" Residual Norm2: "<<scalarResNorm2<<endl;
            Info<<stateName<<" Residual Mean: "<<scalarResMean<<endl;
            Info<<stateName<<" Residual Max: "<<scalarResMax<<endl;
        }
        else if(mode=="set")
        {
            refResL2Norm_.set(resName,scalarResNorm2);
        }
        else if (mode=="verify")
        {
            scalar relError = fabs(refResL2Norm_[resName]-scalarResNorm2) / fabs(refResL2Norm_[resName]+1e-16);
            if( relError>verifyRelTol )
            {
                FatalErrorIn("")<<resName<<" L2 norm "<<scalarResNorm2
                                <<" not equal to ref "<<refResL2Norm_[resName]
                                <<abort(FatalError);
            }
        }
        
        if (writeRes) stateRes.write();
    }
    
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        const word stateName = adjReg_.surfaceScalarStates[idxI];
        const word resName = stateName+"Res";  
        const surfaceScalarField& stateRes = db_.lookupObject<surfaceScalarField>(resName); 

        if (stateRes.size()==0)
        {
            Pout<<"Warning!!!!! Number of cells equals zero"<<endl;
        }
        
        scalar phiResMax=0, phiResNorm2=0, phiResMean=0;
        forAll(stateRes,faceI)
        {
            phiResNorm2+=Foam::pow(stateRes[faceI],2.0);
            phiResMean+=fabs(stateRes[faceI]);
            if(fabs(stateRes[faceI]) > phiResMax) phiResMax=fabs(stateRes[faceI]);
    
        }
        forAll(stateRes.boundaryField(),patchI)
        {
            forAll(stateRes.boundaryField()[patchI],faceI)
            {
                scalar bPhiRes = stateRes.boundaryField()[patchI][faceI];
                phiResNorm2+=Foam::pow(bPhiRes,2.0);
                phiResMean+=fabs(bPhiRes);
                if(fabs(bPhiRes) > phiResMax) phiResMax=fabs(bPhiRes);
            }
        }
        phiResMean=phiResMean/(mesh_.nFaces()+1.0e-8);
        reduce(phiResMean,sumOp<scalar>());
        phiResMean=phiResMean/Pstream::nProcs();
        reduce(phiResNorm2,sumOp<scalar>());
        reduce(phiResMax, maxOp<scalar>());
        phiResNorm2=Foam::pow(phiResNorm2,0.5);
        if(mode=="print")
        {
            Info<<stateName<<" Residual Norm2: "<<phiResNorm2<<endl;
            Info<<stateName<<" Residual Mean: "<<phiResMean<<endl;
            Info<<stateName<<" Residual Max: "<<phiResMax<<endl;
        }
        else if(mode=="set")
        {
            refResL2Norm_.set(resName,phiResNorm2);
        }
        else if (mode=="verify")
        {
            scalar relError = fabs(refResL2Norm_[resName]-phiResNorm2) / fabs(refResL2Norm_[resName]+1e-16);
            if( relError>verifyRelTol )
            {
                FatalErrorIn("")<<resName<<" L2 norm "<<phiResNorm2
                                <<" not equal to ref "<<refResL2Norm_[resName]
                                <<abort(FatalError);
            }
        }

        if (writeRes) stateRes.write();
    }


    return;
}


void AdjointDerivative::printObjFuncValues()
{
    adjObj_.printObjFuncValues();
    return;
}

void AdjointDerivative::clearVars()
{

    forAll(adjReg_.volVectorStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeRes(volVectorStates,volVectorField,adjReg_);
        makeResRef(volVectorStates,volVectorField,adjReg_);
        makeResPartDeriv(volVectorStates,volVectorField,adjReg_);
        stateRes.clear();
        stateResRef.clear();
        stateResPartDeriv.clear();
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeRes(volScalarStates,volScalarField,adjReg_);
        makeResRef(volScalarStates,volScalarField,adjReg_);
        makeResPartDeriv(volScalarStates,volScalarField,adjReg_);
        stateRes.clear();
        stateResRef.clear();
        stateResPartDeriv.clear();
    }

    forAll(adjRAS_.turbStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeRes(turbStates,volScalarField,adjRAS_);
        makeResRef(turbStates,volScalarField,adjRAS_);
        makeResPartDeriv(turbStates,volScalarField,adjRAS_);
        stateRes.clear();
        stateResRef.clear();
        stateResPartDeriv.clear();
    }

    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        // create res, resRef, and resPartDeriv
        makeRes(surfaceScalarStates,surfaceScalarField,adjReg_);
        makeResRef(surfaceScalarStates,surfaceScalarField,adjReg_);
        makeResPartDeriv(surfaceScalarStates,surfaceScalarField,adjReg_);
        stateRes.clear();
        stateResRef.clear();
        stateResPartDeriv.clear();
    }

}

void AdjointDerivative::writeStates()
{
    Info<<"Writting state variables"<<endl;
    
    forAll(adjReg_.volVectorStates,idxI)                                           
    {        
        // create state and stateRef
        makeState(volVectorStates,volVectorField,adjReg_); 
        state.write();                                                                    
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        // create state and stateRef
        makeState(volScalarStates,volScalarField,adjReg_);
        state.write();   
    }

    forAll(adjRAS_.turbStates,idxI)
    {
        // create state and stateRef
        makeState(turbStates,volScalarField,adjRAS_);
        state.write();   

    }

    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        // create state and stateRef
        makeState(surfaceScalarStates,surfaceScalarField,adjReg_);
        state.write();   
    }
}



// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
