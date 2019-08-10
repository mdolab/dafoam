/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointJacobianConnectivity.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointJacobianConnectivity, 0);
defineRunTimeSelectionTable(AdjointJacobianConnectivity, dictionary);

// Constructors
AdjointJacobianConnectivity::AdjointJacobianConnectivity
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx
)
    :
    mesh_(mesh),
    adjIO_(adjIO),
    adjReg_(adjReg),
    adjRAS_(adjRAS),
    adjIdx_(adjIdx)


{
    if(adjIO_.useColoring)
    {
            
        // Calculate the boundary connectivity
        Info <<"Generating Connectivity for Boundaries:"<<endl;
        
        this->calcNeiBFaceGlobalCompact();

        this->calcCyclicAMIBFaceGlobalCompact();
    
        this->setupStateBoundaryCon();

        this->setupStateCyclicAMICon();

        this->combineAllStateCons();

        this->setupStateCyclicAMIConID();

        this->setupStateBoundaryConID();

        if( adjIO_.isInList<word>("Xv", adjIO_.adjDVTypes) ) this->setupXvBoundaryCon();
        
        if(adjIO_.writeMatrices)
        {
            adjIO_.writeMatrixBinary(stateBoundaryCon_,"stateBoundaryCon");
            adjIO_.writeMatrixBinary(stateBoundaryConID_,"stateBoundaryConID");

            PetscViewer    viewer;
            PetscViewerBinaryOpen(PETSC_COMM_SELF,"stateCyclicAMICon.bin",FILE_MODE_WRITE,&viewer);
            MatView(stateCyclicAMICon_,viewer);
            PetscViewerDestroy(&viewer);
            
            PetscViewerBinaryOpen(PETSC_COMM_SELF,"stateCyclicAMIConID.bin",FILE_MODE_WRITE,&viewer);
            MatView(stateCyclicAMIConID_,viewer);
            PetscViewerDestroy(&viewer);

        }

        this->initializePetscVecs();
        
    }
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<AdjointJacobianConnectivity> AdjointJacobianConnectivity::New
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx
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

    Info<< "Selecting " << modelType<<" for AdjointJacobianConnectivity" << endl;

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn
        (
            "AdjointJacobianConnectivity::New"
        )   << "Unknown AdjointJacobianConnectivity type "
            << modelType << nl << nl
            << "Valid AdjointJacobianConnectivity types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<AdjointJacobianConnectivity>
           (
               cstrIter()(mesh,adjIO,adjReg,adjRAS,adjIdx)
           );
}




// * * * * * * * * * * * * * * * * * Member functions * * * * * * * * * * * * * * * //


void AdjointJacobianConnectivity::setupdRdWCon(label isPrealloc,label isPC)
{
    // Calculates connectivity for the state Jacobian connectivity mat
    // if isPrealloc = 1, calculate the preallocation vectors, else, calculate dRdWCon
    // if isPC=1, calc dRdWConPC_, else calculate dRdWCon_
    // dRdWPreallocOn: preallocation vector that stores the number of on-diagonal conectivity for each row 
    // dRdWCon: state Jacobian connectivity mat with dimension sizeAdjStates by sizeAdjStates
    // connectedStatesP: one row matrix that stores the actual connectivity, element value 1 denotes 
    // a connected state. connectedStatesP is then used to assign dRdWPreallocOn or dRdWCon

    Mat* conMat;
    if(isPC) conMat=&dRdWConPC_;
    else conMat=&dRdWCon_;

    if(isPC) this->reduceAdjStateResidualConLevel();

    label globalIdx;
    Mat connectedStatesP;

    PetscInt    nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscInt    nColsID;
    const PetscInt    *colsID;
    const PetscScalar *valsID;

    if(isPrealloc)
    {
        VecZeroEntries(dRdWPreallocOn_);
        VecZeroEntries(dRdWPreallocOff_);
        VecZeroEntries(dRdWTPreallocOn_);
        VecZeroEntries(dRdWTPreallocOff_);
    }

    if(isPrealloc) Info<<"Preallocating state Jacobian connectivity mat"<<endl;
    else Info<<"Setup state Jacobian connectivity mat"<<endl;
    
    // loop over all cell residuals
    forAll(adjIdx_.adjStateNames,idxI) 
    {
        // get stateName and residual names
        word stateName = adjIdx_.adjStateNames[idxI];
        word resName = stateName+"Res";

        // check if this state is a cell state, we do surfaceScalarState residuals separately
        if (adjIdx_.adjStateType[stateName] == "surfaceScalarState") continue; 

        // maximal connectivity level information
        label maxConLevel = adjStateResidualConInfo_[resName].size()-1;
        
        // if it is a vectorState, set compMax=3
        label compMax = 1;
        if (adjIdx_.adjStateType[stateName] == "volVectorState") compMax=3;
        
        forAll(mesh_.cells(), cellI)
        {
            for(label comp=0; comp<compMax; comp++)
            {
            
                //zero the connections
                this->createConnectionMat(&connectedStatesP);
        
                // now add the con. We loop over all the connectivity levels
                forAll(adjStateResidualConInfo_[resName],idxJ) // idxJ: con level
                {
                
                    // set connectedStatesLocal: the locally connected state variables for this level
                    wordList connectedStatesLocal(0);
                    forAll(adjStateResidualConInfo_[resName][idxJ],idxK) 
                    {
                        word conName = adjStateResidualConInfo_[resName][idxJ][idxK];
                        // Exclude surfaceScalarState when appending connectedStatesLocal
                        // whether to add it depends on addFace parameter
                        if (adjIdx_.adjStateType[conName] != "surfaceScalarState") 
                            connectedStatesLocal.append(conName);
                    }
                    
                    // set connectedStatesInterProc: the globally connected state variables for this level
                    List< List<word> > connectedStatesInterProc;
                    if(idxJ==0)
                    {
                        // pass a zero list, no need to add interProc connecitivity for level 0
                        connectedStatesInterProc.setSize(0); 
                    }
                    else if (idxJ != maxConLevel)
                    {
                        connectedStatesInterProc.setSize(maxConLevel-idxJ+1);
                        for(label k=0;k<maxConLevel-idxJ+1;k++)
                        {
                            label conSize = adjStateResidualConInfo_[resName][k+idxJ].size();
                            for(label l=0;l<conSize;l++)
                            {
                                word conName = adjStateResidualConInfo_[resName][k+idxJ][l];
                                // Exclude surfaceScalarState when appending connectedStatesLocal
                                // whether to add it depends on addFace parameter
                                if (adjIdx_.adjStateType[conName] != "surfaceScalarState") 
                                    connectedStatesInterProc[k].append(conName);
                            }
                        }
                        
                    }
                    else
                    {
                        connectedStatesInterProc.setSize(1);
                        label conSize = adjStateResidualConInfo_[resName][maxConLevel].size();
                        for(label l=0;l<conSize;l++)
                        {
                            word conName = adjStateResidualConInfo_[resName][maxConLevel][l];
                            // Exclude surfaceScalarState when appending connectedStatesLocal
                            // whether to add it depends on addFace parameter
                            if (adjIdx_.adjStateType[conName] != "surfaceScalarState") 
                                connectedStatesInterProc[0].append(conName);
                        }
                    }
                    
                    // check if we need to addFace for this level
                    label addFace=0;
                    forAll(adjReg_.surfaceScalarStates,idxK)
                    {
                        const word& conName = adjReg_.surfaceScalarStates[idxK];
                        if ( adjIO_.isInList<word>(conName,adjStateResidualConInfo_[resName][idxJ]) ) 
                            addFace=1;
                    }

                    // Add connectivity
                    this->addStateConnections
                    (
                        connectedStatesP,
                        cellI,
                        idxJ,
                        connectedStatesLocal,
                        connectedStatesInterProc,
                        addFace
                    );
                    
                    //Info<<"lv: "<<idxJ<<" locaStates: "<<connectedStatesLocal<<" interProcStates: "<<connectedStatesInterProc<<" addFace: "<<addFace<<endl;
              
                }   
                
                // get the global index of the current state for the row index
                globalIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,comp);
                
                if(isPrealloc)
                {
                    this->allocateJacobianConnections(dRdWPreallocOn_,dRdWPreallocOff_,
                                                      dRdWTPreallocOn_,dRdWTPreallocOff_,
                                                      connectedStatesP,globalIdx);
                }
                else
                {
                    this->setupJacobianConnections(*conMat,connectedStatesP,globalIdx);
                }

            }
            
        }

    }
    
    
    // loop over all face residuals
    forAll(adjReg_.surfaceScalarStates,idxI) 
    {
        // get stateName and residual names
        word stateName = adjReg_.surfaceScalarStates[idxI];
        word resName = stateName+"Res";

        // maximal connectivity level information
        label maxConLevel = adjStateResidualConInfo_[resName].size()-1;
                
        forAll(mesh_.faces(), faceI)
        {
            
            //zero the connections
            this->createConnectionMat(&connectedStatesP);
            
            // Get the owner and neighbour cells for this face
            label idxO=-1,idxN=-1;
            if (faceI<adjIdx_.nLocalInternalFaces)
            {
                idxO = mesh_.owner()[faceI];
                idxN = mesh_.neighbour()[faceI];
            }
            else
            {
                label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                label patchIdx=adjIdx_.bFacePatchI[relIdx];
                label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                
                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                idxN = pFaceCells[faceIdx];
            }
            
            
            // now add the con. We loop over all the connectivity levels
            forAll(adjStateResidualConInfo_[resName],idxJ) // idxJ: con level
            {
            
                // set connectedStatesLocal: the locally connected state variables for this level
                wordList connectedStatesLocal(0);
                forAll(adjStateResidualConInfo_[resName][idxJ],idxK) 
                {
                    word conName = adjStateResidualConInfo_[resName][idxJ][idxK];
                    // Exclude surfaceScalarState when appending connectedStatesLocal
                    // whether to add it depends on addFace parameter
                    if (adjIdx_.adjStateType[conName] != "surfaceScalarState") 
                        connectedStatesLocal.append(conName);
                }
                
                // set connectedStatesInterProc: the globally connected state variables for this level
                List< List<word> > connectedStatesInterProc;
                if(idxJ==0)
                {
                    // pass a zero list, no need to add interProc connecitivity for level 0
                    connectedStatesInterProc.setSize(0); 
                }
                else if (idxJ != maxConLevel)
                {
                    connectedStatesInterProc.setSize(maxConLevel-idxJ+1);
                    for(label k=0;k<maxConLevel-idxJ+1;k++)
                    {
                        label conSize = adjStateResidualConInfo_[resName][k+idxJ].size();
                        for(label l=0;l<conSize;l++)
                        {
                            word conName = adjStateResidualConInfo_[resName][k+idxJ][l];
                            // Exclude surfaceScalarState when appending connectedStatesLocal
                            // whether to add it depends on addFace parameter
                            if (adjIdx_.adjStateType[conName] != "surfaceScalarState") 
                                connectedStatesInterProc[k].append(conName);
                        }
                    }
                    
                }
                else
                {
                    connectedStatesInterProc.setSize(1);
                    label conSize = adjStateResidualConInfo_[resName][maxConLevel].size();
                    for(label l=0;l<conSize;l++)
                    {
                        word conName = adjStateResidualConInfo_[resName][maxConLevel][l];
                        // Exclude surfaceScalarState when appending connectedStatesLocal
                        // whether to add it depends on addFace parameter
                        if (adjIdx_.adjStateType[conName] != "surfaceScalarState") 
                            connectedStatesInterProc[0].append(conName);
                    }
                }
                
                // check if we need to addFace for this level
                label addFace=0;
                forAll(adjReg_.surfaceScalarStates,idxK)
                {
                    const word& conName = adjReg_.surfaceScalarStates[idxK];
                    // NOTE: we need special treatment for boundary faces for level>0
                    // since addFace for boundary face should add one more extra level of faces
                    // This is because we only have idxN for a boundary face while the idxO can
                    // be on the other side of the inter-proc boundary
                    // In this case, we need to use idxJ-1 instead of idxJ information to tell whether to addFace
                    label levelCheck;
                    if(faceI<adjIdx_.nLocalInternalFaces or idxJ==0) levelCheck = idxJ;
                    else levelCheck = idxJ-1;
                    
                    if ( adjIO_.isInList<word>(conName,adjStateResidualConInfo_[resName][levelCheck]) ) 
                        addFace=1;

                }

                // Add connectivity for idxN
                this->addStateConnections
                (
                    connectedStatesP,
                    idxN,
                    idxJ,
                    connectedStatesLocal,
                    connectedStatesInterProc,
                    addFace
                );
                
                if(faceI<adjIdx_.nLocalInternalFaces)
                {
                    // Add connectivity for idxO
                    this->addStateConnections
                    (
                        connectedStatesP,
                        idxO,
                        idxJ,
                        connectedStatesLocal,
                        connectedStatesInterProc,
                        addFace
                    );
                }
                
                //Info<<"lv: "<<idxJ<<" locaStates: "<<connectedStatesLocal<<" interProcStates: "<<connectedStatesInterProc<<" addFace: "<<addFace<<endl;
                
            } 
            
            // NOTE: if this faceI is on a coupled patch, the above connectivity is not enough to
            // cover the points on the other side of proc domain, we need to add 3 lvs of cells here
            if(faceI>=adjIdx_.nLocalInternalFaces)
            {
                label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                label patchIdx=adjIdx_.bFacePatchI[relIdx];
                
                label maxLevel=adjStateResidualConInfo_[resName].size();

                if ( mesh_.boundaryMesh()[patchIdx].coupled() )
                {
                    
                    label bRow=this->getLocalCoupledBFaceIndex(faceI);
                    label bRowGlobal=adjIdx_.globalCoupledBFaceNumbering.toGlobal(bRow);
                    MatGetRow(stateBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);
                    MatGetRow(stateBoundaryConID_,bRowGlobal,&nColsID,&colsID,&valsID);
                    for(label i=0; i<nCols; i++)
                    {
                        PetscInt idxJ = cols[i];
                        label val = round(vals[i]);
                        // we are going to add some selective states with connectivity level <= 3
                        // first check the state
                        label stateID = round(valsID[i]);
                        word conName = adjIdx_.adjStateNames[stateID];
                        label addState = 0;
                        // NOTE: we use val-1 here since phi actually has 3 levels of connectivity
                        // however, when we assign adjStateResidualConInfo_, we ignore the level 0
                        // connectivity since they are idxN and idxO
                        if ( val!=10 && val<maxLevel+1)
                        {
                            if(adjIO_.isInList<word>(conName,adjStateResidualConInfo_[resName][val-1]))
                                addState =1;
                        }
                        if(addState==1 && val<maxLevel+1 && val>0 ) 
                            this->setConnections(connectedStatesP,idxJ);
                    }
                    MatRestoreRow(stateBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);
                    MatRestoreRow(stateBoundaryConID_,bRowGlobal,&nColsID,&colsID,&valsID);
                }  
                
                if ( mesh_.boundaryMesh()[patchIdx].type() == "cyclicAMI" )
                {
                    
                    label bRow=this->getLocalCyclicAMIFaceIndex(faceI);
                    MatGetRow(stateCyclicAMICon_,bRow,&nCols,&cols,&vals);
                    MatGetRow(stateCyclicAMIConID_,bRow,&nColsID,&colsID,&valsID);
                    for(label i=0; i<nCols; i++)
                    {
                        PetscInt idxJ = cols[i];
                        label val = round(vals[i]);
                        // we are going to add some selective states with connectivity level <= 3
                        // first check the state
                        label stateID = round(valsID[i]);
                        word conName = adjIdx_.adjStateNames[stateID];
                        label addState = 0;
                        // NOTE: we use val-1 here since phi actually has 3 levels of connectivity
                        // however, when we assign adjStateResidualConInfo_, we ignore the level 0
                        // connectivity since they are idxN and idxO
                        if ( val!=10 && val<maxLevel+1)
                        {
                            if(adjIO_.isInList<word>(conName,adjStateResidualConInfo_[resName][val-1]))
                                addState =1;
                        }
                        if(addState==1 && val<maxLevel+1 && val>0 ) 
                            this->setConnections(connectedStatesP,idxJ);
                    }
                    MatRestoreRow(stateCyclicAMICon_,bRow,&nCols,&cols,&vals);
                    MatRestoreRow(stateCyclicAMIConID_,bRow,&nColsID,&colsID,&valsID);
                }  
            }
           
            // get the global index of the current state for the row index
            globalIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,faceI);
            
            if(isPrealloc)
            {
                this->allocateJacobianConnections(dRdWPreallocOn_,dRdWPreallocOff_,
                                                  dRdWTPreallocOn_,dRdWTPreallocOff_,
                                                  connectedStatesP,globalIdx);
            }
            else
            {
                this->setupJacobianConnections(*conMat,connectedStatesP,globalIdx);
            }
            
        }

    }
    
    if(isPC) this->restoreAdjStateResidualConLevel();
    
    if(isPrealloc)
    {
        VecAssemblyBegin(dRdWPreallocOn_);
        VecAssemblyEnd(dRdWPreallocOn_);
        VecAssemblyBegin(dRdWPreallocOff_);
        VecAssemblyEnd(dRdWPreallocOff_);
        VecAssemblyBegin(dRdWTPreallocOn_);
        VecAssemblyEnd(dRdWTPreallocOn_);
        VecAssemblyBegin(dRdWTPreallocOff_);
        VecAssemblyEnd(dRdWTPreallocOff_);
        
        //output the matrix to a file
        if(adjIO_.writeMatrices)
        {
            adjIO_.writeVectorASCII(dRdWTPreallocOn_,"dRdWTPreallocOn");
            adjIO_.writeVectorASCII(dRdWTPreallocOff_,"dRdWTPreallocOff");
            adjIO_.writeVectorASCII(dRdWPreallocOn_,"dRdWPreallocOn");
            adjIO_.writeVectorASCII(dRdWPreallocOff_,"dRdWPreallocOff");
            
        }
    }
    else
    {
        MatAssemblyBegin(*conMat,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*conMat,MAT_FINAL_ASSEMBLY);
        
        //output the matrix to a file
        if(adjIO_.writeMatrices)
        {
            adjIO_.writeMatRowSize(*conMat, "dRdWCon");
            //Mat tmpMat;
            //MatTranspose(dRdWCon_, MAT_INITIAL_MATRIX,&tmpMat);
            //adjIO_.writeMatRowSize(tmpMat, "dRdWTCon");
            //MatDestroy(&tmpMat);
            adjIO_.writeMatrixBinary(*conMat,"dRdWCon");
        }
    }
    
    if(isPrealloc) Info<<"Preallocating state Jacobian connectivity mat: finished!"<<endl;
    else Info<<"Setup state Jacobian connectivity mat: finished!"<<endl;

}


void AdjointJacobianConnectivity::setupdRdXvCon(label isPrealloc)
{
    // Calculates connectivity for dRdXv
    // if isPrealloc = 1, calculate the preallocation vectors, else, calculate dRdXvCon
    // dRdXvPreallocOn: preallocation vector that stores the number of on-diagonal connectivity for each row 
    // dRdXvCon: state Jacobian connectivity mat with dimension sizeAdjStates by sizeXv
    // connectedStatesP: one row matrix that stores the actual connectivity, element value 1 denotes 
    // a connected Xv. connectedStatesP is then used to assign dRdXvPreallocOn or dRdXvCon

    Mat* conMat;
    conMat=&dRdXvCon_;

    label globalIdx;
    Mat connectedStatesP;

    PetscInt    nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    if(isPrealloc)
    {
        VecZeroEntries(dRdXvPreallocOn_);
        VecZeroEntries(dRdXvPreallocOff_);
    }

    if(isPrealloc) Info<<"Preallocating dRdXv connnectivity mat"<<endl;
    else Info<<"Setup dRdXv connnectivity mat"<<endl;
    
    // loop over all cell residuals
    forAll(adjIdx_.adjStateNames,idxI) 
    {
        // get stateName and residual names
        word stateName = adjIdx_.adjStateNames[idxI];
        word resName = stateName+"Res";

        // check if this state is a cell state, we do surfaceScalarState residuals separately
        if (adjIdx_.adjStateType[stateName] == "surfaceScalarState") continue; 

        // maximal connectivity level information
        label maxConLevel = adjStateResidualConInfo_[resName].size()-1;
        
        // if it is a vectorState, set compMax=3
        label compMax = 1;
        if (adjIdx_.adjStateType[stateName] == "volVectorState") compMax=3;
        
        forAll(mesh_.cells(), cellI)
        {
            for(label comp=0; comp<compMax; comp++)
            {
            
                //zero the connections
                this->createConnectionMat(&connectedStatesP);
        
                // now add the con. We loop over all the connectivity levels
                forAll(adjStateResidualConInfo_[resName],idxJ) // idxJ: con level
                {
                    label connectedLevelLocal= idxJ;
                    label connectedLevelInterProc=0;
                    
                    // set connectedLevelInterProc: the globally connected state variables for this level
                    if(idxJ==0)
                    {
                        // pass a zero list, no need to add interProc connecitivity for level 0
                        connectedLevelInterProc=0; 
                    }
                    else
                    {
                        connectedLevelInterProc=maxConLevel-idxJ;
                    }

                    // Add connectivity
                    this->addXvConnections
                    (
                        connectedStatesP,
                        cellI,
                        connectedLevelLocal,
                        connectedLevelInterProc
                    );
                    
                }   
                
                // get the global index of the current state for the row index
                globalIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,comp);
                
                if(isPrealloc)
                {
                    this->allocateJacobianConnectionsXv(dRdXvPreallocOn_,dRdXvPreallocOff_,
                                                      connectedStatesP,globalIdx);
                }
                else
                {
                    this->setupJacobianConnections(*conMat,connectedStatesP,globalIdx);
                }

            }
            
        }

    }
    
    
    // loop over all face residuals
    forAll(adjReg_.surfaceScalarStates,idxI) 
    {
        // get stateName and residual names
        word stateName = adjReg_.surfaceScalarStates[idxI];
        word resName = stateName+"Res";

        // maximal connectivity level information
        label maxConLevel = adjStateResidualConInfo_[resName].size()-1;
                
        forAll(mesh_.faces(), faceI)
        {
            
            //zero the connections
            this->createConnectionMat(&connectedStatesP);
            
            // Get the owner and neighbour cells for this face
            label idxO=-1,idxN=-1;
            if (faceI<adjIdx_.nLocalInternalFaces)
            {
                idxO = mesh_.owner()[faceI];
                idxN = mesh_.neighbour()[faceI];
            }
            else
            {
                label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                label patchIdx=adjIdx_.bFacePatchI[relIdx];
                label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                
                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                idxN = pFaceCells[faceIdx];
            }
            
            
            // now add the con. We loop over all the connectivity levels
            forAll(adjStateResidualConInfo_[resName],idxJ) // idxJ: con level
            {
                label connectedLevelLocal= idxJ;
                label connectedLevelInterProc=0;
                
                // set connectedLevelInterProc: the globally connected state variables for this level
                if(idxJ==0)
                {
                    // pass a zero list, no need to add interProc connecitivity for level 0
                    connectedLevelInterProc=0; 
                }
                else
                {
                    connectedLevelInterProc=maxConLevel-idxJ;
                }
                // Add connectivity for idxN
                this->addXvConnections
                (
                    connectedStatesP,
                    idxN,
                    connectedLevelLocal,
                    connectedLevelInterProc
                );
                
                if(faceI<adjIdx_.nLocalInternalFaces)
                {
                    // Add connectivity for idxO
                    this->addXvConnections
                    (
                        connectedStatesP,
                        idxO,
                        connectedLevelLocal,
                        connectedLevelInterProc
                    );
                }
                
            } 

            // NOTE: if this faceI is on a coupled patch, the above connectivity is not enough to
            // cover the points on the other side of proc domain, we need to add 3 lvs of cells here
            if(faceI>=adjIdx_.nLocalInternalFaces)
            {
                label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                label patchIdx=adjIdx_.bFacePatchI[relIdx];
                
                label maxLevel=adjStateResidualConInfo_[resName].size();

                if ( mesh_.boundaryMesh()[patchIdx].coupled() )
                {
                    
                    label bRow=this->getLocalCoupledBFaceIndex(faceI);
                    label bRowGlobal=adjIdx_.globalCoupledBFaceNumbering.toGlobal(bRow);
                    MatGetRow(xvBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);
                    for(label i=0; i<nCols; i++)
                    {
                        PetscInt idxJ = cols[i];
                        label val = round(vals[i]);
                        // we are going to add some selective states with connectivity level <= 3
                        // NOTE: we use val-1 here since phi actually has 3 levels of connectivity
                        // however, when we assign adjStateResidualConInfo_, we ignore the level 0
                        // connectivity since they are idxN and idxO
                        if( val<maxLevel+1 && val>0 ) 
                            this->setConnections(connectedStatesP,idxJ);
                    }
                    MatRestoreRow(xvBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);
                }  
            }

            // get the global index of the current state for the row index
            globalIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,faceI);
            
            if(isPrealloc)
            {
                this->allocateJacobianConnectionsXv(dRdXvPreallocOn_,dRdXvPreallocOff_,
                                                  connectedStatesP,globalIdx);
            }
            else
            {
                this->setupJacobianConnections(*conMat,connectedStatesP,globalIdx);
            }
            
        }

    }
        
    if(isPrealloc)
    {
        VecAssemblyBegin(dRdXvPreallocOn_);
        VecAssemblyEnd(dRdXvPreallocOn_);
        VecAssemblyBegin(dRdXvPreallocOff_);
        VecAssemblyEnd(dRdXvPreallocOff_);
        
        //output the matrix to a file
        if(adjIO_.writeMatrices)
        {
            adjIO_.writeVectorASCII(dRdXvPreallocOn_,"dRdXvPreallocOn");
            adjIO_.writeVectorASCII(dRdXvPreallocOff_,"dRdXvPreallocOff");
        }
    }
    else
    {
        MatAssemblyBegin(*conMat,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(*conMat,MAT_FINAL_ASSEMBLY);
        
        //output the matrix to a file
        if(adjIO_.writeMatrices)
        {
            adjIO_.writeMatRowSize(*conMat, "dRdXvCon");
            //Mat tmpMat;
            //MatTranspose(dRdWCon_, MAT_INITIAL_MATRIX,&tmpMat);
            //adjIO_.writeMatRowSize(tmpMat, "dRdWTCon");
            //MatDestroy(&tmpMat);
            adjIO_.writeMatrixBinary(*conMat,"dRdXvCon");
        }
    }
    
    if(isPrealloc) Info<<"Preallocating dRdXv Jacobian connnectivity mat: finished!"<<endl;
    else Info<<"Setup dRdXv Jacobian connnectivity mat: finished!"<<endl;

}


void AdjointJacobianConnectivity::reduceAdjStateResidualConLevel()
{
    // if no maxResConLv4JacPCMat is specified, just return;
    if(adjIO_.maxResConLv4JacPCMat.size()==0) return;
    
    // now check if maxResConLv4JacPCMat has all the maxRes level defined
    // and these max levels are <= adjStateResidualConInfo_.size()
    forAll(adjStateResidualConInfo_.toc(),idxJ)
    {
        word key1=adjStateResidualConInfo_.toc()[idxJ];
        bool keyFound=false;
        forAll(adjIO_.maxResConLv4JacPCMat.toc(),idxI)
        {
            word key=adjIO_.maxResConLv4JacPCMat.toc()[idxI];
            if(key==key1)
            {
                keyFound=true;
                label maxLv=adjIO_.maxResConLv4JacPCMat[key];
                label maxLv1=adjStateResidualConInfo_[key1].size()-1;
                if(maxLv>maxLv1)
                {
                    FatalErrorIn("")<<"maxResConLv4JacPCMat maxLevel"
                                    <<maxLv<<" for "<<key
                                    <<" larger than adjStateResidualConInfo maxLevel "
                                    <<maxLv1<<" for "<<key1
                                    <<abort(FatalError);
                }
            }
        }
        if(!keyFound) 
        {
            FatalErrorIn("")<<key1<<" not found in maxResConLv4JacPCMat"<<abort(FatalError);
        }
    }

    Info<<"Reducing max connectivity level of Jacobian PC Mat to:";
    Info<<adjIO_.maxResConLv4JacPCMat<<endl;
    
    forAll(adjStateResidualConInfo_.toc(),idxI)
    {
        word key=adjStateResidualConInfo_.toc()[idxI];
        adjStateResidualConInfoBK_.set(key,adjStateResidualConInfo_[key]);
    }

    // now we can erase adjStateResidualConInfo
    adjStateResidualConInfo_.clearStorage();

    forAll(adjStateResidualConInfoBK_.toc(),idxI)
    {
        word key=adjStateResidualConInfoBK_.toc()[idxI];
        label maxConLevel = adjIO_.maxResConLv4JacPCMat[key];
        label conSize = adjStateResidualConInfoBK_[key].size();
        if (conSize > maxConLevel+1) 
        {
            List< List<word> > conList;
            conList.setSize(maxConLevel+1);
            for(label i=0;i<=maxConLevel;i++) // NOTE: it is <=
            {
                conList[i]=adjStateResidualConInfoBK_[key][i];
            }
            adjStateResidualConInfo_.set(key,conList);
        }
        else
        {
            adjStateResidualConInfo_.set(key,adjStateResidualConInfoBK_[key]);
        }
        
    }
    //Info<<adjStateResidualConInfo_<<endl;
    
}

void AdjointJacobianConnectivity::restoreAdjStateResidualConLevel()
{

    forAll(adjStateResidualConInfoBK_.toc(),idxI)
    {
        word key=adjStateResidualConInfoBK_.toc()[idxI];
        adjStateResidualConInfo_.set(key,adjStateResidualConInfoBK_[key]);
    }
    
    return;

}



void AdjointJacobianConnectivity::setupObjFuncCon(const word objFunc,const word mode)
{
    Info<<"Calculating "<<mode<<" for "<<objFunc<< endl;
    
    if (mode=="dFdW") MatZeroEntries(dFdWCon_);
    else if (mode=="dFdXv") MatZeroEntries(dFdXvCon_);
    else FatalErrorIn("")<<"mode not valid"<<abort(FatalError);

    adjIdx_.initializeObjFuncGeoNumbering(objFunc);
    
    List<word> forceRelatedList={"CD","CL","CMX","CMY","CMZ"};
    
    if( adjIO_.isInList<word>(objFunc,forceRelatedList) )
    {
        // force related con is connected to Level0 of U,p,nut and level1 of U
        word pName=adjIO_.getPName();
#ifdef IncompressibleFlow        
        List< List<word> > forceConInfo=
        {
            {"U",pName,"nut"},
            {"U"}
        };
#endif
#ifdef PhaseIncompressibleFlow        
        List< List<word> > forceConInfo=
        {
            {"U",pName,"nut"},
            {"U"}
        };
#endif
#ifdef CompressibleFlow        
        List< List<word> > forceConInfo=
        {
            {"U",pName,"T","nut"},
            {"U"}
        };
#endif
        // need to correct turb con for nut, e.g., for SA model, we replace nut with nuTilda
        adjRAS_.correctAdjStateResidualTurbCon(forceConInfo);
        
        if (mode=="dFdW") this->addObjFuncConnectivity(objFunc,forceConInfo);
        else if (mode=="dFdXv") this->addObjFuncConnectivityXv(objFunc,forceConInfo);
        else FatalErrorIn("")<<"mode not valid"<<abort(FatalError);

    }
    else if(objFunc=="NUS")
    {
        // NUS con is connected to Level0 of T nut and level1 of T
        // NOTE: if alphatWallFunction is used, NUS also depends on level0 and level1 U
        // since alphatWallFunction depends on yPlus
        List< List<word> > NUSConInfo=
        {
            {"U","T","nut"},
            {"U","T"}
        };
        // need to correct turb con for nut, e.g., for SA model, we replace nut with nuTilda
        adjRAS_.correctAdjStateResidualTurbCon(NUSConInfo);

        if (mode=="dFdW") this->addObjFuncConnectivity("NUS",NUSConInfo);
        else FatalErrorIn("")<<"mode not valid for NUS"<<abort(FatalError);
    }
    else if(objFunc=="CPL")
    {
        // CPL con is connected to Level0 of U and p 
        word pName=adjIO_.getPName();
#ifdef IncompressibleFlow  
        List< List<word> > CPLConInfo=
        {
            {"U",pName}
        };
#endif
#ifdef PhaseIncompressibleFlow  
        List< List<word> > CPLConInfo=
        {
            {"U",pName}
        };
#endif
#ifdef CompressibleFlow  
        List< List<word> > CPLConInfo=
        {
            {"U",pName,"T"}
        };
#endif
        if (mode=="dFdW") this->addObjFuncConnectivity("CPL",CPLConInfo);
        else FatalErrorIn("")<<"mode not valid for CPL"<<abort(FatalError);
    }
    else if (objFunc=="AVGV")
    {
        List< List<word> > tmpConInfo={{"zero"}};
        if (mode=="dFdW") this->addObjFuncConnectivity("AVGV",tmpConInfo);
        else FatalErrorIn("")<<"mode not valid for AVGV"<<abort(FatalError);
    }
    else if (objFunc=="VARV")
    {
        List< List<word> > tmpConInfo={{"all"}};
        if (mode=="dFdW") this->addObjFuncConnectivity("VARV",tmpConInfo);
        else FatalErrorIn("")<<"mode not valid for VARV"<<abort(FatalError);
    }
    else if (objFunc=="AVGS")
    {
        List< List<word> > tmpConInfo={{"zero"}};
        if (mode=="dFdW") this->addObjFuncConnectivity("AVGS",tmpConInfo);
        else FatalErrorIn("")<<"mode not valid for AVGS"<<abort(FatalError);
    }
    else if (objFunc=="VMS")
    {
        // von Mises stress con is connected to 1 level of D 
        List< List<word> > VMSConInfo=
        {
            {"D"},
            {"D"}
        };
        if (mode=="dFdW") this->addObjFuncConnectivity("VMS",VMSConInfo);
        else FatalErrorIn("")<<"mode not valid for VMS"<<abort(FatalError);
    }
    else if(objFunc=="TPR")
    {
        // TPR con is connected to Level0 of U, T, and p 
        word pName=adjIO_.getPName();
        List< List<word> > TPRConInfo=
        {
            {"U",pName,"T"}
        };
        if (mode=="dFdW") this->addObjFuncConnectivity("TPR",TPRConInfo);
        else FatalErrorIn("")<<"mode not valid for TPR"<<abort(FatalError);
    }
    else if(objFunc=="TTR")
    {
        // TTR con is connected to Level0 of U, T
        List< List<word> > TTRConInfo=
        {
            {"U","T"}
        };
        if (mode=="dFdW") this->addObjFuncConnectivity("TTR",TTRConInfo);
        else FatalErrorIn("")<<"mode not valid for TTR"<<abort(FatalError);
    }
    else if(objFunc=="MFR")
    {
        // MFR con is connected to Level0 of U, T, and p 
        word pName=adjIO_.getPName();
        List< List<word> > MFRConInfo=
        {
            {"U",pName,"T"}
        };
        if (mode=="dFdW") this->addObjFuncConnectivity("MFR",MFRConInfo);
        else FatalErrorIn("")<<"mode not valid for MFR"<<abort(FatalError);
    }
    else
    {
        FatalErrorIn("")<<"objFunc not supported!"<<abort(FatalError);
    }
    
    adjIdx_.deleteObjFuncGeoNumbering();
    
    if(mode=="dFdW")
    {
        MatAssemblyBegin(dFdWCon_,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(dFdWCon_,MAT_FINAL_ASSEMBLY);
    }
    
    if(mode=="dFdXv")
    {
        MatAssemblyBegin(dFdXvCon_,MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(dFdXvCon_,MAT_FINAL_ASSEMBLY);
    }
    
    if(adjIO_.writeMatrices)
    {
        if(mode=="dFdW") adjIO_.writeMatrixBinary(dFdWCon_,"dFdWCon_"+objFunc);
        if(mode=="dFdXv") adjIO_.writeMatrixBinary(dFdXvCon_,"dFdXvCon_"+objFunc);
    }
    
    return;
    
}

void AdjointJacobianConnectivity::addObjFuncConnectivity
( 
    word objFunc,
    List< List<word> > objFuncConInfo 
)
{
    /* 
       General function to add connectivity for objective functions

       If the objFunc is a patch-based one, we need to provides all levels of connected state names
       in objFuncConInfo. For example, objFuncConInf={ {"U","P","nut"} {"U"} } indicates that the discrete value of this
       objFunc (on a given patch face) is connected to "U", "P", and "nut" for level-1 and "U" for level-2, e.g., CD
       
       If the objFunc is an volume-based one,  objFuncConInf contains words indicating how many levels of
       surrounding cells are connected. For example, objFuncConInf={{"zero"}} means the discrete value of this
       objFunc (on a given cell) depends on its own cell only (e.g., AVGV), while objFuncConInf={{"all"}} means this objFunc depends on all the cells that is occupied in the user-defined volume (e.g., VARV).
       *** NOTE ***
       If objFuncConInf={{"all"}}, we need to make sure the number of all user-defined volume cells is smaller than a few thousands. Otherwise, the coloring for dfdw will be extremely expensive.
    */  


    Mat connectedStatesP;
    List<word> objFuncGeoInfo = adjIdx_.getObjFuncGeoInfo(objFunc);

    // objFuncConInfo shouldn't include surfaceScalarStates for surface-based obj
    forAll(objFuncGeoInfo,idxI)
    {
        word geoName = objFuncGeoInfo[idxI];
        if (!adjIdx_.isUserDefinedVolume(geoName) and !adjIdx_.isUserDefinedPatch(geoName))
        {
            forAll(objFuncConInfo,idxJ)
            {
                forAll(objFuncConInfo[idxJ],idxK)
                {
                    word stateType = adjIdx_.adjStateType[objFuncConInfo[idxJ][idxK]];
                    if(stateType == "surfaceScalarState")
                        FatalErrorIn("")<<"surfaceScalarState not supported!"<<abort(FatalError);
                }
            }
        }
    }
    
    label addFace=0;
    
    // maximal connectivity level information
    label maxConLevel = objFuncConInfo.size()-1;

    label localObjFuncIdx = 0;
    forAll(objFuncGeoInfo,idxI)
    {
        word geoName = objFuncGeoInfo[idxI];

        // if this patch is UserDefined patch, we need special treatment
        if( adjIdx_.isUserDefinedPatch(geoName) )
        {
            if (objFuncConInfo[0][0]=="zero") // we add only the stateName to dFdWCon, this is for AVGS with user-defined patches
            {
                dictionary geoDict = adjIO_.userDefinedPatchInfo.subDict(geoName);
                word stateName = word(geoDict["stateName"]);
                label comp = readLabel( geoDict["component"]);

                // we will basically add the own cell to rowI
                forAll(adjIdx_.faceIdx4UserDefinedPatches[geoName],idxJ)
                {
                    label faceI = adjIdx_.faceIdx4UserDefinedPatches[geoName][idxJ];

                    if(faceI < adjIdx_.nLocalInternalFaces) 
                    {
                        label idxO=mesh_.owner()[faceI];
                        label idxN=mesh_.neighbour()[faceI];

                        PetscScalar valIn=1.0;
                        label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);

                        label glbIdxN = adjIdx_.getGlobalAdjointStateIndex(stateName,idxN,comp);
                        MatSetValue(dFdWCon_,rowI,glbIdxN,valIn,INSERT_VALUES);

                        label glbIdxO = adjIdx_.getGlobalAdjointStateIndex(stateName,idxO,comp);
                        MatSetValue(dFdWCon_,rowI,glbIdxO,valIn,INSERT_VALUES);
                    }
                    else
                    {
                        label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                        label patchIdx=adjIdx_.bFacePatchI[relIdx];
                        label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    
                        const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                        label idxN = pFaceCells[faceIdx];

                        // calc global index
                        label glbIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,idxN,comp);
                        // set the con mat
                        PetscScalar valIn=1.0;
                        label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);
                        MatSetValue(dFdWCon_,rowI,glbIdx,valIn,INSERT_VALUES);
                    }

                    localObjFuncIdx+=1;
                    
                }

            }
            else // we add connectivity based on the prescribed objFuncConInfo, this is for CPL with user-defined patches
            {
                label idxO=-1,idxN=-1;
                labelList userDefinedPatchFaces = adjIdx_.faceIdx4UserDefinedPatches[geoName];
                forAll(userDefinedPatchFaces,idxF)
                {
                    label faceI = userDefinedPatchFaces[idxF];
                    if(faceI < adjIdx_.nLocalInternalFaces) 
                    {
                        idxO=mesh_.owner()[faceI];
                        idxN=mesh_.neighbour()[faceI];
                    }
                    else
                    {
                        label relIdx=faceI-adjIdx_.nLocalInternalFaces;
                        label patchIdx=adjIdx_.bFacePatchI[relIdx];
                        label faceIdx=adjIdx_.bFaceFaceI[relIdx];
                    
                        const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                        idxN = pFaceCells[faceIdx];
                    }
                    
                    //zero the connections
                    this->createConnectionMat(&connectedStatesP);
                    
                    forAll(objFuncConInfo ,idxJ)  // idxJ: con level
                    {
                        // set connectedStatesLocal: the locally connected state variables for this level
                        wordList connectedStatesLocal=objFuncConInfo[idxJ];
                        
                        // set connectedStatesInterProc: the globally connected state variables for this level
                        List< List<word> > connectedStatesInterProc;
                        
                        if(idxJ==0)
                        {
                            // pass a zero list, no need to add interProc connecitivity for level 0
                            connectedStatesInterProc.setSize(0); 
                        }
                        else if (idxJ != maxConLevel)
                        {
                            connectedStatesInterProc.setSize(maxConLevel-idxJ+1);
                            for(label k=0;k<maxConLevel-idxJ+1;k++)
                            {
                                connectedStatesInterProc[k]=objFuncConInfo[k+idxJ]; 
                            }
                            
                        }
                        else
                        {
                            connectedStatesInterProc.setSize(1);
                            connectedStatesInterProc[0] = objFuncConInfo[maxConLevel];
                        }
                        
                        if(idxN>=0)
                        {
                            this->addStateConnections
                            (
                                connectedStatesP,
                                idxN,
                                idxJ,
                                connectedStatesLocal,
                                connectedStatesInterProc,
                                addFace
                            );
                        }
                        
                        if(idxO>=0)
                        {
                            this->addStateConnections
                            (
                                connectedStatesP,
                                idxO,
                                idxJ,
                                connectedStatesLocal,
                                connectedStatesInterProc,
                                addFace
                            );
                        }
                        
                    }
                    
                    label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);
                    localObjFuncIdx+=1;
                    
                    this->setupJacobianConnections(dFdWCon_,connectedStatesP,rowI);
                }
            }
        }
        else if ( adjIdx_.isUserDefinedVolume(geoName) )
        {
            // first check if objFuncConInfo has only one element
            if (objFuncConInfo.size()!=1)
            {
                FatalErrorIn("")<<"prescribe only one value for objFuncConInfo"<<abort(FatalError);
            }
            else if (objFuncConInfo[0].size()!=1)
            {
                FatalErrorIn("")<<"prescribe only one value for objFuncConInfo"<<abort(FatalError);
            }

            // get state info

            dictionary geoDict = adjIO_.userDefinedVolumeInfo.subDict(geoName);
            label comp = readLabel( geoDict["component"]);
            word stateName = word(geoDict["stateName"]);

            if (objFuncConInfo[0][0]=="zero")
            {
                // we will basically add the own cell to rowI
                forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
                {
                    label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                    // calc global index
                    label glbIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,comp);
                    // set the con mat
                    PetscScalar valIn=1.0;
                    label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);
                    localObjFuncIdx+=1;
                    MatSetValue(dFdWCon_,rowI,glbIdx,valIn,INSERT_VALUES);
                }
            }
            else if (objFuncConInfo[0][0]=="one")
            {
                FatalErrorIn("")<<"objFuncConInfo=one not implemented!"<<abort(FatalError);
            }
            else if (objFuncConInfo[0][0]=="two")
            {
                FatalErrorIn("")<<"objFuncConInfo=two not implemented!"<<abort(FatalError);
            }
            else if (objFuncConInfo[0][0]=="three")
            {
                FatalErrorIn("")<<"objFuncConInfo=three not implemented!"<<abort(FatalError);
            }
            else if (objFuncConInfo[0][0]=="all")
            {
                // since objFuncConInfo=all, we need to add all userDefined volume cells for  each rowI
                // *** NOTE *** this makes dFdWCon very dense! try to reduce
                // the total number of userDefined volume cells

                // first we need to calculate the globalAdjIdx for all cellIdx4UserDefinedVolumes cells that are distributed among processors
                // then we gather them to the master proc, finally the master proc will scatter all the globalAdjIdx to every procs. 
                // This is to make sure we have all globalAdjIdx for each rowI

                label myProc = Pstream::myProcNo();
                label nProcs = Pstream::nProcs();
                // create listlist for gathering 
                List< List<label> > globalAdjIdxAll(nProcs);
                // assign values for the listlists on the local proc
                forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
                {
                    label cellI = adjIdx_.cellIdx4UserDefinedVolumes[geoName][idxJ];
                    // calc global index
                    label glbIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,comp);
                    globalAdjIdxAll[myProc].append(glbIdx);
                }
                // gather all info to the master proc
                Pstream::gatherList(globalAdjIdxAll);
                // scatter all info to every procs
                Pstream::scatterList(globalAdjIdxAll);
                // since globalAdjIdxAll is a listList, we need to convert it to a 1D list 
                labelList globalAdjIdxAllList;
                forAll(globalAdjIdxAll,idxJ)
                {
                    forAll(globalAdjIdxAll[idxJ],idxK)
                    {
                        globalAdjIdxAllList.append(globalAdjIdxAll[idxJ][idxK]);
                    }
                }

                // now we will add all the globalAdjIdx for each rowI
                forAll(adjIdx_.cellIdx4UserDefinedVolumes[geoName],idxJ)
                {
                    label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);

                    forAll(globalAdjIdxAllList,idxK)
                    {
                        label glbIdx  = globalAdjIdxAllList[idxK];
                        // set the con mat
                        PetscScalar valIn=1.0;
                        MatSetValue(dFdWCon_,rowI,glbIdx,valIn,INSERT_VALUES);
                    }

                    localObjFuncIdx+=1;
                }
            }
            else
            {
                FatalErrorIn("")<<"objFuncConInfo not valid! Options are: zero, one, two, three, all"<<abort(FatalError);
            }

        }
        else if (objFunc=="VMS")
        {
            if (objFuncGeoInfo[0]!="allCells")  FatalErrorIn("")<<"set geoInfo to allCells"<<abort(FatalError);
            
            forAll(mesh_.cells(),cellI)
            {

                //zero the connections
                this->createConnectionMat(&connectedStatesP);
                
                forAll(objFuncConInfo ,idxJ)  // idxJ: con level
                {
                    // set connectedStatesLocal: the locally connected state variables for this level
                    wordList connectedStatesLocal=objFuncConInfo[idxJ];
                    
                    // set connectedStatesInterProc: the globally connected state variables for this level
                    List< List<word> > connectedStatesInterProc;
                    
                    if(idxJ==0)
                    {
                        // pass a zero list, no need to add interProc connecitivity for level 0
                        connectedStatesInterProc.setSize(0); 
                    }
                    else if (idxJ != maxConLevel)
                    {
                        connectedStatesInterProc.setSize(maxConLevel-idxJ+1);
                        for(label k=0;k<maxConLevel-idxJ+1;k++)
                        {
                            connectedStatesInterProc[k]=objFuncConInfo[k+idxJ]; 
                        }
                        
                    }
                    else
                    {
                        connectedStatesInterProc.setSize(1);
                        connectedStatesInterProc[0] = objFuncConInfo[maxConLevel];
                    }
                    
                    this->addStateConnections
                    (
                        connectedStatesP,
                        cellI,
                        idxJ,
                        connectedStatesLocal,
                        connectedStatesInterProc,
                        addFace
                    );
                }

                label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(cellI);
                
                this->setupJacobianConnections(dFdWCon_,connectedStatesP,rowI);
                
            }
        }
        else // it is surface patches
        {
            // get the patch id label
            label patchI = mesh_.boundaryMesh().findPatchID( objFuncGeoInfo[idxI] );
            // create a shorter handle for the boundary patch
            const fvPatch& patch = mesh_.boundary()[patchI];
            // get the cells associated with this boundary patch
            const UList<label>& pFaceCells = patch.faceCells();
            
            forAll(patch,faceI)
            {
                // Now get the cell that borders this face
                label idxN = pFaceCells[faceI];
                
                //zero the connections
                this->createConnectionMat(&connectedStatesP);
                
                forAll(objFuncConInfo ,idxJ)  // idxJ: con level
                {
                    // set connectedStatesLocal: the locally connected state variables for this level
                    wordList connectedStatesLocal=objFuncConInfo[idxJ];
                    
                    // set connectedStatesInterProc: the globally connected state variables for this level
                    List< List<word> > connectedStatesInterProc;
                    
                    if(idxJ==0)
                    {
                        // pass a zero list, no need to add interProc connecitivity for level 0
                        connectedStatesInterProc.setSize(0); 
                    }
                    else if (idxJ != maxConLevel)
                    {
                        connectedStatesInterProc.setSize(maxConLevel-idxJ+1);
                        for(label k=0;k<maxConLevel-idxJ+1;k++)
                        {
                            connectedStatesInterProc[k]=objFuncConInfo[k+idxJ]; 
                        }
                        
                    }
                    else
                    {
                        connectedStatesInterProc.setSize(1);
                        connectedStatesInterProc[0] = objFuncConInfo[maxConLevel];
                    }
                    
                    this->addStateConnections
                    (
                        connectedStatesP,
                        idxN,
                        idxJ,
                        connectedStatesLocal,
                        connectedStatesInterProc,
                        addFace
                    );
                }
                
                label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);
                localObjFuncIdx+=1;
                
                this->setupJacobianConnections(dFdWCon_,connectedStatesP,rowI);
            }
        }
    }
            
    return;
}

void AdjointJacobianConnectivity::addObjFuncConnectivityXv
( 
    word objFunc,
    List< List<word> > objFuncConInfo 
)
{

    Mat connectedStatesP;
    List<word> objFuncGeoInfo = adjIdx_.getObjFuncGeoInfo(objFunc);

    // objFuncConInfo shouldn't include surfaceScalarStates for surface-based obj
    forAll(objFuncGeoInfo,idxI)
    {
        word geoName = objFuncGeoInfo[idxI];
        if (!adjIdx_.isUserDefinedVolume(geoName) )
        {
            forAll(objFuncConInfo,idxJ)
            {
                forAll(objFuncConInfo[idxJ],idxK)
                {
                    word stateType = adjIdx_.adjStateType[objFuncConInfo[idxJ][idxK]];
                    if(stateType == "surfaceScalarState")
                        FatalErrorIn("")<<"surfaceScalarState not supported!"<<abort(FatalError);
                }
            }
        }
    }
        
    // maximal connectivity level information
    label maxConLevel = objFuncConInfo.size()-1;

    label localObjFuncIdx = 0;
    forAll(objFuncGeoInfo,idxI)
    {
        word geoName = objFuncGeoInfo[idxI];
        // if this patch is userDefined patch, we need special treatment
        if( adjIdx_.isUserDefinedPatch(geoName) )
        {
            FatalErrorIn("")<<"not implemented"<<abort(FatalError);
        }
        else if ( adjIdx_.isUserDefinedVolume(geoName) )
        {
            FatalErrorIn("")<<"not implemented"<<abort(FatalError);

        }
        else // it is surface patches
        {
            // get the patch id label
            label patchI = mesh_.boundaryMesh().findPatchID( objFuncGeoInfo[idxI] );
            // create a shorter handle for the boundary patch
            const fvPatch& patch = mesh_.boundary()[patchI];
            // get the cells associated with this boundary patch
            const UList<label>& pFaceCells = patch.faceCells();
            
            forAll(patch,faceI)
            {
                // Now get the cell that borders this face
                label idxN = pFaceCells[faceI];
                
                //zero the connections
                this->createConnectionMat(&connectedStatesP);
                
                forAll(objFuncConInfo ,idxJ)  // idxJ: con level
                {
                    label connectedLevelLocal= idxJ;
                    label connectedLevelInterProc=0;

                    // set connectedLevelInterProc: the globally connected state variables for this level
                    if(idxJ==0)
                    {
                        // pass a zero list, no need to add interProc connecitivity for level 0
                        connectedLevelInterProc=0; 
                    }
                    else
                    {
                        connectedLevelInterProc=maxConLevel-idxJ;
                    }

                    // Add connectivity
                    this->addXvConnections
                    (
                        connectedStatesP,
                        idxN,
                        connectedLevelLocal,
                        connectedLevelInterProc
                    );
                }
                
                label rowI = adjIdx_.globalObjFuncGeoNumbering.toGlobal(localObjFuncIdx);
                localObjFuncIdx+=1;
                
                this->setupJacobianConnections(dFdXvCon_,connectedStatesP,rowI);
            }
        }
    }
            
    return;
}

void AdjointJacobianConnectivity::calcNeiBFaceGlobalCompact()
{
    
    // This function calculates neiBFaceGlobalCompact[bFaceI]. Here neiBFaceGlobalCompact stores 
    // the global coupled boundary face index for the face on the other side of the local processor 
    // boundary. bFaceI is the "compact" face index. bFaceI=0 for the first boundary face
    // neiBFaceGlobalCompat.size() = nLocalBoundaryFaces
    // neiBFaceGlobalCompact[bFaceI]=-1 means it is not a coupled face
    // NOTE: neiBFaceGlobalCompact will be used to calculate the connectivity across processors
    // in setupStateBoundaryCon
    //
    // Basically, on proc0 stateBoundaryCon[0] = 1024 for the following example
    //
    //                      localBFaceI = 0        proc0
    //               ---------------------------   coupled boundary face
    //                      globalBFaceI=1024      proc1
    
    // Taken and modified from the extended stencil code in fvMesh
    // Swap the global boundary face index
                          

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    
    neiBFaceGlobalCompact_.setSize(adjIdx_.nLocalBoundaryFaces);

    // initialize the list
    forAll(neiBFaceGlobalCompact_,idx)
    {
        neiBFaceGlobalCompact_[idx] = -1;
    }

    // loop over the patches and store the global indices
    label counter=0;
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];

        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            // For coupled faces set the global face index so that it can be
            // swaped across the interface.
            forAll(pp, i)
            {
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                neiBFaceGlobalCompact_[bFaceI] = adjIdx_.globalCoupledBFaceNumbering.toGlobal(counter);
                faceIStart++;
                counter++;
            }
        }
    }

    // Swap the cell indices, the list now contains the global index for the
    // U state for the cell on the other side of the processor boundary
    syncTools::swapBoundaryFaceList(mesh_, neiBFaceGlobalCompact_);

    return;

}

void AdjointJacobianConnectivity::calcCyclicAMIBFaceGlobalCompact()
{
    
    // This function calculates cyclicAMIBFaceGlobalCompact[bFaceI][amiFaceI]. Here cyclicAMIBFaceGlobalCompact 
    // stores the AMI coupled boundary face indices for the given bFaceI index.
    // bFaceI is the "compact" face index. bFaceI=0 for the first boundary face on the local processor
    // cyclicAMIBFaceGlobalCompat.size() = AdjointIndexing::nLocalBoundaryFaces
    // cyclicAMIBFaceGlobalCompat[0].size() = nAMICoupledFaces: how many faces are coupled to bFaceI=0
    // cyclicAMIBFaceGlobalCompat[bFaceI]={-1} means it is not a coupled AMI face
    // NOTE: cyclicAMIBFaceGlobalCompat will be used to calculate the connectivity across processors
    // in AdjointJacobianConnectivity::setupStateCyclicAMICon
    //
    // Basically, cyclicAMIBFaceGlobalCompat[0] = {1024,96} for the following example

    //                  globalBFaceI=96, globalBFaceI=1024
    // |-----------------------------=-=---------|  <- cyclicAMI patch
    // |                                         |  
    // |                                         |
    // |   simulation domain                     |
    // |                           |-----|       |
    // |                           | cell|       |
    // |-----------------------------------------|  <- cyclicAMI patch
    //                                |
    //                                |
    //                           amiFaceIndex=0 
                          

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    
    cyclicAMIBFaceGlobalCompact_.setSize(adjIdx_.nLocalBoundaryFaces);

    // first we need to check if cyclicAMI patches are owned by only one processor
    // if they are owned by more than one processor, quit and report an error
    // in this case, use singleProcessorFaceSets in the decomposeParDict to fix
    label nProcesOwnAMIPatches=0;
    procHasAMI_=0;
    forAll(patches,patchI)
    {
        if(patches[patchI].size()>0 && patches[patchI].type()=="cyclicAMI")
        {
            nProcesOwnAMIPatches=1;
            procHasAMI_=1;
        }
    }
    reduce(nProcesOwnAMIPatches, sumOp<label>() );
    if(nProcesOwnAMIPatches!=1 && nProcesOwnAMIPatches!=0)
    {
        FatalErrorIn("")<<"cyclicAMI patches are owned by more than one processor!"
                        <<abort(FatalError);
    }

    forAll(patches,patchI)
    {
        if(patches[patchI].size()>0 && patches[patchI].type()=="cyclicAMI")
        {
            // recast patches to patchAMI
            cyclicAMIPolyPatch& patchAMI =refCast< cyclicAMIPolyPatch > 
            ( 
                const_cast<polyPatch&>
                (
                    patches[patchI]
                )
            );

            // only treat the owner patch of the AMI pair
            if (patchAMI.owner())
            {
                // get the AMI class
                const AMIPatchToPatchInterpolation& ami=patchAMI.AMI();

                // here src is the current AMI patch and tgt is the coupled AMI patch
                label srcPatchI=patchI;
                label tgtPatchI=patchAMI.neighbPatchID();
                
                // deal with src cyclicAMI patch
                // srcAddress stores the tgt patch (coupled AMI) indices
                forAll(ami.srcAddress(),idxI) 
                {
                    forAll(ami.srcAddress()[idxI],idxJ) // we may have more than one coupled faces
                    {
                        label srcFaceI = idxI;
                        label tgtFaceI = ami.srcAddress()[idxI][idxJ];

                        label tgtFaceLocalIdx=adjIdx_.BFacePatchIFaceI2LocalIndex(tgtPatchI,tgtFaceI);
                        label bFaceI=tgtFaceLocalIdx-adjIdx_.nLocalInternalFaces;

                        label srcFaceLocalIdx=adjIdx_.BFacePatchIFaceI2LocalIndex(srcPatchI,srcFaceI);
                        label srcCyclicAMIFaceIdx=this->getLocalCyclicAMIFaceIndex(srcFaceLocalIdx);

                        cyclicAMIBFaceGlobalCompact_[bFaceI].append(srcCyclicAMIFaceIdx);

                    }
                    
                }

                // deal with tgt cyclicAMI patch
                // tgtAddress stores the tgt patch (coupled AMI) indices
                forAll(ami.tgtAddress(),idxI)
                {
                    forAll(ami.tgtAddress()[idxI],idxJ) // we may have more than one coupled faces
                    {
                        label tgtFaceI = idxI;
                        label srcFaceI = ami.tgtAddress()[idxI][idxJ];

                        label srcFaceLocalIdx=adjIdx_.BFacePatchIFaceI2LocalIndex(srcPatchI,srcFaceI);
                        label bFaceI=srcFaceLocalIdx-adjIdx_.nLocalInternalFaces;

                        label tgtFaceLocalIdx=adjIdx_.BFacePatchIFaceI2LocalIndex(tgtPatchI,tgtFaceI);
                        label tgtCyclicAMIFaceIdx=this->getLocalCyclicAMIFaceIndex(tgtFaceLocalIdx);
                        
                        cyclicAMIBFaceGlobalCompact_[bFaceI].append(tgtCyclicAMIFaceIdx);

                    }
                    
                }
            }
        }
    }

    //Info<<cyclicAMIBFaceGlobalCompact_<<endl;

    return;

}


void AdjointJacobianConnectivity::addConMatCell
(
    Mat conMat, 
    const label gRow,
    const label cellI,
    const word stateName, 
    const PetscScalar val
)
{

    // Insert a value (val) to the connectivity Matrix (conMat)
    // This value will be inserted at rowI=gRow
    // The column index is dependent on the cellI and stateName 

    PetscInt  idxJ,idxI;
    
    idxI = gRow;

    // find the global index of this state
    label compMax = 1;
    if (adjIdx_.adjStateType[stateName] == "volVectorState") compMax=3;
    
    for(label i=0;i<compMax;i++)
    {
        idxJ = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,i);
        // set it in the matrix
        MatSetValues(conMat,1,&idxI,1,&idxJ,&val,INSERT_VALUES);
    }

    return;

}

void AdjointJacobianConnectivity::addConMatCellXv
(
    Mat conMat, 
    const label gRow,
    const label cellI,
    const PetscScalar val
)
{

    // Calculate the global Xv indices for cellI and add them into conMat.
    // Row index to add: gRow (idxI).
    // Col index to add: global Xv indices of points that are belonged to cellI (idxJ).

    PetscInt  idxI;
    PetscInt  idxJ;

    idxI = gRow;
    // Add the points associated with this cell
    forAll(mesh_.cellPoints()[cellI],pointI)
    {
        label localPoint = mesh_.cellPoints()[cellI][pointI];

        // find the global index
        for(label coord=0; coord<3; coord++)
        {
            idxJ =  adjIdx_.getGlobalXvIndex(localPoint,coord);
            MatSetValue(conMat,idxI,idxJ,val,INSERT_VALUES);
        }

    }

    return;
}


void AdjointJacobianConnectivity::addConMatNeighbourCells
(
    Mat conMat, 
    const label gRow,
    const label cellI,
    const word stateName, 
    const PetscScalar val
)
{

    // Insert a value (val) to the connectivity Matrix (conMat)
    // This value will be inserted at rowI=gRow
    // The column indices are dependent on the cellI's neighbour cells and stateName 

    label     localCellJ;
    PetscInt  idxJ,idxI;

    idxI = gRow;
    // Add the nearest neighbour cells for cell
    forAll(mesh_.cellCells()[cellI],cellJ)
    {
        // get the local neighbour cell
        localCellJ = mesh_.cellCells()[cellI][cellJ];

        // find the global index of this state
        label compMax = 1;
        if (adjIdx_.adjStateType[stateName] == "volVectorState") compMax=3;
        for(label i=0;i<compMax;i++)
        {
            idxJ = adjIdx_.getGlobalAdjointStateIndex(stateName,localCellJ,i);
            // set it in the matrix
            MatSetValues(conMat,1,&idxI,1,&idxJ,&val,INSERT_VALUES);
        }

    }
    
    return;

}

void AdjointJacobianConnectivity::addConMatNeighbourCellsXv
(
    Mat conMat, 
    const label gRow,
    const label cellI,
    const PetscScalar val
)
{

    // Calculate the global Xv indices for the neighbour points connected to cellI,
    // and add them into conMat.
    // Row index to add: gRow (idxI).
    // Col index to add: global point coordinate indices of the neighbour points (idxJ).
    // Value to add: val. this should be the level of the point coordinates

    label     localCellJ;
    PetscInt  idxJ,idxI;
    idxI = gRow;

    // Add the nearest neighbour cells for cell
    forAll(mesh_.cellCells()[cellI],cellJ)
    {

        // get the local neighbour cell
        localCellJ = mesh_.cellCells()[cellI][cellJ];

        // Add the points associated with this cell
        forAll(mesh_.cellPoints()[localCellJ],pointI)
        {
            label localPoint = mesh_.cellPoints()[localCellJ][pointI];

            for(label i=0; i<3; i++)
            {
                idxJ = adjIdx_.getGlobalXvIndex(localPoint,i);
                // set it in the matrix
                MatSetValue(conMat,idxI,idxJ,val,INSERT_VALUES);
            }
        }
    }
    
    return;

}

void AdjointJacobianConnectivity::addConMatCellFaces
(
    Mat conMat,
    const label gRow, 
    const label cellI, 
    const word stateName, 
    const PetscScalar val
)
{

    // Insert a value (val) to the connectivity Matrix (conMat)
    // This value will be inserted at rowI=gRow
    // The column indices are dependent on the cellI's faces

    PetscInt  idxJ,idxI;
    idxI = gRow;
    
    // get the faces connected to this cell, note these are in a single
    // list that includes all internal and boundary faces
    const labelList& faces = mesh_.cells()[cellI];
    forAll(faces,idx)
    {
        //get the appropriate index for this face
        label globalState = adjIdx_.getGlobalAdjointStateIndex(stateName,faces[idx]);
        idxJ = globalState;
        MatSetValues(conMat,1,&idxI,1,&idxJ,&val,INSERT_VALUES);
    }
    
    return;
}

void AdjointJacobianConnectivity::setupStateBoundaryCon()
{
    // This function calculates stateBoundaryCon and stateBoundaryConID
    // stateBoundaryCon stores the level of connected states (on the other side across the boundary) for a
    // given coupled boundary face. stateBoundaryCon is a matrix with sizes of nGlobalCoupledBFaces by nGlobalAdjointStates
    // stateBoundaryCon is mainly used in the addBoundaryFaceConnection function
    
    // Basically, if there are 2 levels of connected states across the inter-proc boundary
    //
    //                                             proc0, globalBFaceI=1024
    //                        -----------------  <-coupled boundary face
    //  globalAdjStateIdx=100 ->   | lv1 |         proc1
    //                             |_____|
    //  globalAdjStateIdx=200 ->   | lv2 |
    //                             |_____| 
    //                           
    //
    // The indices for row 1024 in the stateBoundaryCon matrix will be
    // stateBoundaryCon
    // rowI=1024   
    // Cols: colI=0 .......... colI=100  ............ colI=200 .............  colI=nGlobalAdjointStates
    // Vals:                       1                     2           
    // NOTE: globalBFaceI=1024 is owned by proc0      
    //
    // stateBoundaryConID has the exactly same structure as stateBoundaryCon except that 
    // stateBoundaryConID stores the connected stateID instead of connected levels
    // stateBoundaryConID will be used in addBoundaryFaceConnections                  
    

    //labelList idxJ(adjIdx_.adjStateNames.size());

    MatCreate(PETSC_COMM_WORLD,&stateBoundaryCon_);
    MatSetSizes(stateBoundaryCon_,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(stateBoundaryCon_);
    MatMPIAIJSetPreallocation(stateBoundaryCon_,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(stateBoundaryCon_,1000,NULL);
    MatSetOption(stateBoundaryCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(stateBoundaryCon_);

    Mat stateBoundaryConTmp;
    MatCreate(PETSC_COMM_WORLD,&stateBoundaryConTmp);
    MatSetSizes(stateBoundaryConTmp,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(stateBoundaryConTmp);
    MatMPIAIJSetPreallocation(stateBoundaryConTmp,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(stateBoundaryConTmp,1000,NULL);
    MatSetOption(stateBoundaryConTmp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(stateBoundaryConTmp);

    // loop over the patches and set the boundary connnectivity
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // This cell is already a neighbour cell, so we need this plus two
                // more levels

                // Add connectivity in reverse so that the nearer stencils take
                // priority

                // Start with next to nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    forAll(adjIdx_.adjStateNames,idxI)
                    {
                        word stateName = adjIdx_.adjStateNames[idxI];
                        if(adjIdx_.adjStateType[stateName] != "surfaceScalarState")
                        {
                            // Now add level 3 connectivity, add all vars except for surfaceScalarStates
                            this->addConMatNeighbourCells(stateBoundaryCon_,gRow,localCell,stateName,3.0);
                            this->addConMatNeighbourCells(stateBoundaryConTmp,gRow,localCell,stateName,3.0);
                        }   
                    }
                }

                // now add the nearest neighbour cells, add all vars for level 2 except for surfaceScalarStates
                forAll(adjIdx_.adjStateNames,idxI)
                {
                    word stateName = adjIdx_.adjStateNames[idxI];
                    if(adjIdx_.adjStateType[stateName] != "surfaceScalarState")
                    {
                        this->addConMatNeighbourCells(stateBoundaryCon_,gRow,idxN,stateName,2.0);
                        this->addConMatNeighbourCells(stateBoundaryConTmp,gRow,idxN,stateName,2.0);
                    }
                }

                // and add the surfaceScalarStates for idxN
                forAll(adjReg_.surfaceScalarStates,idxI)
                {
                    const word& stateName = adjReg_.surfaceScalarStates[idxI];
                    this->addConMatCellFaces(stateBoundaryCon_,gRow,idxN,stateName,10.0); // for faces, its connectivity level is 10
                    this->addConMatCellFaces(stateBoundaryConTmp,gRow,idxN,stateName,10.0);
                }
          
                // Add all the cell states for idxN
                forAll(adjIdx_.adjStateNames,idxI)
                {
                    word stateName = adjIdx_.adjStateNames[idxI];
                    if(adjIdx_.adjStateType[stateName] != "surfaceScalarState")
                    {
                        this->addConMatCell(stateBoundaryCon_,gRow,idxN,stateName,1.0);
                        this->addConMatCell(stateBoundaryConTmp,gRow,idxN,stateName,1.0);
                    }
                }

            }
        }
    }

    MatAssemblyBegin(stateBoundaryCon_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryCon_,MAT_FINAL_ASSEMBLY);
    
    // Now repeat loop adding boundary connections from other procs using matrix
    // created in the first loop.
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // This cell is already a neighbour cell, so we need this plus two
                // more levels

                // Add connectivity in reverse so that the nearer stencils take
                // priority

                // Start with nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1={3};
                    // pass a zero list to add all states
                    List< List<word> > connectedStates(0);
                    this->addBoundaryFaceConnections(stateBoundaryConTmp,gRow,localCell,val1,connectedStates,0);  
                }

                // now add the neighbour cells
                labelList vals2= {2,3};
                // pass a zero list to add all states
                List< List<word> > connectedStates(0); 
                this->addBoundaryFaceConnections(stateBoundaryConTmp,gRow,idxN,vals2,connectedStates,0);

            }
        }
    }

    MatAssemblyBegin(stateBoundaryConTmp,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp,MAT_FINAL_ASSEMBLY);

    // the above repeat loop is not enough to cover all the stencil, we need to do more
    this->combineStateBndCon(&stateBoundaryCon_,&stateBoundaryConTmp);
    
    return;
    

}


void AdjointJacobianConnectivity::setupStateCyclicAMICon()
{
    // This function calculates stateCyclicAMICon and stateCyclicAMIConID
    // stateCyclicAMICon stores the level of connected states (on the other side of the cyclicAMI faces) for a
    // given AMI face boundary face. stateCyclicAMICon is a matrix with sizes of nCyclicAMIFaces by nGlobalAdjointStates
    // stateCyclicAMICon is mainly used in the addCyclicAMIFaceConnection function
    
    // Basically, if there are 2 levels of connected states on the other side of the cyclicAMI boundary
    //                                         
    // |-----------------------------------------|  <- cyclicAMI patch
    // |globalAdjStateIdx=100 ->   | lv1 |       |  
    // |                           |_____|       |
    // |globalAdjStateIdx=200 ->   | lv2 |       |
    // |                           |_____|       |
    // |                                         |
    // |   simulation domain                     |
    // |                           |-----|       |
    // |                           | cell|       |
    // |-----------------------------------------|  <- cyclicAMI patch
    //                                |
    //                                |
    //                           amiFaceIndex=1024  
    // The indices for row 1024 in the stateCyclicAMICon matrix will be
    // stateCyclicAMICon
    // rowI=1024   
    // Cols: colI=0 .......... colI=100  ............ colI=200 .............  colI=nGlobalAdjointStates
    // Vals:                       1                     2                
    //
    // stateCyclicAMIConID has the exactly same structure as stateCyclicAMICon except that 
    // stateCyclicAMIConID stores the connected stateID instead of connected levels
    // stateCyclicAMIConID will be used in addCyclicAMIFaceConnections   

    // create a local matrix 
    MatCreateSeqAIJ(PETSC_COMM_SELF,adjIdx_.nLocalCyclicAMIFaces,adjIdx_.nGlobalAdjointStates,1000,NULL,&stateCyclicAMICon_);
    MatSetUp(stateCyclicAMICon_);
    MatZeroEntries(stateCyclicAMICon_);     

    Mat stateCyclicAMIConTmp;
    MatCreateSeqAIJ(PETSC_COMM_SELF,adjIdx_.nLocalCyclicAMIFaces,adjIdx_.nGlobalAdjointStates,1000,NULL,&stateCyclicAMIConTmp);
    MatSetUp(stateCyclicAMIConTmp);
    MatZeroEntries(stateCyclicAMIConTmp);               

    // loop over the patches and set the boundary connnectivity
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    // level 3
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is cyclicAMI
        if (pp.type() == "cyclicAMI")
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                labelList gRowAMI = cyclicAMIBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // Start with next to nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    forAll(adjIdx_.adjStateNames,idxI)
                    {
                        word stateName = adjIdx_.adjStateNames[idxI];
                        if(adjIdx_.adjStateType[stateName] != "surfaceScalarState")
                        {
                            forAll(gRowAMI,aa)
                            {
                                // Now add level 3 connectivity, add all vars except for surfaceScalarStates
                                this->addConMatNeighbourCells(stateCyclicAMICon_,gRowAMI[aa],localCell,stateName,3.0);
                                this->addConMatNeighbourCells(stateCyclicAMIConTmp,gRowAMI[aa],localCell,stateName,3.0);
                            }
                        }   
                    }
                }
            }
        }
    }

    // level 2
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is cyclicAMI
        if (pp.type() == "cyclicAMI")
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                labelList gRowAMI = cyclicAMIBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // now add the nearest neighbour cells, add all vars for level 2 except for surfaceScalarStates
                forAll(adjIdx_.adjStateNames,idxI)
                {
                    word stateName = adjIdx_.adjStateNames[idxI];
                    if(adjIdx_.adjStateType[stateName] != "surfaceScalarState")
                    {
                        forAll(gRowAMI,aa)
                        {
                            this->addConMatNeighbourCells(stateCyclicAMICon_,gRowAMI[aa],idxN,stateName,2.0);
                            this->addConMatNeighbourCells(stateCyclicAMIConTmp,gRowAMI[aa],idxN,stateName,2.0);
                        }
                    }
                }
            }
        }
    }

    // level 1
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is cyclicAMI
        if (pp.type() == "cyclicAMI")
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                labelList gRowAMI = cyclicAMIBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // and add the surfaceScalarStates for idxN
                forAll(adjReg_.surfaceScalarStates,idxI)
                {
                    const word& stateName = adjReg_.surfaceScalarStates[idxI];
                    forAll(gRowAMI,aa)
                    {
                        // for faces, its connectivity level is 10
                        this->addConMatCellFaces(stateCyclicAMICon_,gRowAMI[aa],idxN,stateName,10.0); 
                        this->addConMatCellFaces(stateCyclicAMIConTmp,gRowAMI[aa],idxN,stateName,10.0); 
                    }

                }
          
                // Add all the cell states for idxN
                forAll(adjIdx_.adjStateNames,idxI)
                {
                    word stateName = adjIdx_.adjStateNames[idxI];
                    if(adjIdx_.adjStateType[stateName] != "surfaceScalarState")
                    {
                        forAll(gRowAMI,aa)
                        {
                            this->addConMatCell(stateCyclicAMICon_,gRowAMI[aa],idxN,stateName,1.0);
                            this->addConMatCell(stateCyclicAMIConTmp,gRowAMI[aa],idxN,stateName,1.0);
                        }

                    }
                }
            }
        }
    }

    MatAssemblyBegin(stateCyclicAMIConTmp,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateCyclicAMIConTmp,MAT_FINAL_ASSEMBLY);

    // Now repeat loop adding boundary connections from other procs using matrix
    // created in the first loop.
    // level 3
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.type() == "cyclicAMI")
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                labelList gRowAMI = cyclicAMIBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // Start with nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1={3};
                    // pass a zero list to add all states
                    List< List<word> > connectedStates(0);
                    forAll(gRowAMI,aa)
                    {
                        this->addBoundaryFaceConnections(stateCyclicAMICon_,gRowAMI[aa],localCell,val1,connectedStates,0); 
                    }
                }
            }
        }
    }

    // level 2
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.type() == "cyclicAMI")
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                labelList gRowAMI = cyclicAMIBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // now add the neighbour cells
                labelList vals2= {2,3};
                // pass a zero list to add all states
                List< List<word> > connectedStates(0); 
                forAll(gRowAMI,aa)
                {
                    this->addBoundaryFaceConnections(stateCyclicAMICon_,gRowAMI[aa],idxN,vals2,connectedStates,0);
                }
            }
        }
    }

    // level 2 again, because the previous call will mess up level 2 con
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.type() == "cyclicAMI")
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                labelList gRowAMI = cyclicAMIBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // now add the neighbour cells
                labelList vals1= {2};
                // pass a zero list to add all states
                List< List<word> > connectedStates(0); 
                forAll(gRowAMI,aa)
                {
                    this->addBoundaryFaceConnections(stateCyclicAMICon_,gRowAMI[aa],idxN,vals1,connectedStates,0);
                }
            }
        }
    }

    MatAssemblyBegin(stateCyclicAMICon_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateCyclicAMICon_,MAT_FINAL_ASSEMBLY);
 
    // Now stateCyclicAMICon_ will have all the missing stencil. However, it will also mess up the existing stencil
    // in stateCyclicAMIConTmp. So we need to do a check to make sure that stateCyclicAMICon_ only add stencil, not replacing
    // any existing stencil in stateCyclicAMIConTmp. If anything in stateCyclicAMIConTmp is replaced, rollback the changes.

    PetscInt    nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt    nCols1;
    const PetscInt    *cols1;
    const PetscScalar *vals1;
    label Istart, Iend;
    MatGetOwnershipRange(stateCyclicAMICon_,&Istart,&Iend);
    
    Mat tmpMat; // create a temp mat
    MatCreateSeqAIJ(PETSC_COMM_SELF,adjIdx_.nLocalCyclicAMIFaces,adjIdx_.nGlobalAdjointStates,1000,NULL,&tmpMat);
    MatSetUp(tmpMat);  
    MatZeroEntries(tmpMat); // initialize with zeros
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(stateCyclicAMIConTmp,i,&nCols,&cols,&vals);
        MatGetRow(stateCyclicAMICon_,i,&nCols1,&cols1,&vals1);
        for (PetscInt j=0; j<nCols1; j++)
        {
            // for each col in stateBoundaryConTmp, we need to check if there are any existing values for the same 
            // col in stateBoundaryCon. If yes, assign the val from stateBoundaryCon instead of stateBoundaryConTmp
            PetscScalar newVal = vals1[j];
            PetscInt newCol = cols1[j];
            for (PetscInt k=0; k<nCols; k++)
            {
                if ( int(cols[k]) == int(cols1[j]) )
                {
                    newVal = vals[k];
                    newCol = cols[k];
                    break;
                }
            }
            MatSetValue(tmpMat,i,newCol,newVal,INSERT_VALUES);
        }
        MatRestoreRow(stateCyclicAMIConTmp,i,&nCols,&cols,&vals);
        MatRestoreRow(stateCyclicAMICon_,i,&nCols1,&cols1,&vals1);
    }
    MatAssemblyBegin(tmpMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmpMat,MAT_FINAL_ASSEMBLY);

    // copy ConMat to ConMatTmp
    MatDestroy(&stateCyclicAMICon_);
    MatConvert(tmpMat, MATSAME,MAT_INITIAL_MATRIX,&stateCyclicAMICon_);
    MatAssemblyBegin(stateCyclicAMICon_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateCyclicAMICon_,MAT_FINAL_ASSEMBLY);
        

    return;
    

}

void AdjointJacobianConnectivity::combineAllStateCons()
{
    // now we need to add the missing connectivity in stateBoundaryCon mat,
    // this is because stateBoundaryCon may have connectivity for e.g., cyclicAMI patches
    // this connectivity has not been added!
    // Note the stateCyclicAMI will also have connectivty for stateBoundaryCon, but it has been added 
    // in the  AdjointJacobianConnectivity::setupStateCyclicAMICon() function

    // Now repeat loop adding boundary connections from other procs using matrix
    // created in the first loop.
    // level 3
    // Now repeat loop adding boundary connections from other procs using matrix
    // created in the first loop.

    Mat stateBoundaryConTmp;
    MatCreate(PETSC_COMM_WORLD,&stateBoundaryConTmp);
    MatSetSizes(stateBoundaryConTmp,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(stateBoundaryConTmp);
    MatMPIAIJSetPreallocation(stateBoundaryConTmp,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(stateBoundaryConTmp,1000,NULL);
    MatSetOption(stateBoundaryConTmp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(stateBoundaryConTmp);

    MatConvert(stateBoundaryCon_, MATSAME,MAT_INITIAL_MATRIX,&stateBoundaryConTmp);

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    // level 3
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // Start with nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1={3};
                    // pass a zero list to add all states
                    List< List<word> > connectedStates(0);
                    this->addCyclicAMIFaceConnections(stateBoundaryConTmp,gRow,localCell,val1,connectedStates,0);  
                }
            }
        }
    }

    // level 2 and 3
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // now add the neighbour cells
                labelList vals2= {2,3};
                // pass a zero list to add all states
                List< List<word> > connectedStates(0); 
                this->addCyclicAMIFaceConnections(stateBoundaryConTmp,gRow,idxN,vals2,connectedStates,0);

            }
        }
    }

    // level 2 again
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];
                
                // now add the neighbour cells
                labelList vals2= {2};
                // pass a zero list to add all states
                List< List<word> > connectedStates(0); 
                this->addCyclicAMIFaceConnections(stateBoundaryConTmp,gRow,idxN,vals2,connectedStates,0);

            }
        }
    }

    MatAssemblyBegin(stateBoundaryConTmp,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp,MAT_FINAL_ASSEMBLY);


    MatDestroy(&stateBoundaryCon_);
    MatConvert(stateBoundaryConTmp, MATSAME,MAT_INITIAL_MATRIX,&stateBoundaryCon_);
    MatAssemblyBegin(stateBoundaryCon_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryCon_,MAT_FINAL_ASSEMBLY);

    return;

}

void AdjointJacobianConnectivity::setupStateBoundaryConID()
{
    /*
        This function computes stateBoundaryConID_.
        stateBoundaryConID has the exactly same structure as stateBoundaryCon except that 
        stateBoundaryConID stores the connected stateID instead of connected levels
        stateBoundaryConID will be used in addBoundaryFaceConnections
    */

    PetscInt nCols, colI;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt Istart,Iend;

    PetscScalar valIn;

    // assemble adjStateID4GlobalAdjIdx
    // adjStateID4GlobalAdjIdx stores the adjStateID for given a global adj index
    labelList adjStateID4GlobalAdjIdx;
    adjStateID4GlobalAdjIdx.setSize(adjIdx_.nGlobalAdjointStates);
    adjIdx_.calcAdjStateID4GlobalAdjIdx(adjStateID4GlobalAdjIdx);

    // initialize
    MatCreate(PETSC_COMM_WORLD,&stateBoundaryConID_);
    MatSetSizes(stateBoundaryConID_,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(stateBoundaryConID_);
    MatMPIAIJSetPreallocation(stateBoundaryConID_,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(stateBoundaryConID_,1000,NULL);
    MatSetOption(stateBoundaryConID_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(stateBoundaryConID_);
    MatZeroEntries(stateBoundaryConID_);

    MatGetOwnershipRange(stateBoundaryCon_,&Istart,&Iend);

    // set stateBoundaryConID_ based on stateBoundaryCon_ and adjStateID4GlobalAdjIdx
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(stateBoundaryCon_,i,&nCols,&cols,&vals);
        for (PetscInt j=0; j<nCols; j++)
        {
            if ( !adjIO_.isValueCloseToRef(vals[j],0.0) )
            {
                colI=cols[j];
                valIn = adjStateID4GlobalAdjIdx[colI];
                MatSetValue(stateBoundaryConID_,i,colI,valIn,INSERT_VALUES);
            }
        }
        MatRestoreRow(stateBoundaryCon_,i,&nCols,&cols,&vals);
    }

    adjStateID4GlobalAdjIdx.clear();

    MatAssemblyBegin(stateBoundaryConID_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConID_,MAT_FINAL_ASSEMBLY);

    return;

}


void AdjointJacobianConnectivity::setupStateCyclicAMIConID()
{
    /*
        This function computes stateCyclicAMIConID_.
        stateCyclicAMIConID_ has the exactly same structure as stateCyclicAMICon except that 
        stateCyclicAMIConID_ stores the connected stateID instead of connected levels
        stateCyclicAMIConID_ will be used in addCyclicAMIFaceConnections
    */

    PetscInt nCols, colI;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt Istart,Iend;

    PetscScalar valIn;

    // assemble adjStateID4GlobalAdjIdx
    // adjStateID4GlobalAdjIdx stores the adjStateID for given a global adj index
    labelList adjStateID4GlobalAdjIdx;
    adjStateID4GlobalAdjIdx.setSize(adjIdx_.nGlobalAdjointStates);
    adjIdx_.calcAdjStateID4GlobalAdjIdx(adjStateID4GlobalAdjIdx);

    // initialize
    MatCreateSeqAIJ(PETSC_COMM_SELF,adjIdx_.nLocalCyclicAMIFaces,adjIdx_.nGlobalAdjointStates,1000,NULL,&stateCyclicAMIConID_);
    MatSetUp(stateCyclicAMIConID_);  
    MatZeroEntries(stateCyclicAMIConID_); // initialize with zeros

    MatGetOwnershipRange(stateCyclicAMIConID_,&Istart,&Iend);

    // set stateCyclicAMIConID_ based on stateCyclicAMICon_ and adjStateID4GlobalAdjIdx
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(stateCyclicAMICon_,i,&nCols,&cols,&vals);
        for (PetscInt j=0; j<nCols; j++)
        {
            if ( !adjIO_.isValueCloseToRef(vals[j],0.0) )
            {
                colI=cols[j];
                valIn = adjStateID4GlobalAdjIdx[colI];
                MatSetValue(stateCyclicAMIConID_,i,colI,valIn,INSERT_VALUES);
            }
        }
        MatRestoreRow(stateCyclicAMICon_,i,&nCols,&cols,&vals);
    }

    adjStateID4GlobalAdjIdx.clear();

    MatAssemblyBegin(stateCyclicAMIConID_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateCyclicAMIConID_,MAT_FINAL_ASSEMBLY);

    return;

}

void AdjointJacobianConnectivity::setupXvBoundaryCon()
{
    // This function calculates stateBoundaryCon_
    // xvBoundaryCon stores the level of connected Xv (on the other side across the boundary) for a
    // given coupled boundary face. xvBoundaryCon is a matrix with sizes of nGlobalCoupledBFaces by nGlobalXv
    // xvBoundaryCon is mainly used in the addBoundaryFaceConnectionsXv function           
    

    MatCreate(PETSC_COMM_WORLD,&xvBoundaryCon_);
    MatSetSizes(xvBoundaryCon_,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalXv,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(xvBoundaryCon_);
    MatMPIAIJSetPreallocation(xvBoundaryCon_,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(xvBoundaryCon_,1000,NULL);
    MatSetOption(xvBoundaryCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(xvBoundaryCon_);

    Mat xvBoundaryConTmp;
    MatCreate(PETSC_COMM_WORLD,&xvBoundaryConTmp);
    MatSetSizes(xvBoundaryConTmp,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalXv,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(xvBoundaryConTmp);
    MatMPIAIJSetPreallocation(xvBoundaryConTmp,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(xvBoundaryConTmp,1000,NULL);
    MatSetOption(xvBoundaryConTmp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(xvBoundaryConTmp);

    // loop over the patches and set the boundary connnectivity
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {

            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // This cell is already a neighbour cell, so we need this plus two
                // more levels

                // Add connectivity in reverse so that the nearer stencils take
                // priority

                // Start with next to nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    this->addConMatNeighbourCellsXv(xvBoundaryCon_,gRow,localCell,3.0);
                    this->addConMatNeighbourCellsXv(xvBoundaryConTmp,gRow,localCell,3.0);
                }

                // now add the nearest neighbour cells
                this->addConMatNeighbourCellsXv(xvBoundaryCon_,gRow,idxN,2.0);
                this->addConMatNeighbourCellsXv(xvBoundaryConTmp,gRow,idxN,2.0);

                // Now add the points of idxN itself
                this->addConMatCellXv(xvBoundaryCon_,gRow,idxN,1.0);
                this->addConMatCellXv(xvBoundaryConTmp,gRow,idxN,1.0);
            }
        }
    }

    MatAssemblyBegin(xvBoundaryCon_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(xvBoundaryCon_,MAT_FINAL_ASSEMBLY);

    // Now repeat loop adding boundary connections from other procs using matrix
    // created in the first loop.
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        // get the start index of this patch in the global face list
        label faceIStart = pp.start();

        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {

            forAll(pp, faceI)
            {
                // get the necessary matrix row
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // This cell is already a neighbour cell, so we need this plus two
                // more levels

                // Add connectivity in reverse so that the nearer stencils take
                // priority

                // Start with nearest neighbours
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1={3};
                    this->addBoundaryFaceConnectionsXv(xvBoundaryConTmp,gRow,localCell,val1);
                }

                // now add the neighbour cells
                labelList vals2= {2,3};
                this->addBoundaryFaceConnectionsXv(xvBoundaryConTmp,gRow,idxN,vals2);
            }
        }
    }
    
    // Assemble the final matrix
    MatAssemblyBegin(xvBoundaryConTmp,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(xvBoundaryConTmp,MAT_FINAL_ASSEMBLY);

    // the above repeat loop is not enough to cover all the stencil, we need to do more 
    this->combineXvBndCon(&xvBoundaryCon_,&xvBoundaryConTmp);

    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(xvBoundaryCon_,"xvBoundaryCon");
    }
    
    return;
    

}


void AdjointJacobianConnectivity::addBoundaryFaceConnections
(
    Mat conMat,
    label gRow, 
    label cellI,
    labelList v,
    List< List<word> > connectedStates,
    label addFaces
)
{
    // This function adds inter-proc connectivity into conMat.
    // For all the inter-proc faces owned by cellI, get the global adj state indices from stateBoundaryCon
    // and then add them into conMat
    // Row index to add: gRow
    // Col index to add: the same col index for a given row (bRowGlobal) in the stateBoundaryCon mat if the
    // element value in the stateBoundaryCon mat is less than the input level, i.e., v.size().
    // v: an array denoting the desired values to add, the size of v denotes the maximal levels to add
    // connectedStates: selectively add some states into the conMat for the current level. If its size is 0,
    // add all the possible states (except for surfaceStates). The dimension of connectedStates is nLevel by nStates. 
    //
    // Example:
    // 
    // labelList val2={1,2};
    // PetscInt gRow=1024, idxN = 100, addFaces=1;
    // wordListList connectedStates={{"U","p"},{"U"}}; 
    // addBoundaryFaceConnections(stateBoundaryCon,gRow,idxN,vals2,connectedStates,addFaces);
    // The above call will add 2 levels of connected states for all the inter-proc faces belonged to cellI=idxN 
    // The cols to added are: the level1 connected states (U, p) for all the inter-proc faces belonged to cellI=idxN 
    // the level2 connected states (U only) for all the inter-proc faces belonged to cellI=idxN 
    // The valus "1" will be added to conMat for all the level1 connected states while the value "2" will be added for level2
    // Note: this function will also add all faces belonged to level1 of the inter-proc faces, see the following for reference
    //              
    //                             _______
    //                             | idxN|
    //                             |     |         proc0, idxN=100, globalBFaceI=1024 for the south face of idxN
    //                        -----------------  <-coupled boundary face
    //  add state U and p     ->   | lv1 |         proc1
    //  also add faces        ->   |_____|
    //                             | lv2 |
    //  add state U           ->   |_____| 
    
    if(v.size()!=connectedStates.size() && connectedStates.size()!=0)
    {
        FatalErrorIn("")<<"size of v and connectedStates are not identical!"<<abort(FatalError);
    }
    

    PetscInt  idxJ,idxI,bRow,bRowGlobal;
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscInt nColsID;
    const PetscInt    *colsID;
    const PetscScalar *valsID;
    
    // convert stateNames to stateIDs
    labelListList connectedStateIDs(connectedStates.size());
    forAll(connectedStates,idxI)
    {
        forAll(connectedStates[idxI],idxJ)
        {
            word stateName = connectedStates[idxI][idxJ];
            label stateID = adjIdx_.adjStateID[stateName];
            connectedStateIDs[idxI].append(stateID);
        }
    }

    idxI = gRow;
    // get the faces connected to this cell, note these are in a single
    // list that includes all internal and boundary faces
    const labelList& faces = mesh_.cells()[cellI];

    //get the level
    label level = v.size();

    for (label lv=level; lv>=1; lv--) // we need to start from the largest levels since they have higher priority
    {
        forAll(faces,faceI)
        {
            // Now deal with coupled faces
            label currFace = faces[faceI];
            
            if(adjIdx_.isCoupledFace[currFace])
            {
                //this is a coupled face
    
                // use the boundary connectivity to figure out what is connected
                // to this face for this level
    
                // get bRow in boundaryCon for this face
                bRow=this->getLocalCoupledBFaceIndex(currFace);

                // get the global bRow index
                bRowGlobal = adjIdx_.globalCoupledBFaceNumbering.toGlobal(bRow);
    
                // now extract the boundaryCon row
                MatGetRow(stateBoundaryCon_, bRowGlobal,&nCols,&cols,&vals);
                if (connectedStates.size()!=0)
                {
                    // check if we need to get stateID
                    MatGetRow(stateBoundaryConID_, bRowGlobal,&nColsID,&colsID,&valsID);
                }

                // now loop over the row and set any column that match this level
                // in conMat
                for(label i=0; i<nCols; i++)
                {
                    idxJ = cols[i];
                    label val = round(vals[i]); // val is the connectivity level extracted from stateBoundaryCon_ at this col
                    // selectively add some states into conMat
                    label addState;
                    label stateID=-9999;
                    // check if we need to get stateID
                    if (connectedStates.size()!=0) stateID= round(valsID[i]);

                    if (connectedStates.size()==0) addState=1;
                    else if (adjIO_.isInList<label>(stateID,connectedStateIDs[lv-1]) ) addState=1;
                    else addState =0;
                    // if the level match and the state is what you want
                    if( val==lv && addState )
                    {
                        // need to do v[lv-1] here since v is an array with starting index 0
                        PetscScalar valIn = v[lv-1];
                        MatSetValues(conMat,1,&idxI,1,&idxJ,&valIn,INSERT_VALUES);
                    }
                    if( val==10 && addFaces)
                    {
                        // this is a necessary connection
                        PetscScalar valIn = v[lv-1];
                        MatSetValues(conMat,1,&idxI,1,&idxJ,&valIn,INSERT_VALUES);
                    }
                }
    
                // restore the row of the matrix
                MatRestoreRow(stateBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);
                if (connectedStates.size()!=0)
                {
                    // check if we need to get stateID
                    MatRestoreRow(stateBoundaryConID_,bRowGlobal,&nColsID,&colsID,&valsID);
                }
            }

        }
        
    }
    
    return;
}


void AdjointJacobianConnectivity::addCyclicAMIFaceConnections
(
    Mat conMat,
    label gRow, 
    label cellI,
    labelList v,
    List< List<word> > connectedStates,
    label addFaces
)
{
    // This function adds AMI connectivity into conMat.
    // For all the AMI faces owned by cellI, get the global adj state indices from stateCyclicAMICon
    // and then add them into conMat
    // Row index to add: gRow
    // Col index to add: the same col index for a given row (bRow) in the stateCyclicAMICon mat if the
    // element value in the stateCyclicAMICon mat is less than the input level, i.e., v.size().
    // v: an array denoting the desired values to add, the size of v denotes the maximal levels to add
    // connectedStates: selectively add some states into the conMat for the current level. If its size is 0,
    // add all the possible states (except for surfaceStates). The dimension of connectedStates is nLevel by nStates. 
    //
    // Example:
    // 
    // labelList val2={1,2};
    // PetscInt gRow=1024, idxN = 100, addFaces=1;
    // wordListList connectedStates={{"U","p"},{"U"}}; 
    // addCyclicAMIFaceConnections(conMat,gRow,idxN,vals2,connectedStates,addFaces);
    // The above call will add 2 levels of connected states for all the AMI faces belonged to cellI=idxN 
    // The cols to added are: the level1 connected states (U, p) for all the AMI faces belonged to cellI=idxN 
    // the level2 connected states (U only) for all the AMI faces belonged to cellI=idxN 
    // The valus "1" will be added to conMat for all the level1 connected states while the value "2" will be added for level2
    // Note: this function will also add all faces belonged to level1 of the AMI faces, see the following for reference
    //              
    //                             _______
    //                             | idxN|
    //                             |     |         proc0, idxN=100, globalBFaceI=1024 for the south face of idxN
    //                        -----------------  <-AMI coupled boundary face
    //  add state U and p     ->   | lv1 |         proc1
    //  also add faces        ->   |_____|
    //                             | lv2 |
    //  add state U           ->   |_____| 

    if(!procHasAMI_) return;
    
    if(v.size()!=connectedStates.size() && connectedStates.size()!=0)
    {
        FatalErrorIn("")<<"size of v and connectedStates are not identical!"<<abort(FatalError);
    }
    

    PetscInt  idxJ,idxI,bRow;
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscInt nColsID;
    const PetscInt    *colsID;
    const PetscScalar *valsID;
    
    // convert stateNames to stateIDs
    labelListList connectedStateIDs(connectedStates.size());
    forAll(connectedStates,idxI)
    {
        forAll(connectedStates[idxI],idxJ)
        {
            word stateName = connectedStates[idxI][idxJ];
            label stateID = adjIdx_.adjStateID[stateName];
            connectedStateIDs[idxI].append(stateID);
        }
    }

    idxI = gRow;
    // get the faces connected to this cell, note these are in a single
    // list that includes all internal and boundary faces
    const labelList& faces = mesh_.cells()[cellI];

    //get the level
    label level = v.size();

    for (label lv=level; lv>=1; lv--) // we need to start from the largest levels since they have higher priority
    {
        forAll(faces,faceI)
        {
            // Now deal with coupled faces
            label currFace = faces[faceI];
            
            if(adjIdx_.isCyclicAMIFace[currFace])
            {
                //this is a coupled face
    
                // use the boundary connectivity to figure out what is connected
                // to this face for this level
    
                // get bRow in boundaryCon for this face
                bRow=this->getLocalCyclicAMIFaceIndex(currFace);
    
                // now extract the boundaryCon row
                MatGetRow(stateCyclicAMICon_, bRow,&nCols,&cols,&vals);
                if (connectedStates.size()!=0)
                {
                    // check if we need to get stateID
                    MatGetRow(stateCyclicAMIConID_, bRow,&nColsID,&colsID,&valsID);
                }

                // now loop over the row and set any column that match this level
                // in conMat
                for(label i=0; i<nCols; i++)
                {
                    idxJ = cols[i];
                    label val = round(vals[i]); // val is the connectivity level extracted from stateBoundaryCon_ at this col
                    // selectively add some states into conMat
                    label addState;
                    label stateID=-9999;
                    // check if we need to get stateID
                    if (connectedStates.size()!=0) stateID= round(valsID[i]);

                    if (connectedStates.size()==0) addState=1;
                    else if (adjIO_.isInList<label>(stateID,connectedStateIDs[lv-1]) ) addState=1;
                    else addState =0;
                    // if the level match and the state is what you want
                    if( val==lv && addState )
                    {
                        // need to do v[lv-1] here since v is an array with starting index 0
                        PetscScalar valIn = v[lv-1];
                        MatSetValues(conMat,1,&idxI,1,&idxJ,&valIn,INSERT_VALUES);
                    }
                    if( val==10 && addFaces)
                    {
                        // this is a necessary connection
                        PetscScalar valIn = v[lv-1];
                        MatSetValues(conMat,1,&idxI,1,&idxJ,&valIn,INSERT_VALUES);
                    }
                }
    
                // restore the row of the matrix
                MatRestoreRow(stateCyclicAMICon_,bRow,&nCols,&cols,&vals);
                if (connectedStates.size()!=0)
                {
                    // check if we need to get stateID
                    MatRestoreRow(stateCyclicAMIConID_,bRow,&nColsID,&colsID,&valsID);
                }
            }

        }
        
    }
    
    return;
}


void AdjointJacobianConnectivity::addBoundaryFaceConnectionsXv
(
    Mat conMat,
    label gRow,
    label cellI,
    labelList v
)
{
    // This function adds inter-proc Xv connectivity into conMat.
    // For all the inter-proc faces owned by cellI, get the global point coordinate indices from the xvBoundaryCon
    // and then add them into conMat
    // Row index to add: gRow
    // Col index to add: the same col index for a given row (bRowGlobal) in the xvBoundaryCon
    // if the element value in the xvBoundaryCon mat is less than the input level
    // Value to add: v, a given array denoting the desired connectivity level
    
    // bRow: local row index of the interprocessor boundary faces (see globalBndNumbering_)
    // bRow=0 is the first interprocessor boundary face for a local proc

    PetscInt  idxJ,idxI,bRow,bRowGlobal;
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    
    label level = v.size();
    idxI = gRow;

    // get the faces connected to this cell, note these are in a single
    // list that includes all internal and boundary faces
    const labelList& faces = mesh_.cells()[cellI];

    //get the boundary patches
    //const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    // we need to start with the largest level so that the nearer level takes priority
    // and we need to make it the outermost loop!
    for (label lv=level; lv>=1; lv--)
    {
        forAll(faces,faceI)
        {
            // Now deal with coupled faces
            label currFace = faces[faceI];

            if(adjIdx_.isCoupledFace[currFace])
            {
                //this is a coupled face

                // use the boundary connectivity to figure out what is connected
                // to this face for this level

                // get bRow in boundaryCon for this face
                bRow=this->getLocalCoupledBFaceIndex(currFace);

                // get the global bRow index
                bRowGlobal = adjIdx_.globalCoupledBFaceNumbering.toGlobal(bRow);

                // now extract the boundaryCon row
                MatGetRow(xvBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);

                // Now loop over the row and set any column that match this level
                for(label i=0; i<nCols; i++)
                {
                    idxJ = cols[i];
                    label val = round(vals[i]);
                    if(val==lv)
                    {
                        // this is a necessary connection
                        PetscScalar valIn = v[lv-1];
                        MatSetValues(conMat,1,&idxI,1,&idxJ,&valIn,INSERT_VALUES);
                    }

                }
                // restore the row of the matrix
                MatRestoreRow(xvBoundaryCon_,bRowGlobal,&nCols,&cols,&vals);
            }
        }
    }

    return;

}


PetscInt AdjointJacobianConnectivity::getLocalCoupledBFaceIndex(label localFaceI)
{

    
    //Calculate the index of the local inter-processor boundary face (bRow). bRow is in
    //a list of faces starts with the first inter-processor face. See globalBndNumbering_
    //for more details.
    //localFaceI: The local face index. It is in a list of faces including all the
    //internal and boundary faces.
    

    label counter=0;
    forAll(mesh_.boundaryMesh(), patchI)
    {
        const polyPatch& pp = mesh_.boundaryMesh()[patchI];
        // check whether this face is coupled (cyclic or processor?)
        if (pp.coupled())
        {
            // get the start index of this patch in the global face
            // list and the size of this patch.
            label faceStart = pp.start();
            label patchSize = pp.size();
            label faceEnd = faceStart+patchSize;
            if(localFaceI>=faceStart && localFaceI < faceEnd)
            {
                // this face is on this patch, find the exact index
                label countDelta = localFaceI-pp.start();//-faceStart;
                PetscInt bRow = counter+countDelta;
                return bRow;
            }
            else
            {
                //increment the counter by patchSize
                counter +=patchSize;
            }
        }
    }
    
    // no match found
    FatalErrorIn("getLocalBndFaceIndex")<<abort(FatalError);
    return -1;
}



PetscInt AdjointJacobianConnectivity::getLocalCyclicAMIFaceIndex(label localFaceI)
{
    
    //Calculate the index of the local cyclicAMI boundary face: amiIdx. amiIdx is in
    //a list of faces starts with the first cyclicAMI face.
    //localFaceI: The local face index. It is in a list of faces including all the
    //internal and boundary faces.

    label counter=0;
    forAll(mesh_.boundaryMesh(), patchI)
    {
        const polyPatch& pp = mesh_.boundaryMesh()[patchI];
        // check whether this face is coupled (cyclic or processor?)
        if (pp.type() == "cyclicAMI")
        {
            // get the start index of this patch in the global face
            // list and the size of this patch.
            label faceStart = pp.start();
            label patchSize = pp.size();
            label faceEnd = faceStart+patchSize;
            if(localFaceI>=faceStart && localFaceI < faceEnd)
            {
                // this face is on this patch, find the exact index
                label countDelta = localFaceI-pp.start();//-faceStart;
                PetscInt amiIdx = counter+countDelta;
                return amiIdx;
            }
            else
            {
                //increment the counter by patchSize
                counter +=patchSize;
            }
        }
    }
    
    // no match found
    FatalErrorIn("getLocalCyclicAMIFaceIndex")<<abort(FatalError);
    return -1;
}


void AdjointJacobianConnectivity::combineStateBndCon
(
    Mat* stateBoundaryCon,
    Mat* stateBoundaryConTmp
)
{
    /*
        1. Add additional adj state connectivities if the stateBoundaryCon stencil extends through
        three or more decomposed domains, something like this:
        
        --------       ---------
               |       |       |
          Con3 |  Con2 |  Con1 |  R
               |       |       |
               ---------       --------
               
        Here R is the residual, Con1 to 3 are its connectivity, and dashed lines 
        are the inter-processor boundary
               
        2. Assign stateBoundaryConTmp to stateBoundaryCon.
    */
    
    PetscInt    nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    
    PetscInt    nCols1;
    const PetscInt    *cols1;
    const PetscScalar *vals1;

    // Destroy and initialize stateBoundaryCon with zeros
    MatDestroy(stateBoundaryCon);
    MatCreate(PETSC_COMM_WORLD,stateBoundaryCon);
    MatSetSizes(*stateBoundaryCon,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(*stateBoundaryCon);
    MatMPIAIJSetPreallocation(*stateBoundaryCon,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(*stateBoundaryCon,1000,NULL);
    MatSetOption(*stateBoundaryCon, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*stateBoundaryCon);
    MatZeroEntries(*stateBoundaryCon); // initialize with zeros
    
    // assign stateBoundaryConTmp to stateBoundaryCon
    PetscInt Istart,Iend;
    MatGetOwnershipRange(*stateBoundaryConTmp,&Istart,&Iend);
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(*stateBoundaryConTmp,i,&nCols,&cols,&vals);
        for (PetscInt j=0; j<nCols; j++)
        {
            MatSetValue(*stateBoundaryCon,i,cols[j],vals[j],INSERT_VALUES);
        }
        MatRestoreRow(*stateBoundaryConTmp,i,&nCols,&cols,&vals);
    }
    MatAssemblyBegin(*stateBoundaryCon,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon,MAT_FINAL_ASSEMBLY);
    
    MatDestroy(stateBoundaryConTmp);

    // copy ConMat to ConMatTmp for an extract loop 
    MatConvert(*stateBoundaryCon, MATSAME,MAT_INITIAL_MATRIX,stateBoundaryConTmp);
    
    // We need to do another loop adding boundary connections from other procs using ConMat
    // this will add missing connectivity if the stateBoundaryCon stencil extends through
    // three more more processors 
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        label faceIStart = pp.start();
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];
                label idxN = pFaceCells[faceI];
                
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1={3};
                    // pass a zero list to add all states
                    List< List<word> > connectedStates(0);
                    this->addBoundaryFaceConnections(*stateBoundaryConTmp,gRow,localCell,val1,connectedStates,0);  

                }
                // now add the neighbour cells
                labelList vals2= {2,3};
                // pass a zero list to add all states
                List< List<word> > connectedStates(0); 
                this->addBoundaryFaceConnections(*stateBoundaryConTmp,gRow,idxN,vals2,connectedStates,0);

            }
        }
    }
    MatAssemblyBegin(*stateBoundaryConTmp,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryConTmp,MAT_FINAL_ASSEMBLY);
    
    // Now stateBoundaryConTmp will have all the missing stencil. However, it will also mess up the existing stencil
    // in stateBoundaryCon. So we need to do a check to make sure that stateBoundaryConTmp only add stencil, not replacing
    // any existing stencil in stateBoundaryCon. If anything in stateBoundaryCon is replaced, rollback the changes.
    Mat tmpMat; // create a temp mat
    MatCreate(PETSC_COMM_WORLD,&tmpMat);
    MatSetSizes(tmpMat,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(tmpMat);
    MatMPIAIJSetPreallocation(tmpMat,1000,NULL,1000,NULL);
    MatSeqAIJSetPreallocation(tmpMat,1000,NULL);
    MatSetOption(tmpMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(tmpMat);
    MatZeroEntries(tmpMat); // initialize with zeros
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(*stateBoundaryCon,i,&nCols,&cols,&vals);
        MatGetRow(*stateBoundaryConTmp,i,&nCols1,&cols1,&vals1);
        for (PetscInt j=0; j<nCols1; j++)
        {
            // for each col in stateBoundaryConTmp, we need to check if there are any existing values for the same 
            // col in stateBoundaryCon. If yes, assign the val from stateBoundaryCon instead of stateBoundaryConTmp
            PetscScalar newVal = vals1[j];
            PetscInt newCol = cols1[j];
            for (PetscInt k=0; k<nCols; k++)
            {
                if ( int(cols[k]) == int(cols1[j]) )
                {
                    newVal = vals[k];
                    newCol = cols[k];
                    break;
                }
            }
            MatSetValue(tmpMat,i,newCol,newVal,INSERT_VALUES);
        }
        MatRestoreRow(*stateBoundaryCon,i,&nCols,&cols,&vals);
        MatRestoreRow(*stateBoundaryConTmp,i,&nCols1,&cols1,&vals1);
    }
    MatAssemblyBegin(tmpMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmpMat,MAT_FINAL_ASSEMBLY);

    // copy ConMat to ConMatTmp
    MatDestroy(stateBoundaryCon);
    MatConvert(tmpMat, MATSAME,MAT_INITIAL_MATRIX,stateBoundaryCon);
    MatAssemblyBegin(*stateBoundaryCon,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon,MAT_FINAL_ASSEMBLY);
    
    MatDestroy(stateBoundaryConTmp);
    MatDestroy(&tmpMat);

    return;
    
}


void AdjointJacobianConnectivity::combineXvBndCon
(
    Mat* xvBoundaryCon,
    Mat* xvBoundaryConTmp
)
{

    /*
        1. Add additional X connectivity if the xvBoundaryCon stencil extends through
        three or more decomposed domains, something like this:
        
        --------       ---------
               |       |       |
          Con3 |  Con2 |  Con1 |  R
               |       |       |
               ---------       --------
               
        Here R is the residual, Con1 to 3 are its connectivity, and dashed lines 
        are the inter-processor boundary
        
        2. Assign xvBoundaryConTmp to xvBoundaryCon.
    */
    
    PetscInt    nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    
    PetscInt    nCols1;
    const PetscInt    *cols1;
    const PetscScalar *vals1;

    PetscInt Istart,Iend;
    
    
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    
    Mat tmpMat; // a temp mat
    
    // Step 1: fix conflicting connectivity between xvBoundaryConTmp and xvBoundaryCon
    // One extra level of missing stencil has been add into xvBoundaryCon. However, it will also mess up the existing stencil
    // in xvBoundaryCon. So we need to do a check to make sure that xvBoundaryConTmp only add stencil, not replacing
    // any existing stencil in xvBoundaryCon. If anything in xvBoundaryCon is replaced, rollback the changes.
    MatCreate(PETSC_COMM_WORLD,&tmpMat);
    MatSetSizes(tmpMat,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalXv,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(tmpMat);
    MatMPIAIJSetPreallocation(tmpMat,1000,NULL,1000,NULL);
    MatSetOption(tmpMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(tmpMat);
    MatZeroEntries(tmpMat); // initialize with zeros

    MatGetOwnershipRange(*xvBoundaryConTmp,&Istart,&Iend);
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(*xvBoundaryCon,i,&nCols,&cols,&vals);
        MatGetRow(*xvBoundaryConTmp,i,&nCols1,&cols1,&vals1);
        for (PetscInt j=0; j<nCols1; j++)
        {
            // for each col in xvBoundaryConTmp, we need to check if there are any existing values for the same 
            // col in xvBoundaryCon. If yes, assign the val from xvBoundaryCon instead of xvBoundaryConTmp
            PetscScalar newVal = vals1[j];
            PetscInt newCol = cols1[j];
            for (PetscInt k=0; k<nCols; k++)
            {
                /*
                // NOTE: if the old level is 3 and the new level is lower than it, replace it. This is 
                // for cases when xvBoundaryCon has an initial level-3 stencil which is wrong 
                //if ( round(cols[k]) == round(cols1[j]) and round(vals[k])!=3 and round(vals[k])<round(vals1[j]) )
                */
                if ( round(cols[k]) == round(cols1[j]) )
                {
                    newVal = vals[k];
                    newCol = cols[k];
                    break;
                }
            }
            MatSetValue(tmpMat,i,newCol,newVal,INSERT_VALUES);
        }
        MatRestoreRow(*xvBoundaryCon,i,&nCols,&cols,&vals);
        MatRestoreRow(*xvBoundaryConTmp,i,&nCols1,&cols1,&vals1);
    }
    MatAssemblyBegin(tmpMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmpMat,MAT_FINAL_ASSEMBLY);
    // copy tmpMat to ConMat
    MatDestroy(xvBoundaryCon);
    MatConvert(tmpMat, MATSAME,MAT_INITIAL_MATRIX,xvBoundaryCon);
    MatAssemblyBegin(*xvBoundaryCon,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*xvBoundaryCon,MAT_FINAL_ASSEMBLY);
    MatDestroy(&tmpMat);

    // Step 2: do one more loop to add missing connectivity
    // Destroy and initialize xvBoundaryCon with zeros
    MatDestroy(xvBoundaryConTmp);
    MatConvert(*xvBoundaryCon, MATSAME,MAT_INITIAL_MATRIX,xvBoundaryConTmp);

    // Now repeat loop one more time add missing connectivity to cellBoundaryPointsConTmp
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        label faceIStart = pp.start();

        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                label bFaceI = faceIStart-adjIdx_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];
                label idxN = pFaceCells[faceI];
                
                forAll(mesh_.cellCells()[idxN],cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1={3};
                    this->addBoundaryFaceConnectionsXv(*xvBoundaryConTmp,gRow,localCell,val1);
                }

                labelList vals2= {2,3};
                this->addBoundaryFaceConnectionsXv(*xvBoundaryConTmp,gRow,idxN,vals2);
            }
        }
    }
    
    MatAssemblyBegin(*xvBoundaryConTmp,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*xvBoundaryConTmp,MAT_FINAL_ASSEMBLY);     
        
    // Step 3: we need to fix the conflicting connectivity one more time
    // Now xvBoundaryConTmp will have all the missing stencil. However, it will also mess up the existing stencil
    // in xvBoundaryCon. So we need to do a check to make sure that xvBoundaryConTmp only adds stencil, not replacing
    // any existing stencil in xvBoundaryCon. If anything in xvBoundaryCon is replaced, rollback the changes.
    MatCreate(PETSC_COMM_WORLD,&tmpMat);
    MatSetSizes(tmpMat,adjIdx_.nLocalCoupledBFaces,adjIdx_.nLocalXv,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(tmpMat);
    MatMPIAIJSetPreallocation(tmpMat,1000,NULL,1000,NULL);
    MatSetOption(tmpMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(tmpMat);
    MatZeroEntries(tmpMat); // initialize with zeros
    for(PetscInt i=Istart; i<Iend; i++)
    {
        MatGetRow(*xvBoundaryCon,i,&nCols,&cols,&vals);
        MatGetRow(*xvBoundaryConTmp,i,&nCols1,&cols1,&vals1);
        for (PetscInt j=0; j<nCols1; j++)
        {
            // for each col in cellBoundaryPointsConTmp, we need to check if there are any existing values for the same 
            // col in cellBoundaryPointsCon. If yes, assign the val from cellBoundaryPointsCon instead of cellBoundaryPointsConTmp
            PetscScalar newVal = vals1[j];
            PetscInt newCol = cols1[j];
            for (PetscInt k=0; k<nCols; k++)
            {
                if ( round(cols[k]) ==round(cols1[j]) )
                {
                    newVal = vals[k];
                    newCol = cols[k];
                    break;
                }
            }
            MatSetValue(tmpMat,i,newCol,newVal,INSERT_VALUES);
        }
        MatRestoreRow(*xvBoundaryCon,i,&nCols,&cols,&vals);
        MatRestoreRow(*xvBoundaryConTmp,i,&nCols1,&cols1,&vals1);
    }
    MatAssemblyBegin(tmpMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmpMat,MAT_FINAL_ASSEMBLY);
    
    // copy ConMat to ConMatTmp
    MatDestroy(xvBoundaryCon);
    MatConvert(tmpMat, MATSAME,MAT_INITIAL_MATRIX,xvBoundaryCon);
    MatAssemblyBegin(*xvBoundaryCon,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*xvBoundaryCon,MAT_FINAL_ASSEMBLY);

    MatDestroy(xvBoundaryConTmp);
    MatDestroy(&tmpMat);

    return;
}

void AdjointJacobianConnectivity::initializePetscVecs()
{
    // initialize the preallocation vecs
    VecCreate(PETSC_COMM_WORLD,&dRdWTPreallocOn_);
    VecSetSizes(dRdWTPreallocOn_,adjIdx_.nLocalAdjointStates,PETSC_DECIDE);
    VecSetFromOptions(dRdWTPreallocOn_);

    VecDuplicate(dRdWTPreallocOn_,&dRdWTPreallocOff_);
    VecDuplicate(dRdWTPreallocOn_,&dRdWPreallocOn_);
    VecDuplicate(dRdWTPreallocOn_,&dRdWPreallocOff_);
    
    // initialize coloring vectors
    
    //dRdW Colors
    VecCreate(PETSC_COMM_WORLD,&dRdWColors_);
    VecSetSizes(dRdWColors_,adjIdx_.nLocalAdjointStates,PETSC_DECIDE);
    VecSetFromOptions(dRdWColors_);
    VecDuplicate(dRdWColors_,&dRdWColoredColumns_);
    
    //dFdW Colors, the dFdWColoredColumns will be initialized in initializedFdWCon
    VecCreate(PETSC_COMM_WORLD,&dFdWColors_);
    VecSetSizes(dFdWColors_,adjIdx_.nLocalAdjointStates,PETSC_DECIDE);
    VecSetFromOptions(dFdWColors_);

    if(adjIO_.isInList<word>("Xv",adjIO_.adjDVTypes))
    {
        VecCreate(PETSC_COMM_WORLD,&dRdXvPreallocOn_);
        VecSetSizes(dRdXvPreallocOn_,adjIdx_.nLocalAdjointStates,PETSC_DECIDE);
        VecSetFromOptions(dRdXvPreallocOn_);
        VecDuplicate(dRdXvPreallocOn_,&dRdXvPreallocOff_);
        VecDuplicate(dRdXvPreallocOn_,&dRdXvColoredColumns_);

        //dRdXv Colors
        VecCreate(PETSC_COMM_WORLD,&dRdXvColors_);
        VecSetSizes(dRdXvColors_,adjIdx_.nLocalXv,PETSC_DECIDE);
        VecSetFromOptions(dRdXvColors_);

        //dFdXv Colors, the dFdXvColoredColumns will be initialized in initializedFdXvCon
        VecCreate(PETSC_COMM_WORLD,&dFdXvColors_);
        VecSetSizes(dFdXvColors_,adjIdx_.nLocalXv,PETSC_DECIDE);
        VecSetFromOptions(dFdXvColors_);
    }

    return;
    
}


void AdjointJacobianConnectivity::initializedRdWCon()
{
    
    // initialize dRdWCon_
    MatCreate(PETSC_COMM_WORLD,&dRdWCon_);
    MatSetSizes(dRdWCon_,adjIdx_.nLocalAdjointStates,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(dRdWCon_);

    this->preallocateJacobianMatrix(dRdWCon_,dRdWPreallocOn_,dRdWPreallocOff_);
    //MatSetOption(dRdWCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dRdWCon_);

    Info<<"dRdWCon initialized."<<endl;
    
}

void AdjointJacobianConnectivity::initializedRdWConPC()
{
    
    // initialize dRdWConPC_
    MatCreate(PETSC_COMM_WORLD,&dRdWConPC_);
    MatSetSizes(dRdWConPC_,adjIdx_.nLocalAdjointStates,adjIdx_.nLocalAdjointStates,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(dRdWConPC_);

    this->preallocateJacobianMatrix(dRdWConPC_,dRdWPreallocOn_,dRdWPreallocOff_);
    //MatSetOption(dRdWConPC_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dRdWConPC_);

    Info<<"dRdWConPC initialized."<<endl;
    
}

void AdjointJacobianConnectivity::initializedRdXvCon()
{
    
    // initialize dRdXvCon_
    MatCreate(PETSC_COMM_WORLD,&dRdXvCon_);
    MatSetSizes(dRdXvCon_,adjIdx_.nLocalAdjointStates,adjIdx_.nLocalXv,PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(dRdXvCon_);

    this->preallocateJacobianMatrix(dRdXvCon_,dRdXvPreallocOn_,dRdXvPreallocOff_);
    //MatSetOption(dRdXvCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dRdXvCon_);

    Info<<"dRdXvCon initialized."<<endl;
    
}

void AdjointJacobianConnectivity::initializedFdWCon(const word objFunc)
{
    MatCreate(PETSC_COMM_WORLD,&dFdWCon_);
    MatSetSizes(dFdWCon_,adjIdx_.getNLocalObjFuncGeoElements(objFunc),adjIdx_.nLocalAdjointStates,
                PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(dFdWCon_);
    MatMPIAIJSetPreallocation(dFdWCon_,100,NULL,100,NULL);
    MatSeqAIJSetPreallocation(dFdWCon_,100,NULL);
    MatSetOption(dFdWCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dFdWCon_);
    MatZeroEntries(dFdWCon_);
    
    VecCreate(PETSC_COMM_WORLD,&dFdWColoredColumns_);
    VecSetSizes(dFdWColoredColumns_,adjIdx_.getNLocalObjFuncGeoElements(objFunc),PETSC_DECIDE);
    VecSetFromOptions(dFdWColoredColumns_);
    VecZeroEntries(dFdWColoredColumns_);

    Info<<"dFdWCon Created!"<<endl;
}

void AdjointJacobianConnectivity::initializedFdXvCon(const word objFunc)
{
    MatCreate(PETSC_COMM_WORLD,&dFdXvCon_);
    MatSetSizes(dFdXvCon_,adjIdx_.getNLocalObjFuncGeoElements(objFunc),adjIdx_.nLocalXv,
                PETSC_DETERMINE,PETSC_DETERMINE);
    MatSetFromOptions(dFdXvCon_);
    MatMPIAIJSetPreallocation(dFdXvCon_,500,NULL,500,NULL);
    MatSeqAIJSetPreallocation(dFdXvCon_,500,NULL);
    MatSetOption(dFdXvCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(dFdXvCon_);
    MatZeroEntries(dFdXvCon_);
    
    VecCreate(PETSC_COMM_WORLD,&dFdXvColoredColumns_);
    VecSetSizes(dFdXvColoredColumns_,adjIdx_.getNLocalObjFuncGeoElements(objFunc),PETSC_DECIDE);
    VecSetFromOptions(dFdXvColoredColumns_);
    VecZeroEntries(dFdXvColoredColumns_);

    Info<<"dFdXvCon Created!"<<endl;
}


void AdjointJacobianConnectivity::preallocateJacobianMatrix
(
    Mat dRMat,
    Vec preallocOnProc,
    Vec preallocOffProc
)
{
    PetscScalar    *onVec,*offVec;
    PetscInt   onSize[adjIdx_.nLocalAdjointStates],offSize[adjIdx_.nLocalAdjointStates];

    VecGetArray(preallocOnProc,&onVec);
    VecGetArray(preallocOffProc,&offVec);
    for(label i=0; i<adjIdx_.nLocalAdjointStates; i++)
    {
        onSize[i] = round(onVec[i]); 
        if(onSize[i]>adjIdx_.nLocalAdjointStates)
        {
            onSize[i]=adjIdx_.nLocalAdjointStates;
        }
        offSize[i] = round(offVec[i])+5; // reserve 5 more?
    }

    VecRestoreArray(preallocOnProc,&onVec);
    VecRestoreArray(preallocOffProc,&offVec);

    // MatMPIAIJSetPreallocation(dRMat,NULL,preallocOnProc,NULL,preallocOffProc);
    // MatSeqAIJSetPreallocation(dRMat,NULL,preallocOnProc);

    MatMPIAIJSetPreallocation(dRMat,NULL,onSize,NULL,offSize);
    MatSeqAIJSetPreallocation(dRMat,NULL,onSize);

    return;
}


void AdjointJacobianConnectivity::preallocatedRdW(Mat dRMat,label transposed)
{
    // call the preallocation with the correct vectors
    if(transposed)
    {
        this->preallocateJacobianMatrix(dRMat, dRdWTPreallocOn_, dRdWTPreallocOff_);
    }
    else
    {
        this->preallocateJacobianMatrix(dRMat, dRdWPreallocOn_, dRdWPreallocOff_);

    }
}

void AdjointJacobianConnectivity::preallocatedRdXv(Mat dRMat)
{
    this->preallocateJacobianMatrix(dRMat, dRdXvPreallocOn_, dRdXvPreallocOff_);
}


void AdjointJacobianConnectivity::allocateJacobianConnections
(
    Vec preallocOnProc, 
    Vec preallocOffProc,
    Vec preallocOnProcT, 
    Vec preallocOffProcT,
    Mat connections,
    label row
)
{
    PetscScalar v=1.0;
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatAssemblyBegin(connections,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(connections,MAT_FINAL_ASSEMBLY);

    // Compute the transposed case
    // in this case connections represents a single column, so we need to
    // increment the counter in each row with a non-zero entry.

    label colMin = adjIdx_.globalAdjointStateNumbering.toGlobal(0);
    label colMax = colMin+adjIdx_.nLocalAdjointStates;
    // by construction rows should be limited to local rows
    MatGetRow(connections,0,&nCols,&cols,&vals);

    // for the non-transposed case just sum up the row.
    // count up the total number of non zeros in this row
    label totalCount =0;//2
    label localCount=0;
    //int idx;
    for(label j = 0; j<nCols; j++)
    {
        // int idx = cols[j];
        scalar val = vals[j];
        if( adjIO_.isValueCloseToRef(val,1.0) )
        {
            // We can compute the first part of the non-transposed row here.
            totalCount++;
            label idx = cols[j];
            // Set the transposed version as well
            if(colMin<=idx && idx<colMax)
            {
                //this entry is a local entry, increment the corresponding row
                VecSetValue(preallocOnProcT,idx,v, ADD_VALUES);
                localCount++;
            }
            else
            {
                // this is an off proc entry.
                VecSetValue(preallocOffProcT,idx,v, ADD_VALUES);
            }
        }
    }

    label offProcCount = totalCount-localCount;
    VecSetValue(preallocOnProc,row,localCount, INSERT_VALUES);
    VecSetValue(preallocOffProc,row,offProcCount, INSERT_VALUES);

    // restore the row of the matrix
    MatRestoreRow(connections,0,&nCols,&cols,&vals);
    MatDestroy(&connections);

    return;
}

void AdjointJacobianConnectivity::allocateJacobianConnectionsXv
(
    Vec preallocOnProc, 
    Vec preallocOffProc,
    Mat connections,
    label row
)
{
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatAssemblyBegin(connections,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(connections,MAT_FINAL_ASSEMBLY);

    // Compute the transposed case
    // in this case connections represents a single column, so we need to
    // increment the counter in each row with a non-zero entry.

    label colMin = adjIdx_.globalXvNumbering.toGlobal(0);
    label colMax = colMin+adjIdx_.nLocalXv;
    // by construction rows should be limited to local rows
    MatGetRow(connections,0,&nCols,&cols,&vals);

    // for the non-transposed case just sum up the row.
    // count up the total number of non zeros in this row
    label totalCount =0;//2
    label localCount=0;
    //int idx;
    for(label j = 0; j<nCols; j++)
    {
        // int idx = cols[j];
        scalar val = vals[j];
        if( adjIO_.isValueCloseToRef(val,1.0) )
        {
            // We can compute the first part of the non-transposed row here.
            totalCount++;
            label idx = cols[j];
            // Set the transposed version as well
            if(colMin<=idx && idx<colMax)
            {
                //this entry is a local entry, increment the corresponding row
                localCount++;
            }
        }
    }

    label offProcCount = totalCount-localCount;
    VecSetValue(preallocOnProc,row,localCount, INSERT_VALUES);
    VecSetValue(preallocOffProc,row,offProcCount, INSERT_VALUES);

    // restore the row of the matrix
    MatRestoreRow(connections,0,&nCols,&cols,&vals);
    MatDestroy(&connections);

    return;
}

void AdjointJacobianConnectivity::setupJacobianConnections
(
    Mat conMat,
    Mat connections, 
    PetscInt idxI
)
{
    // Assign connectivity to Jacobian conMat, e.g., dRdWCon, based on the connections input Mat
    // Row index to added: idxI
    // Col index to added: based on connections
    
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatAssemblyBegin(connections,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(connections,MAT_FINAL_ASSEMBLY);

    MatGetRow(connections,0,&nCols,&cols,&vals);
    MatSetValues(conMat,1,&idxI,nCols,cols,vals,INSERT_VALUES);

    // restore the row of the matrix
    MatRestoreRow(connections,0,&nCols,&cols,&vals);
    MatDestroy(&connections);

    return;
}


void AdjointJacobianConnectivity::addStateConnections
(
    Mat connections, 
    label cellI,
    label connectedLevelLocal,
    wordList connectedStatesLocal,
    List< List<word> > connectedStatesInterProc,
    label addFace
)
{
    // add the connections for adj states
    // cellI: cell index to add
    // connectedLevelLocal: level of connectivity
    // connectedStatesLocal: list of states to add for that level
    // connectedStatesInterProc: list of stateI to add for a given level of boundary connectivity 
    // addFace: add face for the current level?
    
    // check if the input parameters are valid
    if(connectedLevelLocal>3 or connectedLevelLocal<0) FatalErrorIn("connectedLevelLocal not valid")<< abort(FatalError);
    if(addFace!=0 && addFace!=1) FatalErrorIn("addFace not valid")<< abort(FatalError);
    if(cellI>=mesh_.nCells()) FatalErrorIn("cellI not valid")<< abort(FatalError);
    if(connectedLevelLocal>=2 && addFace==1) FatalErrorIn("addFace not supported for localLevel>=2")<< abort(FatalError);
        
    labelList val1 = {1};
    labelList vals2= {1,1};
    labelList vals3= {1,1,1};
    
    label interProcLevel = connectedStatesInterProc.size();
    
    if(connectedLevelLocal == 0) 
    {
        // add connectedStatesLocal for level0
        forAll(connectedStatesLocal,idxI)
        {
            word stateName = connectedStatesLocal[idxI];           
            label compMax = 1;
            if (adjIdx_.adjStateType[stateName] == "volVectorState") compMax=3;
            for(label i=0;i<compMax;i++)
            {
                label idxJ = adjIdx_.getGlobalAdjointStateIndex(stateName,cellI,i);
                this->setConnections(connections,idxJ);
            }
        }
        // add faces for level0
        if(addFace) 
        {
            forAll(adjReg_.surfaceScalarStates,idxI)
            {
                const word& stateName = adjReg_.surfaceScalarStates[idxI];
                this->addConMatCellFaces(connections,0,cellI,stateName,1.0);
            }
        }
    }
    else if(connectedLevelLocal == 1)
    {
        // add connectedStatesLocal for level1
        forAll(connectedStatesLocal,idxI)
        {
            word stateName = connectedStatesLocal[idxI];   
            this->addConMatNeighbourCells(connections,0,cellI,stateName,1.0);
        }
        
        // add faces for level1
        if(addFace)
        {
            forAll(mesh_.cellCells()[cellI],cellJ)
            {
                label localCell = mesh_.cellCells()[cellI][cellJ];
                forAll(adjReg_.surfaceScalarStates,idxI)
                {
                    const word& stateName = adjReg_.surfaceScalarStates[idxI];
                    this->addConMatCellFaces(connections,0,localCell,stateName,1.0);
                }
            }
        }
        // add inter-proc connectivity for level1
        if(interProcLevel == 0)
        {
            // pass, not adding anything
        }
        else if(interProcLevel == 1)
        {
            this->addBoundaryFaceConnections(connections,0,cellI,val1,connectedStatesInterProc,addFace);
            this->addCyclicAMIFaceConnections(connections,0,cellI,val1,connectedStatesInterProc,addFace);
        }
        else if (interProcLevel == 2)
        {
            this->addBoundaryFaceConnections(connections,0,cellI,vals2,connectedStatesInterProc,addFace);
            this->addCyclicAMIFaceConnections(connections,0,cellI,vals2,connectedStatesInterProc,addFace);
        }
        else if (interProcLevel == 3)
        {
            this->addBoundaryFaceConnections(connections,0,cellI,vals3,connectedStatesInterProc,addFace);
            this->addCyclicAMIFaceConnections(connections,0,cellI,vals3,connectedStatesInterProc,addFace);
        }
        else 
            FatalErrorIn("interProcLevel not valid")<< abort(FatalError);

    }
    else if(connectedLevelLocal == 2)
    {
        forAll(mesh_.cellCells()[cellI],cellJ)
        {
            label localCell = mesh_.cellCells()[cellI][cellJ];
            // add connectedStatesLocal for level2
            forAll(connectedStatesLocal,idxI)
            {
                word stateName = connectedStatesLocal[idxI];   
                this->addConMatNeighbourCells(connections,0,localCell,stateName,1.0);
            }
            // add inter-proc connecitivty for level2
            if(interProcLevel == 0)
            {
                // pass, not adding anything
            }
            else if(interProcLevel == 1)
            {
                this->addBoundaryFaceConnections(connections,0,localCell,val1,connectedStatesInterProc,addFace);
                this->addCyclicAMIFaceConnections(connections,0,localCell,val1,connectedStatesInterProc,addFace);
            }
            else if (interProcLevel == 2)
            {
                this->addBoundaryFaceConnections(connections,0,localCell,vals2,connectedStatesInterProc,addFace);
                this->addCyclicAMIFaceConnections(connections,0,localCell,vals2,connectedStatesInterProc,addFace);
            }
            else if (interProcLevel == 3)
            {
                this->addBoundaryFaceConnections(connections,0,localCell,vals3,connectedStatesInterProc,addFace);
                this->addCyclicAMIFaceConnections(connections,0,localCell,vals3,connectedStatesInterProc,addFace);
            }
            else 
                FatalErrorIn("interProcLevel not valid")<< abort(FatalError);
        }    
    }
    else if(connectedLevelLocal == 3)
    {
    
        forAll(mesh_.cellCells()[cellI],cellJ)
        {
            label localCell = mesh_.cellCells()[cellI][cellJ];
            forAll(mesh_.cellCells()[localCell],cellK)
            {
                label localCell2 = mesh_.cellCells()[localCell][cellK];
                // add connectedStatesLocal for level3
                forAll(connectedStatesLocal,idxI)
                {
                    word stateName = connectedStatesLocal[idxI];   
                    this->addConMatNeighbourCells(connections,0,localCell2,stateName,1.0);
                }
                // add inter-proc connecitivty for level3
                if(interProcLevel == 0)
                {
                    // pass, not adding anything
                }
                else if(interProcLevel == 1)
                {
                    this->addBoundaryFaceConnections(connections,0,localCell2,val1,connectedStatesInterProc,addFace);
                    this->addCyclicAMIFaceConnections(connections,0,localCell2,val1,connectedStatesInterProc,addFace);
                }
                else if (interProcLevel == 2)
                {
                    this->addBoundaryFaceConnections(connections,0,localCell2,vals2,connectedStatesInterProc,addFace);
                    this->addCyclicAMIFaceConnections(connections,0,localCell2,vals2,connectedStatesInterProc,addFace);
                }
                else if (interProcLevel == 3)
                {
                    this->addBoundaryFaceConnections(connections,0,localCell2,vals3,connectedStatesInterProc,addFace);
                    this->addCyclicAMIFaceConnections(connections,0,localCell2,vals3,connectedStatesInterProc,addFace);
                }
                else 
                    FatalErrorIn("interProcLevel not valid")<< abort(FatalError);

            }
        }
    }
    else
    {
        FatalErrorIn("connectedLevelLocal not valid")<< abort(FatalError);
    }
    
    return;
    
}

void AdjointJacobianConnectivity::addXvConnections
(
    Mat connections, 
    label cellI,
    label connectedLevelLocal,
    label connectedLevelInterProc
)
{
    // add the connections for adj states
    // cellI: cell index to add
    // connectedLevelLocal: level of connectivity
    // connectedLevelInterProc: how many inter-proc levels to add
    
    // check if the input parameters are valid
    if(connectedLevelLocal>3 or connectedLevelLocal<0) FatalErrorIn("connectedLevelLocal not valid")<< abort(FatalError);
    if(cellI>=mesh_.nCells()) FatalErrorIn("cellI not valid")<< abort(FatalError);
    
    labelList val1 = {1};
    labelList vals2= {1,1};
    labelList vals3= {1,1,1};
    
    label interProcLevel = connectedLevelInterProc;
    
    if(connectedLevelLocal == 0) 
    {
        this->addConMatCellXv(connections,0,cellI,1.0);
    }
    else if(connectedLevelLocal == 1)
    {
        // add for level1
        this->addConMatNeighbourCellsXv(connections,0,cellI,1.0);
        
        // add inter-proc connectivity for level1
        if(interProcLevel == 0)
        {
            // pass, not adding anything
        }
        else if(interProcLevel == 1)
            this->addBoundaryFaceConnectionsXv(connections,0,cellI,val1);
        else if (interProcLevel == 2)
            this->addBoundaryFaceConnectionsXv(connections,0,cellI,vals2);
        else if (interProcLevel == 3)
            this->addBoundaryFaceConnectionsXv(connections,0,cellI,vals3);
        else 
            FatalErrorIn("interProcLevel not valid")<< abort(FatalError);

    }
    else if(connectedLevelLocal == 2)
    {
        forAll(mesh_.cellCells()[cellI],cellJ)
        {
            label localCell = mesh_.cellCells()[cellI][cellJ];
            // add for level2
            this->addConMatNeighbourCellsXv(connections,0,localCell,1.0);

            // add inter-proc connecitivity for level2
            if(interProcLevel == 0)
            {
                // pass, not adding anything
            }
            else if(interProcLevel == 1)
                this->addBoundaryFaceConnectionsXv(connections,0,localCell,val1);
            else if (interProcLevel == 2)
                this->addBoundaryFaceConnectionsXv(connections,0,localCell,vals2);
            else if (interProcLevel == 3)
                this->addBoundaryFaceConnectionsXv(connections,0,localCell,vals3);
            else 
                FatalErrorIn("interProcLevel not valid")<< abort(FatalError);
        }    
    }
    else if(connectedLevelLocal == 3)
    {
    
        forAll(mesh_.cellCells()[cellI],cellJ)
        {
            label localCell = mesh_.cellCells()[cellI][cellJ];
            forAll(mesh_.cellCells()[localCell],cellK)
            {
                label localCell2 = mesh_.cellCells()[localCell][cellK];
                // add for level3
                this->addConMatNeighbourCellsXv(connections,0,localCell2,1.0);

                // add inter-proc connecitivity for level3
                if(interProcLevel == 0)
                {
                    // pass, not adding anything
                }
                else if(interProcLevel == 1)
                    this->addBoundaryFaceConnectionsXv(connections,0,localCell2,val1);
                else if (interProcLevel == 2)
                    this->addBoundaryFaceConnectionsXv(connections,0,localCell2,vals2);
                else if (interProcLevel == 3)
                    this->addBoundaryFaceConnectionsXv(connections,0,localCell2,vals3);
                else 
                    FatalErrorIn("interProcLevel not valid")<< abort(FatalError);

            }
        }
    }
    else
    {
        FatalErrorIn("connectedLevelLocal not valid")<< abort(FatalError);
    }
    
    return;
    
}

void AdjointJacobianConnectivity::calcdRdWColoredColumns(label currColor,label isPC)
{

    if (isPC) 
    {    
        this->calcColoredColumns(currColor, dRdWConPC_, dRdWColoredColumns_,dRdWColors_);
    }
    else 
    {
        this->calcColoredColumns(currColor,dRdWCon_,dRdWColoredColumns_,dRdWColors_);
    }
}

void AdjointJacobianConnectivity::calcdRdXvColoredColumns(label currColor)
{
    this->calcColoredColumns(currColor, dRdXvCon_, dRdXvColoredColumns_,dRdXvColors_);
}

void AdjointJacobianConnectivity::calcdFdWColoredColumns(label currColor)
{

    this->calcColoredColumns(currColor, dFdWCon_, dFdWColoredColumns_,dFdWColors_);
}

void AdjointJacobianConnectivity::calcdFdXvColoredColumns(label currColor)
{

    this->calcColoredColumns(currColor, dFdXvCon_, dFdXvColoredColumns_,dFdXvColors_);
}


void AdjointJacobianConnectivity::calcColoredColumns
(
    label currColor,
    Mat conMat, 
    Vec column,
    Vec colors
)
{

    Vec colorIdx;
    label Istart,Iend;

    /* for the current color, determine which row/column pairs match up. */

    // create a vector to hold the column indices associated with this color
    VecDuplicate(colors,&colorIdx);
    VecZeroEntries(colorIdx);

    // Start by looping over the color vector. Set each column index associated
    // with the current color to its own value in the color idx vector
    // get the values on this proc
    VecGetOwnershipRange(colors,&Istart,&Iend);

    // create the arrays to access them directly
    PetscScalar *colColor,*colIdx;
    VecGetArray(colors,&colColor);
    VecGetArray(colorIdx,&colIdx);

    // loop over the entries to find the ones that match this color
    for(label j=Istart; j<Iend; j++)
    {
        label idx = j-Istart;
        if( adjIO_.isValueCloseToRef(colColor[idx],currColor*1.0) )
        {
            // use 1 based indexing here and then subtract 1 from all values in
            // the mat mult. This will handle the zero index case in the first row
            colIdx[idx]=j+1;
        }
    }
    VecRestoreArray(colors,&colColor);
    VecRestoreArray(colorIdx,&colIdx);

    VecAssemblyBegin(colorIdx);
    VecAssemblyEnd(colorIdx);

    //Set column to -1 to account for the 1 based indexing in the above loop
    VecSet(column,-1);
    // Now do a MatVec with the conMat to get the row column pairs.
    //MatMult(conMat,colorIdx,column);
    MatMultAdd(conMat,colorIdx,column,column);

    // destroy the temporary vector
    VecDestroy(&colorIdx);

}

void AdjointJacobianConnectivity::setConnections(Mat conMat,label idx)
{

    // set conMat
    PetscInt idxI = 0;
    PetscScalar v = 1;
    MatSetValues(conMat,1,&idxI,1,&idx,&v,INSERT_VALUES);
    return;
}

void AdjointJacobianConnectivity::createConnectionMat(Mat *connectedStates)
{

    // create a local matrix to store this row's connectivity
    MatCreateSeqAIJ(PETSC_COMM_SELF,1,adjIdx_.nGlobalAdjointStates,2000,NULL,connectedStates);
    //MatSetOption(*connectedStates, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*connectedStates);

    MatZeroEntries(*connectedStates);

    return;
}


void AdjointJacobianConnectivity::readdRdWColoring()
{
    Info<<"Reading dRdW Coloring.."<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dRdWColoring_"<<nProcs;
    word fileName1 = fileName.str();
    VecZeroEntries(dRdWColors_);
    adjIO_.readVectorBinary(dRdWColors_,fileName1);
    
    this->validateColoring(dRdWCon_,dRdWColors_);
    
    PetscReal maxVal;
    VecMax(dRdWColors_,NULL,&maxVal);
    ndRdWColors = maxVal+1;
}

void AdjointJacobianConnectivity::calcdRdWColoring()
{

    VecZeroEntries(dRdWColors_);
    this->parallelD2Coloring(dRdWCon_,dRdWColors_,ndRdWColors);
    this->validateColoring(dRdWCon_,dRdWColors_);
    Info<<" ndRdWColors: "<<ndRdWColors<<endl;
    
    // write dRdW colors
    Info<<"Writing dRdW Colors.."<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dRdWColoring_"<<nProcs;
    word fileName1 = fileName.str();
    adjIO_.writeVectorBinary(dRdWColors_,fileName1);
    
    return;
}

void AdjointJacobianConnectivity::readdRdXvColoring()
{
    Info<<"Reading dRdXv Coloring.."<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dRdXvColoring_"<<nProcs;
    word fileName1 = fileName.str();
    VecZeroEntries(dRdXvColors_);
    adjIO_.readVectorBinary(dRdXvColors_,fileName1);
    
    this->validateColoring(dRdXvCon_,dRdXvColors_);
    
    PetscReal maxVal;
    VecMax(dRdXvColors_,NULL,&maxVal);
    ndRdXvColors = maxVal+1;
}

void AdjointJacobianConnectivity::calcdRdXvColoring()
{

    VecZeroEntries(dRdXvColors_);
    this->parallelD2Coloring(dRdXvCon_,dRdXvColors_,ndRdXvColors);
    this->validateColoring(dRdXvCon_,dRdXvColors_);
    Info<<" ndRdXvColors: "<<ndRdXvColors<<endl;
    
    // write dRdXv colors
    Info<<"Writing dRdXv Colors.."<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dRdXvColoring_"<<nProcs;
    word fileName1 = fileName.str();
    adjIO_.writeVectorBinary(dRdXvColors_,fileName1);
    
    return;
}

void AdjointJacobianConnectivity::readdFdWColoring(const word objFunc)
{
    Info<<"Reading dFdW Coloring for "<<objFunc<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dFdWColoring_"<<objFunc<<"_"<<nProcs;
    word fileName1 = fileName.str();
    VecZeroEntries(dFdWColors_);
    adjIO_.readVectorBinary(dFdWColors_,fileName1);
    
    this->validateColoring(dFdWCon_,dFdWColors_);
    
    PetscReal maxVal;
    VecMax(dFdWColors_,NULL,&maxVal);
    ndFdWColors = maxVal+1;
}

void AdjointJacobianConnectivity::calcdFdWColoring(const word objFunc)
{

    VecZeroEntries(dFdWColors_);
    this->parallelD2Coloring(dFdWCon_,dFdWColors_,ndFdWColors);
    this->validateColoring(dFdWCon_,dFdWColors_);
    Info<<" ndFdWColors: "<<ndFdWColors<<endl;
    
    // write dFdW colors
    Info<<"Writing dFdW Colors.."<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dFdWColoring_"<<objFunc<<"_"<<nProcs;
    word fileName1 = fileName.str();
    adjIO_.writeVectorBinary(dFdWColors_,fileName1);
    
    return;
}

void AdjointJacobianConnectivity::readdFdXvColoring(const word objFunc)
{
    Info<<"Reading dFdXv Coloring for "<<objFunc<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dFdXvColoring_"<<objFunc<<"_"<<nProcs;
    word fileName1 = fileName.str();
    VecZeroEntries(dFdXvColors_);
    adjIO_.readVectorBinary(dFdXvColors_,fileName1);
    
    this->validateColoring(dFdXvCon_,dFdXvColors_);
    
    PetscReal maxVal;
    VecMax(dFdXvColors_,NULL,&maxVal);
    ndFdXvColors = maxVal+1;
}

void AdjointJacobianConnectivity::calcdFdXvColoring(const word objFunc)
{

    VecZeroEntries(dFdXvColors_);
    this->parallelD2Coloring(dFdXvCon_,dFdXvColors_,ndFdXvColors);
    this->validateColoring(dFdXvCon_,dFdXvColors_);
    Info<<" ndFdXvColors: "<<ndFdXvColors<<endl;
    
    // write dFdXv colors
    Info<<"Writing dFdXv Colors.."<<endl;
    label nProcs = Pstream::nProcs();
    std::ostringstream fileName("");
    fileName<<"dFdXvColoring_"<<objFunc<<"_"<<nProcs;
    word fileName1 = fileName.str();
    adjIO_.writeVectorBinary(dFdXvColors_,fileName1);
    
    return;
}


label AdjointJacobianConnectivity::getStateColor
(
    const word mode,
    const word stateName,
    const label idxI,
    label comp
)
{
    label color;
    if (adjIO_.useColoring)
    {
        label globalIdx = adjIdx_.getGlobalAdjointStateIndex(stateName,idxI,comp);
        PetscScalar color1;
        if(mode == "dRdW")
        {
            VecGetValues(dRdWColors_,1,&globalIdx,&color1);
            color=color1;
        }
        else if (mode == "dFdW")
        {
            VecGetValues(dFdWColors_,1,&globalIdx,&color1);
            color=color1;
        }
        else
        {
            FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
            color=-9999;
        }
    }
    else
    {
        color = adjIdx_.getGlobalAdjointStateIndex(stateName,idxI,comp);
    }
    return color;
}

label AdjointJacobianConnectivity::getXvColor
(
    const word mode,
    const label pointI,
    const label coordI
)
{
    label color;
    if (adjIO_.useColoring)
    {
        label globalIdx = adjIdx_.getGlobalXvIndex(pointI,coordI);
        PetscScalar color1;
        if(mode == "dRdXv")
        {
            VecGetValues(dRdXvColors_,1,&globalIdx,&color1);
            color=color1;
        }
        else if (mode == "dFdXv")
        {
            VecGetValues(dFdXvColors_,1,&globalIdx,&color1);
            color=color1;
        }
        else
        {
            FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
            color=-9999;
        }
    }
    else
    {
        color = adjIdx_.getGlobalXvIndex(pointI,coordI);
    }
    return color;
}

label AdjointJacobianConnectivity::getNdRdWColors()
{
    
    if(adjIO_.useColoring)
    {
        return ndRdWColors;
    }
    else
    {
        return adjIdx_.nGlobalAdjointStates;
    }

}

label AdjointJacobianConnectivity::getNdRdXvColors()
{
    
    if(adjIO_.useColoring)
    {
        return ndRdXvColors;
    }
    else
    {
        return adjIdx_.nGlobalXv;
    }

}

label AdjointJacobianConnectivity::getNdFdWColors()
{
    
    if(adjIO_.useColoring)
    {
        return ndFdWColors;
    }
    else
    {
        return adjIdx_.nGlobalAdjointStates;
    }

}

label AdjointJacobianConnectivity::getNdFdXvColors()
{

    if (adjIO_.useColoring)
    {
        return ndFdXvColors;
    }
    else
    {
        return adjIdx_.nGlobalXv;
    }
}


void AdjointJacobianConnectivity::parallelD2Coloring
(
    Mat conMat,
    Vec colors_,
    PetscInt &nColors
)
{

    // if we end up having more than 10000 colors, something must be wrong
    label maxColors=10000;

    PetscInt nCols,nCols2;
    const PetscInt    *cols;
    const PetscScalar *vals;
    const PetscScalar *vals2;

    PetscInt colorStart,colorEnd;
    VecScatter colorScatter;

    label Istart,Iend;
    label currColor;
    label notColored = 1;
    IS globalIS;
    label maxCols = 750;
    scalar allNonZeros;
    Vec globalVec;
    PetscInt nRowG,nColG;

    Info<<"Parallel Distance 2 Graph Coloring...."<<endl;

    // initialize the number of colors to zero
    nColors=0;

    // get the range of colors owned by the local prock
    VecGetOwnershipRange(colors_,&colorStart,&colorEnd);

    // Set the entire color vector to -1
    VecSet(colors_,-1);

    // Determine which rows are on the current processor
    MatGetOwnershipRange(conMat,&Istart,&Iend);

    //then get the global number of rows and columns
    MatGetSize(conMat,&nRowG, &nColG);
    label nRowL = Iend-Istart;
    label nColL = colorEnd-colorStart;

    /* Start by looping over the rows to determine the largest
       number of non-zeros per row. This will determine maxCols
       and the minumum bound for the number of colors.*/
    this->getMatNonZeros(conMat,maxCols,allNonZeros);
    Info<<"MaxCols: "<<maxCols<<endl;
    Info<<"AllNonZeros: "<<allNonZeros<<endl;

    // Create a local sparse matrix with a single row to use as a sparse vector
    Mat localCols;
    MatCreateSeqAIJ(PETSC_COMM_SELF,1,adjIdx_.nGlobalAdjointStates,adjIdx_.nLocalAdjointStates,
                    NULL,&localCols);
    MatSetOption(localCols, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(localCols);
    MatZeroEntries(localCols);

    //Now loop over the owned rows and set the value in any occupied col to 1.
    PetscInt idxI = 0;
    PetscScalar v = 1;
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(conMat, i,&nCols,&cols,&vals);
        // set any columns that have a nonzero entry into localCols
        for(label j=0; j<nCols; j++)
        {
            if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
            {
                PetscInt idx = cols[j];
                MatSetValues(localCols,1,&idxI,1,&idx,&v,INSERT_VALUES);
            }
        }
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);
    }
    MatAssemblyBegin(localCols,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(localCols,MAT_FINAL_ASSEMBLY);

    // now localCols contains the unique set of local columns on each processor
    label nUniqueCols = 0;
    MatGetRow(localCols, 0,&nCols,&cols,&vals);
    for(label j=0; j<nCols; j++)
    {
        if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
        {
            nUniqueCols++;
        }
    }
    Info<<"nUniqueCols: "<<nUniqueCols<<endl;


    //Loop over the local vectors and set nonzero entries in a global vector
    // This lets us determine which columns are strictly local and which have
    // interproccessor overlap.
    VecCreate(PETSC_COMM_WORLD,&globalVec);
    VecSetSizes(globalVec,nColL,PETSC_DECIDE);
    VecSetFromOptions(globalVec);
    VecSet(globalVec,0);

    for(label j=0; j<nCols; j++)
    {
        PetscInt idx = cols[j];
        if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
        {
            VecSetValue(globalVec,idx,vals[j], ADD_VALUES);
        }
    }
    VecAssemblyBegin(globalVec);
    VecAssemblyEnd(globalVec);

    MatRestoreRow(localCols,0,&nCols,&cols,&vals);

    // now create an index set of the strictly local columns
    label *localColumnStat = new label[nUniqueCols];
    label *globalIndexList = new label[nUniqueCols];

    PetscScalar *globalVecArray;
    VecGetArray(globalVec,&globalVecArray);

    label colCounter = 0;
    MatGetRow(localCols, 0,&nCols,&cols,&vals);
    for(label j=0; j<nCols; j++)
    {
        label col = cols[j];
        if( adjIO_.isValueCloseToRef(vals[j],1.0) )
        {
            globalIndexList[colCounter]=col;
            localColumnStat[colCounter]=1;
            if(col>=colorStart && col<colorEnd)
            {
                if( adjIO_.isValueCloseToRef(globalVecArray[col-colorStart],1.0) )
                {
                    localColumnStat[colCounter]=2; // 2: strictly local
                }
            }
            colCounter++;
        }
    }
    MatRestoreRow(localCols,0,&nCols,&cols,&vals);
    MatDestroy(&localCols);

    // create a list of the rows that have any of the strictly local columns included

    label *localRowList = new label[nRowL];
    for(label i=Istart; i<Iend; i++)
    {
        label idx = i-Istart;
        MatGetRow(conMat, i,&nCols,&cols,&vals);
        //check if this row has any strictly local columns

        /*we know that our global index lists are stored sequentially, so we
          don't need to start every index search at zero, we can start at the
          last entry found in this row */
        label kLast = -1;
        for(label j=0; j<nCols; j++)
        {
            label matCol = cols[j];
            if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
            {
                //Info<<"j: "<<j<<vals[j]<<endl;
                kLast = this->find_index(matCol,kLast+1,nUniqueCols,globalIndexList);

                if(kLast>=0)
                {
                    //k was found
                    label localVal = localColumnStat[kLast];
                    if(localVal==2)
                    {
                        localRowList[idx]=1;
                        break;
                    }
                    else
                    {
                        localRowList[idx]=0;
                    }
                }
                else
                {
                    localRowList[idx]=0;
                }
            }
        }
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);
    }
    VecRestoreArray(globalVec,&globalVecArray);

    /* Create the scatter context for the remainder of the function */
    Vec colorsLocal;
    // create a scatter context for these colors
    VecCreateSeq(PETSC_COMM_SELF,nUniqueCols,&colorsLocal);
    VecSet(colorsLocal,-1);

    // now create the Index sets
    ISCreateGeneral(PETSC_COMM_WORLD,nUniqueCols,globalIndexList,PETSC_COPY_VALUES,&globalIS);
    // Create the scatter
    VecScatterCreate(colors_,globalIS,colorsLocal,NULL,&colorScatter);
    
    
    /* Create the conflict resolution scheme*/
    // create tiebreakers locally
    Vec globalTiebreaker;
    VecDuplicate(globalVec,&globalTiebreaker);
    for(label i=colorStart; i<colorEnd; i++)
    {
        srand (i);
        PetscScalar val = rand()%nColG;
        VecSetValue(globalTiebreaker,i,val, INSERT_VALUES);

    }

    // and scatter the random values
    Vec localTiebreaker;
    VecDuplicate(colorsLocal,&localTiebreaker);
    VecScatterBegin(colorScatter,globalTiebreaker,localTiebreaker,
                    INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(colorScatter,globalTiebreaker,localTiebreaker,
                  INSERT_VALUES,SCATTER_FORWARD);


    //initialize conflict columns
    label* conflictCols = new label[maxCols];
    label* conflictLocalColIdx = new label[maxCols];
    for(label j=0; j<maxCols; j++)
    {
        conflictCols[j]=-1;
        conflictLocalColIdx[j]=-1;
    }

    // Create a global distrbuted vector of the only the local portion of
    // localcolumnsstatus
    Vec globalColumnStat;
    VecDuplicate(colors_,&globalColumnStat);
    VecSet(globalColumnStat,0.0);
    for(label k=0; k<nUniqueCols; k++)
    {
        label localCol = globalIndexList[k];
        PetscScalar localVal = localColumnStat[k];
        if(localCol>=colorStart && localCol<colorEnd)
        {
            VecSetValue(globalColumnStat,localCol,localVal, INSERT_VALUES);
        }
    }
    VecAssemblyBegin(globalColumnStat);
    VecAssemblyEnd(globalColumnStat);

    /*
      create a duplicate matrix for conMat that contains its index into the
      local arrays
    */
    Mat conIndMat;
    MatDuplicate(conMat,MAT_SHARE_NONZERO_PATTERN,&conIndMat);

    // now loop over conMat locally, find the index in the local array
    // and store that value in conIndMat
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(conMat, i,&nCols,&cols,&vals);
        label kLast = -1;
        for(label j=0; j<nCols; j++)
        {
            label matCol = cols[j];
            if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
            {
                kLast = this->find_index(matCol,kLast+1,nUniqueCols,globalIndexList);
                PetscScalar val = kLast;
                MatSetValue(conIndMat,i,matCol,val,INSERT_VALUES);
            }
        }
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);
    }
    MatAssemblyBegin(conIndMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(conIndMat,MAT_FINAL_ASSEMBLY);

    /*Now color the locally independent columns using the only the rows
      that contain entries from those columns.*/

    // Retrieve the local portion of the color vector
    PetscScalar *colColor,*tbkrLocal,*tbkrGlobal,*globalStat;
    VecGetArray(colors_,&colColor);
    VecGetArray(localTiebreaker,&tbkrLocal);
    VecGetArray(globalTiebreaker,&tbkrGlobal);
    VecGetArray(globalColumnStat,&globalStat);

    // Loop over the maximum number of colors
    for(label n=0; n<maxColors; n++)
    {
        Info<<"ColorSweep: "<<n<<endl;

        /* Set all entries for strictly local columns that are currently -1
           to the current color */
        for(label k=0; k<nUniqueCols; k++)
        {
            label localCol = globalIndexList[k];
            label localVal = localColumnStat[k];
            if(localCol>=colorStart && localCol<colorEnd && localVal==2)
            {
                // this is a strictly local column;
                label idx = localCol-colorStart;
                if( adjIO_.isValueCloseToRef(colColor[idx],-1.0) )
                {
                    colColor[idx]=n;
                }
            }
        }

        //Now loop over the rows and resolve conflicts
        for(label i=Istart; i<Iend; i++)
        {

            // Get the row local row index
            label idx = i-Istart;

            //create the variables for later sorting
            label smallest = nColG;
            label idxKeep=-1;

            //First check if this is a row that contains strictly local columns
            if(localRowList[idx]>0)
            {

                /* this is a row that contains strictly local columns,get the
                   row information */
                MatGetRow(conMat, i,&nCols,&cols,&vals);

                // set any columns with the current color into conflictCols
                for(label j=0; j<nCols; j++)
                {
                    if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
                    {
                        label colIdx = cols[j];

                        // Check that this is a local column
                        if(colIdx>=colorStart && colIdx<colorEnd)
                        {
                            //now check if it is a strictly local column
                            label localVal = globalStat[colIdx-colorStart];
                            if(localVal==2)
                            {
                                // check if the color in this column is from the
                                // current set
                                if( adjIO_.isValueCloseToRef(colColor[colIdx-colorStart],n*1.0) )
                                {
                                    /* This is a potentially conflicting column
                                       store it */
                                    conflictCols[j]=colIdx;

                                    // now check whether this is the one we keep
                                    label tbkr = tbkrGlobal[colIdx-colorStart];
                                    if(tbkr<smallest)
                                    {
                                        smallest = tbkr;
                                        idxKeep = colIdx;
                                    }
                                }
                            }
                        }
                    }
                }

                // Now reset all columns but the one that wins the tiebreak
                for(label j=0; j<nCols; j++)
                {

                    //check if this is a conflicting column
                    label colIdx = conflictCols[j];
                    if(colIdx>=0)
                    {
                        // Check that this is also a local column
                        if(colIdx>=colorStart && colIdx<colorEnd)
                        {
                            // and now if it is a strictly local column
                            label localVal = globalStat[colIdx-colorStart];
                            if( localVal==2)
                            {
                                // now reset the column
                                if(colIdx>=0 && (colIdx!=idxKeep))
                                {
                                    colColor[colIdx-colorStart]=-1;
                                }
                            }
                        }
                    }
                }
                // reset the changed values in conflictCols
                for(label j=0; j<nCols; j++)
                {
                    if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
                    {
                        //reset all values related to this row in conflictCols
                        conflictCols[j]=-1;
                    }
                }
                MatRestoreRow(conMat,i,&nCols,&cols,&vals);
            }
        }


        // now we want to check the coloring on the strictly local columns
        notColored=0;

        //loop over the columns and check if there are any uncolored rows
        label colorCounter = 0;
        for(label k=0; k<nUniqueCols; k++)
        {
            // get the column info
            label localVal = localColumnStat[k];
            label localCol = globalIndexList[k];
            // check if it is strictly local, if so it should be colored
            if(localVal==2)
            {
                // confirm that it is a local column (is this redundant?
                if(localCol>=colorStart && localCol<colorEnd)
                {
                    label idx = localCol-colorStart;
                    label color = colColor[idx];
                    // now check that it has been colored
                    if(not(color>=0))
                    {
                        // this column is not colored and coloring is not complete
                        notColored=1;
                        colorCounter++;
                        //break;
                    }
                }
            }
        }

        // reduce the logical so that we know that all of the processors are
        // ok
        reduce(notColored, sumOp<label>() );
        reduce(colorCounter, sumOp<label>() );
        Info<<"number of uncolored: "<<colorCounter<<" "<<notColored<<endl;

        if(notColored==0)
        {
            break;
        }
    }
    VecRestoreArray(colors_,&colColor);
    /***** end of local coloring ******/


    // now redo the local row list to handle the global columns
    // create a list of the rows that have any of the global columns included

    for(label i=Istart; i<Iend; i++)
    {
        label idx = i-Istart;
        // get the row information
        MatGetRow(conMat, i,&nCols,&cols,&vals);

        //check if this row has any non-local columns

        /* We know that our localColumnStat is stored sequentially, so we don't
           need to start every index search at zero, we can start at the last
           entry found in this row.*/
        label kLast = -1;

        for(label j=0; j<nCols; j++)
        {
            // get the column of interest
            label matCol = cols[j];
            // confirm that it has an entry
            if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
            {
                // find the index into the local arrays for this column
                kLast = this->find_index(matCol,kLast+1,nUniqueCols,globalIndexList);
                // if this column is present (should always be true?) process the row
                if(kLast>=0)
                {
                    // get the local column type
                    label localVal = localColumnStat[kLast];
                    /* If this is a global column, add the row and move to the next
                       row, otherwise check the next column in the row */
                    if(localVal==1)
                    {
                        localRowList[idx]=1;
                        break;
                    }
                    else
                    {
                        localRowList[idx]=0;
                    }
                }
                else
                {
                    // if this column wasn't found, move to the next column
                    localRowList[idx]=0;
                }
            }
        }
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);
    }


    // now that we know the set of global rows, complete the coloring

    // Loop over the maximum number of colors
    for(label n=0; n<maxColors; n++)
    {
        Info<<"Global ColorSweep: "<<n<<endl;

        // Retrieve the local portion of the color vector
        // and set all entries that are currently -1 to the current color
        PetscScalar *colColor;
        VecGetArray(colors_,&colColor);

        for(label i=colorStart; i<colorEnd; i++)
        {
            label idx = i-colorStart;
            if( adjIO_.isValueCloseToRef(colColor[idx],-1.0) )
            {
                colColor[idx]=n;
            }
        }
        VecRestoreArray(colors_,&colColor);

        /* We will do the confilct resolution in two passes. On the first pass
           we will keep the value with the smallest random index on the local
           column set. We will not touch the off processor columns. On the second
           pass we will keep the value with the smallest random index, regardless
           of processor. This is to prevent deadlocks in the conflict resolution.*/
        for(label conPass=0; conPass<2; conPass++)
        {

            // Scatter the global colors to each processor
            VecScatterBegin(colorScatter,colors_,colorsLocal,
                            INSERT_VALUES,SCATTER_FORWARD);
            VecScatterEnd(colorScatter,colors_,colorsLocal,
                          INSERT_VALUES,SCATTER_FORWARD);

            // compute the number of local rows
            //int nRows = Iend_-Istart_;

            //Allocate a Scalar array to recieve colColors.
            PetscScalar *colColorLocal;
            VecGetArray(colorsLocal,&colColorLocal);
            VecGetArray(colors_,&colColor);

            // set the iteration limits based on conPass
            label start,end;
            if(conPass==0)
            {
                start = colorStart;
                end = colorEnd;
            }
            else
            {
                start = 0;
                end = nColG;
            }

            //Now loop over the rows and resolve conflicts
            for(label i=Istart; i<Iend; i++)
            {
                label idx = i-Istart;
                if(localRowList[idx]==1) //this row includes at least 1 global col.
                {
                    /* Get the connectivity row as well as its index into the
                       local indices. */
                    MatGetRow(conMat, i,&nCols,&cols,&vals);
                    MatGetRow(conIndMat, i,&nCols2,NULL,&vals2);

                    // initialize the sorting variables
                    label smallest = nColG;
                    label idxKeep = -1;

                    //int localColIdx;
                    // set any columns with the current color into conflictCols
                    for(label j=0; j<nCols; j++)
                    {
                        if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
                        {
                            label colIdx = cols[j];
                            label localColIdx = round(vals2[j]);

                            // check if the color in this column is from the
                            // current set
                            if( adjIO_.isValueCloseToRef(colColorLocal[localColIdx],n*1.0) )
                            {
                                /* This matches the current color, so set as a
                                   potential conflict */
                                conflictCols[j]=colIdx;
                                conflictLocalColIdx[j]=localColIdx;
                                /* this is one of the conflicting columns.
                                   If this is a strictly local column, keep it.
                                   Otherwise, compare its random number to the
                                   current smallest one, keep the smaller one and
                                   its index find the index of the smallest
                                   tiebreaker. On the first pass this is only
                                   for the the local columns. On pass two it is
                                   for all columns.*/
                                if(localColumnStat[localColIdx]==2)
                                {
                                    smallest = -1;
                                    idxKeep = colIdx;

                                }
                                else if(tbkrLocal[localColIdx]<smallest and
                                        colIdx>=start and
                                        colIdx<end)
                                {
                                    smallest = tbkrLocal[localColIdx];
                                    idxKeep = cols[j];

                                }
                            }
                        }
                    }

                    // Now reset all the conflicting rows
                    for(label j=0; j<nCols; j++)
                    {
                        label colIdx = conflictCols[j];
                        label localColIdx = conflictLocalColIdx[j];
                        // check that the column is in the range for this conPass.
                        if(colIdx>=start && colIdx<end)
                        {
                            /*this column is local. If this isn't the
                            smallest, reset it.*/
                            if(colIdx!=idxKeep)
                            {
                                if(localColIdx>=0)
                                {
                                    if(localColumnStat[localColIdx]==2)
                                    {
                                        Pout<<"local Array Index: "<<colIdx<<endl;
                                        Info<<"Error, setting a local column!"<<endl;
                                        return;
                                    }
                                    PetscScalar valIn = -1;
                                    VecSetValue(colors_,colIdx,valIn,INSERT_VALUES);
                                    colColorLocal[localColIdx]=-1;
                                }
                            }
                        }
                    }

                    /* reset any columns that have been changed in conflictCols
                    and conflictLocalColIdx */
                    for(label j=0; j<nCols; j++)
                    {
                        if( !adjIO_.isValueCloseToRef(vals[j],0.0) )
                        {
                            //reset all values related to this row in conflictCols
                            conflictCols[j]=-1;
                            conflictLocalColIdx[j] =-1;
                        }
                    }

                    // Restore the row information
                    MatRestoreRow(conIndMat,i,&nCols2,NULL,&vals2);
                    MatRestoreRow(conMat,i,&nCols,&cols,&vals);
                }
            }

            VecRestoreArray(colors_,&colColor);
            VecRestoreArray(colorsLocal,&colColorLocal);
            VecAssemblyBegin(colors_);
            VecAssemblyEnd(colors_);

        }

        //check the coloring for completeness
        this->coloringComplete(colors_,notColored);
        if(notColored==0)
        {
            break;
        }
    }
    VecRestoreArray(globalTiebreaker,&tbkrGlobal);
    VecRestoreArray(globalColumnStat,&globalStat);

    // count the current colors and aggregate
    currColor = 0;
    PetscScalar color;
    for(label i=colorStart; i<colorEnd; i++)
    {
        VecGetValues(colors_,1,&i,&color);
        //Pout<<"Color: "<<i<<" "<<color<<endl;
        if(color>currColor)
        {
            currColor = color;
        }
    }

    reduce(currColor, maxOp<label>() );

    nColors = currColor+1;

    Info<<"Ncolors: "<<nColors<<endl;

    //check the initial coloring for completeness
    this->coloringComplete(colors_,notColored);

    // clean up the unused memory
    VecRestoreArray(localTiebreaker,&tbkrLocal);
    delete [] conflictCols;
    delete [] conflictLocalColIdx;
    delete [] globalIndexList;
    delete [] localColumnStat;
    ISDestroy(&globalIS);
    VecScatterDestroy(&colorScatter);
    VecDestroy(&colorsLocal);
    VecDestroy(&globalTiebreaker);
    VecDestroy(&globalColumnStat);
    VecDestroy(&localTiebreaker);
    delete [] localRowList;
    VecDestroy(&globalVec);
    MatDestroy(&conIndMat);
}

void AdjointJacobianConnectivity::getMatNonZeros
(
    Mat conMat,
    label& maxCols, 
    scalar& allNonZeros
)
{
    // get the max nonzeros per row, and all the nonzeros for this matrix

    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    
    label Istart,Iend;

    // set the counter
    maxCols = 0;
    allNonZeros = 0.0;

    // Determine which rows are on the current processor
    MatGetOwnershipRange(conMat,&Istart,&Iend);

    // loop over the matrix and find the largest number of cols
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(conMat,i,&nCols,&cols,&vals);
        if(nCols<0)
        {
            std::cout<<"Warning! procI: "<<Pstream::myProcNo()<<" nCols <0 at rowI: "<<i<<std::endl;
            std::cout<<"Set nCols to zero "<<std::endl;
            nCols = 0;
        }
        if(nCols>maxCols) // perhaps actually check vals?
        {
            maxCols = nCols;
        }
        allNonZeros += nCols;
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);
    }

    //reduce the maxcols value so that all procs have the same size
    reduce(maxCols, maxOp<label>());
    
    reduce(allNonZeros, sumOp<scalar>());

    return;
}

label AdjointJacobianConnectivity::find_index
(
    label target,
    label start, 
    label size, 
    label *valArray
)
{

    // loop over the valArray from start until target is found
    for(label k=start; k<size; k++)
    {
        if(valArray[k]==target)
        {
            //Info<<"Start: "<<start<<" "<<k<<endl;
            //this is the k of interest
            return k;
        }
    }
    return -1;
}


void AdjointJacobianConnectivity::coloringComplete
(
    Vec colors_,
    label& notColored
)
{

    PetscScalar color;
    PetscInt colorStart,colorEnd;

    notColored=0;
    // get the range of colors owned by the local prock
    VecGetOwnershipRange(colors_,&colorStart,&colorEnd);
    //loop over the columns and check if there are any uncolored rows
    PetscScalar *colColor;
    VecGetArray(colors_,&colColor);
    label colorCounter = 0;
    for(label i=colorStart; i<colorEnd; i++)
    {
        color = colColor[i-colorStart];
        //VecGetValues(colors_,1,&i,&color);
        if(not(color>=0))
        {
            // this columns not colored and coloring is not complete
            //Pout<<"coloring incomplete...: "<<color<<" "<<i<<endl;
            //VecView(colors_,PETSC_VIEWER_STDOUT_WORLD);
            notColored=1;
            colorCounter++;
            //break;
        }
    }
    VecRestoreArray(colors_,&colColor);
    // reduce the logical so that we know that all of the processors are
    // ok
    //Pout<<"local number of uncolored: "<<colorCounter<<" "<<notColored<<endl;
    reduce(notColored, sumOp<label>() );
    reduce(colorCounter, sumOp<label>() );
    Info<<"Number of Uncolored: "<<colorCounter<<" "<<notColored<<endl;
}


void AdjointJacobianConnectivity::validateColoring(Mat conMat, Vec colors)
{
    // loop over the rows and verify that no row has two columns with the same color
    
    Info<<"Validating Coloring..."<<endl;
    
    PetscInt nCols;
    const PetscInt    *cols;
    const PetscScalar *vals;
   
    label Istart,Iend;
    
    // scatter colors to local array for all procs
    Vec vout;
    VecScatter ctx;
    VecScatterCreateToAll(colors,&ctx,&vout);
    VecScatterBegin(ctx,colors,vout,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(ctx,colors,vout,INSERT_VALUES,SCATTER_FORWARD);
    
    PetscScalar* colorsArray;
    VecGetArray(vout,&colorsArray);

    // Determine which rows are on the current processor
    MatGetOwnershipRange(conMat,&Istart,&Iend);
    
    // first calc the largest nCols in conMat
    label colMax=0;
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(conMat, i,&nCols,&cols,&vals);
        if (nCols>colMax) colMax=nCols;
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);
    }
    
    // now check if conMat has conflicting rows
    labelList rowColors(colMax);
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(conMat, i,&nCols,&cols,&vals);
        
        // initialize rowColors with -1
        for(label nn=0;nn<colMax;nn++)
        {
            rowColors[nn]=-1;
        }
        
        // set rowColors for this row
        for(label j=0; j<nCols; j++)
        {
            if( adjIO_.isValueCloseToRef(vals[j],1.0) )
            {
                rowColors[j]=round(colorsArray[cols[j]]);
            }
        }

        // check if rowColors has duplicated colors
        for(label nn=0;nn<nCols;nn++)
        {
            for(label mm=nn+1;mm<nCols;mm++)
            {
                if(rowColors[nn]!=-1 && rowColors[nn]==rowColors[mm])
                {
                    FatalErrorIn("Conflicting Colors Found!")
                        <<" row: "<<i<<" col1: "<<cols[nn]<<" col2: "<<cols[mm]
                        <<" color: "<<rowColors[nn]<<abort(FatalError);
                }
            }
        }
        
        MatRestoreRow(conMat,i,&nCols,&cols,&vals);

    }

    VecRestoreArray(vout,&colorsArray);
    VecScatterDestroy(&ctx);
    VecDestroy(&vout);
    
    Info<<"No Conflicting Colors Found!"<<endl;
    
    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
