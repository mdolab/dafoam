/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAJacCon.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAJacCon, 0);
defineRunTimeSelectionTable(DAJacCon, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAJacCon::DAJacCon(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : modelType_(modelType),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daColoring_(mesh, daOption, daModel, daIndex),
      daField_(mesh, daOption, daModel, daIndex)
{
    // initialize stateInfo_
    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();

    // check if there is special boundary conditions that need special treatment in jacCon_
    this->checkSpecialBCs();
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAJacCon> DAJacCon::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
{
    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAJacCon" << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAJacCon::New"
            "("
            "    const word,"
            "    const fvMesh&,"
            "    const DAOption&,"
            "    const DAModel&,"
            "    const DAIndex&"
            ")")
            << "Unknown DAJacCon type "
            << modelType << nl << nl
            << "Valid DAJacCon types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAJacCon>(
        cstrIter()(modelType, mesh, daOption, daModel, daIndex));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAJacCon::initializeStateBoundaryCon()
{
    /*
    Description:
        Initialize state boundary connectivity matrices and variables. 
        This will be used for Jacobian with respect to states, e.g., dRdW, and dFdW.

        NOTE: no need to call this function for partial derivatives that are not wrt states
    
    Output:
        neiBFaceGlobalCompact_: neibough face global index for a given local boundary face

        stateBoundaryCon_: matrix to store boundary connectivity levels for state Jacobians

        stateBoundaryConID_: matrix to store boundary connectivity ID for state Jacobians
    */

    // Calculate the boundary connectivity
    if (daOption_.getOption<label>("debug"))
    {
        Info << "Generating Connectivity for Boundaries:" << endl;
    }

    this->calcNeiBFaceGlobalCompact(neiBFaceGlobalCompact_);

    this->setupStateBoundaryCon(&stateBoundaryCon_);

    this->setupStateBoundaryConID(&stateBoundaryConID_);

    wordList writeJacobians;
    daOption_.getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("stateBoundaryCon"))
    {
        DAUtility::writeMatrixBinary(stateBoundaryCon_, "stateBoundaryCon");
        DAUtility::writeMatrixBinary(stateBoundaryConID_, "stateBoundaryConID");
    }
}

void DAJacCon::setupJacobianConnections(
    Mat conMat,
    Mat connections,
    const PetscInt idxI)
{
    /*
    Description:
        Assign connectivity to Jacobian conMat, e.g., dRdWCon, based on the connections input Mat
    
    Input:
        idxI: Row index to added, ad the column index to added is based on connections

        connections: the one row matrix with nonzero values to add to conMat

    Output:
        conMat: the connectivity mat to add
    */

    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    MatAssemblyBegin(connections, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(connections, MAT_FINAL_ASSEMBLY);

    MatGetRow(connections, 0, &nCols, &cols, &vals);
    MatSetValues(conMat, 1, &idxI, nCols, cols, vals, INSERT_VALUES);

    // restore the row of the matrix
    MatRestoreRow(connections, 0, &nCols, &cols, &vals);
    MatDestroy(&connections);

    return;
}

void DAJacCon::createConnectionMat(Mat* connectedStates)
{
    /*
    Description:
        Initialize a serial connectivity matrix connectedStates,
        basically, it is one row of connectivity in DAJacCon::jacCon_
    
    Input/Output:
        connectedStates: a 1 row matrix that will be used to store the connectivity
    */

    // create a local matrix to store this row's connectivity
    MatCreateSeqAIJ(
        PETSC_COMM_SELF,
        1,
        daIndex_.nGlobalAdjointStates,
        2000,
        NULL,
        connectedStates);
    //MatSetOption(*connectedStates, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*connectedStates);

    MatZeroEntries(*connectedStates);

    return;
}

void DAJacCon::addStateConnections(
    Mat connections,
    const label cellI,
    const label connectedLevelLocal,
    const wordList connectedStatesLocal,
    const List<List<word>> connectedStatesInterProc,
    const label addFace)
{
    /*
    Description:
        A high level interface to add the connectivity for the row matrix connections
        Note: the connections mat is basically one row of connectivity in DAJacCon::jacCon_

    Input:
        cellI: cell index based on which we want to add the connectivity. We can add any level of 
        connected states to this cellI

        connectedLevelLocal: level of local connectivity, this is useually obtained from
        DAJacCon::adjStateResidualConInfo_

        connectedStatesLocal: list of connected states to add for the level: connectedLevelLocal
    
        connectedStatesInterProc: list of states to add for a given level of boundary connectivity 
        
        addFace: add cell faces for the current level?

    Output:
        connections: one row of connectivity in DAJacCon::jacCon_

    Example:
        If the connectivity list reads:
    
        adjStateResidualConInfo_
        {
            "URes"
            {
                {"U", "p", "phi"}, // level 0 connectivity
                {"U", "p", "phi"}, // level 1 connectivity
                {"U"},             // level 2 connectivity
            }
        }
    
        and the cell topology with a inter-proc boundary cen be either of the following:
        CASE 1:
                           ---------
                           | cellQ |
                    -----------------------
                   | cellP | cellJ | cellO |             <------ proc1
        ------------------------------------------------ <----- inter-processor boundary
           | cellT | cellK | cellI | cellL | cellU |     <------ proc0
           -----------------------------------------
                   | cellN | cellM | cellR |
                    ------------------------
                           | cellS |
                           ---------
        
        CASE 2:
                           ---------
                           | cellQ |                       <------ proc1
        -------------------------------------------------- <----- inter-processor boundary
                   | cellP | cellJ | cellO |               <------ proc0
           ----------------------------------------- 
           | cellT | cellK | cellI | cellL | cellU |    
           -----------------------------------------
                   | cellN | cellM | cellR |
                    ------------------------
                           | cellS |
                           ---------
        
        Then, to add the connectivity correctly, we need to add all levels of connected
        states for cellI.
        Level 0 connectivity is straightforward becasue we don't need
        to provide connectedStatesInterProc
    
        To add level 1 connectivity, we need to:
        set connectedLevelLocal = 1
        set connectedStatesLocal = {U, p}
        set connectedStatesInterProc = {{U,p}, {U}}
        set addFace = 1 
        NOTE: we need set level 1 and level 2 con in connectedStatesInterProc because the 
        north face of cellI is a inter-proc boundary and there are two levels of connected
        state on the other side of the inter-proc boundary for CASE 1. This is the only chance we 
        can add all two levels of connected state across the boundary for CASE 1. For CASE 2, we won't
        add any level 1 inter-proc states because non of the faces for cellI are inter-proc
        faces so calling DAJacCon::addBoundaryFaceConnections for cellI won't add anything
    
        To add level 2 connectivity, we need to
        set connectedLevelLocal = 2
        set connectedStatesLocal = {U}
        set connectedStatesInterProc = {{U}}
        set addFace = 0
        NOTE 1: we need only level 2 con (U) for connectedStatesInterProc because if we are in CASE 1,
        the level 2 of inter-proc states have been added. For CASE 2, we only need to add cellQ
        by calling DAJacCon::addBoundaryFaceConnections with cellJ
        NOTE 2: If we didn't call two levels of connectedStatesInterProc in the previous call for 
        level 1 con, we can not add it for connectedLevelLocal = 2 becasue for CASE 2 there is no
        inter-proc boundary for cellI
    
        NOTE: how to provide connectedLevelLocal, connectedStatesLocal, and connectedStatesInterProc
        are done in DAJacCon::setupJacCon

    */

    // check if the input parameters are valid
    if (connectedLevelLocal > 3 or connectedLevelLocal < 0)
    {
        FatalErrorIn("connectedLevelLocal not valid") << abort(FatalError);
    }
    if (addFace != 0 && addFace != 1)
    {
        FatalErrorIn("addFace not valid") << abort(FatalError);
    }
    if (cellI >= mesh_.nCells())
    {
        FatalErrorIn("cellI not valid") << abort(FatalError);
    }
    //if (connectedLevelLocal>=2 && addFace==1)
    //{
    //    FatalErrorIn("addFace not supported for localLevel>=2")<< abort(FatalError);
    //}

    labelList val1 = {1};
    labelList vals2 = {1, 1};
    labelList vals3 = {1, 1, 1};

    label interProcLevel = connectedStatesInterProc.size();

    if (connectedLevelLocal == 0)
    {
        // add connectedStatesLocal for level0
        forAll(connectedStatesLocal, idxI)
        {
            word stateName = connectedStatesLocal[idxI];
            label compMax = 1;
            if (daIndex_.adjStateType[stateName] == "volVectorState")
            {
                compMax = 3;
            }
            for (label i = 0; i < compMax; i++)
            {
                label idxJ = daIndex_.getGlobalAdjointStateIndex(stateName, cellI, i);
                this->setConnections(connections, idxJ);
            }
        }
        // add faces for level0
        if (addFace)
        {
            forAll(stateInfo_["surfaceScalarStates"], idxI)
            {
                word stateName = stateInfo_["surfaceScalarStates"][idxI];
                this->addConMatCellFaces(connections, 0, cellI, stateName, 1.0);
            }
        }
    }
    else if (connectedLevelLocal == 1)
    {
        // add connectedStatesLocal for level1
        forAll(connectedStatesLocal, idxI)
        {
            word stateName = connectedStatesLocal[idxI];
            this->addConMatNeighbourCells(connections, 0, cellI, stateName, 1.0);
        }

        // add faces for level1
        if (addFace)
        {
            forAll(mesh_.cellCells()[cellI], cellJ)
            {
                label localCell = mesh_.cellCells()[cellI][cellJ];
                forAll(stateInfo_["surfaceScalarStates"], idxI)
                {
                    word stateName = stateInfo_["surfaceScalarStates"][idxI];
                    this->addConMatCellFaces(connections, 0, localCell, stateName, 1.0);
                }
            }
        }
        // add inter-proc connectivity for level1
        if (interProcLevel == 0)
        {
            // pass, not adding anything
        }
        else if (interProcLevel == 1)
        {
            this->addBoundaryFaceConnections(
                connections,
                0,
                cellI,
                val1,
                connectedStatesInterProc,
                addFace);
        }
        else if (interProcLevel == 2)
        {
            this->addBoundaryFaceConnections(
                connections,
                0,
                cellI,
                vals2,
                connectedStatesInterProc,
                addFace);
        }
        else if (interProcLevel == 3)
        {
            this->addBoundaryFaceConnections(
                connections,
                0,
                cellI,
                vals3,
                connectedStatesInterProc,
                addFace);
        }
        else
        {
            FatalErrorIn("interProcLevel not valid") << abort(FatalError);
        }
    }
    else if (connectedLevelLocal == 2)
    {
        forAll(mesh_.cellCells()[cellI], cellJ)
        {
            label localCell = mesh_.cellCells()[cellI][cellJ];

            // add connectedStatesLocal for level2
            forAll(connectedStatesLocal, idxI)
            {
                word stateName = connectedStatesLocal[idxI];
                this->addConMatNeighbourCells(connections, 0, localCell, stateName, 1.0);
            }

            // add faces for level2
            if (addFace)
            {
                forAll(mesh_.cellCells()[localCell], cellK)
                {
                    label localCellK = mesh_.cellCells()[localCell][cellK];
                    forAll(stateInfo_["surfaceScalarStates"], idxI)
                    {
                        word stateName = stateInfo_["surfaceScalarStates"][idxI];
                        this->addConMatCellFaces(connections, 0, localCellK, stateName, 1.0);
                    }
                }
            }

            // add inter-proc connecitivty for level2
            if (interProcLevel == 0)
            {
                // pass, not adding anything
            }
            else if (interProcLevel == 1)
            {
                this->addBoundaryFaceConnections(
                    connections,
                    0,
                    localCell,
                    val1,
                    connectedStatesInterProc,
                    addFace);
            }
            else if (interProcLevel == 2)
            {
                this->addBoundaryFaceConnections(
                    connections,
                    0,
                    localCell,
                    vals2,
                    connectedStatesInterProc,
                    addFace);
            }
            else if (interProcLevel == 3)
            {
                this->addBoundaryFaceConnections(
                    connections,
                    0,
                    localCell,
                    vals3,
                    connectedStatesInterProc,
                    addFace);
            }
            else
            {
                FatalErrorIn("interProcLevel not valid") << abort(FatalError);
            }
        }
    }
    else if (connectedLevelLocal == 3)
    {

        forAll(mesh_.cellCells()[cellI], cellJ)
        {
            label localCell = mesh_.cellCells()[cellI][cellJ];
            forAll(mesh_.cellCells()[localCell], cellK)
            {
                label localCell2 = mesh_.cellCells()[localCell][cellK];

                // add connectedStatesLocal for level3
                forAll(connectedStatesLocal, idxI)
                {
                    word stateName = connectedStatesLocal[idxI];
                    this->addConMatNeighbourCells(connections, 0, localCell2, stateName, 1.0);
                }

                // add faces for level3
                if (addFace)
                {
                    forAll(mesh_.cellCells()[localCell2], cellL)
                    {
                        label localCellL = mesh_.cellCells()[localCell2][cellL];
                        forAll(stateInfo_["surfaceScalarStates"], idxI)
                        {
                            word stateName = stateInfo_["surfaceScalarStates"][idxI];
                            this->addConMatCellFaces(connections, 0, localCellL, stateName, 1.0);
                        }
                    }
                }

                // add inter-proc connecitivty for level3
                if (interProcLevel == 0)
                {
                    // pass, not adding anything
                }
                else if (interProcLevel == 1)
                {
                    this->addBoundaryFaceConnections(
                        connections,
                        0,
                        localCell2,
                        val1,
                        connectedStatesInterProc,
                        addFace);
                }
                else if (interProcLevel == 2)
                {
                    this->addBoundaryFaceConnections(
                        connections,
                        0,
                        localCell2,
                        vals2,
                        connectedStatesInterProc,
                        addFace);
                }
                else if (interProcLevel == 3)
                {
                    this->addBoundaryFaceConnections(
                        connections,
                        0,
                        localCell2,
                        vals3,
                        connectedStatesInterProc,
                        addFace);
                }
                else
                {
                    FatalErrorIn("interProcLevel not valid") << abort(FatalError);
                }
            }
        }
    }
    else
    {
        FatalErrorIn("connectedLevelLocal not valid") << abort(FatalError);
    }

    return;
}

void DAJacCon::setConnections(
    Mat conMat,
    const label idx) const
{

    /*
    Description:
        Set 1.0 for conMat, the column index is idx, the row index is
        always 1 because conMat is a row matrix
    */

    PetscInt idxI = 0;
    PetscScalar v = 1;
    MatSetValues(conMat, 1, &idxI, 1, &idx, &v, INSERT_VALUES);
    return;
}

void DAJacCon::calcNeiBFaceGlobalCompact(labelList& neiBFaceGlobalCompact)
{
    /*
    Description:
        This function calculates DAJacCon::neiBFaceGlobalCompact[bFaceI]. Here neiBFaceGlobalCompact 
        stores the global coupled boundary face index for the face on the other side of the local 
        processor boundary. bFaceI is the "compact" face index. bFaceI=0 for the first boundary face
        neiBFaceGlobalCompat.size() = nLocalBoundaryFaces
        neiBFaceGlobalCompact[bFaceI] = -1 means it is not a coupled face
        NOTE: neiBFaceGlobalCompact will be used to calculate the connectivity across processors
        in DAJacCon::setupStateBoundaryCon

    Output:
        neiBFaceGlobalCompact: the global coupled boundary face index for the face on the other 
        side of the local processor boundary
   
    Example:
        On proc0, neiBFaceGlobalCompact[0] = 1024, then we have the following:
       
                             localBFaceI = 0     <--proc0
                      ---------------------------   coupled boundary face
                             globalBFaceI=1024   <--proc1   
        Taken and modified from the extended stencil code in fvMesh
        Swap the global boundary face index
    */

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    neiBFaceGlobalCompact.setSize(daIndex_.nLocalBoundaryFaces);

    // initialize the list with -1, i.e., non coupled face
    forAll(neiBFaceGlobalCompact, idx)
    {
        neiBFaceGlobalCompact[idx] = -1;
    }

    // loop over the patches and store the global indices
    label counter = 0;
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                neiBFaceGlobalCompact[bFaceI] = daIndex_.globalCoupledBFaceNumbering.toGlobal(counter);
                faceIStart++;
                counter++;
            }
        }
    }

    // Swap the cell indices, the list now contains the global index for the
    // U state for the cell on the other side of the processor boundary
    syncTools::swapBoundaryFaceList(mesh_, neiBFaceGlobalCompact);

    return;
}

label DAJacCon::getLocalCoupledBFaceIndex(const label localFaceI) const
{
    /*
    Description:
        Calculate the index of the local inter-processor boundary face (bRow). 
    
    Input:
        localFaceI: The local face index. It is in a list of faces including all the
        internal and boundary faces.

    Output:
        bRow: A list of faces starts with the first inter-processor face. 
        See DAJacCon::globalBndNumbering_ for more details.
    */

    label counter = 0;
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
            label faceEnd = faceStart + patchSize;
            if (localFaceI >= faceStart && localFaceI < faceEnd)
            {
                // this face is on this patch, find the exact index
                label countDelta = localFaceI - pp.start(); //-faceStart;
                PetscInt bRow = counter + countDelta;
                return bRow;
            }
            else
            {
                //increment the counter by patchSize
                counter += patchSize;
            }
        }
    }

    // no match found
    FatalErrorIn("getLocalBndFaceIndex") << abort(FatalError);
    return -1;
}

void DAJacCon::setupStateBoundaryCon(Mat* stateBoundaryCon)
{
    /*
    Description:
        This function calculates DAJacCon::stateBoundaryCon_

    Output:
        stateBoundaryCon stores the level of connected states (on the other side 
        across the boundary) for a given coupled boundary face. stateBoundaryCon is 
        a matrix with sizes of nGlobalCoupledBFaces by nGlobalAdjointStates
        stateBoundaryCon is mainly used in the addBoundaryFaceConnection function
    
    Example:
        Basically, if there are 2 levels of connected states across the inter-proc boundary
    
                                       |<-----------proc0, globalBFaceI=1024
                               -----------------------------------  <-coupled boundary face
         globalAdjStateIdx=100 ->   | lv1 | <------ proc1
                                    |_____|
         globalAdjStateIdx=200 ->   | lv2 |
                                    |_____| 
                                   
        
        The indices for row 1024 in the stateBoundaryCon matrix will be
        stateBoundaryCon
        rowI=1024   
        Cols: colI=0 ...... colI=100  ........ colI=200 ......... colI=nGlobalAdjointStates
        Vals (level):           1                 2           
        NOTE: globalBFaceI=1024 is owned by proc0      
           
    */

    MatCreate(PETSC_COMM_WORLD, stateBoundaryCon);
    MatSetSizes(
        *stateBoundaryCon,
        daIndex_.nLocalCoupledBFaces,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(*stateBoundaryCon);
    MatMPIAIJSetPreallocation(*stateBoundaryCon, 1000, NULL, 1000, NULL);
    MatSeqAIJSetPreallocation(*stateBoundaryCon, 1000, NULL);
    MatSetOption(*stateBoundaryCon, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*stateBoundaryCon);
    MatZeroEntries(*stateBoundaryCon);

    Mat stateBoundaryConTmp;
    MatCreate(PETSC_COMM_WORLD, &stateBoundaryConTmp);
    MatSetSizes(
        stateBoundaryConTmp,
        daIndex_.nLocalCoupledBFaces,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(stateBoundaryConTmp);
    MatMPIAIJSetPreallocation(stateBoundaryConTmp, 1000, NULL, 1000, NULL);
    MatSeqAIJSetPreallocation(stateBoundaryConTmp, 1000, NULL);
    MatSetOption(stateBoundaryConTmp, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(stateBoundaryConTmp);
    MatZeroEntries(stateBoundaryConTmp);

    // loop over the patches and set the boundary connnectivity
    // Add connectivity in reverse so that the nearer stencils take priority

    // NOTE: we need to start with level 3, then to 2, then to 1, and flush the matrix
    // for each level before going to another level This is necessary because
    // we need to make sure a proper INSERT_VALUE behavior in MatSetValues
    // i.e., we found that if you use INSERT_VALUE to insert different values (e.g., 1, 2, and 3)
    // to a same rowI and colI in MatSetValues, and call Mat_Assembly in the end. The, the actual
    // value in rowI and colI is kind of random, it does not depend on which value is
    // insert first, in this case, it can be 1, 2, or 3... This happens only in parallel and
    // only happens after Petsc-3.8.4

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    // level 3 con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // This cell is already a neighbour cell, so we need this plus two
                // more levels
                // Start with next to nearest neighbours
                forAll(mesh_.cellCells()[idxN], cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    forAll(daIndex_.adjStateNames, idxI)
                    {
                        word stateName = daIndex_.adjStateNames[idxI];
                        if (daIndex_.adjStateType[stateName] != "surfaceScalarState")
                        {
                            // Now add level 3 connectivity, add all vars except for
                            // surfaceScalarStates
                            this->addConMatNeighbourCells(
                                *stateBoundaryCon,
                                gRow,
                                localCell,
                                stateName,
                                3.0);
                            this->addConMatNeighbourCells(
                                stateBoundaryConTmp,
                                gRow,
                                localCell,
                                stateName,
                                3.0);
                        }
                    }
                }
            }
        }
    }
    // NOTE: need to flush the value before assigning the next level
    MatAssemblyBegin(*stateBoundaryCon, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon, MAT_FLUSH_ASSEMBLY);
    MatAssemblyBegin(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // level 2 con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // now add the nearest neighbour cells, add all vars for level 2 except
                // for surfaceScalarStates
                forAll(daIndex_.adjStateNames, idxI)
                {
                    word stateName = daIndex_.adjStateNames[idxI];
                    if (daIndex_.adjStateType[stateName] != "surfaceScalarState")
                    {
                        this->addConMatNeighbourCells(
                            *stateBoundaryCon,
                            gRow,
                            idxN,
                            stateName,
                            2.0);
                        this->addConMatNeighbourCells(
                            stateBoundaryConTmp,
                            gRow,
                            idxN,
                            stateName,
                            2.0);
                    }
                }
            }
        }
    }
    // NOTE: need to flush the value before assigning the next level
    MatAssemblyBegin(*stateBoundaryCon, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon, MAT_FLUSH_ASSEMBLY);
    MatAssemblyBegin(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // face con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];
                // and add the surfaceScalarStates for idxN
                forAll(stateInfo_["surfaceScalarStates"], idxI)
                {
                    word stateName = stateInfo_["surfaceScalarStates"][idxI];
                    this->addConMatCellFaces(
                        *stateBoundaryCon,
                        gRow,
                        idxN,
                        stateName,
                        10.0); // for faces, its connectivity level is 10
                    this->addConMatCellFaces(
                        stateBoundaryConTmp,
                        gRow,
                        idxN,
                        stateName,
                        10.0);
                }
            }
        }
    }
    // NOTE: need to flush the value before assigning the next level
    MatAssemblyBegin(*stateBoundaryCon, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon, MAT_FLUSH_ASSEMBLY);
    MatAssemblyBegin(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // level 1 con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];
                // Add all the cell states for idxN
                forAll(daIndex_.adjStateNames, idxI)
                {
                    word stateName = daIndex_.adjStateNames[idxI];
                    if (daIndex_.adjStateType[stateName] != "surfaceScalarState")
                    {
                        this->addConMatCell(
                            *stateBoundaryCon,
                            gRow,
                            idxN,
                            stateName,
                            1.0);
                        this->addConMatCell(
                            stateBoundaryConTmp,
                            gRow,
                            idxN,
                            stateName,
                            1.0);
                    }
                }
            }
        }
    }
    // Now we can do the final assembly
    MatAssemblyBegin(*stateBoundaryCon, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon, MAT_FINAL_ASSEMBLY);

    // Now repeat loop adding boundary connections from other procs using matrix
    // created in the first loop.
    // Add connectivity in reverse so that the nearer stencils take priority

    // level 3 con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];

                // This cell is already a neighbour cell, so we need this plus two
                // more levels
                // Start with nearest neighbours
                forAll(mesh_.cellCells()[idxN], cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1 = {3};
                    // pass a zero list to add all states
                    List<List<word>> connectedStates(0);
                    this->addBoundaryFaceConnections(
                        stateBoundaryConTmp,
                        gRow,
                        localCell,
                        val1,
                        connectedStates,
                        0);
                }
            }
        }
    }
    // NOTE: need to flush the value before assigning the next level
    MatAssemblyBegin(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // level 2 and 3 con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];
                // now add the neighbour cells
                labelList vals2 = {2, 3};
                // pass a zero list to add all states
                List<List<word>> connectedStates(0);
                this->addBoundaryFaceConnections(
                    stateBoundaryConTmp,
                    gRow,
                    idxN,
                    vals2,
                    connectedStates,
                    0);
            }
        }
    }
    // NOTE: need to flush the value before assigning the next level
    MatAssemblyBegin(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // level 2 again, because the previous call will mess up level 2 con
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
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];

                // Now get the cell that borders this coupled bFace
                label idxN = pFaceCells[faceI];
                // now add the neighbour cells
                labelList vals1 = {2};
                // pass a zero list to add all states
                List<List<word>> connectedStates(0);
                this->addBoundaryFaceConnections(
                    stateBoundaryConTmp,
                    gRow,
                    idxN,
                    vals1,
                    connectedStates,
                    0);
            }
        }
    }

    MatAssemblyBegin(stateBoundaryConTmp, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(stateBoundaryConTmp, MAT_FINAL_ASSEMBLY);

    // the above repeat loop is not enough to cover all the stencil, we need to do more
    this->combineStateBndCon(stateBoundaryCon, &stateBoundaryConTmp);

    return;
}

void DAJacCon::combineStateBndCon(
    Mat* stateBoundaryCon,
    Mat* stateBoundaryConTmp)
{
    /*
    Description:
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
    
    Input/Output:
        stateBoundaryCon, and stateBoundaryConTmp should come from DAJacCon::stateBoundaryCon
    */

    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    PetscInt nCols1;
    const PetscInt* cols1;
    const PetscScalar* vals1;

    // Destroy and initialize stateBoundaryCon with zeros
    MatDestroy(stateBoundaryCon);
    MatCreate(PETSC_COMM_WORLD, stateBoundaryCon);
    MatSetSizes(
        *stateBoundaryCon,
        daIndex_.nLocalCoupledBFaces,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(*stateBoundaryCon);
    MatMPIAIJSetPreallocation(*stateBoundaryCon, 1000, NULL, 1000, NULL);
    MatSeqAIJSetPreallocation(*stateBoundaryCon, 1000, NULL);
    MatSetOption(*stateBoundaryCon, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*stateBoundaryCon);
    MatZeroEntries(*stateBoundaryCon); // initialize with zeros

    // assign stateBoundaryConTmp to stateBoundaryCon
    PetscInt Istart, Iend;
    MatGetOwnershipRange(*stateBoundaryConTmp, &Istart, &Iend);
    for (PetscInt i = Istart; i < Iend; i++)
    {
        MatGetRow(*stateBoundaryConTmp, i, &nCols, &cols, &vals);
        for (PetscInt j = 0; j < nCols; j++)
        {
            MatSetValue(*stateBoundaryCon, i, cols[j], vals[j], INSERT_VALUES);
        }
        MatRestoreRow(*stateBoundaryConTmp, i, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(*stateBoundaryCon, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon, MAT_FINAL_ASSEMBLY);

    MatDestroy(stateBoundaryConTmp);

    // copy ConMat to ConMatTmp for an extract loop
    MatConvert(*stateBoundaryCon, MATSAME, MAT_INITIAL_MATRIX, stateBoundaryConTmp);

    // We need to do another loop adding boundary connections from other procs using ConMat
    // this will add missing connectivity if the stateBoundaryCon stencil extends through
    // three more more processors
    // NOTE: we need to start with level 3, then to 2, then to 1, and flush the matrix
    // for each level before going to another level This is necessary because
    // we need to make sure a proper INSERT_VALUE behavior in MatSetValues
    // i.e., we found that if you use INSERT_VALUE to insert different values (e.g., 1, 2, and 3)
    // to a same rowI and colI in MatSetValues, and call Mat_Assembly in the end. The, the actual
    // value in rowI and colI is kind of random, it does not depend on which value is
    // insert first, in this case, it can be 1, 2, or 3... This happens only in parallel and
    // only happens after Petsc-3.8.4

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    // level 3 con
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        label faceIStart = pp.start();
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];
                label idxN = pFaceCells[faceI];

                forAll(mesh_.cellCells()[idxN], cellI)
                {
                    label localCell = mesh_.cellCells()[idxN][cellI];
                    labelList val1 = {3};
                    // pass a zero list to add all states
                    List<List<word>> connectedStates(0);
                    this->addBoundaryFaceConnections(
                        *stateBoundaryConTmp,
                        gRow,
                        localCell,
                        val1,
                        connectedStates,
                        0);
                }
            }
        }
    }
    MatAssemblyBegin(*stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // level 2, 3 con
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        label faceIStart = pp.start();
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];
                label idxN = pFaceCells[faceI];
                // now add the neighbour cells
                labelList vals2 = {2, 3};
                // pass a zero list to add all states
                List<List<word>> connectedStates(0);
                this->addBoundaryFaceConnections(
                    *stateBoundaryConTmp,
                    gRow,
                    idxN,
                    vals2,
                    connectedStates,
                    0);
            }
        }
    }
    MatAssemblyBegin(*stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryConTmp, MAT_FLUSH_ASSEMBLY);

    // level 2 again, because the previous call will mess up level 2 con
    forAll(patches, patchI)
    {
        const polyPatch& pp = patches[patchI];
        const UList<label>& pFaceCells = pp.faceCells();
        label faceIStart = pp.start();
        if (pp.coupled())
        {
            forAll(pp, faceI)
            {
                label bFaceI = faceIStart - daIndex_.nLocalInternalFaces;
                faceIStart++;
                label gRow = neiBFaceGlobalCompact_[bFaceI];
                label idxN = pFaceCells[faceI];
                // now add the neighbour cells
                labelList vals1 = {2};
                // pass a zero list to add all states
                List<List<word>> connectedStates(0);
                this->addBoundaryFaceConnections(
                    *stateBoundaryConTmp,
                    gRow,
                    idxN,
                    vals1,
                    connectedStates,
                    0);
            }
        }
    }
    MatAssemblyBegin(*stateBoundaryConTmp, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryConTmp, MAT_FINAL_ASSEMBLY);

    // Now stateBoundaryConTmp will have all the missing stencil. However, it will also mess
    // up the existing stencil in stateBoundaryCon. So we need to do a check to make sure that
    // stateBoundaryConTmp only add stencil, not replacing any existing stencil in stateBoundaryCon.
    // If anything in stateBoundaryCon is replaced, rollback the changes.
    Mat tmpMat; // create a temp mat
    MatCreate(PETSC_COMM_WORLD, &tmpMat);
    MatSetSizes(
        tmpMat,
        daIndex_.nLocalCoupledBFaces,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(tmpMat);
    MatMPIAIJSetPreallocation(tmpMat, 1000, NULL, 1000, NULL);
    MatSeqAIJSetPreallocation(tmpMat, 1000, NULL);
    MatSetOption(tmpMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(tmpMat);
    MatZeroEntries(tmpMat); // initialize with zeros
    for (PetscInt i = Istart; i < Iend; i++)
    {
        MatGetRow(*stateBoundaryCon, i, &nCols, &cols, &vals);
        MatGetRow(*stateBoundaryConTmp, i, &nCols1, &cols1, &vals1);
        for (PetscInt j = 0; j < nCols1; j++)
        {
            // for each col in stateBoundaryConTmp, we need to check if there are any existing
            // values for the same col in stateBoundaryCon. If yes, assign the val from
            // stateBoundaryCon instead of stateBoundaryConTmp
            PetscScalar newVal = vals1[j];
            PetscInt newCol = cols1[j];
            for (PetscInt k = 0; k < nCols; k++)
            {
                if (int(cols[k]) == int(cols1[j]))
                {
                    newVal = vals[k];
                    newCol = cols[k];
                    break;
                }
            }
            MatSetValue(tmpMat, i, newCol, newVal, INSERT_VALUES);
        }
        MatRestoreRow(*stateBoundaryCon, i, &nCols, &cols, &vals);
        MatRestoreRow(*stateBoundaryConTmp, i, &nCols1, &cols1, &vals1);
    }
    MatAssemblyBegin(tmpMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(tmpMat, MAT_FINAL_ASSEMBLY);

    // copy ConMat to ConMatTmp
    MatDestroy(stateBoundaryCon);
    MatConvert(tmpMat, MATSAME, MAT_INITIAL_MATRIX, stateBoundaryCon);
    MatAssemblyBegin(*stateBoundaryCon, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryCon, MAT_FINAL_ASSEMBLY);

    MatDestroy(stateBoundaryConTmp);
    MatDestroy(&tmpMat);

    return;
}

void DAJacCon::setupStateBoundaryConID(Mat* stateBoundaryConID)
{
    /*
    Description:
        This function computes DAJacCon::stateBoundaryConID_.

    Output:
        stateBoundaryConID: it has the exactly same structure as DAJacCon::stateBoundaryCon_ 
        except that stateBoundaryConID stores the connected stateID instead of connected 
        levels. stateBoundaryConID will be used in DAJacCon::addBoundaryFaceConnections
    */

    PetscInt nCols, colI;
    const PetscInt* cols;
    const PetscScalar* vals;
    PetscInt Istart, Iend;

    PetscScalar valIn;

    // assemble adjStateID4GlobalAdjIdx
    // adjStateID4GlobalAdjIdx stores the adjStateID for given a global adj index
    labelList adjStateID4GlobalAdjIdx;
    adjStateID4GlobalAdjIdx.setSize(daIndex_.nGlobalAdjointStates);
    daIndex_.calcAdjStateID4GlobalAdjIdx(adjStateID4GlobalAdjIdx);

    // initialize
    MatCreate(PETSC_COMM_WORLD, stateBoundaryConID);
    MatSetSizes(
        *stateBoundaryConID,
        daIndex_.nLocalCoupledBFaces,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(*stateBoundaryConID);
    MatMPIAIJSetPreallocation(*stateBoundaryConID, 1000, NULL, 1000, NULL);
    MatSeqAIJSetPreallocation(*stateBoundaryConID, 1000, NULL);
    MatSetOption(*stateBoundaryConID, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*stateBoundaryConID);
    MatZeroEntries(*stateBoundaryConID);

    MatGetOwnershipRange(stateBoundaryCon_, &Istart, &Iend);

    // set stateBoundaryConID_ based on stateBoundaryCon_ and adjStateID4GlobalAdjIdx
    for (PetscInt i = Istart; i < Iend; i++)
    {
        MatGetRow(stateBoundaryCon_, i, &nCols, &cols, &vals);
        for (PetscInt j = 0; j < nCols; j++)
        {
            if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
            {
                colI = cols[j];
                valIn = adjStateID4GlobalAdjIdx[colI];
                MatSetValue(*stateBoundaryConID, i, colI, valIn, INSERT_VALUES);
            }
        }
        MatRestoreRow(stateBoundaryCon_, i, &nCols, &cols, &vals);
    }

    adjStateID4GlobalAdjIdx.clear();

    MatAssemblyBegin(*stateBoundaryConID, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*stateBoundaryConID, MAT_FINAL_ASSEMBLY);

    return;
}

void DAJacCon::addConMatCell(
    Mat conMat,
    const label gRow,
    const label cellI,
    const word stateName,
    const PetscScalar val)
{
    /* 
    Description:
        Insert a value (val) to the connectivity Matrix (conMat)
        This value will be inserted at rowI=gRow
        The column index is dependent on the cellI and stateName

    Input:
        gRow: which row to insert the value for conMat

        cellI: the index of the cell to compute the column index to add

        stateName: the name of the state variable to compute the column index to add

        val: the value to add to conMat

    Output:
        conMat: the matrix to add value to

    Example:

        If we want to add value 1.0 to conMat for
        column={the U globalAdjointIndice of cellI} where cellI=5
        row = gRow = 100
        Then, call addConMatCell(conMat, 100, 5, "U", 1.0)
    
                 -------  
                | cellI | <----------- value 1.0 will be added to 
                 -------               column = {global index of U 
                                       for cellI}
    */

    PetscInt idxJ, idxI;

    idxI = gRow;

    // find the global index of this state
    label compMax = 1;
    if (daIndex_.adjStateType[stateName] == "volVectorState")
    {
        compMax = 3;
    }

    for (label i = 0; i < compMax; i++)
    {
        idxJ = daIndex_.getGlobalAdjointStateIndex(stateName, cellI, i);
        // set it in the matrix
        MatSetValues(conMat, 1, &idxI, 1, &idxJ, &val, INSERT_VALUES);
    }

    return;
}

void DAJacCon::addConMatNeighbourCells(
    Mat conMat,
    const label gRow,
    const label cellI,
    const word stateName,
    const PetscScalar val)
{

    /*
    Description:
        Insert a value (val) to the connectivity Matrix (conMat)
        This value will be inserted at rowI=gRow
        The column index is dependent on the cellI, and cellI's neibough and stateName

    Input:
        gRow: which row to insert the value for conMat

        cellI: the index of the cell to compute the column index to add

        stateName: the name of the state variable to compute the column index to add

        val: the value to add to conMat

    Output:
        conMat: the matrix to add value to

    Example:

        If we want to add value 1.0 to conMat for
        columns={the U globalAdjointIndice of all the neiboughs of cellI} where cellI=5
        row = gRow = 100
        Then, call addConMatNeighbourCells(conMat, 100, 5, "U", 1.0)
    
                 -------  
                | cellL | <----------- value 1.0 will be added to 
         ------- ------- -------       column = {global index of U 
        | cellJ | cellI | cellK |      for cellL}, similarly for all 
         ------- ------- -------       the neiboughs of cellI
                | cellM |
                 -------
    */

    label localCellJ;
    PetscInt idxJ, idxI;

    idxI = gRow;
    // Add the nearest neighbour cells for cell
    forAll(mesh_.cellCells()[cellI], cellJ)
    {
        // get the local neighbour cell
        localCellJ = mesh_.cellCells()[cellI][cellJ];

        // find the global index of this state
        label compMax = 1;
        if (daIndex_.adjStateType[stateName] == "volVectorState")
        {
            compMax = 3;
        }
        for (label i = 0; i < compMax; i++)
        {
            idxJ = daIndex_.getGlobalAdjointStateIndex(stateName, localCellJ, i);
            // set it in the matrix
            MatSetValues(conMat, 1, &idxI, 1, &idxJ, &val, INSERT_VALUES);
        }
    }

    return;
}

void DAJacCon::addConMatCellFaces(
    Mat conMat,
    const label gRow,
    const label cellI,
    const word stateName,
    const PetscScalar val)
{

    /* 
    Description:
        Insert a value (val) to the connectivity Matrix (conMat)
        This value will be inserted at rowI=gRow
        The column index is dependent on the cellI's faces and stateName

    Input:
        gRow: which row to insert the value for conMat

        cellI: the index of the cell to compute the column index to add

        stateName: the name of the state variable to compute the column index to add

        val: the value to add to conMat

    Output:
        conMat: the matrix to add value to

    Example:

        If we want to add value 10.0 to conMat for
        columns={the phi globalAdjointIndice of cellI's faces} where cellI=5
        row = gRow = 100
        Then, call addConMatCell(conMat, 100, 5, "U", 1.0)
    
                 -------  
                | cellI | <----------- value 10.0 will be added to 
                 -------               column = {global adjoint index 
                                       of all cellI's faces}
    */

    PetscInt idxJ, idxI;
    idxI = gRow;

    // get the faces connected to this cell, note these are in a single
    // list that includes all internal and boundary faces
    const labelList& faces = mesh_.cells()[cellI];
    forAll(faces, idx)
    {
        //get the appropriate index for this face
        label globalState = daIndex_.getGlobalAdjointStateIndex(stateName, faces[idx]);
        idxJ = globalState;
        MatSetValues(conMat, 1, &idxI, 1, &idxJ, &val, INSERT_VALUES);
    }

    return;
}

void DAJacCon::addBoundaryFaceConnections(
    Mat conMat,
    const label gRow,
    const label cellI,
    const labelList v,
    const List<List<word>> connectedStates,
    const label addFaces)
{
    /*
    Description:
        This function adds inter-proc connectivity into conMat.
        For all the inter-proc faces owned by cellI, get the global adj state indices 
        from DAJacCon::stateBoundaryCon_ and then add them into conMat
        Col index to add: the same col index for a given row (bRowGlobal) in the stateBoundaryCon 
        mat if the element value in the stateBoundaryCon mat is less than the input level, 
        i.e., v.size().
    
    Input:
        gRow: Row index to add
    
        cellI: the cell index for getting the faces to add inter-proc connectivity, NOTE: depending on the level
        of requested connections, we may add inter-proc face that are not belonged to cellI
    
        v: an array denoting the desired values to add, the size of v denotes the maximal levels to add
    
        connectedStates: selectively add some states into the conMat for the current level. If its size is 0,
        add all the possible states (except for surfaceStates). The dimension of connectedStates is nLevel 
        by nStates.
    
        addFaces: whether to add indices for face (phi) connectivity
    
    Example:
    
        labelList val2={1,2};
        PetscInt gRow=1024, idxN = 100, addFaces=1;
        wordListList connectedStates={{"U","p"},{"U"}};
        addBoundaryFaceConnections(stateBoundaryCon,gRow,idxN,vals2,connectedStates,addFaces);
        The above call will add 2 levels of connected states for all the inter-proc faces belonged to cellI=idxN
        The cols to added are: the level1 connected states (U, p) for all the inter-proc faces belonged to 
        cellI=idxN. the level2 connected states (U only) for all the inter-proc faces belonged to cellI=idxN
        The valus "1" will be added to conMat for all the level1 connected states while the value "2" will be 
        added for level2. 
        Note: this function will also add all faces belonged to level1 of the inter-proc faces, see the 
        following for reference
        
                                    -------
                                    | idxN|
                                    |     |       proc0, idxN=100, globalBFaceI=1024 for the south face of idxN
                               -----------------  <----coupled boundary face
         add state U and p ----->   | lv1 |       proc1
         also add faces -------->   |     |
                                    -------
                                    | lv2 |
         add state U  ---------->   |     |
                                    -------
        
    
        ****** NOTE: *******
        If the inter-proc boundary is like the following, calling this function will NOT add any 
        inter-proc connection for idxN because there is no inter-proc boundary for cell idxN
    
                -------
                | idxN|
                |     |       
                -------  <--------- there is no inter-proc boundary for idxN, not adding anything
                | lv1 |      
                |     |        proc0
          ------------------- <----coupled boundary face
                | lv2 |        proc1
                |     |
                -------
        
    */

    if (v.size() != connectedStates.size() && connectedStates.size() != 0)
    {
        FatalErrorIn("") << "size of v and connectedStates are not identical!"
                         << abort(FatalError);
    }

    PetscInt idxJ, idxI, bRow, bRowGlobal;
    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    PetscInt nColsID;
    const PetscInt* colsID;
    const PetscScalar* valsID;

    // convert stateNames to stateIDs
    labelListList connectedStateIDs(connectedStates.size());
    forAll(connectedStates, idxI)
    {
        forAll(connectedStates[idxI], idxJ)
        {
            word stateName = connectedStates[idxI][idxJ];
            label stateID = daIndex_.adjStateID[stateName];
            connectedStateIDs[idxI].append(stateID);
        }
    }

    idxI = gRow;
    // get the faces connected to this cell, note these are in a single
    // list that includes all internal and boundary faces
    const labelList& faces = mesh_.cells()[cellI];

    //get the level
    label level = v.size();

    for (label lv = level; lv >= 1; lv--) // we need to start from the largest levels since they have higher priority
    {
        forAll(faces, faceI)
        {
            // Now deal with coupled faces
            label currFace = faces[faceI];

            if (daIndex_.isCoupledFace[currFace])
            {
                //this is a coupled face

                // use the boundary connectivity to figure out what is connected
                // to this face for this level

                // get bRow in boundaryCon for this face
                bRow = this->getLocalCoupledBFaceIndex(currFace);

                // get the global bRow index
                bRowGlobal = daIndex_.globalCoupledBFaceNumbering.toGlobal(bRow);

                // now extract the boundaryCon row
                MatGetRow(stateBoundaryCon_, bRowGlobal, &nCols, &cols, &vals);
                if (connectedStates.size() != 0)
                {
                    // check if we need to get stateID
                    MatGetRow(stateBoundaryConID_, bRowGlobal, &nColsID, &colsID, &valsID);
                }

                // now loop over the row and set any column that match this level
                // in conMat
                for (label i = 0; i < nCols; i++)
                {
                    idxJ = cols[i];
                    label val = round(vals[i]); // val is the connectivity level extracted from stateBoundaryCon_ at this col
                    // selectively add some states into conMat
                    label addState;
                    label stateID = -9999;
                    // check if we need to get stateID
                    if (connectedStates.size() != 0)
                    {
                        stateID = round(valsID[i]);
                    }

                    if (connectedStates.size() == 0)
                    {
                        addState = 1;
                    }
                    else if (DAUtility::isInList<label>(stateID, connectedStateIDs[lv - 1]))
                    {
                        addState = 1;
                    }
                    else
                    {
                        addState = 0;
                    }
                    // if the level match and the state is what you want
                    if (val == lv && addState)
                    {
                        // need to do v[lv-1] here since v is an array with starting index 0
                        PetscScalar valIn = v[lv - 1];
                        MatSetValues(conMat, 1, &idxI, 1, &idxJ, &valIn, INSERT_VALUES);
                    }
                    if (val == 10 && addFaces)
                    {
                        // this is a necessary connection
                        PetscScalar valIn = v[lv - 1];
                        MatSetValues(conMat, 1, &idxI, 1, &idxJ, &valIn, INSERT_VALUES);
                    }
                }

                // restore the row of the matrix
                MatRestoreRow(stateBoundaryCon_, bRowGlobal, &nCols, &cols, &vals);
                if (connectedStates.size() != 0)
                {
                    // check if we need to get stateID
                    MatRestoreRow(stateBoundaryConID_, bRowGlobal, &nColsID, &colsID, &valsID);
                }
            }
        }
    }

    return;
}

label DAJacCon::coloringExists(const word postFix) const
{
    /*
    Description: 
        Check whether the coloring file exists
    
    Input:
        postFix: the post fix of the file name, e.g., the original
        name is dFdWColoring_1.bin, then the new name is
        dFdWColoring_drag_1.bin with postFix = _drag

    Output:
        return 1 if coloring files exist, otherwise, return 0
    */

    Info << "Checking if Coloring file exists.." << endl;
    label nProcs = Pstream::nProcs();
    word fileName = modelType_ + "Coloring" + postFix + "_" + Foam::name(nProcs);
    word checkFile = fileName + ".bin";
    std::ifstream fIn(checkFile);
    if (!fIn.fail())
    {
        Info << checkFile << " exists." << endl;
        return 1;
    }
    else
    {
        return 0;
    }
}

void DAJacCon::calcJacConColoring(const word postFix)
{
    /*
    Description:
        Calculate the coloring for jacCon.

    Input:
        postFix: the post fix of the file name, e.g., the original
        name is dFdWColoring_1.bin, then the new name is
        dFdWColoring_drag_1.bin with postFix = _drag

    Output:
        jacConColors_: jacCon coloring and save to files. 
        The naming convention for coloring vector is 
        coloringVecName_nProcs.bin. This is necessary because 
        using different CPU cores result in different jacCon 
        and therefore different coloring
    
        nJacColors: number of jacCon colors

    */

    // first check if the file name exists, if yes, return and
    // don't compute the coloring
    Info << "Calculating " << modelType_ << " Coloring.." << endl;
    label nProcs = Pstream::nProcs();
    word fileName = modelType_ + "Coloring" + postFix + "_" + Foam::name(nProcs);

    VecZeroEntries(jacConColors_);
    if (daOption_.getOption<label>("adjUseColoring"))
    {
        // use parallelD2 coloring to compute colors
        daColoring_.parallelD2Coloring(jacCon_, jacConColors_, nJacConColors_);
    }
    else
    {
        // use brute force to compute colors, basically, we assign
        // color to its global index
        PetscScalar* jacConColorsArray;
        VecGetArray(jacConColors_, &jacConColorsArray);
        label Istart, Iend;
        VecGetOwnershipRange(jacConColors_, &Istart, &Iend);
        for (label i = Istart; i < Iend; i++)
        {
            label relIdx = i - Istart;
            jacConColorsArray[relIdx] = i * 1.0;
        }
        VecRestoreArray(jacConColors_, &jacConColorsArray);

        PetscReal maxVal;
        VecMax(jacConColors_, NULL, &maxVal);
        nJacConColors_ = maxVal + 1;
    }

    daColoring_.validateColoring(jacCon_, jacConColors_);
    Info << " nJacConColors: " << nJacConColors_ << endl;
    // write jacCon colors
    Info << "Writing Colors to " << fileName << endl;
    DAUtility::writeVectorBinary(jacConColors_, fileName);

    return;
}

void DAJacCon::readJacConColoring(const word postFix)
{
    /*
    Description:
        Read the jacCon coloring from files and 
        compute nJacConColors. The naming convention for
        coloring vector is coloringVecName_nProcs.bin
        This is necessary because using different CPU
        cores result in different jacCon and therefore
        different coloring

    Input:
        postFix: the post fix of the file name, e.g., the original
        name is dFdWColoring_1.bin, then the new name is
        dFdWColoring_drag_1.bin with postFix = _drag

    Output:
        jacConColors_: read from file

        nJacConColors: number of jacCon colors
    */

    label nProcs = Pstream::nProcs();
    word fileName = modelType_ + "Coloring" + postFix + "_" + Foam::name(nProcs);
    Info << "Reading Coloring " << fileName << endl;

    VecZeroEntries(jacConColors_);
    DAUtility::readVectorBinary(jacConColors_, fileName);

    daColoring_.validateColoring(jacCon_, jacConColors_);

    PetscReal maxVal;
    VecMax(jacConColors_, NULL, &maxVal);
    nJacConColors_ = maxVal + 1;
}

void DAJacCon::setupJacConPreallocation(const dictionary& options)
{
    /*
    Description:
        Compute the preallocation vectors.
        NOTE: this need to be implemented in the child class, if not,
        print an error! For example, for dRdW, this needs to be implemented;
        however, for dFdW, no setupJacConPreallocation is needed so users
        shouldn't call this function at all!
    */
    FatalErrorIn("") << "setupJacConPreallocation not implemented " << endl
                     << " in the child class for " << modelType_
                     << abort(FatalError);
}

void DAJacCon::preallocatedRdW(
    Mat dRMat,
    const label transposed) const
{
    /*
    Description:
        Preallocate the dRMat
        NOTE: this need to be implemented in the child class, if not,
        print an error! For example, for dRdW, this needs to be implemented;
        however, for dFdW, no preallocatedRdW is needed so users
        shouldn't call this function at all!
    */
    FatalErrorIn("") << "preallocatedRdW not implemented " << endl
                     << " in the child class for " << modelType_
                     << abort(FatalError);
}

void DAJacCon::setObjFuncVec(
    scalarList objFuncFaceValues,
    scalarList objFuncCellValues,
    Vec objFuncVec) const
{
    /*
    Description:
        Set the objective function vector
        NOTE: this need to be implemented in the child class, if not,
        print an error! For example, for dFdW, this needs to be implemented;
        however, for dRdW, no setObjFuncVec is needed so users
        shouldn't call this function at all!
    */
    FatalErrorIn("") << "setObjFuncVec not implemented " << endl
                     << " in the child class for " << modelType_
                     << abort(FatalError);
}

void DAJacCon::calcColoredColumns(
    const label colorI,
    Vec coloredColumn) const
{
    /*
    Description:
        Compute the colored column vector: coloredColumn. This vector will then
        be used to assign resVec to dRdW in DAPartDeriv::calcPartDeriv

    Input:
        colorI: the ith color index

    Output:
        coloredColumn: For a given colorI, coloredColumn vector contains the column 
        index for non-zero elements in the DAJacCon::jacCon_ matrix. If there is 
        no non-zero element for this color, set the value to -1

    Example:

        If the DAJacCon::jacCon_ matrix reads,
    
               color0  color1
                 |     |
                 1  0  0  0
        jacCon = 0  1  1  0
                 0  0  1  0
                 0  0  0  1
                    |     | 
                color0   color0
    
        and the coloring vector DAJacCon::jacConColors_ = {0, 0, 1, 0}.
        
        **************************
        ***** If colorI = 0 ******
        **************************
        Calling calcColoredColumns(0, coloredColumn) will return 
    
        coloredColumn = {0, 1, -1, 3}
    
        In this case, we have three columns (0, 1, and 3) for color=0, the nonzero pattern is:
    
               color0  
                 |     
                 1  0  0  0
        jacCon = 0  1  0  0
                 0  0  0  0
                 0  0  0  1
                    |     | 
                color0   color0
        
        So for the 0th, 1th, and 3th rows, the non-zero elements in the jacCon matrix are at the 
        0th, 1th, and 3th columns, respectively, which gives coloredColumn = {0, 1, -1, 3}
    
        **************************
        ***** If colorI = 1 ******
        **************************
        Calling calcColoredColumns(1, coloredColumn) will return 
    
        coloredColumn = {-1, 2, 2, -1}
    
        In this case, we have one column (2) for color=1, , the nonzero pattern is:
    
                     color1
                       |
                 0  0  0  0
        jacCon = 0  0  1  0
                 0  0  1  0
                 0  0  0  0
    
        So for the 1th and 2th rows, the non-zero elements in the jacCon matrix are at the 
        2th and 2th columns, respectively, which gives coloredColumn = {-1, 2, 2, -1}

    */

    if (daOption_.getOption<label>("adjUseColoring"))
    {

        Vec colorIdx;
        label Istart, Iend;

        /* for the current color, determine which row/column pairs match up. */

        // create a vector to hold the column indices associated with this color
        VecDuplicate(jacConColors_, &colorIdx);
        VecZeroEntries(colorIdx);

        // Start by looping over the color vector. Set each column index associated
        // with the current color to its own value in the color idx vector
        // get the values on this proc
        VecGetOwnershipRange(jacConColors_, &Istart, &Iend);

        // create the arrays to access them directly
        const PetscScalar* jacConColorsArray;
        PetscScalar* colorIdxArray;
        VecGetArrayRead(jacConColors_, &jacConColorsArray);
        VecGetArray(colorIdx, &colorIdxArray);

        // loop over the entries to find the ones that match this color
        for (label j = Istart; j < Iend; j++)
        {
            label idx = j - Istart;
            if (DAUtility::isValueCloseToRef(jacConColorsArray[idx], colorI * 1.0))
            {
                // use 1 based indexing here and then subtract 1 from all values in
                // the mat mult. This will handle the zero index case in the first row
                colorIdxArray[idx] = j + 1;
            }
        }
        VecRestoreArrayRead(jacConColors_, &jacConColorsArray);
        VecRestoreArray(colorIdx, &colorIdxArray);

        //VecAssemblyBegin(colorIdx);
        //VecAssemblyEnd(colorIdx);

        //Set coloredColumn to -1 to account for the 1 based indexing in the above loop
        VecSet(coloredColumn, -1);
        // Now do a MatVec with the conMat to get the row coloredColumn pairs.
        MatMultAdd(jacCon_, colorIdx, coloredColumn, coloredColumn);

        // destroy the temporary vector
        VecDestroy(&colorIdx);
    }
    else
    {
        // uncolored case, we just set all elements to colorI
        PetscScalar val = colorI * 1.0;
        VecSet(coloredColumn, val);

        VecAssemblyBegin(coloredColumn);
        VecAssemblyEnd(coloredColumn);
    }
}

void DAJacCon::checkSpecialBCs()
{
    /*
    Description:
        Check if there is special boundary conditions that need special treatment in jacCon_
    */

    // *******************************************************************
    //                     pressureInletVelocity
    // *******************************************************************
    if (DAUtility::isInList<word>("pressureInletVelocity", daField_.specialBCs))
    {
        // we need special treatment for connectivity
        Info << "pressureInletVelocity detected, applying special treatment for connectivity." << endl;
        // initialize and compute isPIVBCState_
        VecCreate(PETSC_COMM_WORLD, &isPIVBCState_);
        VecSetSizes(isPIVBCState_, daIndex_.nLocalAdjointStates, PETSC_DECIDE);
        VecSetFromOptions(isPIVBCState_);
        VecZeroEntries(isPIVBCState_);

        // compute isPIVBCState_
        // Note we need to read the U field, instead of getting it from db
        // this is because coloringSolver does not read U
        Info << "Calculating pressureInletVelocity state vec..." << endl;
        volVectorField U(
            IOobject(
                "U",
                mesh_.time().timeName(),
                mesh_,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE,
                false),
            mesh_,
            dimensionedVector("U", dimensionSet(0, 1, -1, 0, 0, 0, 0), vector::zero),
            zeroGradientFvPatchField<vector>::typeName);

        forAll(U.boundaryField(), patchI)
        {
            if (U.boundaryFieldRef()[patchI].type() == "pressureInletVelocity")
            {
                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchI].faceCells();
                forAll(U.boundaryFieldRef()[patchI], faceI)
                {

                    label faceCellI = pFaceCells[faceI]; // lv 1 face neibouring cells
                    this->setPIVVec(isPIVBCState_, faceCellI);

                    forAll(mesh_.cellCells()[faceCellI], cellJ)
                    {

                        label faceCellJ = mesh_.cellCells()[faceCellI][cellJ]; // lv 2 face neibouring cells
                        this->setPIVVec(isPIVBCState_, faceCellJ);

                        forAll(mesh_.cellCells()[faceCellJ], cellK)
                        {
                            label faceCellK = mesh_.cellCells()[faceCellI][cellJ]; // lv 3 face neibouring cells
                            this->setPIVVec(isPIVBCState_, faceCellK);
                        }
                    }
                }
            }
        }

        VecAssemblyBegin(isPIVBCState_);
        VecAssemblyEnd(isPIVBCState_);

        wordList writeJacobians;
        daOption_.getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
        if (writeJacobians.found("isPIVVec"))
        {
            DAUtility::writeVectorASCII(isPIVBCState_, "isPIVVec");
        }
    }

    // *******************************************************************
    //                      append more special BCs
    // *******************************************************************

    return;
}

void DAJacCon::setPIVVec(
    Vec isPIV,
    const label cellI)
{
    forAll(daIndex_.adjStateNames, idxI)
    {
        // get stateName and residual names
        word stateName = daIndex_.adjStateNames[idxI];

        // check if this state is a cell state, we do surfaceScalarState  separately
        if (daIndex_.adjStateType[stateName] == "surfaceScalarState")
        {
            forAll(mesh_.cells()[cellI], faceI)
            {
                label cellFaceI = mesh_.cells()[cellI][faceI];
                label rowI = daIndex_.getGlobalAdjointStateIndex(stateName, cellFaceI);
                VecSetValue(isPIV, rowI, 1.0, INSERT_VALUES);
            }
        }
        else
        {
            // if it is a vectorState, set compMax=3
            label compMax = 1;
            if (daIndex_.adjStateType[stateName] == "volVectorState")
            {
                compMax = 3;
            }

            for (label comp = 0; comp < compMax; comp++)
            {
                label rowI = daIndex_.getGlobalAdjointStateIndex(stateName, cellI, comp);
                VecSetValue(isPIV, rowI, 1.0, INSERT_VALUES);
            }
        }
    }
}

label DAJacCon::addPhi4PIV(
    const word stateName,
    const label idxI,
    const label comp)
{

    label localI = daIndex_.getLocalAdjointStateIndex(stateName, idxI, comp);
    PetscScalar* isPIVArray;
    VecGetArray(isPIVBCState_, &isPIVArray);
    if (DAUtility::isValueCloseToRef(isPIVArray[localI], 1.0))
    {
        VecRestoreArray(isPIVBCState_, &isPIVArray);
        return 1;
    }
    else
    {
        VecRestoreArray(isPIVBCState_, &isPIVArray);
        return 0;
    }

    VecRestoreArray(isPIVBCState_, &isPIVArray);
    return 0;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
