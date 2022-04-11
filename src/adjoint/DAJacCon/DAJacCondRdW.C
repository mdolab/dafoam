/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAJacCondRdW.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAJacCondRdW, 0);
addToRunTimeSelectionTable(DAJacCon, DAJacCondRdW, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAJacCondRdW::DAJacCondRdW(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAJacCon(modelType, mesh, daOption, daModel, daIndex)
{
    this->initializeStateBoundaryCon();
    this->initializePetscVecs();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAJacCondRdW::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAJacCon. Here we need to delete all members
        in the parent and child classes
    */

    VecDestroy(&dRdWTPreallocOn_);
    VecDestroy(&dRdWTPreallocOff_);
    VecDestroy(&dRdWPreallocOn_);
    VecDestroy(&dRdWPreallocOff_);

    this->clearDAJacConMembers();
}
void DAJacCondRdW::initializePetscVecs()
{
    /*
    Description:
        Initialize dRdWTPreallocOn_, dRdWTPreallocOff_, dRdWPreallocOn_,
        dRdWPreallocOff_, and jacConColors_
    
    */

    // initialize the preallocation vecs
    VecCreate(PETSC_COMM_WORLD, &dRdWTPreallocOn_);
    VecSetSizes(dRdWTPreallocOn_, daIndex_.nLocalAdjointStates, PETSC_DECIDE);
    VecSetFromOptions(dRdWTPreallocOn_);

    VecDuplicate(dRdWTPreallocOn_, &dRdWTPreallocOff_);
    VecDuplicate(dRdWTPreallocOn_, &dRdWPreallocOn_);
    VecDuplicate(dRdWTPreallocOn_, &dRdWPreallocOff_);

    // initialize coloring vectors

    // dRdW Colors
    VecCreate(PETSC_COMM_WORLD, &jacConColors_);
    VecSetSizes(jacConColors_, daIndex_.nLocalAdjointStates, PETSC_DECIDE);
    VecSetFromOptions(jacConColors_);

    return;
}

void DAJacCondRdW::setupJacConPreallocation(const dictionary& options)
{
    /*
    Description:
        Setup the connectivity mat preallocation vectors:

        dRdWTPreallocOn_
        dRdWTPreallocOff_
        dRdWPreallocOn_
        dRdWPreallocOff_
    
    Input:
        options.stateResConInfo: a hashtable that contains the connectivity
        information for dRdW, usually obtained from Foam::DAStateInfo
    */

    HashTable<List<List<word>>> stateResConInfo;
    options.readEntry<HashTable<List<List<word>>>>("stateResConInfo", stateResConInfo);

    label isPrealloc = 1;
    this->setupdRdWCon(stateResConInfo, isPrealloc);
}

void DAJacCondRdW::setupJacCon(const dictionary& options)
{
    /*
    Description:
        Setup DAJacCon::jacCon_
    
    Input:
        options.stateResConInfo: a hashtable that contains the connectivity
        information for dRdW, usually obtained from Foam::DAStateInfo
    */

    HashTable<List<List<word>>> stateResConInfo;
    options.readEntry<HashTable<List<List<word>>>>("stateResConInfo", stateResConInfo);

    label isPrealloc = 0;
    this->setupdRdWCon(stateResConInfo, isPrealloc);
}

void DAJacCondRdW::initializeJacCon(const dictionary& options)
{
    /*
    Description:
        Initialize the connectivity matrix and preallocate memory
    
    Input:
        options: it is not used.
    */

    MatCreate(PETSC_COMM_WORLD, &jacCon_);
    MatSetSizes(
        jacCon_,
        daIndex_.nLocalAdjointStates,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(jacCon_);

    this->preallocateJacobianMatrix(
        jacCon_,
        dRdWPreallocOn_,
        dRdWPreallocOff_);
    //MatSetOption(jacCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacCon_);

    if (daOption_.getOption<label>("debug"))
    {
        Info << "Connectivity matrix initialized." << endl;
    }
}

void DAJacCondRdW::setupdRdWCon(
    const HashTable<List<List<word>>>& stateResConInfo,
    const label isPrealloc)
{
    /*
    Description:
        Calculates the state Jacobian connectivity mat DAJacCon::jacCon_ or 
        computing the preallocation vectors DAJacCondRdW::dRdWPreallocOn_ and 
        DAJacCondRdW::dRdWPreallocOff_

    Input:
        stateResConInfo: a hashtable that contains the connectivity
        information for dRdW, usually obtained from Foam::DAStateInfo

        isPrealloc == 1: calculate the preallocation vectors, else, calculate jacCon_

    Output:
        DAJacCondRdW::dRdWPreallocOn_: preallocation vector that stores the number of 
        on-diagonal conectivity for each row
    
        DAJacCon::jacCon_: state Jacobian connectivity mat with dimension 
        sizeAdjStates by sizeAdjStates. jacCon_ has the same non-zero pattern as Jacobian mat.
        The difference is that jacCon_ has values one for all non-zero values, so jacCon_
        may look like this
    
                    1 1 0 0 1 0 
                    1 1 1 0 0 1 
                    0 1 1 1 0 0 
        jacCon_ =   1 0 1 1 1 0 
                    0 1 0 1 1 1
                    0 0 1 0 0 1
    
    Example:
        The way setupJacCon works is that we call the DAJacCon::addStateConnections function
        to add connectivity for each row of DAJacCon::jacCon_.
        
        Here we need to loop over all cellI and add a certain number levels of connected states.
        If the connectivity list reads:
    
        stateResConInfo
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
        are done in this function

    */

    label globalIdx;
    // connectedStatesP: one row matrix that stores the actual connectivity,
    // element value 1 denotes a connected state. connectedStatesP is then used to
    // assign dRdWPreallocOn or dRdWCon
    Mat connectedStatesP;

    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    PetscInt nColsID;
    const PetscInt* colsID;
    const PetscScalar* valsID;

    if (isPrealloc)
    {
        VecZeroEntries(dRdWPreallocOn_);
        VecZeroEntries(dRdWPreallocOff_);
        VecZeroEntries(dRdWTPreallocOn_);
        VecZeroEntries(dRdWTPreallocOff_);
    }

    if (daOption_.getOption<label>("debug"))
    {
        if (isPrealloc)
        {

            Info << "Computing preallocating vectors for Jacobian connectivity mat" << endl;
        }
        else
        {
            Info << "Setup Jacobian connectivity mat" << endl;
        }
    }

    // loop over all cell residuals, we bascially need to compute all
    // the input parameters for the DAJacCondRdW::addStateConnections function, then call
    // the function to get connectedStatesP (matrix that contains one row of connectivity
    // in DAJacCondRdW::jacCon_). Check DAJacCondRdW::addStateConnections for detail usages
    forAll(daIndex_.adjStateNames, idxI)
    {
        // get stateName and residual names
        word stateName = daIndex_.adjStateNames[idxI];
        word resName = stateName + "Res";

        // check if this state is a cell state, we do surfaceScalarState residuals separately
        if (daIndex_.adjStateType[stateName] == "surfaceScalarState")
        {
            continue;
        }

        // maximal connectivity level information
        // Note that stateResConInfo starts with level zero,
        // so the maxConLeve is its size minus one
        label maxConLevel = stateResConInfo[resName].size() - 1;

        // if it is a vectorState, set compMax=3
        label compMax = 1;
        if (daIndex_.adjStateType[stateName] == "volVectorState")
        {
            compMax = 3;
        }

        forAll(mesh_.cells(), cellI)
        {
            for (label comp = 0; comp < compMax; comp++)
            {

                // zero the connections
                this->createConnectionMat(&connectedStatesP);

                // now add the con. We loop over all the connectivity levels
                forAll(stateResConInfo[resName], idxJ) // idxJ: con level
                {

                    // set connectedStatesLocal: the locally connected state variables for this level
                    wordList connectedStatesLocal(0);
                    forAll(stateResConInfo[resName][idxJ], idxK)
                    {
                        word conName = stateResConInfo[resName][idxJ][idxK];
                        // Exclude surfaceScalarState when appending connectedStatesLocal
                        // whether to add it depends on addFace parameter
                        if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                        {
                            connectedStatesLocal.append(conName);
                        }
                    }

                    // set connectedStatesInterProc: the globally connected state variables for this level
                    List<List<word>> connectedStatesInterProc;
                    if (idxJ == 0)
                    {
                        // pass a zero list, no need to add interProc connecitivity for level 0
                        connectedStatesInterProc.setSize(0);
                    }
                    else if (idxJ != maxConLevel)
                    {
                        connectedStatesInterProc.setSize(maxConLevel - idxJ + 1);
                        for (label k = 0; k < maxConLevel - idxJ + 1; k++)
                        {
                            label conSize = stateResConInfo[resName][k + idxJ].size();
                            for (label l = 0; l < conSize; l++)
                            {
                                word conName = stateResConInfo[resName][k + idxJ][l];
                                // Exclude surfaceScalarState when appending connectedStatesLocal
                                // whether to add it depends on addFace parameter
                                if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                                {
                                    connectedStatesInterProc[k].append(conName);
                                }
                            }
                        }
                    }
                    else
                    {
                        connectedStatesInterProc.setSize(1);
                        label conSize = stateResConInfo[resName][maxConLevel].size();
                        for (label l = 0; l < conSize; l++)
                        {
                            word conName = stateResConInfo[resName][maxConLevel][l];
                            // Exclude surfaceScalarState when appending connectedStatesLocal
                            // whether to add it depends on addFace parameter
                            if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                            {
                                connectedStatesInterProc[0].append(conName);
                            }
                        }
                    }

                    // check if we need to addFace for this level
                    label addFace = 0;
                    forAll(stateInfo_["surfaceScalarStates"], idxK)
                    {
                        word conName = stateInfo_["surfaceScalarStates"][idxK];
                        if (DAUtility::isInList<word>(conName, stateResConInfo[resName][idxJ]))
                        {
                            addFace = 1;
                        }
                    }

                    // special treatment for pressureInletVelocity. No such treatment is needed
                    // for dFdW because we have added phi in their conInfo
                    if (DAUtility::isInList<word>("pressureInletVelocity", daField_.specialBCs)
                        && this->addPhi4PIV(stateName, cellI, comp))
                    {
                        addFace = 1;
                    }

                    // Add connectivity
                    this->addStateConnections(
                        connectedStatesP,
                        cellI,
                        idxJ,
                        connectedStatesLocal,
                        connectedStatesInterProc,
                        addFace);

                    //Info<<"lv: "<<idxJ<<" locaStates: "<<connectedStatesLocal<<" interProcStates: "
                    //    <<connectedStatesInterProc<<" addFace: "<<addFace<<endl;
                }

                // get the global index of the current state for the row index
                globalIdx = daIndex_.getGlobalAdjointStateIndex(stateName, cellI, comp);

                if (isPrealloc)
                {
                    this->allocateJacobianConnections(
                        dRdWPreallocOn_,
                        dRdWPreallocOff_,
                        dRdWTPreallocOn_,
                        dRdWTPreallocOff_,
                        connectedStatesP,
                        globalIdx);
                }
                else
                {
                    this->setupJacobianConnections(
                        jacCon_,
                        connectedStatesP,
                        globalIdx);
                }
            }
        }
    }

    // loop over all face residuals
    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        // get stateName and residual names
        word stateName = stateInfo_["surfaceScalarStates"][idxI];
        word resName = stateName + "Res";

        // maximal connectivity level information
        label maxConLevel = stateResConInfo[resName].size() - 1;

        forAll(mesh_.faces(), faceI)
        {

            //zero the connections
            this->createConnectionMat(&connectedStatesP);

            // Get the owner and neighbour cells for this face
            label idxO = -1, idxN = -1;
            if (faceI < daIndex_.nLocalInternalFaces)
            {
                idxO = mesh_.owner()[faceI];
                idxN = mesh_.neighbour()[faceI];
            }
            else
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                label patchIdx = daIndex_.bFacePatchI[relIdx];
                label faceIdx = daIndex_.bFaceFaceI[relIdx];

                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                idxN = pFaceCells[faceIdx];
            }

            // now add the con. We loop over all the connectivity levels
            forAll(stateResConInfo[resName], idxJ) // idxJ: con level
            {

                // set connectedStatesLocal: the locally connected state variables for this level
                wordList connectedStatesLocal(0);
                forAll(stateResConInfo[resName][idxJ], idxK)
                {
                    word conName = stateResConInfo[resName][idxJ][idxK];
                    // Exclude surfaceScalarState when appending connectedStatesLocal
                    // whether to add it depends on addFace parameter
                    if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                    {
                        connectedStatesLocal.append(conName);
                    }
                }

                // set connectedStatesInterProc: the globally connected state variables for this level
                List<List<word>> connectedStatesInterProc;
                if (idxJ == 0)
                {
                    // pass a zero list, no need to add interProc connecitivity for level 0
                    connectedStatesInterProc.setSize(0);
                }
                else if (idxJ != maxConLevel)
                {
                    connectedStatesInterProc.setSize(maxConLevel - idxJ + 1);
                    for (label k = 0; k < maxConLevel - idxJ + 1; k++)
                    {
                        label conSize = stateResConInfo[resName][k + idxJ].size();
                        for (label l = 0; l < conSize; l++)
                        {
                            word conName = stateResConInfo[resName][k + idxJ][l];
                            // Exclude surfaceScalarState when appending connectedStatesLocal
                            // whether to add it depends on addFace parameter
                            if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                            {
                                connectedStatesInterProc[k].append(conName);
                            }
                        }
                    }
                }
                else
                {
                    connectedStatesInterProc.setSize(1);
                    label conSize = stateResConInfo[resName][maxConLevel].size();
                    for (label l = 0; l < conSize; l++)
                    {
                        word conName = stateResConInfo[resName][maxConLevel][l];
                        // Exclude surfaceScalarState when appending connectedStatesLocal
                        // whether to add it depends on addFace parameter
                        if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                        {
                            connectedStatesInterProc[0].append(conName);
                        }
                    }
                }

                // check if we need to addFace for this level
                label addFace = 0;
                forAll(stateInfo_["surfaceScalarStates"], idxK)
                {
                    word conName = stateInfo_["surfaceScalarStates"][idxK];
                    // NOTE: we need special treatment for boundary faces for level>0
                    // since addFace for boundary face should add one more extra level of faces
                    // This is because we only have idxN for a boundary face while the idxO can
                    // be on the other side of the inter-proc boundary
                    // In this case, we need to use idxJ-1 instead of idxJ information to tell whether to addFace
                    label levelCheck;
                    if (faceI < daIndex_.nLocalInternalFaces or idxJ == 0)
                    {
                        levelCheck = idxJ;
                    }
                    else
                    {
                        levelCheck = idxJ - 1;
                    }

                    if (DAUtility::isInList<word>(conName, stateResConInfo[resName][levelCheck]))
                    {
                        addFace = 1;
                    }
                }

                // special treatment for pressureInletVelocity. No such treatment is needed
                // for dFdW because we have added phi in their conInfo
                if (DAUtility::isInList<word>("pressureInletVelocity", daField_.specialBCs)
                    && this->addPhi4PIV(stateName, faceI))
                {
                    addFace = 1;
                }

                // Add connectivity for idxN
                this->addStateConnections(
                    connectedStatesP,
                    idxN,
                    idxJ,
                    connectedStatesLocal,
                    connectedStatesInterProc,
                    addFace);

                if (faceI < daIndex_.nLocalInternalFaces)
                {
                    // Add connectivity for idxO
                    this->addStateConnections(
                        connectedStatesP,
                        idxO,
                        idxJ,
                        connectedStatesLocal,
                        connectedStatesInterProc,
                        addFace);
                }

                //Info<<"lv: "<<idxJ<<" locaStates: "<<connectedStatesLocal<<" interProcStates: "
                //    <<connectedStatesInterProc<<" addFace: "<<addFace<<endl;
            }

            // NOTE: if this faceI is on a coupled patch, the above connectivity is not enough to
            // cover the points on the other side of proc domain, we need to add 3 lvs of cells here
            if (faceI >= daIndex_.nLocalInternalFaces)
            {
                label relIdx = faceI - daIndex_.nLocalInternalFaces;
                label patchIdx = daIndex_.bFacePatchI[relIdx];

                label maxLevel = stateResConInfo[resName].size();

                if (mesh_.boundaryMesh()[patchIdx].coupled())
                {

                    label bRow = this->getLocalCoupledBFaceIndex(faceI);
                    label bRowGlobal = daIndex_.globalCoupledBFaceNumbering.toGlobal(bRow);
                    MatGetRow(stateBoundaryCon_, bRowGlobal, &nCols, &cols, &vals);
                    MatGetRow(stateBoundaryConID_, bRowGlobal, &nColsID, &colsID, &valsID);
                    for (label i = 0; i < nCols; i++)
                    {
                        PetscInt idxJ = cols[i];
                        label val = round(vals[i]);
                        // we are going to add some selective states with connectivity level <= 3
                        // first check the state
                        label stateID = round(valsID[i]);
                        word conName = daIndex_.adjStateNames[stateID];
                        label addState = 0;
                        // NOTE: we use val-1 here since phi actually has 3 levels of connectivity
                        // however, when we assign stateResConInfo, we ignore the level 0
                        // connectivity since they are idxN and idxO
                        if (val != 10 && val < maxLevel + 1)
                        {
                            if (DAUtility::isInList<word>(conName, stateResConInfo[resName][val - 1]))
                            {
                                addState = 1;
                            }
                        }
                        if (addState == 1 && val < maxLevel + 1 && val > 0)
                        {
                            this->setConnections(connectedStatesP, idxJ);
                        }
                    }
                    MatRestoreRow(stateBoundaryCon_, bRowGlobal, &nCols, &cols, &vals);
                    MatRestoreRow(stateBoundaryConID_, bRowGlobal, &nColsID, &colsID, &valsID);
                }
            }

            // get the global index of the current state for the row index
            globalIdx = daIndex_.getGlobalAdjointStateIndex(stateName, faceI);

            if (isPrealloc)
            {
                this->allocateJacobianConnections(
                    dRdWPreallocOn_,
                    dRdWPreallocOff_,
                    dRdWTPreallocOn_,
                    dRdWTPreallocOff_,
                    connectedStatesP,
                    globalIdx);
            }
            else
            {
                this->setupJacobianConnections(
                    jacCon_,
                    connectedStatesP,
                    globalIdx);
            }
        }
    }

    if (isPrealloc)
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
        wordList writeJacobians;
        daOption_.getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
        if (writeJacobians.found("dRdWTPrealloc"))
        {
            DAUtility::writeVectorASCII(dRdWTPreallocOn_, "dRdWTPreallocOn");
            DAUtility::writeVectorASCII(dRdWTPreallocOff_, "dRdWTPreallocOff");
            DAUtility::writeVectorASCII(dRdWPreallocOn_, "dRdWPreallocOn");
            DAUtility::writeVectorASCII(dRdWPreallocOff_, "dRdWPreallocOff");
        }
    }
    else
    {
        MatAssemblyBegin(jacCon_, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jacCon_, MAT_FINAL_ASSEMBLY);

        //output the matrix to a file
        wordList writeJacobians;
        daOption_.getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
        if (writeJacobians.found("dRdWCon"))
        {
            //DAUtility::writeMatRowSize(jacCon_, "dRdWCon");
            DAUtility::writeMatrixBinary(jacCon_, "dRdWCon");
        }
    }

    if (daOption_.getOption<label>("debug"))
    {
        if (isPrealloc)
        {
            Info << "Preallocating state Jacobian connectivity mat: finished!" << endl;
        }
        else
        {
            Info << "Setup state Jacobian connectivity mat: finished!" << endl;
        }
    }
}

void DAJacCondRdW::allocateJacobianConnections(
    Vec preallocOnProc,
    Vec preallocOffProc,
    Vec preallocOnProcT,
    Vec preallocOffProcT,
    Mat connections,
    const label row)
{
    /*
    Description:
        Compute the matrix allocation vector based on one row connection mat

    Input:
        connections: a one row matrix that contains all the nonzeros for one 
        row of DAJacCon::jacCon_

        row: which row to add for the preallocation vector

    Output:
        preallocOnProc: the vector that contains the number of nonzeros for each row
        in dRdW (on-diagonal block elements)
    
        preallocOffProc: the vector that contains the number of nonzeros for each row
        in dRdW (off-diagonal block elements)
    
        preallocOnProcT: the vector that contains the number of nonzeros for each row
        in dRdWT (on-diagonal block elements)
    
        preallocOffProcT: the vector that contains the number of nonzeros for each row
        in dRdWT (off-diagonal block elements)

    */
    PetscScalar v = 1.0;
    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    MatAssemblyBegin(connections, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(connections, MAT_FINAL_ASSEMBLY);

    // Compute the transposed case
    // in this case connections represents a single column, so we need to
    // increment the counter in each row with a non-zero entry.

    label colMin = daIndex_.globalAdjointStateNumbering.toGlobal(0);
    label colMax = colMin + daIndex_.nLocalAdjointStates;
    // by construction rows should be limited to local rows
    MatGetRow(connections, 0, &nCols, &cols, &vals);

    // for the non-transposed case just sum up the row.
    // count up the total number of non zeros in this row
    label totalCount = 0; //2
    label localCount = 0;
    //int idx;
    for (label j = 0; j < nCols; j++)
    {
        // int idx = cols[j];
        scalar val = vals[j];
        if (DAUtility::isValueCloseToRef(val, 1.0))
        {
            // We can compute the first part of the non-transposed row here.
            totalCount++;
            label idx = cols[j];
            // Set the transposed version as well
            if (colMin <= idx && idx < colMax)
            {
                //this entry is a local entry, increment the corresponding row
                VecSetValue(preallocOnProcT, idx, v, ADD_VALUES);
                localCount++;
            }
            else
            {
                // this is an off proc entry.
                VecSetValue(preallocOffProcT, idx, v, ADD_VALUES);
            }
        }
    }

    label offProcCount = totalCount - localCount;
    VecSetValue(preallocOnProc, row, localCount, INSERT_VALUES);
    VecSetValue(preallocOffProc, row, offProcCount, INSERT_VALUES);

    // restore the row of the matrix
    MatRestoreRow(connections, 0, &nCols, &cols, &vals);
    MatDestroy(&connections);

    return;
}

void DAJacCondRdW::preallocateJacobianMatrix(
    Mat dRMat,
    const Vec preallocOnProc,
    const Vec preallocOffProc) const
{
    /*
    Description:
        Preallocate memory for dRMat.

    Input:
        preallocOnProc, preallocOffProc: the on and off diagonal nonzeros for
        dRMat

    Output:
        dRMat: matrix to preallocate memory for
    */

    PetscScalar normOn, normOff;
    VecNorm(preallocOnProc, NORM_2, &normOn);
    VecNorm(preallocOffProc, NORM_2, &normOff);
    PetscScalar normSum = normOn + normOff;
    if (normSum < 1.0e-10)
    {
        FatalErrorIn("preallocateJacobianMatrix")
            << "preallocOnProc and preallocOffProc are not allocated!"
            << abort(FatalError);
    }

    PetscScalar *onVec, *offVec;
    PetscInt onSize[daIndex_.nLocalAdjointStates], offSize[daIndex_.nLocalAdjointStates];

    VecGetArray(preallocOnProc, &onVec);
    VecGetArray(preallocOffProc, &offVec);
    for (label i = 0; i < daIndex_.nLocalAdjointStates; i++)
    {
        onSize[i] = round(onVec[i]);
        if (onSize[i] > daIndex_.nLocalAdjointStates)
        {
            onSize[i] = daIndex_.nLocalAdjointStates;
        }
        offSize[i] = round(offVec[i]) + 5; // reserve 5 more?
    }

    VecRestoreArray(preallocOnProc, &onVec);
    VecRestoreArray(preallocOffProc, &offVec);

    // MatMPIAIJSetPreallocation(dRMat,NULL,preallocOnProc,NULL,preallocOffProc);
    // MatSeqAIJSetPreallocation(dRMat,NULL,preallocOnProc);

    MatMPIAIJSetPreallocation(dRMat, NULL, onSize, NULL, offSize);
    MatSeqAIJSetPreallocation(dRMat, NULL, onSize);

    return;
}

void DAJacCondRdW::preallocatedRdW(
    Mat dRMat,
    const label transposed) const
{
    /*
    Description:
        Call the DAJacCondRdW::preallocateJacobianMatrix function with the 
        correct vectors, depending on transposed

    Input:
        transposed: whether the state Jacobian mat is transposed, i.e., it
        is for dRdW or dRdWT (transposed)

    Output:
        dRMat: the matrix to preallocate
    */
    if (transposed)
    {
        this->preallocateJacobianMatrix(dRMat, dRdWTPreallocOn_, dRdWTPreallocOff_);
    }
    else
    {
        this->preallocateJacobianMatrix(dRMat, dRdWPreallocOn_, dRdWPreallocOff_);
    }
}

} // End namespace Foam

// ************************************************************************* //
