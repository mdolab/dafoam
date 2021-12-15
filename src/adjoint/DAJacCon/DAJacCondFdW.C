/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAJacCondFdW.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAJacCondFdW, 0);
addToRunTimeSelectionTable(DAJacCon, DAJacCondFdW, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAJacCondFdW::DAJacCondFdW(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAJacCon(modelType, mesh, daOption, daModel, daIndex)
{
    // NOTE: we need to initialize petsc vectors and setup state boundary con
    this->initializePetscVecs();
    this->initializeStateBoundaryCon();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAJacCondFdW::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAJacCon. Here we need to delete all members
        in the parent and child classes
    */
    globalObjFuncGeoNumbering_.reset(0);
    this->clearDAJacConMembers();
}

void DAJacCondFdW::initializePetscVecs()
{
    /*
    Description:
        Initialize jacConColors_
    */

    // dFdW Colors the jacConColoredColumn will be initialized in
    // DAJacCondFdW::initializeJacCon
    VecCreate(PETSC_COMM_WORLD, &jacConColors_);
    VecSetSizes(jacConColors_, daIndex_.nLocalAdjointStates, PETSC_DECIDE);
    VecSetFromOptions(jacConColors_);

    return;
}

void DAJacCondFdW::initializeJacCon(const dictionary& options)
{
    /*
    Description:
        Initialize the connectivity matrix

    Input:
        options.objFuncFaceSources: a labelList that contains all the
        face indices for the objective
    
        options.objFuncCellSources: a labelList that contains all the
        cell indices for the objective

    Output:
        jacCon_: connectivity matrix for dFdW, here dFdWCon has a
        size of nLocalObjFuncGeoElements * nGlobalAdjointStates
        The reason that dFdWCon has nLocalObjFuncGeoElements rows is 
        because we need to divide the objective function into 
        nLocalObjFuncGeoElements discrete value such that we can
        use coloring to compute dFdW
    */

    labelList objFuncFaceSources;
    labelList objFuncCellSources;
    options.readEntry<labelList>("objFuncFaceSources", objFuncFaceSources);
    options.readEntry<labelList>("objFuncCellSources", objFuncCellSources);

    objFuncFaceSize_ = objFuncFaceSources.size();
    objFuncCellSize_ = objFuncCellSources.size();

    // nLocalObjFuncGeoElements: the number of objFunc discrete elements for local procs
    label nLocalObjFuncGeoElements = objFuncFaceSize_ + objFuncCellSize_;

    globalObjFuncGeoNumbering_ = DAUtility::genGlobalIndex(nLocalObjFuncGeoElements);

    MatCreate(PETSC_COMM_WORLD, &jacCon_);
    MatSetSizes(
        jacCon_,
        nLocalObjFuncGeoElements,
        daIndex_.nLocalAdjointStates,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(jacCon_);
    MatMPIAIJSetPreallocation(jacCon_, 200, NULL, 200, NULL);
    MatSeqAIJSetPreallocation(jacCon_, 200, NULL);
    //MatSetOption(jacCon_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacCon_);
    MatZeroEntries(jacCon_);

    Info << "dFdWCon Created!" << endl;
}

void DAJacCondFdW::setupJacCon(const dictionary& options)
{
    /*
    Description:
        Setup the jacCon_ matrix
    
    Input:
        options.objFuncConInfo: the connectivity information for
        this objective function, usually obtained from Foam::DAObjFunc
    
        options.objFuncFaceSources: a labelList that contains all the
        face indices for the objective, usually obtained from Foam::DAObjFunc
    
        options.objFuncCellSources: a labelList that contains all the
        cell indices for the objective, usually obtained from Foam::DAObjFunc
    */

    Info << "Setting up dFdWCon.." << endl;

    MatZeroEntries(jacCon_);

    Mat connectedStatesP;

    List<List<word>> objFuncConInfo;
    labelList objFuncFaceSources;
    labelList objFuncCellSources;
    options.readEntry<List<List<word>>>("objFuncConInfo", objFuncConInfo);
    options.readEntry<labelList>("objFuncFaceSources", objFuncFaceSources);
    options.readEntry<labelList>("objFuncCellSources", objFuncCellSources);

    // maximal connectivity level information
    label maxConLevel = objFuncConInfo.size() - 1;

    forAll(objFuncFaceSources, idxI)
    {

        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        // get the cells associated with this boundary patch
        const UList<label>& pFaceCells = patch.faceCells();

        // Now get the cell that borders this face
        label idxN = pFaceCells[faceI];

        //zero the connections
        this->createConnectionMat(&connectedStatesP);

        forAll(objFuncConInfo, idxJ) // idxJ: con level
        {
            // set connectedStatesLocal: the locally connected state variables for this level
            wordList connectedStatesLocal(0);
            forAll(objFuncConInfo[idxJ], idxK)
            {
                word conName = objFuncConInfo[idxJ][idxK];
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
                    label conSize = objFuncConInfo[k + idxJ].size();
                    for (label l = 0; l < conSize; l++)
                    {
                        word conName = objFuncConInfo[k + idxJ][l];
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
                label conSize = objFuncConInfo[maxConLevel].size();
                for (label l = 0; l < conSize; l++)
                {
                    word conName = objFuncConInfo[maxConLevel][l];
                    // Exclude surfaceScalarState when appending connectedStatesLocal
                    // whether to add it depends on addFace parameter
                    if (daIndex_.adjStateType[conName] != "surfaceScalarState")
                    {
                        connectedStatesInterProc[0].append(conName);
                    }
                }
            }

            // check if we need to add face
            label addFace = 0;
            forAll(stateInfo_["surfaceScalarStates"], idxK)
            {
                word conName = stateInfo_["surfaceScalarStates"][idxK];
                if (DAUtility::isInList<word>(conName, objFuncConInfo[idxJ]))
                {
                    addFace = 1;
                }
            }

            this->addStateConnections(
                connectedStatesP,
                idxN,
                idxJ,
                connectedStatesLocal,
                connectedStatesInterProc,
                addFace);
        }

        label glbRowI = this->getGlobalObjFuncGeoIndex("face", idxI);

        this->setupJacobianConnections(jacCon_, connectedStatesP, glbRowI);
    }

    forAll(objFuncCellSources, idxI)
    {
        label cellI = objFuncCellSources[idxI];

        //zero the connections
        this->createConnectionMat(&connectedStatesP);

        forAll(objFuncConInfo, idxJ) // idxJ: con level
        {
            // set connectedStatesLocal: the locally connected state variables for this level
            wordList connectedStatesLocal = objFuncConInfo[idxJ];

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
                    connectedStatesInterProc[k] = objFuncConInfo[k + idxJ];
                }
            }
            else
            {
                connectedStatesInterProc.setSize(1);
                connectedStatesInterProc[0] = objFuncConInfo[maxConLevel];
            }

            // check if we need to add face
            label addFace = 0;
            forAll(stateInfo_["surfaceScalarStates"], idxK)
            {
                word conName = stateInfo_["surfaceScalarStates"][idxK];
                if (DAUtility::isInList<word>(conName, objFuncConInfo[idxJ]))
                {
                    addFace = 1;
                }
            }

            this->addStateConnections(
                connectedStatesP,
                cellI,
                idxJ,
                connectedStatesLocal,
                connectedStatesInterProc,
                addFace);
        }

        label glbRowI = this->getGlobalObjFuncGeoIndex("cell", idxI);

        this->setupJacobianConnections(jacCon_, connectedStatesP, glbRowI);
    }

    MatAssemblyBegin(jacCon_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacCon_, MAT_FINAL_ASSEMBLY);

    // output the matrix to a file
    wordList writeJacobians;
    daOption_.getAllOptions().readEntry<wordList>("writeJacobians", writeJacobians);
    if (writeJacobians.found("dFdWCon"))
    {
        DAUtility::writeMatrixBinary(jacCon_, "dFdWCon");
    }
}

label DAJacCondFdW::getLocalObjFuncGeoIndex(
    const word idxType,
    const label idxI) const
{
    /*
    Description:
        Get the local index of a geometry element of an objective
    
    Input:
        idxType: face means it is a faceSource and cell means it is a cellSource
    
        idxI: the index of the face or cell source
    
    Output:
        The local index

    Example:
        To enable coloring, we divide the objective into discrete face/cells. 
        So the number of local rows in DAJacCon::jacCon_ is nLocalObjFuncGeoElements

        If the faceSource and cellSource on proc0 read
        objFuncFaceSources_ = {11, 12};  <---- these are the indices of the faces for the objective
        objFuncCellSources_ = {23, 24, 25};

        and the faceSource and cellSource on proc1 read
        objFuncFaceSources_ = {21, 22};
        objFuncCellSources_ = {33, 34, 35};

        Then, globalObjFuncGeoNumbering_ reads:
        
    globalObjFuncGeoNumbering:  {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,    }
    Local face/cell Indices:      11   12   23   24   25  | 21   22   33   34   35
                                -------    proc 0    -----|-------    proc 1    ----- 
    */

    label localIdx = -9999;
    if (idxType == "face")
    {
        localIdx = idxI;
    }
    else if (idxType == "cell")
    {
        localIdx = objFuncFaceSize_ + idxI;
    }
    else
    {
        FatalErrorIn("") << "idxType: " << idxType << "not supported!"
                         << abort(FatalError);
    }
    return localIdx;
}

label DAJacCondFdW::getGlobalObjFuncGeoIndex(
    const word idxType,
    const label idxI) const
{
    /*
    Description:
        Get the global index of a geometry element of an objective
    
    Input:
        idxType: face means it is a faceSource and cell means it is a cellSource
    
        idxI: the index of the face or cell source
    
    Output:
        The global index

    Example:
        To enable coloring, we divide the objective into discrete face/cells. 
        So the number of local rows in DAJacCon::jacCon_ is nLocalObjFuncGeoElements

        If the faceSource and cellSource on proc0 read
        objFuncFaceSources_ = {11, 12};  <---- these are the indices of the faces for the objective
        objFuncCellSources_ = {23, 24, 25};

        and the faceSource and cellSource on proc1 read
        objFuncFaceSources_ = {21, 22};
        objFuncCellSources_ = {33, 34, 35};

        Then, globalObjFuncGeoNumbering_ reads:
        
    globalObjFuncGeoNumbering:  {  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,    }
    Local face/cell Indices:      11   12   23   24   25  | 21   22   33   34   35
                                -------    proc 0    -----|-------    proc 1    ----- 
    */
    label localIdx = this->getLocalObjFuncGeoIndex(idxType, idxI);

    return globalObjFuncGeoNumbering_.toGlobal(localIdx);
}

void DAJacCondFdW::setObjFuncVec(
    scalarList objFuncFaceValues,
    scalarList objFuncCellValues,
    Vec objFuncVec) const
{
    /*
    Description:
        Assign the values from objFuncFaceValues and objFuncCellValues
        to the objFuncVec
    
    Input:
        objFuncFaceValues, objFuncCellValues: these two lists are computed
        by calling DAObjFunc::calcObjFunc or DAObjFunc::getObjFuncValue
    
    Output:
        objFuncVec: a vector that contains all the discret values from all processors
        It will be primarily used in Foam::DAPartDerivdFdW
    */
    PetscScalar* objFuncVecArray;
    VecGetArray(objFuncVec, &objFuncVecArray);

    forAll(objFuncFaceValues, idxI)
    {
        label localIdx = getLocalObjFuncGeoIndex("face", idxI);
        assignValueCheckAD(objFuncVecArray[localIdx], objFuncFaceValues[idxI]);
    }

    forAll(objFuncCellValues, idxI)
    {
        label localIdx = getLocalObjFuncGeoIndex("cell", idxI);
        assignValueCheckAD(objFuncVecArray[localIdx], objFuncCellValues[idxI]);
    }

    VecRestoreArray(objFuncVec, &objFuncVecArray);
}

} // End namespace Foam

// ************************************************************************* //
