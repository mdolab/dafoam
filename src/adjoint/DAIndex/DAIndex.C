/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAIndex.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAIndex::DAIndex(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel)
    : mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      pointProcAddressing(
          IOobject(
              "pointProcAddressing",
              mesh_.facesInstance(),
              mesh_.meshSubDir,
              mesh_,
              IOobject::READ_IF_PRESENT,
              IOobject::NO_WRITE),
          {})
{
    // Calculate the sizes
    // local denotes the current MPI process
    // global denotes all the MPI processes

    // initialize stateInfo_
    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();

    // Setup the adjoint state name and type lists.
    forAll(stateInfo_["volVectorStates"], idxI)
    {
        adjStateNames.append(stateInfo_["volVectorStates"][idxI]);
        adjStateType.set(stateInfo_["volVectorStates"][idxI], "volVectorState");
    }
    forAll(stateInfo_["volScalarStates"], idxI)
    {
        adjStateNames.append(stateInfo_["volScalarStates"][idxI]);
        adjStateType.set(stateInfo_["volScalarStates"][idxI], "volScalarState");
    }
    forAll(stateInfo_["modelStates"], idxI)
    {
        adjStateNames.append(stateInfo_["modelStates"][idxI]);
        adjStateType.set(stateInfo_["modelStates"][idxI], "modelState");
    }
    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        adjStateNames.append(stateInfo_["surfaceScalarStates"][idxI]);
        adjStateType.set(stateInfo_["surfaceScalarStates"][idxI], "surfaceScalarState");
    }

    // Local mesh related sizes
    nLocalCells = mesh.nCells();
    nLocalFaces = mesh.nFaces();
    nLocalPoints = mesh.nPoints();
    nLocalXv = nLocalPoints * 3;
    nLocalInternalFaces = mesh.nInternalFaces();
    nLocalBoundaryFaces = nLocalFaces - nLocalInternalFaces;
    nLocalBoundaryPatches = mesh.boundaryMesh().size();

    // get bFacePatchI and bFaceFaceI
    // these two lists store the patchI and faceI for a given boundary mesh face index
    // the index of these lists starts from the first boundary face of the first boundary patch.
    // they will be used to quickly get the patchI and faceI from a given boundary face index
    bFacePatchI.setSize(nLocalBoundaryFaces);
    bFaceFaceI.setSize(nLocalBoundaryFaces);
    label tmpCounter = 0;
    forAll(mesh_.boundaryMesh(), patchI)
    {
        forAll(mesh_.boundaryMesh()[patchI], faceI)
        {
            bFacePatchI[tmpCounter] = patchI;
            bFaceFaceI[tmpCounter] = faceI;
            tmpCounter++;
        }
    }

    // Initialize state local index offset, it will be used in getLocalStateIndex function
    this->calcStateLocalIndexOffset(stateLocalIndexOffset);

    // Initialize adjStateID. It stores the stateID for a given stateName
    this->calcAdjStateID(adjStateID);

    // Local adjoint state sizes
    // first get how many state variables are registered.
    // Note turbStates are treated separatedly
    nVolScalarStates = stateInfo_["volScalarStates"].size();
    nVolVectorStates = stateInfo_["volVectorStates"].size();
    nSurfaceScalarStates = stateInfo_["surfaceScalarStates"].size();
    nModelStates = stateInfo_["modelStates"].size();

    // number of boundary states, NOTE: this does not contain boundary phis becaues
    // phi state already contains boundary phis
    nLocalAdjointBoundaryStates = (nVolVectorStates * 3 + nVolScalarStates + nModelStates) * nLocalBoundaryFaces;

    // we can now calculate adjoint state size
    label nLocalCellStates = (nVolVectorStates * 3 + nVolScalarStates + nModelStates) * nLocalCells;
    label nLocalFaceStates = nSurfaceScalarStates * nLocalFaces;
    nLocalAdjointStates = nLocalCellStates + nLocalFaceStates;

    // Setup the global numbering to convert a local index to the associated global index
    globalAdjointStateNumbering = DAUtility::genGlobalIndex(nLocalAdjointStates);
    globalCellNumbering = DAUtility::genGlobalIndex(nLocalCells);
    globalCellVectorNumbering = DAUtility::genGlobalIndex(nLocalCells * 3);
    globalFaceNumbering = DAUtility::genGlobalIndex(nLocalFaces);
    globalXvNumbering = DAUtility::genGlobalIndex(nLocalXv);

    // global Adjoint state sizes
    nGlobalAdjointStates = globalAdjointStateNumbering.size();
    nGlobalCells = globalCellNumbering.size();
    nGlobalFaces = globalFaceNumbering.size();
    nGlobalXv = globalXvNumbering.size();

    // now compute nUndecomposedPoints based on pointProcAddressing
    // this will be used in generating the total sensitivity for the undecomposed domain
    // for sensitivity map plot
    if (!Pstream::parRun())
    {
        // for serial cases, pointProcAddressing is an empty list, so manually assign it
        for (label i = 0; i < nLocalPoints; i++)
        {
            pointProcAddressing.append(i);
        }

        nUndecomposedPoints = nLocalPoints;
    }
    else
    {
        // for parallel cases, we can read pointProcAddressing
        // get the global point size
        label pointMaxIdx = max(pointProcAddressing);
        reduce(pointMaxIdx, maxOp<label>());
        // +1 since procAddressing point index starts with 0
        nUndecomposedPoints = pointMaxIdx + 1;
    }

    // initialize stuff
    // calculate nLocalCoupledBFaces and isCoupledFace
    isCoupledFace.setSize(nLocalFaces);
    for (label i = 0; i < nLocalFaces; i++)
    {
        isCoupledFace[i] = 0;
    }

    nLocalCoupledBFaces = 0;
    label faceIdx = nLocalInternalFaces;
    forAll(mesh_.boundaryMesh(), patchI)
    {
        forAll(mesh_.boundaryMesh()[patchI], faceI)
        {
            if (mesh_.boundaryMesh()[patchI].coupled())
            {
                // this is a coupled patch
                isCoupledFace[faceIdx] = 1;
                nLocalCoupledBFaces++;
            }
            faceIdx++;
        }
    }
    // now initialize the gloalindex for bFace
    globalCoupledBFaceNumbering = DAUtility::genGlobalIndex(nLocalCoupledBFaces);
    nGlobalCoupledBFaces = globalCoupledBFaceNumbering.size();

    // calculate some local lists for indexing
    this->calcLocalIdxLists(adjStateName4LocalAdjIdx, cellIFaceI4LocalAdjIdx);

    if (daOption_.getOption<label>("debug"))
    {
        this->writeAdjointIndexing();
    }
}

void DAIndex::calcStateLocalIndexOffset(HashTable<label>& offset)
{
    /*
    Description:
        Calculate the indexing offset for all states (stateLocalIndexOffset),
        this will be used in the DAIndex::getLocalAdjointStateIndex function
    
        For state-by-state ordering, we set u_0, v_0, w_0, u_1, v_1, w_1,
        ...., p_0, p_1, ... nuTilda_0, nuTilda_1, ... with subscript being the
        cell index. stateLocalIndexingOffset will return how many states are
        before a specific stateName
    
        For cell by cell ordering, we set u_0, v_0, w_0, p_0, nuTilda_0, phi_0, .... 
        u_N, v_N, w_N, p_N, nuTilda_N, phi_N with subscript being the cell index. 
        so stateLocalIndexingOffset will return how many states are before a specific 
        stateName for a given cell index

    Output:
        offset: hash table of  local state variable index offset, This will be used in 
        determing the local indexing for adjoint states. It differs depending on whether 
        we use state-by-state or cell-by-cell ordering
    */

    word adjStateOrdering = daOption_.getOption<word>("adjStateOrdering");

    if (adjStateOrdering == "state")
    {

        forAll(adjStateNames, idxI)
        {
            word stateName = adjStateNames[idxI];

            label counter = 0;

            forAll(stateInfo_["volVectorStates"], idx)
            {
                if (stateInfo_["volVectorStates"][idx] == stateName)
                {
                    offset.set(stateName, counter * nLocalCells);
                }
                counter += 3;
            }

            forAll(stateInfo_["volScalarStates"], idx)
            {
                if (stateInfo_["volScalarStates"][idx] == stateName)
                {
                    offset.set(stateName, counter * nLocalCells);
                }
                counter++;
            }

            forAll(stateInfo_["modelStates"], idx)
            {
                if (stateInfo_["modelStates"][idx] == stateName)
                {
                    offset.set(stateName, counter * nLocalCells);
                }
                counter++;
            }

            forAll(stateInfo_["surfaceScalarStates"], idx)
            {
                if (stateInfo_["surfaceScalarStates"][idx] == stateName && idx == 0)
                {
                    offset.set(stateName, counter * nLocalCells);
                }
                if (stateInfo_["surfaceScalarStates"][idx] == stateName && idx > 0)
                {
                    offset.set(stateName, counter * nLocalFaces);
                }
                counter++;
            }
        }
    }
    else if (adjStateOrdering == "cell")
    {

        forAll(adjStateNames, idxI)
        {
            word stateName = adjStateNames[idxI];

            label counter = 0;

            forAll(stateInfo_["volVectorStates"], idx)
            {
                if (stateInfo_["volVectorStates"][idx] == stateName)
                {
                    offset.set(stateName, counter);
                }
                counter += 3;
            }

            forAll(stateInfo_["volScalarStates"], idx)
            {
                if (stateInfo_["volScalarStates"][idx] == stateName)
                {
                    offset.set(stateName, counter);
                }
                counter++;
            }

            forAll(stateInfo_["modelStates"], idx)
            {
                if (stateInfo_["modelStates"][idx] == stateName)
                {
                    offset.set(stateName, counter);
                }
                counter++;
            }

            forAll(stateInfo_["surfaceScalarStates"], idx)
            {
                if (stateInfo_["surfaceScalarStates"][idx] == stateName)
                {
                    offset.set(stateName, counter);
                }
                counter++;
            }
        }

        // We also need a few more offsets

        // calculate faceOwner
        faceOwner.setSize(nLocalFaces);
        const UList<label>& internalFaceOwner = mesh_.owner(); // these only include internal faces owned cellI
        forAll(faceOwner, idxI)
        {
            if (idxI < nLocalInternalFaces)
            {
                faceOwner[idxI] = internalFaceOwner[idxI];
            }
            else
            {
                label relIdx = idxI - nLocalInternalFaces;
                label patchIdx = bFacePatchI[relIdx];
                label faceIdx = bFaceFaceI[relIdx];
                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                faceOwner[idxI] = pFaceCells[faceIdx];
            }
        }

        // Calculate the cell owned face index. Note: we can't use mesh.cells here since it will have
        // duplicated face indices
        List<List<label>> cellOwnedFaces;
        cellOwnedFaces.setSize(nLocalCells);
        forAll(faceOwner, idxI)
        {
            label ownedCellI = faceOwner[idxI];
            cellOwnedFaces[ownedCellI].append(idxI);
        }
        //Info<<"cellOwnedFaces "<<cellOwnedFaces<<endl;
        // check if every cells have owned faces
        // This is unnecessary since some cells may own no face.
        //forAll(cellOwnedFaces,idxI)
        //{
        //    if(cellOwnedFaces[idxI].size()==0) FatalErrorIn("")<<"cell "<<idxI<<" owns no faces"<<abort(FatalError);
        //}

        // We first calculate phiAccumulatedOffset
        phiAccumulatdOffset.setSize(nLocalCells);
        forAll(phiAccumulatdOffset, idxI)
        {
            phiAccumulatdOffset[idxI] = 0;
        }
        // if we have no surfaceScalarStates, phiAccumulatdOffset remains zeros
        if (stateInfo_["surfaceScalarStates"].size() > 0)
        {
            forAll(phiAccumulatdOffset, idxI)
            {
                if (idxI == 0)
                {
                    phiAccumulatdOffset[idxI] = 0;
                }
                else
                {
                    phiAccumulatdOffset[idxI] = cellOwnedFaces[idxI - 1].size() + phiAccumulatdOffset[idxI - 1];
                }
            }
        }
        //Info<<"phiAccumulatdOffset "<<phiAccumulatdOffset<<endl;

        // Now calculate the phiLocalOffset
        phiLocalOffset.setSize(nLocalFaces);
        forAll(phiLocalOffset, idxI)
        {
            phiLocalOffset[idxI] = 0;
        }
        // if we have no surfaceScalarStates, phiAccumulatdOffset remains zeros
        if (stateInfo_["surfaceScalarStates"].size() > 0)
        {
            forAll(cellOwnedFaces, idxI) // idxI is cell Index
            {
                forAll(cellOwnedFaces[idxI], offsetI)
                {
                    label ownedFace = cellOwnedFaces[idxI][offsetI];
                    phiLocalOffset[ownedFace] = offsetI;
                }
            }
        }
        //Info<<"phiLocalOffset "<<phiLocalOffset<<endl;
        //Info<<"stateLocalIndexOffset "<<stateLocalIndexOffset<<endl;
    }
    else
    {
        FatalErrorIn("") << "adjStateOrdering invalid" << abort(FatalError);
    }

    return;
}

void DAIndex::calcAdjStateID(HashTable<label>& adjStateID)
{
    /* 
    Description:
       The stateID is an alternative for the stateNames
       stateID starts from 0 for the first volVector state
       e.g., if the state variables are U, p, nut, phi, their 
       state ID are U=0, p=1, nut=2, phi=3

    Output:
        adjStateID: the state ID list
    */

    label id = 0;
    forAll(stateInfo_["volVectorStates"], idx)
    {
        word stateName = stateInfo_["volVectorStates"][idx];
        adjStateID.set(stateName, id);
        id++;
    }

    forAll(stateInfo_["volScalarStates"], idx)
    {
        word stateName = stateInfo_["volScalarStates"][idx];
        adjStateID.set(stateName, id);
        id++;
    }

    forAll(stateInfo_["modelStates"], idx)
    {
        word stateName = stateInfo_["modelStates"][idx];
        adjStateID.set(stateName, id);
        id++;
    }

    forAll(stateInfo_["surfaceScalarStates"], idx)
    {
        word stateName = stateInfo_["surfaceScalarStates"][idx];
        adjStateID.set(stateName, id);
        id++;
    }
    return;
}

void DAIndex::calcLocalIdxLists(
    wordList& stateName4LocalAdjIdx,
    scalarList& cellIFaceI4LocalIdx)
{
    /*
    Description:
        Calculate indexing lists:
        cellIFaceI4LocalAdjIdx
        adjStateName4LocalAdjIdx

    Output:
        cellIFaceI4LocalAdjIdx: stores the cell/face index for a local adjoint index
        For vector fields, the decima of cellIFaceI4LocalIdx denotes the vector component
        e.g., 10.1 means cellI=10, y compoent of U
    
        adjStateName4LocalAdjIdx: stores the state name for a local adjoint index
    */

    cellIFaceI4LocalIdx.setSize(nLocalAdjointStates);
    stateName4LocalAdjIdx.setSize(nLocalAdjointStates);

    forAll(stateInfo_["volVectorStates"], idx)
    {
        word stateName = stateInfo_["volVectorStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label localIdx = this->getLocalAdjointStateIndex(stateName, cellI, i);
                cellIFaceI4LocalIdx[localIdx] = cellI + i / 10.0;

                stateName4LocalAdjIdx[localIdx] = stateName;
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idx)
    {
        word stateName = stateInfo_["volScalarStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            label localIdx = this->getLocalAdjointStateIndex(stateName, cellI);
            cellIFaceI4LocalIdx[localIdx] = cellI;

            stateName4LocalAdjIdx[localIdx] = stateName;
        }
    }

    forAll(stateInfo_["modelStates"], idx)
    {
        word stateName = stateInfo_["modelStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            label localIdx = this->getLocalAdjointStateIndex(stateName, cellI);
            cellIFaceI4LocalIdx[localIdx] = cellI;

            stateName4LocalAdjIdx[localIdx] = stateName;
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idx)
    {
        word stateName = stateInfo_["surfaceScalarStates"][idx];
        forAll(mesh_.faces(), faceI)
        {
            label localIdx = this->getLocalAdjointStateIndex(stateName, faceI);
            cellIFaceI4LocalIdx[localIdx] = faceI;

            stateName4LocalAdjIdx[localIdx] = stateName;
        }
    }

    return;
}

label DAIndex::getLocalAdjointStateIndex(
    const word stateName,
    const label idxJ,
    const label comp) const
{
    /*
    Description:
        Return the global adjoint index given a state name, a local index, 
        and vector component (optional)

    Input:
        stateName: name of the state variable for the global indexing
    
        idxJ: the local index for the state variable, typically it is the state's 
        local cell index or face index
    
        comp: if the state is a vector, give its componet for global indexing. 
        NOTE: for volVectorState, one need to set comp; while for other states, 
        comp is simply ignored in this function

    Example:
        Image we have two state variables (p, T) and we have three cells, the state
        variable vector reads (state-by-state ordering):
    
        w= [p0, p1, p2, T0, T1, T2]  <- p0 means p for the 0th cell
             0   1   2   3   4   5   <- adjoint local index
    
        Then getLocalAdjointStateIndex("p",1) returns 1 and 
        getLocalAdjointStateIndex("T",1) returns 4
    
        If we use cell-by-cell ordering, the state variable vector reads 
        w= [p0, T0, p1, T1, p2, T2]
             0   1   2   3   4   5   <- adjoint local index
        
        Then getLocalAdjointStateIndex("p",1) returns 2 and 
        getLocalAdjointStateIndex("T",1) returns 3
    
        Similarly, we can apply this functions for vector state variables, again
        we assume we have two state variables (U, p) and three cells, then the 
        state-by-state adjoint ordering gives
    
        w= [u0, v0, w0, u1, v1, w1, u2, v2, w2, p0, p1, p2]
             0   1   2   3   4   5   6   7   8   9  10  11 <- adjoint local index
    
        Then getLocalAdjointStateIndex("U", 1, 2) returns 5 and 
        getLocalAdjointStateIndex("p",1) returns 10
    
        NOTE: the three compoent for U are [u,v,w]

    */

    word adjStateOrdering = daOption_.getOption<word>("adjStateOrdering");

    if (adjStateOrdering == "state")
    {
        /*
        state by state indexing
        we set u_0, v_1, w_2, u_3, v_4, w_5, ...., p_np, p_np+1, ... nuTilda_nnu, 
        nuTilda_nnu+1, ... so getStateLocalIndexingOffset(p) will return np
        For vector, one need to provide comp, for scalar, comp is not needed.
        */
        forAll(adjStateNames, idxI)
        {
            if (adjStateNames[idxI] == stateName)
            {
                if (adjStateType[stateName] == "volVectorState")
                {
                    if (comp == -1)
                    {
                        FatalErrorIn("") << "comp needs to be set for vector states!"
                                         << abort(FatalError);
                    }
                    else
                    {
                        return stateLocalIndexOffset[stateName] + idxJ * 3 + comp;
                    }
                }
                else
                {
                    return stateLocalIndexOffset[stateName] + idxJ;
                }
            }
        }
    }
    else if (adjStateOrdering == "cell")
    {
        // cell by cell ordering
        // We set u_0, v_0, w_0, p_0, nuTilda_0, phi_0a,phi_0b,phi_0c.... u_N, v_N, w_N, p_N, nuTilda_N, phi_N
        // To get the local index, we need to do:
        // idxLocal =
        //   cellI*(nVectorStates*3+nScalarStates+nTurbStates)
        // + phiAccumulatedOffset (how many phis have been accumulated. Note: we have mulitple phis for a cellI)
        // + stateLocalIndexOffset+comp
        // + phiLocalOffset (only for phi idx, ie. 0a or 0b or oc. Note: we have mulitple phis for a cellI)

        label nCellStates = 3 * nVolVectorStates
            + nVolScalarStates
            + nModelStates;

        const word& stateType = adjStateType[stateName];
        label returnV = -99999999;
        if (stateType == "surfaceScalarState") // for surfaceScalarState idxJ is faceI
        {
            label idxN = faceOwner[idxJ]; // idxN is the cell who owns faceI
            returnV = idxN * nCellStates
                + stateLocalIndexOffset[stateName]
                + phiAccumulatdOffset[idxN]
                + phiLocalOffset[idxJ];
            return returnV;
        }
        else if (stateType == "volVectorState") // for other states idxJ is cellI
        {
            if (comp == -1)
            {
                FatalErrorIn("") << "comp needs to be set for vector states!"
                                 << abort(FatalError);
            }
            else
            {

                returnV = idxJ * nCellStates
                    + stateLocalIndexOffset[stateName]
                    + comp
                    + phiAccumulatdOffset[idxJ];
            }
            return returnV;
        }
        else
        {
            returnV = idxJ * nCellStates
                + stateLocalIndexOffset[stateName]
                + phiAccumulatdOffset[idxJ];
            return returnV;
        }
    }
    else
    {
        FatalErrorIn("") << "adjStateOrdering invalid" << abort(FatalError);
    }

    // if no stateName found, return an error
    FatalErrorIn("") << "stateName not found!" << abort(FatalError);
    return -1;
}

label DAIndex::getGlobalAdjointStateIndex(
    const word stateName,
    const label idxI,
    const label comp) const
{
    /*
    Description:
        This function has the same input as DAIndex::getLocalAdjointStateIndex
        the only difference is that this function returns the global adjoint 
        state index by calling globalAdjointStateNumbering.toGlobal()

    Input:
        stateName: name of the state variable for the global indexing
    
        idxJ: the local index for the state variable, typically it is the state's 
        local cell index or face index
    
        comp: if the state is a vector, give its componet for global indexing. 
        NOTE: for volVectorState, one need to set comp; while for other states, 
        comp is simply ignored in this function


    Example:
        Image we have two state variables (p,T) and five cells, running on two CPU
        processors, the proc0 owns two cells and the proc1 owns three cells,
        then the global adjoint state variables reads (state-by-state)
    
        w = [p0, p1, T0, T1 | p0, p1, p2, T0, T1, T2] <- p0 means p for the 0th cell on local processor
              0   1   2   3 |  4   5   6   7   8   9  <- global adjoint index
            ---- proc0 -----|--------- proc1 ------- 
        
        Then, on proc0, getGlobalAdjointStateIndex("T", 1) returns 3 
          and on proc1, getGlobalAdjointStateIndex("T", 1) returns 8

    */
    // For vector, one need to provide comp, for scalar, comp is not needed.
    label localIdx = this->getLocalAdjointStateIndex(stateName, idxI, comp);
    label globalIdx = globalAdjointStateNumbering.toGlobal(localIdx);
    return globalIdx;
}

label DAIndex::getGlobalXvIndex(
    const label idxPoint,
    const label idxCoord) const
{
    /*
    Description:
        This function has the same input as DAIndex::getLocalXvIndex except that 
        this function returns the global xv index 

    Input:
        idxPoint: local point index
    
        idxCoord: the compoent of the point

    Example:
        Image we have three points, running on two CPU cores, and the proc0 owns
        one point and proc1 owns two points, and the Xv vector reads
    
        Xv = [x0, y0, z0 | x0, y0, z0, x1, y1, z1] <- x0 means the x for the 0th point
               0   1   2    3   4   5   6   7   8  <- global Xv index
              -- proc0 --|--------- proc1 ------- 
        Then, on proc0, getGlobalXvIndex(0,1) returns 1
          and on proc1, getGlobalXvIndex(0,1) returns 4

    */

    label localXvIdx = this->getLocalXvIndex(idxPoint, idxCoord);
    label globalXvIdx = globalXvNumbering.toGlobal(localXvIdx);
    return globalXvIdx;
}

label DAIndex::getLocalXvIndex(
    const label idxPoint,
    const label idxCoord) const
{
    /*
    Description:
        Returns the local xv index for a given local cell index and its component

    Input:
        idxPoint: local point index
    
        idxCoord: the compoent of the point

    Example:
        Image we have two points, and the Xv vector reads
    
        Xv = [x0, y0, z0, x1, y1, z1] <- x0 means the x for the 0th point
               0   1   2   3   4   5  <- local Xv index
    
        Then, getLocalXvIndex(1,1) returns 4

    */

    label localXvIdx = idxPoint * 3 + idxCoord;
    return localXvIdx;
}

label DAIndex::getLocalCellIndex(const label cellI) const
{
    /*
    Description:
        Returns the local cell index for a given local cell index

    Input:
        cellI: local mesh cell index

    Example:
        Image we have three cells, the local vector reads
    
        cVec = [c0, c1, c2] <- c0 means the 0th cell
                 0   1   2  <- local cell index
    
        Then, getLocalCellIndex(1) returns 1
        NOTE: we essentially just return cellI

    */
    return cellI;
}

label DAIndex::getGlobalCellIndex(const label cellI) const
{
    /*
    Description:
        Returns the global cell index for a given local cell index

    Input:
        cellI: local cell index

    Example:
        Image we have nine cells, running on two CPU cores, and the proc0 owns
        three cell and proc1 owns six cell, and the cell vector reads
    
        cVec = [c0, c1, c2 | c0, c1, c2, c3, c4, c5] <- c0 means the 0th cell
                 0   1   2    3   4   5   6   7   8  <- global cell index
                -- proc0 --|--------- proc1 ------- 
        Then, on proc0, getGlobalCellIndex(1) returns 1
          and on proc1, getGlobalCellIndex(1) returns 4

    */

    label localIdx = this->getLocalCellIndex(cellI);
    label globalIdx = globalCellNumbering.toGlobal(localIdx);
    return globalIdx;
}

label DAIndex::getLocalCellVectorIndex(
    const label cellI,
    const label comp) const
{
    /*
    Description:
        Returns the local cell index for a given local cell vector index

    Input:
        cellI: local mesh cell index

        comp: the vector component 

    Example:
        Image we have two cells, the local vector reads
    
        cVec = [c0a, c0b, c0c, c1a, c1b, c1c] <- c0b means the 0th cell, 1st vector component
                 0    1    2    3    4    5   <- local cell vector index
    
        Then, getLocalCellIndex(1,1) returns 4

    */
    label localIdx = cellI * 3 + comp;
    return localIdx;
}

label DAIndex::getGlobalCellVectorIndex(
    const label cellI,
    const label comp) const
{
    /*
    Description:
        Returns the global cell index for a given local cell vector index

    Input:
        cellI: local cell index

        comp: the vector component 

    Example:
        Image we have three cells, running on two CPU cores, and the proc0 owns
        two cells and proc1 owns one cell, and the cell vector reads
    
        cVec = [c0a, c0b, c0c, c1a, c1b, c1c | c0a, c0b, c0c] <- c0b means the 0th cell, 1st vector component
                 0   1     2    3    4    5     6    7    8  <- global cell index
                --------- proc0 -------------|---- proc1 ---
        Then, on proc0, getGlobalCellVectorIndex(0, 1) returns 1
          and on proc1, getGlobalCellVectorIndex(0, 1) returns 7

    */

    label localIdx = this->getLocalCellVectorIndex(cellI, comp);
    label globalIdx = globalCellVectorNumbering.toGlobal(localIdx);
    return globalIdx;
}

label DAIndex::getLocalFaceIndex(const label faceI) const
{
    /*
    Description:
        Returns the local face index for a given local face index

    Input:
        faceI: local mesh face index (including the boundary faces)

    Example:
        Image we have three faces, the local vector reads
    
        fVec = [f0, f1, f2] <- f0 means the 0th face
                 0   1   2  <- local face index
    
        Then, getLocalFaceIndex(1) returns 1
        NOTE: we essentially just return faceI

    */
    return faceI;
}

label DAIndex::getGlobalFaceIndex(const label faceI) const
{
    /*
    Description:
        Returns the global face index for a given local face index

    Input:
        faceI: local mesh face index (including the boundary faces)

    Example:
        Image we have nine faces, running on two CPU cores, and the proc0 owns
        three faces and proc1 owns six faces, and the face vector reads
    
        fVec = [f0, f1, f2 | f0, f1, f2, f3, f4, f5] <- f0 means the 0th face
                 0   1   2    3   4   5   6   7   8  <- global face index
                -- proc0 --|--------- proc1 ------- 
        Then, on proc0, getGlobalFaceIndex(1) returns 1
          and on proc1, getGlobalFaceIndex(1) returns 4

    */

    label localIdx = this->getLocalFaceIndex(faceI);
    label globalIdx = globalFaceNumbering.toGlobal(localIdx);
    return globalIdx;
}

void DAIndex::calcAdjStateID4GlobalAdjIdx(labelList& adjStateID4GlobalAdjIdx) const
{
    /*
    Description:
        Compute adjStateID4GlobalAdjIdx

    Output:
        adjStateID4GlobalAdjIdx: labelList that stores the adjStateID for given a global adj index
        NOTE: adjStateID4GlobalAdjIdx contains all the global adj indices, so its memory usage 
        is high. We should avoid having any sequential list; however, to make the connectivity
        calculation easier, we keep it for now. 
        *******delete this list after used!************
    */

    if (adjStateID4GlobalAdjIdx.size() != nGlobalAdjointStates)
    {
        FatalErrorIn("") << "adjStateID4GlobalAdjIdx.size()!=nGlobalAdjointStates"
                         << abort(FatalError);
    }

    Vec stateIVec;
    VecCreate(PETSC_COMM_WORLD, &stateIVec);
    VecSetSizes(stateIVec, nLocalAdjointStates, PETSC_DECIDE);
    VecSetFromOptions(stateIVec);
    VecSet(stateIVec, 0); // default value

    forAll(stateInfo_["volVectorStates"], idx)
    {
        word stateName = stateInfo_["volVectorStates"][idx];
        PetscScalar valIn = adjStateID[stateName] + 1; // we need to use 1-based indexing here for scattering
        forAll(mesh_.cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label globalIdx = this->getGlobalAdjointStateIndex(stateName, cellI, i);
                VecSetValues(stateIVec, 1, &globalIdx, &valIn, INSERT_VALUES);
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idx)
    {
        word stateName = stateInfo_["volScalarStates"][idx];
        PetscScalar valIn = adjStateID[stateName] + 1; // we need to use 1-based indexing here for scattering
        forAll(mesh_.cells(), cellI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName, cellI);
            VecSetValues(stateIVec, 1, &globalIdx, &valIn, INSERT_VALUES);
        }
    }

    forAll(stateInfo_["modelStates"], idx)
    {
        word stateName = stateInfo_["modelStates"][idx];
        PetscScalar valIn = adjStateID[stateName] + 1; // we need to use 1-based indexing here for scattering
        forAll(mesh_.cells(), cellI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName, cellI);
            VecSetValues(stateIVec, 1, &globalIdx, &valIn, INSERT_VALUES);
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idx)
    {
        word stateName = stateInfo_["surfaceScalarStates"][idx];
        PetscScalar valIn = adjStateID[stateName] + 1; // we need to use 1-based indexing here for scattering
        forAll(mesh_.faces(), faceI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName, faceI);
            VecSetValues(stateIVec, 1, &globalIdx, &valIn, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(stateIVec);
    VecAssemblyEnd(stateIVec);

    // scatter to local array for all procs
    Vec vout;
    VecScatter ctx;
    VecScatterCreateToAll(stateIVec, &ctx, &vout);
    VecScatterBegin(ctx, stateIVec, vout, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, stateIVec, vout, INSERT_VALUES, SCATTER_FORWARD);

    PetscScalar* stateIVecArray;
    VecGetArray(vout, &stateIVecArray);

    for (label i = 0; i < nGlobalAdjointStates; i++)
    {
        adjStateID4GlobalAdjIdx[i] = static_cast<label>(stateIVecArray[i]) - 1; // subtract 1 and return to 0-based indexing
    }

    VecRestoreArray(vout, &stateIVecArray);
    VecScatterDestroy(&ctx);
    VecDestroy(&vout);

    return;
}

void DAIndex::printMatChars(const Mat matIn) const
{
    /*
    Description:
        Calculate and print some matrix statistics such as the 
        max ratio of on and off diagonal elements
    */

    PetscInt nCols, Istart, Iend;
    const PetscInt* cols;
    const PetscScalar* vals;
    scalar maxRatio = 0.0;
    label maxRatioRow = -1;
    scalar diagV = -1.0;
    scalar maxNonDiagV = -1.0;
    label maxNonDiagCol = -1;
    scalar small = 1e-12;
    scalar allNonZeros = 0.0;
    label maxCols = -1;

    this->getMatNonZeros(matIn, maxCols, allNonZeros);

    MatGetOwnershipRange(matIn, &Istart, &Iend);
    for (label i = Istart; i < Iend; i++)
    {
        scalar diag = 0;
        scalar nonDiagSum = 0;
        scalar maxV = 0.0;
        label maxVIdx = -1;
        MatGetRow(matIn, i, &nCols, &cols, &vals);
        for (label n = 0; n < nCols; n++)
        {
            if (i == cols[n])
            {
                diag = vals[n];
            }
            if (vals[n] != 0)
            {
                if (i != cols[n])
                {
                    nonDiagSum = nonDiagSum + fabs(vals[n]);
                }
                if (fabs(vals[n]) > maxV)
                {
                    maxV = fabs(vals[n]);
                    maxVIdx = cols[n];
                }
            }
        }

        if (fabs(nonDiagSum / (diag + small)) > maxRatio)
        {
            maxRatio = fabs(nonDiagSum / (diag + small));
            maxRatioRow = i;
            maxNonDiagCol = maxVIdx;
            diagV = diag;
            maxNonDiagV = maxV;
        }

        MatRestoreRow(matIn, i, &nCols, &cols, &vals);
    }

    label rowStateID = -1;
    label colStateID = -1;
    vector rowCoord(0, 0, 0), colCoord(0, 0, 0);

    forAll(stateInfo_["volVectorStates"], idx)
    {
        const word& stateName = stateInfo_["volVectorStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            for (label i = 0; i < 3; i++)
            {
                label idxJ = getGlobalAdjointStateIndex(stateName, cellI, i);
                if (idxJ == maxRatioRow)
                {
                    rowStateID = adjStateID[stateName];
                    rowCoord = mesh_.C()[cellI];
                }
                if (idxJ == maxNonDiagCol)
                {
                    colStateID = adjStateID[stateName];
                    colCoord = mesh_.C()[cellI];
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idx)
    {
        const word& stateName = stateInfo_["volScalarStates"][idx];
        forAll(mesh_.cells(), cellI)
        {

            label idxJ = getGlobalAdjointStateIndex(stateName, cellI);
            if (idxJ == maxRatioRow)
            {
                rowStateID = adjStateID[stateName];
                rowCoord = mesh_.C()[cellI];
            }
            if (idxJ == maxNonDiagCol)
            {
                colStateID = adjStateID[stateName];
                colCoord = mesh_.C()[cellI];
            }
        }
    }

    forAll(stateInfo_["modelStates"], idx)
    {
        const word& stateName = stateInfo_["modelStates"][idx];
        forAll(mesh_.cells(), cellI)
        {

            label idxJ = getGlobalAdjointStateIndex(stateName, cellI);
            if (idxJ == maxRatioRow)
            {
                rowStateID = adjStateID[stateName];
                rowCoord = mesh_.C()[cellI];
            }
            if (idxJ == maxNonDiagCol)
            {
                colStateID = adjStateID[stateName];
                colCoord = mesh_.C()[cellI];
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idx)
    {
        const word& stateName = stateInfo_["surfaceScalarStates"][idx];
        forAll(mesh_.faces(), faceI)
        {

            label idxJ = getGlobalAdjointStateIndex(stateName, faceI);

            if (idxJ == maxRatioRow)
            {
                rowStateID = adjStateID[stateName];
                if (faceI < nLocalInternalFaces)
                {
                    rowCoord = mesh_.Cf()[faceI];
                }
                else
                {
                    label relIdx = faceI - nLocalInternalFaces;
                    label patchIdx = bFacePatchI[relIdx];
                    label faceIdx = bFaceFaceI[relIdx];
                    rowCoord = mesh_.Cf().boundaryField()[patchIdx][faceIdx];
                }
            }
            if (idxJ == maxNonDiagCol)
            {
                colStateID = adjStateID[stateName];
                if (faceI < nLocalInternalFaces)
                {
                    colCoord = mesh_.Cf()[faceI];
                }
                else
                {
                    label relIdx = faceI - nLocalInternalFaces;
                    label patchIdx = bFacePatchI[relIdx];
                    label faceIdx = bFaceFaceI[relIdx];
                    colCoord = mesh_.Cf().boundaryField()[patchIdx][faceIdx];
                }
            }
        }
    }

    // create a list to store the info
    List<scalar> matCharInfo(13);
    matCharInfo[0] = maxRatio;
    matCharInfo[1] = diagV;
    matCharInfo[2] = maxNonDiagV;
    matCharInfo[3] = maxRatioRow;
    matCharInfo[4] = rowStateID;
    matCharInfo[5] = rowCoord.x();
    matCharInfo[6] = rowCoord.y();
    matCharInfo[7] = rowCoord.z();
    matCharInfo[8] = maxNonDiagCol;
    matCharInfo[9] = colStateID;
    matCharInfo[10] = colCoord.x();
    matCharInfo[11] = colCoord.y();
    matCharInfo[12] = colCoord.z();

    // now gather all the info
    label myProc = Pstream::myProcNo();
    label nProcs = Pstream::nProcs();
    // create listlist for gathering
    List<List<scalar>> gatheredList(nProcs);
    // assign values for the listlists
    gatheredList[myProc] = matCharInfo;
    // gather all info to the master proc
    Pstream::gatherList(gatheredList);
    // scatter all info to every procs
    Pstream::scatterList(gatheredList);

    scalar maxRatioGathered = -1.0;
    label procI = -1;
    for (label i = 0; i < nProcs; i++)
    {
        if (fabs(gatheredList[i][0]) > maxRatioGathered)
        {
            maxRatioGathered = fabs(gatheredList[i][0]);
            procI = i;
        }
    }

    Info << endl;
    Info << "Jacobian Matrix Characteristics: " << endl;
    Info << " Mat maxCols: " << maxCols << endl;
    Info << " Mat allNonZeros: " << allNonZeros << endl;
    Info << " Max nonDiagSum/Diag: " << gatheredList[procI][0]
         << " Diag: " << gatheredList[procI][1] << " MaxNonDiag: " << gatheredList[procI][2] << endl;
    Info << " MaxRatioRow: " << gatheredList[procI][3] << " RowState: " << gatheredList[procI][4]
         << " RowCoord: (" << gatheredList[procI][5] << " " << gatheredList[procI][6]
         << " " << gatheredList[procI][7] << ")" << endl;
    Info << " MaxNonDiagCol: " << gatheredList[procI][8] << " ColState: " << gatheredList[procI][9]
         << " ColCoord: (" << gatheredList[procI][10] << " " << gatheredList[procI][11]
         << " " << gatheredList[procI][12] << ")" << endl;
    Info << " Max nonDiagSum/Diag ProcI: " << procI << endl;
    Info << endl;

    return;
}

void DAIndex::getMatNonZeros(
    const Mat matIn,
    label& maxCols,
    scalar& allNonZeros) const
{
    /*
    Description:
        Get the max nonzeros per row, and all the nonzeros for this matrix
    */

    PetscInt nCols, Istart, Iend;
    const PetscInt* cols;
    const PetscScalar* vals;

    // set the counter
    maxCols = 0;
    allNonZeros = 0.0;

    // Determine which rows are on the current processor
    MatGetOwnershipRange(matIn, &Istart, &Iend);

    // loop over the matrix and find the largest number of cols
    for (label i = Istart; i < Iend; i++)
    {
        MatGetRow(matIn, i, &nCols, &cols, &vals);
        if (nCols < 0)
        {
            std::cout << "Warning! procI: " << Pstream::myProcNo() << " nCols <0 at rowI: " << i << std::endl;
            std::cout << "Set nCols to zero " << std::endl;
            nCols = 0;
        }
        if (nCols > maxCols) // perhaps actually check vals?
        {
            maxCols = nCols;
        }
        allNonZeros += nCols;
        MatRestoreRow(matIn, i, &nCols, &cols, &vals);
    }

    //reduce the maxcols value so that all procs have the same size
    reduce(maxCols, maxOp<label>());

    reduce(allNonZeros, sumOp<scalar>());

    return;
}

void DAIndex::writeAdjointIndexing()
{

    scalar xx, yy, zz; // face owner coordinates

    //output the matrix to a file
    label myProc = Pstream::myProcNo();
    label nProcs = Pstream::nProcs();
    std::ostringstream fileNameStream("");
    fileNameStream << "AdjointIndexing"
                   << "_" << myProc << "_of_" << nProcs << ".txt";
    word fileName = fileNameStream.str();
    OFstream aOut(fileName);
    aOut.precision(9);

    forAll(stateInfo_["volVectorStates"], idx)
    {
        const word& stateName = stateInfo_["volVectorStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            xx = mesh_.C()[cellI].x();
            yy = mesh_.C()[cellI].y();
            zz = mesh_.C()[cellI].z();
            for (label i = 0; i < 3; i++)
            {
                label glbIdx = getGlobalAdjointStateIndex(stateName, cellI, i);
                aOut << "Cell: " << cellI << " State: " << stateName << i << " glbIdx: " << glbIdx
                     << " x: " << xx << " y: " << yy << " z: " << zz << endl;
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idx)
    {
        const word& stateName = stateInfo_["volScalarStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            xx = mesh_.C()[cellI].x();
            yy = mesh_.C()[cellI].y();
            zz = mesh_.C()[cellI].z();
            label glbIdx = getGlobalAdjointStateIndex(stateName, cellI);
            aOut << "Cell: " << cellI << " State: " << stateName << " glbIdx: " << glbIdx
                 << " x: " << xx << " y: " << yy << " z: " << zz << endl;
        }
    }

    forAll(stateInfo_["modelStates"], idx)
    {
        const word& stateName = stateInfo_["modelStates"][idx];
        forAll(mesh_.cells(), cellI)
        {
            xx = mesh_.C()[cellI].x();
            yy = mesh_.C()[cellI].y();
            zz = mesh_.C()[cellI].z();
            label glbIdx = getGlobalAdjointStateIndex(stateName, cellI);
            aOut << "Cell: " << cellI << " State: " << stateName << " glbIdx: " << glbIdx
                 << " x: " << xx << " y: " << yy << " z: " << zz << endl;
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idx)
    {
        const word& stateName = stateInfo_["surfaceScalarStates"][idx];
        label cellI = -1;
        forAll(mesh_.faces(), faceI)
        {
            if (faceI < nLocalInternalFaces)
            {
                xx = mesh_.Cf()[faceI].x();
                yy = mesh_.Cf()[faceI].y();
                zz = mesh_.Cf()[faceI].z();
            }
            else
            {
                label relIdx = faceI - nLocalInternalFaces;
                label patchIdx = bFacePatchI[relIdx];
                label faceIdx = bFaceFaceI[relIdx];
                xx = mesh_.Cf().boundaryField()[patchIdx][faceIdx].x();
                yy = mesh_.Cf().boundaryField()[patchIdx][faceIdx].y();
                zz = mesh_.Cf().boundaryField()[patchIdx][faceIdx].z();

                const polyPatch& pp = mesh_.boundaryMesh()[patchIdx];
                const UList<label>& pFaceCells = pp.faceCells();
                cellI = pFaceCells[faceIdx];
            }

            label glbIdx = getGlobalAdjointStateIndex(stateName, faceI);
            aOut << "Face: " << faceI << " State: " << stateName << " glbIdx: " << glbIdx
                 << " x: " << xx << " y: " << yy << " z: " << zz
                 << " OwnerCellI: " << cellI << endl;
        }
    }

    // write point indexing
    std::ostringstream fileNameStreamPoint("");
    fileNameStreamPoint << "PointIndexing"
                        << "_" << myProc << "_of_" << nProcs << ".txt";
    word fileNamePoint = fileNameStreamPoint.str();
    OFstream aOutPoint(fileNamePoint);
    aOutPoint.precision(9);

    forAll(mesh_.points(), idxI)
    {
        xx = mesh_.points()[idxI].x();
        yy = mesh_.points()[idxI].y();
        zz = mesh_.points()[idxI].z();
        for (label i = 0; i < 3; i++)
        {
            label glbIdx = getGlobalXvIndex(idxI, i);
            aOutPoint << "Point: " << idxI << " Coords: " << i << " glbIdx: " << glbIdx
                      << " x: " << xx << " y: " << yy << " z: " << zz << endl;
        }
    }

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
