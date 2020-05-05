/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/


#include "AdjointIndexing.H"

namespace Foam
{

// Constructors
AdjointIndexing::AdjointIndexing
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    const AdjointRASModel& adjRAS
)  
    :
    mesh_(mesh),
    adjIO_(adjIO),
    adjReg_(adjReg),
    adjRAS_(adjRAS),
    pointProcAddressing
    (
        IOobject
        (
            "pointProcAddressing",
            mesh_.facesInstance(),
            mesh_.meshSubDir,
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        {}
    )
    
{

    // Calculate the sizes
    // local denotes the current MPI process
    // global denotes all the MPI processes

    // Setup the adjoint state name and type lists.
    forAll(adjReg_.volVectorStates,idxI) 
    {
        adjStateNames.append( adjReg_.volVectorStates[idxI] );
        adjStateType.set( adjReg_.volVectorStates[idxI], "volVectorState" );
    }
    forAll(adjReg_.volScalarStates,idxI) 
    {
        adjStateNames.append( adjReg_.volScalarStates[idxI] );
        adjStateType.set( adjReg_.volScalarStates[idxI], "volScalarState" );
    }
    forAll(adjRAS_.turbStates,idxI) 
    {
        adjStateNames.append( adjRAS_.turbStates[idxI] );
        adjStateType.set( adjRAS_.turbStates[idxI], "turbState" );
    }
    forAll(adjReg_.surfaceScalarStates,idxI) 
    {
        adjStateNames.append( adjReg_.surfaceScalarStates[idxI] );
        adjStateType.set( adjReg_.surfaceScalarStates[idxI], "surfaceScalarState" );
    }
    
    //Info<<"adjStateNames"<<adjStateNames<<endl;
    Info<<"Adjoint States: "<<adjStateType<<endl;
    
    // Local mesh related sizes 
    nLocalCells = mesh.nCells();
    nLocalFaces = mesh.nFaces();
    nLocalPoints = mesh.nPoints();
    nLocalXv = nLocalPoints*3;
    nLocalInternalFaces = mesh.nInternalFaces();
    nLocalBoundaryFaces = nLocalFaces - nLocalInternalFaces;
    nLocalBoundaryPatches = mesh.boundaryMesh().size();
    
    // get bFacePatchI and bFaceFaceI
    // these two lists store the patchI and faceI for a given boundary mesh face index
    // the index of these lists starts from the first boundary face of the first boundary patch.
    // they will be used to quickly get the patchI and faceI from a given boundary face index
    bFacePatchI.setSize(nLocalBoundaryFaces);
    bFaceFaceI.setSize(nLocalBoundaryFaces);
    label tmpCounter=0;
    forAll(mesh_.boundaryMesh(),patchI)
    {
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            bFacePatchI[tmpCounter] = patchI;
            bFaceFaceI[tmpCounter] = faceI;
            tmpCounter++;
        }
    }
    
    // Initialize state local index offset, it will be used in getLocalStateIndex function
    this->initializeStateLocalIndexOffset(stateLocalIndexOffset);
    
    // Initialize adjStateID. It stores the stateID for a given stateName
    this->initializeAdjStateID(adjStateID);
     
    // Local adjoint state sizes 
    // first get how many state variables are registered.
    // Note turbStates are treated separatedly
    nVolScalarStates = adjReg_.volScalarStates.size();
    nVolVectorStates = adjReg_.volVectorStates.size();
    nSurfaceScalarStates = adjReg_.surfaceScalarStates.size();
    nTurbStates = adjRAS_.turbStates.size();

    // we can now calculate adjoint state size
    label nLocalCellStates =(nVolVectorStates*3+nVolScalarStates+nTurbStates) * nLocalCells;
    label nLocalFaceStates = nSurfaceScalarStates * nLocalFaces;
    nLocalAdjointStates = nLocalCellStates + nLocalFaceStates;
    
    // Setup the global numbering to convert a local index to the associated global index
    globalAdjointStateNumbering = this->genGlobalIndex(nLocalAdjointStates);
    globalCellNumbering = this->genGlobalIndex(nLocalCells);
    globalCellVectorNumbering = this->genGlobalIndex(nLocalCells*3);
    globalFaceNumbering = this->genGlobalIndex(nLocalFaces);
    globalXvNumbering = this->genGlobalIndex(nLocalXv);
    
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
        for(label i=0;i<nLocalPoints;i++)
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
        nUndecomposedPoints = pointMaxIdx+1;
    }

    // Print relevant sizes to screen
    Info<<"Global Cells: "<<nGlobalCells<<endl;
    Info<<"Global Faces: "<<nGlobalFaces<<endl;
    Info<<"Global Xv: "<<nGlobalXv<<endl;
    Info<<"Undecomposed points: "<<nUndecomposedPoints<<endl;
    Info<<"Global Adjoint States: "<<nGlobalAdjointStates<<endl;
    
    // initialize stuff for coloring
    if(adjIO_.useColoring)
    {
        // calculate nLocalCoupledBFaces and isCoupledFace
        isCoupledFace.setSize(nLocalFaces);
        for(label i=0;i<nLocalFaces;i++) isCoupledFace[i]=0;
        
        nLocalCoupledBFaces=0;
        label faceIdx = nLocalInternalFaces;
        forAll(mesh_.boundaryMesh(),patchI)
        {
            forAll(mesh_.boundaryMesh()[patchI],faceI)
            {
                if(mesh_.boundaryMesh()[patchI].coupled())
                {
                    // this is a coupled patch
                    isCoupledFace[faceIdx]=1;
                    nLocalCoupledBFaces++;
                }
                faceIdx++;
            }
        }
        
        globalCoupledBFaceNumbering = this->genGlobalIndex(nLocalCoupledBFaces);
        nGlobalCoupledBFaces = globalCoupledBFaceNumbering.size();

        // calculate nLocalCyclicAMIFaces and isCyclicAMIFace
        isCyclicAMIFace.setSize(nLocalFaces);
        for(label i=0;i<nLocalFaces;i++) isCyclicAMIFace[i]=0;
        
        nLocalCyclicAMIFaces=0;
        faceIdx = nLocalInternalFaces;
        forAll(mesh_.boundaryMesh(),patchI)
        {
            forAll(mesh_.boundaryMesh()[patchI],faceI)
            {
                if(mesh_.boundaryMesh()[patchI].type()=="cyclicAMI")
                {
                    // this is a cyclicAMI patch
                    isCyclicAMIFace[faceIdx]=1;
                    nLocalCyclicAMIFaces++;
                }
                faceIdx++;
            }
        }
    }

    this->initializeLocalIdxLists();
    
    // check if we have user-defined patches or volumes, if yes, calculate their face and cell indices
    if (adjIO_.userDefinedPatchInfo.size()!=0)
    {
        this->calcFaceIndx4UserDefinedPatches();
    }
    if (adjIO_.userDefinedVolumeInfo.size()!=0)
    {
        this->calcCellIndx4UserDefinedVolumes();
    }
   
}

globalIndex AdjointIndexing::genGlobalIndex(const label index)
{

    globalIndex result(index);
    return result;
    
}


label AdjointIndexing::getLocalAdjointStateIndex
(
    const word stateName,
    const label idxJ,
    label comp
)
{
    // NOTE: for volVectorState, one need to set comp; while for other states, comp is simply ignored in this function

    if(adjIO_.adjJacMatOrdering == "state")
    {
        // state by state indexing
        // we set u_0, v_1, w_2, u_3, v_4, w_5, ...., p_np, p_np+1, ... nuTilda_nnu, nuTilda_nnu+1, ...
        // so getStateLocalIndexingOffset(p) will return np
        // For vector, one need to provide comp, for scalar, comp is not needed.
        
        forAll(adjStateNames,idxI)
        {
            if (adjStateNames[idxI] == stateName)
            {
                if(adjStateType[stateName] == "volVectorState")
                {
                    if (comp==-1) 
                    {
                        FatalErrorIn("")<<"comp needs to be set for vector states!"
                                        <<abort(FatalError);
                    }
                    else
                    {
                        return stateLocalIndexOffset[stateName]+idxJ*3+comp;
                    }
                }
                else
                {
                    return stateLocalIndexOffset[stateName]+idxJ;
                }
            }
        }

    }
    else if(adjIO_.adjJacMatOrdering == "cell")
    {
        // cell by cell ordering
        // We set u_0, v_0, w_0, p_0, nuTilda_0, phi_0a,phi_0b,phi_0c.... u_N, v_N, w_N, p_N, nuTilda_N, phi_N
        // To get the local index, we need to do:
        // idxLocal = 
        //   cellI*(nVectorStates*3+nScalarStates+nTurbStates) 
        // + phiAccumulatedOffset (how many phis have been accumulated. Note: we have mulitple phis for a cellI)
        // + stateLocalIndexOffset+comp
        // + phiLocalOffset (only for phi idx, ie. 0a or 0b or oc. Note: we have mulitple phis for a cellI)

        label nCellStates =3*adjReg_.volVectorStates.size()
                           +adjReg_.volScalarStates.size()
                           +adjRAS_.turbStates.size();

        
        const word& stateType = adjStateType[stateName];
        label returnV=-99999999;
        if (stateType == "surfaceScalarState") // for surfaceScalarState idxJ is faceI
        {
            label idxN = faceOwner[idxJ]; // idxN is the cell who owns faceI 
            returnV = idxN*nCellStates
                     +stateLocalIndexOffset[stateName]
                     +phiAccumulatdOffset[idxN]
                     +phiLocalOffset[idxJ];
            return returnV;
        }
        else if(stateType == "volVectorState") // for other states idxJ is cellI
        {
            if (comp==-1) 
            {
                FatalErrorIn("")<<"comp needs to be set for vector states!"
                                <<abort(FatalError);
            }
            else
            {
                
                returnV = idxJ*nCellStates
                         +stateLocalIndexOffset[stateName]
                         +comp
                         +phiAccumulatdOffset[idxJ];
            }
            return returnV;
        }
        else
        {
            returnV = idxJ*nCellStates
                     +stateLocalIndexOffset[stateName]
                     +phiAccumulatdOffset[idxJ];
            return returnV; 
        }
    }
    else
    {
        FatalErrorIn("")<< "adjJacMatOrdering invalid"<<abort(FatalError);
    }
    
    // if no stateName found, return an error
    FatalErrorIn("")<< "stateName not found!"<<abort(FatalError);
    return -1;
}


label AdjointIndexing::getGlobalAdjointStateIndex
(
    const word stateName,
    const label idxI,
    label comp
)
{
    // For vector, one need to provide comp, for scalar, comp is not needed.
    label localIdx = this->getLocalAdjointStateIndex(stateName,idxI,comp);
    label globalIdx = globalAdjointStateNumbering.toGlobal(localIdx);
    return globalIdx;
}

label AdjointIndexing::getLocalSegregatedAdjointStateIndex
(
    const word stateName,
    const label idxJ,
    label comp
)
{
    
    if(adjStateType[stateName] == "volVectorState")
    {
        if (comp==-1) 
        {
            FatalErrorIn("")<<"comp needs to be set for vector states!"
                            <<abort(FatalError);
            return -999999;
        }
        else
        {
            return idxJ*3+comp;
        }
    }
    else
    {
        return idxJ;
    }

}


label AdjointIndexing::getGlobalSegregatedAdjointStateIndex
(
    const word stateName,
    const label idxI,
    label comp
)
{
    // For vector, one need to provide comp, for scalar, comp is not needed.
    label localIdx = this->getLocalSegregatedAdjointStateIndex(stateName,idxI,comp);

    label globalIdx = -99999;

    if(adjStateType[stateName] == "volVectorState")
    {
        globalIdx = globalCellVectorNumbering.toGlobal(localIdx);
    }
    else if (adjStateType[stateName] == "surfaceScalarState")
    {
        globalIdx = globalFaceNumbering.toGlobal(localIdx);
    }
    else
    {
        globalIdx = globalCellNumbering.toGlobal(localIdx);
    }
    return globalIdx;
}

label AdjointIndexing::getGlobalXvIndex
(
    const label idxPoint,
    const label idxCoord
)
{
    label localXvIdx = this->getLocalXvIndex(idxPoint,idxCoord);
    label globalXvIdx = globalXvNumbering.toGlobal(localXvIdx);
    return globalXvIdx;
}

label AdjointIndexing::getLocalXvIndex
(
    const label idxPoint,
    const label idxCoord
)
{
    label localXvIdx = idxPoint*3 + idxCoord;
    return localXvIdx;
}

void AdjointIndexing::initializeStateLocalIndexOffset
(
    HashTable<label>& offset
)
{
    // Calculate the indexing offset for all states (stateLocalIndexOffset), 
    // this will be used in the getLocalAdjointStateIndex function

    if(adjIO_.adjJacMatOrdering == "state")
    {
        // For state-by-state ordering, we set u_0, v_0, w_0, u_1, v_1, w_1, 
        // ...., p_0, p_1, ... nuTilda_0, nuTilda_1, ... with subscript being the
        // cell index. stateLocalIndexingOffset will return how many states are
        // before a specific stateName
        forAll(adjStateNames,idxI)
        {
            word& stateName = adjStateNames[idxI];
            
            label counter=0;
            
            forAll(adjReg_.volVectorStates,idx)
            {
                if (adjReg_.volVectorStates[idx] == stateName)
                {
                    offset.set(stateName,counter*nLocalCells);
                }
                counter+=3;
            }
            
            forAll(adjReg_.volScalarStates,idx)
            {
                if (adjReg_.volScalarStates[idx] == stateName)
                {
                    offset.set(stateName,counter*nLocalCells);
                }
                counter++;
            }
            
            forAll(adjRAS_.turbStates,idx)
            {
                if (adjRAS_.turbStates[idx] == stateName)
                {
                    offset.set(stateName,counter*nLocalCells);
                }
                counter++;
            }
            
            forAll(adjReg_.surfaceScalarStates,idx)
            {
                if (adjReg_.surfaceScalarStates[idx] == stateName && idx==0)
                {
                    offset.set(stateName,counter*nLocalCells);
                }
                if (adjReg_.surfaceScalarStates[idx] == stateName && idx>0)
                {
                    offset.set(stateName,counter*nLocalFaces);
                }
                counter++;
            }
        }
            
    }
    else if(adjIO_.adjJacMatOrdering == "cell")
    {
        // For cell by cell ordering
        // we set u_0, v_0, w_0, p_0, nuTilda_0, phi_0, .... u_N, v_N, w_N, p_N, nuTilda_N, phi_N
        // with subscript being the cell index. so stateLocalIndexingOffset will return 
        // how many states are before a specific stateName for a given cell index 
        forAll(adjStateNames,idxI)
        {
            word& stateName = adjStateNames[idxI];
            
            label counter=0;
            
            forAll(adjReg_.volVectorStates,idx)
            {
                if (adjReg_.volVectorStates[idx] == stateName)
                {
                    offset.set(stateName,counter);
                }
                counter+=3;
            }
            
            forAll(adjReg_.volScalarStates,idx)
            {
                if (adjReg_.volScalarStates[idx] == stateName)
                {
                    offset.set(stateName,counter);
                }
                counter++;
            }
            
            forAll(adjRAS_.turbStates,idx)
            {
                if (adjRAS_.turbStates[idx] == stateName)
                {
                    offset.set(stateName,counter);
                }
                counter++;
            }
            
            forAll(adjReg_.surfaceScalarStates,idx)
            {
                if (adjReg_.surfaceScalarStates[idx] == stateName)
                {
                    offset.set(stateName,counter);
                }
                counter++;
            }
        }
        
        // We also need a few more offsets
        
        // calculate faceOwner
        faceOwner.setSize(nLocalFaces);
        const UList<label>& internalFaceOwner = mesh_.owner(); // these only include internal faces owned cellI
        forAll(faceOwner,idxI)
        {
            if(idxI<nLocalInternalFaces) 
            {
                faceOwner[idxI] =  internalFaceOwner[idxI];
            }
            else
            {
                label relIdx=idxI-nLocalInternalFaces;
                label patchIdx=bFacePatchI[relIdx];
                label faceIdx=bFaceFaceI[relIdx];
                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                faceOwner[idxI] = pFaceCells[faceIdx];
            }
        }
        
        // Calculate the cell owned face index. Note: we can't use mesh.cells here since it will have 
        // duplicated face indices
        List< List<label> > cellOwnedFaces;
        cellOwnedFaces.setSize(nLocalCells);
        forAll(faceOwner,idxI)
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
        forAll(phiAccumulatdOffset,idxI) phiAccumulatdOffset[idxI] = -9999999;
        forAll(phiAccumulatdOffset,idxI)
        {
            if(idxI==0) phiAccumulatdOffset[idxI]=0;
            else phiAccumulatdOffset[idxI] = cellOwnedFaces[idxI-1].size()+phiAccumulatdOffset[idxI-1];
        }
        //Info<<"phiAccumulatdOffset "<<phiAccumulatdOffset<<endl;
        
        // Now calculate the phiLocalOffset
        phiLocalOffset.setSize(nLocalFaces);
        forAll(phiLocalOffset,idxI) phiLocalOffset[idxI] = -9999999;
        forAll(cellOwnedFaces,idxI) // idxI is cell Index
        {
            forAll(cellOwnedFaces[idxI],offsetI) 
            {
                label ownedFace = cellOwnedFaces[idxI][offsetI];
                phiLocalOffset[ownedFace] = offsetI;
            }
        }
        //Info<<"phiLocalOffset "<<phiLocalOffset<<endl;
        //Info<<"stateLocalIndexOffset "<<stateLocalIndexOffset<<endl;
        
        
    }
    else
    {
        FatalErrorIn("")<< "adjJacMatOrdering invalid"<<abort(FatalError);
    }
    
    return;
}


void AdjointIndexing::initializeAdjStateID(HashTable<label>& adjStateID)
{
    // the stateID is an alternative for the stateNames
    // stateID starts from 0 for the first volVector state
    
    label id=0;
    forAll(adjReg_.volVectorStates,idx)
    {
        word stateName = adjReg_.volVectorStates[idx];
        adjStateID.set(stateName,id);
        id++;
    }
    
    forAll(adjReg_.volScalarStates,idx)
    {
        word stateName = adjReg_.volScalarStates[idx];
        adjStateID.set(stateName,id);
        id++;
    }
    
    forAll(adjRAS_.turbStates,idx)
    {
        word stateName = adjRAS_.turbStates[idx];
        adjStateID.set(stateName,id);
        id++;
    }
    
    forAll(adjReg_.surfaceScalarStates,idx)
    {
        word stateName = adjReg_.surfaceScalarStates[idx];
        adjStateID.set(stateName,id);
        id++;
    }
    return;
}


List<word> AdjointIndexing::getObjFuncGeoInfo(const word objFunc)
{
    // Return the obj func patches for a given objFunc
    forAll(adjIO_.objFuncs,idxI)
    {
        if (adjIO_.objFuncs[idxI] == objFunc)
        {
            return adjIO_.objFuncGeoInfo[idxI];
        }
    }
    
    // objFunc not found
    FatalErrorIn("")<< "objFunc not found!"<<abort(FatalError);
    return adjIO_.objFuncGeoInfo[0]; // to make C++ happy
}

label AdjointIndexing::getNLocalObjFuncGeoElements(const word objFunc)
{
    // Return local objective function face sizes for a given objFunc
    label nLocalObjFuncGeoElements = 0;
    List<word> objFuncGeoInfo=this->getObjFuncGeoInfo(objFunc);
    forAll(objFuncGeoInfo,idxI)
    {
        if( this->isUserDefinedPatch(objFuncGeoInfo[idxI]) )
        {
            labelList userDefinedPatchFaces = faceIdx4UserDefinedPatches[objFuncGeoInfo[idxI]];
            nLocalObjFuncGeoElements += userDefinedPatchFaces.size();
        }
        else if ( this->isUserDefinedVolume(objFuncGeoInfo[idxI]) )
        {
            labelList userDefinedVolumeCells = cellIdx4UserDefinedVolumes[objFuncGeoInfo[idxI]];
            nLocalObjFuncGeoElements += userDefinedVolumeCells.size();
        }
        else if ( objFuncGeoInfo[idxI]=="allCells" )
        {
            nLocalObjFuncGeoElements = mesh_.nCells();
            return nLocalObjFuncGeoElements;
        }
        else
        {
            // get the patch id
            label patchI = mesh_.boundaryMesh().findPatchID( objFuncGeoInfo[idxI] );
            if (patchI <0) 
            {
                FatalErrorIn("")<< "objFuncPatches not found in the boundary mesh!"<<abort(FatalError);
            }
            // create a shorter handle for the boundary patch
            const fvPatch& patch = mesh_.boundary()[patchI];
            forAll(patch, faceI)
            {
                nLocalObjFuncGeoElements += 1 ;
            }
        }
    }
    return nLocalObjFuncGeoElements;
}

void AdjointIndexing::initializeObjFuncGeoNumbering(const word objFunc)
{
    label nLocalObjFuncGeoElements = this->getNLocalObjFuncGeoElements(objFunc);
    globalObjFuncGeoNumbering = this->genGlobalIndex(nLocalObjFuncGeoElements);
    return;
}

void AdjointIndexing::deleteObjFuncGeoNumbering()
{
    // How to delete this or we don't need to delete it?
    //globalObjFuncGeoNumbering.clear();
}


label AdjointIndexing::BFacePatchIFaceI2LocalIndex
(
    const label patchI,
    const label faceI
)
{
    // given a local mesh patchI and faceI, return its local face index
    label localIdx;
    localIdx = mesh_.boundaryMesh()[patchI].start()+faceI;
    return localIdx;
}

void AdjointIndexing::BFaceLocalIndex2PatchIFaceI
(
    const label idxI,
    label& patchIdx,
    label& faceIdx
)
{
    // given a local face index, return its boundary mesh patchI and faceI
    
    // first check if idxI is valid
    if (idxI <= nLocalInternalFaces or idxI >= nLocalFaces)
    {
        FatalErrorIn("")<< "idxI is not valid!"<<abort(FatalError);
    }
    
    forAll(mesh_.boundaryMesh(),patchI)
    {
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            if ( (mesh_.boundaryMesh()[patchI].start()+faceI)==idxI  )
            {
                patchIdx=patchI;
                faceIdx=faceI;
                return;
            }
        }
    }
    
    // if no idxI found, return an error
    FatalErrorIn("")<< "idxI not found in the boundary mesh!"<<abort(FatalError);
    return;
}


void AdjointIndexing::calcAdjStateID4GlobalAdjIdx(labelList& adjStateID4GlobalAdjIdx)
{ 
    // compute adjStateID4GlobalAdjIdx
    
    // adjStateID4GlobalAdjIdx stores the adjStateID for given a global adj index
    // NOTE: adjStateID4GlobalAdjIdx contains all the global adj indices, so its memory usage 
    // is high. We should avoid having any sequential list; however, to make the connectivity
    // calculation easier, we keep it for now. 
    // *******delete this list after used!************
    
    if (adjStateID4GlobalAdjIdx.size()!=nGlobalAdjointStates)
    {
        FatalErrorIn("")<<"adjStateID4GlobalAdjIdx.size()!=nGlobalAdjointStates"<<abort(FatalError);
    }

    Vec stateIVec;
    VecCreate(PETSC_COMM_WORLD,&stateIVec);
    VecSetSizes(stateIVec,nLocalAdjointStates,PETSC_DECIDE);
    VecSetFromOptions(stateIVec);
    VecSet(stateIVec,0); // default value
    
    forAll(adjReg_.volVectorStates,idx)
    {
        word stateName = adjReg_.volVectorStates[idx];
        PetscScalar valIn=adjStateID[stateName]+1;  // we need to use 1-based indexing here for scattering
        forAll(mesh_.cells(),cellI)
        {
            for(label i=0;i<3;i++)
            {
                label globalIdx = this->getGlobalAdjointStateIndex(stateName,cellI,i);
                VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
            }
            
        }
    }
    
    forAll(adjReg_.volScalarStates,idx)
    {
        word stateName = adjReg_.volScalarStates[idx];
        PetscScalar valIn=adjStateID[stateName]+1;  // we need to use 1-based indexing here for scattering
        forAll(mesh_.cells(),cellI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName,cellI);
            VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
        }
    }
    
    forAll(adjRAS_.turbStates,idx)
    {
        word stateName = adjRAS_.turbStates[idx];
        PetscScalar valIn=adjStateID[stateName]+1;  // we need to use 1-based indexing here for scattering
        forAll(mesh_.cells(),cellI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName,cellI);
            VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
        }
    }
    
    forAll(adjReg_.surfaceScalarStates,idx)
    {
        word stateName = adjReg_.surfaceScalarStates[idx];
        PetscScalar valIn=adjStateID[stateName]+1;  // we need to use 1-based indexing here for scattering
        forAll(mesh_.faces(),faceI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName,faceI);
            VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
        }
    }
    

    VecAssemblyBegin(stateIVec);
    VecAssemblyEnd(stateIVec);
    
    // scatter to local array for all procs
    Vec vout;
    VecScatter ctx;
    VecScatterCreateToAll(stateIVec,&ctx,&vout);
    VecScatterBegin(ctx,stateIVec,vout,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(ctx,stateIVec,vout,INSERT_VALUES,SCATTER_FORWARD);
    
    PetscScalar* stateIVecArray;
    VecGetArray(vout,&stateIVecArray);

    for(label i=0;i<nGlobalAdjointStates;i++)
    {
        adjStateID4GlobalAdjIdx[i]=static_cast<label>(stateIVecArray[i])-1; // subtract 1 and return to 0-based indexing
    }

    VecRestoreArray(vout,&stateIVecArray);
    VecScatterDestroy(&ctx);
    VecDestroy(&vout);
    
    return;
}

void AdjointIndexing::calcCellIFaceI4GlobalAdjIdx(scalarList& cellIFaceI4GlobalAdjIdx)
{ 
    // compute adjStateID4GlobalAdjIdx
    
    // adjStateID4GlobalAdjIdx stores the adjStateID for given a global adj index
    // NOTE: adjStateID4GlobalAdjIdx contains all the global adj indices, so its memory usage 
    // is high. We should avoid having any sequential list; however, to make the connectivity
    // calculation easier, we keep it for now. 
    // *******delete this list after used!************
    
    if (cellIFaceI4GlobalAdjIdx.size()!=nGlobalAdjointStates)
    {
        FatalErrorIn("")<<"adjStateID4GlobalAdjIdx.size()!=nGlobalAdjointStates"<<abort(FatalError);
    }

    Vec stateIVec;
    VecCreate(PETSC_COMM_WORLD,&stateIVec);
    VecSetSizes(stateIVec,nLocalAdjointStates,PETSC_DECIDE);
    VecSetFromOptions(stateIVec);
    VecSet(stateIVec,0); // default value
    
    forAll(adjReg_.volVectorStates,idx)
    {
        word stateName = adjReg_.volVectorStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            for(label i=0;i<3;i++)
            {
                label globalIdx = this->getGlobalAdjointStateIndex(stateName,cellI,i);
                scalar valIn = cellI+i*0.1+1;  // need to use 1-based indexing for scattering
                VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
            }
            
        }
    }
    
    forAll(adjReg_.volScalarStates,idx)
    {
        word stateName = adjReg_.volScalarStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName,cellI);
            scalar valIn = cellI+1; // need to use 1-based indexing for scattering
            VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
        }
    }
    
    forAll(adjRAS_.turbStates,idx)
    {
        word stateName = adjRAS_.turbStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName,cellI);
            scalar valIn = cellI+1; // need to use 1-based indexing for scattering
            VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
        }
    }
    
    forAll(adjReg_.surfaceScalarStates,idx)
    {
        word stateName = adjReg_.surfaceScalarStates[idx];
        forAll(mesh_.faces(),faceI)
        {
            label globalIdx = this->getGlobalAdjointStateIndex(stateName,faceI);
            scalar valIn = faceI+1; // need to use 1-based indexing for scattering
            VecSetValues(stateIVec,1,&globalIdx,&valIn,INSERT_VALUES);
        }
    }
    

    VecAssemblyBegin(stateIVec);
    VecAssemblyEnd(stateIVec);
    
    // scatter to local array for all procs
    Vec vout;
    VecScatter ctx;
    VecScatterCreateToAll(stateIVec,&ctx,&vout);
    VecScatterBegin(ctx,stateIVec,vout,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(ctx,stateIVec,vout,INSERT_VALUES,SCATTER_FORWARD);
    
    PetscScalar* stateIVecArray;
    VecGetArray(vout,&stateIVecArray);

    for(label i=0;i<nGlobalAdjointStates;i++)
    {
        cellIFaceI4GlobalAdjIdx[i]=stateIVecArray[i]-1; // subtract 1 and return to 0-based indexing
    }

    VecRestoreArray(vout,&stateIVecArray);
    VecScatterDestroy(&ctx);
    VecDestroy(&vout);
    
    return;
}

void AdjointIndexing::initializeLocalIdxLists()
{
    // Initialize indexing lists: 
    // cellIFaceI4LocalAdjIdx
    // adjStateName4LocalAdjIdx
        
    // cellIFaceI4LocalAdjIdx stores the cell/face index for a local adjoint index
    // For vector fields, the decima of cellIFaceI4LocalAdjIdx denotes the vector component
    // e.g., 10.1 means cellI=10, y compoent of U
    
    // adjStateName4LocalAdjIdx stores the state name for a local adjoint index

 
    cellIFaceI4LocalAdjIdx.setSize(nLocalAdjointStates);
    adjStateName4LocalAdjIdx.setSize(nLocalAdjointStates);
        
    forAll(adjReg_.volVectorStates,idx)
    {
        word stateName = adjReg_.volVectorStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            for(label i=0;i<3;i++)
            {               
                label localIdx = this->getLocalAdjointStateIndex(stateName,cellI,i);
                cellIFaceI4LocalAdjIdx[localIdx] = cellI+i/10.0;
                
                adjStateName4LocalAdjIdx[localIdx] = stateName;
            }
            
        }
    }
    
    forAll(adjReg_.volScalarStates,idx)
    {
        word stateName = adjReg_.volScalarStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            label localIdx = this->getLocalAdjointStateIndex(stateName,cellI);
            cellIFaceI4LocalAdjIdx[localIdx] = cellI;
            
            adjStateName4LocalAdjIdx[localIdx] = stateName;
        }
    }
    
    forAll(adjRAS_.turbStates,idx)
    {
        word stateName = adjRAS_.turbStates[idx];
        forAll(mesh_.cells(),cellI)
        {            
            label localIdx = this->getLocalAdjointStateIndex(stateName,cellI);
            cellIFaceI4LocalAdjIdx[localIdx] = cellI;
            
            adjStateName4LocalAdjIdx[localIdx] = stateName;
        }
    }
    
    forAll(adjReg_.surfaceScalarStates,idx)
    {
        word stateName = adjReg_.surfaceScalarStates[idx];
        forAll(mesh_.faces(),faceI)
        {            
            label localIdx = this->getLocalAdjointStateIndex(stateName,faceI);
            cellIFaceI4LocalAdjIdx[localIdx] = faceI;
            
            adjStateName4LocalAdjIdx[localIdx] = stateName;
        }
    }
    
    return;
    
}


void AdjointIndexing::writeAdjointIndexing()
{


    scalar xx,yy,zz; // face owner coordinates

    //output the matrix to a file
    label myProc = Pstream::myProcNo();
    label nProcs = Pstream::nProcs();
    std::ostringstream fileNameStream("");
    fileNameStream<<"AdjointIndexing"<<"_"<<myProc<<"_of_"<<nProcs<<".txt";
    word fileName = fileNameStream.str();
    OFstream aOut(fileName);
    aOut.precision(9);

/*
    std::ostringstream fileNameStreamPoint("");
    fileNameStreamPoint<<"PointCoordinates"<<"_"<<myProc<<"_of_"<<nProcs<<".txt";
    word fileNamePoint = fileNameStreamPoint.str();
    OFstream aOutPoint(fileNamePoint);
    aOutPoint.precision(9);
*/
    forAll(adjReg_.volVectorStates,idx)
    {
        const word& stateName = adjReg_.volVectorStates[idx];
        forAll(mesh_.cells(), cellI)
        {
            xx=mesh_.C()[cellI].x();
            yy=mesh_.C()[cellI].y();
            zz=mesh_.C()[cellI].z();
            for(label i=0; i<3; i++)
            {  
                label glbIdx = getGlobalAdjointStateIndex(stateName,cellI,i);
                aOut << "Cell: "<<cellI<<" State: "<<stateName<<i<<" glbIdx: "<<glbIdx <<" x: "<<xx<<" y: "<<yy<<" z: "<<zz<<endl;
            }
        }
    }
    
    forAll(adjReg_.volScalarStates,idx)
    {
        const word& stateName = adjReg_.volScalarStates[idx];
        forAll(mesh_.cells(), cellI)
        {
            xx=mesh_.C()[cellI].x();
            yy=mesh_.C()[cellI].y();
            zz=mesh_.C()[cellI].z();
            label glbIdx = getGlobalAdjointStateIndex(stateName,cellI);
            aOut << "Cell: "<<cellI<<" State: "<<stateName<<" glbIdx: "<<glbIdx <<" x: "<<xx<<" y: "<<yy<<" z: "<<zz<<endl;
        }
    }
    
    forAll(adjRAS_.turbStates,idx)
    {
        const word& stateName = adjRAS_.turbStates[idx];
        forAll(mesh_.cells(), cellI)
        {
            xx=mesh_.C()[cellI].x();
            yy=mesh_.C()[cellI].y();
            zz=mesh_.C()[cellI].z();
            label glbIdx = getGlobalAdjointStateIndex(stateName,cellI);
            aOut << "Cell: "<<cellI<<" State: "<<stateName<<" glbIdx: "<<glbIdx <<" x: "<<xx<<" y: "<<yy<<" z: "<<zz<<endl;
        }
    }
    
    forAll(adjReg_.surfaceScalarStates,idx)
    {
        const word& stateName = adjReg_.surfaceScalarStates[idx];
        label cellI=-1;
        forAll(mesh_.faces(), faceI)
        {
            if (faceI < nLocalInternalFaces)
            {
                xx=mesh_.Cf()[faceI].x();
                yy=mesh_.Cf()[faceI].y();
                zz=mesh_.Cf()[faceI].z();
                
            }
            else
            {
                label relIdx=faceI-nLocalInternalFaces;
                label patchIdx=bFacePatchI[relIdx];
                label faceIdx=bFaceFaceI[relIdx];
                xx=mesh_.Cf().boundaryField()[patchIdx][faceIdx].x();
                yy=mesh_.Cf().boundaryField()[patchIdx][faceIdx].y();
                zz=mesh_.Cf().boundaryField()[patchIdx][faceIdx].z();

                const polyPatch& pp = mesh_.boundaryMesh()[patchIdx];
                const UList<label>& pFaceCells = pp.faceCells();
                cellI=pFaceCells[faceIdx];
            } 
            
            label glbIdx = getGlobalAdjointStateIndex(stateName,faceI);
            aOut << "Face: "<<faceI<<" State: "<<stateName<<" glbIdx: "<<glbIdx <<" x: "<<xx<<" y: "<<yy<<" z: "<<zz<<" OwnerCellI: "<<cellI<<endl;
  
        }
        
    }

    // write point indexing
    std::ostringstream fileNameStreamPoint("");
    fileNameStreamPoint<<"PointIndexing"<<"_"<<myProc<<"_of_"<<nProcs<<".txt";
    word fileNamePoint = fileNameStreamPoint.str();
    OFstream aOutPoint(fileNamePoint);
    aOutPoint.precision(9);

    forAll(mesh_.points(),idxI)
    {
        xx=mesh_.points()[idxI].x();
        yy=mesh_.points()[idxI].y();
        zz=mesh_.points()[idxI].z();
        for(label i=0;i<3;i++)
        {
            label glbIdx = getGlobalXvIndex(idxI,i);
            aOutPoint << "Point: "<<idxI<<" Coords: "<<i<<" glbIdx: "<<glbIdx <<" x: "<<xx<<" y: "<<yy<<" z: "<<zz<<endl;
        }
    }
    
    return;
}


void AdjointIndexing::getMatNonZeros
(
    Mat matIn,
    label& maxCols, 
    scalar& allNonZeros
)
{
    // get the max nonzeros per row, and all the nonzeros for this matrix

    PetscInt nCols, Istart, Iend;
    const PetscInt    *cols;
    const PetscScalar *vals;

    // set the counter
    maxCols = 0;
    allNonZeros = 0.0;

    // Determine which rows are on the current processor
    MatGetOwnershipRange(matIn,&Istart,&Iend);

    // loop over the matrix and find the largest number of cols
    for(label i=Istart; i<Iend; i++)
    {
        MatGetRow(matIn,i,&nCols,&cols,&vals);
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
        MatRestoreRow(matIn,i,&nCols,&cols,&vals);
    }

    //reduce the maxcols value so that all procs have the same size
    reduce(maxCols, maxOp<int>());
    
    reduce(allNonZeros, sumOp<scalar>());

    return;
}

void AdjointIndexing::printMatChars(Mat matIn)
{

    PetscInt nCols, Istart, Iend;
    const PetscInt    *cols;
    const PetscScalar *vals;
    scalar maxRatio=0.0;
    label maxRatioRow=-1;
    scalar diagV=-1.0;
    scalar maxNonDiagV=-1.0;
    label maxNonDiagCol=-1;
    scalar small = 1e-12;
    scalar allNonZeros=0.0;
    label maxCols=-1;
    
    this->getMatNonZeros(matIn,maxCols,allNonZeros);

    MatGetOwnershipRange(matIn,&Istart,&Iend);
    for(label i=Istart; i<Iend; i++)
    {
        scalar diag = 0;
        scalar nonDiagSum = 0;
        scalar maxV=0.0;
        label maxVIdx=-1;
        MatGetRow(matIn,i,&nCols,&cols,&vals);
        for(label n=0; n<nCols; n++)
        {
            if(i==cols[n])
            {
                diag=vals[n];
            }
            if(vals[n]!=0)
            {
                if(i!=cols[n])
                {
                    nonDiagSum=nonDiagSum+fabs(vals[n]);
                }
                if(fabs(vals[n])>maxV)
                {
                    maxV=fabs(vals[n]);
                    maxVIdx=cols[n];
                }

            }
        }

        if ( fabs(nonDiagSum/(diag+small)) > maxRatio )
        {
            maxRatio=fabs(nonDiagSum/(diag+small));
            maxRatioRow=i;
            maxNonDiagCol=maxVIdx;
            diagV=diag;
            maxNonDiagV=maxV;
        }

        MatRestoreRow(matIn,i,&nCols,&cols,&vals);
    }

    label rowStateID=-1;
    label colStateID=-1;
    vector rowCoord(0,0,0), colCoord(0,0,0);

    forAll(adjReg_.volVectorStates,idx)
    {
        const word& stateName = adjReg_.volVectorStates[idx];
        forAll(mesh_.cells(), cellI)
        {
            for(label i=0; i<3; i++)
            {
                label idxJ = getGlobalAdjointStateIndex(stateName,cellI,i);
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
    
    forAll(adjReg_.volScalarStates,idx)
    {
        const word& stateName = adjReg_.volScalarStates[idx];
        forAll(mesh_.cells(), cellI)
        {

            label idxJ = getGlobalAdjointStateIndex(stateName,cellI);
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
    
    forAll(adjRAS_.turbStates,idx)
    {
        const word& stateName = adjRAS_.turbStates[idx];
        forAll(mesh_.cells(), cellI)
        {

            label idxJ = getGlobalAdjointStateIndex(stateName,cellI);
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
    
    forAll(adjReg_.surfaceScalarStates,idx)
    {
        const word& stateName = adjReg_.surfaceScalarStates[idx];
        forAll(mesh_.faces(), faceI)
        {

            label idxJ = getGlobalAdjointStateIndex(stateName,faceI);
            
            if (idxJ == maxRatioRow)
            {
                rowStateID = adjStateID[stateName];
                if(faceI < nLocalInternalFaces)
                {
                    rowCoord = mesh_.Cf()[faceI];
                }
                else
                {
                    label relIdx=faceI-nLocalInternalFaces;
                    label patchIdx=bFacePatchI[relIdx];
                    label faceIdx=bFaceFaceI[relIdx];
                    rowCoord=mesh_.Cf().boundaryField()[patchIdx][faceIdx];
                }
                
            }
            if (idxJ == maxNonDiagCol)
            {
                colStateID = adjStateID[stateName];
                if(faceI < nLocalInternalFaces)
                {
                    colCoord = mesh_.Cf()[faceI];
                }
                else
                {
                    label relIdx=faceI-nLocalInternalFaces;
                    label patchIdx=bFacePatchI[relIdx];
                    label faceIdx=bFaceFaceI[relIdx];
                    colCoord=mesh_.Cf().boundaryField()[patchIdx][faceIdx];
                }
            }
        }
    }


    // create a list to store the info
    List<scalar> matCharInfo(13);
    matCharInfo[0]=maxRatio;
    matCharInfo[1]=diagV;
    matCharInfo[2]=maxNonDiagV;
    matCharInfo[3]=maxRatioRow;
    matCharInfo[4]=rowStateID;
    matCharInfo[5]=rowCoord.x();
    matCharInfo[6]=rowCoord.y();
    matCharInfo[7]=rowCoord.z();
    matCharInfo[8]=maxNonDiagCol;
    matCharInfo[9]=colStateID;
    matCharInfo[10]=colCoord.x();
    matCharInfo[11]=colCoord.y();
    matCharInfo[12]=colCoord.z();
    
    // now gather all the info
    label myProc = Pstream::myProcNo();
    label nProcs = Pstream::nProcs();
    // create listlist for gathering 
    List< List<scalar> > gatheredList(nProcs);
    // assign values for the listlists
    gatheredList[myProc]=matCharInfo;
    // gather all info to the master proc
    Pstream::gatherList(gatheredList);
    // scatter all info to every procs
    Pstream::scatterList(gatheredList);
    
    scalar maxRatioGathered = -1.0;
    label procI=-1;
    for(label i=0;i<nProcs;i++)
    {
        if( fabs(gatheredList[i][0]) > maxRatioGathered )
        {
            maxRatioGathered=fabs(gatheredList[i][0]);
            procI=i;
        }
    }

    Info<<endl;
    Info<<"Jacobian Matrix Characteristics: "<<endl;
    Info<<" Mat maxCols: "<<maxCols<<endl;
    Info<<" Mat allNonZeros: "<<allNonZeros<<endl;
    Info<<" Max nonDiagSum/Diag: "<<gatheredList[procI][0]
        <<" Diag: "<<gatheredList[procI][1]<<" MaxNonDiag: "<<gatheredList[procI][2]<<endl;
    Info<<" MaxRatioRow: "<<gatheredList[procI][3]<<" RowState: "<<gatheredList[procI][4]
        <<" RowCoord: ("<<gatheredList[procI][5]<<" "<<gatheredList[procI][6]
        <<" "<<gatheredList[procI][7]<<")"<<endl;
    Info<<" MaxNonDiagCol: "<<gatheredList[procI][8]<<" ColState: "<<gatheredList[procI][9]
        <<" ColCoord: ("<<gatheredList[procI][10]<<" "<<gatheredList[procI][11]
        <<" "<<gatheredList[procI][12]<<")"<<endl;
    Info<<" Max nonDiagSum/Diag ProcI: "<<procI<<endl;
    Info<<endl;

    return;

}

label AdjointIndexing::isUserDefinedPatch(word geoInfo)
{
    label returnV;
    // we only consider names "userDefinedPatch0", "userDefinedPatch1", etc
    // as user defined patches. NOTE: they are case sensitive

    wordReList wordrelist
    {
        {"userDefinedPatch.*", wordRe::REGEX},
    };
    wordRes wrelist(wordrelist);

    returnV=wrelist(geoInfo);
    
    return returnV;
}

label AdjointIndexing::isUserDefinedVolume(word geoInfo)
{
    label returnV;
    // we only consider names "userDefinedVolume0", "userDefinedVolume1", etc
    // as user defined patches. NOTE: they are case sensitive

    wordReList wordrelist
    {
        {"userDefinedVolume.*", wordRe::REGEX},
    };
    wordRes wrelist(wordrelist);

    returnV=wrelist(geoInfo);
    
    return returnV;
}

void AdjointIndexing::calcCellIndx4UserDefinedVolumes()
{
    // basically, we add the cell index that is inside the prescribed volumes
    // defined in userDefinedVolumeInfo, and set it to cellIndx4UserDefinedVolumes.

    // first check if the names are valid
    forAll(adjIO_.userDefinedVolumeInfo.toc(),idxI)
    {
        word geoName=adjIO_.userDefinedVolumeInfo.toc()[idxI];
        if (!isUserDefinedVolume(geoName))
        {
            Info<<geoName<<endl;
            FatalErrorIn("")<<"valid userDefinedVolume format is userDefinedVolume0, userDefinedVolume1, ..."<<abort(FatalError);
        }
    }

    forAll(adjIO_.userDefinedVolumeInfo.toc(),idxI)
    {
        word geoName=adjIO_.userDefinedVolumeInfo.toc()[idxI];

        dictionary geoDict = adjIO_.userDefinedVolumeInfo.subDict(geoName);

        word geoType = word(geoDict["type"]);

        if( geoType== "box") // it is a box
        {
            /*
                example:
                userDefinedVolume0
                {
                    type box;
                    stateName U;
                    component 0;
                    scale 1.0;
                    centerX 0.1;
                    centerY 0.1;
                    centerZ 0.1;
                    sizeX   1.0;
                    sizeY   1.0;
                    sizeZ   1.0;
                }
            */

            scalar centerX = readScalar( geoDict["centerX"] );
            scalar centerY = readScalar( geoDict["centerY"] );
            scalar centerZ = readScalar( geoDict["centerZ"] );
            scalar sizeX   = readScalar( geoDict["sizeX"] );
            scalar sizeY   = readScalar( geoDict["sizeY"] );
            scalar sizeZ   = readScalar( geoDict["sizeZ"] );

            scalar xMax = centerX + sizeX/2.0;
            scalar xMin = centerX - sizeX/2.0;
            scalar yMax = centerY + sizeY/2.0;
            scalar yMin = centerY - sizeY/2.0;
            scalar zMax = centerZ + sizeZ/2.0;
            scalar zMin = centerZ - sizeZ/2.0;
           
            labelList cellList;
            
            forAll(mesh_.C(),cellI)
            {
                scalar cX=mesh_.C()[cellI].x();
                scalar cY=mesh_.C()[cellI].y();
                scalar cZ=mesh_.C()[cellI].z();           
                if( cX >= xMin && cX <= xMax && cY >= yMin && cY <= yMax && cZ >= zMin && cZ <= zMax )
                {
                    cellList.append(cellI);
                    //Info<<"cX "<<cX<<" cY "<<cY<<" cZ "<<cZ<<" cellI "<<cellI<<endl;
                }

            }

            cellIdx4UserDefinedVolumes.set(geoName,cellList);
        }
        else if (geoType=="sphere") // it is a sphere
        {
            FatalErrorIn("")<<"sphere not implemented"<<abort(FatalError);
        }
        else if (geoType=="cylinder") // it is a cylinder
        {
            
            /*
                example:
                userDefinedVolume0
                {
                    type cylinder;
                    stateName U;
                    component 0;
                    scale 1.0;
                    centerX 0.1;
                    centerY 0.1;
                    centerZ 0.1;
                    width 0.2;
                    radius 0.5;
                    axis x;
                }
            */

            scalar centerX = readScalar( geoDict["centerX"] );
            scalar centerY = readScalar( geoDict["centerY"] );
            scalar centerZ = readScalar( geoDict["centerZ"] );
            scalar width   = readScalar( geoDict["width"] );
            scalar radius  = readScalar( geoDict["radius"] );
            word   axis    = word(geoDict["axis"]);

            scalar aMax=0.0,aMin=0.0;
            if (axis=="x")
            {
                aMax=centerX+width/2.0;
                aMin=centerX-width/2.0;
            }
            else if (axis=="y")
            {
                aMax=centerY+width/2.0;
                aMin=centerY-width/2.0;
            }
            else if (axis=="z")
            {
                aMax=centerZ+width/2.0;
                aMin=centerZ-width/2.0;
            }
            else
            {
                FatalErrorIn("")<<"axis not valid"<<abort(FatalError);
            }
            
            labelList cellList;
            
            forAll(mesh_.C(),cellI)
            {
                scalar cX=mesh_.C()[cellI].x();
                scalar cY=mesh_.C()[cellI].y();
                scalar cZ=mesh_.C()[cellI].z();
                scalar aCell=0.0,rCell=0.0;
                if (axis=="x")
                {
                    aCell=cX;
                    scalar comp1=(cY-centerY)*(cY-centerY);
                    scalar comp2=(cZ-centerZ)*(cZ-centerZ);
                    rCell=Foam::pow(comp1+comp2,0.5);
                }
                else if (axis=="y")
                {
                    aCell=cY;
                    scalar comp1=(cX-centerX)*(cX-centerX);
                    scalar comp2=(cZ-centerZ)*(cZ-centerZ);
                    rCell=Foam::pow(comp1+comp2,0.5);
                }
                else if (axis=="z")
                {
                    aCell=cZ;
                    scalar comp1=(cY-centerY)*(cY-centerY);
                    scalar comp2=(cX-centerX)*(cX-centerX);
                    rCell=Foam::pow(comp1+comp2,0.5);
                }
                else
                {
                    FatalErrorIn("")<<"axis not valid"<<abort(FatalError);
                }
                
                if( aCell >= aMin and aCell <= aMax and rCell <= radius  )
                {
                    cellList.append(cellI);
                    //Info<<"cX "<<cX<<" cY "<<cY<<" cZ "<<cZ<<" cellI "<<cellI<<endl;
                }

            }

            cellIdx4UserDefinedVolumes.set(geoName,cellList);

        }
        else if (geoType=="annulus") // it is an annulus
        {
            /*
                example:
                userDefinedVolume0
                {
                    type annulus;
                    stateName U;
                    component 0;
                    scale 1.0;
                    centerX 0.1;
                    centerY 0.1;
                    centerZ 0.1;
                    width 0.2;
                    radiusInner 0.1;
                    radiusOuter 0.5;
                    axis x;
                }
            */

            scalar centerX = readScalar( geoDict["centerX"] );
            scalar centerY = readScalar( geoDict["centerY"] );
            scalar centerZ = readScalar( geoDict["centerZ"] );
            scalar width   = readScalar( geoDict["width"] );
            scalar radius1 = readScalar( geoDict["radiusInner"] );
            scalar radius2 = readScalar( geoDict["radiusOuter"] );
            word axis      = word(geoDict["axis"]);

            scalar aMax=0.0,aMin=0.0;
            if (axis=="x")
            {
                aMax=centerX+width/2.0;
                aMin=centerX-width/2.0;
            }
            else if (axis=="y")
            {
                aMax=centerY+width/2.0;
                aMin=centerY-width/2.0;
            }
            else if (axis=="z")
            {
                aMax=centerZ+width/2.0;
                aMin=centerZ-width/2.0;
            }
            else
            {
                FatalErrorIn("")<<"axis not valid"<<abort(FatalError);
            }
            
            labelList cellList;
            
            forAll(mesh_.C(),cellI)
            {
                scalar cX=mesh_.C()[cellI].x();
                scalar cY=mesh_.C()[cellI].y();
                scalar cZ=mesh_.C()[cellI].z();
                scalar aCell=0.0,rCell=0.0;
                if (axis=="x")
                {
                    aCell=cX;
                    scalar comp1=(cY-centerY)*(cY-centerY);
                    scalar comp2=(cZ-centerZ)*(cZ-centerZ);
                    rCell=Foam::pow(comp1+comp2,0.5);
                }
                else if (axis=="y")
                {
                    aCell=cY;
                    scalar comp1=(cX-centerX)*(cX-centerX);
                    scalar comp2=(cZ-centerZ)*(cZ-centerZ);
                    rCell=Foam::pow(comp1+comp2,0.5);
                }
                else if (axis=="z")
                {
                    aCell=cZ;
                    scalar comp1=(cY-centerY)*(cY-centerY);
                    scalar comp2=(cX-centerX)*(cX-centerX);
                    rCell=Foam::pow(comp1+comp2,0.5);
                }
                else
                {
                    FatalErrorIn("")<<"axis not valid"<<abort(FatalError);
                }
                
                if( aCell >= aMin and aCell <= aMax and rCell <= radius2 and rCell >= radius1  )
                {
                    cellList.append(cellI);
                    //Info<<"cX "<<cX<<" cY "<<cY<<" cZ "<<cZ<<" cellI "<<cellI<<endl;
                }

            }    
            
            cellIdx4UserDefinedVolumes.set(geoName,cellList);
        }
        else
        {
            FatalErrorIn("")<<"geoType not supported"<<abort(FatalError);
        }
        
    }

    //output the cell information into userDefinedVolume0.info
    forAll(cellIdx4UserDefinedVolumes.toc(),idxI)
    {
        word key=cellIdx4UserDefinedVolumes.toc()[idxI];
        if (cellIdx4UserDefinedVolumes[key].size()!=0)
        {
            label myProc = Pstream::myProcNo();
            label nProcs = Pstream::nProcs();
            std::ostringstream fileNameStream("");
            fileNameStream<<key<<"_"<<myProc<<"_of_"<<nProcs<<".info";
            word fileName = fileNameStream.str();
            OFstream aOut(fileName);
            aOut.precision(8);
            forAll(cellIdx4UserDefinedVolumes[key],idxJ)
            {
                label cellI=cellIdx4UserDefinedVolumes[key][idxJ];
                scalar cX=mesh_.C()[cellI].x();
                scalar cY=mesh_.C()[cellI].y();
                scalar cZ=mesh_.C()[cellI].z();
                aOut<<cX<<" "<<cY<<" "<<cZ<<endl;
            }
        }
    }
}

void AdjointIndexing::calcFaceIndx4UserDefinedPatches()
{
    // basically, we add the face index that is inside the rectangular domain 
    // defined in userDefinedPatchInfo, and set it to faceIndx4UserDefinedPatches

    // first check if the names are valid
    forAll(adjIO_.userDefinedPatchInfo.toc(),idxI)
    {
        word geoName=adjIO_.userDefinedPatchInfo.toc()[idxI];
        if (!isUserDefinedPatch(geoName))
        {
            Info<<geoName<<endl;
            FatalErrorIn("")<<"valid userDefinedPatch format is userDefinedPatch0, userDefinedPatch1, ..."<<abort(FatalError);
        }
    }
    
    forAll(adjIO_.userDefinedPatchInfo.toc(),idxI)
    {
        word patchName = adjIO_.userDefinedPatchInfo.toc()[idxI];

        dictionary patchDict = adjIO_.userDefinedPatchInfo.subDict(patchName);

        word patchType = word(patchDict["type"]);

        if( patchType=="box")
        {
            /*
                example:
                userDefinedPatch0
                {
                    type box;
                    stateName U;
                    component 0;
                    scale 1.0;
                    centerX 0.1;
                    centerY 0.1;
                    centerZ 0.1;
                    sizeX 0.1;
                    sizeY 0.1;
                    sizeZ 0.1;
                }
            */
            // box patch domain
            scalar centerX = readScalar( patchDict["centerX"] );
            scalar centerY = readScalar( patchDict["centerY"] );
            scalar centerZ = readScalar( patchDict["centerZ"] );
            scalar sizeX =   readScalar( patchDict["sizeX"] );
            scalar sizeY =   readScalar( patchDict["sizeY"] );
            scalar sizeZ =   readScalar( patchDict["sizeZ"] );
            
            scalar xMax=centerX+sizeX/2.0;
            scalar xMin=centerX-sizeX/2.0;
            scalar yMax=centerY+sizeY/2.0;
            scalar yMin=centerY-sizeY/2.0;
            scalar zMax=centerZ+sizeZ/2.0;
            scalar zMin=centerZ-sizeZ/2.0;
            
            labelList faceList;
            
            forAll(mesh_.Cf(),faceI)
            {
                scalar cfX=mesh_.Cf()[faceI].x();
                scalar cfY=mesh_.Cf()[faceI].y();
                scalar cfZ=mesh_.Cf()[faceI].z();
                
                if( cfX >= xMin and cfX <= xMax and   
                    cfY >= yMin and cfY <= yMax and
                    cfZ >= zMin and cfZ <= zMax )
                {
                    faceList.append(faceI);
                }
                
            }
            // for decomposed domains, the userDefinedPatch may be a boundary patch
            forAll(mesh_.Cf().boundaryField(),patchI)
            {
                forAll(mesh_.Cf().boundaryField()[patchI],faceI)
                {
                    scalar cfX=mesh_.Cf().boundaryField()[patchI][faceI].x();
                    scalar cfY=mesh_.Cf().boundaryField()[patchI][faceI].y();
                    scalar cfZ=mesh_.Cf().boundaryField()[patchI][faceI].z();
                    
                    if( cfX >= xMin and cfX <= xMax and   
                        cfY >= yMin and cfY <= yMax and
                        cfZ >= zMin and cfZ <= zMax )
                    {
                        label tmp=BFacePatchIFaceI2LocalIndex(patchI,faceI);
                        faceList.append(tmp);
                    }
                }
            }
            
            faceIdx4UserDefinedPatches.set(patchName,faceList);
        }
        else if( patchType=="patch" )
        {
            /*
                example:
                userDefinedPatch0
                {
                    type patch;
                    patchName inlet;
                    stateName U;
                    component 0;
                    scale 1.0;
                }
            */
            word userDefinedPatchName = word(patchDict["patchName"]);
            label patchI = mesh_.boundaryMesh().findPatchID( userDefinedPatchName );
            labelList faceList;
            forAll(mesh_.boundaryMesh()[patchI],faceI)
            {
                 label bFaceI = BFacePatchIFaceI2LocalIndex(patchI,faceI);
                 faceList.append(bFaceI);
            }
            faceIdx4UserDefinedPatches.set(patchName,faceList);
        }
        else
        {
            FatalErrorIn("")<<"patch type not supported"<<abort(FatalError);
        }
    }

    //output the face information into userDefinedPatch0.info
    forAll(faceIdx4UserDefinedPatches.toc(),idxI)
    {
        word key=faceIdx4UserDefinedPatches.toc()[idxI];
        if (faceIdx4UserDefinedPatches[key].size()!=0)
        {
            label myProc = Pstream::myProcNo();
            label nProcs = Pstream::nProcs();
            std::ostringstream fileNameStream("");
            fileNameStream<<key<<"_"<<myProc<<"_of_"<<nProcs<<".info";
            word fileName = fileNameStream.str();
            OFstream aOut(fileName);
            aOut.precision(8);
            forAll(faceIdx4UserDefinedPatches[key],idxJ)
            {
                label faceI=faceIdx4UserDefinedPatches[key][idxJ];
                if (faceI<nLocalInternalFaces)
                {
                    scalar cX=mesh_.Cf()[faceI].x();
                    scalar cY=mesh_.Cf()[faceI].y();
                    scalar cZ=mesh_.Cf()[faceI].z();
                    aOut<<cX<<" "<<cY<<" "<<cZ<<endl;
                }
                else
                {
                    label relIdx=faceI-nLocalInternalFaces;
                    label patchIdx=bFacePatchI[relIdx];
                    label faceIdx=bFaceFaceI[relIdx];
                    scalar cX=mesh_.Cf().boundaryField()[patchIdx][faceIdx].x();
                    scalar cY=mesh_.Cf().boundaryField()[patchIdx][faceIdx].y();
                    scalar cZ=mesh_.Cf().boundaryField()[patchIdx][faceIdx].z();
                    aOut<<cX<<" "<<cY<<" "<<cZ<<endl;
                }
                
            }
        }
    }

    return ;
}

// ************************************************************************* //

} // End namespace Foam
