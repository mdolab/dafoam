/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointNewtonKrylov.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
AdjointNewtonKrylov::AdjointNewtonKrylov
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointObjectiveFunction& adjObj,
    AdjointDerivative& adjDev
)  
    :
    mesh_(mesh),
    adjIO_(adjIO),
    adjReg_(adjReg),
    adjRAS_(adjRAS),
    adjObj_(adjObj),
    adjDev_(adjDev),
    db_(mesh.thisDb())
{
    // initialize stuff
    nFuncEvals_ = 0;
    this->initializeIndexing();
    this->initializePetscVectors();
    
}

//  ---------------------------------------
//    member functions
//  ---------------------------------------
globalIndex AdjointNewtonKrylov::genGlobalIndex(const label index)
{

    globalIndex result(index);
    return result;
    
}

void AdjointNewtonKrylov::initializeIndexing()
{
    // this is essentially a copy of adjointIndexing functions except that we may ignore phi and turb vars

    // initialize nkStateType and stateNames
    forAll(adjReg_.volVectorStates,idxI) 
    {
        nkStateNames_.append( adjReg_.volVectorStates[idxI] );
        nkStateType_.set( adjReg_.volVectorStates[idxI], "volVectorState" );
    }
    forAll(adjReg_.volScalarStates,idxI) 
    {
        nkStateNames_.append( adjReg_.volScalarStates[idxI] );
        nkStateType_.set( adjReg_.volScalarStates[idxI], "volScalarState" );
    }
    if(!adjIO_.nkSegregatedTurb)
    {
        forAll(adjRAS_.turbStates,idxI) 
        {
            nkStateNames_.append( adjRAS_.turbStates[idxI] );
            nkStateType_.set( adjRAS_.turbStates[idxI], "turbState" );
        }
    }
    if(!adjIO_.nkSegregatedPhi)
    {
        forAll(adjReg_.surfaceScalarStates,idxI) 
        {
            nkStateNames_.append( adjReg_.surfaceScalarStates[idxI] );
            nkStateType_.set( adjReg_.surfaceScalarStates[idxI], "surfaceScalarState" );
        }
    }

    // initialize cell sizes
    label nLocalCells = mesh_.nCells();
    label nLocalFaces = mesh_.nFaces();
    label nLocalInternalFaces = mesh_.nInternalFaces();
    label nLocalBoundaryFaces = nLocalFaces - nLocalInternalFaces;

    // get bFacePatchI and bFaceFaceI
    // these two lists store the patchI and faceI for a given boundary mesh face index
    // the index of these lists starts from the first boundary face of the first boundary patch.
    // they will be used to quickly get the patchI and faceI from a given boundary face index
    bFacePatchI_.setSize(nLocalBoundaryFaces);
    bFaceFaceI_.setSize(nLocalBoundaryFaces);
    label tmpCounter=0;
    forAll(mesh_.boundaryMesh(),patchI)
    {
        forAll(mesh_.boundaryMesh()[patchI],faceI)
        {
            bFacePatchI_[tmpCounter] = patchI;
            bFaceFaceI_[tmpCounter] = faceI;
            tmpCounter++;
        }
    }

    // initialize nkStateLocalIndexOffset
    // NOTE: for NK we use only state-by-state ordering
    forAll(nkStateNames_,idxI)
    {
        word& stateName = nkStateNames_[idxI];
        
        label counter=0;
        
        forAll(adjReg_.volVectorStates,idx)
        {
            if (adjReg_.volVectorStates[idx] == stateName)
            {
                nkStateLocalIndexOffset_.set(stateName,counter*nLocalCells);
            }
            counter+=3;
        }
        
        forAll(adjReg_.volScalarStates,idx)
        {
            if (adjReg_.volScalarStates[idx] == stateName)
            {
                nkStateLocalIndexOffset_.set(stateName,counter*nLocalCells);
            }
            counter++;
        }
        
        if(!adjIO_.nkSegregatedTurb)
        {
            forAll(adjRAS_.turbStates,idx)
            {
                if (adjRAS_.turbStates[idx] == stateName)
                {
                    nkStateLocalIndexOffset_.set(stateName,counter*nLocalCells);
                }
                counter++;
            }
        }

        if(!adjIO_.nkSegregatedPhi)
        {
            forAll(adjReg_.surfaceScalarStates,idx)
            {
                if (adjReg_.surfaceScalarStates[idx] == stateName && idx==0)
                {
                    nkStateLocalIndexOffset_.set(stateName,counter*nLocalCells);
                }
                if (adjReg_.surfaceScalarStates[idx] == stateName && idx>0)
                {
                    nkStateLocalIndexOffset_.set(stateName,counter*nLocalFaces);
                }
                counter++;
            }
        }
        
    }

    // initialize nk sizes
    label nVolScalarStates = adjReg_.volScalarStates.size();
    label nVolVectorStates = adjReg_.volVectorStates.size();
    label nSurfaceScalarStates = adjReg_.surfaceScalarStates.size();
    if (adjIO_.nkSegregatedPhi) nSurfaceScalarStates=0;
    label nTurbStates = adjRAS_.turbStates.size();
    if (adjIO_.nkSegregatedTurb) nTurbStates=0;

    // we can now calculate adjoint state size
    nLocalNKStates_ = (nVolVectorStates*3+nVolScalarStates+nTurbStates)*nLocalCells + nSurfaceScalarStates*nLocalFaces;


    // Setup the global numbering to convert a local index to the associated global index
    globalNKStateNumbering_ = this->genGlobalIndex(nLocalNKStates_);

    // Initialize indexing lists: 
    // cellIFaceI4LocalNKIdx
    // stateName4LocalNKIdx
    // cellIFaceI4LocalNKIdx stores the cell index for a local NK index
    // For vector fields, the decima of cellIFaceI4LocalNKIdx denotes the vector component
    // e.g., 10.1 means cellI=10, y compnent of U
    // stateName4LocalNKIdx stores the state name for a local NK index
    cellIFaceI4LocalNKIdx_.setSize(nLocalNKStates_);
    stateName4LocalNKIdx_.setSize(nLocalNKStates_);
    forAll(adjReg_.volVectorStates,idx)
    {
        word stateName = adjReg_.volVectorStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            for(label i=0;i<3;i++)
            {               
                label localIdx = this->getLocalNKStateIndex(stateName,cellI,i);
                cellIFaceI4LocalNKIdx_[localIdx] = cellI+i/10.0;
                stateName4LocalNKIdx_[localIdx] = stateName;
            }
            
        }
    }
    forAll(adjReg_.volScalarStates,idx)
    {
        word stateName = adjReg_.volScalarStates[idx];
        forAll(mesh_.cells(),cellI)
        {
            label localIdx = this->getLocalNKStateIndex(stateName,cellI);
            cellIFaceI4LocalNKIdx_[localIdx] = cellI;
            stateName4LocalNKIdx_[localIdx] = stateName;
        }
    }
    if (!adjIO_.nkSegregatedTurb)
    {
        forAll(adjRAS_.turbStates,idx)
        {
            word stateName = adjRAS_.turbStates[idx];
            forAll(mesh_.cells(),cellI)
            {            
                label localIdx = this->getLocalNKStateIndex(stateName,cellI);
                cellIFaceI4LocalNKIdx_[localIdx] = cellI;
                stateName4LocalNKIdx_[localIdx] = stateName;
            }
        }
    }
    if (!adjIO_.nkSegregatedPhi)
    {
        forAll(adjReg_.surfaceScalarStates,idx)
        {
            word stateName = adjReg_.surfaceScalarStates[idx];
            forAll(mesh_.faces(),faceI)
            {            
                label localIdx = this->getLocalNKStateIndex(stateName,faceI);
                cellIFaceI4LocalNKIdx_[localIdx] = faceI;
                stateName4LocalNKIdx_[localIdx] = stateName;
            }
        }
    }
}           


void AdjointNewtonKrylov::initializePetscVectors()
{
    label localSize = nLocalNKStates_;
    
    VecCreate(PETSC_COMM_WORLD,&rVec0_);
    VecSetSizes(rVec0_,localSize,PETSC_DETERMINE);
    VecSetFromOptions(rVec0_);
    VecDuplicate(rVec0_,&rVec1_);
    VecDuplicate(rVec0_,&wVec0_);
    VecDuplicate(rVec0_,&wVec1_);
    VecDuplicate(rVec0_,&dWVec_);

    VecZeroEntries(rVec0_);
    VecZeroEntries(rVec1_);
    VecZeroEntries(wVec0_);
    VecZeroEntries(wVec1_);
    VecZeroEntries(dWVec_);

    if(adjIO_.nkSegregatedPhi)
    {
        label phiSize=adjReg_.surfaceScalarStates.size()*mesh_.nFaces();
        VecCreate(PETSC_COMM_WORLD,&phiVec0_);
        VecSetSizes(phiVec0_,phiSize,PETSC_DETERMINE);
        VecSetFromOptions(phiVec0_);

        VecDuplicate(phiVec0_,&dPhiVec_);

        VecZeroEntries(phiVec0_);
        VecZeroEntries(dPhiVec_);
    }

    if(adjIO_.nkPseudoTransient)
    {
        VecDuplicate(rVec0_,&dTInv_);
        VecZeroEntries(dTInv_);
    }

}

label AdjointNewtonKrylov::getLocalNKStateIndex
(
    const word stateName,
    const label idxJ,
    label comp
)
{
    // NOTE: for volVectorState, one need to set comp; while for other states, comp is simply ignored in this function
    
    forAll(nkStateNames_,idxI)
    {
        if (nkStateNames_[idxI] == stateName)
        {
            if(nkStateType_[stateName] == "volVectorState")
            {
                if (comp==-1) 
                {
                    FatalErrorIn("")<<"comp needs to be set for vector states!"
                                    <<abort(FatalError);
                }
                else
                {
                    return nkStateLocalIndexOffset_[stateName]+idxJ*3+comp;
                }
            }
            else
            {
                return nkStateLocalIndexOffset_[stateName]+idxJ;
            }
        }
    }

    
    // if no stateName found, return an error
    FatalErrorIn("")<< "stateName not found!"<<abort(FatalError);
    return -1;
}

label AdjointNewtonKrylov::getGlobalNKStateIndex
(
    const word stateName,
    const label idxI,
    label comp
)
{
    // For vector, one need to provide comp, for scalar, comp is not needed.
    label localIdx = this->getLocalNKStateIndex(stateName,idxI,comp);
    label globalIdx = globalNKStateNumbering_.toGlobal(localIdx);
    return globalIdx;
}

void AdjointNewtonKrylov::calcdRdWPCFast
(
    Mat jacMat
)
{
    // TODO: need to check if all the fvMatrix include boundary effect!

    const PetscScalar *dTInvArray;
    if(adjIO_.nkPseudoTransient) VecGetArrayRead(dTInv_,&dTInvArray);

    label isRef=0,isPC=1;
    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();

    forAll(adjReg_.volVectorStates,idxI)
    {
        
        const word stateName = adjReg_.volVectorStates[idxI];
        const word resName = stateName+"Res";
        const word fvMatrixName = stateName+"Eqn";
        scalar resScaling=1.0;
        scalar stateScaling= adjDev_.getStateScaling(stateName);
        
        adjDev_.updateStateVariableBCs();
        adjDev_.calcResiduals(isRef,isPC,fvMatrixName);
        nFuncEvals_++;

        // set diag
        for(label cellI=0;cellI<mesh_.nCells();cellI++)
        {
            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[cellI];
            }    

            for(label i=0;i<3;i++)
            {
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,cellI,i);
                PetscInt colI = rowI;
                PetscScalar val=adjDev_.fvMatrixDiag[cellI]/resScaling*stateScaling;
                if(adjIO_.nkPseudoTransient) 
                {
                    label localIdx = this->getLocalNKStateIndex(stateName,cellI,i);
                    val += dTInvArray[localIdx];  
                }
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }

        // set lower/owner
        for(label faceI=0;faceI<mesh_.nInternalFaces();faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[neighbourCellI];
            } 

            for(label i=0;i<3;i++)
            {
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,neighbourCellI,i);
                PetscInt colI = this->getGlobalNKStateIndex(stateName,ownerCellI,i);
                PetscScalar val=adjDev_.fvMatrixLower[faceI]/resScaling*stateScaling;
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }

        // set upper/neighbour
        for(label faceI=0;faceI<mesh_.nInternalFaces();faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[ownerCellI];
            } 

            for(label i=0;i<3;i++)
            {
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,ownerCellI,i);
                PetscInt colI = this->getGlobalNKStateIndex(stateName,neighbourCellI,i);
                PetscScalar val=adjDev_.fvMatrixUpper[faceI]/resScaling*stateScaling;
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }
    }

    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI];
        const word resName = stateName+"Res";
        const word fvMatrixName = stateName+"Eqn";
        scalar resScaling=1.0;
        scalar stateScaling= adjDev_.getStateScaling(stateName);
        
        adjDev_.updateStateVariableBCs();
        adjDev_.calcResiduals(isRef,isPC,fvMatrixName);
        nFuncEvals_++;

        // set diag
        for(label cellI=0;cellI<mesh_.nCells();cellI++)
        {
            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[cellI];
            } 

            PetscInt rowI = this->getGlobalNKStateIndex(stateName,cellI);
            PetscInt colI = rowI;
            PetscScalar val=adjDev_.fvMatrixDiag[cellI]/resScaling*stateScaling;
            if(adjIO_.nkPseudoTransient) 
            {
                label localIdx = this->getLocalNKStateIndex(stateName,cellI);
                val += dTInvArray[localIdx];  
            }
            MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
        }

        // set lower/owner
        for(label faceI=0;faceI<mesh_.nInternalFaces();faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[neighbourCellI];
            } 

            PetscInt rowI = this->getGlobalNKStateIndex(stateName,neighbourCellI);
            PetscInt colI = this->getGlobalNKStateIndex(stateName,ownerCellI);
            PetscScalar val=adjDev_.fvMatrixLower[faceI]/resScaling*stateScaling;
            MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
        }

        // set upper/neighbour
        for(label faceI=0;faceI<mesh_.nInternalFaces();faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
            {
                resScaling = mesh_.V()[ownerCellI];
            } 

            PetscInt rowI = this->getGlobalNKStateIndex(stateName,ownerCellI);
            PetscInt colI = this->getGlobalNKStateIndex(stateName,neighbourCellI);
            PetscScalar val=adjDev_.fvMatrixUpper[faceI]/resScaling*stateScaling;
            MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
        }
    }

    if(!adjIO_.nkSegregatedTurb)
    {
        forAll(adjRAS_.turbStates,idxI)
        {
            const word stateName = adjRAS_.turbStates[idxI];
            const word resName = stateName+"Res";
            const word fvMatrixName = stateName+"Eqn";
            scalar resScaling=1.0;
            scalar stateScaling= adjDev_.getStateScaling(stateName);
            
            adjRAS_.correctTurbBoundaryConditions();
            adjRAS_.calcTurbResiduals(isRef,isPC,fvMatrixName);
            nFuncEvals_++;
    
            // set diag
            for(label cellI=0;cellI<mesh_.nCells();cellI++)
            {
    
                if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
                {
                    resScaling = mesh_.V()[cellI];
                } 
    
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,cellI);
                PetscInt colI = rowI;
                PetscScalar val=adjRAS_.fvMatrixDiag[cellI]/resScaling*stateScaling;
                if(adjIO_.nkPseudoTransient) 
                {
                    label localIdx = this->getLocalNKStateIndex(stateName,cellI);
                    val += dTInvArray[localIdx];  
                }
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
    
            // set lower/owner
            for(label faceI=0;faceI<mesh_.nInternalFaces();faceI++)
            {
                label ownerCellI = owner[faceI];
                label neighbourCellI = neighbour[faceI];
    
                if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
                {
                    resScaling = mesh_.V()[neighbourCellI];
                } 
    
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,neighbourCellI);
                PetscInt colI = this->getGlobalNKStateIndex(stateName,ownerCellI);
                PetscScalar val=adjRAS_.fvMatrixLower[faceI]/resScaling*stateScaling;
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
    
            // set upper/neighbour
            for(label faceI=0;faceI<mesh_.nInternalFaces();faceI++)
            {
                label ownerCellI = owner[faceI];
                label neighbourCellI = neighbour[faceI];
    
                if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
                {
                    resScaling = mesh_.V()[ownerCellI];
                } 
    
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,ownerCellI);
                PetscInt colI = this->getGlobalNKStateIndex(stateName,neighbourCellI);
                PetscScalar val=adjRAS_.fvMatrixUpper[faceI]/resScaling*stateScaling;
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }
    }

    if(!adjIO_.nkSegregatedPhi)
    {
        forAll(adjReg_.surfaceScalarStates,idxI)
        {
            const word stateName = adjReg_.surfaceScalarStates[idxI];
            const word resName = stateName+"Res";
            scalar resScaling=1.0,stateScaling=1.0;
    
            // for phi, we keep only the diagonal component
            for(label faceI=0;faceI<mesh_.nFaces();faceI++)
            {
                if (adjIO_.isInList<word>(resName,adjIO_.normalizeResiduals)) 
                {
                    if (faceI<mesh_.nInternalFaces())
                    {
                        resScaling = mesh_.magSf()[faceI];
                    }
                    else
                    {
                        label relIdx=faceI-mesh_.nInternalFaces();
                        label patchIdx=bFacePatchI_[relIdx];
                        label faceIdx=bFaceFaceI_[relIdx];
                        resScaling = mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                    }
                } 
    
                if (adjIO_.isInList<word>(stateName,adjIO_.normalizeStates)) 
                {
                    stateScaling = adjDev_.getStateScaling(stateName,faceI);
                } 
    
                PetscInt rowI = this->getGlobalNKStateIndex(stateName,faceI);
                PetscInt colI = rowI;
                PetscScalar val=-1.0/resScaling*stateScaling;
                if(adjIO_.nkPseudoTransient) 
                {
                    label localIdx = this->getLocalNKStateIndex(stateName,faceI);
                    val += dTInvArray[localIdx];  
                }
                MatSetValues(jacMat,1,&rowI,1,&colI,&val,INSERT_VALUES);
            }
        }
    }

    MatAssemblyBegin(jacMat,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat,MAT_FINAL_ASSEMBLY);

    if(adjIO_.nkPseudoTransient) VecRestoreArrayRead(dTInv_,&dTInvArray);
    
    if(adjIO_.writeMatrices)
    {
        adjIO_.writeMatrixBinary(jacMat,"fastPCMat");
    }
}
    

void AdjointNewtonKrylov::calcPCMat4NK(Mat* PCMat,label initMat)
{

    
    // initialize and calculate PCMat
    if (initMat)
    {
        label localSize = nLocalNKStates_;
        MatCreate(PETSC_COMM_WORLD,PCMat);
        MatSetSizes(*PCMat,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE);
        MatSetFromOptions(*PCMat);
        MatMPIAIJSetPreallocation(*PCMat,20,NULL,20,NULL);
        MatSeqAIJSetPreallocation(*PCMat,20,NULL);
        MatSetUp(*PCMat);
    }
    this->calcdRdWPCFast(*PCMat);
    
}

void AdjointNewtonKrylov::setNormalizeStatesScaling2Vec(Vec vecY)
{

    PetscScalar *vecYArray;
    VecGetArray(vecY,&vecYArray);
    
    forAll(adjReg_.volVectorStates,idxI)                                           
    {        
        const word stateName = adjReg_.volVectorStates[idxI];    
        scalar scalingFactor = adjDev_.getStateScaling(stateName); 
        if(scalingFactor==1.0) continue;

        forAll(mesh_.cells(),cellI)                                            
        {                                                                      
            for(label i=0;i<3;i++)                                         
            {                                                                  
                label localIdx = this->getLocalNKStateIndex(stateName,cellI,i);
                PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor; 
                vecYArray[localIdx] = scaledVal;        
            }                                                                  
        }                                                                   
    }
    
    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI]; 
        scalar scalingFactor = adjDev_.getStateScaling(stateName);
        if(scalingFactor==1.0) continue;

        forAll(mesh_.cells(),cellI)
        {         
            label localIdx = this->getLocalNKStateIndex(stateName,cellI);    
            PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor; 
            vecYArray[localIdx] = scaledVal;          
        }
         
    }
    
    if(!adjIO_.nkSegregatedTurb)
    {
        forAll(adjRAS_.turbStates,idxI)
        {
            const word stateName = adjRAS_.turbStates[idxI]; 
            scalar scalingFactor = adjDev_.getStateScaling(stateName);
            if(scalingFactor==1.0) continue;
    
            forAll(mesh_.cells(),cellI)
            {         
                label localIdx = this->getLocalNKStateIndex(stateName,cellI);   
                PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor;
                vecYArray[localIdx] = scaledVal;        
            }
        
        }
    }

    if(!adjIO_.nkSegregatedPhi)
    {
        forAll(adjReg_.surfaceScalarStates,idxI)
        {
            const word stateName = adjReg_.surfaceScalarStates[idxI]; 
        
            forAll(mesh_.faces(),faceI)
            {
                label localIdx = this->getLocalNKStateIndex(stateName,faceI); 
                scalar scalingFactor = adjDev_.getStateScaling(stateName,faceI);
                if(scalingFactor!=1.0)
                {
                    PetscScalar scaledVal = vecYArray[localIdx]*scalingFactor;
                    vecYArray[localIdx] = scaledVal;
                }     
            } 
        }
    }
        
    VecRestoreArray(vecY,&vecYArray);

    return;

}

scalar AdjointNewtonKrylov::getEWTol
(
    scalar norm, 
    scalar oldNorm, 
    scalar rTolLast
)
{
    // There are the default EW Parameters from PETSc. They seem to work well
    // version:  2
    // rTolLast: 0.1
    // rTolMax:  0.9
    // gamma:    1.0
    // alpha:    1.61803398874989
    // threshold: 0.1

    scalar rTolMax   = adjIO_.nkEWRTolMax;
    scalar gamma     = 1.0;
    scalar alpha     = (1.0+Foam::sqrt(5.0))/2.0;
    scalar threshold = 0.10;
    // We use version 2:
    scalar rTol = gamma*Foam::pow(norm/oldNorm,alpha);
    scalar sTol = gamma*Foam::pow(rTolLast,alpha);

    if (sTol > threshold)
    {
       rTol = max(rTol, sTol);
    }

    // Safeguard: avoid rtol greater than one
    rTol = min(rTol, rTolMax);

    return rTol;
}

void AdjointNewtonKrylov::NKCalcResiduals(Vec wVec,Vec rVec)
{
    // Given an input vec wVec, calculate the function vec rWec
    // wVec: vector storing all the states
    // rVec: vector storing the residuals
    
    this->NKSetVecs(wVec,"Vec2Var",1.0,"");

    label isRef=0,isPC=0;
    adjDev_.updateStateVariableBCs();
    adjDev_.calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);

    this->NKSetVecs(rVec,"Var2Vec",1.0,"Res");

    return;

}

void AdjointNewtonKrylov::NKCalcResidualsUnsteady
(
    Vec wVec,
    Vec dWVec, 
    scalar step,
    Vec rVec
)
{
    // Given an input vec wVec, calculate the function vec rWec
    // wVec: vector storing all the states
    // rVec: vector storing the residuals
    
    this->NKSetVecs(wVec,"Vec2Var",1.0,"");

    label isRef=0,isPC=0;
    adjDev_.updateStateVariableBCs();
    adjDev_.calcResiduals(isRef,isPC);
    adjRAS_.calcTurbResiduals(isRef,isPC);

    this->NKSetVecs(rVec,"Var2Vec",1.0,"Res");

    Vec unsteadyVec;
    VecDuplicate(dWVec,&unsteadyVec);
    VecCopy(dWVec,unsteadyVec);
    VecScale(unsteadyVec,step);

    this->NKAddUnsteadyTerms(unsteadyVec,rVec);

    return;

}


scalar AdjointNewtonKrylov::NKLineSearchNew
(
    const Vec wVec,
    const Vec rVec,
    const Vec dWVec,
    Vec wVecNew,
    Vec rVecNew
)
{
    // basic non monotonic line search using backtracking 
    
    scalar gamma = 0.999; // line search sufficient decrease coefficient
    scalar sigma = 0.5;    // line search reduce factor
    scalar alpha = 1.0;      // initial step size

    PetscErrorCode ierr;

    // compute the norms
    scalar rVecNorm=0.0, rVecNewNorm=0.0;
    VecNorm(rVec,NORM_2,&rVecNorm);

    // initial step
    alpha=1.0;

    // actual backtracking
    for(label i=0;i<10;i++)
    {
        // compute new w value wNew = w-dW Note: dW is the solution from KSP it is dW=wOld-wNew
        VecWAXPY(wVecNew,-alpha,dWVec,wVec);

        // using the current state to compute rVecNew
        if(adjIO_.nkPseudoTransient) this->NKCalcResidualsUnsteady(wVecNew,dWVec,-alpha,rVecNew);
        else this->NKCalcResiduals(wVecNew,rVecNew);
        nFuncEvals_++;

        // compute the rVecNorm at new w
        ierr=VecNorm(rVecNew,NORM_2,&rVecNewNorm);

        if(ierr == PETSC_ERR_FP)
        {
            // floating point seg fault, just reduce the step size
            alpha = alpha*sigma;
            Info<<"Seg Fault in line search"<<endl;
            continue;
        }
        else if(this->checkNegativeTurb())
        {
            // have negative turbulence, reduce the step size
            alpha = alpha*sigma;
            Info<<"Negative turbulence in line search"<<endl;
            continue;
        }
        else if (rVecNewNorm <= rVecNorm*gamma)  // Sufficient reduction
        {
            return alpha;
        }
        else // reduce step
        {
            alpha = alpha * sigma;
        }

    }

    return alpha;

}

scalar AdjointNewtonKrylov::NKLineSearch
(
    const Mat dRdW,
    const Vec wVec,
    const Vec rVec,
    const Vec dWVec,
    Vec wVecNew,
    Vec rVecNew
)
{
    // basic non monotonic line search using backtracking 

    // Note that for line search purposes we work with with the related
    // minimization problem:
    //    min  z(w):  R^n -> R,
    // where z(w) = .5 * rVecNorm*rVecNorm, and rVecNorm = || rVec ||_2.
    
    scalar gamma = 1.0e-3; // line search sufficient decrease coefficient
    scalar sigma = 0.5;    // line search reduce factor
    scalar initSlope=0.0;
    scalar alpha=1.0;      // initial step size
    scalar maxVal = 0.0;

    PetscErrorCode ierr;

    // compute the norms
    scalar rVecNorm=0.0, rVecNewNorm=0.0;
    VecNorm(rVec,NORM_2,&rVecNorm);

    // current z(w)
    scalar zW0=0.5*rVecNorm*rVecNorm;

    // compute dRdW*dW and save it to wVecNew
    MatMult(dRdW,dWVec,wVecNew);
    nFuncEvals_++;

    // compute dRdW*dW*rVec as the initSlope for Wolf condition
    VecDot(rVec,wVecNew,&initSlope);

    // make sure this slope is always negative.
    if (initSlope > 0.0) initSlope = -initSlope;
    if (initSlope == 0.0) initSlope = -1.0;

    // initial step
    alpha=1.0;

    // actual backtracking
    for(label i=0;i<10;i++)
    {
        // compute new w value wNew = w-dW Note: dW is the solution from KSP it is dW=wOld-wNew
        VecWAXPY(wVecNew,-alpha,dWVec,wVec);

        // using the current state to compute rVecNew
        if(adjIO_.nkPseudoTransient) this->NKCalcResidualsUnsteady(wVecNew,dWVec,-alpha,rVecNew);
        else this->NKCalcResiduals(wVecNew,rVecNew);
        nFuncEvals_++;

        // compute the rVecNorm at new w
        ierr=VecNorm(rVecNew,NORM_2,&rVecNewNorm);

        if(ierr == PETSC_ERR_FP)
        {
            // floating point seg fault, just reduce the step size
            alpha = alpha*sigma;
            Info<<"Seg Fault in line search"<<endl;
            continue;
        }
        else if(this->checkNegativeTurb())
        {
            // have negative turbulence, reduce the step size
            alpha = alpha*sigma;
            Info<<"Negative turbulence in line search"<<endl;
            continue;
        }
        else
        {
            // this is a valid step, check if we get sufficient decrease in z(x)
            maxVal = zW0 + alpha*gamma*initSlope;
        }

        if (0.5*rVecNewNorm*rVecNewNorm <= maxVal)  // Sufficient reduction
        {
            return alpha;
        }
        else // reduce step
        {
            alpha = alpha * sigma;
        }

    }

    return alpha;

}


label AdjointNewtonKrylov::checkNegativeTurb()
{
    if(adjIO_.nkSegregatedTurb) return 0;

    const objectRegistry& db=mesh_.thisDb();
    forAll(nkStateNames_,idxI)
    {
        word stateName = nkStateNames_[idxI];
        word stateType = nkStateType_[stateName];
        if (stateType == "turbState")
        {
            const volScalarField& var =  db.lookupObject<volScalarField>(stateName) ;
            forAll(var,idxJ)
            {
                if (var[idxJ]<0)
                {
                    return 1;
                }
            }
            forAll(var.boundaryField(),patchI)
            {
                forAll(var.boundaryField()[patchI],faceI)
                {
                    if (var.boundaryField()[patchI][faceI]<0)
                    {
                        return 1;
                    }
                }
            }
        }
    }

    return 0;
}

void AdjointNewtonKrylov::printConvergenceInfo
(
    word mode,
    HashTable<scalar> objFuncs,
    label mainIter,
    label nFuncEval,
    word solverType,
    scalar step,
    scalar linRes,
    scalar CFL,
    scalar runTime,
    scalar turbNorm,
    scalar phiNorm,
    scalar totalNorm
    
)
{
    if (mode=="printHeader")
    {
        word printInfo="";
        printInfo +=   "--------------------------------------------------------------------------------------------------------";
        forAll(objFuncs.toc(),idxI)
        {
            printInfo += "------------------------";
        }
        printInfo += "\n";

        printInfo +=   "| Main | Func | Solv |  Step  |   Lin   |   CFL   |   RunTime   |  Res Norm  |  Res Norm  |  Res Norm  |";
        forAll(objFuncs.toc(),idxI)
        {
            word key = objFuncs.toc()[idxI];
            label keySize=key.size();
            for(label i=0;i<13-keySize;i++) printInfo += " ";
            printInfo += key+"          |";
        }
        printInfo += "\n";

        printInfo +=   "| Iter | Eval | Type |        |   Res   |         |   (s)       |  (Turb)    |  (Phi)     |  (Total)   |";
        forAll(objFuncs.toc(),idxI)
        {
            word key = objFuncs.toc()[idxI];
            printInfo += "                       |";
        }
        printInfo += "\n";

        printInfo +=   "--------------------------------------------------------------------------------------------------------";
        forAll(objFuncs.toc(),idxI)
        {
            printInfo += "------------------------";
        }
        printInfo += "\n";
    
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            printInfo.c_str()
        );
    }
    else if (mode=="printConvergence")
    {
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            " %6d %6d %s  %.4f   %.5f   %.2e   %.4e   %.4e   %.4e   %.4e",
            mainIter,
            nFuncEval,
            solverType.c_str(),
            step,
            linRes,
            CFL,
            runTime,
            turbNorm,
            phiNorm,
            totalNorm
        );
        forAll(objFuncs.toc(),idxI)
        {
            word key = objFuncs.toc()[idxI];
            scalar val = objFuncs[key];
            PetscPrintf
            (
                PETSC_COMM_WORLD,
                "   %.15e",
                val
            );
        }
        PetscPrintf
        (
            PETSC_COMM_WORLD,
            "\n"
        );
    }
    else
    {
        FatalErrorIn("")<< "mode not found!"<<abort(FatalError);
    }
}

void AdjointNewtonKrylov::solve()
{
    if(adjIO_.nkPseudoTransient) this->solveANK();
    else this->solveNK();
}

void AdjointNewtonKrylov::solveANK()
{
    Info<<endl<<"********* Solving Flow Using Approximated Newton-Krylov Solver *********"<<endl;
    
    // KSP
    KSP ksp;
    // Jacobians
    Mat dRdW,dRdWPC;
    // number of GMRES linear iterations
    label GMRESIters;
    // EW parameters
    scalar rTol=0.05;
    scalar rVecNorm;
    //scalar rVecNormOld;
    scalar aTol=1e-16;
    // initial residuals tolerance before the simulation
    scalar totalResNormFS;
    // turbulence Res norm
    scalar turbResNorm;
    // phi residual norm
    scalar phiResNorm;
    // total residual norm including all variables
    scalar totalResNorm;
    // total residual norm including all variables last step
    scalar totalResNormOld;

    scalar stepSize=1.0;

    // local size of dRdW
    label localSize = nLocalNKStates_;
    // these vars are for store the tolerance for GMRES linear solution
    scalar rGMRESHist[adjIO_.nkGMRESMaxIters+1];
    label nGMRESIters=adjIO_.nkGMRESMaxIters+1;

    // initialize ANK CFL and dTInv
    ANKCFL_=5.0e5;
    this->setdTInv(dTInv_);

    //adjIO_.writeVectorASCII(dTInv_,"dT");

    // create dRdW
    MatCreateMFFD(PETSC_COMM_WORLD,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE,&dRdW);
    MatMFFDSetFunction(dRdW,FormFunctionANK,this);

    // initialize and calculate PCMat
    this->calcPCMat4NK(&dRdWPC,1);

    // first compute/assign the initial wVec and rVec and compute the initial totalResNormFS
    this->NKSetVecs(wVec0_,"Var2Vec",1.0,"");
    this->NKCalcResidualsUnsteady(wVec0_,wVec0_,0.01,rVec0_); // initially, we set dW=0.01*w
    VecNorm(rVec0_,NORM_2,&rVecNorm);
    Info<<"rVecNorm "<<rVecNorm<<endl;

    totalResNormFS=this->getResNorm("total");
    totalResNorm=totalResNormFS;
    totalResNormOld=totalResNormFS;
    turbResNorm=this->getResNorm("turb");
    phiResNorm=this->getResNorm("phi");

    // add options and initialize ksp
    dictionary adjOptions;
    adjOptions.add("GMRESRestart",adjIO_.nkGMRESRestart);
    adjOptions.add("GlobalPCIters",adjIO_.nkGlobalPCIters);
    adjOptions.add("ASMOverlap",adjIO_.nkASMOverlap);
    adjOptions.add("LocalPCIters",adjIO_.nkLocalPCIters);
    adjOptions.add("JacMatReOrdering",adjIO_.nkJacMatReOrdering);
    adjOptions.add("PCFillLevel",adjIO_.nkPCFillLevel);
    adjOptions.add("GMRESMaxIters",adjIO_.nkGMRESMaxIters);
    adjOptions.add("GMRESRelTol",rTol);
    adjOptions.add("GMRESAbsTol",aTol);
    adjOptions.add("printInfo",0);
    //Info<<adjOptions<<endl;
    adjDev_.createMLRKSP(&ksp,dRdW,dRdWPC,adjOptions);

    // initialize objFuncs
    HashTable<scalar> objFuncs;
    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];
        adjObj_.calcObjFuncs(objFunc,0);
        scalar val = adjObj_.getObjFunc(objFunc);
        objFuncs.set(objFunc,val);
    }

    // print the initial convergence information
    this->printConvergenceInfo("printHeader",objFuncs);
    this->printConvergenceInfo
    (
        "printConvergence",
        objFuncs,
        0,
        0,
        "  ANK ",
        1.0,
        0.0,
        ANKCFL_,
        scalar(adjDev_.getRunTime()),
        turbResNorm,
        phiResNorm,
        totalResNormFS
    );

    // main loop for ANK
    
    for(label iterI=1;iterI<adjIO_.nkMaxIters+1;iterI++)
    {
    
        // update the w and v vectors
        if (iterI>1) 
        {
            ANKCFL_=this->updateANKCFL(ANKCFL_,totalResNormFS,totalResNorm,totalResNormOld,stepSize);

            VecCopy(rVec1_,rVec0_);
            VecCopy(wVec1_,wVec0_);
            // check if we need to recompute PC
            if(iterI%adjIO_.nkPCLag==0)
            {
                this->calcPCMat4NK(&dRdWPC,0);
                KSPDestroy(&ksp);
                adjDev_.createMLRKSP(&ksp,dRdW,dRdWPC,adjOptions);
            }
        }
    
        // set up rGMRESHist to save the tolerance history for the GMRES solution
        KSPSetResidualHistory(ksp,rGMRESHist,nGMRESIters,PETSC_TRUE);

        // before solving the ksp, form the baseVector for matrix-vector products.
        // Note: we need to apply normalize-states to the baseVector
        // Note that we also scale the dRdW*psi 
        // in AdjointNewtonKrylov::FormFunction
        Vec rVecBase;
        VecDuplicate(rVec0_,&rVecBase);
        VecZeroEntries(rVecBase);
        //VecCopy(rVec0_,rVecBase); 
        this->FormFunctionANK(this,wVec0_,rVecBase);
        //this->setNormalizeStatesScaling2Vec(rVecBase);
        MatMFFDSetBase(dRdW,wVec0_,rVecBase);

        // solve the linear system
        // we should use rVec0 as the rhs, however, we need to normalize
        // the states, so we create this temporary rhs vec to store 
        // the scaled rVec. Note that we also scale the dRdW*psi 
        // in AdjointNewtonKrylov::FormFunction
        Vec rhs;
        VecDuplicate(rVec0_,&rhs);
        VecCopy(rVec0_,rhs);
        this->setNormalizeStatesScaling2Vec(rhs);
        KSPSolve(ksp,rhs,dWVec_);
    
        // get linear solution rTol: linRes
        KSPGetIterationNumber(ksp,&GMRESIters);
        nFuncEvals_+=GMRESIters;
        scalar linRes=rGMRESHist[GMRESIters]/rGMRESHist[0];
    
        // do a line search and update states
        VecZeroEntries(rVec1_);
        VecZeroEntries(wVec1_);
        stepSize=this->NKLineSearchNew(wVec0_,rVec0_,dWVec_,wVec1_,rVec1_);

        if(stepSize<0.01) 
        {    
            VecCopy(rVec0_,rVec1_);
            VecCopy(wVec0_,wVec1_);
        }

        // update dTInv
        this->setdTInv(dTInv_);

        // update the norm for printting convergence info
        VecNorm(rVec1_,NORM_2,&rVecNorm);
        Info<<"rVecNorm "<<rVecNorm<<endl;
        totalResNormOld=totalResNorm;
        totalResNorm=this->getResNorm("total");
        turbResNorm=this->getResNorm("turb");
        phiResNorm=this->getResNorm("phi");

        // update objective function values
        forAll(adjIO_.objFuncs,idxI)
        {
            word objFunc = adjIO_.objFuncs[idxI];
            adjObj_.calcObjFuncs(objFunc,0);
            scalar val = adjObj_.getObjFunc(objFunc);
            objFuncs.set(objFunc,val);
        }

        // print convergence info
        if (iterI>1 && iterI%20==0)
        {
            this->printConvergenceInfo("printHeader",objFuncs);
        }
        this->printConvergenceInfo
        (
            "printConvergence",
            objFuncs,
            iterI,
            nFuncEvals_,
            "  ANK ",
            stepSize,
            linRes,
            ANKCFL_,
            scalar(adjDev_.getRunTime()),
            turbResNorm,
            phiResNorm,
            totalResNorm
        );
        
    }

    // assign the latest wVec to variables
    this->NKSetVecs(wVec1_,"Vec2Var",1.0,"");

    return;
}


void AdjointNewtonKrylov::solveNK()
{
    
    Info<<endl<<"********* Solving Flow Using Newton-Krylov Solver *********"<<endl;
    
    // KSP
    KSP ksp;
    // Jacobians
    Mat dRdW,dRdWPC;
    // number of GMRES linear iterations
    label GMRESIters;
    // EW parameters
    scalar rTolLast=adjIO_.nkEWRTol0; // EW rTol0, EW default is 0.1 but we can increase a bit
    scalar rTol=rTolLast;
    scalar rVecNorm;
    scalar rVecNormOld=0.0;
    scalar aTol=1e-16;
    // initial residuals tolerance before the simulation
    scalar totalResNorm0;
    // turbulence Res norm
    scalar turbResNorm;
    // phi residual norm
    scalar phiResNorm;
    // total residual norm including all variables
    scalar totalResNorm;

    // local size of dRdW
    label localSize = nLocalNKStates_;
    // these vars are for store the tolerance for GMRES linear solution
    scalar rGMRESHist[adjIO_.nkGMRESMaxIters+1];
    label nGMRESIters=adjIO_.nkGMRESMaxIters+1;

    // create dRdW
    MatCreateMFFD(PETSC_COMM_WORLD,localSize,localSize,PETSC_DETERMINE,PETSC_DETERMINE,&dRdW);
    MatMFFDSetFunction(dRdW,FormFunction,this);

    // initialize and calculate PCMat
    this->calcPCMat4NK(&dRdWPC,1);

    // first compute/assign the initial wVec and rVec and compute the initial totalResNorm0
    this->NKSetVecs(wVec0_,"Var2Vec",1.0,"");
    this->NKCalcResiduals(wVec0_,rVec0_);
    //VecNorm(rVec0_,NORM_2,&rVecNorm);

    totalResNorm0=this->getResNorm("total");
    totalResNorm=totalResNorm0;
    turbResNorm=this->getResNorm("turb");
    phiResNorm=this->getResNorm("phi");

    // add options and initialize ksp
    dictionary adjOptions;
    adjOptions.add("GMRESRestart",adjIO_.nkGMRESRestart);
    adjOptions.add("GlobalPCIters",adjIO_.nkGlobalPCIters);
    adjOptions.add("ASMOverlap",adjIO_.nkASMOverlap);
    adjOptions.add("LocalPCIters",adjIO_.nkLocalPCIters);
    adjOptions.add("JacMatReOrdering",adjIO_.nkJacMatReOrdering);
    adjOptions.add("PCFillLevel",adjIO_.nkPCFillLevel);
    adjOptions.add("GMRESMaxIters",adjIO_.nkGMRESMaxIters);
    adjOptions.add("GMRESRelTol",rTol);
    adjOptions.add("GMRESAbsTol",aTol);
    adjOptions.add("printInfo",0);
    //Info<<adjOptions<<endl;
    adjDev_.createMLRKSP(&ksp,dRdW,dRdWPC,adjOptions);

    // initialize objFuncs
    HashTable<scalar> objFuncs;
    forAll(adjIO_.objFuncs,idxI)
    {
        word objFunc = adjIO_.objFuncs[idxI];
        adjObj_.calcObjFuncs(objFunc,0);
        scalar val = adjObj_.getObjFunc(objFunc);
        objFuncs.set(objFunc,val);
    }

    // print the initial convergence information
    this->printConvergenceInfo("printHeader",objFuncs);
    this->printConvergenceInfo
    (
        "printConvergence",
        objFuncs,
        0,
        0,
        "  NK  ",
        1.0,
        0.0,
        1.0,
        scalar(adjDev_.getRunTime()),
        turbResNorm,
        phiResNorm,
        totalResNorm
    );

    // main loop for NK
    for(label iterI=1;iterI<adjIO_.nkMaxIters+1;iterI++)
    {
        // check if the presribed tolerances are met
        if (totalResNorm<adjIO_.nkAbsTol)
        {
            Info<<"Absolute Tolerance "<<totalResNorm<<" less than the presribed nkAbsTol "<<adjIO_.nkAbsTol<<endl;
            Info<<"NK completed!"<<endl;
            break;
        }
        else if (totalResNorm/totalResNorm0<adjIO_.nkRelTol)
        {
            Info<<"Relative Tolerance "<<totalResNorm/totalResNorm0<<" less than the presribed nkRelTol "<<adjIO_.nkRelTol<<endl;
            Info<<"NK completed!"<<endl;
            break;
        }
    
        // update the w and v vectors
        if (iterI>1) 
        {
            VecCopy(rVec1_,rVec0_);
            VecCopy(wVec1_,wVec0_);
        }
       
        // compute relative tol using EW
        VecNorm(rVec0_,NORM_2,&rVecNorm);
        if (iterI>1) rTol=this->getEWTol(rVecNorm,rVecNormOld,rTolLast);
        rVecNormOld=rVecNorm;
        rTolLast=rTol;

        if(iterI>1)
        {
            // check if we need to recompute PC
            if(iterI%adjIO_.nkPCLag==0)
            {
                this->calcPCMat4NK(&dRdWPC,0);
                KSPDestroy(&ksp);
                adjOptions.set("GMRESRelTol",rTol);
                adjDev_.createMLRKSP(&ksp,dRdW,dRdWPC,adjOptions);
            }
            else
            {
                // we need to reassign the relative tolerance computed from EW
                KSPSetTolerances(ksp,rTol,aTol,PETSC_DEFAULT,adjIO_.nkGMRESMaxIters);
            }

        }
    
        // set up rGMRESHist to save the tolerance history for the GMRES solution
        KSPSetResidualHistory(ksp,rGMRESHist,nGMRESIters,PETSC_TRUE);

        // before solving the ksp, form the baseVector for matrix-vector products.
        // Note: we need to apply normalize-states to the baseVector
        // Note that we also scale the dRdW*psi 
        // in AdjointNewtonKrylov::FormFunction
        Vec rVecBase;
        VecDuplicate(rVec0_,&rVecBase);
        VecCopy(rVec0_,rVecBase); 
        this->setNormalizeStatesScaling2Vec(rVecBase);
        MatMFFDSetBase(dRdW,wVec0_,rVecBase);

        // solve the linear system
        // we should use rVec0 as the rhs, however, we need to normalize
        // the states, so we create this temporary rhs vec to store 
        // the scaled rVec. Note that we also scale the dRdW*psi 
        // in AdjointNewtonKrylov::FormFunction
        Vec rhs;
        VecDuplicate(rVec0_,&rhs);
        VecCopy(rVec0_,rhs);
        this->setNormalizeStatesScaling2Vec(rhs);
        KSPSolve(ksp,rhs,dWVec_);
    
        // get linear solution rTol: linRes
        KSPGetIterationNumber(ksp,&GMRESIters);
        nFuncEvals_+=GMRESIters;
        scalar linRes=rGMRESHist[GMRESIters]/rGMRESHist[0];
    
        // do a line search and update states
        VecZeroEntries(rVec1_);
        VecZeroEntries(wVec1_);
        scalar stepSize=this->NKLineSearchNew(wVec0_,rVec0_,dWVec_,wVec1_,rVec1_);
        // we also update phi if nkSegregatedPhi is true
        if(adjIO_.nkSegregatedPhi)
        {
            label isRef=0,isPC=0,updatePhi=1;
            this->NKSetVecs(wVec1_,"Vec2Var",1.0,"");
            this->NKSetPhiVec(phiVec0_,"Var2Vec",1.0,"");
            adjDev_.calcResiduals(isRef,isPC,"None",updatePhi);
            nFuncEvals_++;
            this->NKSetPhiVec(dPhiVec_,"Var2Vec",1.0,"");
            VecAXPY(dPhiVec_,-1.0,phiVec0_);

            Vec phiVec1_;
            VecDuplicate(phiVec0_,&phiVec1_);

            // do a line search for phi, 
            // we select a phi step that reduces rVecNorm
            scalar step=1.0;
            for(label ls=0;ls<10;ls++)
            {
                this->NKCalcResiduals(wVec1_,rVec1_);
                VecNorm(rVec1_,NORM_2,&rVecNorm);
                if (rVecNorm<rVecNormOld)
                {
                    break;
                }
                else
                {
                    step *=0.5; // need to tweak this
                    VecWAXPY(phiVec1_,step,dPhiVec_,phiVec0_);
                    this->NKSetPhiVec(phiVec1_,"Vec2Var",1.0,"");
                }
                
            }
        }

        // update the norm for printting convergence info
        VecNorm(rVec1_,NORM_2,&rVecNorm);
        totalResNorm=this->getResNorm("total");
        turbResNorm=this->getResNorm("turb");
        phiResNorm=this->getResNorm("phi");

        // update objective function values
        forAll(adjIO_.objFuncs,idxI)
        {
            word objFunc = adjIO_.objFuncs[idxI];
            adjObj_.calcObjFuncs(objFunc,0);
            scalar val = adjObj_.getObjFunc(objFunc);
            objFuncs.set(objFunc,val);
        }

        // print convergence info
        if (iterI>1 && iterI%20==0)
        {
            this->printConvergenceInfo("printHeader",objFuncs);
        }
        this->printConvergenceInfo
        (
            "printConvergence",
            objFuncs,
            iterI,
            nFuncEvals_,
            "  NK  ",
            stepSize,
            linRes,
            1.0,
            scalar(adjDev_.getRunTime()),
            turbResNorm,
            phiResNorm,
            totalResNorm
        );

        // check if the presribed tolerances are met
        if (Foam::mag(rVecNorm-rVecNormOld)/rVecNormOld<adjIO_.nkSTol)
        {
            Info<<"S Tolerance "<<Foam::mag(rVecNorm-rVecNormOld)/rVecNormOld<<" less than the presribed nkSTol "<<adjIO_.nkSTol<<endl;
            Info<<"NK completed!"<<endl;
            break;
        }
        if (linRes>0.999)
        {
            Info<<"GMRES residual drop "<<linRes<<" too small! Quit! "<<endl;
            break;
        }
        
    }

    // assign the latest wVec to variables
    this->NKSetVecs(wVec1_,"Vec2Var",1.0,"");

    return;
    

}


PetscErrorCode AdjointNewtonKrylov::FormFunction(void* ctx,Vec wVec,Vec rVec)
{
    
    // Given an input vec wVec, calculate the function vec rWec
    // wVec: vector storing all the states
    // rVec: vector storing the residuals

    AdjointNewtonKrylov *adjNK = (AdjointNewtonKrylov*) ctx;

    adjNK->NKCalcResiduals(wVec,rVec);

    adjNK->setNormalizeStatesScaling2Vec(rVec);
    
    return 0;
    
}

PetscErrorCode AdjointNewtonKrylov::FormFunctionANK(void* ctx,Vec wVec,Vec rVec)
{
    
    // Given an input vec wVec, calculate the function vec rWec
    // wVec: vector storing all the states
    // rVec: vector storing the residuals

    AdjointNewtonKrylov *adjNK = (AdjointNewtonKrylov*) ctx;

    adjNK->NKCalcResiduals(wVec,rVec);

    adjNK->NKAddUnsteadyTerms(wVec,rVec);

    adjNK->setNormalizeStatesScaling2Vec(rVec);
    
    return 0;
    
}

scalar AdjointNewtonKrylov::getResNorm(word mode)
{

    scalar totalResNorm2=0.0;
    scalar turbResNorm2=0.0;
    scalar phiResNorm2=0.0;

    forAll(adjReg_.volVectorStates,idxI)
    {
        const word stateName = adjReg_.volVectorStates[idxI];
        const word resName = stateName+"Res";                   
        const volVectorField& stateRes = db_.lookupObject<volVectorField>(resName); 
        
        vector vecResNorm2(0,0,0);
        forAll(stateRes,cellI)
        {
            vecResNorm2.x()+=Foam::pow(stateRes[cellI].x(),2.0);
            vecResNorm2.y()+=Foam::pow(stateRes[cellI].y(),2.0);
            vecResNorm2.z()+=Foam::pow(stateRes[cellI].z(),2.0);
        }
        totalResNorm2 += vecResNorm2.x() + vecResNorm2.y() + vecResNorm2.z();
    }
    
    forAll(adjReg_.volScalarStates,idxI)
    {
        const word stateName = adjReg_.volScalarStates[idxI];
        const word resName = stateName+"Res";                   
        const volScalarField& stateRes = db_.lookupObject<volScalarField>(resName); 
        
        scalar scalarResNorm2=0;
        forAll(stateRes,cellI)
        {
            scalarResNorm2+=Foam::pow(stateRes[cellI],2.0);
        }
        totalResNorm2 += scalarResNorm2;
    }
    
    forAll(adjRAS_.turbStates,idxI)
    {
        const word stateName = adjRAS_.turbStates[idxI];
        const word resName = stateName+"Res";  
        const volScalarField& stateRes = db_.lookupObject<volScalarField>(resName); 
        
        scalar scalarResNorm2=0;
        forAll(stateRes,cellI)
        {
            scalarResNorm2+=Foam::pow(stateRes[cellI],2.0);
        }
        totalResNorm2 += scalarResNorm2;
        turbResNorm2  += scalarResNorm2;
    }
    
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        const word stateName = adjReg_.surfaceScalarStates[idxI];
        const word resName = stateName+"Res";  
        const surfaceScalarField& stateRes = db_.lookupObject<surfaceScalarField>(resName); 
        
        forAll(stateRes,faceI)
        {
            phiResNorm2+=Foam::pow(stateRes[faceI],2.0);
    
        }
        forAll(stateRes.boundaryField(),patchI)
        {
            forAll(stateRes.boundaryField()[patchI],faceI)
            {
                scalar bPhiRes = stateRes.boundaryField()[patchI][faceI];
                phiResNorm2+=Foam::pow(bPhiRes,2.0);
            }
        }
        totalResNorm2 += phiResNorm2;
        
    }


    reduce(totalResNorm2,sumOp<scalar>());
    totalResNorm2=Foam::pow(totalResNorm2,0.5);

    reduce(turbResNorm2,sumOp<scalar>());
    turbResNorm2=Foam::pow(turbResNorm2,0.5);

    reduce(phiResNorm2,sumOp<scalar>());
    phiResNorm2=Foam::pow(phiResNorm2,0.5);

    if(mode=="total") return totalResNorm2;
    else if (mode=="turb") return turbResNorm2;
    else if (mode=="phi") return phiResNorm2;
    else FatalErrorIn("")<<"mode not valid"<< abort(FatalError);

    FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    return -10000.0;

}

void AdjointNewtonKrylov::NKSetVecs
(
    Vec vecX, 
    word mode, 
    scalar scaleFactor, 
    word postFix
)
{
    
    // this is a general function to assign vec to/from states/residuals

    PetscInt    Istart, Iend;
    VecGetOwnershipRange(vecX,&Istart,&Iend);
    PetscScalar* vecXArray;
    const PetscScalar* vecXArrayConst;
    if(mode == "Vec2Var" || mode == "VecAdd2Var")
    {
        VecGetArrayRead(vecX,&vecXArrayConst);
    }
    else if (mode == "Var2Vec" || mode == "VarAdd2Vec")
    {
        VecGetArray(vecX,&vecXArray);
    }
    else
    {
        FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    }
    
    for(PetscInt i=Istart;i<Iend;i++)
    {
        label localIdx =  i-Istart;

        word stateName = stateName4LocalNKIdx_[localIdx];
        word varName = stateName+postFix;
        const word& stateType = nkStateType_[stateName];
        
        scalar cellIFaceI = cellIFaceI4LocalNKIdx_[localIdx];

        if(stateType == "volVectorState")
        {

            volVectorField& var = const_cast<volVectorField&>( db_.lookupObject<volVectorField>(varName) );
            label cellI,comp;
            cellI = round(cellIFaceI);
            comp = round(10*(cellIFaceI-cellI));

            if(mode == "Vec2Var")
            {
                var[cellI][comp] = scaleFactor*vecXArrayConst[localIdx];
            }
            else if (mode == "Var2Vec")
            {
                vecXArray[localIdx] = scaleFactor*var[cellI][comp];
            }
            else if (mode == "VecAdd2Var")
            {
                var[cellI][comp] += scaleFactor*vecXArrayConst[localIdx];
            }
            else if (mode == "VarAdd2Vec")
            {
                vecXArray[localIdx] += scaleFactor*var[cellI][comp];
            }
            else
            {
                 FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
            }
            
        }
        else if(stateType == "volScalarState")
        {

            volScalarField& var = const_cast<volScalarField&>( db_.lookupObject<volScalarField>(varName) );
            label cellI;
            cellI = round(cellIFaceI);

            if(mode == "Vec2Var")
            {
                var[cellI] = scaleFactor*vecXArrayConst[localIdx];
            }
            else if (mode == "Var2Vec")
            {
                vecXArray[localIdx] = scaleFactor*var[cellI];
            }
            else if (mode == "VecAdd2Var")
            {
                var[cellI] += scaleFactor*vecXArrayConst[localIdx];
            }
            else if (mode == "VarAdd2Vec")
            {
                vecXArray[localIdx] += scaleFactor*var[cellI];
            }
            else
            {
                 FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
            }

        }
        else if(stateType == "turbState" and !adjIO_.nkSegregatedTurb)
        {

            volScalarField& var = const_cast<volScalarField&>( db_.lookupObject<volScalarField>(varName) );
            label cellI;
            cellI = round(cellIFaceI);

            if(mode == "Vec2Var")
            {
                var[cellI] = scaleFactor*vecXArrayConst[localIdx];
            }
            else if (mode == "Var2Vec")
            {
                vecXArray[localIdx] = scaleFactor*var[cellI];
            }
            else if (mode == "VecAdd2Var")
            {
                var[cellI] += scaleFactor*vecXArrayConst[localIdx];
            }
            else if (mode == "VarAdd2Vec")
            {
                vecXArray[localIdx] += scaleFactor*var[cellI];
            }
            else
            {
                 FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
            }
        }
        else if (stateType == "surfaceScalarState")
        {

            surfaceScalarField& var = const_cast<surfaceScalarField&>( db_.lookupObject<surfaceScalarField>(varName) );
            label faceI=round(cellIFaceI);
            if(faceI<mesh_.nInternalFaces())
            {
                if(mode == "Vec2Var")
                {
                    var[faceI] = scaleFactor*vecXArrayConst[localIdx];
                }
                else if (mode == "Var2Vec")
                {
                    vecXArray[localIdx] = scaleFactor*var[faceI];
                }
                else if (mode == "VecAdd2Var")
                {
                    var[faceI] += scaleFactor*vecXArrayConst[localIdx];
                }
                else if (mode == "VarAdd2Vec")
                {
                    vecXArray[localIdx] += scaleFactor*var[faceI];
                }
                else
                {
                     FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
                }
            }
            else
            {
                label relIdx=faceI-mesh_.nInternalFaces();
                label patchIdx=bFacePatchI_[relIdx];
                label faceIdx=bFaceFaceI_[relIdx];

                if(mode == "Vec2Var")
                {
                    var.boundaryFieldRef()[patchIdx][faceIdx] = scaleFactor*vecXArrayConst[localIdx];
                }
                else if (mode == "Var2Vec")
                {
                    vecXArray[localIdx] = scaleFactor*var.boundaryFieldRef()[patchIdx][faceIdx];
                }
                else if (mode == "VecAdd2Var")
                {
                    var.boundaryFieldRef()[patchIdx][faceIdx] += scaleFactor*vecXArrayConst[localIdx];
                }
                else if (mode == "VarAdd2Vec")
                {
                    vecXArray[localIdx] += scaleFactor*var.boundaryFieldRef()[patchIdx][faceIdx];
                }
                else
                {
                     FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
                }

            }
        }
        else
        {
            FatalErrorIn("")<<"statetype not valid"<< abort(FatalError);
        }
    }
    
    if(mode == "Vec2Var" || mode == "VecAdd2Var")
    {
        VecRestoreArrayRead(vecX,&vecXArrayConst);
    }
    else if (mode == "Var2Vec" || mode == "VarAdd2Vec")
    {
        VecRestoreArray(vecX,&vecXArray);
        //Info<<"Pass NKFormFunctoin5"<<endl;
        VecAssemblyBegin(vecX);
        VecAssemblyEnd(vecX);
    }
    else
    {
        FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    }

    if(mode == "Vec2Var" or mode == "VecAdd2Var")
    {
        adjDev_.updateStateVariableBCs();
    }
    
    return;

}

void AdjointNewtonKrylov::NKSetPhiVec
(
    Vec vecX, 
    word mode,
    scalar scaleFactor, 
    word postFix
)
{
    // this function only sets or gets phiVec

    PetscScalar* vecXArray;
    const PetscScalar* vecXArrayConst;
    if(mode == "Vec2Var")
    {
        VecGetArrayRead(vecX,&vecXArrayConst);
    }
    else if (mode == "Var2Vec")
    {
        VecGetArray(vecX,&vecXArray);
    }
    else
    {
        FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    }
    
    forAll(adjReg_.surfaceScalarStates,idxI)
    {
        word stateName = adjReg_.surfaceScalarStates[idxI];
        word varName = stateName+postFix;
        surfaceScalarField& var = const_cast<surfaceScalarField&>( db_.lookupObject<surfaceScalarField>(varName) );
        for(label faceI=0;faceI<mesh_.nFaces();faceI++)
        {
            label localIdx = faceI+mesh_.nFaces()*idxI; // state-by-state ordering
            if(faceI<mesh_.nInternalFaces())
            {
                if(mode == "Vec2Var")
                {
                    var[faceI] = scaleFactor * vecXArrayConst[localIdx];
                }
                else if (mode == "Var2Vec")
                {
                    vecXArray[localIdx] = scaleFactor * var[faceI];
                }
                else
                {
                    FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
                }
            }
            else
            {
                label relIdx=faceI-mesh_.nInternalFaces();
                label patchIdx=bFacePatchI_[relIdx];
                label faceIdx=bFaceFaceI_[relIdx];

                if(mode == "Vec2Var")
                {
                    var.boundaryFieldRef()[patchIdx][faceIdx] = scaleFactor * vecXArrayConst[localIdx];
                }
                else if (mode == "Var2Vec")
                {
                    vecXArray[localIdx] = scaleFactor * var.boundaryFieldRef()[patchIdx][faceIdx];
                }
                else
                {
                     FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
                }
            }
        }    
    }

    
    if(mode == "Vec2Var")
    {
        VecRestoreArrayRead(vecX,&vecXArrayConst);
    }
    else if (mode == "Var2Vec")
    {
        VecRestoreArray(vecX,&vecXArray);
        //Info<<"Pass NKFormFunctoin5"<<endl;
        VecAssemblyBegin(vecX);
        VecAssemblyEnd(vecX);
    }
    else
    {
        FatalErrorIn("")<<"mode not valid"<< abort(FatalError);
    }
    
    return;

}

void AdjointNewtonKrylov::setdTInv(Vec dTInv)
{

    VecZeroEntries(dTInv);

    PetscInt    Istart, Iend;
    VecGetOwnershipRange(dTInv,&Istart,&Iend);

    PetscScalar* dTInvArray;
    VecGetArray(dTInv,&dTInvArray);

    const volVectorField& U = db_.lookupObject<volVectorField>("U") ;

    for(PetscInt i=Istart;i<Iend;i++)
    {
        label localIdx =  i-Istart;

        word stateName = stateName4LocalNKIdx_[localIdx];
        const word& stateType = nkStateType_[stateName];
        
        scalar cellIFaceI = cellIFaceI4LocalNKIdx_[localIdx];

        if(stateType == "surfaceScalarState")
        {
            // Get the owner cells for this face
            label faceI=round(cellIFaceI);
            label ownerCellI=-100;
            if (faceI<mesh_.nInternalFaces())
            {
                ownerCellI = mesh_.owner()[faceI];
            }
            else
            {
                label relIdx=faceI-mesh_.nInternalFaces();
                label patchIdx=bFacePatchI_[relIdx];
                label faceIdx=bFaceFaceI_[relIdx];
                const UList<label>& pFaceCells = mesh_.boundaryMesh()[patchIdx].faceCells();
                ownerCellI = pFaceCells[faceIdx];
            }

            const scalar meshV = mesh_.V()[ownerCellI];
            scalar dX = Foam::pow(meshV,1.0/3.0);
            scalar vel = mag(U[ownerCellI]);
            dTInvArray[localIdx] = vel/ANKCFL_/dX;

        }
        else
        {
            label cellI=round(cellIFaceI);
            const scalar meshV = mesh_.V()[cellI];
            scalar dX = Foam::pow(meshV,1.0/3.0);
            scalar vel = mag(U[cellI]);

            dTInvArray[localIdx] = vel/ANKCFL_/dX;
        }
    }

    VecRestoreArray(dTInv,&dTInvArray);

    return;
}

void AdjointNewtonKrylov::NKAddUnsteadyTerms(Vec unsteadyVec, Vec rVec)
{
    // add unsteady terms
    PetscScalar* rVecArray;
    VecGetArray(rVec,&rVecArray);

    const PetscScalar* unsteadyVecArray;
    VecGetArrayRead(unsteadyVec,&unsteadyVecArray);

    const PetscScalar* dTInvArray;
    VecGetArrayRead(dTInv_,&dTInvArray);

    PetscInt    Istart, Iend;
    VecGetOwnershipRange(rVec,&Istart,&Iend);

    for(PetscInt i=Istart;i<Iend;i++)
    {
        label localIdx =  i-Istart;
        rVecArray[localIdx] = rVecArray[localIdx] + dTInvArray[localIdx]*unsteadyVecArray[localIdx];
    } 

    VecRestoreArray(rVec,&rVecArray);
    VecRestoreArrayRead(unsteadyVec,&unsteadyVecArray);
    VecRestoreArrayRead(dTInv_,&dTInvArray);

    return;
}


scalar AdjointNewtonKrylov::updateANKCFL
(
    scalar CFL0,
    scalar totalResNormFS,
    scalar totalResNorm,
    scalar totalResNormOld,
    scalar stepSize
)
{
    scalar CFLMax=1.0e5;
    scalar CFLMin=Foam::pow(totalResNormFS/totalResNorm,0.5);

    scalar stepRamp=0.4;
    scalar stepMin=0.01;

    scalar newCFL=-1000;
    if(stepSize>stepRamp)
    {
        scalar gamma = max( (totalResNormOld-totalResNorm)/totalResNormOld,0.0 );
        newCFL = min(CFL0*Foam::pow(10.0,gamma),CFLMax);
        return newCFL;
    }
    else if(stepSize>stepMin)
    {
        newCFL = max(CFL0,CFLMin);
        return newCFL;
    }
    else
    {
        newCFL = max(0.5*CFL0,CFLMin);
        return newCFL;
    }

    return newCFL;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
