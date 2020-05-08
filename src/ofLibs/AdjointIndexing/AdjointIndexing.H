/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Handle Jacobian or state indexing for the adjoint

\*---------------------------------------------------------------------------*/

#ifndef AdjointIndexing_H
#define AdjointIndexing_H

#include "AdjointSolverRegistry.H"
#include "AdjointIO.H"
#include "AdjointRASModel.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "globalIndex.H"
#include "HashTable.H"
#include "syncTools.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointIndexing Declaration
\*---------------------------------------------------------------------------*/
class AdjointIndexing
{

protected:

    const fvMesh& mesh_;
    const AdjointIO& adjIO_;
    const AdjointSolverRegistry& adjReg_;
    const AdjointRASModel& adjRAS_;

public:

    // Constructors

    //- Construct from components
    AdjointIndexing
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        const AdjointRASModel& adjRAS
    );

    //- Destructor
    virtual ~AdjointIndexing()
    {};
    
    // adjoint indexing info
    /// list of adjoint state names for a specific solver
    List<word> adjStateNames;
    
    /// hash table of adjoint state types, e.g., volVectorState for a given state name
    HashTable<word> adjStateType;
    
    /** hash table of  local state variable index offset, see initializeStateLocalIndexOffset for definition. 
        This will be used in determing the local indexing  for adjoint states. It differs depending on whether
        we use state-by-state or cell-by-cell ordering
    */
    HashTable<label> stateLocalIndexOffset;
    
    /// a unique number ID for adjoint states, it depends on the sequence of adjStateNames
    HashTable<label> adjStateID;

    // mesh sizes
    // local sizes
    
    /// local cell size
    label nLocalCells;
    
    /// local face size
    label nLocalFaces;
    
    /// local point size
    label nLocalPoints;
    
    /// local Xv size (point size*3)
    label nLocalXv;
    
    /// local boundary face size
    label nLocalBoundaryFaces;
    
    /// local internal face size
    label nLocalInternalFaces;
    
    /// local boundary patch size
    label nLocalBoundaryPatches;
    
    /// local coupled boundary patch size
    label nLocalCoupledBFaces;

    /// local cyclicAMI patch size
    label nLocalCyclicAMIFaces;
    
    /** given a local boundary face index (starting with zero for the first boundary face, 
        and ends with the last boundary faces), return its boundary patchI, 
        e.g., for face index = 100, its patchI = 2
    */
    labelList bFacePatchI;

    /** given a local boundary face index (starting with zero for the first boundary face, 
        and ends with the last boundary faces), return its boundary faceI associated with a boundary patchI,
        e.g., for face index = 100, its patchI = 2, faceI=20
    */
    labelList bFaceFaceI;

    /// for a given face index, return whether this face is a coupled boundary face
    labelList isCoupledFace;

    /// for a given face index, return whether this face is a cyclicAMI boundary face
    labelList isCyclicAMIFace;

    // glocal sizes
    /// global cell size
    label nGlobalCells;

    /// global face size
    label nGlobalFaces;

    /// global Xv size (global face size*3)
    label nGlobalXv;

    /// global objective function face size
    label nGlobalObjFuncFaces;

    /// global coupled boundary face size
    label nGlobalCoupledBFaces;
    
    // adjoint sizes
    /// number of local adjoint states (including all cells and faces)
    label nLocalAdjointStates;

    /// number of global adjoint states (including all cells and faces)
    label nGlobalAdjointStates;

    /// number of state variables for volScalarField, volVectorField, and surfaceScalarField
    label nVolScalarStates;

    /// number of state variables for volScalarField, volVectorField, and surfaceScalarField
    label nVolVectorStates;

    /// number of state variables for volScalarField, volVectorField, and surfaceScalarField
    label nSurfaceScalarStates;

    /// number of turbulence states, NOTE: they are counted separately
    label nTurbStates;
    
    /// \name global indexing variables, they will be used to transfer local and global indices
    //@{
    globalIndex globalAdjointStateNumbering;
    globalIndex globalCellNumbering;
    globalIndex globalCellVectorNumbering; // similar to globalCellNumbering but has 3 components per cell
    globalIndex globalFaceNumbering;
    globalIndex globalCoupledBFaceNumbering;
    globalIndex globalObjFuncGeoNumbering;
    globalIndex globalXvNumbering;
    //@}

    /// given a local adjoint state index, return its state name
    wordList adjStateName4LocalAdjIdx;

    /// given a local adjoint state index, return its cell/face index
    scalarList cellIFaceI4LocalAdjIdx;

    /// a list to map the point index from decomposed domains to the original un-decomposed domain
    labelIOList pointProcAddressing;

    /// number of points for the un-decomposed domain
    label nUndecomposedPoints;
        
    // Member functions
    /// generate global index numbering for local-global index transferring
    globalIndex genGlobalIndex(const label index);
    
    /// get local adjoint index for a given state name, cell/face indxI and its component (optional, only for vector states)
    label getLocalAdjointStateIndex
    (
        const word stateName,
        const label idxI,
        label comp=-1
    );
    
    /// get global adjoint index for a given state name, cell/face indxI and its component (optional, only for vector states)
    label getGlobalAdjointStateIndex
    (
        const word stateName,
        const label idxI,
        label comp=-1
    );

    /// get local segregated adjoint index for a given state name, cell/face indxI and its component (optional, only for vector states)
    label getLocalSegregatedAdjointStateIndex
    (
        const word stateName,
        const label idxI,
        label comp=-1
    );
    
    /// get global segregated adjoint index for a given state name, cell/face indxI and its component (optional, only for vector states)
    label getGlobalSegregatedAdjointStateIndex
    (
        const word stateName,
        const label idxI,
        label comp=-1
    );

    /// get global Xv index for a given point index and coordinate component (x, y, or z)
    label getGlobalXvIndex
    (
        const label idxPoint,
        const label idxCoord
    );

    /// get local Xv index for a given point index and coordinate component (x, y, or z)
    label getLocalXvIndex
    (
        const label idxPoint,
        const label idxCoord
    );
    
    /// calculate stateLocalIndexOffset
    void initializeStateLocalIndexOffset
    (
        HashTable<label>& offset
    );

    /// the accumulated phi indexing offset for cell-by-cell indexing
    labelList phiAccumulatdOffset;

    /// phi local indexing offset for cell-by-cell indexing
    labelList phiLocalOffset;

    /// owner cell of a given face
    labelList faceOwner;
    
    /// set adjoint state unique ID: adjStateID
    void initializeAdjStateID(HashTable<label>& adjStateID);
    
    /// get the objective function geometry information from the adjIO object
    List<word> getObjFuncGeoInfo(const word objFunc);

    /// get the number of objective function geometry elements, this could be the total number of discrete surfaces for all the design surfaces, or total cells for all the user-defined volumes. 
    label getNLocalObjFuncGeoElements(const word objFunc);
    
    /// compute local lists such as adjStateName4LocalAdjIdx and  cellIFaceI4LocalAdjIdx;
    void initializeLocalIdxLists();

    /// compute global list adjStateID4GlobalAdjIdx that stores the stateID for a given globalAdjIndx
    void calcAdjStateID4GlobalAdjIdx(labelList& adjStateID4GlobalAdjIdx);

    void calcCellIFaceI4GlobalAdjIdx(scalarList& cellIFaceI4GlobalAdjIdx);
    
    /// given a boundary patchI and faceI, return its local boundary face index
    label BFacePatchIFaceI2LocalIndex(const label patchI, const label faceI);
    
    /// given a local boundary face index, return its boundary patchI and faceI
    void BFaceLocalIndex2PatchIFaceI
    (
        const label idxI,
        label& patchIdx,
        label& faceIdx
    );
    
    /// write adjoint indexing information to disk for debugging
    void writeAdjointIndexing();
    
    /// how many non-zeros in the petsc matrix
    void getMatNonZeros
    (
        Mat matIn,
        label& maxCols, 
        scalar& allNonZeros
    );
    
    /// print matrix non-zero information (e.g., diag/off-diag ratio)
    void printMatChars(Mat matIn);

    /// initialize globalObjFuncGeoNumbering for a given objective function   
    void initializeObjFuncGeoNumbering(const word objFunc);

    /// delete globalObjFuncGeoNumbering
    void deleteObjFuncGeoNumbering();
    
    /// check if a patch is a user-defined patch
    label isUserDefinedPatch(word geoInfo);

    /// check if a volume is user-defined volume
    label isUserDefinedVolume(word geoInfo);

    /// calculate the face indices for all the user-defined patches: faceIdx4UserDefinedPatches
    void calcFaceIndx4UserDefinedPatches();

    /// calculate the cell indices for all the user-defined volumes: cellIdx4UserDefinedVolumes
    void calcCellIndx4UserDefinedVolumes();
    
    ///  face indices for all the user-defined patches
    HashTable< labelList > faceIdx4UserDefinedPatches;

    ///  cell indices for all the user-defined volumes
    HashTable< labelList > cellIdx4UserDefinedVolumes;

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
