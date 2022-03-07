/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Handel state indexing in serial and in parallel

\*---------------------------------------------------------------------------*/

#ifndef DAIndex_H
#define DAIndex_H

#include "fvOptions.H"
#include "surfaceFields.H"
#include "DAOption.H"
#include "DAUtility.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "globalIndex.H"
#include "DAMacroFunctions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAIndex Declaration
\*---------------------------------------------------------------------------*/

class DAIndex
{

private:
    /// Disallow default bitwise copy construct
    DAIndex(const DAIndex&);

    /// Disallow default bitwise assignment
    void operator=(const DAIndex&);

protected:
    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// Foam::DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// the StateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// write the adjoint indexing for debugging
    void writeAdjointIndexing();

public:
    /// Constructors
    DAIndex(
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel);

    /// Destructor
    virtual ~DAIndex()
    {
    }

    // Members

    // adjoint indexing info
    /// list of adjoint state names for a specific solver
    List<word> adjStateNames;

    /// hash table of adjoint state types, e.g., volVectorState for a given state name
    HashTable<word> adjStateType;

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

    /** hash table of  local state variable index offset, see initializeStateLocalIndexOffset for definition. 
        This will be used in determing the local indexing  for adjoint states. It differs depending on whether
        we use state-by-state or cell-by-cell ordering
    */
    HashTable<label> stateLocalIndexOffset;

    /// a unique number ID for adjoint states, it depends on the sequence of adjStateNames
    HashTable<label> adjStateID;

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

    /// local adjoint boundary states that do not contain phis
    label nLocalAdjointBoundaryStates;

    /// number of state variables for volScalarField
    label nVolScalarStates;

    /// number of state variables for volVectorField
    label nVolVectorStates;

    /// number of state variables for surfaceScalarField
    label nSurfaceScalarStates;

    /// number of model states, NOTE: they are counted separately
    label nModelStates;

    /// \name global indexing variables, they will be used to transfer local and global indices
    //@{
    globalIndex globalAdjointStateNumbering;
    globalIndex globalCellNumbering;
    globalIndex globalCellVectorNumbering; // similar to globalCellNumbering but has 3 components per cell
    globalIndex globalFaceNumbering;
    globalIndex globalCoupledBFaceNumbering;
    globalIndex globalObjFuncGeoNumbering;
    globalIndex globalXvNumbering;
    globalIndex globalVolScalarFieldNumbering;
    globalIndex globalVolVectorFieldNumbering;
    globalIndex globalSurfaceScalarFieldNumbering;
    //@}

    /// a list to map the point index from decomposed domains to the original un-decomposed domain
    labelIOList pointProcAddressing;

    /// number of points for the un-decomposed domain
    label nUndecomposedPoints;

    /// for a given face index, return whether this face is a coupled boundary face
    labelList isCoupledFace;

    /// given a local adjoint state index, return its state name
    wordList adjStateName4LocalAdjIdx;

    /// given a local adjoint state index, return its cell/face index
    scalarList cellIFaceI4LocalAdjIdx;

    /// owner cell of a given face
    labelList faceOwner;

    /// the accumulated phi indexing offset for cell-by-cell indexing
    labelList phiAccumulatdOffset;

    /// phi local indexing offset for cell-by-cell indexing
    labelList phiLocalOffset;

    // Member functions

    /// calculate stateLocalIndexOffset
    void calcStateLocalIndexOffset(HashTable<label>& offset);

    /// set adjoint state unique ID: adjStateID
    void calcAdjStateID(HashTable<label>& adjStateID);

    /// compute local lists such as adjStateName4LocalAdjIdx and  cellIFaceI4LocalAdjIdx;
    void calcLocalIdxLists(
        wordList& adjStateName4LocalAdjIdx,
        scalarList& cellIFaceI4LocalAdjIdx);

    /// get local adjoint index for a given state name, cell/face indxI and its component (optional, only for vector states)
    label getLocalAdjointStateIndex(
        const word stateName,
        const label idxI,
        const label comp = -1) const;

    /// get global adjoint index for a given state name, cell/face indxI and its component (optional, only for vector states)
    label getGlobalAdjointStateIndex(
        const word stateName,
        const label idxI,
        const label comp = -1) const;

    /// get global Xv index for a given point index and coordinate component (x, y, or z)
    label getGlobalXvIndex(
        const label idxPoint,
        const label idxCoord) const;

    /// get local Xv index for a given point index and coordinate component (x, y, or z)
    label getLocalXvIndex(
        const label idxPoint,
        const label idxCoord) const;

    /// get local cell index for a local cell index
    label getLocalCellIndex(const label cellI) const;

    /// get global cell index for a local cell index
    label getGlobalCellIndex(const label cellI) const;

    /// get local cell index for a local cell vector index
    label getLocalCellVectorIndex(
        const label cellI,
        const label comp) const;

    /// get global cell index for a local cell vector index
    label getGlobalCellVectorIndex(
        const label cellI,
        const label comp) const;

    /// get local face index for a local face index
    label getLocalFaceIndex(const label faceI) const;

    /// get global face index for a local face index
    label getGlobalFaceIndex(const label faceI) const;

    /// compute global list adjStateID4GlobalAdjIdx that stores the stateID for a given globalAdjIndx
    void calcAdjStateID4GlobalAdjIdx(labelList& adjStateID4GlobalAdjIdx) const;

    /// print all the sizes and indices
    void printIndices() const
    {
        //Info<<"adjStateNames"<<adjStateNames<<endl;
        Info << "Adjoint States: " << adjStateType << endl;

        // Print relevant sizes to screen
        Info << "Global Cells: " << nGlobalCells << endl;
        Info << "Global Faces: " << nGlobalFaces << endl;
        Info << "Global Xv: " << nGlobalXv << endl;
        Info << "Undecomposed points: " << nUndecomposedPoints << endl;
        Info << "Global Adjoint States: " << nGlobalAdjointStates << endl;
    }

    /// print petsc matrix statistics such as the max on/off diagonral ratio
    void printMatChars(const Mat matIn) const;

    /// get number of non zeros for a matrix
    void getMatNonZeros(
        const Mat matIn,
        label& maxCols,
        scalar& allNonZeros) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
