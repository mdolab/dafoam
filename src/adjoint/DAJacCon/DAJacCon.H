/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Compute Jacobian connectivity and coloring

\*---------------------------------------------------------------------------*/

#ifndef DAJacCon_H
#define DAJacCon_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "DAUtility.H"
#include "DAOption.H"
#include "DAIndex.H"
#include "DAModel.H"
#include "DAStateInfo.H"
#include "syncTools.H"
#include "DAObjFunc.H"
#include "DAColoring.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DAJacCon Declaration
\*---------------------------------------------------------------------------*/

class DAJacCon
{

private:
    /// Disallow default bitwise copy construct
    DAJacCon(const DAJacCon&);

    /// Disallow default bitwise assignment
    void operator=(const DAJacCon&);

protected:
    /// the name of the jacCon matrix
    const word modelType_;

    /// fvMesh
    const fvMesh& mesh_;

    /// DAOption object
    const DAOption& daOption_;

    /// DAModel object
    const DAModel& daModel_;

    /// DAIndex object
    const DAIndex& daIndex_;

    /// DAColoring object
    DAColoring daColoring_;

    /// DAField object
    DAField daField_;

    /// the regState_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// matrix to store boundary connectivity levels for state Jacobians
    Mat stateBoundaryCon_;

    /// matrix to store boundary connectivity ID for state Jacobians
    Mat stateBoundaryConID_;

    /// neibough face global index for a given local boundary face
    labelList neiBFaceGlobalCompact_;

    /// Jacobian connectivity mat
    Mat jacCon_;

    /// jacCon matrix colors
    Vec jacConColors_;

    /// number of jacCon colors
    label nJacConColors_;

    /// initialize state boundary connection
    void initializeStateBoundaryCon();

    /// calculate DAJacCon::neiBFaceGlobalCompact_
    void calcNeiBFaceGlobalCompact(labelList& neiBFaceGlobalCompact);

    /// given a local face index, return the local index of the coupled boundary face
    label getLocalCoupledBFaceIndex(const label localFaceI) const;

    /// calculate DAJacCon::stateBoundaryCon_
    void setupStateBoundaryCon(Mat* stateBoundaryCon);

    /// calculate DAJacCon::stateBoundaryConID_
    void setupStateBoundaryConID(Mat* stateBoundaryConID);

    /// allocate connectedState matrix
    void createConnectionMat(Mat* connectedStates);

    /// a high-level function to add connected state column indices to the connectivity matrix
    void addStateConnections(
        Mat connections,
        const label cellI,
        const label connectedLevelLocal,
        const wordList connectedStatesLocal,
        const List<List<word>> connectedStateInterProc,
        const label addFace);

    /// add value 1 for the colume idx to conMat
    void setConnections(
        Mat conMat,
        const label idx) const;

    /// assign values in connections to a specific row idxI in conMat
    void setupJacobianConnections(
        Mat conMat,
        Mat connections,
        const label idxI);

    /// combine stateBoundaryCon and stateBoundaryConTmp, this is to ensure including all connected states for parallel cases
    void combineStateBndCon(
        Mat* stateBoundaryCon,
        Mat* stateBoundaryConTmp);

    /// add val to the gRow row in conMat, the column indice are the state (given by stateName) for a given cellI
    void addConMatCell(
        Mat conMat,
        const label gRow,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// add val to gRow row in conMat, column indice are the neighbouring states (given by stateName) for a given cellI
    void addConMatNeighbourCells(
        Mat conMat,
        const label gRow,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// add val to gRow row in conMat, column indice are the faces (given by stateName) for a given cellI
    void addConMatCellFaces(
        Mat conMat,
        const label gRow,
        const label cellI,
        const word stateName,
        const PetscScalar val);

    /// add the column index of the (iner-proc) connected states and faces to conMat, given a local face index
    void addBoundaryFaceConnections(
        Mat conMat,
        const label gRow,
        const label cellI,
        const labelList v,
        const List<List<word>> connectedStates,
        const label addFaces);

    /// clear members in DAJacCon
    void clearDAJacConMembers()
    {
        // Need to clean up all the matrices
        MatDestroy(&jacCon_);
        MatDestroy(&stateBoundaryCon_);
        MatDestroy(&stateBoundaryConID_);
        VecDestroy(&jacConColors_);
        stateInfo_.clear();
        neiBFaceGlobalCompact_.clear();
    }

    /// check if there is special boundary conditions that need special treatment in jacCon_
    void checkSpecialBCs();

    /// a vector to show whether a state is connected to a pressureInletVelocity boundary face (3 level max)
    Vec isPIVBCState_;

    /// function used to add connectivity for pressureInletVelocity
    void setPIVVec(
        Vec iSPIV,
        const label idxI);

    /// add connectivity phi for pressureInletVelocity
    label addPhi4PIV(
        const word stateName,
        const label idxI,
        const label comp = -1);

public:
    /// Runtime type information
    TypeName("DAJacCon");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DAJacCon,
        dictionary,
        (const word modelType,
         const fvMesh& mesh,
         const DAOption& daOption,
         const DAModel& daModel,
         const DAIndex& daIndex),
        (modelType,
         mesh,
         daOption,
         daModel,
         daIndex));

    // Constructors

    //- Construct from components
    DAJacCon(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DAJacCon> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAJacCon()
    {
    }

    // Member functions

    /// clear members in parent and child objects
    virtual void clear() = 0;

    /// calculate the preallocation vector for initializing the JacCon mat, if necessary
    virtual void setupJacConPreallocation(const dictionary& options);

    /// initialize the state Jacobian connectivity matrix
    virtual void initializeJacCon(const dictionary& options) = 0;

    /// assign 1 to all non-zero elements for the Jacobian connecitivyt matrix
    virtual void setupJacCon(const dictionary& options) = 0;

    /// get the number of JacCon colors
    label getNJacConColors() const
    {
        return nJacConColors_;
    }

    /// preallocate dRdW matrix using the preallocVec
    virtual void preallocatedRdW(
        Mat dRMat,
        const label transposed) const;

    /// compute graph coloring for Jacobian connectivity matrix
    void calcJacConColoring(const word postFix = "");

    /// read colors for JacCon
    void readJacConColoring(const word postFix = "");

    /// whether the coloring file exists
    label coloringExists(const word postFix = "") const;

    /// return DAJacCon::jacConColors_
    Vec getJacConColor() const
    {
        return jacConColors_;
    }

    /// calculate the colored column vector
    void calcColoredColumns(
        const label colorI,
        Vec coloredColumn) const;

    /// assign values for the objective function vector based on the face and cell value lists
    virtual void setObjFuncVec(
        scalarList objFuncFaceValues,
        scalarList objFuncCellValues,
        Vec objFuncVec) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
