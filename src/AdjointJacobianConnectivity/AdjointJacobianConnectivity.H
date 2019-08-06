/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Handle Jacobian connectivity

\*---------------------------------------------------------------------------*/

#ifndef AdjointJacobianConnectivity_H
#define AdjointJacobianConnectivity_H

#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "UList.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "cyclicAMIPolyPatch.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointJacobianConnectivity Declaration
\*---------------------------------------------------------------------------*/

class AdjointJacobianConnectivity
{

private:

    //- Disallow default bitwise copy construct
    AdjointJacobianConnectivity(const AdjointJacobianConnectivity&);

    //- Disallow default bitwise assignment
    void operator=(const AdjointJacobianConnectivity&);

protected:

    const fvMesh& mesh_;
    const AdjointIO& adjIO_;
    const AdjointSolverRegistry& adjReg_;
    AdjointRASModel& adjRAS_;
    AdjointIndexing& adjIdx_;
    
    /// matrix to store boundary connectivity levels for state Jacobians
    Mat stateBoundaryCon_;

    /// matrix to store boundary connectivity ID for state Jacobians
    Mat stateBoundaryConID_;

    /// matrix to store cyclicAMI boundary connectivity levels for state Jacobians
    Mat stateCyclicAMICon_;

    /// matrix to store cyclicAMI boundary connectivity ID for state Jacobians
    Mat stateCyclicAMIConID_;

    /// matrix to store boundary connectivity levels for Xv Jacobians
    Mat xvBoundaryCon_;

    /// neibough face global index for a given local boundary face
    labelList neiBFaceGlobalCompact_;

    /// cyclicAMI face global indices for a given local boundary face
    List< List<label> > cyclicAMIBFaceGlobalCompact_;
    
    /// dRdW connectivity mat
    Mat dRdWCon_;

    /// dRdWPC connectivity mat
    Mat dRdWConPC_;

    /// dRdXv connectivity mat
    Mat dRdXvCon_;

    /// \name on- and off-diagonal preallocation sizes for each row of dRdW, dRdWT, and dRdXv
    //@{
    Vec dRdWTPreallocOn_, dRdWTPreallocOff_;
    Vec dRdWPreallocOn_, dRdWPreallocOff_;
    Vec dRdXvPreallocOn_, dRdXvPreallocOff_;
    //@}

    /// dRdW colors
    Vec dRdWColors_;
    
    /// dRdW colored columns. It contains the global column index of perturbed state for the current color
    Vec dRdWColoredColumns_;
    
    /// dRdXv colors
    Vec dRdXvColors_;
    
    /// dRdXv colored columns. It contains the global column index of perturbed Xv for the current color
    Vec dRdXvColoredColumns_;

    /// dFdW connectivity
    Mat dFdWCon_;

    /// dFdW colors
    Vec dFdWColors_;
    
    /// dFdW colored columns. It contains the global column index of perturbed state for the current color
    Vec dFdWColoredColumns_;

    /// dFdXv connectivity
    Mat dFdXvCon_;

    /// dFdXv colors
    Vec dFdXvColors_;
    
    /// dFdXv colored columns. It contains the global column index of perturbed Xv for the current color
    Vec dFdXvColoredColumns_;

    /// table to specify how the states are connected to the residuals for a given solver 
    HashTable< List< List<word> > > adjStateResidualConInfo_;

    /// a backup for AdjointJacobianConnectivity::adjStateResidualConInfo_ when reducing the connectivity levels
    HashTable< List< List<word> > > adjStateResidualConInfoBK_;

    /// does this processor has AMI patches?
    label procHasAMI_;
    
public:

    //- Runtime type information
    TypeName("AdjointJacobianConnectivity");
    
    // Declare run-time constructor selection table
    declareRunTimeSelectionTable
    (
        autoPtr,
        AdjointJacobianConnectivity,
        dictionary,
        (
            const fvMesh& mesh,
            const AdjointIO& adjIO,
            const AdjointSolverRegistry& adjReg,
            AdjointRASModel& adjRAS,
            AdjointIndexing& adjIdx
        ),
        (mesh,adjIO,adjReg,adjRAS,adjIdx)
    );

    // Constructors

    //- Construct from components
    AdjointJacobianConnectivity
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
        
    );
    
    // Selectors

    //- Return a reference to the selected model
    static autoPtr<AdjointJacobianConnectivity> New
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx
    );

    //- Destructor
    virtual ~AdjointJacobianConnectivity()
    {}
    
    // public variables
    /// number of dRdW colors
    label ndRdWColors;

    /// number of dRdW colors
    label ndRdXvColors;
    
    /// number of dFdW colors
    label ndFdWColors;

    /// number of dFdXv colors
    label ndFdXvColors;
    
    
    // Member functions
    
    /// calculate AdjointJacobianConnectivity::neiBFaceGlobalCompact_
    void calcNeiBFaceGlobalCompact();

    /// calculate AdjointJacobianConnectivity::cyclicAMIBFaceGlobalCompact_
    void calcCyclicAMIBFaceGlobalCompact();
    
    /// calculate AdjointJacobianConnectivity::stateBoundaryCon_
    void setupStateBoundaryCon();
    
    /// calculate AdjointJacobianConnectivity::stateBoundaryConID_
    void setupStateBoundaryConID();

    /// calculate AdjointJacobianConnectivity::stateBoundaryCon_
    void setupStateCyclicAMICon();
    
    /// calculate AdjointJacobianConnectivity::stateBoundaryConID_
    void setupStateCyclicAMIConID();

    /// combine the connectivity for all state con mats
    void combineAllStateCons();

    /// calculate AdjointJacobianConnectivity::xvBoundaryCon_
    void setupXvBoundaryCon();

    /// add val to the gRow row in conMat, the column indice are the state (given by stateName) for a given cellI 
    void addConMatCell
    (
        Mat conMat, 
        const label gRow,
        const label cellI,
        const word stateName, 
        const PetscScalar val
    );

    /// add val to gRow row in conMat, column indice are all points that are belonged a given cellI 
    void addConMatCellXv
    (
        Mat conMat, 
        const label gRow,
        const label cellI,
        const PetscScalar val
    );

    /// add val to gRow row in conMat, column indice are the neighbouring states (given by stateName) for a given cellI 
    void addConMatNeighbourCells
    (
        Mat conMat, 
        const label gRow,
        const label cellI,
        const word stateName, 
        const PetscScalar val
    );

    /// add val to gRow row in conMat, column indice are the neighbouring Xv for a given cellI 
    void addConMatNeighbourCellsXv
    (
        Mat conMat, 
        const label gRow,
        const label cellI,
        const PetscScalar val
    );
    
    /// add val to gRow row in conMat, column indice are the faces (given by stateName) for a given cellI 
    void addConMatCellFaces
    (
        Mat conMat,
        const label gRow, 
        const label cellI, 
        const word stateName, 
        const PetscScalar val
    );
    
    /// add the column index of the (iner-proc) connected states and faces to conMat, given a local face index 
    void addBoundaryFaceConnections
    (
        Mat conMat,
        label gRow, 
        label cellI,
        labelList v,
        List< List<word> > connectedStates,
        label addFaces
    );

    /// add the column index of the AMI connected states and faces to conMat, given a local face index 
    void addCyclicAMIFaceConnections
    (
        Mat conMat,
        label gRow, 
        label cellI,
        labelList v,
        List< List<word> > connectedStates,
        label addFaces
    );

    /// add the column index of the (iner-proc) connected Xv to conMat, given a local face index 
    void addBoundaryFaceConnectionsXv
    (
        Mat conMat,
        label gRow,
        label cellI,
        labelList v
    );
    
    /// combine stateBoundaryCon and stateBoundaryConTmp, this is to ensure including all connected states for parallel cases
    void combineStateBndCon
    (
        Mat* stateBoundaryCon,
        Mat* stateBoundaryConTmp
    );

    /// combine xvBoundaryCon and xvBoundaryConTmp, this is to ensure including all connected Xv for parallel cases
    void combineXvBndCon
    (
        Mat* xvBoundaryCon,
        Mat* xvBoundaryConTmp
    );
    
    /// given a local face index, return the local index of the coupled boundary face
    PetscInt getLocalCoupledBFaceIndex(label localFaceI);

    /// given a local face index, return the index of cyclicAMI boundary face
    PetscInt getLocalCyclicAMIFaceIndex(label localFaceI);
    
    /// initialize petsc vectors
    void initializePetscVecs();
    
    /// initialize AdjointJacobianConnectivity::dRdWCon_
    void initializedRdWCon();

    /// initialize AdjointJacobianConnectivity::dRdWConPC_
    void initializedRdWConPC();

    /// initialize AdjointJacobianConnectivity::dRdXvCon_
    void initializedRdXvCon();
    
    /// delete AdjointJacobianConnectivity::dRdWCon_
    void deletedRdWCon()
    {
        MatDestroy(&dRdWCon_);
        //VecDestroy(&dRdWColoredColumns_);
    }

     /// delete AdjointJacobianConnectivity::dRdWConPC_
    void deletedRdWConPC()
    {
        MatDestroy(&dRdWConPC_);
    }

    /// delete AdjointJacobianConnectivity::dRdXvCon_
    void deletedRdXvCon()
    {
        MatDestroy(&dRdXvCon_);
    }
    
    /// delete AdjointJacobianConnectivity::dFdWCon_
    void deletedFdWCon()
    {
        MatDestroy(&dFdWCon_);
        //VecDestroy(&dFdWColoredColumns_);
    }

    /// delete AdjointJacobianConnectivity::dFdXvCon_
    void deletedFdXvCon()
    {
        MatDestroy(&dFdXvCon_);
    }
    
    /// compute preallocation vectors
    void preallocateJacobianMatrix
    (
        Mat dRMat,
        Vec preallocOnProc,
        Vec preallocOffProc
    );
    
    /// preallocate memory for dRdW using the computed preallocation vectors
    void preallocatedRdW(Mat dRMat,label transposed);

    /// preallocate memory for dRdXv using the computed preallocation vectors
    void preallocatedRdXv(Mat dRMat);
    
    /// a high-level function to add connected state column indices to the connectivity matrix 
    void addStateConnections
    (
        Mat connections, 
        label cellI,
        label connectedLevelLocal,
        wordList connectedStatesLocal,
        List< List<word> > connectedStateInterProc,
        label addFace
    );

    /// a high-level function to add connected Xv column indices to the connectivity matrix 
    void addXvConnections
    (
        Mat connections, 
        label cellI,
        label connectedLevelLocal,
        label connectedLevelInterProc
    );
    
    
    /// add value 1 for the colume idx to conMat
    void setConnections(Mat conMat,label idx);
    
    /// allocate connectedState matrix
    void createConnectionMat(Mat *connectedStates);
    
    /// assign 1 to all non-zero elements for AdjointJacobianConnectivity::dRdWCon_
    void setupdRdWCon(label isPrealloc,label isPC=0);

    /// assign 1 to all non-zero elements for AdjointJacobianConnectivity::dRdXvCon_
    void setupdRdXvCon(label isPrealloc);
    
    /// assign 1 to all non-zero elements for AdjointJacobianConnectivity::dFdWCon_ or AdjointJacobianConnectivity::dFdXvCon_
    void setupObjFuncCon(const word objFunc,const word mode);
    
    /// compute preallocation vectors
    void allocateJacobianConnections
    (
        Vec preallocOnProc, 
        Vec preallocOffProc,
        Vec preallocOnProcT, 
        Vec preallocOffProcT,
        Mat connections,
        label row
    );

    void allocateJacobianConnectionsXv
    (
        Vec preallocOnProc, 
        Vec preallocOffProc,
        Mat connections,
        label row
    );
    
    /// assign values in connections to a specific row idxI in conMat
    void setupJacobianConnections
    (
        Mat conMat, 
        Mat connections, 
        PetscInt idxI
    );
    
    /// compute graph coloring for dRdW: AdjointJacobianConnectivity::dRdWColors_
    void calcdRdWColoring();

    /// read colors for dRdW
    void readdRdWColoring();

    /// compute graph coloring for dRdXv: AdjointJacobianConnectivity::dRdXvColors_
    void calcdRdXvColoring();

    /// read colors for dRdXv
    void readdRdXvColoring();

    /// a parallel distance-2 graph coloring function
    void parallelD2Coloring
    (
        Mat conMat,
        Vec colors_,
        PetscInt &nColors
    );
    
    /// number of non-zero elements in a matrix
    void getMatNonZeros
    (
        Mat conMat,
        label& maxCols, 
        scalar& allNonZeros
    );

    /// find the index of a prescribed value in an array
    label find_index
    (
        label target,
        label start, 
        label size, 
        label *valArray
    );
    
    /// check if there is non-colored columns
    void coloringComplete
    (
        Vec colors_,
        label& notColored
    );
    
    /// validate if there is coloring conflict
    void validateColoring(Mat conMat, Vec colors);
    
    /// given a cell or face index, return the state color
    label getStateColor
    (
        const word mode,
        const word stateName,
        const label idxI,
        label comp=-1
    );

    /// given point and coord indices, return the Xv color
    label getXvColor
    (
        const word mode,
        const label pointI,
        const label coordI
    );
    
    /// get the number of dRdW colors
    label getNdRdWColors();

    /// get the number of dRdXv colors
    label getNdRdXvColors();

    /// get the number of dFdW colors
    label getNdFdWColors();

    /// get the number of dFdXv colors
    label getNdFdXvColors();

    /// calculate AdjointJacobianConnectivity::dRdWColoredColumns_
    void calcdRdWColoredColumns(label currColor, label isPC);

    /// calculate AdjointJacobianConnectivity::dRdXvColoredColumns_
    void calcdRdXvColoredColumns(label currColor);

    /// calculate AdjointJacobianConnectivity::dFdWColoredColumns_
    void calcdFdWColoredColumns(label currColor);

    /// calculate AdjointJacobianConnectivity::dFdXvColoredColumns_
    void calcdFdXvColoredColumns(label currColor);

    /// return AdjointJacobianConnectivity::dRdWColoredColumns_
    Vec getdRdWColoredColumns()
    {
        return dRdWColoredColumns_;
    }

    /// return AdjointJacobianConnectivity::dRdXvColoredColumns_
    Vec getdRdXvColoredColumns()
    {
        return dRdXvColoredColumns_;
    }

    /// return AdjointJacobianConnectivity::dFdWColoredColumns_
    Vec getdFdWColoredColumns()
    {
        return dFdWColoredColumns_;
    }

    /// return AdjointJacobianConnectivity::dFdXvColoredColumns_
    Vec getdFdXvColoredColumns()
    {
        return dFdXvColoredColumns_;
    }
    
    /// a general function to calculate coloredColumn vecs
    void calcColoredColumns
    (
        label currColor,
        Mat conMat, 
        Vec column,
        Vec colors
    );
    
    /// initialize AdjointJacobianConnectivity::dFdWCon_
    void initializedFdWCon(const word objFunc);

    /// initialize AdjointJacobianConnectivity::dFdXvCon_
    void initializedFdXvCon(const word objFunc);

    /// add connectivity info into AdjointJacobianConnectivity::dFdWCon_ for a given objective function
    void addObjFuncConnectivity
    ( 
        word objFunc,
        List< List<word> > objFuncConInfo 
    );

    /// add connectivity info into AdjointJacobianConnectivity::dFdXvCon_ for a given objective function
    void addObjFuncConnectivityXv
    ( 
        word objFunc,
        List< List<word> > objFuncConInfo 
    );
    
    /// calculate AdjointJacobianConnectivity::dFdWColors_
    void calcdFdWColoring(const word objFunc);

    /// calculate AdjointJacobianConnectivity::dFdXvColors_
    void calcdFdXvColoring(const word objFunc);

    /// read AdjointJacobianConnectivity::dFdWColors_
    void readdFdWColoring(const word objFunc);

    /// read AdjointJacobianConnectivity::dFdXvColors_
    void readdFdXvColoring(const word objFunc);

    /// reduce the connectivity level for AdjointJacobianConnectivity::dRdWCon_
    void reduceAdjStateResidualConLevel();

    /// restore the connectivity level for AdjointJacobianConnectivity::dRdWCon_
    void restoreAdjStateResidualConLevel();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
