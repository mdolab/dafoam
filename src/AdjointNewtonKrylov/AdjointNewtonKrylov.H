/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

    Description:
    Newton-Krylov method to solve the flow

\*---------------------------------------------------------------------------*/

#ifndef AdjointNewtonKrylov_H
#define AdjointNewtonKrylov_H

#include <petscsnes.h>
#include "fvMesh.H"
#include "fvOptions.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointObjectiveFunction.H"
#include "AdjointDerivative.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointNewtonKrylov Declaration
\*---------------------------------------------------------------------------*/

class AdjointNewtonKrylov
{

private:

    fvMesh& mesh_;
    const AdjointIO& adjIO_;
    const AdjointSolverRegistry& adjReg_;
    AdjointRASModel& adjRAS_;
    AdjointObjectiveFunction& adjObj_;
    AdjointDerivative& adjDev_;
    const objectRegistry& db_;

protected:

    /// number of function evaluation
    label nFuncEvals_;

    /// state variable vectors for the current step
    Vec wVec0_;

    /// residual vectors for the current step
    Vec rVec0_;

    /// direction vector, NOTE: dWVec_=wVec0_-wVec1 instead of wVec1_-wVec0
    Vec dWVec_;

    /// state variable vectors for the new step
    Vec wVec1_;

    /// state variable vectors for the new step
    Vec rVec1_;

    ///
    Vec phiVec0_;
    Vec dPhiVec_;

    /// hash table of nk state types, e.g., volVectorState for a given state name
    HashTable<word> nkStateType_;

    /// list of nk state names for a specific solver
    List<word> nkStateNames_;

    /// hash table of  local state variable index offset,
    HashTable<label> nkStateLocalIndexOffset_;

    /// number of local NK states 
    label nLocalNKStates_;

    /// name global indexing variables, they will be used to transfer local and global indices
    globalIndex globalNKStateNumbering_;

    /// given a local NK state index, return its state name
    wordList stateName4LocalNKIdx_;

    /// given a local NK state index, return its cell/face index
    scalarList cellIFaceI4LocalNKIdx_;

    /** given a local boundary face index (starting with zero for the first boundary face, 
        and ends with the last boundary faces), return its boundary patchI, 
        e.g., for face index = 100, its patchI = 2
    */
    labelList bFacePatchI_;

    /** given a local boundary face index (starting with zero for the first boundary face, 
        and ends with the last boundary faces), return its boundary faceI associated with a boundary patchI,
        e.g., for face index = 100, its patchI = 2, faceI=20
    */
    labelList bFaceFaceI_;

    /// initialize indexing stuff for NK 
    void initializeIndexing();

    /// initialize petsc vectors
    void initializePetscVectors();

    /// get local NK index for a given state name, cell indxI and its component (optional, only for vector states)
    label getLocalNKStateIndex
    (
        const word stateName,
        const label idxI,
        label comp=-1
    );

    /// get global NK index for a given state name, cell indxI and its component (optional, only for vector states)
    label getGlobalNKStateIndex
    (
        const word stateName,
        const label idxI,
        label comp=-1
    );

    /// generate global index numbering for local-global index transferring
    globalIndex genGlobalIndex(const label index);

    /// compute PC using the OpenFOAOM built-in fvMatrix coefficients
    void calcdRdWPCFast
    (
        Mat jacMat
    );

    void NKCalcResiduals(Vec wVec,Vec rVec);

    void calcPCMat4NK(Mat* PCMat,label initMat);

    scalar getEWTol(scalar norm, scalar oldNorm, scalar rTolLast);

    scalar NKLineSearch
    (
        const Mat dRdW,
        const Vec wVec,
        const Vec rVec,
        const Vec dWVec,
        Vec wVecNew,
        Vec rVecNew
    );

    scalar NKLineSearchNew
    (
        const Vec wVec,
        const Vec rVec,
        const Vec dWVec,
        Vec wVecNew,
        Vec rVecNew
    );

    label checkNegativeTurb();

    void printConvergenceInfo
    (
        word mode,
        HashTable<scalar> objFuncs,
        label mainIter=0,
        label nFuncEval=0,
        word solverType="",
        scalar step=1.0,
        scalar linRes=1.0,
        scalar CFL=1.0,
        scalar runTime=1.0,
        scalar turbNorm=1.0,
        scalar phiNorm=1.0,
        scalar totalNorm=1.0
    );

    /// general function to set vecX to/from states/residual variables
    void NKSetVecs
    (
        Vec vecX, 
        word mode, 
        scalar scaleFactor, 
        word postFix
    );

    /// set or get phiVec
    void NKSetPhiVec
    (
        Vec vecX, 
        word mode,
        scalar scaleFactor, 
        word postFix
    );

    /// special function to only calculate norm for existing residuals without recomputing/updating them
    scalar getResNorm(word mode);

    /// set normalization in the AD adjoint solution function
    void setNormalizeStatesScaling2Vec(Vec vecY);

    /// local 1/dt for ANK transient term
    Vec dTInv_;

    /// CFL for ANK
    scalar ANKCFL_;

    void setdTInv(Vec dTInv);

    void NKAddUnsteadyTerms(Vec unsteadyVec,Vec rVec);

    void NKCalcResidualsUnsteady(Vec wVec,Vec dWVec,scalar step,Vec rVec);

    scalar updateANKCFL
    (
        scalar CFL0,
        scalar totalResNormFS,
        scalar totalResNorm,
        scalar totalResNormOld,
        scalar stepSize
    );

public:

    // Constructors

    //- Construct from components
    AdjointNewtonKrylov
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointObjectiveFunction& adjObj,
        AdjointDerivative& adjDev
    );

    //- Destructor
    virtual ~AdjointNewtonKrylov(){};
    
    // Member Functions
    static PetscErrorCode FormFunction(void* ctx,Vec wVec,Vec rVec);

    static PetscErrorCode FormFunctionANK(void* ctx,Vec wVec,Vec rVec);

    void solve();

    void solveANK();

    void solveNK();


};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
