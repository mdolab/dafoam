/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Compute derivatives using the discrete adjoint method

\*---------------------------------------------------------------------------*/

#ifndef AdjointDerivative_H
#define AdjointDerivative_H

#include <petscksp.h>
#include "runTimeSelectionTables.H"
#include "fvOptions.H"
#include "surfaceFields.H"
#include "macroFunctions.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointObjectiveFunction.H"

#ifdef CompressibleFlow
    #include "fluidThermo.H"
#endif 

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointDerivative Declaration
\*---------------------------------------------------------------------------*/

class AdjointDerivative
{

private:

    //- Disallow default bitwise copy construct
    AdjointDerivative(const AdjointDerivative&);

    //- Disallow default bitwise assignment
    void operator=(const AdjointDerivative&);

protected:
    
    fvMesh& mesh_;
    const AdjointIO& adjIO_;
    const AdjointSolverRegistry& adjReg_;
    AdjointRASModel& adjRAS_;
    AdjointIndexing& adjIdx_;
    AdjointJacobianConnectivity& adjCon_;
    AdjointObjectiveFunction& adjObj_;
#ifdef CompressibleFlow
    fluidThermo& thermo_;
#endif    
    const objectRegistry& db_;
    
    /// state Jacobian matrix transposed
    Mat dRdWT_;
    
    /// PC state Jacobian matrix transposed
    Mat dRdWTPC_;

    /// Jacobian for inlet velocity
    Mat dRdUIn_;

    /// Jacobian for FFD points
    Mat dRdFFD_;

    /// Jacobian for volume coordinates Xv
    Mat dRdXv_;

    /// Jacobian for molecular viscosity nu
    Mat dRdVis_;

    /// partial derivative rhs vector of objective function wrt state variable including all the prescribed objective functions
    Mat dFdWAll_;

    /// partial derivative rhs vector of objective function wrt state variable for a given objective function
    Vec dFdW_;

    /// partial derivative rhs vector of objective function wrt FFD points
    Vec dFdFFD_;

    /// partial derivative rhs vector of objective function wrt volume coordinate Xv
    Vec dFdXv_;

    /// partial derivative rhs vector of objective function wrt volume coordinate Xv for undecomposed domain
    Vec dFdXvSerial_;

    /// partial derivative rhs vector of objective function wrt velocity inlet
    Vec dFdUIn_;

    /// partial derivative rhs vector of objective function wrt molecular viscosity nu
    Vec dFdVis_;

    /// adjoint vector; solution of adjoint equation
    Vec psi_;
    
    /// total derivative vector of objective function wrt velocity inlet
    Vec dFdUInTotal_;

    /// total derivative vector of objective function wrt FFD points
    Vec dFdFFDTotal_;

    /// total derivative vector of objective function wrt volume coordinate Xv
    Vec dFdXvTotal_;

    /// total derivative vector of objective function wrt volume coordinate Xv for undecomposed domain
    Vec dFdXvTotalSerial_;

    /// total derivative vector of objective function wrt molecular viscosity nu
    Vec dFdVisTotal_;
    
    /// KSP object of PETSc
    KSP ksp_;
    
    /// a reference point field that contains all the unperturbed volume coordinates
    const pointField pointsRef_;

    /// store the L2 norm for all ResRef, we use this to ensure consistent refRes after perturbing and resetting states
    HashTable<scalar> refResL2Norm_;

    /// Multiple Reference Frame 
    IOMRFZoneList MRF_;
    
public:

    //- Runtime type information
    TypeName("AdjointDerivative");

#ifdef CompressibleFlow    
    // Declare run-time constructor selection table
    declareRunTimeSelectionTable
    (
        autoPtr,
        AdjointDerivative,
        dictionary,
        (
            fvMesh& mesh,
            const AdjointIO& adjIO,
            const AdjointSolverRegistry& adjReg,
            AdjointRASModel& adjRAS,
            AdjointIndexing& adjIdx,
            AdjointJacobianConnectivity& adjCon,
            AdjointObjectiveFunction& adjObj,
            fluidThermo& thermo
        ),
        (mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj,thermo)
    );

    // Constructors

    //- Construct from components
    AdjointDerivative
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj,
        fluidThermo& thermo
    );
    
    // Selectors

    //- Return a reference to the selected model
    static autoPtr<AdjointDerivative> New
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj,
        fluidThermo& thermo
    );

#else

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable
    (
        autoPtr,
        AdjointDerivative,
        dictionary,
        (
            fvMesh& mesh,
            const AdjointIO& adjIO,
            const AdjointSolverRegistry& adjReg,
            AdjointRASModel& adjRAS,
            AdjointIndexing& adjIdx,
            AdjointJacobianConnectivity& adjCon,
            AdjointObjectiveFunction& adjObj
        ),
        (mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj)
    );

    // Constructors

    //- Construct from components
    AdjointDerivative
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj
    );
    
    // Selectors

    //- Return a reference to the selected model
    static autoPtr<AdjointDerivative> New
    (
        fvMesh& mesh,
        const AdjointIO& adjIO,
        const AdjointSolverRegistry& adjReg,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon,
        AdjointObjectiveFunction& adjObj
    );

#endif

    //- Destructor
    virtual ~AdjointDerivative()
    {}

    /// diagonal components of fvMatrix, which fvMatrix to use depends on fvMatrixName in calcResiduals function 
    scalarList fvMatrixDiag;

    /// lower components of fvMatrix, which fvMatrix to use depends on fvMatrixName in calcResiduals function
    scalarList fvMatrixLower;

    /// upper component of fvMatrix, which fvMatrix to use depends on fvMatrixName in calcResiduals function
    scalarList fvMatrixUpper; 
    
    /// main function to compute total derivatives, this is also the main APIs of the libAdjointDerivative*.so for adjoint solvers
    void solve();

    /// initialize AdjointDerivative::dFdW_
    void initializedFdW();

    /// initialize AdjointDerivative::dRdWT_
    void initializedRdW(Mat* jacMat,const label transposed);
    
    /// AdjointDerivative::dRdUIn_
    void initializedRdUIn();

    /// AdjointDerivative::dFdUIn_
    void initializedFdUIn();
    
    /// AdjointDerivative::dRdXv_
    void initializedRdXv();

    /// AdjointDerivative::dFdXv_
    void initializedFdXv();

    /// AdjointDerivative::dRdVis_
    void initializedRdVis();

    /// AdjointDerivative::dFdVis_
    void initializedFdVis();

    /// perturb state variables given a color index
    void perturbStates(const label colorI,const word mode);
    
    /// perturb FFD point based on the perturbed Xv in deltaVolPointMat
    void perturbFFD
    (
        Mat deltaVolPointMat,
        PetscInt iDV,
        pointField& newPoints
    );

    /// perturb volume coordinates
    void perturbXv
    (
        label colorI,
        const word mode,
        pointField& newPoints
    );
    
    /// sync boundary points (this function has been deprecated)
    void syncBoundaryPointCoords(pointField &newPoints);
    
    /// calculate flow residuals
    virtual void calcResiduals
    (
        const label isRef,
        const label isPC,
        const word fvMatrixName="None",
        const label updatePhi=0
    )=0;
    
    /// reset state perturbation
    virtual void updateIntermediateVariables()=0;
    
    /// calculate L2norm, mean, and max flow residuals
    void calcFlowResidualStatistics
    (
        const word mode,
        const label writeRes=0
    );
    
    /// calculate FD-based partial derivatives
    void calcResPartDeriv(const scalar eps,const label isPC);

    /// update boundary values
    void updateStateVariableBCs();
    
    /// calculate AdjointDerivative::dRdWT_
    void calcdRdW(Mat jacMat,const label transposed,const label isPC);

    /// compute AdjointDerivative::dRdWTPC_ or AdjointDerivative::dRdWPC_ using the OpenFOAOM built-in fvMatrix coefficients
    void calcdRdWPCFast
    (
        Mat jacMat,
        const label transposed,
        const label printStatistics=1
    );

    /// assemble AdjointDerivative::dRdWT_
    void assembledRdW4Adjoint();
    
    /// calculate AdjointDerivative::dFdW_
    void calcdFdW(const word objFunc);

    /// assign AdjointDerivative::dFdW_ to AdjointDerivative::dFdWAll_
    void dFdW2dFdWAll(const word objFunc);

    /// assign AdjointDerivative::dFdWAll_ to AdjointDerivative::dFdW_
    void dFdWAll2dFdW(const word objFunc);
    
    /// calculate AdjointDerivative::dRdFFD_
    void calcdRdFFD(Mat dRdFFD);

    /// calculate AdjointDerivative::dRdXv_
    void calcdRdXv();
    
    /// calculate AdjointDerivative::dFdFFD_
    void calcdFdFFD(const word objFunc);
    
    /// calculate AdjointDerivative::dFdXv_
    void calcdFdXv(const word objFunc);
    
    /// copy state variables to reference state variable or vice versa, depending on option
    void copyStates(const word option);
    
    /// assign values to the Jacobian matrices based on the partial derivative values (no color)
    void setJacobianMat
    (
        Mat matIn,
        const label colorI,
        const label transposed
    );
    
    /// assign values to the Jacobian matrices based on the partial derivative values (with color)
    void setJacobianMatColored
    (
        Mat matIn,
        Vec coloredColumns,
        const label transposed,
        const label isPC
    );
    
    /// given a adjoint state local index, return its partial derivative
    scalar adjStateLocalIdx2PartDerivVal(const label localIdx);
    
    /// assign values to the dFdW or dFdXv matrices based on the partial derivative values (no color)
    void setdFVec
    (
        Vec dFVec,
        const word objFunc,
        const label colorI
    );
    
    /// assign values to the dFdW or dFdXv matrices based on the partial derivative values (with color)
    void setdFVecColored
    (
        Vec dFVec,
        Vec coloredColumns,
        const word objFunc
    );
    
    /// solve adjoint equation
    void solveAdjoint(const word objFunc);

    /// solve adjoint equation using the segregated method
    void solveAdjointSegregated(const word objFunc);

    /// set the coupled psi rows that are belonged to stateName to psiS or vice versa depending on mode  
    void setSegregatedVecs
    (
        Vec psi,
        Vec psiS, 
        const word stateName,
        const scalar scale,
        const word mode
    );

    /// under-relax a vector
    void underRelaxVec
    (
        Vec vecNew,
        Vec vecOld,
        scalar alpha
    );

    /// initialize segregated Jacobian matrices
    void initializeSegregatedMat(Mat* matIn,const word stateName);

    /// extract segregated matrices from the full coupled matrices
    void extractSegregatedAdjointMat
    (
        Mat coupledMat, 
        Mat segregatedMat, 
        const word stateName
    );

    /// extract segregated vectors from the full coupled vectors
    void extractSegregatedAdjointVec
    (
        Vec coupledVec, 
        Vec segregatedVec, 
        const word stateName
    );


    /// create a multi-level Richardson KSP (MLRKSP) object
    void createMLRKSP
    (
        KSP *genksp,
        Mat jac, 
        Mat jacPC,
        dictionary options
    );
    
    /// ksp monitor function
    static PetscErrorCode myKSPMonitor(KSP,PetscInt,PetscReal,void*);
    
    /// perturb velocity inlet
    void perturbUIn(scalar eps);
    
    /// calculate AdjointDerivative::dRdUIn_
    void calcdRdUIn();

    /// calculate AdjointDerivative::dFdUIn_
    void calcdFdUIn(const word objFunc);

    /// calculate AdjointDerivative::dRdVis_
    void calcdRdVis();

    /// calculate AdjointDerivative::dFdVis_
    void calcdFdVis(const word objFunc);
    
    /// calculate total derivatives and save them to files
    void calcTotalDeriv();

    /// calculate total derivatives for velocity inlet
    void calcTotalDerivUIn();

    /// calculate total derivatives for FFD points
    void calcTotalDerivFFD();

    /// calculate total derivatives for volume coordinates
    void calcTotalDerivXv();

    /// calculate total derivatives for molecular viscosity
    void calcTotalDerivVis();
    
    /// return the scaling factors for a given states
    scalar getStateScaling(const word stateName, label faceI=-1);

    /// initialize AdjointDerivative::dRdFFD_
    void initializedRdFFD();
    
    /// initialize AdjointDerivative::dFdFFD_
    void initializedFdFFD();

    /// matrix free matrix-vector function
    static PetscErrorCode dRdwTMatMult(Mat A,Vec x,Vec y);

    /// set normalization in the AD adjoint solution function
    void setNormalizeStatesScaling2Vec(Vec vecY);

    /// return the runtime for the adjoint solver
    label getRunTime();

    /// clean variables to reduce peak memory usage at the adjoint solution step
    void clearVars4AdjSol();

    /// clean variables to reduce peak memory usage at the adjoint solution step
    void clearVars();

    /// write state variables to files
    void writeStates();

    /// set the undecomposed objFuncsSens_dXXdXv_Serial based on a parallel vector objFuncsSens_dXXdXv
    void setObjFuncsSensSerial(Vec decompVec, Vec undecompVec, word objFunc,word mode);

    /// call adjObj and print the objective function values to screen
    void printObjFuncValues();

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
