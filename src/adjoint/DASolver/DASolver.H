/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Solver class that calls primal and adjoint solvers,
        and compute the total derivatives

\*---------------------------------------------------------------------------*/

#ifndef DASolver_H
#define DASolver_H

#include <petscksp.h>
#include "Python.h"
#include "fvCFD.H"
#include "runTimeSelectionTables.H"
#include "OFstream.H"
#include "functionObjectList.H"
#include "DAUtility.H"
#include "DACheckMesh.H"
#include "DAOption.H"
#include "DAStateInfo.H"
#include "DAModel.H"
#include "DAIndex.H"
#include "DAObjFunc.H"
#include "DAJacCon.H"
#include "DAColoring.H"
#include "DAResidual.H"
#include "DAField.H"
#include "DAPartDeriv.H"
#include "DALinearEqn.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DASolver Declaration
\*---------------------------------------------------------------------------*/

class DASolver
{

private:
    /// Disallow default bitwise copy construct
    DASolver(const DASolver&);

    /// Disallow default bitwise assignment
    void operator=(const DASolver&);

protected:
    /// all the arguments
    char* argsAll_;

    /// all options in DAFoam
    PyObject* pyOptions_;

    /// args pointer
    autoPtr<argList> argsPtr_;

    /// runTime pointer
    autoPtr<Time> runTimePtr_;

    /// fvMesh pointer
    autoPtr<fvMesh> meshPtr_;

    /// DAOption pointer
    autoPtr<DAOption> daOptionPtr_;

    /// DAModel pointer
    autoPtr<DAModel> daModelPtr_;

    /// DAIndex pointer
    autoPtr<DAIndex> daIndexPtr_;

    /// DAField pointer
    autoPtr<DAField> daFieldPtr_;

    /// a list of DAObjFunc pointers
    UPtrList<DAObjFunc> daObjFuncPtrList_;

    /// DACheckMesh object pointer
    autoPtr<DACheckMesh> daCheckMeshPtr_;

    /// DALinearEqn pointer
    autoPtr<DALinearEqn> daLinearEqnPtr_;

    /// DAResidual pointer
    autoPtr<DAResidual> daResidualPtr_;

    /// DAStateInfo pointer
    autoPtr<DAStateInfo> daStateInfoPtr_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// the list of objective function names that requires adjoint solution
    wordList objFuncNames4Adj_;

    /// the dictionary of adjoint vector (psi) values for all objectives
    dictionary psiVecDict_;

    /// the dictionary that stores the total derivatives reduced from processors
    dictionary totalDerivDict_;

    /// pointer to the objective function file used in unsteady primal solvers
    autoPtr<OFstream> objFuncAvgHistFilePtr_;

    /// number of iterations since the start of objFunc averaging
    label nItersObjFuncAvg_ = -9999;

    /// the averaged objective function values used in unsteady flow
    scalarList avgObjFuncValues_;

    /// the preconditioner matrix for the adjoint linear equation solution
    Mat dRdWTPC_;

    /// how many times the DASolver::solveAdjoint function is called
    label nSolveAdjointCalls_ = 0;

    /// the matrix that stores the partials dXv/dFFD computed from the idwarp and pygeo in the pyDAFoam.py
    Mat dXvdFFDMat_;

    /// the vector that stores the AD seeds that propagate from FFD to Xv and will be used in forward mode AD
    Vec FFD2XvSeedVec_;

    /// the derivative value computed by the forward mode primal solution.
    HashTable<PetscScalar> forwardADDerivVal_;

    /// the maximal residual for primal solution
    scalar primalMinRes_ = 1e10;

    /// the solution time for the previous primal solution
    scalar prevPrimalSolTime_ = -1e10;

    /// setup maximal residual control and print the residual as needed
    template<class classType>
    void primalResidualControl(
        const SolverPerformance<classType>& solverP,
        const label printToScreen,
        const label printInterval,
        const word varName);

    label isPrintTime(
        const Time& runTime,
        const label printInterval) const;

    /// check whether the min residual in primal satisfy the prescribed tolerance
    label checkResidualTol();

    /// reduce the connectivity level for Jacobian connectivity mat
    void reduceStateResConLevel(
        const dictionary& maxResConLv4JacPCMat,
        HashTable<List<List<word>>>& stateResConInfo) const;

    /// set velocity boundary condition for rotating walls
    void setRotingWallVelocity();

    /// write associated fields such as relative velocity
    void writeAssociatedFields();

    /// matrix-free dRdWT matrix used in GMRES solution
    Mat dRdWTMF_;

    /// a flag in dRdWTMatVecMultFunction to determine if the global tap is initialized
    label globalADTape4dRdWTInitialized = 0;

    /// state variable list for all instances (unsteady)
    List<List<scalar>> stateAllInstances_;

    /// state boundary variable list for all instances (unsteady)
    List<List<scalar>> stateBoundaryAllInstances_;

    /// objective function for all instances (unsteady)
    List<dictionary> objFuncsAllInstances_;

    /// the runtime value for all instances (unsteady)
    List<scalar> runTimeAllInstances_;

    /// the runtime index for all instances (unsteady)
    List<label> runTimeIndexAllInstances_;

    /// number of time instances for hybird adjoint (unsteady)
    label nTimeInstances_ = -9999;

    /// periodicity of oscillating flow variables (unsteady)
    scalar periodicity_ = 0.0;

    /// save primal variable to time instance list for hybrid adjoint (unsteady)
    void saveTimeInstanceFieldHybrid(label& timeInstanceI);

    /// save primal variable to time instance list for time accurate adjoint (unsteady)
    void saveTimeInstanceFieldTimeAccurate(label& timeInstanceI);

public:
    /// Runtime type information
    TypeName("DASolver");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DASolver,
        dictionary,
        (
            char* argsAll,
            PyObject* pyOptions),
        (argsAll, pyOptions));

    // Constructors

    //- Construct from components
    DASolver(
        char* argsAll,
        PyObject* pyOptions);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DASolver> New(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DASolver()
    {
    }

    // Member functions

    /// initialize fields and variables
    virtual void initSolver() = 0;

    /// solve the primal equations
    virtual label solvePrimal(
        const Vec xvVec,
        Vec wVec) = 0;

    /// assign primal variables based on the current time instance
    void setTimeInstanceField(const label instanceI);

    /// return the value of objective function at the given time instance and name
    scalar getTimeInstanceObjFunc(
        const label instanceI,
        const word objFuncName);

    /// assign the time instance mats to/from the lists in the OpenFOAM layer depending on the mode
    void setTimeInstanceVar(
        const word mode,
        Mat stateMat,
        Mat stateBCMat,
        Vec timeVec,
        Vec timeIdxVec);

    /// initialize the oldTime for all state variables
    void initOldTimes();

    /// compute dRdWT
    void calcdRdWT(
        const Vec xvVec,
        const Vec wVec,
        const label isPC,
        Mat dRdWT);

    /// compute [dRdW]^T*Psi
    void calcdRdWTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        Vec dRdWTPsi);

    /// compute dFdW
    void calcdFdW(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        Vec dFdW);

    /// create a multi-level, Richardson KSP object
    void createMLRKSP(
        const Mat jacMat,
        const Mat jacPCMat,
        KSP ksp);

    /// solve the linear equation given a ksp and right-hand-side vector
    label solveLinearEqn(
        const KSP ksp,
        const Vec rhsVec,
        Vec solVec);

    /// convert the mpi vec to a seq vec
    void convertMPIVec2SeqVec(
        const Vec mpiVec,
        Vec seqVec);

    /// Update the OpenFOAM field values (including both internal and boundary fields) based on the state vector wVec
    void updateOFField(const Vec wVec);

    /// Update the OpenFoam mesh point coordinates based on the point vector xvVec
    void updateOFMesh(const Vec xvVec);

    /// compute dRdBC
    void calcdRdBC(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        Mat dRdBC);

    /// compute dFdBC
    void calcdFdBC(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdBC);

    /// compute dRdBCAD
    void calcdRdBCTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdBCTPsi);

    /// compute dRdAOA
    void calcdRdAOA(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        Mat dRdAOA);

    /// compute dFdAOA
    void calcdFdAOA(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdAOA);

    /// compute dRdAOAAD
    void calcdRdAOATPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdAOATPsi);

    /// compute dRdFFD
    void calcdRdFFD(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        Mat dRdFFD);

    /// compute dFdFFD
    void calcdFdFFD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdFFD);

    /// compute dRdACT
    void calcdRdACT(
        const Vec xvVec,
        const Vec wVec,
        const word designVarName,
        const word designVarType,
        Mat dRdACT);

    /// compute dFdACT
    void calcdFdACT(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        const word designVarType,
        Vec dFdACT);

    /// compute dRdField^T*Psi
    void calcdRdFieldTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdFieldTPsi);

    /// compute dFdField
    void calcdFdFieldAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdField);

    /// create a multi-level, Richardson KSP object with matrix-free Jacobians
    void createMLRKSPMatrixFree(
        const Mat jacPCMat,
        KSP ksp);

    /// compute dFdW using AD
    void calcdFdWAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        Vec dFdW);

    /// compute dRdXv^T*Psi
    void calcdRdXvTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        Vec dRdXvTPsi);

    /// compute dFdXv AD
    void calcdFdXvAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdXv);

    void calcdRdActTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdActTPsi);

    /// compute dFdACT AD
    void calcdFdACTAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdACT);

    /// compute dRdWOld^T*Psi
    void calcdRdWOldTPsiAD(
        const label oldTimeLevel,
        const Vec psi,
        Vec dRdWOldTPsi);

    /// matrix free matrix-vector product function to compute vecY=dRdWT*vecX
    static PetscErrorCode dRdWTMatVecMultFunction(
        Mat dRdWT,
        Vec vecX,
        Vec vecY);

    /// initialize matrix free dRdWT
    void initializedRdWTMatrixFree(
        const Vec xvVec,
        const Vec wVec);

    /// destroy the matrix free dRdWT
    void destroydRdWTMatrixFree();

    /// register all state variables as the input for reverse-mode AD
    void registerStateVariableInput4AD(const label oldTimeLevel = 0);

    /// register field variables as the input for reverse-mode AD
    void registerFieldVariableInput4AD(
        const word fieldName,
        const word fieldType);

    /// register all residuals as the output for reverse-mode AD
    void registerResidualOutput4AD();

    /// assign the reverse-mode AD input seeds from vecX to the residuals in OpenFOAM
    void assignVec2ResidualGradient(Vec vecX);

    /// set the reverse-mode AD derivatives from the state variables in OpenFOAM to vecY
    void assignStateGradient2Vec(
        Vec vecY,
        const label oldTimeLevel = 0);

    /// set the reverse-mode AD derivatives from the field variables in OpenFOAM to vecY
    void assignFieldGradient2Vec(
        const word fieldName,
        const word fieldType,
        Vec vecY);

    /// normalize the reverse-mode AD derivatives stored in vecY
    void normalizeGradientVec(Vec vecY);

    /// initialize the CoDiPack reverse-mode AD global tape for computing dRdWT*psi
    void initializeGlobalADTape4dRdWT();

    /// return whether to loop the primal solution, similar to runTime::loop() except we don't do file IO
    label loop(Time& runTime);

    /// basically, we call DAIndex::getGlobalXvIndex
    label getGlobalXvIndex(
        const label idxPoint,
        const label idxCoord) const
    {
        return daIndexPtr_->getGlobalXvIndex(idxPoint, idxCoord);
    }

    /// set the state vector based on the latest fields in OpenFOAM
    void ofField2StateVec(Vec stateVec) const
    {
        daFieldPtr_->ofField2StateVec(stateVec);
    }

    /// assign the fields in OpenFOAM based on the state vector
    void stateVec2OFField(const Vec stateVec) const
    {
        daFieldPtr_->stateVec2OFField(stateVec);
    }

    /// assign the points in fvMesh of OpenFOAM based on the point vector
    void pointVec2OFMesh(const Vec xvVec) const
    {
        daFieldPtr_->pointVec2OFMesh(xvVec);
    }

    /// assign the point vector based on the points in fvMesh of OpenFOAM
    void ofMesh2PointVec(Vec xvVec) const
    {
        daFieldPtr_->ofMesh2PointVec(xvVec);
    }

    /// assign the OpenFOAM residual fields based on the resVec
    void resVec2OFResField(const Vec resVec) const
    {
        daFieldPtr_->resVec2OFResField(resVec);
    }

    /// assign the resVec based on OpenFOAM residual fields
    void ofResField2ResVec(Vec resVec) const
    {
        daFieldPtr_->resVec2OFResField(resVec);
    }

    /// write the matrix in binary format
    void writeMatrixBinary(
        const Mat matIn,
        const word prefix)
    {
        DAUtility::writeMatrixBinary(matIn, prefix);
    }

    /// write the matrix in ASCII format
    void writeMatrixASCII(
        const Mat matIn,
        const word prefix)
    {
        DAUtility::writeMatrixASCII(matIn, prefix);
    }

    /// read petsc matrix in binary format
    void readMatrixBinary(
        Mat matIn,
        const word prefix)
    {
        DAUtility::readMatrixBinary(matIn, prefix);
    }

    /// write petsc vector in ascii format
    void writeVectorASCII(
        const Vec vecIn,
        const word prefix)
    {
        DAUtility::writeVectorASCII(vecIn, prefix);
    }

    /// read petsc vector in binary format
    void readVectorBinary(
        Vec vecIn,
        const word prefix)
    {
        DAUtility::readVectorBinary(vecIn, prefix);
    }

    /// write petsc vector in binary format
    void writeVectorBinary(
        const Vec vecIn,
        const word prefix)
    {
        DAUtility::writeVectorBinary(vecIn, prefix);
    }

    /// return the number of local adjoint states
    label getNLocalAdjointStates() const
    {
        return daIndexPtr_->nLocalAdjointStates;
    }

    /// return the number of local adjoint boundary states
    label getNLocalAdjointBoundaryStates() const
    {
        return daIndexPtr_->nLocalAdjointBoundaryStates;
    }

    /// return the number of local cells
    label getNLocalCells() const
    {
        return meshPtr_->nCells();
    }

    /// initialize DASolver::daObjFuncPtrList_ one needs to call this before calling printAllObjFuncs
    void setDAObjFuncList();

    /// calculate the values of all objective functions and print them to screen
    void printAllObjFuncs();

    /// check the mesh quality and return meshOK
    label checkMesh() const
    {
        return daCheckMeshPtr_->run();
    }

    /// return the value of the objective function
    scalar getObjFuncValue(const word objFuncName);

    /// print all DAOption
    void printAllOptions()
    {
        Info << "DAFoam option dictionary: ";
        Info << daOptionPtr_->getAllOptions() << endl;
    }

    /// calculate the norms of all residuals and print to screen
    void calcPrimalResidualStatistics(
        const word mode,
        const label writeRes = 0);

    /// set the value for DASolver::dXvdFFDMat_
    void setdXvdFFDMat(const Mat dXvdFFDMat);

    /// set the value for DASolver::FFD2XvSeedVec_
    void setFFD2XvSeedVec(Vec vecIn);

    /// update the allOptions_ dict in DAOption based on the pyOptions from pyDAFoam
    void updateDAOption(PyObject* pyOptions)
    {
        daOptionPtr_->updateDAOption(pyOptions);
    }

    /// get the solution time folder for previous primal solution
    scalar getPrevPrimalSolTime()
    {
        return prevPrimalSolTime_;
    }

    /// set the field value
    void setFieldValue4GlobalCellI(
        const word fieldName,
        const scalar val,
        const label globalCellI,
        const label compI = 0);

    /// update the boundary condition for a field
    void updateBoundaryConditions(
        const word fieldName,
        const word fieldType);

    /// synchronize the values in DAOption and actuatorDiskDVs_
    void syncDAOptionToActuatorDVs()
    {
        DAFvSource& fvSource = const_cast<DAFvSource&>(
            meshPtr_->thisDb().lookupObject<DAFvSource>("DAFvSource"));
        fvSource.syncDAOptionToActuatorDVs();
    }

    /// Accessing members
    /// return the mesh object
    const fvMesh& getMesh()
    {
        return meshPtr_();
    }

    /// return the runTime object
    const Time& getRunTime()
    {
        return runTimePtr_();
    }

    /// get DAOption object
    const DAOption& getDAOption()
    {
        return daOptionPtr_();
    }

    /// get DAStateInfo object
    const DAStateInfo& getDAStateInfo()
    {
        return daStateInfoPtr_();
    }

    /// get DAIndex object
    const DAIndex& getDAIndex()
    {
        return daIndexPtr_();
    }

    /// get DAModel object
    const DAModel& getDAModel()
    {
        return daModelPtr_();
    }

    /// get DAResidual object
    const DAResidual& getDAResidual()
    {
        return daResidualPtr_();
    }

    /// get DAField object
    const DAField& getDAField()
    {
        return daFieldPtr_();
    }

    /// get DALinearEqn object
    const DALinearEqn& getDALinearEqn()
    {
        return daLinearEqnPtr_();
    }

    /// get DACheckMesh object
    const DACheckMesh& getDACheckMesh()
    {
        return daCheckMeshPtr_();
    }

    /// get forwardADDerivVal_
    PetscScalar getForwardADDerivVal(const word objFuncName)
    {
        return forwardADDerivVal_[objFuncName];
    }

    /// calculate the residual and assign it to the resVec vector
    void calcResidualVec(Vec resVec);

#ifdef CODI_AD_REVERSE

    /// global tape for reverse-mode AD
    codi::RealReverse::TapeType& globalADTape_;

#endif
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class classType>
void DASolver::primalResidualControl(
    const SolverPerformance<classType>& solverP,
    const label printToScreen,
    const label printInterval,
    const word varName)
{
    /*
    Description:
        Setup maximal residual control and print the residual as needed
    */
    if (mag(solverP.initialResidual()) < primalMinRes_)
    {
        primalMinRes_ = mag(solverP.initialResidual());
    }
    if (printToScreen)
    {
        Info << varName << " Initial residual: " << solverP.initialResidual() << endl
             << varName << "   Final residual: " << solverP.finalResidual() << endl;
    }
}

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
