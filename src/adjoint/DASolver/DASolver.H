/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Solver class that calls primal and adjoint solvers,
        and compute the total derivatives

\*---------------------------------------------------------------------------*/

#ifndef DASolver_H
#define DASolver_H

#include <petscksp.h>
#include "Python.h"
#include "fvCFD.H"
#include "fvMesh.H"
#include "runTimeSelectionTables.H"
#include "OFstream.H"
#include "functionObjectList.H"
#include "fvOptions.H"
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
#include "DARegression.H"
#include "volPointInterpolation.H"
#include "IOMRFZoneListDF.H"
#include "interpolateSplineXY.H"

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

    /// DARegression pointer
    autoPtr<DARegression> daRegressionPtr_;

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

    /// the solution time for the previous primal solution
    scalar prevPrimalSolTime_ = -1e10;

    label isPrintTime(
        const Time& runTime,
        const label printInterval) const;

    /// check whether the min residual in primal satisfy the prescribed tolerance
    label checkResidualTol();

    /// reduce the connectivity level for Jacobian connectivity mat
    void reduceStateResConLevel(
        const dictionary& maxResConLv4JacPCMat,
        HashTable<List<List<word>>>& stateResConInfo) const;

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

    /// primal residual tolerance
    scalar primalMinResTol_ = 0.0;

    /// primal min number of iterations
    label primalMinIters_ = -1;

    /// save primal variable to time instance list for hybrid adjoint (unsteady)
    void saveTimeInstanceFieldHybrid(label& timeInstanceI);

    /// save primal variable to time instance list for time accurate adjoint (unsteady)
    void saveTimeInstanceFieldTimeAccurate(label& timeInstanceI);

    /// a hash table that save the unsteady obj funcs, e.g., time-averaged drag
    HashTable<scalar> unsteadyObjFuncs_;

    /// a scaling factor related to the unsteady obj func timeOperator e.g., in dFdW calculation
    scalar unsteadyObjFuncsScaling_;

    /// initial values for validateStates
    HashTable<scalar> initStateVals_;

    /// whether to use mean states for steady-solvers
    label useMeanStates_ = 0;
    /// if the meanStates is used, what is the step-averaging start
    scalar meanStateStart_ = 0.5;
    /// whether we have calculated the meanStates, if not (flow converge before meanStateStart_), we will not use meanStates
    label meanStatesCalculated_ = 0;

    /// step-averaged scalar states
    PtrList<volScalarField> meanVolScalarStates_;

    /// step-averaged model states
    PtrList<volScalarField> meanModelStates_;

    /// step-averaged vector states
    PtrList<volVectorField> meanVolVectorStates_;

    /// step-averaged surfaceScalar states
    PtrList<surfaceScalarField> meanSurfaceScalarStates_;

    /// whether to use the primal's std as the convergence criterion
    label primalObjStdActive_;

    /// the step series of the primal objective, used if primalObjStdTol->active is True
    scalarList primalObjSeries_;

    /// the standard deviation of the primal objective, used if primalObjStdTol->active is True
    scalar primalObjStd_;

    /// the tolerance of the primal objective std, used if primalObjStdTol->active is True
    scalar primalObjStdTol_;

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

    /// solve the adjoint equation using the fixed-point iteration method
    virtual label runFPAdj(
        const Vec xvVec,
        const Vec wVec,
        Vec dFdW,
        Vec psi);

    /// calculate the step-averaged mean states for steady solvers
    void calcMeanStates();

    /// initialize the meanStates if needed
    void initMeanStates();

    /// zero out the mean states before calculating the meanStates
    void zeroMeanStates();

    /// assign meanStates to states when the primal is done.
    void assignMeanStatesToStates();

    /// assign primal variables based on the current time instance
    void setTimeInstanceField(const label instanceI);

    /// setTime for OF fields
    void setTime(scalar time, label timeIndex)
    {
        runTimePtr_->setTime(time, timeIndex);
    }

    /// get the ddtScheme order
    label getDdtSchemeOrder()
    {
        const fvSchemes& myFvSchemes = meshPtr_->thisDb().lookupObject<fvSchemes>("fvSchemes");

        word ddtSchemeName = myFvSchemes.subDict("ddtSchemes").getWord("default");

        if (ddtSchemeName == "steadyState")
        {
            return 0;
        }
        if (ddtSchemeName == "Euler")
        {
            return 1;
        }
        else if (ddtSchemeName == "backward")
        {
            return 2;
        }
        else
        {
            FatalErrorIn("") << "ddtScheme " << ddtSchemeName << " not supported! Options: steadyState, Euler, or backward"
                             << abort(FatalError);
        }
    }

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

    /// compute [dRdW]^T*Psi but without updating xvVec and wVec to the OF fields
    void calcdRdWTPsiAD(
        const label isInit,
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

    /// Update the preconditioner matrix for the ksp object
    void updateKSPPCMat(
        Mat PCMat,
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

    /// Update the OpenFOAM field values (including both internal and boundary fields) based on the state array
    void updateOFField(const scalar* volCoords);

    /// Update the OpenFoam mesh point coordinates based on the volume point coords array
    void updateOFMesh(const scalar* states);

    /// reset the seeds (gradient to zeros) for all OpenFOAM variables
    void resetOFSeeds();

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

    /// assign the BC val to OF var
    void setBCToOFVars(
        const dictionary& dvSubDict,
        const scalar& BC);

    /// get the BC val from OF var
    void getBCFromOFVars(
        const dictionary& dvSubDict,
        scalar& BC);

    /// compute dRdBCAD
    void calcdRdBCTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        const word designVarName,
        Vec dRdBCTPsi);

    /// compute dFdBCAD
    void calcdFdBCAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdBC);

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

    void calcdRdThermalTPsiAD(
        const double* volCoords,
        const double* states,
        const double* thermal,
        const double* seeds,
        double* product);

    /// compute dRdXv^T*Psi
    void calcdRdXvTPsiAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec psi,
        Vec dRdXvTPsi);

    /// compute dForcedXv
    void calcdForcedXvAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dForcedXv);

    /// compute dAcoudXv
    void calcdAcousticsdXvAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dForcedXv,
        word varName,
        word groupName);

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

    void calcdForcedWAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dForcedW);

    void calcdAcousticsdWAD(
        const Vec xvVec,
        const Vec wVec,
        const Vec fBarVec,
        Vec dForcedW,
        word varName,
        word groupName);

    /// compute dFdACT AD
    void calcdFdACTAD(
        const Vec xvVec,
        const Vec wVec,
        const word objFuncName,
        const word designVarName,
        Vec dFdACT);

    /// compute dFdRegPar AD
    void calcdFdRegParAD(
        const double* volCoords,
        const double* states,
        const double* parameters,
        const word objFuncName,
        const word designVarName,
        const word modelName,
        double* dFdRegPar);

    /// compute dRdWOld^T*Psi
    void calcdRdWOldTPsiAD(
        const label oldTimeLevel,
        const Vec psi,
        Vec dRdWOldTPsi);

    /// return the coupling patch list if any scenario is active on couplingInfo dict otherwise return design surface
    void getCouplingPatchList(wordList& patchList, word groupName = "NONE");

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

    /// deactivate all state variables as the input for reverse-mode AD
    void deactivateStateVariableInput4AD(const label oldTimeLevel = 0);

    /// register field variables as the input for reverse-mode AD
    void registerFieldVariableInput4AD(
        const word fieldName,
        const word fieldType);

    /// deactivate the field variables for reverse-mode AD
    void deactivateFieldVariableInput4AD(
        const word fieldName,
        const word fieldType);

    /// register all residuals as the output for reverse-mode AD
    void registerResidualOutput4AD();

    /// register all force as the ouptut for referse-mod AD
    void registerForceOutput4AD(List<scalar>& fX, List<scalar>& fY, List<scalar>& fZ);

    /// register acoustic as the ouptut for referse-mod AD
    void registerAcousticOutput4AD(List<scalar>& a);

    /// assign the reverse-mode AD input seeds from vecX to the residuals in OpenFOAM
    void assignVec2ResidualGradient(Vec vecX);

    /// assign the reverse-mode AD input seeds from vecX to the residuals in OpenFOAM
    void assignSeeds2ResidualGradient(const double* seeds);

    /// assign the reverse-mode AD input seeds from fBarVec to the force vectors in OpenFOAM
    void assignVec2ForceGradient(Vec fBarVec, List<scalar>& fX, List<scalar>& fY, List<scalar>& fZ);

    /// assign the reverse-mode AD input seeds from fBarVec to the acoustic vectors in OpenFOAM
    void assignVec2AcousticGradient(Vec fBarVec, List<scalar>& a, label offset, label step);

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

    /// get a variable from OF layer
    void getOFField(
        const word fieldName,
        const word fieldType,
        Vec field) const;

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

    /// return the number of local points
    label getNLocalPoints() const
    {
        return meshPtr_->nPoints();
    }

    /// initialize DASolver::daObjFuncPtrList_ one needs to call this before calling printAllObjFuncs
    void setDAObjFuncList();

    /// calculate the values of all objective functions and print them to screen
    void printAllObjFuncs();

    /// reset the unsteady objFunction to zero
    void initUnsteadyObjFuncs()
    {
        forAll(daObjFuncPtrList_, idxI)
        {
            DAObjFunc& daObjFunc = daObjFuncPtrList_[idxI];
            word objFuncName = daObjFunc.getObjFuncName();
            word objFuncPart = daObjFunc.getObjFuncPart();
            word uKey = objFuncName + objFuncPart;
            unsteadyObjFuncs_[uKey] = 0.0;
        }
    }

    /// return the index of a give objFuncName in daObjFuncPtrList_
    label getObjFuncListIndex(
        const word objFuncName,
        const word objFuncPart)
    {
        forAll(daObjFuncPtrList_, idxI)
        {
            DAObjFunc& daObjFunc = daObjFuncPtrList_[idxI];
            word objFuncName1 = daObjFunc.getObjFuncName();
            word objFuncPart1 = daObjFunc.getObjFuncPart();
            if (objFuncPart1 == objFuncPart && objFuncName1 == objFuncName)
            {
                return idxI;
            }
        }
    }

    /// calculate and save the unsteady objective function
    void calcUnsteadyObjFuncs();

    /// return the unsteady objFunc values
    scalar getObjFuncValueUnsteady(const word objFuncName);

    /// get the objective func scaling for unsteady adjoint
    scalar getObjFuncUnsteadyScaling()
    {
        return unsteadyObjFuncsScaling_;
    }

    /// get the unsteady adjoint start time index
    label getUnsteadyObjFuncStartTimeIndex()
    {
        // we assume the adjoint is for deltaT to endTime
        // but users can also prescribed a custom time range
        scalar startTime = this->getRunTime().deltaT().value();
        scalar startTimeRead = daOptionPtr_->getSubDictOption<scalar>("unsteadyAdjoint", "objFuncStartTime");
        if (startTimeRead >= 0)
        {
            startTime = startTimeRead;
        }
        scalar deltaT = this->getRunTime().deltaT().value();
        return round(startTime / deltaT);
    }

    /// get the unsteady adjoint end time index
    label getUnsteadyObjFuncEndTimeIndex()
    {
        // we assume the adjoint is for deltaT to endTime
        // but users can also prescribed a custom time range
        scalar endTime = this->getRunTime().endTime().value();
        scalar endTimeRead = daOptionPtr_->getSubDictOption<scalar>("unsteadyAdjoint", "objFuncEndTime");
        if (endTimeRead >= 0)
        {
            endTime = endTimeRead;
        }
        scalar deltaT = this->getRunTime().deltaT().value();
        return round(endTime / deltaT);
    }

    /// check the mesh quality and return meshOK
    label checkMesh() const
    {
        return daCheckMeshPtr_->run();
    }

    /// return the value of the objective function
    scalar getObjFuncValue(const word objFuncName);

    /// return the forces of the desired fluid-structure-interaction patches
    void getForces(
        Vec fX,
        Vec fY,
        Vec fZ);

    /// return the positions, normals, areas, and forces of the desired acoustic patches
    void getAcousticData(
        Vec x,
        Vec y,
        Vec z,
        Vec nX,
        Vec nY,
        Vec nZ,
        Vec a,
        Vec fX,
        Vec fY,
        Vec fZ,
        word groupName);

    /// return the number of points and faces for MDO patches
    void getPatchInfo(
        label& nPoints,
        label& nFaces,
        List<word>& patchList);

    /// compute the forces of the desired fluid-structure-interation patches
    void getForcesInternal(
        List<scalar>& fX,
        List<scalar>& fY,
        List<scalar>& fZ,
        List<word>& patchList);

    /// Get the number of faces for the MDO coupling patches
    label getNCouplingFaces();

    /// Get the number of points for the MDO coupling patches
    label getNCouplingPoints();

    /// return the face coordinates based on vol coords
    void calcCouplingFaceCoords(
        const scalar* volCoords,
        scalar* surfCoords);

    /// calc matrix-vector products for calcCouplingFaceCoords
    void calcCouplingFaceCoordsAD(
        const double* volCoords,
        const double* seeds,
        double* product);

    /// compute the temperature on the conjugate heat transfer patches
    void getThermal(
        const scalar* volCoords,
        const scalar* states,
        scalar* thermal);

    /// compute the temperature on the conjugate heat transfer patches AD
    void getThermalAD(
        const word inputName,
        const double* volCoords,
        const double* states,
        const double* seeds,
        double* product);

    /// assign temperature values to the conjugate heat transfer patches
    void setThermal(scalar* thermal);

    /// compute the positions, normals, areas, and forces of the desired acoustic patches
    void getAcousticDataInternal(
        List<scalar>& x,
        List<scalar>& y,
        List<scalar>& z,
        List<scalar>& nX,
        List<scalar>& nY,
        List<scalar>& nZ,
        List<scalar>& a,
        List<scalar>& fX,
        List<scalar>& fY,
        List<scalar>& fZ,
        List<word>& patchList);

    /// calculate the radial profile of the force on the propeller surface
    void calcForceProfile(
        const word propName,
        Vec aForce,
        Vec tForce,
        Vec rDist,
        Vec integralForce);

    void calcForceProfileInternal(
        const word propName,
        scalarList& aForce,
        scalarList& tForce,
        scalarList& rDist,
        scalarList& integralForce);

    void calcdForceProfiledXvWAD(
        const word propName,
        const word inputMode,
        const word outputMode,
        const Vec xvVec,
        const Vec wVec,
        Vec psi,
        Vec dForcedXvW);

    void calcdForcedStateTPsiAD(
        const word mode,
        Vec xvVec,
        Vec stateVec,
        Vec psiVec,
        Vec prodVec);

    /// calculate the fvSource based on the radial force profile
    void calcFvSourceInternal(
        const word propName,
        const scalarField& aForce,
        const scalarField& tForce,
        const scalarField& rDistList,
        const scalarList& targetForce,
        const vector& center,
        volVectorField& fvSource);

    void calcFvSource(
        const word propName,
        Vec aForce,
        Vec tForce,
        Vec rDist,
        Vec targetForce,
        Vec center,
        Vec xvVec,
        Vec fvSource);

    void calcdFvSourcedInputsTPsiAD(
        const word propName,
        const word mode,
        Vec aForce,
        Vec tForce,
        Vec rDist,
        Vec targetForce,
        Vec center,
        Vec xvVec,
        Vec psi,
        Vec dFvSource);

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

    void setFieldValue4LocalCellI(
        const word fieldName,
        const scalar val,
        const label localCellI,
        const label compI = 0);

    /// update the boundary condition for a field
    void updateBoundaryConditions(
        const word fieldName,
        const word fieldType);

    /// update the boundary conditions for all states and intermediate variables
    void updateStateBoundaryConditions();

    /// calculate the residuals
    void calcResiduals(label isPC = 0);

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

    /// get the number of regression model parameters
    label getNRegressionParameters(word modelName)
    {
#ifdef SolidDASolver
        return 0;
#else
        return daRegressionPtr_->nParameters(modelName);
#endif
    }

    /// get the regression parameter
    scalar getRegressionParameter(word modelName, const label idxI)
    {
#ifdef SolidDASolver
        return 0.0;
#else
        return daRegressionPtr_->getParameter(modelName, idxI);
#endif
    }

    /// set the regression parameter
    void setRegressionParameter(word modelName, const label idxI, scalar val)
    {
#ifndef SolidDASolver
        daRegressionPtr_->setParameter(modelName, idxI, val);
#endif
    }

    /// call the compute method of the regression model
    void regressionModelCompute()
    {
#ifndef SolidDASolver
        daRegressionPtr_->compute();
#endif
    }

    /// calculte [dR/dRegParameters]^T * psi
    void calcdRdRegParTPsiAD(
        const double* volCoords,
        const double* states,
        const double* parameters,
        const double* seeds,
        const word modelName,
        double* product);

    /// update the primal state boundary condition based on the primalBC dict
    void setPrimalBoundaryConditions(const label printInfo = 1);

    /// calculate the residual and assign it to the resVec vector
    void calcResidualVec(Vec resVec);

    /// write the failed mesh to disk
    void writeFailedMesh();

    /// read the state variables from the disk and assign the value to the prescribe time level
    void readStateVars(
        scalar timeVal,
        label oldTimeLevel = 0);

    /// calculate the PC mat using fvMatrix
    void calcPCMatWithFvMatrix(Mat PCMat);

    /// initialize tensorflow functions and interfaces for callback
    void initTensorFlowFuncs(
        pyComputeInterface computeInterface,
        void* compute,
        pyJacVecProdInterface jacVecProdInterface,
        void* jacVecProd,
        pySetCharInterface setModelNameInterface,
        void* setModelName)
    {
        DAUtility::pyCalcBetaInterface = computeInterface;
        DAUtility::pyCalcBeta = compute;

        DAUtility::pyCalcBetaJacVecProdInterface = jacVecProdInterface;
        DAUtility::pyCalcBetaJacVecProd = jacVecProd;

        DAUtility::pySetModelNameInterface = setModelNameInterface;
        DAUtility::pySetModelName = setModelName;
    }

    /// write state variables that are NO_WRITE to disk
    void writeAdjStates(
        const label writeMesh,
        const wordList& additionalOutput);

    /// return the elapsed clock time for testing speed
    scalar getElapsedClockTime()
    {
        return runTimePtr_->elapsedClockTime();
    }

    /// return the elapsed CPU time for testing speed
    scalar getElapsedCpuTime()
    {
        return runTimePtr_->elapsedCpuTime();
    }

#ifdef CODI_AD_REVERSE

    /// global tape for reverse-mode AD
    codi::RealReverse::Tape& globalADTape_;

#endif

    /// check if a field variable has nan
    template<class classType>
    label validateField(const classType& field);

    /// check if a field variable has nan
    template<class classType>
    label validateVectorField(const classType& field);

    /// check if the state variables have valid values
    label validateStates();

    /// calculate the initial value for validate states
    void getInitStateVals(HashTable<scalar>& initState);

    /// reset the states to its initial values this usually happens when we have nan in states
    void resetStateVals();

    /// write the sensitivity map for all wall surfaces
    void writeSensMapSurface(
        const word name,
        const double* dFdXs,
        const double* Xs,
        const label size,
        const double timeName);

    /// write the sensitivity map for the entire field
    void writeSensMapField(
        const word name,
        const double* dFdField,
        const word fieldType,
        const double timeName);

    /// get the latest time solution from the case folder.
    scalar getLatestTime()
    {
        instantList timeDirs = runTimePtr_->findTimes(runTimePtr_->path(), runTimePtr_->constant());
        scalar latestTime = timeDirs.last().value();
        return latestTime;
    }

    /// initialize the objStd vars
    void initObjStd();

    /// calculate the objective function's std
    void calcObjStd(Time& runTime);

    /// write the adjoint variables for all states
    void writeAdjointFields(
        const word objFunc,
        const double writeTime,
        const double* psi);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

template<class classType>
label DASolver::validateField(const classType& field)
{
    /*
    Description:
        Check if a field variable has invalid in it. If the values are valid, return 0
    */

    forAll(field, idxI)
    {
        const scalar& val = field[idxI];
        if (std::isnan(val) || std::isinf(val) || fabs(val) > 1e15)
        {
            return 1;
        }
    }
    forAll(field.boundaryField(), patchI)
    {
        forAll(field.boundaryField()[patchI], faceI)
        {
            const scalar& val = field.boundaryField()[patchI][faceI];
            if (std::isnan(val) || std::isinf(val) || fabs(val) > 1e15)
            {
                return 1;
            }
        }
    }

    return 0;
}

template<class classType>
label DASolver::validateVectorField(const classType& field)
{
    /*
    Description:
        Check if a field variable has invalid in it. If the values are valid, return 0
    */

    forAll(field, idxI)
    {
        for (label compI = 0; compI < 3; compI++)
        {
            const scalar& val = field[idxI][compI];
            if (std::isnan(val) || std::isinf(val) || fabs(val) > 1e15)
            {
                return 1;
            }
        }
    }
    forAll(field.boundaryField(), patchI)
    {
        forAll(field.boundaryField()[patchI], faceI)
        {
            for (label compI = 0; compI < 3; compI++)
            {
                const scalar& val = field.boundaryField()[patchI][faceI][compI];
                if (std::isnan(val) || std::isinf(val) || fabs(val) > 1e15)
                {
                    return 1;
                }
            }
        }
    }

    return 0;
}

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
