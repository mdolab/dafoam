/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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
#include "DAFunction.H"
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
#include "DAInput.H"
#include "DAOutput.H"
#include "DAGlobalVar.H"
#include "DATimeOp.H"

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

    /// a list of DAFunction pointers
    UPtrList<DAFunction> daFunctionPtrList_;

    /// a list of DATimeOp pointers
    UPtrList<DATimeOp> daTimeOpPtrList_;

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

    /// DAGlobalVar pointer
    autoPtr<DAGlobalVar> daGlobalVarPtr_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// the list of objective function names that requires adjoint solution
    wordList functionNames4Adj_;

    /// the dictionary of adjoint vector (psi) values for all objectives
    dictionary psiVecDict_;

    /// the dictionary that stores the total derivatives reduced from processors
    dictionary totalDerivDict_;

    /// pointer to the objective function file used in unsteady primal solvers
    autoPtr<OFstream> functionAvgHistFilePtr_;

    /// number of iterations since the start of function averaging
    label nItersFunctionAvg_ = -9999;

    /// the averaged objective function values used in unsteady flow
    scalarList avgFunctionValues_;

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
    label checkResidualTol(const scalar& primalMaxRes);

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
    List<dictionary> functionsAllInstances_;

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

    /// primal maximal residual computed from all the governing equations
    scalar primalMaxRes_ = 0.0;

    /// whether to print primal information to the screen
    label printToScreen_ = 0;

    /// how frequent do you want to print the primal info default is every 100 steps
    label printInterval_ = 100;

    /// primal min number of iterations
    label primalMinIters_ = -1;

    /// how frequent do you want to print the primal info default is every 100 steps
    label printIntervalUnsteady_ = 1;

    /// a list list that saves the function value for all time steps
    List<scalarList> functionTimeSteps_;

    /// the final time index from the primal solve. for steady state cases it can converge before endTime
    label primalFinalTimeIndex_;

    /// a scaling factor related to the unsteady obj func timeOperator e.g., in dFdW calculation
    scalar unsteadyFunctionsScaling_;

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
        (char* argsAll,
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
    virtual label solvePrimal() = 0;

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
            return -1;
        }
    }

    /// return the value of objective function at the given time instance and name
    scalar getTimeInstanceFunction(
        const label instanceI,
        const word functionName);

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
    void updateOFField(const scalar* states);

    /// Update the OpenFoam mesh point coordinates based on the volume point coords array
    void updateOFMesh(const scalar* volCoords);

    /// assign the state variables from OpenFoam layer to the states array
    void getOFField(scalar* states)
    {
        daFieldPtr_->ofField2State(states);
    }

    /// reset the seeds (gradient to zeros) for all OpenFOAM variables
    void resetOFSeeds();

    /// assign the BC val to OF var
    void setBCToOFVars(
        const dictionary& dvSubDict,
        const scalar& BC);

    /// get the BC val from OF var
    void getBCFromOFVars(
        const dictionary& dvSubDict,
        scalar& BC);

    /// get the array size of an input type
    label getInputSize(
        const word inputName,
        const word inputType);

    /// get the array size of an output type
    label getOutputSize(
        const word outputName,
        const word outputType);
    
    /// get whether the input is distributed among processors
    label getInputDistributed(
        const word inputName,
        const word inputType);
    
    /// get whether the output is distributed among processors
    label getOutputDistributed(
        const word outputName,
        const word outputType);
    
    /// get whether the output is distributed among processors
    void calcOutput(
        const word outputName,
        const word outputType,
        double* output);
    
    /// run the coloring solver
    void runColoring();

    /// calculate the Jacobian-matrix-transposed and vector product for product = [dOutput/dInput]^T * seed
    void calcJacTVecProduct(
        const word inputName,
        const word inputType,
        const int inputSize,
        const double* input,
        const word outputName,
        const word outputType,
        const int outputSize,
        const double* seed,
        double* product);

    void setSolverInput(
        const word inputName,
        const word inputType,
        const int inputSize,
        const double* input,
        const double* seed);

    /// create a multi-level, Richardson KSP object with matrix-free Jacobians
    void createMLRKSPMatrixFree(
        const Mat jacPCMat,
        KSP ksp);

    /// compute dRdWOld^T*Psi
    void calcdRdWOldTPsiAD(
        const label oldTimeLevel,
        const double* psi,
        double* dRdWOldTPsi);

    /// return the coupling patch list if any scenario is active on couplingInfo dict otherwise return design surface
    void getCouplingPatchList(wordList& patchList, word groupName = "NONE");

    /// matrix free matrix-vector product function to compute vecY=dRdWT*vecX
    static PetscErrorCode dRdWTMatVecMultFunction(
        Mat dRdWT,
        Vec vecX,
        Vec vecY);

    /// initialize matrix free dRdWT
    void initializedRdWTMatrixFree();

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
    void assignVec2ResidualGradient(const double* vecX);

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
    
    /// set the reverse-mode AD derivatives from the state variables in OpenFOAM to vecY
    void assignStateGradient2Vec(
        double* vecY,
        const label oldTimeLevel = 0);

    /// set the reverse-mode AD derivatives from the field variables in OpenFOAM to vecY
    void assignFieldGradient2Vec(
        const word fieldName,
        const word fieldType,
        Vec vecY);

    /// normalize the reverse-mode AD derivatives stored in vecY
    void normalizeGradientVec(Vec vecY);

    /// normalize the reverse-mode AD derivatives stored in vecY
    void normalizeGradientVec(double* vecY);

    /// normalize the jacobian vector product that has states as the input such as dFdW and dRdW
    void normalizeJacTVecProduct(
        const word inputName,
        double* product);

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

    /// initialize DASolver::daFunctionPtrList_ one needs to call this before calling printAllFunctions
    void setDAFunctionList();

    /// calculate the values of all objective functions and print them to screen
    void calcAllFunctions(label print=0);

    /// get the function value based on timeOp
    double getTimeOpFuncVal(const word functionName);

    /// return the index of a give functionName in daFunctionPtrList_
    label getFunctionListIndex(const word functionName)
    {
        forAll(daFunctionPtrList_, idxI)
        {
            DAFunction& daFunction = daFunctionPtrList_[idxI];
            word functionName1 = daFunction.getFunctionName();
            if (functionName1 == functionName)
            {
                return idxI;
            }
        }
    }

    /// get the unsteady adjoint start time index
    label getUnsteadyFunctionStartTimeIndex()
    {
        // we assume the adjoint is for deltaT to endTime
        // but users can also prescribed a custom time range
        scalar startTime = this->getRunTime().deltaT().value();
        scalar startTimeRead = daOptionPtr_->getSubDictOption<scalar>("unsteadyAdjoint", "functionStartTime");
        if (startTimeRead >= 0)
        {
            startTime = startTimeRead;
        }
        scalar deltaT = this->getRunTime().deltaT().value();
        return round(startTime / deltaT);
    }

    /// get the unsteady adjoint end time index
    label getUnsteadyFunctionEndTimeIndex()
    {
        // we assume the adjoint is for deltaT to endTime
        // but users can also prescribed a custom time range
        scalar endTime = this->getRunTime().endTime().value();
        scalar endTimeRead = daOptionPtr_->getSubDictOption<scalar>("unsteadyAdjoint", "functionEndTime");
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

    /// calculate and return the value of the objective function
    scalar calcFunction(const word functionName);

    /// get the scaling factor for dF/d? derivative computation
    scalar getdFScaling(const word functionName, const label timeIdx);

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
    PetscScalar getForwardADDerivVal(const word functionName)
    {
        return forwardADDerivVal_[functionName];
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

#ifdef CODI_ADR

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

    /// write the adjoint variables for all states
    void writeAdjointFields(
        const word function,
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
