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

    /// the initial points for dynamicMesh without volCoord inputs
    autoPtr<pointField> points0Ptr_;

    /// the stateInfo_ list from DAStateInfo object
    HashTable<wordList> stateInfo_;

    /// the solution time for the previous primal solution
    scalar prevPrimalSolTime_ = -1e10;

    label isPrintTime(
        const Time& runTime,
        const label printInterval) const;

    /// check whether the primal fails based on residual and regression fail flag
    label checkPrimalFailure();

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

    /// primal residual tolerance
    scalar primalMinResTol_ = 0.0;

    /// whether to print primal information to the screen
    label printToScreen_ = 0;

    /// whether the regModel compute fails
    label regModelFail_ = 0;

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

    /// initial values for validateStates
    HashTable<scalar> initStateVals_;

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
        Vec dFdW,
        Vec psi);

    /// solve the adjoint equation using the fixed-point iteration method
    virtual label solveAdjointFP(
        Vec dFdW,
        Vec psi);

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

    void printElapsedTime(const Time& runTime, const label printToScreen)
    {
        if (printToScreen)
        {
            Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                 << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                 << nl << endl;
        }
    }

    /// compute dRdWT
    void calcdRdWT(
        const label isPC,
        Mat dRdWT);

    /// Update the preconditioner matrix for the ksp object
    void updateKSPPCMat(
        Mat PCMat,
        KSP ksp);

    /// solve the linear equation given a ksp and right-hand-side vector
    label solveLinearEqn(
        const KSP ksp,
        const Vec rhsVec,
        Vec solVec);

    /// Update the OpenFOAM field values (including both internal and boundary fields) based on the state array
    void updateOFFields(const scalar* states);

    /// Update the OpenFoam mesh point coordinates based on the volume point coords array
    void updateOFMesh(const scalar* volCoords);

    /// assign the state variables from OpenFoam layer to the states array
    void getOFFields(scalar* states)
    {
        daFieldPtr_->ofField2State(states);
    }

    /// get a field variable from OF layer
    void getOFField(
        const word fieldName,
        const word fieldType,
        double* field);

    /// get the flatten mesh points coordinates
    void getOFMeshPoints(double* points);

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
        const double* input,
        const word outputName,
        const word outputType,
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

    /// return the face coordinates based on vol coords
    void calcCouplingFaceCoords(
        const scalar* volCoords,
        scalar* surfCoords);

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

    /// register all residuals as the output for reverse-mode AD
    void registerResidualOutput4AD();

    /// assign the reverse-mode AD input seeds from vecX to the residuals in OpenFOAM
    void assignVec2ResidualGradient(const double* vecX);

    /// set the reverse-mode AD derivatives from the state variables in OpenFOAM to vecY
    void assignStateGradient2Vec(
        double* vecY,
        const label oldTimeLevel = 0);

    /// normalize the reverse-mode AD derivatives stored in vecY
    void normalizeGradientVec(double* vecY);

    /// normalize the jacobian vector product that has states as the input such as dFdW and dRdW
    void normalizeJacTVecProduct(
        const word inputName,
        double* product);

    /// initialize the CoDiPack reverse-mode AD global tape for computing dRdWT*psi
    void initializeGlobalADTape4dRdWT();

    /// whether the volCoord input is defined
    label hasVolCoordInput();

    /// resetting internal info in fvMesh, which is needed for multiple primal runs
    void initDynamicMesh();

    /// return whether to loop the primal solution, similar to runTime::loop() except we don't do file IO
    label loop(Time& runTime);

    /// assign the inputFieldUnsteady values to the OF field vars
    void updateInputFieldUnsteady();

    /// initialize inputFieldUnsteady from the GlobalVar class
    void initInputFieldUnsteady();

    /// assign the mean states values to states
    void meanStatesToStates();

    /// basically, we call DAIndex::getGlobalXvIndex
    label getGlobalXvIndex(
        const label idxPoint,
        const label idxCoord) const
    {
        return daIndexPtr_->getGlobalXvIndex(idxPoint, idxCoord);
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

    /// return the number of local points
    label getNLocalPoints() const
    {
        return meshPtr_->nPoints();
    }

    /// initialize DASolver::daFunctionPtrList_ one needs to call this before calling printAllFunctions
    void setDAFunctionList();

    /// calculate the values of all objective functions and print them to screen
    void calcAllFunctions(label print = 0);

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

    /// check the mesh quality and return meshOK
    label checkMesh() const
    {
        return daCheckMeshPtr_->run();
    }

    /// get the scaling factor for dF/d? derivative computation
    scalar getdFScaling(const word functionName, const label timeIdx);

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

    /// update the boundary condition for a field
    void updateBoundaryConditions(
        const word fieldName,
        const word fieldType);

    /// update the boundary conditions for all states and intermediate variables
    void updateStateBoundaryConditions();

    /// calculate the residuals
    void calcResiduals(label isPC = 0);

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

    /// get the number of regression model parameters
    label getNRegressionParameters(word modelName)
    {
        return daRegressionPtr_->nParameters(modelName);
    }

    /// get the regression parameter
    scalar getRegressionParameter(word modelName, const label idxI)
    {
        return daRegressionPtr_->getParameter(modelName, idxI);
    }

    /// set the regression parameter
    void setRegressionParameter(word modelName, const label idxI, scalar val)
    {
        daRegressionPtr_->setParameter(modelName, idxI, val);
    }

    /// call the compute method of the regression model
    void regressionModelCompute()
    {
        daRegressionPtr_->compute();
    }

    /// update the primal state boundary condition based on the primalBC dict
    void setPrimalBoundaryConditions(const label printInfo = 1);

    /// write the failed mesh to disk
    void writeFailedMesh();

    /// read the state variables from the disk and assign the value to the prescribe time level
    void readStateVars(
        scalar timeVal,
        label oldTimeLevel = 0);

    /// read the mesh points from the disk and run movePoints to deform the mesh
    void readMeshPoints(const scalar timeVal);

    /// write the mesh points to the disk for the given timeVal
    void writeMeshPoints(const double* points, const scalar timeVal);

    /// calculate the PC mat using fvMatrix
    void calcPCMatWithFvMatrix(Mat PCMat, const label turbOnly = 0);

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
