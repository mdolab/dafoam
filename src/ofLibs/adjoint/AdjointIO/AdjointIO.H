/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Handle IO parameters for the adjoint

\*---------------------------------------------------------------------------*/

#ifndef AdjointIO_H
#define AdjointIO_H

#include <petscksp.h>
#include "fvOptions.H"
#include "mixedFvPatchFields.H" // for setFlowBCs
#include "wordRe.H"
#include "wordRes.H"
#include "tractionDisplacementFvPatchVectorField.H"
#include "fixedGradientFvPatchField.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointIO Declaration
\*---------------------------------------------------------------------------*/

class AdjointIO
{

private:

    const fvMesh& mesh_;
    label readOptionBoolean
    (
        const dictionary& dict,
        word option
    );

protected:

    /// adjoint dict
    IOdictionary adjointDict_;

    /// A temp dict to store all the adjoint parameters and print them to screen later 
    IOdictionary adjointParameters_; 
    //Note: if we dont specify any parmeters in system/adjointDict, the default values will be used. All the dict options will be added to adjointParameters
    
public:

    // Constructors

    //- Construct from components
    AdjointIO
    (
        const fvMesh& mesh
    );

    //- Destructor
    virtual ~AdjointIO();
    
    // adjoint options
    /// finite-difference step size for state variables
    scalar epsDeriv;
    /// finite-difference step size for FFD point coordinates
    scalar epsDerivFFD;
    /// finite-difference step size for mesh volume coordinates
    scalar epsDerivXv;
    /// finite-difference step for molecular viscosity nu
    scalar epsDerivVis;
    /// finite-difference step for velocity inlet
    scalar epsDerivUIn;
    /// whether to construct the Jacobians and solve the adjoint equation
    label solveAdjoint;
    /// whether to use graph coloring to accelerate FD-based Jacobian computation
    label useColoring;
    /// whether to correct wall distance when volume coordinates are changed
    label correctWallDist;
    /// whether to reduce connectivity levels for state Jacobian matrices
    label reduceResCon4JacMat;
    /// whether to delete the turbulence production terms for PC state Jacobian matrix
    label delTurbProd4PCMat;
    /// whether to compute PC Jacobian matrix
    label calcPCMat;
    /// whether to compute PC Jacobian matrix using the OpenFOAM built-in fvMatrix (fast)
    label fastPCMat;
    /// minimal values for Jacobian matrices
    scalar minTolJac;
    /// maximal values for Jacobian matrices
    scalar maxTolJac;
    /// minimal values for PC Jacobian matrices
    scalar minTolPC;
    /// maximal values for PC Jacobian matrices
    scalar maxTolPC;
    /// which option to use for adjoint transonic preconditioner
    label transonicPCOption;
    /// residual L2 diff tolerance for checking if the state perturbation is properly reset 
    scalar stateResetTol;
    ///  maximal iteration number to iteratively correct D and gradD
    label tractionBCMaxIter;
    /// original ordering strategy for Jacobian matrices, can be either cell-by-cell (cell) or state-by-state (state)
    word adjJacMatOrdering;
    /// how to reorder the Jacobian matrices to save memory and improve convergence, can be natural, nd, rcm, etc.
    word adjJacMatReOrdering;
    /// whether to use the segregated approach to solve the adjoint equation.
    label adjSegregated;
    /// coefficients for segregated adjoint
    HashTable<scalar> segAdjParameters;
    /// GMRES restart interval
    label adjGMRESRestart;
    /// interations for global PC
    label adjGlobalPCIters;
    /// interations for local PC
    label adjLocalPCIters;
    /// levels of ASM PC overlap
    label adjASMOverlap;
    /// fill-in levels for PC Jacobian matrices
    label adjPCFillLevel;
    /// max iteration number for solving adjoint equations
    label adjGMRESMaxIters;
    /// absolute convergence tolerance for solving adjoint equations
    scalar adjGMRESAbsTol;
    /// relative convergence tolerance for solving adjoint equations
    scalar adjGMRESRelTol;
    /// whether to compute the preconditioned eigenvalues during the adjoint GMRES solution
    label adjGMRESCalcEigen;
    /// whether to write all Jacobian matrices and vectors to disk (only for debugging)
    label writeMatrices;
    /// whether to read state Jacobian matrices from disk 
    label readMatrices;
    /// a list of residuals to normalize
    wordList normalizeResiduals;
    /// a list of state variables to normalize
    wordList normalizeStates;
    /// hash table of scaling for state variables
    HashTable<scalar> stateScaling;
    /// hash table of scaling for residuals
    HashTable<scalar> residualScaling;
    /// a list of adjoint design variable types: can you UIn, Vis, FFD, and Xv
    wordList adjDVTypes;
    /// number of design variable FFD points
    label nFFDPoints;
    /// hash table to limit the maximal connectivity levels for PC Jacobian matrices. this will save memory when doing PC fill-in
    HashTable<label> maxResConLv4JacPCMat;
    ///  whether to print the solver parameters
    label printSolverPars;
    
    // flow options
    /// whether to set boundary condition for flow solvers
    label setFlowBCs;
    /// boundary conditions for flow
    dictionary  flowBCs;
    /// reference values for objective function computation, e.g., ARef for frontal area of CD
    HashTable<scalar> referenceValues;
    /// flow condition: Compressible or Incompressible
    word flowCondition;
    /// parameters defined in thermophysicalProperties or transportProperties
    HashTable<scalar> flowProperties;
    /// list of inlet patches
    List<word> inletPatches;
    /// list of outlet patches
    List<word> outletPatches;
    /// information for UIn derivative
    dictionary derivUInInfo;
    /// dictionary for user-defined patches, see detailed format in AdjointIndex::calcFaceIndx4UserDefinedPatches
    dictionary userDefinedPatchInfo;
    /// dictionary for user-defined volumes, see detailed format in AdjointIndex::calcFaceIndx4UserDefinedVolumes
    dictionary userDefinedVolumeInfo;
    /// whether to use div2 scheme
    label divDev2;
    /// whether to use NK solver
    label useNKSolver;
    /// whether to segregate turbulence variables in NK solution
    label nkSegregatedTurb;
    /// whether to segregate phi in NK solution
    label nkSegregatedPhi;
    /// whether to switch on ANK option
    label nkPseudoTransient;
    /// initial tolerance for EW
    scalar nkEWRTol0;
    /// max tolerance for EW
    scalar nkEWRTolMax;
    /// how frequent to update the PC matrix for NK
    label nkPCLag;
    /// GMRES restart number for NK
    label nkGMRESRestart;
    /// ASM PC overlap for NK
    label nkASMOverlap;
    /// interations of global PC for NK
    label nkGlobalPCIters;
    /// interations of local PC for NK
    label nkLocalPCIters;
    /// ILU pc fill-in level for NK
    label nkPCFillLevel;
    /// max GMRES iteration for NK
    label nkGMRESMaxIters;
    /// nk Jacobian matrix reordering
    word nkJacMatReOrdering;
    /// relative tolerance for NK
    scalar nkRelTol;
    /// absolute tolerance for NK
    scalar nkAbsTol;
    /// S tolerance for NK
    scalar nkSTol;
    /// max iteration for NK
    label nkMaxIters;
    /// max function evaluation for NK
    label nkMaxFuncEvals;
    /// whether to use ANK solver
    label useANKSolver;

    // actuator disk options  
    /// whether to add actuator disk source terms  
    label actuatorActive; 
    /// list of internal volume names for adding the source terms   
    List<word> actuatorVolumeNames;        
    /// target thrust coefficient for the source terms
    scalarList actuatorThrustCoeff; 
    /// propeller pitch-diameter ratio
    scalarList actuatorPOverD;       
    /// whether to adjust source terms values based on the computed thrust of design surfaces
    label actuatorAdjustThrust; 
    /// propeller rotation direction
    List<word> actuatorRotationDir;
    
    // objective function options
    /// list of objective functions for adjoint, see AdjointObjectiveFunction::objFuncsReg_ in AdjointObjectiveFunction for the possible objFuncs
    List<word> objFuncs;
    /// list list of objective function geometry information (patches names, volume names)
    List< List<word> > objFuncGeoInfo;
    /// drag direction for CD
    vector dragDir; 
    /// lift direction for CL
    vector liftDir;
    /// center of rotation
    vector CofR;
    /// rotational speed vector in rad/s
    vector rotRad;

    /// read an option from dict and add them to adjointParameters_
    template<class type>
    type readOptionOrDefault
    (
        const dictionary& dict,
        word option,
        type defaultValue
    );
    
    /// read a boolean option from dict and add them to adjointParameters_
    label readBooleanOptionOrDefault
    (
        const dictionary& dict,
        word option,
        word defaultValue
    );
    
    /// read a list option from dict and add them to adjointParameters_
    template<class listType>
    void readListOption
    (
        listType& list,
        const dictionary& dict,
        word option
    );

    /// whether the value is in the list
    template<class type>
    label isInList(const type value, const List<type>& list) const;
    
    /// write petsc matrix in binary format
    void writeMatrixBinary(const Mat matIn, const word prefix) const;
    
    /// write petsc matrix in ascii format
    void writeMatrixASCII(const Mat matIn, const word prefix) const;
    
    /// read petsc matrix in binary format
    void readMatrixBinary(const Mat matIn, const word prefix) const;
    
    /// write petsc vector in ascii format
    void writeVectorASCII(const Vec vecIn, const word prefix) const;
    
    /// read petsc vector in binary format
    void readVectorBinary(const Vec vecIn, const word prefix) const;
    
    /// write petsc vector in binary format
    void writeVectorBinary(const Vec vecIn, const word prefix) const;
    
    /// write petsc matrix size information
    void writeMatRowSize(Mat matIn, word prefix) const;

    inline label isValueCloseToRef(scalar val, scalar refVal,scalar tol=1.0e-6) const
    {
        if (fabs(val-refVal)<tol) return 1;
        else return 0;
    }

    /// set flow boundary conditions and write them to files
    void setFlowBoundaryConditions();

    /// get the name of pressure state. can be either p or p_rgh
    word getPName() const;

    /// bound a volScalar variable based on parametes defined in fvSolution-SIMPLE
    void boundVar(volScalarField& var);

    /// bound a volVector variable based on parametes defined in fvSolution-SIMPLE
    void boundVar(volVectorField& var);

    /// compute absolute velocity
    void calcUabs(const volVectorField& Urel,volVectorField& Uabs) const;

};


template<class type>
type AdjointIO::readOptionOrDefault
(
    const dictionary& dict,
    word option,
    type defaultValue
)
{
    
    type value = dict.lookupOrDefault<type>(option,defaultValue); 
    if (value == defaultValue)
    {
        // notify the user the default value is used.
        word optionPrint = option+"(D)"; 
        adjointParameters_.add(optionPrint,value);
    }
    else
    {
        adjointParameters_.add(option,value);
    }
    return value;
};


template<class listType>
void AdjointIO::readListOption
(
    listType& list,
    const dictionary& dict,
    word option
)
{
    
    if (dict.found(option))
    {
        listType tmpList(dict.lookup(option));
        list=tmpList;
    }
    else
    {
        FatalErrorIn("")<<option<<" not found in "<<dict<< abort(FatalError);
    }
    adjointParameters_.add(option,list);
    return;
}

template<class type>
label AdjointIO::isInList(const type value, const List<type>& list) const
{
    forAll(list,idxI)
    {
        if(list[idxI] == value)
        {
            return 1;
        }
    }
    return 0;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
