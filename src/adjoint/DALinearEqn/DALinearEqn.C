/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DALinearEqn.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DALinearEqn::DALinearEqn(
    const fvMesh& mesh,
    const DAOption& daOption)
    : mesh_(mesh),
      daOption_(daOption)
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DALinearEqn::createMLRKSP(
    const Mat jacMat,
    const Mat jacPCMat,
    KSP ksp)
{
    /*
    Description:
        This is the main function we need to call to initialize the KSP and set
        up parameters for solving the linear equations
    
    Input:
        gmresRestart: how many Krylov spaces to keep before resetting them.
        Usually, this is set to the gmresMaxIters

        gmresMaxIters: how many GMRES iteration to run at most

        gmresRelTol: the relative tolerance for GMRES

        gmresAbsTol: the absolute tolerance for GMRES

        globalPCIters: globa iteration for PC, usually set it to 0

        asmOverlap: ASM overlap for solving the linearEqn in parallel. 
        Usually set it to 1. Setting a higher number increases the convergence but
        significantly increase the memory usage

        localPCIters: local iteraction for PC. usually set it to 1
        
        jacMatReOrdering: re-order the lhs matrix to reduce memory usage.
        Usually we use nd, rcm, or natural (not re-ordered)
    
        pcFillLevel: how many leve fill-in to use for PC. This is a critical
        parameters for convergence rate. Usually set it to 1. Setting it to a higher
        number increase the convergence, however, the memory usage generally grows 
        exponetially. We rarely set it more than 2.

        printInfo: whether to print summary information before solving 

        jacMat: the right-hand-side petsc matrix 

        jacPCMat: the preconditioner matrix from which we constructor our preconditioners
    
    Output:
        genksp: the set KSP object 
    */

    label gmresRestart =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresRestart");
    label globalPCIters =
        daOption_.getSubDictOption<label>("adjEqnOption", "globalPCIters");
    label asmOverlap =
        daOption_.getSubDictOption<label>("adjEqnOption", "asmOverlap");
    label localPCIters =
        daOption_.getSubDictOption<label>("adjEqnOption", "localPCIters");
    word jacMatReOrdering =
        daOption_.getSubDictOption<word>("adjEqnOption", "jacMatReOrdering");
    label pcFillLevel =
        daOption_.getSubDictOption<label>("adjEqnOption", "pcFillLevel");
    label gmresMaxIters =
        daOption_.getSubDictOption<label>("adjEqnOption", "gmresMaxIters");
    scalar gmresRelTol =
        daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresRelTol");
    scalar gmresAbsTol =
        daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresAbsTol");
    label useNonZeroInitGuess =
        daOption_.getSubDictOption<label>("adjEqnOption", "useNonZeroInitGuess");
    label useMGSO =
        daOption_.getSubDictOption<label>("adjEqnOption", "useMGSO");
    label printInfo =
        daOption_.getSubDictOption<label>("adjEqnOption", "printInfo");

    PC MLRMasterPC, MLRGlobalPC;
    PC MLRsubpc;
    KSP MLRMasterPCKSP;
    KSP* MLRsubksp;
    // ASM Preconditioner variables
    PetscInt MLRoverlap; // width of subdomain overlap
    PetscInt MLRnlocal, MLRfirst; // number of local subblocks, first local subblock

    // Create linear solver context
    //KSPCreate(PETSC_COMM_WORLD, &ksp);

    // Set operators. Here the matrix that defines the linear
    // system also serves as the preconditioning matrix.
    KSPSetOperators(ksp, jacMat, jacPCMat);

    // This code sets up the supplied kspObject in the following
    // specific fashion.
    //
    // The hierarchy of the setup is:
    //  kspObject --> Supplied KSP object
    //  |
    //  --> master_PC --> Preconditioner type set to KSP
    //      |
    //      --> master_PC_KSP --> KSP type set to Richardson with 'globalPreConIts'
    //          |
    //           --> globalPC --> PC type set to 'globalPCType'
    //               |            Usually Additive Schwartz and overlap is set
    //               |            with 'ASMOverlap'. Use 0 to get BlockJacobi
    //               |
    //               --> subKSP --> KSP type set to Richardon with 'LocalPreConIts'
    //                   |
    //                   --> subPC -->  PC type set to 'localPCType'.
    //                                  Usually ILU. 'localFillLevel' is
    //                                  set and 'localMatrixOrder' is used.
    //
    // Note that if globalPreConIts=1 then maser_PC_KSP is NOT created and master_PC=globalPC
    // and if localPreConIts=1 then subKSP is set to preOnly.

    // First, KSPSetFromOptions MUST be called
    KSPSetFromOptions(ksp);

    // Set GMRES
    // Set the type of solver to GMRES
    KSPType kspObjectType = KSPGMRES;

    KSPSetType(ksp, kspObjectType);
    // Set the gmres restart
    PetscInt restartGMRES = gmresRestart;

    // whether to use non-zero initial guess
    if (useNonZeroInitGuess)
    {
        KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    }
    else
    {
        KSPSetInitialGuessNonzero(ksp, PETSC_FALSE);
    }

    KSPGMRESSetRestart(ksp, restartGMRES);
    // Set the GMRES refinement type
    KSPGMRESSetCGSRefinementType(ksp, KSP_GMRES_CGS_REFINE_IFNEEDED);

    // set orthogonalization for the GMRES, useMGSO=1: modified Gram Schmidt
    // useMGSO=0: classical Gram Schmidt
    if (useMGSO)
    {
        KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization);
    }

    // Set the preconditioner side
    KSPSetPCSide(ksp, PC_RIGHT);

    // Set global and local PC iters
    PetscInt globalPreConIts = globalPCIters;

    // Since there is an extraneous matMult required when using the
    // richardson precondtiter with only 1 iteration, only use it when we need
    // to do more than 1 iteration.
    if (globalPreConIts > 1)
    {
        // Extract preconditioning context for main KSP solver: (MLRMasterPC)
        KSPGetPC(ksp, &MLRMasterPC);

        // Set the type of MLRMasterPC to ksp. This lets us do multiple
        // iterations of preconditioner application
        PCSetType(MLRMasterPC, PCKSP);

        // Get the ksp context from MLRMasterPC which is the actual preconditioner:
        PCKSPGetKSP(MLRMasterPC, &MLRMasterPCKSP);

        // MLRMasterPCKSP type will always be of type richardson. If the
        // number  of iterations is set to 1, this ksp object is transparent.
        KSPSetType(MLRMasterPCKSP, KSPRICHARDSON);

        // Important to set the norm-type to None for efficiency.
        KSPSetNormType(MLRMasterPCKSP, KSP_NORM_NONE);

        // Do one iteration of the outer ksp preconditioners. Note the
        // tolerances are unsued since we have set KSP_NORM_NONE
        KSPSetTolerances(MLRMasterPCKSP, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, globalPreConIts);

        // Get the 'preconditioner for MLRMasterPCKSP, called 'MLRGlobalPC'. This
        // preconditioner is potentially run multiple times.
        KSPGetPC(MLRMasterPCKSP, &MLRGlobalPC);
    }
    else
    {
        // Just pull out the pc-object if we are not using kspRichardson
        KSPGetPC(ksp, &MLRGlobalPC);
    }

    // Set the type of 'MLRGlobalPC'. This will almost always be additive schwartz
    PCSetType(MLRGlobalPC, PCASM);

    // Set the overlap required
    MLRoverlap = asmOverlap;
    PCASMSetOverlap(MLRGlobalPC, MLRoverlap);

    //label KSPCalcEigen = readLabel(options.lookup("KSPCalcEigen"));
    //if (KSPCalcEigen)
    //{
    //    KSPSetComputeEigenvalues(*genksp, PETSC_TRUE);
    //}

    //Setup the main ksp context before extracting the subdomains
    KSPSetUp(ksp);

    // Extract the ksp objects for each subdomain
    PCASMGetSubKSP(MLRGlobalPC, &MLRnlocal, &MLRfirst, &MLRsubksp);

    //Loop over the local blocks, setting various KSP options
    //for each block.
    PetscInt localPreConIts = localPCIters;
    word matOrdering = jacMatReOrdering;
    PetscInt localFillLevel = pcFillLevel;
    for (PetscInt i = 0; i < MLRnlocal; i++)
    {
        // Since there is an extraneous matMult required when using the
        // richardson precondtiter with only 1 iteration, only use it we need
        // to do more than 1 iteration.
        if (localPreConIts > 1)
        {
            // This 'subksp' object will ALSO be of type richardson so we can do
            // multiple iterations on the sub-domains
            KSPSetType(MLRsubksp[i], KSPRICHARDSON);

            // Set the number of iterations to do on local blocks. Tolerances are ignored.
            KSPSetTolerances(MLRsubksp[i], PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, localPreConIts);

            // Again, norm_type is NONE since we don't want to check error
            KSPSetNormType(MLRsubksp[i], KSP_NORM_NONE);
        }
        else
        {
            KSPSetType(MLRsubksp[i], KSPPREONLY);
        }

        // Extract the preconditioner for subksp object.
        KSPGetPC(MLRsubksp[i], &MLRsubpc);

        // The subpc type will almost always be ILU
        PCType localPCType = PCILU;
        PCSetType(MLRsubpc, localPCType);

        // Set PC factor
        PCFactorSetPivotInBlocks(MLRsubpc, PETSC_TRUE);
        PCFactorSetShiftType(MLRsubpc, MAT_SHIFT_NONZERO);
        PCFactorSetShiftAmount(MLRsubpc, PETSC_DECIDE);

        // Setup the matrix ordering for the subpc object:
        // 'natural':'natural',
        // 'rcm':'rcm',
        // 'nested dissection':'nd' (default),
        // 'one way dissection':'1wd',
        // 'quotient minimum degree':'qmd',
        MatOrderingType localMatrixOrdering;
        if (matOrdering == "natural")
        {
            localMatrixOrdering = MATORDERINGNATURAL;
        }
        else if (matOrdering == "nd")
        {
            localMatrixOrdering = MATORDERINGND;
        }
        else if (matOrdering == "rcm")
        {
            localMatrixOrdering = MATORDERINGRCM;
        }
        else if (matOrdering == "1wd")
        {
            localMatrixOrdering = MATORDERING1WD;
        }
        else if (matOrdering == "qmd")
        {
            localMatrixOrdering = MATORDERINGQMD;
        }
        else
        {
            Info << "matOrdering not known. Using default: nested dissection" << endl;
            localMatrixOrdering = MATORDERINGND;
        }
        PCFactorSetMatOrderingType(MLRsubpc, localMatrixOrdering);

        // Set the ILU parameters
        PCFactorSetLevels(MLRsubpc, localFillLevel);
    }

    // Set the norm to unpreconditioned
    KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED);
    // Setup monitor if necessary:
    if (printInfo)
    {
        KSPMonitorSet(ksp, myKSPMonitor, this, 0);
    }

    PetscInt maxIts = gmresMaxIters;
    PetscScalar rtol, atol;
    assignValueCheckAD(rtol, gmresRelTol);
    assignValueCheckAD(atol, gmresAbsTol);
    KSPSetTolerances(ksp, rtol, atol, PETSC_DEFAULT, maxIts);

    if (printInfo)
    {
        Info << "Solver Type: " << kspObjectType << endl;
        Info << "GMRES Restart: " << restartGMRES << endl;
        Info << "ASM Overlap: " << MLRoverlap << endl;
        Info << "Global PC Iters: " << globalPreConIts << endl;
        Info << "Local PC Iters: " << localPreConIts << endl;
        Info << "Mat ReOrdering: " << matOrdering << endl;
        Info << "ILU PC Fill Level: " << localFillLevel << endl;
        Info << "GMRES Max Iterations: " << maxIts << endl;
        Info << "GMRES Relative Tolerance: " << rtol << endl;
        Info << "GMRES Absolute Tolerance: " << atol << endl;
    }
}

label DALinearEqn::solveLinearEqn(
    const KSP ksp,
    const Vec rhsVec,
    Vec solVec)
{
    /*
    Description:
        Solve a linear equation.
    
    Input:
        ksp: the KSP object, obtained from calling Foam::createMLRKSP

        rhsVec: the right-hand-side petsc vector

    Output:
        solVec: the solution vector

        Return 0 if the linear equation solution finished successfully otherwise return 1
    */

    Info << "Solving Linear Equation... " << this->getRunTime() << " s" << endl;

    //Solve adjoint
    // VecZeroEntries(solVec);

    // set up rGMRESHist to save the tolerance history for the GMRES solution
    // these vars are for store the tolerance for GMRES linear solution
    label gmresMaxIters = daOption_.getSubDictOption<label>("adjEqnOption", "gmresMaxIters");
    PetscScalar rGMRESHist[gmresMaxIters + 1];
    label nGMRESIters = gmresMaxIters + 1;
    KSPSetResidualHistory(ksp, rGMRESHist, nGMRESIters, PETSC_TRUE);

    // solve KSP
    KSPSolve(ksp, rhsVec, solVec);

    //Print convergence information
    label its;
    KSPGetIterationNumber(ksp, &its);
    PetscScalar initResNorm = rGMRESHist[0];
    PetscScalar finalResNorm = rGMRESHist[its];
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    PetscPrintf(
        PETSC_COMM_WORLD,
        "Main iteration %D KSP Residual norm %14.12e %.2f s \n",
        its,
        finalResNorm,
        this->getRunTime());

    Info << "**Completed**! Total iterations: " << its
         << ". PetscConvergedReason: " << reason << ". " << this->getRunTime() << " s" << endl;

    VecAssemblyBegin(solVec);
    VecAssemblyEnd(solVec);

    // now we need to check if the linear equation solution is successful

    scalar absResRatio = finalResNorm / daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresAbsTol");
    scalar relResRatio = finalResNorm / initResNorm / daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresRelTol");
    scalar resDiff = daOption_.getSubDictOption<scalar>("adjEqnOption", "gmresTolDiff");
    if (relResRatio > resDiff && absResRatio > resDiff)
    {
        Info << "Residual tolerance not satisfied, solution failed!" << endl;
        return 1;
    }
    else
    {
        Info << "Residual tolerance satisfied, solution finished!" << endl;
        return 0;
    }

    return 1;
}

PetscErrorCode DALinearEqn::myKSPMonitor(
    KSP ksp,
    PetscInt n,
    PetscReal rnorm,
    void* ctx)
{

    /*
    Descripton:
        Write the solution vector and residual norm to stdout.
        - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
        - The parallel viewer PETSC_VIEWER_STDOUT_WORLD handles
        data from multiple processors so that the output
        is not jumbled.
    */

    DALinearEqn* daLinearEqn = (DALinearEqn*)ctx;

    // residual print frequency
    PetscInt printFrequency = daLinearEqn->getPrintInterval();
    PetscScalar runTime = daLinearEqn->getRunTime();
    if (n % printFrequency == 0)
    {
        PetscPrintf(
            PETSC_COMM_WORLD,
            "Main iteration %D KSP Residual norm %14.12e %.2f s\n",
            n,
            rnorm,
            runTime);
    }
    return 0;
}

double DALinearEqn::getRunTime()
{
    /*
    Descripton:
        Return the runtime
    */
    return mesh_.time().elapsedCpuTime();
}

label DALinearEqn::getPrintInterval()
{
    /*
    Descripton:
        Return the printInterval from DAOption
    */
    return daOption_.getOption<label>("printInterval");
}

} // End namespace Foam

// ************************************************************************* //
