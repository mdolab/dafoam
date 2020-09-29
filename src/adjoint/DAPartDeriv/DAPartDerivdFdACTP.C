/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdFdACTP.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdFdACTP, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdFdACTP, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdFdACTP::DAPartDerivdFdACTP(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAJacCon& daJacCon,
    const DAResidual& daResidual)
    : DAPartDeriv(modelType,
                  mesh,
                  daOption,
                  daModel,
                  daIndex,
                  daJacCon,
                  daResidual)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAPartDerivdFdACTP::initializePartDerivMat(
    const dictionary& options,
    Mat* jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options. This is not used
    */

    // create dFdACTP
    MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        *jacMat,
        PETSC_DECIDE,
        PETSC_DECIDE,
        1,
        nActPointDVs_);
    MatSetFromOptions(*jacMat);
    MatMPIAIJSetPreallocation(*jacMat, nActPointDVs_, NULL, nActPointDVs_, NULL);
    MatSeqAIJSetPreallocation(*jacMat, nActPointDVs_, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(*jacMat);
    MatZeroEntries(*jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdFdACTP::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. Note for dFdACTP, we do brute force finite-difference 
        there is no need to do coloring
    
    Input:
        options.objFuncSubDictPart: the objFunc subDict, obtained from DAOption

        options.objFuncName: the name of the objective

        options.objFuncPart: the part of the objective

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dFdACTP to compute
    */

    word actuatorPointName = options.getWord("actuatorPointName");

    word objFuncName, objFuncPart;
    dictionary objFuncSubDictPart = options.subDict("objFuncSubDictPart");
    options.readEntry<word>("objFuncName", objFuncName);
    options.readEntry<word>("objFuncPart", objFuncPart);

    autoPtr<DAObjFunc> daObjFunc(
        DAObjFunc::New(
            mesh_,
            daOption_,
            daModel_,
            daIndex_,
            daResidual_,
            objFuncName,
            objFuncPart,
            objFuncSubDictPart)
            .ptr());

    // zero all the matrices
    MatZeroEntries(jacMat);

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 0);
    scalar fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "ACTP");
    scalar rDelta = 1.0 / delta;

    dictionary& pointModelSubDict = const_cast<dictionary&>(
        daOption_.getAllOptions().subDict("fvSource").subDict(actuatorPointName));

    scalarList center, amplitude;
    pointModelSubDict.readEntry<scalarList>("center", center);
    pointModelSubDict.readEntry<scalarList>("amplitude", amplitude);
    scalar periodicity = pointModelSubDict.getScalar("periodicity");
    scalar phase = pointModelSubDict.getScalar("phase");
    scalar scale = pointModelSubDict.getScalar("scale");

    for (label i = 0; i < nActPointDVs_; i++)
    {

        // perturb ACTP
        if (i == 0)
        {
            // perturb x
            center[0] += delta;
            pointModelSubDict.set("center", center);
        }
        else if (i == 1)
        {
            // perturb y
            center[1] += delta;
            pointModelSubDict.set("center", center);
        }
        else if (i == 2)
        {
            // perturb z
            center[2] += delta;
            pointModelSubDict.set("center", center);
        }
        else if (i == 3)
        {
            // perturb amplitude x
            amplitude[0] += delta;
            pointModelSubDict.set("amplitude", amplitude);
        }
        else if (i == 4)
        {
            // perturb amplitude y
            amplitude[1] += delta;
            pointModelSubDict.set("amplitude", amplitude);
        }
        else if (i == 5)
        {
            // perturb amplitude z
            amplitude[2] += delta;
            pointModelSubDict.set("amplitude", amplitude);
        }
        else if (i == 6)
        {
            // perturb periodicity
            periodicity += delta;
            pointModelSubDict.set("periodicity", periodicity);
        }
        else if (i == 7)
        {
            // perturb phase
            phase += delta;
            pointModelSubDict.set("phase", phase);
        }
        else if (i == 8)
        {
            // perturb scale
            scale += delta;
            pointModelSubDict.set("scale", scale);
        }

        // compute object
        scalar fNew = daObjFunc->masterFunction(mOptions, xvVec, wVec);

        scalar partDeriv = (fNew - fRef) * rDelta;

        MatSetValue(jacMat, 0, i, partDeriv, INSERT_VALUES);

        // reset perturbation
        if (i == 0)
        {
            // reset x
            center[0] -= delta;
            pointModelSubDict.set("center", center);
        }
        else if (i == 1)
        {
            // reset y
            center[1] -= delta;
            pointModelSubDict.set("center", center);
        }
        else if (i == 2)
        {
            // reset z
            center[2] -= delta;
            pointModelSubDict.set("center", center);
        }
        else if (i == 3)
        {
            // reset amplitude x
            amplitude[0] -= delta;
            pointModelSubDict.set("amplitude", amplitude);
        }
        else if (i == 4)
        {
            // reset amplitude y
            amplitude[1] -= delta;
            pointModelSubDict.set("amplitude", amplitude);
        }
        else if (i == 5)
        {
            // reset amplitude z
            amplitude[2] -= delta;
            pointModelSubDict.set("amplitude", amplitude);
        }
        else if (i == 6)
        {
            // reset periodicity
            periodicity -= delta;
            pointModelSubDict.set("periodicity", periodicity);
        }
        else if (i == 7)
        {
            // reset phase
            phase -= delta;
            pointModelSubDict.set("phase", phase);
        }
        else if (i == 8)
        {
            // reset scale
            scale -= delta;
            pointModelSubDict.set("scale", scale);
        }
    }

    // call masterFunction again to reset the wVec to OpenFOAM field
    fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    if (daOption_.getOption<label>("debug"))
    {
        Info << objFuncName << ": " << fRef << endl;
    }

    label eTime = mesh_.time().elapsedClockTime();

    Info << modelType_ << " ExecutionTime: " << eTime << " s" << endl;

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
