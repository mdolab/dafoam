/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAPartDerivdFdACTD.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAPartDerivdFdACTD, 0);
addToRunTimeSelectionTable(DAPartDeriv, DAPartDerivdFdACTD, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDerivdFdACTD::DAPartDerivdFdACTD(
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

void DAPartDerivdFdACTD::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options. This is not used
    */

    // create dFdACTD
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        PETSC_DECIDE,
        PETSC_DECIDE,
        nActDVs_,
        1
        );
    MatSetFromOptions(jacMat);
    MatMPIAIJSetPreallocation(jacMat, 1, NULL, 1, NULL);
    MatSeqAIJSetPreallocation(jacMat, 1, NULL);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedClockTime() << " s" << endl;
}

void DAPartDerivdFdACTD::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. Note for dFdACTD, we have only one column so we can do brute 
        force finite-difference there is no need to do coloring
    
    Input:
        options.objFuncSubDictPart: the objFunc subDict, obtained from DAOption

        options.objFuncName: the name of the objective

        options.objFuncPart: the part of the objective

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dFdACTD to compute
    */

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

    word actuatorName = options.getWord("actuatorName");

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 0);
    scalar fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "ACTD");
    scalar rDelta = 1.0 / delta;

    dictionary& diskModelSubDict = const_cast<dictionary&>(
        daOption_.getAllOptions().subDict("fvSource").subDict(actuatorName));

    scalarList center;
    diskModelSubDict.readEntry<scalarList>("center", center);
    scalar innerRadius = diskModelSubDict.getScalar("innerRadius");
    scalar outerRadius = diskModelSubDict.getScalar("outerRadius");
    scalar POD = diskModelSubDict.getScalar("POD");
    scalar scale = diskModelSubDict.getScalar("scale");
    scalar expM = diskModelSubDict.getScalar("expM");
    scalar expN = diskModelSubDict.getScalar("expN");

    DAFvSource& fvSource = const_cast<DAFvSource&>(
        mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource"));

    for (label i = 0; i < nActDVs_; i++)
    {
        // perturb ACTD
        if (i == 0)
        {
            // perturb x
            center[0] += delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 1)
        {
            // perturb y
            center[1] += delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 2)
        {
            // perturb z
            center[2] += delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 3)
        {
            // perturb innerRadius
            innerRadius += delta;
            diskModelSubDict.set("innerRadius", innerRadius);
        }
        else if (i == 4)
        {
            // perturb outerRadius
            outerRadius += delta;
            diskModelSubDict.set("outerRadius", outerRadius);
        }
        else if (i == 5)
        {
            // perturb scale
            scale += delta;
            diskModelSubDict.set("scale", scale);
        }
        else if (i == 6)
        {
            // perturb POD
            POD += delta;
            diskModelSubDict.set("POD", POD);
        }
        else if (i == 7)
        {
            // perturb expM
            expM += delta;
            diskModelSubDict.set("expM", expM);
        }
        else if (i == 8)
        {
            // perturb expN
            expN += delta;
            diskModelSubDict.set("expN", expN);
        }

        // we need to synchronize the DAOption to actuatorDVs
        fvSource.syncDAOptionToActuatorDVs();
        fvSource.updateFvSource();

        // compute object
        scalar fNew = daObjFunc->masterFunction(mOptions, xvVec, wVec);

        // reset perturbation
        if (i == 0)
        {
            // reset x
            center[0] -= delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 1)
        {
            // reset y
            center[1] -= delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 2)
        {
            // reset z
            center[2] -= delta;
            diskModelSubDict.set("center", center);
        }
        else if (i == 3)
        {
            // reset innerRadius
            innerRadius -= delta;
            diskModelSubDict.set("innerRadius", innerRadius);
        }
        else if (i == 4)
        {
            // reset outerRadius
            outerRadius -= delta;
            diskModelSubDict.set("outerRadius", outerRadius);
        }
        else if (i == 5)
        {
            // reset scale
            scale -= delta;
            diskModelSubDict.set("scale", scale);
        }
        else if (i == 6)
        {
            // reset POD
            POD -= delta;
            diskModelSubDict.set("POD", POD);
        }
        else if (i == 7)
        {
            // reset expM
            expM -= delta;
            diskModelSubDict.set("expM", expM);
        }
        else if (i == 8)
        {
            // reset expN
            expN -= delta;
            diskModelSubDict.set("expN", expN);
        }

        // we need to synchronize the DAOption to actuatorDVs
        fvSource.syncDAOptionToActuatorDVs();
        fvSource.updateFvSource();

        scalar partDeriv = (fNew - fRef) * rDelta;
        PetscScalar partDerivValue = 0.0;
        assignValueCheckAD(partDerivValue, partDeriv);

        MatSetValue(jacMat, i, 0, partDerivValue, INSERT_VALUES);

    }

    // call masterFunction again to reset the wVec to OpenFOAM field
    fRef = daObjFunc->masterFunction(mOptions, xvVec, wVec);

    if (daOption_.getOption<label>("debug"))
    {
        Info << objFuncName << ": " << fRef << endl;
    }

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);
}

} // End namespace Foam

// ************************************************************************* //
