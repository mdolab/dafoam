/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAPartDeriv.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAPartDeriv::DAPartDeriv(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAJacCon& daJacCon,
    const DAResidual& daResidual)
    : modelType_(modelType),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daJacCon_(daJacCon),
      daResidual_(daResidual),
      allOptions_(daOption.getAllOptions())
{
    // initialize stateInfo_
    word solverName = daOption.getOption<word>("solverName");
    autoPtr<DAStateInfo> daStateInfo(DAStateInfo::New(solverName, mesh, daOption, daModel));
    stateInfo_ = daStateInfo->getStateInfo();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAPartDeriv::perturbStates(
    const Vec jacConColors,
    const Vec normStatePerturbVec,
    const label colorI,
    const scalar delta,
    Vec wVec)
{
    /*
    Descripton:
        Perturb state variable vec such that it can be used to compute 
        perturbed residual vector later
    
    Input:
        jacConColors: the coloring vector for this Jacobian, obtained from Foam::DAJacCon
    
        normStatePerturbVec: state normalization vector, 1.0 means no normalization, the 
        actually perturbation added on the wVec is normStatePerturbVec * delta

        colorI: we perturb the rows that associated with the coloring index colorI

        delta: the delta value for perturbation the actually perturbation added on 
        the wVec is normStatePerturbVec * delta

    Input/Output:
        wVec: the perturbed state variable vector

    Example:
        Assuming we have five element in wVec {0.1, 0.5, 1.0, 1.5, 2.0}
        If jacConColors reads {0, 0, 1, 0, 1},
        normStatePerturbVec reads {1.0, 1.0, 0.5, 1.0, 0.1}
        colorI = 1, delta = 0.01

        Then after calling this function wVec reads
        {0.1, 0.5, 1.005, 1.5, 2.001}
    */
    PetscInt Istart, Iend;
    VecGetOwnershipRange(jacConColors, &Istart, &Iend);

    const PetscScalar* colorArray;
    VecGetArrayRead(jacConColors, &colorArray);

    const PetscScalar* normStateArray;
    VecGetArrayRead(normStatePerturbVec, &normStateArray);

    PetscScalar* wVecArray;
    VecGetArray(wVec, &wVecArray);

    PetscScalar deltaVal = 0.0;
    assignValueCheckAD(deltaVal, delta);

    for (label i = Istart; i < Iend; i++)
    {
        label relIdx = i - Istart;
        label colorJ = colorArray[relIdx];
        if (colorI == colorJ)
        {
            wVecArray[relIdx] += deltaVal * normStateArray[relIdx];
        }
    }

    VecRestoreArrayRead(jacConColors, &colorArray);
    VecRestoreArray(wVec, &wVecArray);
    VecRestoreArrayRead(normStatePerturbVec, &normStateArray);

    return;
}

void DAPartDeriv::setPartDerivMat(
    const Vec resVec,
    const Vec coloredColumn,
    const label transposed,
    Mat jacMat,
    const scalar jacLowerBound) const
{
    /*
    Description:
        Set the values from resVec to jacMat
    
    Input:
        resVec: residual vector, obtained after calling the DAResidual::masterFunction

        coloredColumn: a vector to determine the element in resVec is associated with 
        which column in the jacMat. coloredColumn is computed in DAJacCon::calcColoredColumns
        -1 in coloredColumn means don't set values to jacMat for this column

        transposed: whether the jacMat is transposed or not, for dRdWT transpoed = 1, for 
        all the other cases, set it to 0

        jacLowerBound: any |value| that is smaller than lowerBound will be set to zero in PartDerivMat

    Output:
        jacMat: the jacobian matrix to set

    Example:
        Condering a 5 by 5 jacMat with all zeros, and

        resVec
        {
            0.1,
            0.2,
            0.3,
            0.4,
            0.5
        }

        coloredColumn
        {
            -1
            1
            3
            -1
            0
        }

        transposed = 0

        Then, after calling this function, jacMat reads

        0.0  0.0  0.0  0.0  0.0  
        0.0  0.2  0.0  0.0  0.0  
        0.0  0.0  0.0  0.3  0.0  
        0.0  0.0  0.0  0.0  0.0  
        0.5  0.0  0.0  0.0  0.0  
    */

    label rowI, colI;
    PetscScalar val;
    PetscInt Istart, Iend;
    const PetscScalar* resVecArray;
    const PetscScalar* coloredColumnArray;

    VecGetArrayRead(resVec, &resVecArray);
    VecGetArrayRead(coloredColumn, &coloredColumnArray);

    // get the local ownership range
    VecGetOwnershipRange(resVec, &Istart, &Iend);

    // Loop over the owned values of this row and set the corresponding
    // Jacobian entries
    for (PetscInt i = Istart; i < Iend; i++)
    {
        label relIdx = i - Istart;
        colI = coloredColumnArray[relIdx];
        if (colI >= 0)
        {
            rowI = i;
            val = resVecArray[relIdx];
            // if val < bound, don't set the matrix. The exception is that
            // we always set values for diagonal elements (colI==rowI)
            // Another exception is that jacLowerBound is less than 1e-16
            if (jacLowerBound < 1.0e-16 || fabs(val) > jacLowerBound || colI == rowI)
            {
                if (transposed)
                {
                    MatSetValue(jacMat, colI, rowI, val, INSERT_VALUES);
                }
                else
                {
                    MatSetValue(jacMat, rowI, colI, val, INSERT_VALUES);
                }
            }
        }
    }

    VecRestoreArrayRead(resVec, &resVecArray);
    VecRestoreArrayRead(coloredColumn, &coloredColumnArray);
}

void DAPartDeriv::setNormStatePerturbVec(Vec* normStatePerturbVec)
{
    /*
    Description:
        Set the values for the normStatePerturbVec

    Input/Output:
        normStatePerturbVec: the vector to store the state normalization
        values, this will be used in DAPartDeriv::perturbStates

    The normalization referene values are set in "normalizeStates" in DAOption
    */
    label localSize = daIndex_.nLocalAdjointStates;
    VecCreate(PETSC_COMM_WORLD, normStatePerturbVec);
    VecSetSizes(*normStatePerturbVec, localSize, PETSC_DETERMINE);
    VecSetFromOptions(*normStatePerturbVec);
    VecSet(*normStatePerturbVec, 1.0);

    dictionary normStateDict = allOptions_.subDict("normalizeStates");

    wordList normStateNames = normStateDict.toc();

    forAll(stateInfo_["volVectorStates"], idxI)
    {
        word stateName = stateInfo_["volVectorStates"][idxI];
        if (normStateNames.found(stateName))
        {
            scalar scale = normStateDict.getScalar(stateName);
            PetscScalar scaleValue = 0.0;
            assignValueCheckAD(scaleValue, scale);
            forAll(mesh_.cells(), idxI)
            {
                for (label comp = 0; comp < 3; comp++)
                {
                    label glbIdx = daIndex_.getGlobalAdjointStateIndex(stateName, idxI, comp);
                    VecSetValue(*normStatePerturbVec, glbIdx, scaleValue, INSERT_VALUES);
                }
            }
        }
    }

    forAll(stateInfo_["volScalarStates"], idxI)
    {
        const word stateName = stateInfo_["volScalarStates"][idxI];
        if (normStateNames.found(stateName))
        {
            scalar scale = normStateDict.getScalar(stateName);
            PetscScalar scaleValue = 0.0;
            assignValueCheckAD(scaleValue, scale);
            forAll(mesh_.cells(), idxI)
            {
                label glbIdx = daIndex_.getGlobalAdjointStateIndex(stateName, idxI);
                VecSetValue(*normStatePerturbVec, glbIdx, scaleValue, INSERT_VALUES);
            }
        }
    }

    forAll(stateInfo_["modelStates"], idxI)
    {
        const word stateName = stateInfo_["modelStates"][idxI];
        if (normStateNames.found(stateName))
        {
            scalar scale = normStateDict.getScalar(stateName);
            PetscScalar scaleValue = 0.0;
            assignValueCheckAD(scaleValue, scale);
            forAll(mesh_.cells(), idxI)
            {
                label glbIdx = daIndex_.getGlobalAdjointStateIndex(stateName, idxI);
                VecSetValue(*normStatePerturbVec, glbIdx, scaleValue, INSERT_VALUES);
            }
        }
    }

    forAll(stateInfo_["surfaceScalarStates"], idxI)
    {
        const word stateName = stateInfo_["surfaceScalarStates"][idxI];
        if (normStateNames.found(stateName))
        {
            scalar scale = normStateDict.getScalar(stateName);
            PetscScalar scaleValue = 0.0;

            forAll(mesh_.faces(), faceI)
            {
                if (faceI < daIndex_.nLocalInternalFaces)
                {
                    scale = mesh_.magSf()[faceI];
                    assignValueCheckAD(scaleValue, scale);
                }
                else
                {
                    label relIdx = faceI - daIndex_.nLocalInternalFaces;
                    label patchIdx = daIndex_.bFacePatchI[relIdx];
                    label faceIdx = daIndex_.bFaceFaceI[relIdx];
                    scale = mesh_.magSf().boundaryField()[patchIdx][faceIdx];
                    assignValueCheckAD(scaleValue, scale);
                }

                label glbIdx = daIndex_.getGlobalAdjointStateIndex(stateName, faceI);
                VecSetValue(*normStatePerturbVec, glbIdx, scaleValue, INSERT_VALUES);
            }
        }
    }

    VecAssemblyBegin(*normStatePerturbVec);
    VecAssemblyEnd(*normStatePerturbVec);
}

void DAPartDeriv::initializePartDerivMat(
    const dictionary& options,
    Mat jacMat)
{
    /*
    Description:
        Initialize jacMat
    
    Input:
        options.transposed. Whether to compute the transposed of dRdW
    */

    label transposed = options.getLabel("transposed");

    // now initialize the memory for the jacobian itself
    label localSize = daIndex_.nLocalAdjointStates;

    // create dRdWT
    //MatCreate(PETSC_COMM_WORLD, jacMat);
    MatSetSizes(
        jacMat,
        localSize,
        localSize,
        PETSC_DETERMINE,
        PETSC_DETERMINE);
    MatSetFromOptions(jacMat);
    daJacCon_.preallocatedRdW(jacMat, transposed);
    //MatSetOption(jacMat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(jacMat);
    MatZeroEntries(jacMat);
    Info << "Partial derivative matrix created. " << mesh_.time().elapsedCpuTime() << " s" << endl;
}

void DAPartDeriv::calcPartDerivMat(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Mat jacMat)
{
    /*
    Description:
        Compute jacMat. We use coloring accelerated finite-difference
    
    Input:

        options.transposed. Whether to compute the transposed of dRdW

        options.isPC: whether to compute the jacMat for preconditioner

        options.lowerBound: any |value| that is smaller than lowerBound will be set to zero in dRdW

        xvVec: the volume mesh coordinate vector

        wVec: the state variable vector
    
    Output:
        jacMat: the partial derivative matrix dRdW to compute
    */

    label transposed = options.getLabel("transposed");

    // initialize coloredColumn vector
    Vec coloredColumn;
    VecDuplicate(wVec, &coloredColumn);
    VecZeroEntries(coloredColumn);

    DAResidual& daResidual = const_cast<DAResidual&>(daResidual_);

    // zero all the matrices
    MatZeroEntries(jacMat);

    Vec wVecNew;
    VecDuplicate(wVec, &wVecNew);
    VecCopy(wVec, wVecNew);

    // initialize residual vectors
    Vec resVecRef, resVec;
    VecDuplicate(wVec, &resVec);
    VecDuplicate(wVec, &resVecRef);
    VecZeroEntries(resVec);
    VecZeroEntries(resVecRef);

    // set up state normalization vector
    Vec normStatePerturbVec;
    this->setNormStatePerturbVec(&normStatePerturbVec);

    dictionary mOptions;
    mOptions.set("updateState", 1);
    mOptions.set("updateMesh", 0);
    mOptions.set("setResVec", 1);
    mOptions.set("isPC", options.getLabel("isPC"));
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    scalar jacLowerBound = options.getScalar("lowerBound");

    scalar delta = daOption_.getSubDictOption<scalar>("adjPartDerivFDStep", "State");
    scalar rDelta = 1.0 / delta;
    PetscScalar rDeltaValue = 0.0;
    assignValueCheckAD(rDeltaValue, rDelta);

    label nColors = daJacCon_.getNJacConColors();

    word partDerivName = modelType_;
    if (transposed)
    {
        partDerivName += "T";
    }
    if (options.getLabel("isPC"))
    {
        partDerivName += "PC";
    }

    label printInterval = daOption_.getOption<label>("printInterval");
    for (label color = 0; color < nColors; color++)
    {
        scalar eTime = mesh_.time().elapsedCpuTime();
        // print progress
        if (color % printInterval == 0 or color == nColors - 1)
        {
            Info << partDerivName << ": " << color << " of " << nColors
                 << ", ExecutionTime: " << eTime << " s" << endl;
        }

        // perturb states
        this->perturbStates(
            daJacCon_.getJacConColor(),
            normStatePerturbVec,
            color,
            delta,
            wVecNew);

        // compute residual
        daResidual.masterFunction(mOptions, xvVec, wVecNew, resVec);

        // reset state perburbation
        VecCopy(wVec, wVecNew);

        // compute residual partial using finite-difference
        VecAXPY(resVec, -1.0, resVecRef);
        VecScale(resVec, rDeltaValue);

        // compute the colored coloumn and assign resVec to jacMat
        daJacCon_.calcColoredColumns(color, coloredColumn);
        this->setPartDerivMat(resVec, coloredColumn, transposed, jacMat, jacLowerBound);
    }

    // call masterFunction again to reset the wVec to OpenFOAM field
    daResidual.masterFunction(mOptions, xvVec, wVec, resVecRef);

    MatAssemblyBegin(jacMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jacMat, MAT_FINAL_ASSEMBLY);

    if (daOption_.getOption<label>("debug"))
    {
        daIndex_.printMatChars(jacMat);
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
