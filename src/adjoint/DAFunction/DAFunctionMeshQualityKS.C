/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAFunctionMeshQualityKS.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFunctionMeshQualityKS, 0);
addToRunTimeSelectionTable(DAFunction, DAFunctionMeshQualityKS, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFunctionMeshQualityKS::DAFunctionMeshQualityKS(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const word functionName)
    : DAFunction(
        mesh,
        daOption,
        daModel,
        daIndex,
        functionName)
{

    functionDict_.readEntry<scalar>("coeffKS", coeffKS_);

    functionDict_.readEntry<word>("metric", metric_);

    // the polyMeshTools funcs are not fully AD in parallel, so the mesh quality
    // computed at the processor faces will not be properly back-propagate in AD
    // we can ignore the mesh quality at proc patches if includeProcPatches = False (default)
    includeProcPatches_ = functionDict_.lookupOrDefault<label>("includeProcPatches", 0);
    includeFaceList_.setSize(mesh_.nFaces(), 1);
    if (!includeProcPatches_)
    {
        label nInterF = mesh_.nInternalFaces();
        label faceCounter = 0;
        forAll(mesh.boundaryMesh(), patchI)
        {
            if (mesh.boundaryMesh()[patchI].type() == "processor")
            {
                forAll(mesh.boundaryMesh()[patchI], faceI)
                {
                    includeFaceList_[nInterF + faceCounter] = 0;
                    faceCounter += 1;
                }
            }
            else
            {
                faceCounter += mesh.boundaryMesh()[patchI].size();
            }
        }
    }

    if (daOption.getOption<label>("debug"))
    {
        Info << "includeFaceList " << includeFaceList_ << endl;
    }
}

/// calculate the value of objective function
scalar DAFunctionMeshQualityKS::calcFunction()
{
    /*
    Description:
        Calculate the mesh quality and aggregate with the KS function
        e.g., if metric is the faceSkewness, the function value will be the
        approximated max skewness
    */

    // initialize objFunValue
    scalar functionValue = 0.0;

    if (metric_ == "faceOrthogonality")
    {
        // faceOrthogonality ranges from 0 to 1 and the nonOrthoAngle is acos(faceOrthogonality)
        const scalarField faceOrthogonality(
            polyMeshTools::faceOrthogonality(
                mesh_,
                mesh_.faceAreas(),
                mesh_.cellCentres()));

        // calculate the KS mesh quality
        forAll(faceOrthogonality, faceI)
        {
            if (includeFaceList_[faceI] == 1)
            {
                functionValue += exp(coeffKS_ * faceOrthogonality[faceI]);
            }

            if (functionValue > 1e200)
            {
                FatalErrorIn(" ") << "KS function summation term too large! "
                                  << "Reduce coeffKS! " << abort(FatalError);
            }
        }
    }
    else if (metric_ == "nonOrthoAngle")
    {
        const scalarField faceOrthogonality(
            polyMeshTools::faceOrthogonality(
                mesh_,
                mesh_.faceAreas(),
                mesh_.cellCentres()));

        // Face based non ortho angle
        scalarField nonOrthoAngle = faceOrthogonality;
        forAll(faceOrthogonality, faceI)
        {
            scalar val = faceOrthogonality[faceI];
            // bound it to less than 1.0 - 1e-6. We can't let val = 1
            // because its derivative will be divided by zero
            scalar boundV = 1.0 - 1e-6;
            if (val > boundV)
            {
                val = boundV;
            }
            if (val < -boundV)
            {
                val = -boundV;
            }
            // compute non ortho angle
            scalar angleRad = acos(val);
            // convert rad to degree
            scalar pi = constant::mathematical::pi;
            scalar angleDeg = angleRad * 180.0 / pi;
            nonOrthoAngle[faceI] = angleDeg;
        }

        // calculate the KS mesh quality
        forAll(nonOrthoAngle, faceI)
        {
            if (includeFaceList_[faceI] == 1)
            {
                functionValue += exp(coeffKS_ * nonOrthoAngle[faceI]);
            }

            if (functionValue > 1e200)
            {
                FatalErrorIn(" ") << "KS function summation term too large! "
                                  << "Reduce coeffKS! " << abort(FatalError);
            }
        }
    }
    else if (metric_ == "faceSkewness")
    {
        const scalarField faceSkewness(
            polyMeshTools::faceSkewness(
                mesh_,
                mesh_.points(),
                mesh_.faceCentres(),
                mesh_.faceAreas(),
                mesh_.cellCentres()));

        // calculate the KS mesh quality
        forAll(faceSkewness, faceI)
        {
            if (includeFaceList_[faceI] == 1)
            {
                functionValue += exp(coeffKS_ * faceSkewness[faceI]);
            }

            if (functionValue > 1e200)
            {
                FatalErrorIn(" ") << "KS function summation term too large! "
                                  << "Reduce coeffKS! " << abort(FatalError);
            }
        }
    }
    else
    {
        FatalErrorIn(" ") << "metric not valid! "
                          << "Options: faceOrthogonality, nonOrthoAngle, or faceSkewness "
                          << abort(FatalError);
    }

    // need to reduce the sum of force across all processors
    reduce(functionValue, sumOp<scalar>());

    functionValue = log(functionValue) / coeffKS_;

    // check if we need to calculate refDiff.
    this->calcRefVar(functionValue);

    return functionValue;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
