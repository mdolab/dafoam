/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAOutputForceCoupling.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAOutputForceCoupling, 0);
addToRunTimeSelectionTable(DAOutput, DAOutputForceCoupling, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAOutputForceCoupling::DAOutputForceCoupling(
    const word outputName,
    const word outputType,
    fvMesh& mesh,
    const DAOption& daOption,
    DAModel& daModel,
    const DAIndex& daIndex,
    DAResidual& daResidual,
    UPtrList<DAFunction>& daFunctionList)
    : DAOutput(
        outputName,
        outputType,
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        daFunctionList)
{
    daOption_.getAllOptions().subDict("outputInfo").subDict(outputName_).readEntry("patches", patches_);
    pRef_ = daOption_.getAllOptions().subDict("outputInfo").subDict(outputName_).getScalar("pRef");
    // NOTE: always sort the patch because the order of the patch element matters in FSI coupling
    sort(patches_);

    // calculate how many nodal points are in the above patches
    // NOTE: we will have duplicated points for parallel cases
    size_ = 0;
    const pointMesh& pMesh = pointMesh::New(mesh_);
    const pointBoundaryMesh& boundaryMesh = pMesh.boundary();
    forAll(patches_, cI)
    {
        // Get number of points in patch
        word patchName = patches_[cI];
        label patchIPoints = boundaryMesh.findPatchID(patchName);
        size_ += boundaryMesh[patchIPoints].size();
    }
    // we have x, y, z coords for each point
    size_ *= 3;
}

void DAOutputForceCoupling::run(scalarList& output)
{
    /*
    Description:
        Assign output based on OF fields
    */

    scalarList fX(size_ / 3);
    scalarList fY(size_ / 3);
    scalarList fZ(size_ / 3);

    // Initialize surface field for face-centered forces
    volVectorField volumeForceField(
        IOobject(
            "volumeForceField",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE),
        mesh_,
        dimensionedVector("surfaceForce", dimensionSet(1, 1, -2, 0, 0, 0, 0), vector::zero),
        "fixedValue");

    // this code is pulled from:
    // src/functionObjects/forces/forces.C
    // modified slightly
    vector force(vector::zero);

    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");

    const surfaceVectorField::Boundary& Sfb = mesh_.Sf().boundaryField();

    const DATurbulenceModel& daTurb = daModel_.getDATurbulenceModel();
    tmp<volSymmTensorField> tdevRhoReff = daTurb.devRhoReff();
    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff().boundaryField();

    const pointMesh& pMesh = pointMesh::New(mesh_);
    const pointBoundaryMesh& boundaryMesh = pMesh.boundary();

    // iterate over patches and extract boundary surface forces
    forAll(patches_, cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID(patches_[cI]);
        // create a shorter handle for the boundary patch
        const fvPatch& patch = mesh_.boundary()[patchI];
        // normal force
        vectorField fN(Sfb[patchI] * (p.boundaryField()[patchI] - pRef_));
        // tangential force
        vectorField fT(Sfb[patchI] & devRhoReffb[patchI]);
        // sum them up
        forAll(patch, faceI)
        {
            force.x() = fN[faceI].x() + fT[faceI].x();
            force.y() = fN[faceI].y() + fT[faceI].y();
            force.z() = fN[faceI].z() + fT[faceI].z();
            volumeForceField.boundaryFieldRef()[patchI][faceI] = force;
        }
    }
    volumeForceField.write();

    // The above volumeForceField is face-centered, we need to interpolate it to point-centered
    pointField meshPoints = mesh_.points();

    vector nodeForce(vector::zero);

    label patchStart = 0;
    forAll(patches_, cI)
    {
        // get the patch id label
        label patchI = mesh_.boundaryMesh().findPatchID(patches_[cI]);
        label patchIPoints = boundaryMesh.findPatchID(patches_[cI]);

        label nPointsPatch = boundaryMesh[patchIPoints].size();
        List<scalar> fXTemp(nPointsPatch);
        List<scalar> fYTemp(nPointsPatch);
        List<scalar> fZTemp(nPointsPatch);
        List<label> pointListTemp(nPointsPatch);
        pointListTemp = -1;

        label pointCounter = 0;
        // Loop over Faces
        forAll(mesh_.boundaryMesh()[patchI], faceI)
        {
            // Get number of points
            const label nPoints = mesh_.boundaryMesh()[patchI][faceI].size();

            // Divide force to nodes
            nodeForce = volumeForceField.boundaryFieldRef()[patchI][faceI] / nPoints;

            forAll(mesh_.boundaryMesh()[patchI][faceI], pointI)
            {
                // this is the index that corresponds to meshPoints, which contains both volume and surface points
                // so we can't directly reuse this index because we want to have only surface points
                label faceIPointIndexI = mesh_.boundaryMesh()[patchI][faceI][pointI];

                // Loop over pointListTemp array to check if this node is already included in this patch
                bool found = false;
                label iPoint = -1;
                for (label i = 0; i < pointCounter; i++)
                {
                    if (faceIPointIndexI == pointListTemp[i])
                    {
                        found = true;
                        iPoint = i;
                        break;
                    }
                }

                // If node is already included, add value to its entry
                if (found)
                {
                    // Add Force
                    fXTemp[iPoint] += nodeForce[0];
                    fYTemp[iPoint] += nodeForce[1];
                    fZTemp[iPoint] += nodeForce[2];
                }
                // If node is not already included, add it as the newest point and add global index mapping
                else
                {
                    // Add Force
                    fXTemp[pointCounter] = nodeForce[0];
                    fYTemp[pointCounter] = nodeForce[1];
                    fZTemp[pointCounter] = nodeForce[2];

                    // Add to Node Order Array
                    pointListTemp[pointCounter] = faceIPointIndexI;

                    // Increment counter
                    pointCounter += 1;
                }
            }
        }

        // Sort Patch Indices and Insert into Global Arrays
        SortableList<label> pointListSort(pointListTemp);
        forAll(pointListSort.indices(), indexI)
        {
            fX[patchStart + indexI] = fXTemp[pointListSort.indices()[indexI]];
            fY[patchStart + indexI] = fYTemp[pointListSort.indices()[indexI]];
            fZ[patchStart + indexI] = fZTemp[pointListSort.indices()[indexI]];
        }

        // Increment Patch Start Index
        patchStart += nPointsPatch;
    }

    label counterI = 0;
    forAll(fX, idxI)
    {
        output[counterI] = fX[idxI];
        output[counterI + 1] = fY[idxI];
        output[counterI + 2] = fZ[idxI];
        counterI += 3;
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
