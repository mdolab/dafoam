#include "writeFields.H"
#include "volFields.H"
#include "polyMeshTools.H"
#include "zeroGradientFvPatchFields.H"
#include "syncTools.H"
#include "tetPointRef.H"
#include "regionSplit.H"

using namespace Foam;

void maxFaceToCell
(
    const scalarField& faceData,
    volScalarField& cellData
)
{
    const cellList& cells = cellData.mesh().cells();

    scalarField& cellFld = cellData.ref();

    cellFld = -GREAT;
    forAll(cells, cellI)
    {
        const cell& cFaces = cells[cellI];
        forAll(cFaces, i)
        {
            cellFld[cellI] = max(cellFld[cellI], faceData[cFaces[i]]);
        }
    }

    forAll(cellData.boundaryField(), patchI)
    {
        fvPatchScalarField& fvp = cellData.boundaryFieldRef()[patchI];

        fvp = fvp.patch().patchSlice(faceData);
    }
    cellData.correctBoundaryConditions();
}


void minFaceToCell
(
    const scalarField& faceData,
    volScalarField& cellData
)
{
    const cellList& cells = cellData.mesh().cells();

    scalarField& cellFld = cellData.ref();

    cellFld = GREAT;
    forAll(cells, cellI)
    {
        const cell& cFaces = cells[cellI];
        forAll(cFaces, i)
        {
            cellFld[cellI] = min(cellFld[cellI], faceData[cFaces[i]]);
        }
    }

    forAll(cellData.boundaryField(), patchI)
    {
        fvPatchScalarField& fvp = cellData.boundaryFieldRef()[patchI];

        fvp = fvp.patch().patchSlice(faceData);
    }
    cellData.correctBoundaryConditions();
}


void Foam::writeFields
(
    const fvMesh& mesh,
    const wordHashSet& selectedFields
)
{
    if (selectedFields.empty())
    {
        return;
    }

    Info << "Writing fields with mesh quality parameters" << endl;

    if (selectedFields.found("nonOrthoAngle"))
    {
        //- Face based orthogonality
        const scalarField faceOrthogonality
        (
            polyMeshTools::faceOrthogonality
            (
                mesh,
                mesh.faceAreas(),
                mesh.cellCentres()
            )
        );

        //- Face based angle
        const scalarField nonOrthoAngle
        (
            radToDeg
            (
                Foam::acos(min(scalar(1), faceOrthogonality))
            )
        );

        //- Cell field - max of either face
        volScalarField cellNonOrthoAngle
        (
            IOobject
            (
                "nonOrthoAngle",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            calculatedFvPatchScalarField::typeName
        );
        //- Take max
        maxFaceToCell(nonOrthoAngle, cellNonOrthoAngle);
        Info << "    Writing non-orthogonality (angle) to "
             << cellNonOrthoAngle.name() << endl;
        cellNonOrthoAngle.write();
    }

    if (selectedFields.found("faceWeight"))
    {
        const scalarField faceWeights
        (
            polyMeshTools::faceWeights
            (
                mesh,
                mesh.faceCentres(),
                mesh.faceAreas(),
                mesh.cellCentres()
            )
        );

        volScalarField cellWeights
        (
            IOobject
            (
                "faceWeight",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            calculatedFvPatchScalarField::typeName
        );
        //- Take min
        minFaceToCell(faceWeights, cellWeights);
        Info << "    Writing face interpolation weights (0..0.5) to "
             << cellWeights.name() << endl;
        cellWeights.write();
    }


    // Skewness
    // ~~~~~~~~

    if (selectedFields.found("skewness"))
    {
        //- Face based skewness
        const scalarField faceSkewness
        (
            polyMeshTools::faceSkewness
            (
                mesh,
                mesh.points(),
                mesh.faceCentres(),
                mesh.faceAreas(),
                mesh.cellCentres()
            )
        );

        //- Cell field - max of either face
        volScalarField cellSkewness
        (
            IOobject
            (
                "skewness",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            calculatedFvPatchScalarField::typeName
        );
        //- Take max
        maxFaceToCell(faceSkewness, cellSkewness);
        Info << "    Writing face skewness to " << cellSkewness.name() << endl;
        cellSkewness.write();
    }


    // cellDeterminant
    // ~~~~~~~~~~~~~~~

    if (selectedFields.found("cellDeterminant"))
    {
        volScalarField cellDeterminant
        (
            IOobject
            (
                "cellDeterminant",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                false
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            zeroGradientFvPatchScalarField::typeName
        );
        cellDeterminant.primitiveFieldRef() =
            primitiveMeshTools::cellDeterminant
            (
                mesh,
                mesh.geometricD(),
                mesh.faceAreas(),
                syncTools::getInternalOrCoupledFaces(mesh)
            );
        cellDeterminant.correctBoundaryConditions();
        Info << "    Writing cell determinant to "
             << cellDeterminant.name() << endl;
        cellDeterminant.write();
    }


    // Aspect ratio
    // ~~~~~~~~~~~~
    if (selectedFields.found("aspectRatio"))
    {
        volScalarField aspectRatio
        (
            IOobject
            (
                "aspectRatio",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                false
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            zeroGradientFvPatchScalarField::typeName
        );


        scalarField cellOpenness;
        polyMeshTools::cellClosedness
        (
            mesh,
            mesh.geometricD(),
            mesh.faceAreas(),
            mesh.cellVolumes(),
            cellOpenness,
            aspectRatio.ref()
        );

        aspectRatio.correctBoundaryConditions();
        Info << "    Writing aspect ratio to " << aspectRatio.name() << endl;
        aspectRatio.write();
    }


    // cell type
    // ~~~~~~~~~

    if (selectedFields.found("cellShapes"))
    {
        volScalarField shape
        (
            IOobject
            (
                "cellShapes",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                false
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            zeroGradientFvPatchScalarField::typeName
        );
        const cellShapeList& cellShapes = mesh.cellShapes();
        forAll(cellShapes, cellI)
        {
            const cellModel& model = cellShapes[cellI].model();
            shape[cellI] = model.index();
        }
        shape.correctBoundaryConditions();
        Info << "    Writing cell shape (hex, tet etc.) to " << shape.name()
             << endl;
        shape.write();
    }

    if (selectedFields.found("cellVolume"))
    {
        volScalarField V
        (
            IOobject
            (
                "cellVolume",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                false
            ),
            mesh,
            dimensionedScalar(dimVolume, Zero),
            calculatedFvPatchScalarField::typeName
        );
        V.ref() = mesh.V();
        Info << "    Writing cell volume to " << V.name() << endl;
        V.write();
    }

    if (selectedFields.found("cellVolumeRatio"))
    {
        const scalarField faceVolumeRatio
        (
            polyMeshTools::volRatio
            (
                mesh,
                mesh.V()
            )
        );

        volScalarField cellVolumeRatio
        (
            IOobject
            (
                "cellVolumeRatio",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            calculatedFvPatchScalarField::typeName
        );
        //- Take min
        minFaceToCell(faceVolumeRatio, cellVolumeRatio);
        Info << "    Writing cell volume ratio to "
             << cellVolumeRatio.name() << endl;
        cellVolumeRatio.write();
    }

    // minTetVolume
    if (selectedFields.found("minTetVolume"))
    {
        volScalarField minTetVolume
        (
            IOobject
            (
                "minTetVolume",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                false
            ),
            mesh,
            dimensionedScalar("minTetVolume", dimless, GREAT),
            zeroGradientFvPatchScalarField::typeName
        );


        const labelList& own = mesh.faceOwner();
        const labelList& nei = mesh.faceNeighbour();
        const pointField& p = mesh.points();
        forAll(own, facei)
        {
            const face& f = mesh.faces()[facei];
            const point& fc = mesh.faceCentres()[facei];

            {
                const point& ownCc = mesh.cellCentres()[own[facei]];
                scalar& ownVol = minTetVolume[own[facei]];
                forAll(f, fp)
                {
                    scalar tetQual = tetPointRef
                                     (
                                         p[f[fp]],
                                         p[f.nextLabel(fp)],
                                         ownCc,
                                         fc
                                     ).quality();
                    ownVol = min(ownVol, tetQual);
                }
            }
            if (mesh.isInternalFace(facei))
            {
                const point& neiCc = mesh.cellCentres()[nei[facei]];
                scalar& neiVol = minTetVolume[nei[facei]];
                forAll(f, fp)
                {
                    scalar tetQual = tetPointRef
                                     (
                                         p[f[fp]],
                                         p[f.nextLabel(fp)],
                                         fc,
                                         neiCc
                                     ).quality();
                    neiVol = min(neiVol, tetQual);
                }
            }
        }

        minTetVolume.correctBoundaryConditions();
        Info << "    Writing minTetVolume to " << minTetVolume.name() << endl;
        minTetVolume.write();
    }

    if (selectedFields.found("cellRegion"))
    {
        volScalarField cellRegion
        (
            IOobject
            (
                "cellRegion",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar(dimless, Zero),
            calculatedFvPatchScalarField::typeName
        );

        regionSplit rs(mesh);
        forAll(rs, celli)
        {
            cellRegion[celli] = rs[celli];
        }
        cellRegion.correctBoundaryConditions();
        Info << "    Writing cell region to " << cellRegion.name() << endl;
        cellRegion.write();
    }

    Info << endl;
}
