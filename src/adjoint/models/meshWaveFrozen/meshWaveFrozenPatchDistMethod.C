/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        A modified version of Poisson method for computing wall distance
        Basically, we replace the original max function with the softmax 
        function to avoid discontinuity

\*---------------------------------------------------------------------------*/

#include "meshWaveFrozenPatchDistMethod.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace patchDistMethods
{
defineTypeNameAndDebug(meshWaveFrozen, 0);
addToRunTimeSelectionTable(patchDistMethod, meshWaveFrozen, dictionary);
}
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::patchDistMethods::meshWaveFrozen::meshWaveFrozen(
    const dictionary& dict,
    const fvMesh& mesh,
    const labelHashSet& patchIDs)
    : patchDistMethod(mesh, patchIDs),
      correctWalls_(dict.lookupOrDefault("correctWalls", true)),
      nUnset_(0),
      y_(IOobject("yWallFrozen", mesh.time().timeName(), mesh),
         mesh,
         dimensionedScalar("yWallFrozen", dimLength, SMALL),
         patchDistMethod::patchTypes<scalar>(mesh, patchIDs)),
      n_(IOobject("nWallFrozen", mesh.time().timeName(), mesh),
         mesh,
         dimensionedVector(dimless, Zero),
         patchDistMethod::patchTypes<vector>(mesh, patchIDs))
{
}

Foam::patchDistMethods::meshWaveFrozen::meshWaveFrozen(
    const fvMesh& mesh,
    const labelHashSet& patchIDs,
    const bool correctWalls)
    : patchDistMethod(mesh, patchIDs),
      correctWalls_(correctWalls),
      nUnset_(0),
      y_(IOobject("yWallFrozen", mesh.time().timeName(), mesh),
         mesh,
         dimensionedScalar("yWallFrozen", dimLength, SMALL),
         patchDistMethod::patchTypes<scalar>(mesh, patchIDs)),
      n_(IOobject("nWallFrozen", mesh.time().timeName(), mesh),
         mesh,
         dimensionedVector(dimless, Zero),
         patchDistMethod::patchTypes<vector>(mesh, patchIDs))
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::patchDistMethods::meshWaveFrozen::correct(volScalarField& y)
{
    if (isComputed_)
    {
        y = y_;
        return nUnset_ > 0;
    }
    else
    {
        y = dimensionedScalar("yWall", dimLength, GREAT);

        // Calculate distance starting from patch faces
        patchWave wave(mesh_, patchIDs_, correctWalls_);

        // Transfer cell values from wave into y
        y.transfer(wave.distance());

        // Transfer values on patches into boundaryField of y
        volScalarField::Boundary& ybf = y.boundaryFieldRef();

        forAll(ybf, patchi)
        {
            if (!isA<emptyFvPatchScalarField>(ybf[patchi]))
            {
                scalarField& waveFld = wave.patchDistance()[patchi];

                ybf[patchi].transfer(waveFld);
            }
        }

        // Transfer number of unset values
        nUnset_ = wave.nUnset();

        y_ = y;

        isComputed_ = 1;

        return nUnset_ > 0;
    }
}

bool Foam::patchDistMethods::meshWaveFrozen::correct(
    volScalarField& y,
    volVectorField& n)
{
    if (isComputed_)
    {
        y = y_;
        n = n_;
        return nUnset_ > 0;
    }
    else
    {
        y = dimensionedScalar("yWall", dimLength, GREAT);

        // Collect pointers to data on patches
        UPtrList<vectorField> patchData(mesh_.boundaryMesh().size());

        volVectorField::Boundary& nbf = n.boundaryFieldRef();

        forAll(nbf, patchi)
        {
            patchData.set(patchi, &nbf[patchi]);
        }

        // Do mesh wave
        patchDataWave<wallPointData<vector>> wave(
            mesh_,
            patchIDs_,
            patchData,
            correctWalls_);

        // Transfer cell values from wave into y and n
        y.transfer(wave.distance());

        n.transfer(wave.cellData());

        // Transfer values on patches into boundaryField of y and n
        volScalarField::Boundary& ybf = y.boundaryFieldRef();

        forAll(ybf, patchi)
        {
            scalarField& waveFld = wave.patchDistance()[patchi];

            if (!isA<emptyFvPatchScalarField>(ybf[patchi]))
            {
                ybf[patchi].transfer(waveFld);

                vectorField& wavePatchData = wave.patchData()[patchi];

                nbf[patchi].transfer(wavePatchData);
            }
        }

        // Transfer number of unset values
        nUnset_ = wave.nUnset();

        y_ = y;
        n_ = n;

        isComputed_ = 1;

        return nUnset_ > 0;
    }
}

// ************************************************************************* //
