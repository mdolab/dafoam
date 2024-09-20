/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        A modified version of Poisson method for computing wall distance
        Basically, we replace the original max function with the softmax 
        function to avoid discontinuity

    This file is modified from OpenFOAM's source code
    src/finiteVolume/fvMesh/wallDist/patchDistMethods/meshWave/meshWavePatchDistMethod.C

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

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
    // NOT implemented!
    FatalErrorIn("") << "NOT implemented" << abort(FatalError);
    return nUnset_ > 0;
}

// ************************************************************************* //
