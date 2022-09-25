/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        A modified version of MRF that allows changing the rotation speed at
        the runtime to enable derivatives wrt the rotation speed

    This class is modified from OpenFOAM's source code
    src/finiteVolume/cfdTools/general/MRF

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

#include "MRFZoneDF.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvMatrices.H"
#include "faceSet.H"
#include "geometricOneField.H"
#include "syncTools.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
defineTypeNameAndDebug(MRFZoneDF, 0);
}

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::MRFZoneDF::setMRFFaces()
{
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    // Type per face:
    //  0:not in zone
    //  1:moving with frame
    //  2:other
    labelList faceType(mesh_.nFaces(), 0);

    // Determine faces in cell zone
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // (without constructing cells)

    const labelList& own = mesh_.faceOwner();
    const labelList& nei = mesh_.faceNeighbour();

    // Cells in zone
    boolList zoneCell(mesh_.nCells(), false);

    if (cellZoneID_ != -1)
    {
        const labelList& cellLabels = mesh_.cellZones()[cellZoneID_];
        forAll(cellLabels, i)
        {
            zoneCell[cellLabels[i]] = true;
        }
    }

    label nZoneFaces = 0;

    for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
    {
        if (zoneCell[own[facei]] || zoneCell[nei[facei]])
        {
            faceType[facei] = 1;
            nZoneFaces++;
        }
    }

    labelHashSet excludedPatches(excludedPatchLabels_);

    forAll(patches, patchi)
    {
        const polyPatch& pp = patches[patchi];

        if (pp.coupled() || excludedPatches.found(patchi))
        {
            forAll(pp, i)
            {
                label facei = pp.start() + i;

                if (zoneCell[own[facei]])
                {
                    faceType[facei] = 2;
                    nZoneFaces++;
                }
            }
        }
        else if (!isA<emptyPolyPatch>(pp))
        {
            forAll(pp, i)
            {
                label facei = pp.start() + i;

                if (zoneCell[own[facei]])
                {
                    faceType[facei] = 1;
                    nZoneFaces++;
                }
            }
        }
    }

    // Synchronize the faceType across processor patches
    syncTools::syncFaceList(mesh_, faceType, maxEqOp<label>());

    // Now we have for faceType:
    //  0   : face not in cellZone
    //  1   : internal face or normal patch face
    //  2   : coupled patch face or excluded patch face

    // Sort into lists per patch.

    internalFaces_.setSize(mesh_.nFaces());
    label nInternal = 0;

    for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
    {
        if (faceType[facei] == 1)
        {
            internalFaces_[nInternal++] = facei;
        }
    }
    internalFaces_.setSize(nInternal);

    labelList nIncludedFaces(patches.size(), 0);
    labelList nExcludedFaces(patches.size(), 0);

    forAll(patches, patchi)
    {
        const polyPatch& pp = patches[patchi];

        forAll(pp, patchFacei)
        {
            label facei = pp.start() + patchFacei;

            if (faceType[facei] == 1)
            {
                nIncludedFaces[patchi]++;
            }
            else if (faceType[facei] == 2)
            {
                nExcludedFaces[patchi]++;
            }
        }
    }

    includedFaces_.setSize(patches.size());
    excludedFaces_.setSize(patches.size());
    forAll(nIncludedFaces, patchi)
    {
        includedFaces_[patchi].setSize(nIncludedFaces[patchi]);
        excludedFaces_[patchi].setSize(nExcludedFaces[patchi]);
    }
    nIncludedFaces = 0;
    nExcludedFaces = 0;

    forAll(patches, patchi)
    {
        const polyPatch& pp = patches[patchi];

        forAll(pp, patchFacei)
        {
            label facei = pp.start() + patchFacei;

            if (faceType[facei] == 1)
            {
                includedFaces_[patchi][nIncludedFaces[patchi]++] = patchFacei;
            }
            else if (faceType[facei] == 2)
            {
                excludedFaces_[patchi][nExcludedFaces[patchi]++] = patchFacei;
            }
        }
    }
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::MRFZoneDF::MRFZoneDF(
    const word& name,
    const fvMesh& mesh,
    const dictionary& dict,
    const word& cellZoneName)
    : mesh_(mesh),
      name_(name),
      coeffs_(dict),
      active_(coeffs_.lookupOrDefault("active", true)),
      cellZoneName_(cellZoneName),
      cellZoneID_(),
      excludedPatchNames_(
          coeffs_.lookupOrDefault<wordRes>("nonRotatingPatches", wordRes())),
      origin_(coeffs_.lookup("origin")),
      axis_(coeffs_.lookup("axis")),
      omega_(coeffs_.lookupOrDefault<scalar>("omega", 0.0))
{
    if (cellZoneName_ == word::null)
    {
        coeffs_.readEntry("cellZone", cellZoneName_);
    }

    if (!active_)
    {
        cellZoneID_ = -1;
    }
    else
    {
        cellZoneID_ = mesh_.cellZones().findZoneID(cellZoneName_);

        axis_ = axis_ / mag(axis_);

        const labelHashSet excludedPatchSet(
            mesh_.boundaryMesh().patchSet(excludedPatchNames_));

        excludedPatchLabels_.setSize(excludedPatchSet.size());

        label i = 0;
        for (const label patchi : excludedPatchSet)
        {
            excludedPatchLabels_[i++] = patchi;
        }

        bool cellZoneFound = (cellZoneID_ != -1);

        reduce(cellZoneFound, orOp<bool>());

        if (!cellZoneFound)
        {
            FatalErrorInFunction
                << "cannot find MRF cellZone " << cellZoneName_
                << exit(FatalError);
        }

        setMRFFaces();
    }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::vector Foam::MRFZoneDF::Omega() const
{
    return omega_ * axis_;
}

void Foam::MRFZoneDF::addCoriolis(
    const volVectorField& U,
    volVectorField& ddtU) const
{
    if (cellZoneID_ == -1)
    {
        return;
    }

    const labelList& cells = mesh_.cellZones()[cellZoneID_];
    vectorField& ddtUc = ddtU.primitiveFieldRef();
    const vectorField& Uc = U;

    const vector Omega = this->Omega();

    forAll(cells, i)
    {
        label celli = cells[i];
        ddtUc[celli] += (Omega ^ Uc[celli]);
    }
}

void Foam::MRFZoneDF::makeRelative(volVectorField& U) const
{
    if (cellZoneID_ == -1)
    {
        return;
    }

    const volVectorField& C = mesh_.C();

    const vector Omega = this->Omega();

    const labelList& cells = mesh_.cellZones()[cellZoneID_];

    forAll(cells, i)
    {
        label celli = cells[i];
        U[celli] -= (Omega ^ (C[celli] - origin_));
    }

    // Included patches

    volVectorField::Boundary& Ubf = U.boundaryFieldRef();

    forAll(includedFaces_, patchi)
    {
        forAll(includedFaces_[patchi], i)
        {
            label patchFacei = includedFaces_[patchi][i];
            Ubf[patchi][patchFacei] = Zero;
        }
    }

    // Excluded patches
    forAll(excludedFaces_, patchi)
    {
        forAll(excludedFaces_[patchi], i)
        {
            label patchFacei = excludedFaces_[patchi][i];
            Ubf[patchi][patchFacei] -=
                (Omega
                 ^ (C.boundaryField()[patchi][patchFacei] - origin_));
        }
    }
}

void Foam::MRFZoneDF::makeRelative(surfaceScalarField& phi) const
{
    makeRelativeRhoFlux(geometricOneField(), phi);
}

void Foam::MRFZoneDF::makeRelative(FieldField<fvsPatchField, scalar>& phi) const
{
    //makeRelativeRhoFlux(oneFieldField(), phi);
}

void Foam::MRFZoneDF::makeRelative(Field<scalar>& phi, const label patchi) const
{
    //makeRelativeRhoFlux(oneField(), phi, patchi);
}

void Foam::MRFZoneDF::makeRelative(
    const surfaceScalarField& rho,
    surfaceScalarField& phi) const
{
    makeRelativeRhoFlux(rho, phi);
}

void Foam::MRFZoneDF::correctBoundaryVelocity(volVectorField& U) const
{
    if (!active_)
    {
        return;
    }

    const vector Omega = this->Omega();

    // Included patches
    volVectorField::Boundary& Ubf = U.boundaryFieldRef();

    forAll(includedFaces_, patchi)
    {
        const vectorField& patchC = mesh_.Cf().boundaryField()[patchi];

        vectorField pfld(Ubf[patchi]);

        forAll(includedFaces_[patchi], i)
        {
            label patchFacei = includedFaces_[patchi][i];

            pfld[patchFacei] = (Omega ^ (patchC[patchFacei] - origin_));
        }

        Ubf[patchi] == pfld;
    }
}
/*
void Foam::MRFZoneDF::writeData(Ostream& os) const
{
    // do nothing
}

bool Foam::MRFZoneDF::read(const dictionary& dict)
{
    // do nothing
    return true;
}

void Foam::MRFZoneDF::update()
{
    if (mesh_.topoChanging())
    {
        setMRFFaces();
    }
}
*/
// ************************************************************************* //
