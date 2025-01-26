/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class RhoFieldType>
void Foam::MRFZoneDF::makeRelativeRhoFlux(
    const RhoFieldType& rho,
    surfaceScalarField& phi) const
{
    if (!active_)
    {
        return;
    }

    const surfaceVectorField& Cf = mesh_.Cf();
    const surfaceVectorField& Sf = mesh_.Sf();

    const vector Omega = omega_ * axis_;

    const vectorField& Cfi = Cf;
    const vectorField& Sfi = Sf;
    scalarField& phii = phi.primitiveFieldRef();

    // Internal faces
    forAll(internalFaces_, i)
    {
        label facei = internalFaces_[i];
        phii[facei] -= rho[facei] * (Omega ^ (Cfi[facei] - origin_)) & Sfi[facei];
    }

    makeRelativeRhoFlux(rho.boundaryField(), phi.boundaryFieldRef());
}

template<class RhoFieldType>
void Foam::MRFZoneDF::makeRelativeRhoFlux(
    const RhoFieldType& rho,
    FieldField<fvsPatchField, scalar>& phi) const
{
    if (!active_)
    {
        return;
    }

    const surfaceVectorField& Cf = mesh_.Cf();
    const surfaceVectorField& Sf = mesh_.Sf();

    const vector Omega = omega_ * axis_;

    // Included patches
    forAll(includedFaces_, patchi)
    {
        forAll(includedFaces_[patchi], i)
        {
            label patchFacei = includedFaces_[patchi][i];

            phi[patchi][patchFacei] = 0.0;
        }
    }

    // Excluded patches
    forAll(excludedFaces_, patchi)
    {
        forAll(excludedFaces_[patchi], i)
        {
            label patchFacei = excludedFaces_[patchi][i];

            phi[patchi][patchFacei] -=
                rho[patchi][patchFacei]
                    * (Omega ^ (Cf.boundaryField()[patchi][patchFacei] - origin_))
                & Sf.boundaryField()[patchi][patchFacei];
        }
    }
}

template<class RhoFieldType>
void Foam::MRFZoneDF::makeRelativeRhoFlux(
    const RhoFieldType& rho,
    Field<scalar>& phi,
    const label patchi) const
{
    /*
    if (!active_)
    {
        return;
    }

    const surfaceVectorField& Cf = mesh_.Cf();
    const surfaceVectorField& Sf = mesh_.Sf();

    const vector Omega = omega_ * axis_;

    // Included patches
    forAll(includedFaces_[patchi], i)
    {
        label patchFacei = includedFaces_[patchi][i];

        phi[patchFacei] = 0.0;
    }

    // Excluded patches
    forAll(excludedFaces_[patchi], i)
    {
        label patchFacei = excludedFaces_[patchi][i];

        phi[patchFacei] -=
            rho[patchFacei]
                * (Omega ^ (Cf.boundaryField()[patchi][patchFacei] - origin_))
            & Sf.boundaryField()[patchi][patchFacei];
    }
    */
}

// ************************************************************************* //
