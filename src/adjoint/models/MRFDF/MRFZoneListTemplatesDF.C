/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

#include "MRFZoneListDF.H"

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Type>
Foam::tmp<Foam::GeometricField<Type, Foam::fvsPatchField, Foam::surfaceMesh>>
Foam::MRFZoneListDF::zeroFilter(
    const tmp<GeometricField<Type, fvsPatchField, surfaceMesh>>& tphi) const
{
    if (size())
    {
        tmp<surfaceScalarField> zphi(
            New(
                tphi,
                "zeroFilter(" + tphi().name() + ')',
                tphi().dimensions(),
                true));

        forAll(*this, i)
        {
            operator[](i).zero(zphi.ref());
        }

        return zphi;
    }
    else
    {
        return tmp<surfaceScalarField>(tphi, true);
    }
}

// ************************************************************************* //
