/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2017 OpenCFD Ltd
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

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

#include "alphatWallFunctionIncompFvPatchScalarField.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

alphatWallFunctionIncompressibleFvPatchScalarField::
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF)
    : fixedValueFvPatchScalarField(p, iF),
      Prt_(0.85)
{
}

alphatWallFunctionIncompressibleFvPatchScalarField::
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const alphatWallFunctionIncompressibleFvPatchScalarField& ptf,
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const fvPatchFieldMapper& mapper)
    : fixedValueFvPatchScalarField(ptf, p, iF, mapper)
{
}

alphatWallFunctionIncompressibleFvPatchScalarField::
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const dictionary& dict)
    : fixedValueFvPatchScalarField(p, iF, dict),
      Prt_(dict.get<scalar>("Prt")) // force read to avoid ambiguity
{
}

alphatWallFunctionIncompressibleFvPatchScalarField::
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const alphatWallFunctionIncompressibleFvPatchScalarField& wfpsf)
    : fixedValueFvPatchScalarField(wfpsf)
{
}

alphatWallFunctionIncompressibleFvPatchScalarField::
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const alphatWallFunctionIncompressibleFvPatchScalarField& wfpsf,
        const DimensionedField<scalar, volMesh>& iF)
    : fixedValueFvPatchScalarField(wfpsf, iF)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void alphatWallFunctionIncompressibleFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const label patchi = patch().index();

    // Retrieve turbulence properties from model

    const turbulenceModel& turbModel = db().lookupObject<turbulenceModel>(
        IOobject::groupName(
            turbulenceModel::propertiesName,
            internalField().group()));

    const tmp<scalarField> tnutw = turbModel.nut(patchi);

    operator==(tnutw / Prt_);

    fixedValueFvPatchField<scalar>::updateCoeffs();
}

void alphatWallFunctionIncompressibleFvPatchScalarField::write(Ostream& os) const
{
    fvPatchField<scalar>::write(os);
    os.writeEntry("Prt", Prt_);
    writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField(
    fvPatchScalarField,
    alphatWallFunctionIncompressibleFvPatchScalarField);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace incompressible
} // End namespace Foam

// ************************************************************************* //
