/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2017 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2015-2017 OpenCFD Ltd.
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

#include "wallHeatFluxTransferFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::wallHeatFluxTransferFvPatchScalarField::
wallHeatFluxTransferFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(p, iF),
    h_(),
    Ta_(293.0)
{
    refValue() = 0.0;
    refGrad() = 0.0;
    valueFraction() = 1.0;
}


Foam::wallHeatFluxTransferFvPatchScalarField::
wallHeatFluxTransferFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchScalarField(p, iF),
    h_(),
    Ta_(dict.lookupOrDefault<scalar>("Ta", 293.0))
{

    h_ = scalarField("h", dict, p.size());

    fvPatchScalarField::operator=(scalarField("value", dict, p.size()));

    if (dict.found("refValue"))
    {
        // Full restart
        refValue() = scalarField("refValue", dict, p.size());
        refGrad() = scalarField("refGradient", dict, p.size());
        valueFraction() = scalarField("valueFraction", dict, p.size());
    }
    else
    {
        // Start from user entered data. Assume fixedValue.
        refValue() = *this;
        refGrad() = 0.0;
        valueFraction() = 1.0;
    }
}


Foam::wallHeatFluxTransferFvPatchScalarField::
wallHeatFluxTransferFvPatchScalarField
(
    const wallHeatFluxTransferFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    mixedFvPatchScalarField(ptf, p, iF, mapper),
    h_(),
    Ta_(ptf.Ta_)
{

    h_.setSize(mapper.size());
    h_.map(ptf.h_, mapper);

}


Foam::wallHeatFluxTransferFvPatchScalarField::
wallHeatFluxTransferFvPatchScalarField
(
    const wallHeatFluxTransferFvPatchScalarField& ewhftpsf
)
:
    mixedFvPatchScalarField(ewhftpsf),
    h_(ewhftpsf.h_),
    Ta_(ewhftpsf.Ta_)
{}


Foam::wallHeatFluxTransferFvPatchScalarField::
wallHeatFluxTransferFvPatchScalarField
(
    const wallHeatFluxTransferFvPatchScalarField& ewhftpsf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(ewhftpsf, iF),
    h_(ewhftpsf.h_),
    Ta_(ewhftpsf.Ta_)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::wallHeatFluxTransferFvPatchScalarField::autoMap
(
    const fvPatchFieldMapper& m
)
{
    mixedFvPatchScalarField::autoMap(m);


    h_.autoMap(m);
}


void Foam::wallHeatFluxTransferFvPatchScalarField::rmap
(
    const fvPatchScalarField& ptf,
    const labelList& addr
)
{
    mixedFvPatchScalarField::rmap(ptf, addr);

    const wallHeatFluxTransferFvPatchScalarField& ewhftpsf =
        refCast<const wallHeatFluxTransferFvPatchScalarField>(ptf);


    h_.rmap(ewhftpsf.h_, addr);
}


void Foam::wallHeatFluxTransferFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const scalarField& Tp(*this);

    /// not sure how to deal this, directly get kappa from filed K? or recalculate kappa from T? 
    const volScalarField& kappa = db().lookupObject<volScalarField>("k");
    const fvPatchField<scalar>& kappaBF = kappa.boundaryField()[patch().index()];


    const scalarField kappaDeltaCoeffs
    (
        kappaBF*patch().deltaCoeffs()
    );

    refGrad() = 0.0;

    forAll(Tp, i)
    {
        refValue()[i] = Ta_;
        valueFraction()[i] = h_[i]/(h_[i] + kappaDeltaCoeffs[i]);
    }



    mixedFvPatchScalarField::updateCoeffs();
}


void Foam::wallHeatFluxTransferFvPatchScalarField::write
(
    Ostream& os
) const
{
    fvPatchScalarField::write(os);

    h_.writeEntry("h", os);

    os.writeEntry("Ta", Ta_);

    refValue().writeEntry("refValue", os);
    refGrad().writeEntry("refGradient", os);
    valueFraction().writeEntry("valueFraction", os);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchScalarField,
        wallHeatFluxTransferFvPatchScalarField
    );
}

// ************************************************************************* //
