/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    src/finiteVolume/fields/fvPatchFields/basic/fixedValue/fixedValueFvPatchField.C

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

#include "multiFreqScalarFvPatchField.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::multiFreqScalarFvPatchField::multiFreqScalarFvPatchField(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF)
    : fixedValueFvPatchScalarField(p, iF),
      refValue_(0.0),
      amplitudes_({}),
      frequencies_({}),
      phases_({})
{
}

Foam::multiFreqScalarFvPatchField::multiFreqScalarFvPatchField(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict)
    : fixedValueFvPatchScalarField(p, iF),
      refValue_(dict.lookupOrDefault<scalar>("refValue", 0.0)),
      amplitudes_(dict.lookupOrDefault<scalarList>("amplitudes", {})),
      frequencies_(dict.lookupOrDefault<scalarList>("frequencies", {})),
      phases_(dict.lookupOrDefault<scalarList>("phases", {}))
{

    if (dict.found("value"))
    {
        fvPatchScalarField::operator=(
            scalarField("value", dict, p.size()));
    }
    else
    {
        this->evaluate();
    }
}

Foam::multiFreqScalarFvPatchField::multiFreqScalarFvPatchField(
    const multiFreqScalarFvPatchField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper)
    : fixedValueFvPatchScalarField(p, iF),
      refValue_(ptf.refValue_),
      amplitudes_(ptf.amplitudes_),
      frequencies_(ptf.frequencies_),
      phases_(ptf.phases_)
{
    this->evaluate();
}

Foam::multiFreqScalarFvPatchField::multiFreqScalarFvPatchField(
    const multiFreqScalarFvPatchField& wbppsf)
    : fixedValueFvPatchScalarField(wbppsf),
      refValue_(wbppsf.refValue_),
      amplitudes_(wbppsf.amplitudes_),
      frequencies_(wbppsf.frequencies_),
      phases_(wbppsf.phases_)
{
    this->evaluate();
}

Foam::multiFreqScalarFvPatchField::multiFreqScalarFvPatchField(
    const multiFreqScalarFvPatchField& wbppsf,
    const DimensionedField<scalar, volMesh>& iF)
    : fixedValueFvPatchScalarField(wbppsf, iF),
      refValue_(wbppsf.refValue_),
      amplitudes_(wbppsf.amplitudes_),
      frequencies_(wbppsf.frequencies_),
      phases_(wbppsf.phases_)
{
    this->evaluate();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void Foam::multiFreqScalarFvPatchField::updateCoeffs()
{
    // calculate patch values
    const scalar t = this->db().time().timeOutputValue();
    scalar bcVal = 0.0;
    const label& nFreqs = frequencies_.size();

    for (label i = 0; i < nFreqs; i++)
    {
        const scalar& a = amplitudes_[i];
        const scalar& f = frequencies_[i];
        const scalar& p = phases_[i];
        bcVal += a * sin(constant::mathematical::twoPi * f * t + p);
    }

    scalarField& thisPatchRef = *this;
    scalarField thisPatch = thisPatchRef;
    forAll(thisPatch, idxI) thisPatch[idxI] = refValue_;
    forAll(thisPatch, idxI) thisPatch[idxI] += bcVal;

    fvPatchScalarField::operator==(thisPatch);

    fixedValueFvPatchScalarField::updateCoeffs();
}

void Foam::multiFreqScalarFvPatchField::write(Ostream& os) const
{
    fixedValueFvPatchScalarField::write(os);
    os.writeEntry("refValue", refValue_);
    os.writeEntry("amplitudes", amplitudes_);
    os.writeEntry("frequencies", frequencies_);
    os.writeEntry("phases", phases_);
    //writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
makePatchTypeField(
    fvPatchScalarField,
    multiFreqScalarFvPatchField);
}

// ************************************************************************* //
