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

#include "multiFreqVectorFvPatchField.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::multiFreqVectorFvPatchField::multiFreqVectorFvPatchField(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF)
    : fixedValueFvPatchVectorField(p, iF),
      refValue_(vector::zero),
      amplitudes_({}),
      frequencies_({}),
      phases_({}),
      component_(0),
      endTime_(1.0e8)
{
}

Foam::multiFreqVectorFvPatchField::multiFreqVectorFvPatchField(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict)
    : fixedValueFvPatchVectorField(p, iF),
      refValue_(dict.lookupOrDefault<vector>("refValue", vector::zero)),
      amplitudes_(dict.lookupOrDefault<scalarList>("amplitudes", {})),
      frequencies_(dict.lookupOrDefault<scalarList>("frequencies", {})),
      phases_(dict.lookupOrDefault<scalarList>("phases", {})),
      component_(dict.lookupOrDefault<label>("component", 0)),
      endTime_(dict.lookupOrDefault<scalar>("endTime", 1.0e8))
{
    if (dict.found("value"))
    {
        fvPatchVectorField::operator=(
            vectorField("value", dict, p.size()));
    }
    else
    {
        this->evaluate();
    }
}

Foam::multiFreqVectorFvPatchField::multiFreqVectorFvPatchField(
    const multiFreqVectorFvPatchField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper)
    : fixedValueFvPatchVectorField(p, iF),
      refValue_(ptf.refValue_),
      amplitudes_(ptf.amplitudes_),
      frequencies_(ptf.frequencies_),
      phases_(ptf.phases_),
      component_(ptf.component_),
      endTime_(ptf.endTime_)
{
    this->evaluate();
}

Foam::multiFreqVectorFvPatchField::multiFreqVectorFvPatchField(
    const multiFreqVectorFvPatchField& wbppsf)
    : fixedValueFvPatchVectorField(wbppsf),
      refValue_(wbppsf.refValue_),
      amplitudes_(wbppsf.amplitudes_),
      frequencies_(wbppsf.frequencies_),
      phases_(wbppsf.phases_),
      component_(wbppsf.component_),
      endTime_(wbppsf.endTime_)
{
    this->evaluate();
}

Foam::multiFreqVectorFvPatchField::multiFreqVectorFvPatchField(
    const multiFreqVectorFvPatchField& wbppsf,
    const DimensionedField<vector, volMesh>& iF)
    : fixedValueFvPatchVectorField(wbppsf, iF),
      refValue_(wbppsf.refValue_),
      amplitudes_(wbppsf.amplitudes_),
      frequencies_(wbppsf.frequencies_),
      phases_(wbppsf.phases_),
      component_(wbppsf.component_),
      endTime_(wbppsf.endTime_)
{
    this->evaluate();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void Foam::multiFreqVectorFvPatchField::updateCoeffs()
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

    vectorField& thisPatchRef = *this;
    vectorField thisPatch = thisPatchRef;
    forAll(thisPatch, idxI) thisPatch[idxI] = refValue_;
    if (t < endTime_) // if t > endTime, do not add oscillation.
    {
        forAll(thisPatch, idxI) thisPatch[idxI][component_] += bcVal;
    }

    fvPatchVectorField::operator==(thisPatch);

    fixedValueFvPatchVectorField::updateCoeffs();
}

void Foam::multiFreqVectorFvPatchField::write(Ostream& os) const
{
    fixedValueFvPatchVectorField::write(os);
    os.writeEntry("refValue", refValue_);
    os.writeEntry("amplitudes", amplitudes_);
    os.writeEntry("frequencies", frequencies_);
    os.writeEntry("phases", phases_);
    os.writeEntry("component", component_);
    os.writeEntry("endTime", endTime_);
    //writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
makePatchTypeField(
    fvPatchVectorField,
    multiFreqVectorFvPatchField);
}

// ************************************************************************* //
