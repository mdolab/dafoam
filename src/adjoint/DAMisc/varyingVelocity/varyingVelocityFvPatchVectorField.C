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

#include "varyingVelocityFvPatchVectorField.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::varyingVelocityFvPatchVectorField::varyingVelocityFvPatchVectorField(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF)
    : fixedValueFvPatchVectorField(p, iF),
      U0_(0.0),
      URate_(0.0),
      flowComponent_(0),
      normalComponent_(1),
      alpha0_(0),
      alphaRate_(0)
{
}

Foam::varyingVelocityFvPatchVectorField::varyingVelocityFvPatchVectorField(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict)
    : fixedValueFvPatchVectorField(p, iF),
      U0_(dict.lookupOrDefault<scalar>("U0", 0.0)),
      URate_(dict.lookupOrDefault<scalar>("URate", 0.0)),
      flowComponent_(dict.lookupOrDefault<label>("flowComponent", 0)),
      normalComponent_(dict.lookupOrDefault<label>("normalComponent", 1)),
      alpha0_(dict.lookupOrDefault<scalar>("alpha0", 0.0)),
      alphaRate_(dict.lookupOrDefault<scalar>("alphaRate", 0.0))
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

Foam::varyingVelocityFvPatchVectorField::varyingVelocityFvPatchVectorField(
    const varyingVelocityFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper)
    : fixedValueFvPatchVectorField(p, iF),
      U0_(ptf.U0_),
      URate_(ptf.URate_),
      flowComponent_(ptf.flowComponent_),
      normalComponent_(ptf.normalComponent_),
      alpha0_(ptf.alpha0_),
      alphaRate_(ptf.alphaRate_)
{
    this->evaluate();
}

Foam::varyingVelocityFvPatchVectorField::varyingVelocityFvPatchVectorField(
    const varyingVelocityFvPatchVectorField& wbppsf)
    : fixedValueFvPatchVectorField(wbppsf),
      U0_(wbppsf.U0_),
      URate_(wbppsf.URate_),
      flowComponent_(wbppsf.flowComponent_),
      normalComponent_(wbppsf.normalComponent_),
      alpha0_(wbppsf.alpha0_),
      alphaRate_(wbppsf.alphaRate_)
{
    this->evaluate();
}

Foam::varyingVelocityFvPatchVectorField::varyingVelocityFvPatchVectorField(
    const varyingVelocityFvPatchVectorField& wbppsf,
    const DimensionedField<vector, volMesh>& iF)
    : fixedValueFvPatchVectorField(wbppsf, iF),
      U0_(wbppsf.U0_),
      URate_(wbppsf.URate_),
      flowComponent_(wbppsf.flowComponent_),
      normalComponent_(wbppsf.normalComponent_),
      alpha0_(wbppsf.alpha0_),
      alphaRate_(wbppsf.alphaRate_)
{
    this->evaluate();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void Foam::varyingVelocityFvPatchVectorField::updateCoeffs()
{
    // calculate patch values
    const scalar t = this->db().time().timeOutputValue();
    scalar alpha = alpha0_ + t * alphaRate_;
    scalar U = U0_ + t * URate_;

    vectorField& thisPatchRef = *this;
    vectorField thisPatch = thisPatchRef;
    forAll(thisPatch, idxI)
    {
        thisPatch[idxI][flowComponent_] = U * cos(alpha);
        thisPatch[idxI][normalComponent_] = U * sin(alpha);
    }

    fvPatchVectorField::operator==(thisPatch);

    fixedValueFvPatchVectorField::updateCoeffs();
}

void Foam::varyingVelocityFvPatchVectorField::write(Ostream& os) const
{
    fixedValueFvPatchVectorField::write(os);
    os.writeEntry("U0", U0_);
    os.writeEntry("URate", URate_);
    os.writeEntry("flowComponent", flowComponent_);
    os.writeEntry("normalComponent", normalComponent_);
    os.writeEntry("alpha0", alpha0_);
    os.writeEntry("alphaRate", alphaRate_);
    //writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
makePatchTypeField(
    fvPatchVectorField,
    varyingVelocityFvPatchVectorField);
}

// ************************************************************************* //
