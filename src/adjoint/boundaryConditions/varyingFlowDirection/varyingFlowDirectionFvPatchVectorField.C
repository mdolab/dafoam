/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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

#include "varyingFlowDirectionFvPatchVectorField.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::varyingFlowDirectionFvPatchVectorField::varyingFlowDirectionFvPatchVectorField(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF)
    : fixedValueFvPatchVectorField(p, iF),
      UMag_(0.0),
      flowDir_("x"),
      normalDir_("y"),
      alpha0_(0),
      rate_(0)
{
}

Foam::varyingFlowDirectionFvPatchVectorField::varyingFlowDirectionFvPatchVectorField(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict)
    : fixedValueFvPatchVectorField(p, iF),
      UMag_(dict.lookupOrDefault<scalar>("UMag", 0.0)),
      flowDir_(dict.lookupOrDefault<word>("flowDir", "x")),
      normalDir_(dict.lookupOrDefault<word>("normalDir", "y")),
      alpha0_(dict.lookupOrDefault<scalar>("alpha0", 0.0)),
      rate_(dict.lookupOrDefault<scalar>("rate", 0.0))
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

Foam::varyingFlowDirectionFvPatchVectorField::varyingFlowDirectionFvPatchVectorField(
    const varyingFlowDirectionFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper)
    : fixedValueFvPatchVectorField(p, iF),
      UMag_(ptf.UMag_),
      flowDir_(ptf.flowDir_),
      normalDir_(ptf.normalDir_),
      alpha0_(ptf.alpha0_),
      rate_(ptf.rate_)
{
    this->evaluate();
}

Foam::varyingFlowDirectionFvPatchVectorField::varyingFlowDirectionFvPatchVectorField(
    const varyingFlowDirectionFvPatchVectorField& wbppsf)
    : fixedValueFvPatchVectorField(wbppsf),
      UMag_(wbppsf.UMag_),
      flowDir_(wbppsf.flowDir_),
      normalDir_(wbppsf.normalDir_),
      alpha0_(wbppsf.alpha0_),
      rate_(wbppsf.rate_)
{
    this->evaluate();
}

Foam::varyingFlowDirectionFvPatchVectorField::varyingFlowDirectionFvPatchVectorField(
    const varyingFlowDirectionFvPatchVectorField& wbppsf,
    const DimensionedField<vector, volMesh>& iF)
    : fixedValueFvPatchVectorField(wbppsf, iF),
      UMag_(wbppsf.UMag_),
      flowDir_(wbppsf.flowDir_),
      normalDir_(wbppsf.normalDir_),
      alpha0_(wbppsf.alpha0_),
      rate_(wbppsf.rate_)
{
    this->evaluate();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void Foam::varyingFlowDirectionFvPatchVectorField::updateCoeffs()
{
    // calculate patch values
    const scalar t = this->db().time().timeOutputValue();
    scalar alpha = alpha0_ + t * rate_;

    HashTable<label> compTable;
    compTable.set("x", 0);
    compTable.set("y", 1);
    compTable.set("z", 2);

    vectorField& thisPatchRef = *this;
    vectorField thisPatch = thisPatchRef;
    forAll(thisPatch, idxI) 
    {
        label compFlow = compTable[flowDir_];
        label compNormal = compTable[normalDir_];
        thisPatch[idxI][compFlow] = UMag_ * cos(alpha);
        thisPatch[idxI][compNormal] = UMag_ * sin(alpha);
    }

    fvPatchVectorField::operator==(thisPatch);

    fixedValueFvPatchVectorField::updateCoeffs();
}

void Foam::varyingFlowDirectionFvPatchVectorField::write(Ostream& os) const
{
    fixedValueFvPatchVectorField::write(os);
    os.writeEntry("UMag", UMag_);
    os.writeEntry("flowDir", flowDir_);
    os.writeEntry("normalDir", normalDir_);
    os.writeEntry("alpha0", alpha0_);
    os.writeEntry("rate", rate_);
    //writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
makePatchTypeField(
    fvPatchVectorField,
    varyingFlowDirectionFvPatchVectorField);
}

// ************************************************************************* //
