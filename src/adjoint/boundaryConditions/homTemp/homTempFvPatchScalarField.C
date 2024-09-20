/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

#include "homTempFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "pressureInletOutletVelocityFvPatchVectorField.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::homTempFvPatchScalarField::homTempFvPatchScalarField(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF)
    : fixedValueFvPatchScalarField(p, iF),
      kS_(0.0),
      kF_(0.0),
      solidThickness_(0.0),
      baseTemperature_(p.size(), 0.0)
{
}

Foam::homTempFvPatchScalarField::homTempFvPatchScalarField(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict)
    : fixedValueFvPatchScalarField(p, iF, dict),
      kS_(readScalar(dict.lookup("kS"))),
      kF_(readScalar(dict.lookup("kF"))),
      solidThickness_(readScalar(dict.lookup("solidThickness"))),
      baseTemperature_("baseTemperature", dict, p.size())
{
}

Foam::homTempFvPatchScalarField::homTempFvPatchScalarField(
    const homTempFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper)
    : fixedValueFvPatchScalarField(ptf, p, iF, mapper),
      kS_(ptf.kS_),
      kF_(ptf.kF_),
      solidThickness_(ptf.solidThickness_),
      baseTemperature_(ptf.baseTemperature_)
{
}

Foam::homTempFvPatchScalarField::homTempFvPatchScalarField(
    const homTempFvPatchScalarField& tppsf,
    const DimensionedField<scalar, volMesh>& iF)
    : fixedValueFvPatchScalarField(tppsf, iF),
      kS_(tppsf.kS_),
      kF_(tppsf.kF_),
      solidThickness_(tppsf.solidThickness_),
      baseTemperature_(tppsf.baseTemperature_)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::homTempFvPatchScalarField::updateCoeffs()
{

    // store in "Tp" the field of scalar temperature on the patch
    const scalarField& Tp(patch().lookupPatchField<volScalarField, scalar>("T"));

    // store in "Tc" the field of scalar temperature in the computational domain
    const scalarField& Tc(patch().lookupPatchField<volScalarField, scalar>("T").internalField());
    //Info << "Size of Tc: " << Tc.size() << endl;

    // initialize cell's temperature variable
    scalarField Tcell(Tp.size(), 0); //Zero

    // distance from the face center to the cell center on patch's cells
    const scalarField& deltaInv(patch().deltaCoeffs());

    // separately calculate the homogenization coefficient "lambda_theta"
    const scalarField Coeff(kF_ / kS_ * solidThickness_ * deltaInv);

    // store in "cells" the labels of the cells adjacent to the face
    const labelList cells(patch().faceCells());

    // initialize the counter
    label cellN(0);
    forAll(cells, patchCell)
    {
        cellN = cells[patchCell];
        Tcell[patchCell] = Tc[cellN];
    }

    // face temperature
    const scalarField faceTemp((baseTemperature_ + Tcell * Coeff) / (1 + Coeff));

    // value assigned to the patch
    operator==(
        faceTemp);

    fixedValueFvPatchScalarField::updateCoeffs();
}

void Foam::homTempFvPatchScalarField::write(Ostream& os) const
{
    fixedValueFvPatchScalarField::write(os);

    os.writeEntry("kS", kS_);
    os.writeEntry("kF", kF_);
    os.writeEntry("solidThickness", solidThickness_);
    baseTemperature_.writeEntry("baseTemperature", os);
    this->writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
makePatchTypeField(
    fvPatchScalarField,
    homTempFvPatchScalarField);
}

// ************************************************************************* //
