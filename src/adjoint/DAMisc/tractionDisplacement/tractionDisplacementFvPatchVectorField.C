/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    applications/solvers/stressAnalysis/solidDisplacementFoam/tractionDisplacement/
    tractionDisplacementFvPatchVectorField.C

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

#include "tractionDisplacementFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF)
    : fixedGradientFvPatchVectorField(p, iF)
{
}

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField& tdpvf,
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const fvPatchFieldMapper& mapper)
    : fixedGradientFvPatchVectorField(tdpvf, p, iF, mapper)
{
}

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const dictionary& dict)
    : fixedGradientFvPatchVectorField(p, iF),
      traction_("traction", dict, p.size()),
      pressure_("pressure", dict, p.size())
{
    fvPatchVectorField::operator=(patchInternalField());
    gradient() = Zero;
}

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField& tdpvf)
    : fixedGradientFvPatchVectorField(tdpvf)
{
}

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField& tdpvf,
        const DimensionedField<vector, volMesh>& iF)
    : fixedGradientFvPatchVectorField(tdpvf, iF)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void tractionDisplacementFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const fvPatchField<scalar>& rho =
        patch().lookupPatchField<volScalarField, scalar>("solid:rho");

    const fvPatchField<scalar>& lambda =
        patch().lookupPatchField<volScalarField, scalar>("solid:lambda");

    const fvPatchField<scalar>& mu =
        patch().lookupPatchField<volScalarField, scalar>("solid:mu");

    const fvPatchField<tensor>& gradD =
        patch().lookupPatchField<volTensorField, tensor>("gradD");

    // here we use the BC implementation from:
    // Tang, Tian: Implementation of solid body stress analysis in OpenFOAM, 2013
    vectorField n(patch().nf());
    gradient() = ((traction_ - pressure_ * n) / rho
                  - (n & (mu * gradD.T() - (mu + lambda) * gradD))
                  - n * tr(gradD) * lambda)
        / (2.0 * mu + lambda);

    fixedGradientFvPatchVectorField::updateCoeffs();
}

void tractionDisplacementFvPatchVectorField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
    traction_.writeEntry("traction", os);
    pressure_.writeEntry("pressure", os);
    writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField(
    fvPatchVectorField,
    tractionDisplacementFvPatchVectorField);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
