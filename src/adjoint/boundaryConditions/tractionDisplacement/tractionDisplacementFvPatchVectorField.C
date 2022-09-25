/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

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
    : fixedGradientFvPatchVectorField(p, iF),
      traction_(p.size(), Zero),
      pressure_(p.size(), 0.0)
{
    fvPatchVectorField::operator=(patchInternalField());
    gradient() = Zero;
}

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField& tdpvf,
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const fvPatchFieldMapper& mapper)
    : fixedGradientFvPatchVectorField(tdpvf, p, iF, mapper),
      traction_(tdpvf.traction_, mapper),
      pressure_(tdpvf.pressure_, mapper)
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
    : fixedGradientFvPatchVectorField(tdpvf),
      traction_(tdpvf.traction_),
      pressure_(tdpvf.pressure_)
{
}

tractionDisplacementFvPatchVectorField::
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField& tdpvf,
        const DimensionedField<vector, volMesh>& iF)
    : fixedGradientFvPatchVectorField(tdpvf, iF),
      traction_(tdpvf.traction_),
      pressure_(tdpvf.pressure_)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void tractionDisplacementFvPatchVectorField::autoMap(
    const fvPatchFieldMapper& m)
{

    fixedGradientFvPatchVectorField::autoMap(m);
    traction_.autoMap(m);
    pressure_.autoMap(m);

}

void tractionDisplacementFvPatchVectorField::rmap(
    const fvPatchVectorField& ptf,
    const labelList& addr)
{

    fixedGradientFvPatchVectorField::rmap(ptf, addr);

    const tractionDisplacementFvPatchVectorField& dmptf =
        refCast<const tractionDisplacementFvPatchVectorField>(ptf);

    traction_.rmap(dmptf.traction_, addr);
    pressure_.rmap(dmptf.pressure_, addr);

}

void tractionDisplacementFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const dictionary& mechanicalProperties =
        db().lookupObject<IOdictionary>("mechanicalProperties");

    const fvPatchField<scalar>& rho =
        patch().lookupPatchField<volScalarField, scalar>("solid:rho");

    const fvPatchField<scalar>& rhoE =
        patch().lookupPatchField<volScalarField, scalar>("E");

    const fvPatchField<scalar>& nu =
        patch().lookupPatchField<volScalarField, scalar>("solid:nu");

    scalarField E(rhoE / rho);
    scalarField mu(E / (2.0 * (1.0 + nu)));
    scalarField lambda(nu * E / ((1.0 + nu) * (1.0 - 2.0 * nu)));

    Switch planeStress(mechanicalProperties.lookup("planeStress"));

    if (planeStress)
    {
        lambda = nu * E / ((1.0 + nu) * (1.0 - nu));
    }

    scalarField twoMuLambda(2 * mu + lambda);

    vectorField n(patch().nf());

    // NOTE: we can't use the built-in gradient() computation because
    // it is designed for transient problem, i.e., snGrad() is actually
    // related to gradient()
    // Here we implement our owned BC for steady-state
    const fvPatchField<tensor>& gradD =
        patch().lookupPatchField<volTensorField, tensor>("gradD");
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
