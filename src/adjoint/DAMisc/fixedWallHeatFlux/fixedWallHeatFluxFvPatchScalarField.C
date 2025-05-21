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

#include "fixedWallHeatFluxFvPatchScalarField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

fixedWallHeatFluxFvPatchScalarField::
    fixedWallHeatFluxFvPatchScalarField(
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF)
    : fixedGradientFvPatchScalarField(p, iF)
{
}

fixedWallHeatFluxFvPatchScalarField::
    fixedWallHeatFluxFvPatchScalarField(
        const fixedWallHeatFluxFvPatchScalarField& tdpvf,
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const fvPatchFieldMapper& mapper)
    : fixedGradientFvPatchScalarField(tdpvf, p, iF, mapper)
{
}

fixedWallHeatFluxFvPatchScalarField::
    fixedWallHeatFluxFvPatchScalarField(
        const fvPatch& p,
        const DimensionedField<scalar, volMesh>& iF,
        const dictionary& dict)
    : fixedGradientFvPatchScalarField(p, iF),
      heatFlux_(dict.getScalar("heatFlux")),
      nu_(dict.getScalar("nu")),
      Pr_(dict.getScalar("Pr")),
      Prt_(dict.getScalar("Prt")),
      Cp_(dict.getScalar("Cp"))
{
    fvPatchScalarField::operator=(patchInternalField());
    gradient() = 0.0;
}

fixedWallHeatFluxFvPatchScalarField::
    fixedWallHeatFluxFvPatchScalarField(
        const fixedWallHeatFluxFvPatchScalarField& tdpvf)
    : fixedGradientFvPatchScalarField(tdpvf)
{
}

fixedWallHeatFluxFvPatchScalarField::
    fixedWallHeatFluxFvPatchScalarField(
        const fixedWallHeatFluxFvPatchScalarField& tdpvf,
        const DimensionedField<scalar, volMesh>& iF)
    : fixedGradientFvPatchScalarField(tdpvf, iF)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void fixedWallHeatFluxFvPatchScalarField::updateCoeffs()
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

    const tmp<scalarField> nutw = turbModel.nut(patchi);

    gradient() = heatFlux_ / (nutw / Prt_ + nu_ / Pr_) / Cp_;

    fixedGradientFvPatchScalarField::updateCoeffs();
}

void fixedWallHeatFluxFvPatchScalarField::write(Ostream& os) const
{
    fvPatchScalarField::write(os);
    writeEntry("value", os);
    os.writeEntry("Pr", Pr_);
    os.writeEntry("Prt", Prt_);
    os.writeEntry("nu", nu_);
    os.writeEntry("Cp", Cp_);
    os.writeEntry("heatFlux", heatFlux_);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField(
    fvPatchScalarField,
    fixedWallHeatFluxFvPatchScalarField);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
