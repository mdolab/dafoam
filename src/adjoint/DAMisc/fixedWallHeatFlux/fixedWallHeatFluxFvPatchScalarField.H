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

    Description:
        Fixed traction boundary condition for the standard linear elastic, 
        fixed coefficient displacement equation.

\*---------------------------------------------------------------------------*/

#ifndef fixedWallHeatFluxFvPatchScalarField_H
#define fixedWallHeatFluxFvPatchScalarField_H

#include "fvPatchFields.H"
#include "fixedGradientFvPatchFields.H"
#include "turbulenceModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                   Class fixedWallHeatFlux Declaration
\*---------------------------------------------------------------------------*/

class fixedWallHeatFluxFvPatchScalarField
    : public fixedGradientFvPatchScalarField
{

    // Private Data

    scalar heatFlux_;

    scalar nu_;

    scalar Pr_;

    scalar Prt_;

    scalar Cp_;

public:
    //- Runtime type information
    TypeName("fixedWallHeatFlux");

    // Constructors

    //- Construct from patch and internal field
    fixedWallHeatFluxFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    fixedWallHeatFluxFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given
    //  fixedWallHeatFluxFvPatchScalarField onto a new patch
    fixedWallHeatFluxFvPatchScalarField(
        const fixedWallHeatFluxFvPatchScalarField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    fixedWallHeatFluxFvPatchScalarField(
        const fixedWallHeatFluxFvPatchScalarField&);

    //- Construct as copy setting internal field reference
    fixedWallHeatFluxFvPatchScalarField(
        const fixedWallHeatFluxFvPatchScalarField&,
        const DimensionedField<scalar, volMesh>&);

    // Member functions

    //- Update the coefficients associated with the patch field
    virtual void updateCoeffs();

    //- Write
    virtual void write(Ostream&) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
