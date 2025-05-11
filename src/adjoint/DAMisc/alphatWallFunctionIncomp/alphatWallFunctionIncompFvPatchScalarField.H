/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/TurbulenceModels/compressible/turbulentFluidThermoModels/derivedFvPatchFields/
    wallFunctions/alphatWallFunctionIncompressibles/alphatWallFunctionIncompressible

    The turbulent thermal diffusivity calculated using:

        \f[
            \alpha_t = \frac{\nu_t}{Pr_t}
        \f]

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

#ifndef alphatWallFunctionIncompressibleFvPatchScalarField_H
#define alphatWallFunctionIncompressibleFvPatchScalarField_H

#include "fixedValueFvPatchFields.H"
#include "turbulenceModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace incompressible
{

/*---------------------------------------------------------------------------*\
       Class alphatWallFunctionIncompressibleFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class alphatWallFunctionIncompressibleFvPatchScalarField
    : public fixedValueFvPatchScalarField
{
protected:
    // Protected data

    //- Turbulent Prandtl number
    scalar Prt_;

public:
    //- Runtime type information
    TypeName("incompressible::alphatWallFunction");

    // Constructors

    //- Construct from patch and internal field
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given
    //  alphatWallFunctionIncompressibleFvPatchScalarField
    //  onto a new patch
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const alphatWallFunctionIncompressibleFvPatchScalarField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const alphatWallFunctionIncompressibleFvPatchScalarField&);

    //- Construct as copy setting internal field reference
    alphatWallFunctionIncompressibleFvPatchScalarField(
        const alphatWallFunctionIncompressibleFvPatchScalarField&,
        const DimensionedField<scalar, volMesh>&);

    // Member functions

    // Evaluation functions

    //- Update the coefficients associated with the patch field
    virtual void updateCoeffs();

    // I-O

    //- Write
    virtual void write(Ostream&) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace incompressible
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
