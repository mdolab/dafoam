/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    One layer CHT boundary condition obtained through homogenization (for 
	laminar flows only)

    Contribution from Eric Segalerba

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

#ifndef homTempFvPatchScalarField_H
#define homTempFvPatchScalarField_H

#include "fixedValueFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
               Class homTempFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class homTempFvPatchScalarField
    : public fixedValueFvPatchScalarField
{
protected:
    // Protected Data

    //- Solid thermal conductivity
    const scalar kS_;

    //- Fluid thermal conductivity
    const scalar kF_;

    //- Solid thickness
    const scalar solidThickness_;

    //- Base temperature field
    Field<scalar> baseTemperature_;

    //- Old temperature value
    Field<scalar> oldT;

    //- Cell temperature initialization
    //Field<scalar> Tcell;
public:
    //- Runtime type information
    TypeName("homTemp");

    // Constructors

    //- Construct from patch and internal field
    homTempFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    homTempFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given homTempFvPatchScalarField
    //  onto a new patch
    homTempFvPatchScalarField(
        const homTempFvPatchScalarField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Disallow copy without setting internal field reference
    homTempFvPatchScalarField(
        const homTempFvPatchScalarField&) = delete;

    //- Copy constructor setting internal field reference
    homTempFvPatchScalarField(
        const homTempFvPatchScalarField&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchScalarField> clone(
        const DimensionedField<scalar, volMesh>& iF) const
    {
        return tmp<fvPatchScalarField>(
            new homTempFvPatchScalarField(*this, iF));
    }

    // Member Functions

    // Evaluation functions

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
