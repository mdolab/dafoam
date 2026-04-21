/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2017 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

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

Class
    Foam::wallHeatFluxTransferFvPatchScalarField

Description
    This boundary condition applies a Heat transfer coefficient condition to temperature

    where:
    \vartable
        h  | Heat transfer coefficient [W/m^2/K]
        Ta | Ambient temperature [K]
    \endvartable

    The thermal conductivity \c kappa is use polynomial read from solidProperties.

Usage
    \table
    Property    | Description                              | Required | Default
    h           | Heat transfer coefficient [W/m^2/K]       | yes |
    Ta          | Ambient temperature [K]                   | yes |
    \endtable

    Example of the boundary condition specification:
    \verbatim
    <patchName>
    {
        type            wallHeatFluxTransfer;

        Ta              300.0;
        h               uniform 10.0;
        value           $internalField;
    }
    \endverbatim

See also
    Foam::mixedFvPatchScalarField

SourceFiles
    wallHeatFluxTransferFvPatchScalarField.C

\*---------------------------------------------------------------------------*/

#ifndef wallHeatFluxTransferFvPatchScalarField_H
#define wallHeatFluxTransferFvPatchScalarField_H

#include "mixedFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class wallHeatFluxTransferFvPatchScalarField Declaration
\*---------------------------------------------------------------------------*/

class wallHeatFluxTransferFvPatchScalarField
    : public mixedFvPatchScalarField
{

private:
    // Private data

    //- Heat transfer coefficient [W/m2K]
    scalarField h_;

    //- Ambient temperature [K]
    const scalar Ta_;

public:
    //- Runtime type information
    TypeName("wallHeatFluxTransfer");

    // Constructors

    //- Construct from patch and internal field
    wallHeatFluxTransferFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    wallHeatFluxTransferFvPatchScalarField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given
    //  externalWallHeatFluxTemperatureFvPatchScalarField
    //  onto a new patch
    wallHeatFluxTransferFvPatchScalarField(
        const wallHeatFluxTransferFvPatchScalarField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    wallHeatFluxTransferFvPatchScalarField(
        const wallHeatFluxTransferFvPatchScalarField&);

    //- Construct and return a clone
    virtual tmp<fvPatchScalarField> clone() const
    {
        return tmp<fvPatchScalarField>(
            new wallHeatFluxTransferFvPatchScalarField(*this));
    }

    //- Construct as copy setting internal field reference
    wallHeatFluxTransferFvPatchScalarField(
        const wallHeatFluxTransferFvPatchScalarField&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchScalarField> clone(
        const DimensionedField<scalar, volMesh>& iF) const
    {
        return tmp<fvPatchScalarField>(
            new wallHeatFluxTransferFvPatchScalarField(*this, iF));
    }

    // Member functions

    // Access

    //- Allow manipulation of the boundary values
    virtual bool fixesValue() const
    {
        return false;
    }

    // Mapping functions

    //- Map (and resize as needed) from self given a mapping object
    virtual void autoMap(
        const fvPatchFieldMapper&);

    //- Reverse map the given fvPatchField onto this fvPatchField
    virtual void rmap(
        const fvPatchScalarField&,
        const labelList&);

    // Evaluation functions

    //- Update the coefficients associated with the patch field
    virtual void updateCoeffs();

    // I-O

    //- Write
    void write(Ostream&) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
