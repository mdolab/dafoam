/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This file is modified from OpenFOAM's source code
    applications/solvers/stressAnalysis/solidDisplacementFoam/tractionDisplacement/
    tractionDisplacementFvPatchVectorField.H

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

#ifndef tractionDisplacementFvPatchVectorField_H
#define tractionDisplacementFvPatchVectorField_H

#include "fvPatchFields.H"
#include "fixedGradientFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                   Class tractionDisplacementFvPatch Declaration
\*---------------------------------------------------------------------------*/

class tractionDisplacementFvPatchVectorField
    : public fixedGradientFvPatchVectorField
{

    // Private Data

    vectorField traction_;
    scalarField pressure_;

public:
    //- Runtime type information
    TypeName("tractionDisplacement");

    // Constructors

    //- Construct from patch and internal field
    tractionDisplacementFvPatchVectorField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&);

    //- Construct from patch, internal field and dictionary
    tractionDisplacementFvPatchVectorField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const dictionary&);

    //- Construct by mapping given
    //  tractionDisplacementFvPatchVectorField onto a new patch
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField&,
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField&);

    //- Construct as copy setting internal field reference
    tractionDisplacementFvPatchVectorField(
        const tractionDisplacementFvPatchVectorField&,
        const DimensionedField<vector, volMesh>&);

    // Member functions

    // Access
    /*
    virtual const vectorField& traction() const
    {
        return traction_;
    }

    virtual vectorField& traction()
    {
        return traction_;
    }

    virtual const scalarField& pressure() const
    {
        return pressure_;
    }

    virtual scalarField& pressure()
    {
        return pressure_;
    }
*/
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
