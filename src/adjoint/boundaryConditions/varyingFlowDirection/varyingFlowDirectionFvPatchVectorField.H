/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is modified from OpenFOAM's source code
    src/finiteVolume/fields/fvPatchFields/basic/fixedValue/fixedValueFvPatchField.H

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

        This boundary condition provides a varying flow direction at the far field
    
        U_flowDir = UMag * cos(alpha)
        U_normalDir = UMag * sin(alpha)
        alpha = alpha0 + rate * t
    
        where
    
        U_flowDir    : velocity component parallel to the flow. Options are: x, y, or z
        U_normalDir  : velocity component normal to the flow. Options are: x, y, or z
        UMag         : velocity magnitude [m/s]
        alpha0       : initial flow angle [rad]
        rate         : rate of change for the flow angle [rad/s]
    
        Example of the boundary condition specification:
        myPatch
        {
            type             varyingFlowDirection;
            UMag             10;
            flowDir          x;
            normalDir        y;
            alpha0           0;
            rate             0.1;
        }
    
        This will change the flow direction with a rate of 0.1 rad/s starting from 0 rad
        U[0] = cos(0+0.1*t) 
        U[1] = sin(0+0.1*t) 

\*---------------------------------------------------------------------------*/

#ifndef varyingFlowDirectionFvPatchVectorFields_H
#define varyingFlowDirectionFvPatchVectorFields_H

#include "fixedValueFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class varyingFlowDirectionFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class varyingFlowDirectionFvPatchVectorField
    : public fixedValueFvPatchVectorField
{
    // Private data

    scalar UMag_;
    word flowDir_;
    word normalDir_;
    scalar alpha0_;
    scalar rate_;

public:
    //- Runtime type information
    TypeName("varyingFlowDirection");

    // Constructors

    //- Construct from patch and internal field
    varyingFlowDirectionFvPatchVectorField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&);

    //- Construct from patch, internal field and dictionary
    varyingFlowDirectionFvPatchVectorField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const dictionary&);

    //- Construct by mapping given varyingFlowDirectionFvPatchVectorField onto
    //  a new patch
    varyingFlowDirectionFvPatchVectorField(
        const varyingFlowDirectionFvPatchVectorField&,
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    varyingFlowDirectionFvPatchVectorField(
        const varyingFlowDirectionFvPatchVectorField&);

    //- Construct and return a clone
    virtual tmp<fvPatchVectorField> clone() const
    {
        return tmp<fvPatchVectorField>(
            new varyingFlowDirectionFvPatchVectorField(*this));
    }

    //- Construct as copy setting internal field reference
    varyingFlowDirectionFvPatchVectorField(
        const varyingFlowDirectionFvPatchVectorField&,
        const DimensionedField<vector, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchVectorField> clone(
        const DimensionedField<vector, volMesh>& iF) const
    {
        return tmp<fvPatchVectorField>(
            new varyingFlowDirectionFvPatchVectorField(*this, iF));
    }

    // Member functions

    //- Update the patch field
    virtual void updateCoeffs();

    //- Write
    virtual void write(Ostream&) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
