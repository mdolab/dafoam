/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

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

        This boundary condition provides a varying velocity for the patch
    
        U[flowComponent] = U * cos(alpha)
        U[normalComponent] = U * sin(alpha)
        alpha = alpha0 + alphaRate * t
        U = U0 + URate * t
    
        where
    
        flowComponent   : velocity component parallel to the flow. Options are: 0, 1, or 2, which corresponds to x, y, and z
        normalComponent : velocity component normal to the flow. Options are: 0, 1, or 2, which corresponds to x, y, and z
        U0           : initial velocity magnitude [m/s]
        URate        : rate of change for the velocity magnitude
        alpha0       : initial flow angle [rad]
        alphaRate    : rate of change for the flow angle [rad/s]
    
        Example of the boundary condition specification:
        myPatch
        {
            type             varyingVelocity;
            U0               10;
            URate            -0.2;
            flowComponent     0;
            normalComponent   1;
            alpha0           0;
            alphaRate        0.1;
        }
    
        This will change the flow direction with a rate of 0.1 rad/s starting from 0 rad
        U[0] = (10-0.2*t)*cos(0+0.1*t) 
        U[1] = (10-0.2*t)*sin(0+0.1*t) 

\*---------------------------------------------------------------------------*/

#ifndef varyingVelocityFvPatchVectorFields_H
#define varyingVelocityFvPatchVectorFields_H

#include "fixedValueFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class varyingVelocityFvPatchVectorField Declaration
\*---------------------------------------------------------------------------*/

class varyingVelocityFvPatchVectorField
    : public fixedValueFvPatchVectorField
{
    // Private data

    scalar U0_;
    scalar URate_;
    label flowComponent_;
    label normalComponent_;
    scalar alpha0_;
    scalar alphaRate_;

public:
    //- Runtime type information
    TypeName("varyingVelocity");

    // Constructors

    //- Construct from patch and internal field
    varyingVelocityFvPatchVectorField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&);

    //- Construct from patch, internal field and dictionary
    varyingVelocityFvPatchVectorField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const dictionary&);

    //- Construct by mapping given varyingVelocityFvPatchVectorField onto
    //  a new patch
    varyingVelocityFvPatchVectorField(
        const varyingVelocityFvPatchVectorField&,
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    varyingVelocityFvPatchVectorField(
        const varyingVelocityFvPatchVectorField&);

    //- Construct and return a clone
    virtual tmp<fvPatchVectorField> clone() const
    {
        return tmp<fvPatchVectorField>(
            new varyingVelocityFvPatchVectorField(*this));
    }

    //- Construct as copy setting internal field reference
    varyingVelocityFvPatchVectorField(
        const varyingVelocityFvPatchVectorField&,
        const DimensionedField<vector, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchVectorField> clone(
        const DimensionedField<vector, volMesh>& iF) const
    {
        return tmp<fvPatchVectorField>(
            new varyingVelocityFvPatchVectorField(*this, iF));
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
