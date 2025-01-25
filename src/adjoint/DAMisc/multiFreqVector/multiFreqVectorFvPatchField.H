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

        This boundary condition provides an oscillating condition in terms of
        (possibly multiple) amplitude and frequency.
    
        x_p[i] = x_ref[i] + a * sin(2 * pi * f * t + phase) 
    
        where
    
        x_p    : patch values
        x_ref  : reference value 
        a      : amplitude
        f      : frequency [1/s]
        t      : time [s]
        phase  : phase in radian
        i      : component of the vector
    
        The input values are:

        Property     | Description                        | Required  
        refVal       | reference BC value                 | yes      
        amplitudes   | oscillation amplitudes             | yes      
        frequencies  | oscillation frequencies            | yes      
        phases       | phase angles at zero               | yes      
        component    | component of the vector            | yes 
        endTime      | oscillation end time (default 1e8) | no
    
        Example of the boundary condition specification:
        myPatch
        {
            type             multiFreqVector;
            refValue         (1.0 0.0 0.0);
            amplitudes       (0.5 1.0);
            frequencies      (10.0 5.0);
            phases           (0.0 1.0);
            component        0;
        }
    
        This will add the oscillation with two frequencies to the 0th component 
        of the refValue (1 0 0) at myPatch

\*---------------------------------------------------------------------------*/

#ifndef multiFreqVectorFvPatchFields_H
#define multiFreqVectorFvPatchFields_H

#include "fixedValueFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class multiFreqVectorFvPatchField Declaration
\*---------------------------------------------------------------------------*/

class multiFreqVectorFvPatchField
    : public fixedValueFvPatchVectorField
{
    // Private data

    vector refValue_;
    scalarList amplitudes_;
    scalarList frequencies_;
    scalarList phases_;
    label component_;
    scalar endTime_;

public:
    //- Runtime type information
    TypeName("multiFreqVector");

    // Constructors

    //- Construct from patch and internal field
    multiFreqVectorFvPatchField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&);

    //- Construct from patch, internal field and dictionary
    multiFreqVectorFvPatchField(
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const dictionary&);

    //- Construct by mapping given multiFreqVectorFvPatchField onto
    //  a new patch
    multiFreqVectorFvPatchField(
        const multiFreqVectorFvPatchField&,
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    multiFreqVectorFvPatchField(
        const multiFreqVectorFvPatchField&);

    //- Construct and return a clone
    virtual tmp<fvPatchVectorField> clone() const
    {
        return tmp<fvPatchVectorField>(
            new multiFreqVectorFvPatchField(*this));
    }

    //- Construct as copy setting internal field reference
    multiFreqVectorFvPatchField(
        const multiFreqVectorFvPatchField&,
        const DimensionedField<vector, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchVectorField> clone(
        const DimensionedField<vector, volMesh>& iF) const
    {
        return tmp<fvPatchVectorField>(
            new multiFreqVectorFvPatchField(*this, iF));
    }

    // Member functions

    //- Update the patch field
    virtual void updateCoeffs();

    //- Write
    virtual void write(Ostream&) const;

    //- Access the BC parameters
    vector& refValue()
    {
        return refValue_;
    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
