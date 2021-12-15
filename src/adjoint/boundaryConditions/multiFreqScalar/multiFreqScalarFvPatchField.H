/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:

        This boundary condition provides an oscillating condition in terms of
        (possibly multiple) amplitude and frequency.
    
        x_p = x_ref + a * sin(2 * pi * f * t + phase) 
    
        where
    
        x_p    : patch values
        x_ref  : reference value 
        a      : amplitude
        f      : frequency [1/s]
        t      : time [s]
        phase  : phase in radian
    
        The input values are:

        Property     | Description                        | Required  
        refVal       | reference BC value                 | yes      
        amplitudes   | oscillation amplitudes             | yes      
        frequencies  | oscillation frequencies            | yes      
        phases       | phase angles at zero               | yes      
        endTime      | oscillation end time (default 1e8) | no
    
        Example of the boundary condition specification:
        myPatch
        {
            type             multiFreqVector;
            refValue         1.0;
            amplitudes       (0.5 1.0);
            frequencies      (10.0 5.0);
            phases           (0.0 1.0);
        }
    
        This will add the oscillation with two frequencies to the refValue 
        1.0 at myPatch

\*---------------------------------------------------------------------------*/

#ifndef multiFreqScalarFvPatchFields_H
#define multiFreqScalarFvPatchFields_H

#include "fixedValueFvPatchFields.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
             Class multiFreqScalarFvPatchField Declaration
\*---------------------------------------------------------------------------*/

class multiFreqScalarFvPatchField
    : public fixedValueFvPatchScalarField
{
    // Private data

    scalar refValue_;
    scalarList amplitudes_;
    scalarList frequencies_;
    scalarList phases_;

public:
    //- Runtime type information
    TypeName("multiFreqScalar");

    // Constructors

    //- Construct from patch and internal field
    multiFreqScalarFvPatchField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct from patch, internal field and dictionary
    multiFreqScalarFvPatchField(
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const dictionary&);

    //- Construct by mapping given multiFreqScalarFvPatchField onto
    //  a new patch
    multiFreqScalarFvPatchField(
        const multiFreqScalarFvPatchField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&);

    //- Construct as copy
    multiFreqScalarFvPatchField(
        const multiFreqScalarFvPatchField&);

    //- Construct and return a clone
    virtual tmp<fvPatchScalarField> clone() const
    {
        return tmp<fvPatchScalarField>(
            new multiFreqScalarFvPatchField(*this));
    }

    //- Construct as copy setting internal field reference
    multiFreqScalarFvPatchField(
        const multiFreqScalarFvPatchField&,
        const DimensionedField<scalar, volMesh>&);

    //- Construct and return a clone setting internal field reference
    virtual tmp<fvPatchScalarField> clone(
        const DimensionedField<scalar, volMesh>& iF) const
    {
        return tmp<fvPatchScalarField>(
            new multiFreqScalarFvPatchField(*this, iF));
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
