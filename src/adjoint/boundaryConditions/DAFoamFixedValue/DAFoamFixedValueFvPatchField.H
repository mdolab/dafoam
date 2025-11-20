#ifndef DAFoamFixedValueFvPatchField_H
#define DAFoamFixedValueFvPatchField_H

#include "fixedValueFvPatchFields.H"
#include "DAModel.H"
#include "DAOption.H"
#include "fvMesh.H"
namespace Foam
{


class DAFoamFixedValueFvPatchField
:
    public fixedValueFvPatchVectorField
{
private:
        //- Helper function to find or create the design variable field
       volVectorField& registerDVField(const objectRegistry& db);
public:

    //- Runtime type information
    TypeName("DAFoamFixedValue");

    volVectorField& dvField_;
    // Constructors

        //- Construct from patch and internal field
        DAFoamFixedValueFvPatchField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct from patch, internal field and dictionary
        DAFoamFixedValueFvPatchField
        (
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const dictionary&
        );

        //- Construct by mapping given noSlipFvPatchVectorField
        //  onto a new patch
        DAFoamFixedValueFvPatchField
        (
            const DAFoamFixedValueFvPatchField&,
            const fvPatch&,
            const DimensionedField<vector, volMesh>&,
            const fvPatchFieldMapper&
        );

        //- Construct as copy
        DAFoamFixedValueFvPatchField
        (
            const DAFoamFixedValueFvPatchField&
        );

        //- Construct and return a clone
        virtual tmp<fvPatchVectorField> clone() const
        {
            return tmp<fvPatchVectorField>
            (
                new DAFoamFixedValueFvPatchField(*this)
            );
        }

        //- Construct as copy setting internal field reference
        DAFoamFixedValueFvPatchField
        (
            const DAFoamFixedValueFvPatchField&,
            const DimensionedField<vector, volMesh>&
        );

        //- Construct and return a clone setting internal field reference
        virtual tmp<fvPatchVectorField> clone
        (
            const DimensionedField<vector, volMesh>& iF
        ) const
        {
            return tmp<fvPatchVectorField>
            (
                new DAFoamFixedValueFvPatchField(*this, iF)
            );
        }
            // Member function
            
        //- Update the coefficients of the patch
        virtual void updateCoeffs();
        //- Write
        virtual void write(Ostream&) const;
};

} // End namespace Foam
#endif
