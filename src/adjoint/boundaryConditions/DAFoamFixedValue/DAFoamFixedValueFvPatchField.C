#include "DAFoamFixedValueFvPatchField.H"
#include "addToRunTimeSelectionTable.H"

namespace Foam
{

// --- Constructors ---

// Corrected syntax: uses a colon ':' to call the base class constructor
DAFoamFixedValueFvPatchField::DAFoamFixedValueFvPatchField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(p, iF, Zero),
    dvField_(registerDVField(this->db())) // Initialize here


{}

// Corrected syntax for all other constructors as well
DAFoamFixedValueFvPatchField::DAFoamFixedValueFvPatchField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchVectorField(p, iF, Zero),
    dvField_(registerDVField(this->db())) // Initialize here

{
    patchType() = dict.lookupOrDefault<word>("patchType", word::null);
}

DAFoamFixedValueFvPatchField::DAFoamFixedValueFvPatchField
(
    const DAFoamFixedValueFvPatchField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchVectorField(p, iF, Zero),
    dvField_(registerDVField(this->db())) // Initialize here

{}

DAFoamFixedValueFvPatchField::DAFoamFixedValueFvPatchField
(
    const DAFoamFixedValueFvPatchField& mwvpvf
)
:
    fixedValueFvPatchVectorField(mwvpvf),
    dvField_(registerDVField(this->db())) // Initialize here

{}

DAFoamFixedValueFvPatchField::DAFoamFixedValueFvPatchField
(
    const DAFoamFixedValueFvPatchField& mwvpvf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchVectorField(mwvpvf, iF),
    dvField_(registerDVField(this->db())) // Initialize here

{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAFoamFixedValueFvPatchField::write(Ostream& os) const
{
    fvPatchVectorField::write(os);
}

// --- NEW OVERRIDDEN FUNCTION ---
void DAFoamFixedValueFvPatchField::updateCoeffs()
{
    // The dvField_ member variable was initialized by the constructor and is
    // now a valid reference to the design variable field in the database.

    // 1. Get the patch field from our stored member variable
    const fvPatchField<vector>& dvPatchField =
        dvField_.boundaryField()[this->patch().index()];

    // 2. Assign the values from the DV field to this patch
    this->operator==(dvPatchField);
}

volVectorField& DAFoamFixedValueFvPatchField::registerDVField
(
    const objectRegistry& db // Pass in the mesh database
)
{
    // Define the name of the design variable field
    word dvFieldName = "plateVelocityDV";

    // Check if the field already exists
    if (!this->db().foundObject<volVectorField>(dvFieldName))
    {
        Info << "DAFoamFixedValue BC: Creating and registering design variable field '"
             << dvFieldName << "'" << endl;

        // If it doesn't exist, create it...
        volVectorField* dvFieldPtr = new volVectorField
        (
            IOobject
            (
                dvFieldName,
                this->db().time().timeName(),
                this->db(), // Register it to the same database
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            // Create a field with the same mesh and zero value
            this->patch().boundaryMesh().mesh(),
            dimensionedVector(dvFieldName, dimVelocity, vector::zero),
            fixedValueFvPatchVectorField::typeName
        );
    }

    // Now that we know it exists, look it up and return a non-const reference
    return const_cast<volVectorField&>
    (
        db.lookupObject<volVectorField>(dvFieldName)
    );
}

    makePatchTypeField
    (
        fvPatchVectorField,
        DAFoamFixedValueFvPatchField
    );


} // End namespace Foam