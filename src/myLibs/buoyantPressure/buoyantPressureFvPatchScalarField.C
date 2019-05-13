/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "buoyantPressureFvPatchScalarField.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "addToRunTimeSelectionTable.H"
#include "uniformDimensionedFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::buoyantPressureFvPatchScalarField::buoyantPressureFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedGradientFvPatchScalarField(p, iF)
{}


Foam::buoyantPressureFvPatchScalarField::buoyantPressureFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    fixedGradientFvPatchScalarField(p, iF)
{
    patchType() = dict.lookupOrDefault<word>("patchType", word::null);
    if (dict.found("value") && dict.found("gradient"))
    {
        fvPatchField<scalar>::operator=
        (
            scalarField("value", dict, p.size())
        );
        gradient() = scalarField("gradient", dict, p.size());
    }
    else
    {
        fvPatchField<scalar>::operator=(patchInternalField());
        gradient() = 0.0;
    }
}


Foam::buoyantPressureFvPatchScalarField::buoyantPressureFvPatchScalarField
(
    const buoyantPressureFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedGradientFvPatchScalarField(p, iF)
{
    patchType() = ptf.patchType();

    // Map gradient. Set unmapped values and overwrite with mapped ptf
    gradient() = 0.0;
    gradient().map(ptf.gradient(), mapper);

    // Evaluate the value field from the gradient if the internal field is valid
    if (notNull(iF))
    {
        if (iF.size())
        {
            // Note: cannot ask for nf() if zero faces

            scalarField::operator=
            (
                //patchInternalField() + gradient()/patch().deltaCoeffs()
                // ***HGW Hack to avoid the construction of mesh.deltaCoeffs
                // which fails for AMI patches for some mapping operations
                patchInternalField()
              + gradient()*(patch().nf() & patch().delta())
            );
    }
    else
    {
        // Enforce mapping of values so we have a valid starting value. This
        // constructor is used when reconstructing fields
        this->map(ptf, mapper);
        }
    }
    else
    {
        // Enforce mapping of values so we have a valid starting value
        this->map(ptf, mapper);
    }
}


Foam::buoyantPressureFvPatchScalarField::buoyantPressureFvPatchScalarField
(
    const buoyantPressureFvPatchScalarField& wbppsf
)
:
    fixedGradientFvPatchScalarField(wbppsf)
{}


Foam::buoyantPressureFvPatchScalarField::buoyantPressureFvPatchScalarField
(
    const buoyantPressureFvPatchScalarField& wbppsf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    fixedGradientFvPatchScalarField(wbppsf, iF)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
void Foam::buoyantPressureFvPatchScalarField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const uniformDimensionedVectorField& g =
        db().lookupObject<uniformDimensionedVectorField>("g");

    const fvPatchField<scalar>& rho =
        patch().lookupPatchField<volScalarField, scalar>("rhok");

    // If the variable name is "p_rgh", "ph_rgh" or "pd"
    // assume it is p? - rho*g.h and set the gradient appropriately.
    // Otherwise assume the variable is the static pressure.
    if
    (
        this->internalField().name() == "p_rgh"
     || this->internalField().name() == "ph_rgh"
     || this->internalField().name() == "pd"
    )
    {
        gradient() = -rho.snGrad()*(g.value() & patch().Cf());
    }
    else
    {
        gradient() = rho*(g.value() & patch().nf());
    }

    fixedGradientFvPatchScalarField::updateCoeffs();
}


void Foam::buoyantPressureFvPatchScalarField::write(Ostream& os) const
{
    fixedGradientFvPatchScalarField::write(os);
    writeEntry("value", os);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField
    (
        fvPatchScalarField,
        buoyantPressureFvPatchScalarField
    );
}


// ************************************************************************* //
