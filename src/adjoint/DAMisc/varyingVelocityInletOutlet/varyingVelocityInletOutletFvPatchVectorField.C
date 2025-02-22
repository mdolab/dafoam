/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2017 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2017 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "varyingVelocityInletOutletFvPatchVectorField.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::varyingVelocityInletOutletFvPatchVectorField::
    varyingVelocityInletOutletFvPatchVectorField(
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF)
    : mixedFvPatchField<vector>(p, iF),
      phiName_("phi"),
      U0_(0.0),
      URate_(0.0),
      flowComponent_(0),
      normalComponent_(1),
      alpha0_(0),
      alphaRate_(0)
{
    forAll(this->refValue(), idxI)
    {
        this->refValue()[idxI] = pTraits<vector>::zero;
    }
    this->refGrad() = pTraits<vector>::zero;
    this->valueFraction() = 0.0;
}

Foam::varyingVelocityInletOutletFvPatchVectorField::
    varyingVelocityInletOutletFvPatchVectorField(
        const varyingVelocityInletOutletFvPatchVectorField& ptf,
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const fvPatchFieldMapper& mapper)
    : mixedFvPatchField<vector>(ptf, p, iF, mapper),
      phiName_(ptf.phiName_),
      U0_(ptf.U0_),
      URate_(ptf.URate_),
      flowComponent_(ptf.flowComponent_),
      normalComponent_(ptf.normalComponent_),
      alpha0_(ptf.alpha0_),
      alphaRate_(ptf.alphaRate_)
{
}

Foam::varyingVelocityInletOutletFvPatchVectorField::
    varyingVelocityInletOutletFvPatchVectorField(
        const fvPatch& p,
        const DimensionedField<vector, volMesh>& iF,
        const dictionary& dict)
    : mixedFvPatchField<vector>(p, iF),
      phiName_(dict.lookupOrDefault<word>("phi", "phi")),
      U0_(dict.lookupOrDefault<scalar>("U0", 0.0)),
      URate_(dict.lookupOrDefault<scalar>("URate", 0.0)),
      flowComponent_(dict.lookupOrDefault<label>("flowComponent", 0)),
      normalComponent_(dict.lookupOrDefault<label>("normalComponent", 1)),
      alpha0_(dict.lookupOrDefault<scalar>("alpha0", 0.0)),
      alphaRate_(dict.lookupOrDefault<scalar>("alphaRate", 0.0))
{
    this->patchType() = dict.lookupOrDefault<word>("patchType", word::null);

    vector UInit = vector::zero;
    UInit[flowComponent_] = U0_ * cos(alpha0_);
    UInit[normalComponent_] = U0_ * sin(alpha0_);
    forAll(this->refValue(), idxI)
    {
        this->refValue()[idxI] = UInit;
    }

    if (dict.found("value"))
    {
        fvPatchField<vector>::operator=(
            vectorField("value", dict, p.size()));
    }
    else
    {
        fvPatchField<vector>::operator=(this->refValue());
    }

    this->refGrad() = pTraits<vector>::zero;
    this->valueFraction() = 0.0;
}

Foam::varyingVelocityInletOutletFvPatchVectorField::
    varyingVelocityInletOutletFvPatchVectorField(
        const varyingVelocityInletOutletFvPatchVectorField& ptf)
    : mixedFvPatchField<vector>(ptf),
      phiName_(ptf.phiName_),
      U0_(ptf.U0_),
      URate_(ptf.URate_),
      flowComponent_(ptf.flowComponent_),
      normalComponent_(ptf.normalComponent_),
      alpha0_(ptf.alpha0_),
      alphaRate_(ptf.alphaRate_)
{
}

Foam::varyingVelocityInletOutletFvPatchVectorField::
    varyingVelocityInletOutletFvPatchVectorField(
        const varyingVelocityInletOutletFvPatchVectorField& ptf,
        const DimensionedField<vector, volMesh>& iF)
    : mixedFvPatchField<vector>(ptf, iF),
      phiName_(ptf.phiName_),
      U0_(ptf.U0_),
      URate_(ptf.URate_),
      flowComponent_(ptf.flowComponent_),
      normalComponent_(ptf.normalComponent_),
      alpha0_(ptf.alpha0_),
      alphaRate_(ptf.alphaRate_)
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::varyingVelocityInletOutletFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const scalar t = this->db().time().timeOutputValue();
    scalar alpha = alpha0_ + t * alphaRate_;
    scalar U = U0_ + t * URate_;

    const Field<scalar>& phip =
        this->patch().template lookupPatchField<surfaceScalarField, scalar>(phiName_);

    forAll(this->refValue(), faceI)
    {
        this->refValue()[faceI][flowComponent_] = U * cos(alpha) * (1.0 - pos0(phip[faceI]));
        this->refValue()[faceI][normalComponent_] = U * sin(alpha) * (1.0 - pos0(phip[faceI]));
    }

    this->valueFraction() = 1.0 - pos0(phip);

    mixedFvPatchField<vector>::updateCoeffs();
}

void Foam::varyingVelocityInletOutletFvPatchVectorField::write(Ostream& os)
    const
{
    fvPatchVectorField::write(os);
    os.writeEntry("phiName", phiName_);
    os.writeEntry("U0", U0_);
    os.writeEntry("URate", URate_);
    os.writeEntry("flowComponent", flowComponent_);
    os.writeEntry("normalComponent", normalComponent_);
    os.writeEntry("alpha0", alpha0_);
    os.writeEntry("alphaRate", alphaRate_);
    //writeEntry("value", os);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::varyingVelocityInletOutletFvPatchVectorField::operator=(
    const fvPatchVectorField& ptf)
{
    fvPatchVectorField::operator=(
        this->valueFraction() * this->refValue()
        + (1 - this->valueFraction()) * ptf);
}

namespace Foam
{
makePatchTypeField(
    fvPatchVectorField,
    varyingVelocityInletOutletFvPatchVectorField);
}

// ************************************************************************* //
