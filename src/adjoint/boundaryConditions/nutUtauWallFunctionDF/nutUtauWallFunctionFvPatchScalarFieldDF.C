/*---------------------------------------------------------------------------*\
    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    This file is a modified combination of the OpenFOAM's source codes
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nutWallFunctions
    /nutUSpaldingWallFunction/nutUSpaldingWallFunctionFvPatchScalarField.C,
    src/TurbulenceModels/turbulenceModels/derivedFvPatchFields/wallFunctions/nutWallFunctions
    /nutkRoughWallFunction/nutkRoughWallFunctionFvPatchScalarField.C

i.e. a simplified nut wall function usable with any turbulence model (only available for flat plates for now).

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

\*---------------------------------------------------------------------------*/
#include "nutUtauWallFunctionFvPatchScalarFieldDF.H"
#include "DATurbulenceModel.H"
#include "IOstreams.H"
#include "error.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
addToRunTimeSelectionTable(nutWallFunctionFvPatchScalarField, nutUtauWallFunctionFvPatchScalarFieldDF, dictionary);
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


scalar nutUtauWallFunctionFvPatchScalarFieldDF::fnRough
(
    const scalar KsPlus,
    const scalar Cs
) const
{
    // Return fn based on non-dimensional roughness height

    if (KsPlus < 90.0)
    {
        return pow
        (
            (KsPlus - 2.25)/87.75 + Cs*KsPlus,
            sin(0.4258*(log(KsPlus) - 0.811))
        );
    }
    else
    {
        return (1.0 + Cs*KsPlus);
    }
}

tmp<scalarField> nutUtauWallFunctionFvPatchScalarFieldDF::calcNut() const
{
    const label patchi = patch().index();

    const turbulenceModel& turbModel = fvPatchField<scalar>::db().lookupObject<turbulenceModel>
    (
        IOobject::groupName
        (
            turbulenceModel::propertiesName,
            internalField().group()
        )
    );

    const fvPatchVectorField& Uw = turbModel.U().boundaryField()[patchi];
    const scalarField magGradU(mag(Uw.snGrad()));

    const tmp<scalarField> tnuw = turbModel.nu(patchi);
    const scalarField& nuw = tnuw();

    tmp<scalarField> tnutw(new scalarField(*this));
    scalarField& nutw = tnutw.ref();

    const tmp<scalarField> tuTau = calcUTau(magGradU);
    const scalarField& uTau = tuTau();

    forAll(nutw, facei)
    {
        scalar limitingNutw = max(nutw[facei], nuw[facei]);
        nutw[facei] = max(min(
                        (sqr(uTau[facei])/(magGradU[facei] + ROOTVSMALL)) - nuw[facei],
                        2*limitingNutw),
                        0.5*limitingNutw);
    }

    return tnutw;
}

tmp<scalarField> nutUtauWallFunctionFvPatchScalarFieldDF::calcUTau
(
    const scalarField& magGradU
) const
{
    const label patchi = patch().index();

    const turbulenceModel& turbModel = fvPatchField<scalar>::db().lookupObject<turbulenceModel>
    (
        IOobject::groupName
        (
            turbulenceModel::propertiesName,
            internalField().group()
        )
    );
    const scalarField& y = turbModel.y()[patchi];

    const fvPatchVectorField& Uw = turbModel.U().boundaryField()[patchi];
    const scalarField magUp(mag(Uw.patchInternalField() - Uw));

    const tmp<scalarField> tnuw = turbModel.nu(patchi);
    const scalarField& nuw = tnuw();

    const scalarField& nutw = *this;

    tmp<scalarField> tuTau(new scalarField(patch().size(), 0.0));
    scalarField& uTau = tuTau.ref();

    scalarField etaPatch_ = etaWallDV.boundaryField()[patchi];

    forAll(uTau, facei)
    {
        scalar ut = sqrt((nutw[facei] + nuw[facei])*magGradU[facei]);

        if (ut > ROOTVSMALL)
        {
            int iter = 0;
            scalar err = GREAT;

            do
            {
                scalar yPlus = ut * y[facei] / nuw[facei];
                scalar KsPlus = ut * Ks_[facei] / nuw[facei];
                scalar Edash = E_;
                yPlus = max(yPlus, VSMALL);
                Edash = max(Edash, VSMALL);


                scalar dfn = 0.0;

                if (2.25 < KsPlus)
                {
		    //Info<< "Wall is rough " << endl;
                    scalar fn = fnRough(KsPlus, Cs_[facei]);
                    Edash /= fn;
                    scalar logTerm = log(yPlus * Edash / etaPatch_[facei]);

                    scalar f = ut / kappa_ * logTerm - magUp[facei];

                    if (KsPlus < 90.0)
                    {
			//Info<< "Transitionally rough" << endl;
                        const scalar a = max((KsPlus - 2.25)/87.75 + Cs_[facei] * KsPlus, VSMALL);
                        const scalar b = sin(0.4258 * (log(KsPlus) - 0.811));

                        const scalar aPrime = 1.0/87.75 + Cs_[facei];
                        const scalar bPrime = 0.4258 * cos(0.4258 * (log(KsPlus) - 0.811)) / KsPlus;

                        dfn = pow(a, b) * (bPrime * log(a) + b * aPrime / a);
                    }
                    else
                    {
			//Info<< "Fully rough" << endl;
                        dfn = Cs_[facei] * Ks_[facei] / nuw[facei];
                    }

                    scalar df = (1.0 / kappa_) * (1.0 + logTerm - ut / fn * dfn);
                    df = max(df, VSMALL);

                    scalar uTauNew = ut - f / df;
                    err = mag((ut - uTauNew) / ut);
                    ut = uTauNew;
                }
                else
                {
		    //Info<< "Wall is smooth"<< endl;
                    scalar logTerm = log(yPlus * Edash / etaPatch_[facei]);

                    scalar f = ut / kappa_ * logTerm - magUp[facei];
                    scalar df = (1.0 + logTerm) / kappa_;
                    df = max(df, VSMALL);

                    scalar uTauNew = ut - f / df;
                    uTauNew = min(max(uTauNew, 0.01), 10.0);

                    err = mag((ut - uTauNew) / ut);
                    ut = uTauNew;
                }

            } while (ut > ROOTVSMALL && err > 0.01 && ++iter < 10);

            uTau[facei] = max(0.0, ut);
//            Info<< "uTau = " << uTau[facei] <<
//                   ", nutw = " << nutw[facei] <<
//                   ", gradU = " << magGradU[facei] <<
//                   ", nuw = " << nuw[facei] << endl;

        }
    }

    return tuTau;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //
//1)
nutUtauWallFunctionFvPatchScalarFieldDF::
nutUtauWallFunctionFvPatchScalarFieldDF
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    nutWallFunctionFvPatchScalarField(p, iF),
    Ks_(p.size(), 0.0),
    Cs_(p.size(), 0.0),
    etaWallDV(registerEtaField(iF.db()))
{
}

//2)
nutUtauWallFunctionFvPatchScalarFieldDF::
nutUtauWallFunctionFvPatchScalarFieldDF
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    nutWallFunctionFvPatchScalarField(p, iF, dict),
    Ks_("Ks", dict, p.size()),
    Cs_("Cs", dict, p.size()),
    etaWallDV(registerEtaField(iF.db()))
{
}

//3)
nutUtauWallFunctionFvPatchScalarFieldDF::
nutUtauWallFunctionFvPatchScalarFieldDF
(
    const nutUtauWallFunctionFvPatchScalarFieldDF& wfpsf
)
:
    nutWallFunctionFvPatchScalarField(wfpsf),
    Ks_(wfpsf.Ks_),
    Cs_(wfpsf.Cs_),
    etaWallDV(wfpsf.etaWallDV)
{
}

//4)
nutUtauWallFunctionFvPatchScalarFieldDF::
nutUtauWallFunctionFvPatchScalarFieldDF
(
    const nutUtauWallFunctionFvPatchScalarFieldDF& wfpsf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    nutWallFunctionFvPatchScalarField(wfpsf, iF),
    Ks_(wfpsf.Ks_),
    Cs_(wfpsf.Cs_),
    etaWallDV(registerEtaField(iF.db()))
{
}

//5)
nutUtauWallFunctionFvPatchScalarFieldDF::
nutUtauWallFunctionFvPatchScalarFieldDF
(
    const nutUtauWallFunctionFvPatchScalarFieldDF& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    nutWallFunctionFvPatchScalarField(ptf, p, iF, mapper),
    Ks_(ptf.Ks_, mapper),
    Cs_(ptf.Cs_, mapper),
    etaWallDV(registerEtaField(iF.db()))
{
}


// * * * * * * * * * * * * Member Functions * * * * * * * * * * * * * * * * * //

tmp<scalarField> nutUtauWallFunctionFvPatchScalarFieldDF::yPlus() const
{
    const label patchi = patch().index();

    const turbulenceModel& turbModel = fvPatchField<scalar>::db().lookupObject<turbulenceModel>
    (
        IOobject::groupName
        (
            turbulenceModel::propertiesName,
            internalField().group()
        )
    );
    const scalarField& y = turbModel.y()[patchi];
    const fvPatchVectorField& Uw = turbModel.U().boundaryField()[patchi];
    const tmp<scalarField> tnuw = turbModel.nu(patchi);
    const scalarField& nuw = tnuw();

    return y*calcUTau(mag(Uw.snGrad()))/nuw;
}

void Foam::nutUtauWallFunctionFvPatchScalarFieldDF::autoMap
(
    const fvPatchFieldMapper& m
)
{
    nutWallFunctionFvPatchScalarField::autoMap(m);
    Ks_.autoMap(m);
    Cs_.autoMap(m);
    etaWallDV.autoMap(m);
}

void Foam::nutUtauWallFunctionFvPatchScalarFieldDF::rmap
(
    const fvPatchScalarField& ptf,
    const labelList& addr
)
{
    nutWallFunctionFvPatchScalarField::rmap(ptf, addr);

    const nutUtauWallFunctionFvPatchScalarFieldDF& nrwfpsf =
        refCast<const nutUtauWallFunctionFvPatchScalarFieldDF>(ptf);

    Ks_.rmap(nrwfpsf.Ks_, addr);
    Cs_.rmap(nrwfpsf.Cs_, addr);
    etaWallDV.rmap(nrwfpsf.etaWallDV, addr);
}

volScalarField& nutUtauWallFunctionFvPatchScalarFieldDF::registerEtaField(const objectRegistry& db)
{
    if (!this->db().foundObject<volScalarField>("etaWallDV"))
    {

        volScalarField* etaWallDVPtr = new volScalarField
        (
            IOobject(
                "etaWallDV",
                this->db().time().timeName(),
                this->db(),
                IOobject::NO_READ,
                IOobject::AUTO_WRITE,
                true  // no register automatically
            ),
            this->patch().boundaryMesh().mesh(),
            dimensionedScalar("etaWallDV", dimensionSet(0,0,0,0,0,0,0), 20.0),
	    fixedValueFvPatchScalarField::typeName
        );
    }
    // Now that we know it exists, look it up and return it
    return const_cast<volScalarField&>
    (
        db.lookupObject<volScalarField>("etaWallDV")
    );
}

void nutUtauWallFunctionFvPatchScalarFieldDF::write(Ostream& os) const
{
    nutWallFunctionFvPatchScalarField::write(os);

    os.writeKeyword("etaWallDV") << etaWallDV << token::END_STATEMENT << nl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

makePatchTypeField(
    fvPatchScalarField,
    nutUtauWallFunctionFvPatchScalarFieldDF);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
