/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        A modified version of MRF that allows changing the rotation speed at
        the runtime to enable derivatives wrt the rotation speed

    This class is modified from OpenFOAM's source code
    src/finiteVolume/cfdTools/general/MRF

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

#include "MRFZoneListDF.H"
#include "volFields.H"
#include "fixedValueFvsPatchFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::MRFZoneListDF::MRFZoneListDF(
    const fvMesh& mesh,
    const dictionary& dict)
    : PtrList<MRFZoneDF>(),
      mesh_(mesh)
{
    reset(dict);

    active(true);
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::MRFZoneListDF::~MRFZoneListDF()
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::MRFZoneListDF::active(const bool warn) const
{
    bool a = false;
    forAll(*this, i)
    {
        a = a || this->operator[](i).active();
    }

    if (warn && this->size() && !a)
    {
        Info << "    No MRF zones active" << endl;
    }

    return a;
}

void Foam::MRFZoneListDF::reset(const dictionary& dict)
{
    label count = 0;
    for (const entry& dEntry : dict)
    {
        if (dEntry.isDict())
        {
            ++count;
        }
    }

    this->resize(count);

    count = 0;
    for (const entry& dEntry : dict)
    {
        if (dEntry.isDict())
        {
            const word& name = dEntry.keyword();
            const dictionary& modelDict = dEntry.dict();

            Info << "    creating MRF zone: " << name << endl;

            this->set(
                count++,
                new MRFZoneDF(name, mesh_, modelDict));
        }
    }
}

bool Foam::MRFZoneListDF::read(const dictionary& dict)
{
    bool allOk = true;
    forAll(*this, i)
    {
        MRFZoneDF& pm = this->operator[](i);
        bool ok = pm.read(dict.subDict(pm.name()));
        allOk = (allOk && ok);
    }
    return allOk;
}

bool Foam::MRFZoneListDF::writeData(Ostream& os) const
{
    forAll(*this, i)
    {
        os << nl;
        this->operator[](i).writeData(os);
    }

    return os.good();
}

void Foam::MRFZoneListDF::addAcceleration(
    const volVectorField& U,
    volVectorField& ddtU) const
{
    forAll(*this, i)
    {
        operator[](i).addCoriolis(U, ddtU);
    }
}

void Foam::MRFZoneListDF::addAcceleration(fvVectorMatrix& UEqn) const
{
    forAll(*this, i)
    {
        operator[](i).addCoriolis(UEqn);
    }
}

void Foam::MRFZoneListDF::addAcceleration(
    const volScalarField& rho,
    fvVectorMatrix& UEqn) const
{
    forAll(*this, i)
    {
        operator[](i).addCoriolis(rho, UEqn);
    }
}

Foam::tmp<Foam::volVectorField> Foam::MRFZoneListDF::DDt(
    const volVectorField& U) const
{
    tmp<volVectorField> tacceleration(
        new volVectorField(
            IOobject(
                "MRFZoneListDF:acceleration",
                U.mesh().time().timeName(),
                U.mesh()),
            U.mesh(),
            dimensionedVector(U.dimensions() / dimTime, Zero)));
    volVectorField& acceleration = tacceleration.ref();

    forAll(*this, i)
    {
        operator[](i).addCoriolis(U, acceleration);
    }

    return tacceleration;
}

Foam::tmp<Foam::volVectorField> Foam::MRFZoneListDF::DDt(
    const volScalarField& rho,
    const volVectorField& U) const
{
    return rho * DDt(U);
}

void Foam::MRFZoneListDF::makeRelative(volVectorField& U) const
{
    forAll(*this, i)
    {
        operator[](i).makeRelative(U);
    }
}

void Foam::MRFZoneListDF::makeRelative(surfaceScalarField& phi) const
{
    forAll(*this, i)
    {
        operator[](i).makeRelative(phi);
    }
}

Foam::tmp<Foam::surfaceScalarField> Foam::MRFZoneListDF::relative(
    const tmp<surfaceScalarField>& tphi) const
{
    if (size())
    {
        tmp<surfaceScalarField> rphi(
            New(
                tphi,
                "relative(" + tphi().name() + ')',
                tphi().dimensions(),
                true));

        makeRelative(rphi.ref());

        tphi.clear();

        return rphi;
    }
    else
    {
        return tmp<surfaceScalarField>(tphi, true);
    }
}

Foam::tmp<Foam::FieldField<Foam::fvsPatchField, Foam::scalar>>
Foam::MRFZoneListDF::relative(
    const tmp<FieldField<fvsPatchField, scalar>>& tphi) const
{
    if (size())
    {
        tmp<FieldField<fvsPatchField, scalar>> rphi(New(tphi, true));

        forAll(*this, i)
        {
            operator[](i).makeRelative(rphi.ref());
        }

        tphi.clear();

        return rphi;
    }
    else
    {
        return tmp<FieldField<fvsPatchField, scalar>>(tphi, true);
    }
}

Foam::tmp<Foam::Field<Foam::scalar>>
Foam::MRFZoneListDF::relative(
    const tmp<Field<scalar>>& tphi,
    const label patchi) const
{
    if (size())
    {
        tmp<Field<scalar>> rphi(New(tphi, true));

        forAll(*this, i)
        {
            operator[](i).makeRelative(rphi.ref(), patchi);
        }

        tphi.clear();

        return rphi;
    }
    else
    {
        return tmp<Field<scalar>>(tphi, true);
    }
}

void Foam::MRFZoneListDF::makeRelative(
    const surfaceScalarField& rho,
    surfaceScalarField& phi) const
{
    forAll(*this, i)
    {
        operator[](i).makeRelative(rho, phi);
    }
}

void Foam::MRFZoneListDF::makeAbsolute(volVectorField& U) const
{
    forAll(*this, i)
    {
        operator[](i).makeAbsolute(U);
    }
}

void Foam::MRFZoneListDF::makeAbsolute(surfaceScalarField& phi) const
{
    forAll(*this, i)
    {
        operator[](i).makeAbsolute(phi);
    }
}

Foam::tmp<Foam::surfaceScalarField> Foam::MRFZoneListDF::absolute(
    const tmp<surfaceScalarField>& tphi) const
{
    if (size())
    {
        tmp<surfaceScalarField> rphi(
            New(
                tphi,
                "absolute(" + tphi().name() + ')',
                tphi().dimensions(),
                true));

        makeAbsolute(rphi.ref());

        tphi.clear();

        return rphi;
    }
    else
    {
        return tmp<surfaceScalarField>(tphi, true);
    }
}

void Foam::MRFZoneListDF::makeAbsolute(
    const surfaceScalarField& rho,
    surfaceScalarField& phi) const
{
    forAll(*this, i)
    {
        operator[](i).makeAbsolute(rho, phi);
    }
}

void Foam::MRFZoneListDF::correctBoundaryVelocity(volVectorField& U) const
{
    forAll(*this, i)
    {
        operator[](i).correctBoundaryVelocity(U);
    }
}

void Foam::MRFZoneListDF::correctBoundaryFlux(
    const volVectorField& U,
    surfaceScalarField& phi) const
{
    FieldField<fvsPatchField, scalar> Uf(
        relative(mesh_.Sf().boundaryField() & U.boundaryField()));

    surfaceScalarField::Boundary& phibf = phi.boundaryFieldRef();

    forAll(mesh_.boundary(), patchi)
    {
        if (
            isA<fixedValueFvsPatchScalarField>(phibf[patchi]))
        {
            phibf[patchi] == Uf[patchi];
        }
    }
}

const Foam::scalar& Foam::MRFZoneListDF::getOmegaRef() const
{
    label nObjs = 0;
    forAll(*this, i)
    {
        nObjs++;
    }

    if (nObjs > 1)
    {
        FatalErrorInFunction
            << "Do not support more than one MRF zones!"
            << exit(FatalError);
    }

    return operator[](0).getOmegaRef();
}

const Foam::vector& Foam::MRFZoneListDF::getOriginRef() const
{
    label nObjs = 0;
    forAll(*this, i)
    {
        nObjs++;
    }

    if (nObjs > 1)
    {
        FatalErrorInFunction
            << "Do not support more than one MRF zones!"
            << exit(FatalError);
    }

    return operator[](0).getOriginRef();
}

const Foam::vector& Foam::MRFZoneListDF::getAxisRef() const
{
    label nObjs = 0;
    forAll(*this, i)
    {
        nObjs++;
    }

    if (nObjs > 1)
    {
        FatalErrorInFunction
            << "Do not support more than one MRF zones!"
            << exit(FatalError);
    }

    return operator[](0).getAxisRef();
}

const Foam::wordRes& Foam::MRFZoneListDF::getExcludedPatchNamesRef() const
{
    label nObjs = 0;
    forAll(*this, i)
    {
        nObjs++;
    }

    if (nObjs > 1)
    {
        FatalErrorInFunction
            << "Do not support more than one MRF zones!"
            << exit(FatalError);
    }

    return operator[](0).getExcludedPatchNamesRef();
}

void Foam::MRFZoneListDF::update()
{
    if (mesh_.topoChanging())
    {
        forAll(*this, i)
        {
            operator[](i).update();
        }
    }
}

// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

Foam::Ostream& Foam::operator<<(
    Ostream& os,
    const MRFZoneListDF& models)
{
    models.writeData(os);
    return os;
}

// ************************************************************************* //
