/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "SpalartAllmarasFv3.H"
#include "fvOptions.H"
#include "bound.H"
#include "wallDist.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace RASModels
{

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::chi() const
{
    return nuTilda_/this->nu();
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::fv1
(
    const volScalarField& chi
) const
{
    const volScalarField chi3(pow3(chi));
    return chi3/(chi3 + pow3(Cv1_));
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::fv2
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    return 1.0/pow3(scalar(1) + chi/Cv2_);
}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::fv3
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{

    const volScalarField chiByCv2((1/Cv2_)*chi);

    return
        (scalar(1) + chi*fv1)
       *(1/Cv2_)
       *(3*(scalar(1) + chiByCv2) + sqr(chiByCv2))
       /pow3(scalar(1) + chiByCv2);

}

template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::fw
(
    const volScalarField& Stilda
) const
{
    volScalarField r
    (
        min
        (
            nuTilda_
           /(
               max
               (
                   Stilda,
                   dimensionedScalar("SMALL", Stilda.dimensions(), SMALL)
               )
              *sqr(kappa_*y_)
            ),
            scalar(10.0)
        )
    );
    r.boundaryFieldRef() == 0.0;

    const volScalarField g(r + Cw2_*(pow6(r) - r));

    return g*pow((1.0 + pow6(Cw3_))/(pow6(g) + pow6(Cw3_)), 1.0/6.0);
}


template<class BasicTurbulenceModel>
void SpalartAllmarasFv3<BasicTurbulenceModel>::correctNut
(
    const volScalarField& fv1
)
{
    this->nut_ = nuTilda_*fv1;
    this->nut_.correctBoundaryConditions();
    fv::options::New(this->mesh_).correct(this->nut_);

    BasicTurbulenceModel::correctNut();
}


template<class BasicTurbulenceModel>
void SpalartAllmarasFv3<BasicTurbulenceModel>::correctNut()
{
    correctNut(fv1(this->chi()));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
SpalartAllmarasFv3<BasicTurbulenceModel>::SpalartAllmarasFv3
(
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName,
    const word& type
)
:
    eddyViscosity<RASModel<BasicTurbulenceModel>>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),

    sigmaNut_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaNut",
            this->coeffDict_,
            0.66666
        )
    ),
    kappa_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "kappa",
            this->coeffDict_,
            0.41
        )
    ),

    Cb1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cb1",
            this->coeffDict_,
            0.1355
        )
    ),
    Cb2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cb2",
            this->coeffDict_,
            0.622
        )
    ),
    Cw1_(Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_),
    Cw2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cw2",
            this->coeffDict_,
            0.3
        )
    ),
    Cw3_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cw3",
            this->coeffDict_,
            2.0
        )
    ),
    Cv1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cv1",
            this->coeffDict_,
            7.1
        )
    ),
    Cv2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cv2",
            this->coeffDict_,
            5.0
        )
    ),
    nuTildaMin_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "nuTildaMin",
            this->coeffDict_,
            SMALL
        )
    ),

    nuTilda_
    (
        IOobject
        (
            "nuTilda",
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        this->mesh_
    ),

    y_(wallDist::New(this->mesh_).y())
{
    if (type == typeName)
    {
        this->printCoeffs(type);
    }

}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool SpalartAllmarasFv3<BasicTurbulenceModel>::read()
{
    if (eddyViscosity<RASModel<BasicTurbulenceModel>>::read())
    {
        sigmaNut_.readIfPresent(this->coeffDict());
        kappa_.readIfPresent(this->coeffDict());

        Cb1_.readIfPresent(this->coeffDict());
        Cb2_.readIfPresent(this->coeffDict());
        Cw1_ = Cb1_/sqr(kappa_) + (1.0 + Cb2_)/sigmaNut_;
        Cw2_.readIfPresent(this->coeffDict());
        Cw3_.readIfPresent(this->coeffDict());
        Cv1_.readIfPresent(this->coeffDict());
        Cv2_.readIfPresent(this->coeffDict());

        return true;
    }
    else
    {
        return false;
    }
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::DnuTildaEff() const
{
    return tmp<volScalarField>
    (
        new volScalarField("DnuTildaEff", (nuTilda_ + this->nu())/sigmaNut_)
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::k() const
{
    WarningInFunction
        << "Turbulence kinetic energy not defined for "
        << "Spalart-Allmaras model. Returning zero field"
        << endl;

    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                "k",
                this->runTime_.timeName(),
                this->mesh_
            ),
            this->mesh_,
            dimensionedScalar("0", dimensionSet(0, 2, -2, 0, 0), 0),
            zeroGradientFvPatchField<vector>::typeName
        )
    );
}


template<class BasicTurbulenceModel>
tmp<volScalarField> SpalartAllmarasFv3<BasicTurbulenceModel>::epsilon() const
{
    WarningInFunction
        << "Turbulence kinetic energy dissipation rate not defined for "
        << "Spalart-Allmaras model. Returning zero field"
        << endl;

    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                "epsilon",
                this->runTime_.timeName(),
                this->mesh_
            ),
            this->mesh_,
            dimensionedScalar("0", dimensionSet(0, 2, -3, 0, 0), 0),
            zeroGradientFvPatchField<vector>::typeName
        )
    );
}

template<class BasicTurbulenceModel>
void SpalartAllmarasFv3<BasicTurbulenceModel>::boundVar(volScalarField& var)
{
    const scalar vGreat = 1e200;
    label useUpperBound = 0, useLowerBound=0;

    const dictionary& simple = this->mesh_.solutionDict().subDict("SIMPLE");

    scalar varMin = simple.lookupOrDefault<scalar>(var.name()+"LowerBound",-vGreat); 
    scalar varMax = simple.lookupOrDefault<scalar>(var.name()+"UpperBound",vGreat); 

    
    forAll(var,cellI)
    {
        if(var[cellI]<=varMin) 
        {
            var[cellI]=varMin;
            useLowerBound = 1;
        }
        if(var[cellI]>=varMax) 
        {
            var[cellI]=varMax;
            useUpperBound = 1;
        }
    }

    if (useUpperBound) Info<<"Bounding "<<var.name()<<"<"<<varMax<<endl;
    if (useLowerBound) Info<<"Bounding "<<var.name()<<">"<<varMin<<endl;

    return;
}


template<class BasicTurbulenceModel>
void SpalartAllmarasFv3<BasicTurbulenceModel>::correct()
{
    if (!this->turbulence_)
    {
        return;
    }

    // Local references
    const alphaField& alpha = this->alpha_;
    const rhoField& rho = this->rho_;
    const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    fv::options& fvOptions(fv::options::New(this->mesh_));

    eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda
    (
        fv3(chi, fv1)*::sqrt(2.0)*mag(skew(fvc::grad(this->U_)))
      + fv2(chi, fv1)*nuTilda_/sqr(kappa_*y_)
    );

    tmp<fvScalarMatrix> nuTildaEqn
    (
        fvm::ddt(alpha, rho, nuTilda_)
      + fvm::div(alphaRhoPhi, nuTilda_)
      - fvm::laplacian(alpha*rho*DnuTildaEff(), nuTilda_)
      - Cb2_/sigmaNut_*alpha*rho*magSqr(fvc::grad(nuTilda_))
      ==
        Cb1_*alpha*rho*Stilda*nuTilda_
      - fvm::Sp(Cw1_*alpha*rho*fw(Stilda)*nuTilda_/sqr(y_), nuTilda_)
      + fvOptions(alpha, rho, nuTilda_)
    );

    nuTildaEqn.ref().relax();
    fvOptions.constrain(nuTildaEqn.ref());
    solve(nuTildaEqn);
    fvOptions.correct(nuTilda_);
    bound(nuTilda_, dimensionedScalar("0", nuTilda_.dimensions(), nuTildaMin_.value()));
    boundVar(nuTilda_);
    nuTilda_.correctBoundaryConditions();
    
    correctNut(fv1);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace RASModels
} // End namespace Foam

// ************************************************************************* //
