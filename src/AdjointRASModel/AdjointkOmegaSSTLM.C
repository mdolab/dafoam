/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointkOmegaSSTLM.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointkOmegaSSTLM, 0);
addToRunTimeSelectionTable(AdjointRASModel, AdjointkOmegaSSTLM, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointkOmegaSSTLM::AdjointkOmegaSSTLM
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    nearWallDist& d,
#ifdef IncompressibleFlow
    const singlePhaseTransportModel& laminarTransport
#endif
#ifdef CompressibleFlow
    const fluidThermo& thermo
#endif
)
    :
#ifdef IncompressibleFlow
    AdjointRASModel(mesh,adjIO,d,laminarTransport),
#endif
#ifdef CompressibleFlow
    AdjointRASModel(mesh,adjIO,d,thermo),
#endif
    
    // SST parameters
    alphaK1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK1",
            this->coeffDict_,
            0.85
        )
    ),
    alphaK2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK2",
            this->coeffDict_,
            1.0
        )
    ),
    alphaOmega1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega1",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega2",
            this->coeffDict_,
            0.856
        )
    ),
    gamma1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma1",
            this->coeffDict_,
            5.0/9.0
        )
    ),
    gamma2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma2",
            this->coeffDict_,
            0.44
        )
    ),
    beta1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta1",
            this->coeffDict_,
            0.075
        )
    ),
    beta2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta2",
            this->coeffDict_,
            0.0828
        )
    ),
    betaStar_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    a1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "a1",
            this->coeffDict_,
            0.31
        )
    ),
    b1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "b1",
            this->coeffDict_,
            1.0
        )
    ),
    c1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "c1",
            this->coeffDict_,
            10.0
        )
    ),
    F3_
    (
        Switch::lookupOrAddToDict
        (
            "F3",
            this->coeffDict_,
            false
        )
    ),
    ca1_
    (
        dimensionedScalar::lookupOrAddToDict
        (
            "ca1",
            this->coeffDict_,
            2
        )
    ),
    ca2_
    (
        dimensionedScalar::lookupOrAddToDict
        (
            "ca2",
            this->coeffDict_,
            0.06
        )
    ),
    ce1_
    (
        dimensionedScalar::lookupOrAddToDict
        (
            "ce1",
            this->coeffDict_,
            1
        )
    ),
    ce2_
    (
        dimensionedScalar::lookupOrAddToDict
        (
            "ce2",
            this->coeffDict_,
            50
        )
    ),
    cThetat_
    (
        dimensionedScalar::lookupOrAddToDict
        (
            "cThetat",
            this->coeffDict_,
            0.03
        )
    ),
    sigmaThetat_
    (
        dimensionedScalar::lookupOrAddToDict
        (
            "sigmaThetat",
            this->coeffDict_,
            2
        )
    ),
    lambdaErr_
    (
        this->coeffDict_.lookupOrDefault("lambdaErr", 1e-6)
    ),
    maxLambdaIter_
    (
        this->coeffDict_.lookupOrDefault("maxLambdaIter", 10)
    ),
    deltaU_("deltaU", dimVelocity, SMALL),
    
    // Augmented variables
    omega_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("omega")
        )
    ),
    omegaRes_
    (
        IOobject
        (
            "omegaRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("omegaRes",dimensionSet(1,-3,-2,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("omegaRes",dimensionSet(0,0,-2,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    omegaResRef_
    (
        IOobject
        (
            "omegaResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        omegaRes_
    ),
    omegaResPartDeriv_
    (
        IOobject
        (
            "omegaResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        omegaRes_
    ),
    omegaRef_
    (
        IOobject
        (
            "omegaRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        omega_
    ), 
    k_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("k")
        )
    ),
    kRes_
    (
        IOobject
        (
            "kRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("kRes",dimensionSet(1,-1,-3,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("kRes",dimensionSet(0,2,-3,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    kResRef_
    (
        IOobject
        (
            "kResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        kRes_
    ),
    kResPartDeriv_
    (
        IOobject
        (
            "kResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        kRes_
    ),
    kRef_
    (
        IOobject
        (
            "kRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        k_
    ),
    ReThetat_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("ReThetat")
        )
    ),
    ReThetatRes_
    (
        IOobject
        (
            "ReThetatRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("ReThetatRes",dimensionSet(1,-3,-1,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("ReThetatRes",dimensionSet(0,0,-1,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    ReThetatResRef_
    (
        IOobject
        (
            "ReThetatResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        ReThetatRes_
    ),
    ReThetatResPartDeriv_
    (
        IOobject
        (
            "ReThetatResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        ReThetatRes_
    ),
    ReThetatRef_
    (
        IOobject
        (
            "ReThetatRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        ReThetat_
    ), 
    gammaInt_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("gammaInt")
        )
    ),
    gammaIntRes_
    (
        IOobject
        (
            "gammaIntRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("gammaIntRes",dimensionSet(1,-3,-1,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("gammaIntRes",dimensionSet(0,0,-1,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    gammaIntResRef_
    (
        IOobject
        (
            "gammaIntResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        gammaIntRes_
    ),
    gammaIntResPartDeriv_
    (
        IOobject
        (
            "gammaIntResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        gammaIntRes_
    ),
    gammaIntRef_
    (
        IOobject
        (
            "gammaIntRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        gammaInt_
    ),
    gammaIntEff_
    (
        const_cast<volScalarField::Internal&>
        (
            db_.lookupObject<volScalarField::Internal>("gammaIntEff")
        )
    ),
    y_(db_.lookupObject<volScalarField>("yWall"))
    
{
    // add the turbulence state variables, this will be used in AdjointIndexing
    this->turbStates.append("k"); 
    this->turbStates.append("omega");
    this->turbStates.append("ReThetat");
    this->turbStates.append("gammaInt");
    this->copyTurbStates("Var2Ref"); // copy turbVars to turbVarsRef
    //this->calcTurbResiduals(1,0); 
    //Info<<nuTildaResidualRef_<<endl;
    
    
    // calculate the size of omegaWallFunction faces
    label nWallFaces=0;
    forAll(omega_.boundaryField(),patchI)
    {
        if( omega_.boundaryField()[patchI].type() == "omegaWallFunction" 
            and omega_.boundaryField()[patchI].size()>0 )
        {
            forAll(omega_.boundaryField()[patchI],faceI)
            {
                nWallFaces++;
                //Info<<"patchI: "<<patchI<<" faceI: "<<faceI<<endl;
            }
        }
    }
    
    // initialize omegaNearWall
    omegaNearWall_.setSize(nWallFaces);
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// SST member functions
tmp<volScalarField> AdjointkOmegaSSTLM::F1SST
(
    const volScalarField& CDkOmega
) const
{

    tmp<volScalarField> CDkOmegaPlus = max
    (
        CDkOmega,
        dimensionedScalar("1.0e-10", dimless/sqr(dimTime), 1.0e-10)
    );

    tmp<volScalarField> arg1 = min
    (
        min
        (
            max
            (
                (scalar(1)/betaStar_)*sqrt(k_)/(omega_*y_),
                scalar(500)*(this->getNu())/(sqr(y_)*omega_)
            ),
            (scalar(4)*alphaOmega2_)*k_/(CDkOmegaPlus*sqr(y_))
        ),
        scalar(10)
    );

    return tanh(pow4(arg1));
}


tmp<volScalarField> AdjointkOmegaSSTLM::F2() const
{
    
    tmp<volScalarField> arg2 = min
    (
        max
        (
            (scalar(2)/betaStar_)*sqrt(k_)/(omega_*y_),
            scalar(500)*(this->getNu())/(sqr(y_)*omega_)
        ),
        scalar(100)
    );

    return tanh(sqr(arg2));
}



tmp<volScalarField> AdjointkOmegaSSTLM::F3() const
{
     
    tmp<volScalarField> arg3 = min
    (
        150*(this->getNu())/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}

tmp<volScalarField> AdjointkOmegaSSTLM::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


tmp<volScalarField::Internal> AdjointkOmegaSSTLM::GbyNu
(
    const volScalarField::Internal& GbyNu0,
    const volScalarField::Internal& F2,
    const volScalarField::Internal& S2
) const
{
    return min
    (
        GbyNu0,
        (c1_/a1_)*betaStar_*omega_()
       *max(a1_*omega_(), b1_*F2*sqrt(S2))
    );
}


tmp<volScalarField::Internal> AdjointkOmegaSSTLM::PkSST
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*k_()*omega_());
}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::epsilonBykSST
(
    const volScalarField& F1,
    const volTensorField& gradU
) const
{
    return betaStar_*omega_();
}


tmp<fvScalarMatrix> AdjointkOmegaSSTLM::kSource() const
{
    const volScalarField& rho = rho_;
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            k_,
            dimVolume*rho.dimensions()*k_.dimensions()/dimTime
        )
    );
}


tmp<fvScalarMatrix> AdjointkOmegaSSTLM::omegaSource() const
{
    const volScalarField& rho = rho_;
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*rho.dimensions()*omega_.dimensions()/dimTime
        )
    );
}

tmp<fvScalarMatrix> AdjointkOmegaSSTLM::Qsas
(
    const volScalarField::Internal& S2,
    const volScalarField::Internal& gamma,
    const volScalarField::Internal& beta
) const
{
    const volScalarField& rho = rho_;
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            omega_,
            dimVolume*rho.dimensions()*omega_.dimensions()/dimTime
        )
    );
}

// SSTLM functions
tmp<volScalarField> AdjointkOmegaSSTLM::F1
(
    const volScalarField& CDkOmega
) const
{
    const volScalarField Ry(y_*sqrt(k_)/this->getNu());
    const volScalarField F3(exp(-pow(Ry/120.0, 8)));

    return max(this->F1SST(CDkOmega), F3);
}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::Pk
(
    const volScalarField::Internal& G
) const
{
    return gammaIntEff_*this->PkSST(G);
}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::epsilonByk
(
    const volScalarField& F1,
    const volTensorField& gradU
) const
{
    return
        min(max(gammaIntEff_, scalar(0.1)), scalar(1))
       *this->epsilonBykSST(F1, gradU);
}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::Fthetat
(
    const volScalarField::Internal& Us,
    const volScalarField::Internal& Omega,
    const volScalarField::Internal& nu
) const
{
    const volScalarField::Internal& omega = omega_();
    const volScalarField::Internal& y = y_();

    const volScalarField::Internal delta(375*Omega*nu*ReThetat_()*y/sqr(Us));
    const volScalarField::Internal ReOmega(sqr(y)*omega/nu);
    const volScalarField::Internal Fwake(exp(-sqr(ReOmega/1e5)));
    
    return tmp<volScalarField::Internal>
    (
        new volScalarField::Internal
        (
            IOobject::groupName("Fthetat", U_.group()),
            min
            (
                max
                (
                    Fwake*exp(-pow4((y/delta))),
                    (1 - sqr((gammaInt_() - 1.0/ce2_)/(1 - 1.0/ce2_)))
                ),
                scalar(1)
            )
        )
    );

}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::ReThetac() const
{

    tmp<volScalarField::Internal> tReThetac
    (
        new volScalarField::Internal
        (
            IOobject
            (
                IOobject::groupName("ReThetac", U_.group()),
                mesh_.time().timeName(),
                mesh_
            ),
            mesh_,
            dimless
        )
    );
    volScalarField::Internal& ReThetac = tReThetac.ref();

    forAll(ReThetac, celli)
    {
        const scalar ReThetat = ReThetat_[celli];

        ReThetac[celli] =
            ReThetat <= 1870
          ?
            ReThetat
          - 396.035e-2
          + 120.656e-4*ReThetat
          - 868.230e-6*sqr(ReThetat)
          + 696.506e-9*pow3(ReThetat)
          - 174.105e-12*pow4(ReThetat)
          :
            ReThetat - 593.11 - 0.482*(ReThetat - 1870);
    }

    return tReThetac;
}


tmp<volScalarField::Internal> AdjointkOmegaSSTLM::Flength
(
    const volScalarField::Internal& nu
) const
{

    tmp<volScalarField::Internal> tFlength
    (
        new volScalarField::Internal
        (
            IOobject
            (
                IOobject::groupName("Flength", U_.group()),
                mesh_.time().timeName(),
                mesh_
            ),
            mesh_,
            dimless
        )
    );
    volScalarField::Internal& Flength = tFlength.ref();

    const volScalarField::Internal& omega = omega_();
    const volScalarField::Internal& y = y_();

    forAll(ReThetat_, celli)
    {
        const scalar ReThetat = ReThetat_[celli];

        if (ReThetat < 400)
        {
            Flength[celli] =
                398.189e-1
              - 119.270e-4*ReThetat
              - 132.567e-6*sqr(ReThetat);
        }
        else if (ReThetat < 596)
        {
            Flength[celli] =
                263.404
              - 123.939e-2*ReThetat
              + 194.548e-5*sqr(ReThetat)
              - 101.695e-8*pow3(ReThetat);
        }
        else if (ReThetat < 1200)
        {
            Flength[celli] = 0.5 - 3e-4*(ReThetat - 596);
        }
        else
        {
            Flength[celli] = 0.3188;
        }

        const scalar Fsublayer =
            exp(-sqr(sqr(y[celli])*omega[celli]/(200*nu[celli])));

        Flength[celli] = Flength[celli]*(1 - Fsublayer) + 40*Fsublayer;
    }

    return tFlength;
}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::Fonset
(
    const volScalarField::Internal& Rev,
    const volScalarField::Internal& ReThetac,
    const volScalarField::Internal& RT
) const
{
    const volScalarField::Internal Fonset1(Rev/(2.193*ReThetac));

    const volScalarField::Internal Fonset2
    (
        min(max(Fonset1, pow4(Fonset1)), scalar(2))
    );

    const volScalarField::Internal Fonset3(max(1 - pow3(RT/2.5), scalar(0)));

    return tmp<volScalarField::Internal>
    (
        new volScalarField::Internal
        (
            IOobject::groupName("Fonset", U_.group()),
            max(Fonset2 - Fonset3, scalar(0))
        )
    );
}

tmp<volScalarField::Internal> AdjointkOmegaSSTLM::ReThetat0
(
    const volScalarField::Internal& Us,
    const volScalarField::Internal& dUsds,
    const volScalarField::Internal& nu
) const
{

    tmp<volScalarField::Internal> tReThetat0
    (
        new volScalarField::Internal
        (
            IOobject
            (
                IOobject::groupName("ReThetat0", U_.group()),
                mesh_.time().timeName(),
                mesh_
            ),
            mesh_,
            dimless
        )
    );
    volScalarField::Internal& ReThetat0 = tReThetat0.ref();

    const volScalarField& k = k_;

    label maxIter = 0;

    forAll(ReThetat0, celli)
    {
        const scalar Tu
        (
            max(100*sqrt((2.0/3.0)*k[celli])/Us[celli], scalar(0.027))
        );

        // Initialize lambda to zero.
        // If lambda were cached between time-steps convergence would be faster
        // starting from the previous time-step value.
        scalar lambda = 0;

        scalar lambdaErr;
        scalar thetat;
        label iter = 0;

        do
        {
            // Previous iteration lambda for convergence test
            const scalar lambda0 = lambda;

            if (Tu <= 1.3)
            {
                const scalar Flambda =
                    dUsds[celli] <= 0
                  ?
                    1
                  - (
                     - 12.986*lambda
                     - 123.66*sqr(lambda)
                     - 405.689*pow3(lambda)
                    )*exp(-pow(Tu/1.5, 1.5))
                  :
                    1
                  + 0.275*(1 - exp(-35*lambda))
                   *exp(-Tu/0.5);

                thetat =
                    (1173.51 - 589.428*Tu + 0.2196/sqr(Tu))
                   *Flambda*nu[celli]
                   /Us[celli];
            }
            else
            {
                const scalar Flambda =
                    dUsds[celli] <= 0
                  ?
                    1
                  - (
                      -12.986*lambda
                      -123.66*sqr(lambda)
                      -405.689*pow3(lambda)
                    )*exp(-pow(Tu/1.5, 1.5))
                  :
                    1
                  + 0.275*(1 - exp(-35*lambda))
                   *exp(-2*Tu);

                thetat =
                    331.50*pow((Tu - 0.5658), -0.671)
                   *Flambda*nu[celli]/Us[celli];
            }

            lambda = sqr(thetat)/nu[celli]*dUsds[celli];
            lambda = max(min(lambda, 0.1), -0.1);

            lambdaErr = mag(lambda - lambda0);

            maxIter = max(maxIter, ++iter);

        } while (lambdaErr > lambdaErr_);

        ReThetat0[celli] = max(thetat*Us[celli]/nu[celli], scalar(20));
    }

    if (maxIter > maxLambdaIter_)
    {
        WarningInFunction
            << "Number of lambda iterations exceeds maxLambdaIter("
            << maxLambdaIter_ << ')'<< endl;
    }

    return tReThetat0;
}

// Augmented functions

void AdjointkOmegaSSTLM::updateNut()
{
    const volVectorField U=db_.lookupObject<volVectorField>("U");
    tmp<volTensorField> tgradU = fvc::grad(U);
    volScalarField S2(2*magSqr(symm(tgradU())));

    nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));
    
    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k

    // for SSTLM also update gammaIntEff
    const tmp<volScalarField> tnu = this->getNu();
    const volScalarField::Internal& nu = tnu()();
    const volScalarField::Internal& y = y_();
    tmp<volTensorField> tgradULM = fvc::grad(U_);
    volScalarField::Internal Omega(sqrt(2*magSqr(skew(tgradULM()()))));
    forAll(Omega,idxI) 
    {
        if(fabs(Omega[idxI])<1e-15)
        {
            Omega[idxI]=1e-15; // to make sure Omega is not zero at the initial step
        }
    }
    const volScalarField::Internal S(sqrt(2*magSqr(symm(tgradULM()()))));
    const volScalarField::Internal Us(max(mag(U_()), deltaU_));
    const volScalarField::Internal dUsds((U_() & (U_() & tgradULM()()))/sqr(Us));
    tgradULM.clear();
    const volScalarField::Internal ReThetac(this->ReThetac());
    const volScalarField::Internal Rev(sqr(y)*S/nu);
    const volScalarField::Internal RT(k_()/(nu*omega_()));
    const volScalarField::Internal Fthetat(this->Fthetat(Us, Omega, nu));
    const volScalarField::Internal Freattach(exp(-pow4(RT/20.0)));
    const volScalarField::Internal gammaSep
    (
        min(2*max(Rev/(3.235*ReThetac) - 1, scalar(0))*Freattach, scalar(2))
       *Fthetat
    );

    gammaIntEff_ = max(gammaInt_(), gammaSep);
    
    return;
}


void AdjointkOmegaSSTLM::copyTurbStates(const word option)
{
    // "Ref2Var", assign ref states to states
    // "Var2Ref", assign states to ref states
    if(option=="Ref2Var")
    {
        omega_=omegaRef_;
        k_=kRef_;
        ReThetat_=ReThetatRef_;
        gammaInt_=gammaIntRef_;
        this->correctTurbBoundaryConditions();
    }
    else if (option=="Var2Ref")
    {
        omegaRef_ = omega_;
        kRef_ = k_;
        ReThetatRef_ = ReThetat_;
        gammaIntRef_ = gammaInt_;
    }
    else
    {
        FatalErrorIn("option not valid! Should be either Var2Ref or Ref2Var")
        << abort(FatalError);
    }

    this->updateNut();
    
    return;
    
}


void AdjointkOmegaSSTLM::correctTurbBoundaryConditions()
{

    k_.correctBoundaryConditions(); // kqWallFunction is a zero-gradient BC
    //this->correctOmegaBoundaryConditions(); // special treatment for omega

    // Note: we need to update nut since we may have perturbed other turbulence vars
    // that affect the nut values
    this->updateNut(); 
    
    ReThetat_.correctBoundaryConditions(); 
    gammaInt_.correctBoundaryConditions(); 

    return;
}

void AdjointkOmegaSSTLM::correctOmegaBoundaryConditions()
{
    // this is a special treatment for omega BC
    // we cant directly call omega.correctBoundaryConditions() because it will
    // modify the internal omega and G that are right next to walls. This will mess up adjoint Jacobians
    // To solve this issue, 
    // 1. we store the near wall omega before calling omega.correctBoundaryConditions()
    // 2. call omega.correctBoundaryConditions()
    // 3. Assign the stored near wall omega back
    // 4. Apply a zeroGradient BC for omega at the wall patches
    // *********** NOTE *************
    // this treatment will obviously downgrade the accuracy of adjoint derivative since it is not 100% consistent
    // with what is used for the flow solver; however, our observation shows that the impact is not very large. 

    // save the perturbed omega at the wall
    this->saveOmegaNearWall();
    // correct omega boundary conditions, this includes updating wall face and near wall omega values,
    // updating the inter-proc BCs
    omega_.correctBoundaryConditions(); 
    // reset the corrected omega near wall cell to its perturbed value
    this->setOmegaNearWall();

}


void AdjointkOmegaSSTLM::calcTurbResiduals
(  
    const label isRef,
    const label isPC,
    const word fvMatrixName
)
{    

    // Copied and modified based on the "correct" function

    // ************ LM part ********
    {
        word divGammaIntScheme="div(phi,gammaInt)";
        word divReThetatScheme="div(phi,ReThetat)";
        if(isPC) 
        {
            divGammaIntScheme="div(pc)";
            divReThetatScheme="div(pc)";
        }
    
        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        this->boundTurbVars(ReThetat_, 0.0);
    
        // Local references
        const tmp<volScalarField> tnu = this->getNu();
        const volScalarField::Internal& nu = tnu()();
        const volScalarField::Internal& y = y_();
    
        // Fields derived from the velocity gradient
        tmp<volTensorField> tgradULM = fvc::grad(U_);
        const volScalarField::Internal Omega(sqrt(2*magSqr(skew(tgradULM()()))));
        const volScalarField::Internal S(sqrt(2*magSqr(symm(tgradULM()()))));
        const volScalarField::Internal Us(max(mag(U_()), deltaU_));
        const volScalarField::Internal dUsds((U_() & (U_() & tgradULM()()))/sqr(Us));
        tgradULM.clear();
    
        const volScalarField::Internal Fthetat(this->Fthetat(Us, Omega, nu));
    
        {
            const volScalarField::Internal t(500*nu/sqr(Us));
            const volScalarField::Internal Pthetat
            (
                phase_()*rho_()*(cThetat_/t)*(1 - Fthetat)
            );
    
            // Transition onset momentum-thickness Reynolds number equation
            tmp<fvScalarMatrix> ReThetatEqn
            (
                fvm::ddt(phase_, rho_, ReThetat_)
              + fvm::div(phaseRhoPhi_, ReThetat_, divReThetatScheme)
              - fvm::laplacian(phase_*rho_*DReThetatEff(), ReThetat_)
             ==
                Pthetat*ReThetat0(Us, dUsds, nu) - fvm::Sp(Pthetat, ReThetat_)
            );
    
            ReThetatEqn.ref().relax();
    
            if(fvMatrixName=="ReThetatEqn") 
            {
                fvMatrixDiag.clear();
                fvMatrixLower.clear();
                fvMatrixUpper.clear();
                fvMatrixDiag = ReThetatEqn.ref().diag();
                fvMatrixLower = ReThetatEqn.ref().lower();
                fvMatrixUpper = ReThetatEqn.ref().upper();
            }
            
            // calculate residuals
            if(isRef)
            {
                ReThetatResRef_ = ReThetatEqn() & ReThetat_;
                // need to normalize Res. NOTE: ReThetatRes is normalized by its cell volume by default!
                if(!adjIO_.isInList<word>("ReThetatRes",adjIO_.normalizeResiduals)) 
                {
                    forAll(ReThetatResRef_,cellI)
                    {
                        ReThetatResRef_[cellI] *= mesh_.V()[cellI];  
                    }
                }
                if(adjIO_.isInList<word>("ReThetatResScaling",adjIO_.residualScaling.toc())) 
                {
                    forAll(ReThetatResRef_,cellI)
                    {
                        ReThetatResRef_[cellI] /= adjIO_.residualScaling["ReThetatResScaling"];  
                    }
                }
            }
            else
            {
                ReThetatRes_ = ReThetatEqn() & ReThetat_;
                // need to normalize Res. NOTE: ReThetatRes is normalized by its cell volume by default!
                if(!adjIO_.isInList<word>("ReThetatRes",adjIO_.normalizeResiduals)) 
                {
                    forAll(ReThetatRes_,cellI)
                    {
                        ReThetatRes_[cellI] *= mesh_.V()[cellI];  
                    }
                }
                if(adjIO_.isInList<word>("ReThetatResScaling",adjIO_.residualScaling.toc())) 
                {
                    forAll(ReThetatRes_,cellI)
                    {
                        ReThetatRes_[cellI] /= adjIO_.residualScaling["ReThetatResScaling"];  
                    }
                }
            }
        }
    
        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        this->boundTurbVars(gammaInt_, 0.0);
    
        const volScalarField::Internal ReThetac(this->ReThetac());
        const volScalarField::Internal Rev(sqr(y)*S/nu);
        const volScalarField::Internal RT(k_()/(nu*omega_()));
    
        {
            const volScalarField::Internal Pgamma
            (
                phase_()*rho_()
               *ca1_*Flength(nu)*S*sqrt(gammaInt_()*Fonset(Rev, ReThetac, RT))
            );
    
            const volScalarField::Internal Fturb(exp(-pow4(0.25*RT)));
    
            const volScalarField::Internal Egamma
            (
                phase_()*rho_()*ca2_*Omega*Fturb*gammaInt_()
            );
    
            // Intermittency equation
            tmp<fvScalarMatrix> gammaIntEqn
            (
                fvm::ddt(phase_, rho_, gammaInt_)
              + fvm::div(phaseRhoPhi_, gammaInt_, divGammaIntScheme)
              - fvm::laplacian(phase_*rho_*DgammaIntEff(), gammaInt_)
            ==
                Pgamma - fvm::Sp(ce1_*Pgamma, gammaInt_)
              + Egamma - fvm::Sp(ce2_*Egamma, gammaInt_)
            );
    
            gammaIntEqn.ref().relax();
            
            if(fvMatrixName=="gammaIntEqn") 
            {
                fvMatrixDiag.clear();
                fvMatrixLower.clear();
                fvMatrixUpper.clear();
                fvMatrixDiag = gammaIntEqn.ref().diag();
                fvMatrixLower = gammaIntEqn.ref().lower();
                fvMatrixUpper = gammaIntEqn.ref().upper();
            }
            
            // calculate residuals
            if(isRef)
            {
                gammaIntResRef_ = gammaIntEqn() & gammaInt_;
                // need to normalize Res. NOTE: gammaIntRes is normalized by its cell volume by default!
                if(!adjIO_.isInList<word>("gammaIntRes",adjIO_.normalizeResiduals)) 
                {
                    forAll(gammaIntResRef_,cellI)
                    {
                        gammaIntResRef_[cellI] *= mesh_.V()[cellI];  
                    }
                }
                if(adjIO_.isInList<word>("gammaIntResScaling",adjIO_.residualScaling.toc())) 
                {
                    forAll(gammaIntResRef_,cellI)
                    {
                        gammaIntResRef_[cellI] /= adjIO_.residualScaling["gammaIntResScaling"];  
                    }
                }
            }
            else
            {
                gammaIntRes_ = gammaIntEqn() & gammaInt_;
                // need to normalize Res. NOTE: gammaIntRes is normalized by its cell volume by default!
                if(!adjIO_.isInList<word>("gammaIntRes",adjIO_.normalizeResiduals)) 
                {
                    forAll(gammaIntRes_,cellI)
                    {
                        gammaIntRes_[cellI] *= mesh_.V()[cellI];  
                    }
                }
                if(adjIO_.isInList<word>("gammaIntResScaling",adjIO_.residualScaling.toc())) 
                {
                    forAll(gammaIntRes_,cellI)
                    {
                        gammaIntRes_[cellI] /= adjIO_.residualScaling["gammaIntResScaling"];  
                    }
                }
            }
    
        }
    }

    // ************ SST part ********
    {
        word divKScheme="div(phi,k)";
        word divOmegaScheme="div(phi,omega)";
        if(isPC) 
        {
            divKScheme="div(pc)";
            divOmegaScheme="div(pc)";
        }
    
        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        this->boundTurbVars(omega_, omegaMin_);
        
        volScalarField& phase = phase_;
        volScalarField& rho = rho_;
        surfaceScalarField& phaseRhoPhi = phaseRhoPhi_;
    
        //BasicEddyViscosityModel::correct();
        
        // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho), 
        // while for the incompresssible "this->phi()" returns phi only
        // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
        volScalarField::Internal divU(fvc::div(fvc::absolute(phi_/fvc::interpolate(rho_), U_)));
    
        tmp<volTensorField> tgradU = fvc::grad(U_);
        volScalarField S2(2*magSqr(symm(tgradU())));
        volScalarField::Internal GbyNu0((tgradU() && dev(twoSymm(tgradU()))));
        volScalarField::Internal G("kOmegaSSTLM:G", nut_*GbyNu0);
    
        // Update omega and G at the wall. NOTE: this will not update the inter-proc face values  
        //omega_.boundaryFieldRef().updateCoeffs();
        
        this->correctOmegaBoundaryConditions();
    
        volScalarField CDkOmega
        (
            (scalar(2)*alphaOmega2_)*(fvc::grad(k_) & fvc::grad(omega_))/omega_
        );
    
        volScalarField F1(this->F1(CDkOmega));
        volScalarField F23(this->F23());
    
        scalar prodSwitch=1.0;
        if(adjIO_.delTurbProd4PCMat and isPC) prodSwitch=0.0;
    
        {
            volScalarField::Internal gamma(this->gamma(F1));
            volScalarField::Internal beta(this->beta(F1));
    
            // Turbulent frequency equation
            tmp<fvScalarMatrix> omegaEqn
            (
                fvm::ddt(phase, rho, omega_)
              + fvm::div(phaseRhoPhi, omega_, divOmegaScheme)
              - fvm::laplacian(phase*rho*DomegaEff(F1), omega_)
             ==
                prodSwitch*phase()*rho()*gamma*GbyNu(GbyNu0, F23(), S2())
              - fvm::SuSp((2.0/3.0)*phase()*rho()*gamma*divU, omega_)
              - fvm::Sp(phase()*rho()*beta*omega_(), omega_)
              - fvm::SuSp
                (
                    phase()*rho()*(F1() - scalar(1))*CDkOmega()/omega_(),
                    omega_
                )
                + Qsas(S2(), gamma, beta)
                + omegaSource()
              //+ fvOptions(phase, rho, omega_)
    
            );
    
            omegaEqn.ref().relax();
            //fvOptions.constrain(omegaEqn.ref());
            omegaEqn.ref().boundaryManipulate(omega_.boundaryFieldRef());
            
            // reset the corrected omega near wall cell to its perturbed value
            this->setOmegaNearWall();
            
            //solve(omegaEqn);
            //fvOptions.correct(omega_);
            //bound(omega_, omegaMin_);
    
            if(fvMatrixName=="omegaEqn") 
            {
                fvMatrixDiag.clear();
                fvMatrixLower.clear();
                fvMatrixUpper.clear();
                fvMatrixDiag = omegaEqn.ref().diag();
                fvMatrixLower = omegaEqn.ref().lower();
                fvMatrixUpper = omegaEqn.ref().upper();
            }
            
            // calculate residuals
            if(isRef)
            {
                omegaResRef_ = omegaEqn() & omega_;
                // need to normalize Res. NOTE: omegaRes is normalized by its cell volume by default!
                if(!adjIO_.isInList<word>("omegaRes",adjIO_.normalizeResiduals)) 
                {
                    forAll(omegaResRef_,cellI)
                    {
                        omegaResRef_[cellI] *= mesh_.V()[cellI];  
                    }
                }
                if(adjIO_.isInList<word>("omegaResScaling",adjIO_.residualScaling.toc())) 
                {
                    forAll(omegaResRef_,cellI)
                    {
                        omegaResRef_[cellI] /= adjIO_.residualScaling["omegaResScaling"];  
                    }
                }
            }
            else
            {
                omegaRes_ = omegaEqn() & omega_;
                // need to normalize Res. NOTE: omegaRes is normalized by its cell volume by default!
                if(!adjIO_.isInList<word>("omegaRes",adjIO_.normalizeResiduals)) 
                {
                    forAll(omegaRes_,cellI)
                    {
                        omegaRes_[cellI] *= mesh_.V()[cellI];  
                    }
                }
                if(adjIO_.isInList<word>("omegaResScaling",adjIO_.residualScaling.toc())) 
                {
                    forAll(omegaRes_,cellI)
                    {
                        omegaRes_[cellI] /= adjIO_.residualScaling["omegaResScaling"];  
                    }
                }
            }
    
        }
    
        // we need to bound before computing residuals
        // this will avoid having NaN residuals
        this->boundTurbVars(k_, kMin_);
    
        // Turbulent kinetic energy equation
        tmp<fvScalarMatrix> kEqn
        (
            fvm::ddt(phase, rho, k_)
          + fvm::div(phaseRhoPhi, k_, divKScheme)
          - fvm::laplacian(phase*rho*DkEff(F1), k_)
         ==
            prodSwitch*phase()*rho()*Pk(G)
          - fvm::SuSp((2.0/3.0)*phase()*rho()*divU, k_)
          - fvm::Sp(phase()*rho()*epsilonByk(F1, tgradU()), k_)
          + kSource()
          //+ fvOptions(phase, rho, k_)
        );
    
        tgradU.clear();
    
        kEqn.ref().relax();
        //fvOptions.constrain(kEqn.ref());
        //solve(kEqn);
        //fvOptions.correct(k_);
        //bound(k_, kMin_);
    
        //correctNut(S2);
    
        if(fvMatrixName=="kEqn") 
        {
            fvMatrixDiag.clear();
            fvMatrixLower.clear();
            fvMatrixUpper.clear();
            fvMatrixDiag = kEqn.ref().diag();
            fvMatrixLower = kEqn.ref().lower();
            fvMatrixUpper = kEqn.ref().upper();
        }
        
        // calculate residuals
        if(isRef)
        {
            kResRef_ = kEqn() & k_;
            // need to normalize Res. NOTE: kRes is normalized by its cell volume by default!
            if(!adjIO_.isInList<word>("kRes",adjIO_.normalizeResiduals)) 
            {
                forAll(kResRef_,cellI)
                {
                    kResRef_[cellI] *= mesh_.V()[cellI];  
                }
            }
            if(adjIO_.isInList<word>("kResScaling",adjIO_.residualScaling.toc())) 
            {
                forAll(kResRef_,cellI)
                {
                    kResRef_[cellI] /= adjIO_.residualScaling["kResScaling"];  
                }
            }
        }
        else
        {
            kRes_ = kEqn() & k_;
            // need to normalize Res. NOTE: kRes is normalized by its cell volume by default!
            if(!adjIO_.isInList<word>("kRes",adjIO_.normalizeResiduals)) 
            {
                forAll(kRes_,cellI)
                {
                    kRes_[cellI] *= mesh_.V()[cellI];  
                }
            }
            if(adjIO_.isInList<word>("kResScaling",adjIO_.residualScaling.toc())) 
            {
                forAll(kRes_,cellI)
                {
                    kRes_[cellI] /= adjIO_.residualScaling["kResScaling"];  
                }
            }
        }
    }
    
    return;
}

void AdjointkOmegaSSTLM::saveOmegaNearWall()
{
    // save the current near wall omega values to omegaNearWall_
    label counterI=0;
    forAll(omega_.boundaryField(),patchI)
    {
        if( omega_.boundaryField()[patchI].type() == "omegaWallFunction" 
            and omega_.boundaryField()[patchI].size()>0 )
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells,faceI)
            {
                //Info<<"Near Wall cellI: "<<faceCells[faceI]<<endl;
                omegaNearWall_[counterI] = omega_[faceCells[faceI]];
                counterI++;
            }
        }
    }
    return;
}

void AdjointkOmegaSSTLM::setOmegaNearWall()
{
    // set the current near wall omega values from omegaNearWall_
    // here we also apply a zeroGradient BC to the wall faces
    label counterI=0;
    forAll(omega_.boundaryField(),patchI)
    {
        if( omega_.boundaryField()[patchI].type() == "omegaWallFunction" 
            and omega_.boundaryField()[patchI].size()>0 )
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells,faceI)
            {
                omega_[faceCells[faceI]] = omegaNearWall_[counterI];
                omega_.boundaryFieldRef()[patchI][faceI] = omega_[faceCells[faceI]]; // zeroGradient BC
                counterI++;
            }
        }
    }
    return;
}


void AdjointkOmegaSSTLM::correctAdjStateResidualTurbCon
(
    List< List<word> >& adjStateResidualConInfo
)
{

    // For SST model, we need to replace nut with k, omega, and U
    // also we need to make sure we have one more level of U connectivity
    // since nut is based on k, omega, and grad(U)
    forAll(adjStateResidualConInfo,idxI) // idxI is the con level
    {
        label addUCon=0;
        forAll(adjStateResidualConInfo[idxI],idxJ)
        {
            word conStateName = adjStateResidualConInfo[idxI][idxJ];
            if( conStateName == "nut" ) 
            {
                adjStateResidualConInfo[idxI][idxJ]="omega"; // replace nut with omega
                adjStateResidualConInfo[idxI].append("k");  // also add k for that level
                addUCon=1;
            }
        }
        
        // add U for the current level and level+1 if it is not there yet
        label isU;
        if(addUCon==1)
        {
            // first add U for the current level
            isU=0;
            forAll(adjStateResidualConInfo[idxI],idxJ)
            {
                word conStateName = adjStateResidualConInfo[idxI][idxJ];
                if( conStateName == "U" ) isU=1;
            }
            if(!isU)
            {
                adjStateResidualConInfo[idxI].append("U"); 
            }
            
            // now add U for level+1
            isU=0;
            forAll(adjStateResidualConInfo[idxI+1],idxJ)
            {
                word conStateName = adjStateResidualConInfo[idxI+1][idxJ];
                if( conStateName == "U" ) isU=1;
            }
            if(!isU)
            {
                adjStateResidualConInfo[idxI+1].append("U"); 
            }
        }
    }
    
    return; 

}

void AdjointkOmegaSSTLM::setAdjStateResidualTurbCon
(
    HashTable< List< List<word> > >& adjStateResidualConInfo
)
{

    word pName=adjIO_.getPName();

#ifdef IncompressibleFlow
    adjStateResidualConInfo.set
    (
        "omegaRes",
        {
            {"U","omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","omega","k","ReThetat","gammaInt"},       // lv1
            {"U","omega","k","ReThetat","gammaInt"}        // lv2
        }
    );
    
    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","omega","k","ReThetat","gammaInt"},       // lv1
            {"U","omega","k","ReThetat","gammaInt"}        // lv2
        }
    );

    adjStateResidualConInfo.set
    (
        "ReThetatRes",
        {
            {"U","omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","omega","k","ReThetat","gammaInt"},       // lv1
            {"U","omega","k","ReThetat","gammaInt"}        // lv2
        }
    );

    adjStateResidualConInfo.set
    (
        "gammaIntRes",
        {
            {"U","omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","omega","k","ReThetat","gammaInt"},       // lv1
            {"U","omega","k","ReThetat","gammaInt"}        // lv2
        }
    );
#endif 

#ifdef CompressibleFlow
    adjStateResidualConInfo.set
    (
        "omegaRes",
        {
            {"U","T",pName,"omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","T",pName,"omega","k","ReThetat","gammaInt"},       // lv1
            {"U","T",pName,"omega","k","ReThetat","gammaInt"}        // lv2
        }
    );
    
    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","T",pName,"omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","T",pName,"omega","k","ReThetat","gammaInt"},       // lv1
            {"U","T",pName,"omega","k","ReThetat","gammaInt"}        // lv2
        }
    );

    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","T",pName,"omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","T",pName,"omega","k","ReThetat","gammaInt"},       // lv1
            {"U","T",pName,"omega","k","ReThetat","gammaInt"}        // lv2
        }
    );

    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","T",pName,"omega","k","ReThetat","gammaInt","phi"}, // lv0
            {"U","T",pName,"omega","k","ReThetat","gammaInt"},       // lv1
            {"U","T",pName,"omega","k","ReThetat","gammaInt"}        // lv2
        }
    );
#endif
}

void AdjointkOmegaSSTLM::clearTurbVars()
{
    // clear any variables that are not needed for solveAdjoint
    //kRef_.clear();
    kRes_.clear();
    kResRef_.clear();
    kResPartDeriv_.clear();

    //omegaRef_.clear();
    omegaRes_.clear();
    omegaResRef_.clear();
    omegaResPartDeriv_.clear();

    ReThetatRes_.clear();
    ReThetatResRef_.clear();
    ReThetatResPartDeriv_.clear();

    gammaIntRes_.clear();
    gammaIntResRef_.clear();
    gammaIntResPartDeriv_.clear();
}

void AdjointkOmegaSSTLM::writeTurbStates()
{
    nut_.write();
    k_.write();
    omega_.write();
    ReThetat_.write();
    gammaInt_.write();
}

} // End namespace Foam

// ************************************************************************* //
