/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointRASModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(AdjointRASModel, 0);
defineRunTimeSelectionTable(AdjointRASModel, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointRASModel::AdjointRASModel
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
#ifdef PhaseIncompressibleFlow
    const immiscibleIncompressibleTwoPhaseMixture& mixture
#endif
)
    :
    mesh_(mesh),
    adjIO_(adjIO),
    d_(d),
    db_(mesh.thisDb()),
    nut_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("nut")
        )
    ),
    U_
    (
        const_cast<volVectorField&>
        (
            db_.lookupObject<volVectorField>("U")
        )
    ),
    phi_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("phi")
        )
    ),
#ifdef IncompressibleFlow 
    laminarTransport_(laminarTransport),
    phase_
    (
        IOobject
        (
            "phase",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh,
        dimensionedScalar("phase",dimensionSet(0,0,0,0,0,0,0),1.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    rho_
    (
        IOobject
        (
            "rho",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh,
        dimensionedScalar("rho",dimensionSet(0,0,0,0,0,0,0),1.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    phaseRhoPhi_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("phi")
        )
    ),
#endif
#ifdef CompressibleFlow
    thermo_(thermo),
    phase_
    (
        IOobject
        (
            "phase",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh,
        dimensionedScalar("phase",dimensionSet(0,0,0,0,0,0,0),1.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    rho_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("rho")
        )
    ),
    phaseRhoPhi_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("phi")
        )
    ),
#endif
#ifdef PhaseIncompressibleFlow
    mixture_(mixture),
    phase_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("alpha")
        )
    ),
    rho_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("rho")
        )
    ),
    phaseRhoPhi_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("phi") // note: this should be something else?
        )
    ),
#endif
    turbDict_
    (
        IOobject
        (
            "turbulenceProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false
        )
    ),
    coeffDict_(turbDict_.subDict("RAS")),
    kMin_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "kMin",
            coeffDict_,
            sqr(dimVelocity),
            SMALL
        )
    ),
    epsilonMin_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "epsilonMin",
            coeffDict_,
            kMin_.dimensions()/dimTime,
            SMALL
        )
    ),
    omegaMin_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "omegaMin",
            coeffDict_,
            dimless/dimTime,
            SMALL
        )
    ),
    nuTildaMin_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "nuTildaMin",
            coeffDict_,
            nut_.dimensions(),
            SMALL
        )
    )
    
{  
    Info<<"AdjointRASDict:"<<coeffDict_<<endl;
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<AdjointRASModel> AdjointRASModel::New
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
#ifdef PhaseIncompressibleFlow
    const immiscibleIncompressibleTwoPhaseMixture& mixture
#endif
)
{
    // get model name, but do not register the dictionary
    // otherwise it is registered in the database twice
    const word modelType
    (
        IOdictionary
        (
            IOobject
            (
                "turbulenceProperties",
                mesh.time().constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false
            )
        ).subDict("RAS").lookup("RASModel")
    );

    Info<< "Selecting " << modelType <<" for AdjointRASModel"<< endl;

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn
        (
            "AdjointRASModel::New"
            "("
            "    const fvMesh&"
            ")"
        )   << "Unknown AdjointRASModel type "
            << modelType << nl << nl
            << "Valid AdjointRASModel types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    return autoPtr<AdjointRASModel>
           (
#ifdef IncompressibleFlow
                cstrIter()(mesh,adjIO,d,laminarTransport)
#endif
#ifdef CompressibleFlow
                cstrIter()(mesh,adjIO,d,thermo)
#endif
#ifdef PhaseIncompressibleFlow
                cstrIter()(mesh,adjIO,d,mixture)
#endif 
           );
}



// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

tmp<volScalarField> AdjointRASModel::nuEff()
{
    
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "nuEff",
            this->getNu()+nut_
        )
    );
}

tmp<volScalarField> AdjointRASModel::alphaEff()
{

#ifdef IncompressibleFlow
    const volScalarField& alphat = db_.lookupObject<volScalarField>("alphat");
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "alphaEff",
            this->getAlpha()+alphat
        )
    );
#endif

#ifdef CompressibleFlow
    const volScalarField& alphat = db_.lookupObject<volScalarField>("alphat");
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "alphaEff",
            thermo_.alphaEff(alphat)
        )
    );
#endif

#ifdef PhaseIncompressibleFlow
    const volScalarField& alphat = db_.lookupObject<volScalarField>("alphat");
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "alphaEff",
            this->getAlpha()+alphat
        )
    );
#endif

}

tmp<volScalarField> AdjointRASModel::getNu() const
{

#ifdef IncompressibleFlow
    return laminarTransport_.nu();
#endif

#ifdef CompressibleFlow
    return thermo_.mu()/rho_;
#endif

#ifdef PhaseIncompressibleFlow
    return mixture_.nu();
#endif

}

tmp<volScalarField> AdjointRASModel::getAlpha() const
{
    return this->getNu()/adjIO_.flowProperties["Pr"];
}

tmp<Foam::volScalarField> AdjointRASModel::getMu() const
{

#ifdef CompressibleFlow
    return thermo_.mu();
#else
    FatalErrorIn("flowCondition not valid!")<< abort(FatalError);
    return nut_;
#endif
        
}

tmp<volSymmTensorField> AdjointRASModel::devRhoReff()
{

    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("devRhoReff", U_.group()),
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            ( -phase_ * rho_ * nuEff() ) * dev(twoSymm(fvc::grad(U_)))
        )
    );
} 

tmp<fvVectorMatrix> AdjointRASModel::divDevRhoReff
(
    volVectorField& U
)
{

#ifdef IncompressibleFlow
    word divScheme="div((nuEff*dev2(T(grad(U)))))";
#endif
#ifdef CompressibleFlow
    word divScheme="div(((rho*nuEff)*dev2(T(grad(U)))))";
#endif
#ifdef PhaseIncompressibleFlow
    word divScheme="div(((rho*nuEff)*dev2(T(grad(U)))))";
#endif

    volScalarField& phase = phase_;
    volScalarField& rho = rho_;
    
    if (adjIO_.divDev2)
    {
        return
        (
          - fvm::laplacian(phase*rho*nuEff(), U)
          - fvc::div( (phase*rho*nuEff())*dev2(T(fvc::grad(U))),divScheme )
        );
    }
    else
    {
        return
        (
          - fvm::laplacian(phase*rho*nuEff(), U)
          - fvc::div( (phase*rho*nuEff())*dev(T(fvc::grad(U))),divScheme )
        );
    }
}

tmp<fvVectorMatrix> AdjointRASModel::divDevReff
(
    volVectorField& U
)
{
    return divDevRhoReff(U);
}

void AdjointRASModel::boundTurbVars(volScalarField& vsf, const dimensionedScalar& lowerBound)
{
    // copy from src/finiteVolume/cfdTools/general/bound/bound.C
    // the only difference is that we don't print the info
    
    const scalar minVsf = min(vsf).value();

    if (minVsf < lowerBound.value())
    {
        /*
        Info<< "bounding " << vsf.name()
            << ", min: " << minVsf
            << " max: " << max(vsf).value()
            << " average: " << gAverage(vsf.primitiveField())
            << endl;
        */
        vsf.primitiveFieldRef() = max
        (
            max
            (
                vsf.primitiveField(),
                fvc::average(max(vsf, lowerBound))().primitiveField()
              * pos(-vsf.primitiveField())
            ),
            lowerBound.value()
        );

        vsf.boundaryFieldRef() = max(vsf.boundaryField(), lowerBound.value());
    }

}


void AdjointRASModel::correctWallDist()
{
    d_.correct();
    // need to correct turbulence boundary conditions
    // this is because when the near wall distance changes, the nut, omega, epsilon at wall
    // may change if you use wall functions
    this->correctTurbBoundaryConditions();
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
