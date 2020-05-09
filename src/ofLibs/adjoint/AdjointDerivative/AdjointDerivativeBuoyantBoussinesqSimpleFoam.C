/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeBuoyantBoussinesqSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeBuoyantBoussinesqSimpleFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeBuoyantBoussinesqSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeBuoyantBoussinesqSimpleFoam::AdjointDerivativeBuoyantBoussinesqSimpleFoam
(
    fvMesh& mesh,
    const AdjointIO& adjIO,
    const AdjointSolverRegistry& adjReg,
    AdjointRASModel& adjRAS,
    AdjointIndexing& adjIdx,
    AdjointJacobianConnectivity& adjCon,
    AdjointObjectiveFunction& adjObj
)
    :
    AdjointDerivative(mesh,adjIO,adjReg,adjRAS,adjIdx,adjCon,adjObj),
    // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
    setResidualClassMemberVector(U,dimensionSet(0,1,-2,0,0,0,0)),
    setResidualClassMemberScalar(p_rgh,dimensionSet(0,0,-1,0,0,0,0)),
    setResidualClassMemberScalar(T,dimensionSet(0,0,-1,1,0,0,0)),
    setResidualClassMemberScalar(G,dimensionSet(1,-1,-3,0,0,0,0)),
    setResidualClassMemberPhi(phi),
    // these are intermediate variables or objects
    alphat_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("alphat")
        )
    ),
    rhok_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("rhok")
        )
    ),
    p_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("p")
        )
    ),
    g_
    (
        const_cast<uniformDimensionedVectorField&>
        (
            db_.lookupObject<uniformDimensionedVectorField>("g")
        )
    ),
    hRef_
    (
        const_cast<uniformDimensionedScalarField&>
        (
            db_.lookupObject<uniformDimensionedScalarField>("hRef")
        )
    ),
    gh_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("gh")
        )
    ),
    ghf_
    (
        const_cast<surfaceScalarField&>
        (
            db_.lookupObject<surfaceScalarField>("ghf")
        )
    ),
    rhoCpRef_
    (
        "rhoCpRef",
        dimDensity*dimEnergy/dimMass/dimTemperature,
        1.0
    ),
    radiationProperties_
    (
        IOobject
        (
            "radiationProperties",
            mesh_.time().constant(),
            mesh_.time(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false // Do not register
        )
    ),
    radiationOn_
    (
        radiationProperties_.lookup("radiation")
    ),
    a_
    (
        radiationProperties_.subDict("constantAbsorptionEmissionCoeffs").lookup("absorptivity")
    ),
    e_
    (
        radiationProperties_.subDict("constantAbsorptionEmissionCoeffs").lookup("emissivity")
    ),
    E_
    (
        radiationProperties_.subDict("constantAbsorptionEmissionCoeffs").lookup("E")
    ),
    gamma_
    (
        IOobject
        (
            "gammaRad",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh_,
        dimensionedScalar("gammaRad",dimensionSet(0,1,0,0,0,0,0),1.0/(3.0*a_.value() + ROOTVSMALL)),
        zeroGradientFvPatchScalarField::typeName
    ),
     
    // create simpleControl
    simple_(mesh) 
    
{
    this->copyStates("Var2Ref"); // copy states to statesRef

    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            mesh_.time().constant(),
            mesh_.time(),
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false // Do not register
        )
    );
    dimensionedScalar rhoRef
    (
        "rhoRef",
        dimDensity,
        transportProperties
    );
    dimensionedScalar CpRef
    (
        "CpRef",
        dimSpecificHeatCapacity,
        transportProperties
    );
    rhoCpRef_ = rhoRef*CpRef;

    //Info<<"Radiation for Adjoint: "<<radiationOn_<<endl;

}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::tmp<Foam::fvScalarMatrix> AdjointDerivativeBuoyantBoussinesqSimpleFoam::ST
(
    const dimensionedScalar& rhoCp,
    volScalarField& T
) const
{
    
    return
    (
        this->Ru()/rhoCp
      - fvm::Sp(this->Rp()*pow3(T_)/rhoCp, T_)
    );
    
}

Foam::tmp<Foam::volScalarField> AdjointDerivativeBuoyantBoussinesqSimpleFoam::Rp() const
{
    if (radiationOn_ == "on")
    {
        volScalarField e
        (
            IOobject
            (
                "e",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            mesh_,
            e_
        );

        return tmp<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    "Rp",
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false
                ),
                4.0*e*Foam::constant::physicoChemical::sigma
            )
        );
    }
    else if (radiationOn_ == "off")
    {
        return tmp<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    "Rp0",
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar
                (
                    "Rp0",
                    Foam::constant::physicoChemical::sigma.dimensions()/dimLength,
                    0.0
                )
            )
        );
    }
    else
    {
        FatalErrorIn("")<<"radiation must be either on or off"<< abort(FatalError);
        return tmp<volScalarField>
        (
            new volScalarField
            (
                IOobject
                (
                    "Rp0",
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar
                (
                    "Rp0",
                    Foam::constant::physicoChemical::sigma.dimensions()/dimLength,
                    0.0
                )
            )
        );
    }
    
}


Foam::tmp<Foam::DimensionedField<Foam::scalar, Foam::volMesh>>
AdjointDerivativeBuoyantBoussinesqSimpleFoam::Ru() const
{
    if (radiationOn_ == "on")
    {
        volScalarField a
        (
            IOobject
            (
                "a",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            mesh_,
            a_
        );
    
        volScalarField E
        (
            IOobject
            (
                "E",
                mesh_.time().timeName(),
                mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            mesh_,
            E_
        );
    
        return a()*G_() - E();
    }
    else if (radiationOn_ == "off")
    {
        return tmp<volScalarField::Internal>
        (
            new volScalarField::Internal
            (
                IOobject
                (
                    "Ru0",
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar
                (
                    "Ru0", dimMass/dimLength/pow3(dimTime), 0.0
                )
            )
        );
    }
    else
    {
        FatalErrorIn("")<<"radiation must be either on or off"<< abort(FatalError);

        return tmp<volScalarField::Internal>
        (
            new volScalarField::Internal
            (
                IOobject
                (
                    "Ru0",
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh_,
                dimensionedScalar
                (
                    "Ru0", dimMass/dimLength/pow3(dimTime), 0.0
                )
            )
        );
    }
}

void AdjointDerivativeBuoyantBoussinesqSimpleFoam::calcResiduals
(
    const label isRef,
    const label isPC,
    const word fvMatrixName,
    const label updatePhi
)
{
    // We dont support MRF and fvOptions so all the related lines are commented 
    // out for now
    
    word divUScheme="div(phi,U)";
    word divTScheme="div(phi,T)";
    if(isPC) 
    {
        divUScheme="div(pc)"; 
        divTScheme="div(pc)";
    }
    
    // ******** U Residuals **********
    // copied and modified from UEqn.H
    tmp<fvVectorMatrix> tUEqn
    (
        fvm::div(phi_, U_, divUScheme)
      + adjRAS_.divDevReff(U_)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();
    
    if (!updatePhi)
    {
        if(isRef) 
        {
            UResRef_  = (UEqn&U_) - fvc::reconstruct
                                    (
                                        (
                                          - ghf_*fvc::snGrad(rhok_)
                                          - fvc::snGrad(p_rgh_)
                                        )*mesh_.magSf()
                                    );
        }
        else
        {
            URes_  = (UEqn&U_) - fvc::reconstruct
                                 (
                                     (
                                       - ghf_*fvc::snGrad(rhok_)
                                       - fvc::snGrad(p_rgh_)
                                     )*mesh_.magSf()
                                 );
        }
        // need to normalize Res.
        normalizeResiduals(URes);
                                
    }
    
    // ******** T Residuals **********
    // copied and modified from TEqn.H
    
    dimensionedScalar Pr1
    (
        "Pr1",
        dimless,
        adjIO_.flowProperties["Pr"]
    );

    volScalarField alphaEff("alphaEff", adjRAS_.getNu()/Pr1 + alphat_);

    fvScalarMatrix TEqn
    (
        fvm::div(phi_, T_, divTScheme)
      - fvm::laplacian(alphaEff, T_)
      ==
        this->ST(rhoCpRef_, T_)
    );

    if (!updatePhi)
    {
        if(isRef) TResRef_  = TEqn&T_;
        else TRes_  = TEqn&T_;
        // need to normalize Res.
        normalizeResiduals(TRes);
    }

    // ******** G Residuals **********
    // copied and modified from P1.C

    volScalarField a
    (
        IOobject
        (
            "a",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh_,
        a_
    );

    volScalarField e
    (
        IOobject
        (
            "e",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh_,
        e_
    );

    volScalarField E
    (
        IOobject
        (
            "E",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            false
        ),
        mesh_,
        E_
    );

    // Solve G transport equation
    fvScalarMatrix GEqn
    (
        fvm::laplacian(gamma_, G_)
      - fvm::Sp(a, G_)
     ==
      - 4.0*(e*Foam::constant::physicoChemical::sigma*pow4(T_) ) - E
    );

    if (!updatePhi)
    {
        if(isRef) GResRef_  = GEqn&G_;
        else GRes_  = GEqn&G_;
        // need to normalize Res.
        normalizeResiduals(GRes);
    }
    
    // ******** p Residuals **********
    // copied and modified from pEqn.H
    // NOTE manually set pRefCell and pRefValue
    label pRefCell=0;
    
    // Note: relax UEqn after the URes is calculated
    UEqn.relax();
    
    volScalarField rAU("rAU", 1.0/UEqn.A());
    surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU));
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U_, p_rgh_));
    //***************** NOTE *******************
    // we should not use the constrainHbyA function above since it
    // will degrade the accuracy of shape derivatives. Basically, we should
    // not constrain any variable because it will create discontinuity
    volVectorField HbyA("HbyA", U_);
    HbyA = rAU*UEqn.H();

    tUEqn.clear();

    surfaceScalarField phig(-rAUf*ghf_*fvc::snGrad(rhok_)*mesh_.magSf());

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        fvc::flux(HbyA)
    );

    adjustPhi(phiHbyA, U_, p_rgh_);

    phiHbyA += phig;

    fvScalarMatrix p_rghEqn
    (
        fvm::laplacian(rAUf, p_rgh_) == fvc::div(phiHbyA)
    );
    p_rghEqn.setReference(pRefCell, getRefCellValue(p_rgh_, pRefCell));

    if (!updatePhi)
    {
        if(isRef) p_rghResRef_  = p_rghEqn&p_rgh_;
        else p_rghRes_  = p_rghEqn&p_rgh_;
        // need to normalize Res.
        normalizeResiduals(p_rghRes);
    }
    
    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    if (!updatePhi)
    {
        if(isRef) phiResRef_ = phiHbyA - p_rghEqn.flux() - phi_;
        else phiRes_ = phiHbyA - p_rghEqn.flux() - phi_;
        // need to normalize phiRes
        normalizePhiResiduals(phiRes);
    }

    if(updatePhi) phi_=phiHbyA - p_rghEqn.flux();
    
    return;

}

void AdjointDerivativeBuoyantBoussinesqSimpleFoam::updateIntermediateVariables()
{
    // we need to update the rhok and p 
    dimensionedScalar beta1
    (
        "beta1",
        dimless/dimTemperature,
        adjIO_.flowProperties["beta"]
    );
    dimensionedScalar TRef1
    (
        "TRef1",
        dimTemperature,
        adjIO_.flowProperties["TRef"]
    );
    dimensionedScalar Prt1
    (
        "Prt1",
        dimless,
        adjIO_.flowProperties["Prt"]
    );

    rhok_ = 1.0 - beta1*(T_ - TRef1);
    // need to update p_rgh again because the buoyantPressure BC depends on rhok
    p_rgh_.correctBoundaryConditions();

    dimensionedScalar ghRef = - mag(g_)*hRef_;
    gh_ =  (g_ & mesh_.C()) - ghRef;
    ghf_ = (g_ & mesh_.Cf()) - ghRef;
    p_ = p_rgh_ + rhok_*gh_;
    
    alphat_ = adjRAS_.getNut()/Prt1;
    alphat_.correctBoundaryConditions();
}


} // End namespace Foam

// ************************************************************************* //
