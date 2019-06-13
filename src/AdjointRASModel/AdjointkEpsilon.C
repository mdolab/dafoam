/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointkEpsilon.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointkEpsilon, 0);
addToRunTimeSelectionTable(AdjointRASModel, AdjointkEpsilon, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointkEpsilon::AdjointkEpsilon
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
    
    // KE parameters
    Cmu_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cmu",
            this->coeffDict_,
            0.09
        )
    ),
    C1_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C1",
            this->coeffDict_,
            1.44
        )
    ),
    C2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C2",
            this->coeffDict_,
            1.92
        )
    ),
    C3_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C3",
            this->coeffDict_,
            0
        )
    ),
    sigmak_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmak",
            this->coeffDict_,
            1.0
        )
    ),
    sigmaEps_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaEps",
            this->coeffDict_,
            1.3
        )
    ),
    
    // Augmented variables
    epsilon_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("epsilon")
        )
    ),
    epsilonRes_
    (
        IOobject
        (
            "epsilonRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("epsilonRes",dimensionSet(1,-1,-4,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("epsilonRes",dimensionSet(0,2,-4,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    epsilonResRef_
    (
        IOobject
        (
            "epsilonResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        epsilonRes_
    ),
    epsilonResPartDeriv_
    (
        IOobject
        (
            "epsilonResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        epsilonRes_
    ),
    epsilonRef_
    (
        IOobject
        (
            "epsilonRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        epsilon_
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
    )
    
{
    // add the turbulence state variables, this will be used in AdjointIndexing
    this->turbStates.append("k"); 
    this->turbStates.append("epsilon");
    this->copyTurbStates("Var2Ref"); // copy turbVars to turbVarsRef
    //this->calcTurbResiduals(1,0); 
    //Info<<nuTildaResidualRef_<<endl;
    
    // calculate the size of epsilonWallFunction faces
    label nWallFaces=0;
    forAll(epsilon_.boundaryField(),patchI)
    {
        if( epsilon_.boundaryField()[patchI].type() == "epsilonWallFunction" 
            and epsilon_.boundaryField()[patchI].size()>0 )
        {
            forAll(epsilon_.boundaryField()[patchI],faceI)
            {
                nWallFaces++;
                //Info<<"patchI: "<<patchI<<" faceI: "<<faceI<<endl;
            }
        }
    }
    
    // initialize epsilonNearWall
    epsilonNearWall_.setSize(nWallFaces);
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// KE member functions
tmp<fvScalarMatrix> AdjointkEpsilon::kSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            k_,
            dimVolume*rho_.dimensions()*k_.dimensions()
            /dimTime
        )
    );
}


tmp<fvScalarMatrix> AdjointkEpsilon::epsilonSource() const
{
    return tmp<fvScalarMatrix>
    (
        new fvScalarMatrix
        (
            epsilon_,
            dimVolume*rho_.dimensions()*epsilon_.dimensions()
            /dimTime
        )
    );
}


tmp<volScalarField> AdjointkEpsilon::DkEff() const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "DkEff",
            (nut_/sigmak_ + this->getNu())
        )
    );
}

tmp<volScalarField> AdjointkEpsilon::DepsilonEff() const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "DepsilonEff",
            (nut_/sigmaEps_ + this->getNu())
        )
    );
}


// Augmented functions

void AdjointkEpsilon::updateNut()
{
    nut_ = Cmu_*sqr(k_)/epsilon_;    
    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k
    
    return;
}


void AdjointkEpsilon::copyTurbStates(const word option)
{
    // "Ref2Var", assign ref states to states
    // "Var2Ref", assign states to ref states
    if(option=="Ref2Var")
    {
        epsilon_=epsilonRef_;
        k_=kRef_;
        this->correctTurbBoundaryConditions();
    }
    else if (option=="Var2Ref")
    {
        epsilonRef_ = epsilon_;
        kRef_ = k_;
    }
    else
    {
        FatalErrorIn("option not valid! Should be either Var2Ref or Ref2Var")
        << abort(FatalError);
    }

    this->updateNut();
    
    return;
    
}


void AdjointkEpsilon::correctTurbBoundaryConditions()
{

    k_.correctBoundaryConditions(); // kqWallFunction is a zero-gradient BC
    //this->correctEpsilonBoundaryConditions(); // special treatment for epsilon

    // Note: we need to update nut since we may have perturbed other turbulence vars
    // that affect the nut values
    this->updateNut(); 

    return;
}

void AdjointkEpsilon::correctEpsilonBoundaryConditions()
{
    // this is a special treatment for epsilon BC
    // we cant directly call epsilon.correctBoundaryConditions() because it will
    // modify the internal epsilon and G that are right next to walls. This will mess up adjoint Jacobians
    // To solve this issue, 
    // 1. we store the near wall epsilon before calling epsilon.correctBoundaryConditions()
    // 2. call epsilon.correctBoundaryConditions()
    // 3. Assign the stored near wall epsilon back
    // 4. Apply a zeroGradient BC for epsilon at the wall patches
    // *********** NOTE *************
    // this treatment will obviously downgrade the accuracy of adjoint derivative since it is not 100% consistent
    // with what is used for the flow solver; however, our observation shows that the impact is not very large. 

    // save the perturbed epsilon at the wall
    this->saveEpsilonNearWall();
    // correct epsilon boundary conditions, this includes updating wall face and near wall epsilon values,
    // updating the inter-proc BCs
    epsilon_.correctBoundaryConditions(); 
    // reset the corrected epsilon near wall cell to its perturbed value
    this->setEpsilonNearWall();

}


void AdjointkEpsilon::calcTurbResiduals
(  
    const label isRef,
    const label isPC,
    const word fvMatrixName
)
{    

    // Copied and modified based on the "correct" function
    
    word divKScheme="div(phi,k)";
    word divEpsilonScheme="div(phi,epsilon)";
    if(isPC) 
    {
        divKScheme="div(pc)";
        divEpsilonScheme="div(pc)";
    }

    //if (!this->turbulence_)
    //{
    //    return;
    //}

    // Local references
    //const alphaField& alpha = this->alpha_;
    //const rhoField& rho = this->rho_;
    //const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    //const volVectorField& U = this->U_;
    //volScalarField& nut = this->nut_;
    //fv::options& fvOptions(fv::options::New(this->mesh_));

    // we need to bound before computing residuals
    // this will avoid having NaN residuals
    this->boundTurbVars(epsilon_, epsilonMin_);
    
    volScalarField& phase = phase_;
    volScalarField& rho = rho_;
    surfaceScalarField& phaseRhoPhi = phaseRhoPhi_;

    //eddyViscosity<RASModel<BasicTurbulenceModel>>::correct();

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho), 
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField::Internal divU
    (
        fvc::div(fvc::absolute(phi_/fvc::interpolate(rho_),U_))().v()
    );

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField::Internal G
    (
        "kEpsilon:G",
        nut_.v()*(dev(twoSymm(tgradU().v())) && tgradU().v())
    );
    tgradU.clear();

    // Update epsilon and G at the wall
    //epsilon_.boundaryFieldRef().updateCoeffs();

    // special treatment for epsilon BC
    this->correctEpsilonBoundaryConditions();

    // Dissipation equation
    tmp<fvScalarMatrix> epsEqn
    (
        fvm::ddt(phase, rho, epsilon_)
      + fvm::div(phaseRhoPhi, epsilon_)
      - fvm::laplacian(phase*rho*DepsilonEff(), epsilon_)
     ==
        C1_*phase()*rho()*G*epsilon_()/k_()
      - fvm::SuSp((scalar(2.0/3.0)*C1_ - C3_)*phase()*rho()*divU, epsilon_)
      - fvm::Sp(C2_*phase()*rho()*epsilon_()/k_(), epsilon_)
      + epsilonSource()
      //+ fvOptions(alpha, rho, epsilon_)
    );

    epsEqn.ref().relax();
    //fvOptions.constrain(epsEqn.ref());
    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());

    // reset the corrected epsilon near wall cell to its perturbed value
    this->setEpsilonNearWall();

    //solve(epsEqn);
    //fvOptions.correct(epsilon_);
    //bound(epsilon_, epsilonMin_);

    if(fvMatrixName=="epsilonEqn") 
    {
        fvMatrixDiag.clear();
        fvMatrixLower.clear();
        fvMatrixUpper.clear();
        fvMatrixDiag = epsEqn.ref().diag();
        fvMatrixLower = epsEqn.ref().lower();
        fvMatrixUpper = epsEqn.ref().upper();
    }

    // calculate residuals
    if(isRef)
    {
        epsilonResRef_ = epsEqn() & epsilon_;
        // need to normalize Res. NOTE: epsilonRes is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("epsilonRes",adjIO_.normalizeResiduals)) 
        {
            forAll(epsilonResRef_,cellI)
            {
                epsilonResRef_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }
    else
    {
        epsilonRes_ = epsEqn() & epsilon_;
        // need to normalize Res. NOTE: epsilonRes is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("epsilonRes",adjIO_.normalizeResiduals)) 
        {
            forAll(epsilonRes_,cellI)
            {
                epsilonRes_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }

    // we need to bound before computing residuals
    // this will avoid having NaN residuals
    this->boundTurbVars(k_, this->kMin_);

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(phase, rho, k_)
      + fvm::div(phaseRhoPhi, k_)
      - fvm::laplacian(phase*rho*DkEff(), k_)
     ==
        phase()*rho()*G 
      - fvm::SuSp(2.0/3.0*phase()*rho()*divU, k_)
      - fvm::Sp(phase()*rho()*epsilon_()/k_(), k_)
      + kSource()
     // + fvOptions(alpha, rho, k_)
    );

    kEqn.ref().relax();
    //fvOptions.constrain(kEqn.ref());
    //solve(kEqn);
    //fvOptions.correct(k_);
    //bound(k_, this->kMin_);

    //correctNut();

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
    }
    
    return;
}

void AdjointkEpsilon::saveEpsilonNearWall()
{
    // save the current near wall epsilon values to epsilonNearWall_
    label counterI=0;
    forAll(epsilon_.boundaryField(),patchI)
    {
        if( epsilon_.boundaryField()[patchI].type() == "epsilonWallFunction" 
            and epsilon_.boundaryField()[patchI].size()>0 )
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells,faceI)
            {
                //Info<<"Near Wall cellI: "<<faceCells[faceI]<<endl;
                epsilonNearWall_[counterI] = epsilon_[faceCells[faceI]];
                counterI++;
            }
        }
    }
    return;
}

void AdjointkEpsilon::setEpsilonNearWall()
{
    // set the current near wall epsilon values from epsilonNearWall_
    // here we also apply a zeroGradient BC to the wall faces
    label counterI=0;
    forAll(epsilon_.boundaryField(),patchI)
    {
        if( epsilon_.boundaryField()[patchI].type() == "epsilonWallFunction" 
            and epsilon_.boundaryField()[patchI].size()>0 )
        {
            const UList<label>& faceCells = mesh_.boundaryMesh()[patchI].faceCells();
            forAll(faceCells,faceI)
            {
                epsilon_[faceCells[faceI]] = epsilonNearWall_[counterI];
                epsilon_.boundaryFieldRef()[patchI][faceI] = epsilon_[faceCells[faceI]]; // zeroGradient BC
                counterI++;
            }
        }
    }
    return;
}


void AdjointkEpsilon::correctAdjStateResidualTurbCon
(
    List< List<word> >& adjStateResidualConInfo
)
{

    // For KE model, we need to replace nut with k, epsilon
    forAll(adjStateResidualConInfo,idxI) // idxI is the con level
    {
        forAll(adjStateResidualConInfo[idxI],idxJ)
        {
            word conStateName = adjStateResidualConInfo[idxI][idxJ];
            if( conStateName == "nut" ) 
            {
                adjStateResidualConInfo[idxI][idxJ]="epsilon"; // replace nut with epsilon
                adjStateResidualConInfo[idxI].append("k");  // also add k for that level
            }
        }
        
    }
    
    return; 

}

void AdjointkEpsilon::setAdjStateResidualTurbCon
(
    HashTable< List< List<word> > >& adjStateResidualConInfo
)
{

    word pName=adjIO_.getPName();

#ifdef IncompressibleFlow
    adjStateResidualConInfo.set
    (
        "epsilonRes",
        {
            {"U","epsilon","k","phi"}, // lv0
            {"U","epsilon","k"},       // lv1
            {"U","epsilon","k"}        // lv2
        }
    );
    
    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","epsilon","k","phi"}, // lv0
            {"U","epsilon","k"},       // lv1
            {"U","epsilon","k"}        // lv2
        }
    );
#endif 

#ifdef CompressibleFlow
    adjStateResidualConInfo.set
    (
        "epsilonRes",
        {
            {"U","T",pName,"epsilon","k","phi"}, // lv0
            {"U","T",pName,"epsilon","k"},       // lv1
            {"U","T",pName,"epsilon","k"}        // lv2
        }
    );
    
    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","T",pName,"epsilon","k","phi"}, // lv0
            {"U","T",pName,"epsilon","k"},       // lv1
            {"U","T",pName,"epsilon","k"}        // lv2
        }
    );
#endif

}

void AdjointkEpsilon::clearTurbVars()
{
    // clear any variables that are not needed for solveAdjoint
    //kRef_.clear();
    kRes_.clear();
    kResRef_.clear();
    kResPartDeriv_.clear();

    //epsilonRef_.clear();
    epsilonRes_.clear();
    epsilonResRef_.clear();
    epsilonResPartDeriv_.clear();
}

void AdjointkEpsilon::writeTurbStates()
{
    nut_.write();
    k_.write();
    epsilon_.write();
}


} // End namespace Foam

// ************************************************************************* //
