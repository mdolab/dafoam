/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "Adjointv2f.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(Adjointv2f, 0);
addToRunTimeSelectionTable(AdjointRASModel, Adjointv2f, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Adjointv2f::Adjointv2f
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
    
    // v2f parameters
    Cmu_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cmu",
            this->coeffDict_,
            0.22
        )
    ),
    CmuKEps_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "CmuKEps",
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
            1.4
        )
    ),
    C2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "C2",
            this->coeffDict_,
            0.3
        )
    ),
    CL_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "CL",
            this->coeffDict_,
            0.23
        )
    ),
    Ceta_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceta",
            this->coeffDict_,
            70.0
        )
    ),
    Ceps2_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceps2",
            this->coeffDict_,
            1.9
        )
    ),
    Ceps3_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Ceps3",
            this->coeffDict_,
            -0.33
        )
    ),
    sigmaK_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "sigmaK",
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
    ),
    
    v2_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("v2")
        )
    ),
    v2Res_
    (
        IOobject
        (
            "v2Res",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("v2Res",dimensionSet(1,-1,-3,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("v2Res",dimensionSet(0,2,-3,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    v2ResRef_
    (
        IOobject
        (
            "v2ResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        v2Res_
    ),
    v2ResPartDeriv_
    (
        IOobject
        (
            "v2ResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        v2Res_
    ),
    v2Ref_
    (
        IOobject
        (
            "v2Ref",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        v2_
    ),

    f_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("f")
        )
    ),
    fRes_
    (
        IOobject
        (
            "fRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("fRes",dimensionSet(0,-2,-1,0,0,0,0),0.0),
        zeroGradientFvPatchScalarField::typeName
    ),
    fResRef_
    (
        IOobject
        (
            "fResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        fRes_
    ),
    fResPartDeriv_
    (
        IOobject
        (
            "fResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        fRes_
    ),
    fRef_
    (
        IOobject
        (
            "fRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        f_
    ),
    v2Min_(dimensionedScalar("v2Min", v2_.dimensions(), SMALL)),
    fMin_(dimensionedScalar("fMin", f_.dimensions(), 0.0))
    
{
    // add the turbulence state variables, this will be used in AdjointIndexing
    this->turbStates.append("k"); 
    this->turbStates.append("epsilon");
    this->turbStates.append("v2");
    this->turbStates.append("f");
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


// v2f member functions

tmp<volScalarField> Adjointv2f::DkEff() const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            "DkEff",
            (nut_/sigmaK_ + this->getNu())
        )
    );
}

tmp<volScalarField> Adjointv2f::DepsilonEff() const
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

tmp<volScalarField> Adjointv2f::Ts() const
{
    // SAF: limiting thermo->nu(). If psiThermo is used rho might be < 0
    // temporarily when p < 0 then nu < 0 which needs limiting
    return
        max
        (
            k_/epsilon_,
            6.0*sqrt
            (
                max
                (
                    this->getNu(),
                    dimensionedScalar(this->getNu()().dimensions(), Zero)
                )
              / epsilon_
            )
        );
}


tmp<volScalarField> Adjointv2f::Ls() const
{
    // SAF: limiting thermo->nu(). If psiThermo is used rho might be < 0
    // temporarily when p < 0 then nu < 0 which needs limiting
    return
        CL_
      * max
        (
            pow(k_, 1.5)/epsilon_,
            Ceta_*pow025
            (
                pow3
                (
                    max
                    (
                        this->getNu(),
                        dimensionedScalar(this->getNu()().dimensions(), Zero)
                    )
                )/epsilon_
            )
        );
}



// Augmented functions

void Adjointv2f::updateNut()
{
    nut_ = min(CmuKEps_*sqr(k_)/epsilon_, Cmu_*v2_*this->Ts());
    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k
    
    return;
}


void Adjointv2f::copyTurbStates(const word option)
{
    // "Ref2Var", assign ref states to states
    // "Var2Ref", assign states to ref states
    if(option=="Ref2Var")
    {
        epsilon_=epsilonRef_;
        k_=kRef_;
        v2_=v2Ref_;
        f_=fRef_;
        this->correctTurbBoundaryConditions();
    }
    else if (option=="Var2Ref")
    {
        epsilonRef_ = epsilon_;
        kRef_ = k_;
        v2Ref_ = v2_;
        fRef_ = f_;
    }
    else
    {
        FatalErrorIn("option not valid! Should be either Var2Ref or Ref2Var")
        << abort(FatalError);
    }

    this->updateNut();
    
    return;
    
}


void Adjointv2f::correctTurbBoundaryConditions()
{

    k_.correctBoundaryConditions(); // kqWallFunction is a zero-gradient BC
    //this->correctEpsilonBoundaryConditions(); // special treatment for epsilon

    v2_.correctBoundaryConditions();

    f_.correctBoundaryConditions();

    // Note: we need to update nut since we may have perturbed other turbulence vars
    // that affect the nut values
    this->updateNut(); 

    return;
}

void Adjointv2f::correctEpsilonBoundaryConditions()
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


void Adjointv2f::calcTurbResiduals
(  
    const label isRef,
    const label isPC,
    const word fvMatrixName
)
{    

    // Copied and modified based on the "correct" function
    
    word divKScheme="div(phi,k)";
    word divEpsilonScheme="div(phi,epsilon)";
    word divV2Scheme="div(phi,v2)";
    if(isPC) 
    {
        divKScheme="div(pc)";
        divEpsilonScheme="div(pc)";
        divV2Scheme="div(pc)";
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

    // Note: for compressible flow, the "this->phi()" function divides phi by fvc:interpolate(rho), 
    // while for the incompresssible "this->phi()" returns phi only
    // see src/TurbulenceModels/compressible/compressibleTurbulenceModel.C line 62 to 73
    volScalarField divU
    (
        fvc::div
        (
            fvc::absolute
            (
                phi_/fvc::interpolate(rho_),
                U_
            )
        )
    );

    // Use N=6 so that f=0 at walls
    const dimensionedScalar N("N", dimless, 6.0);

    const volTensorField gradU(fvc::grad(U_));
    const volScalarField S2(2*magSqr(dev(symm(gradU))));

    const volScalarField G("v2f:G", nut_*S2);
    const volScalarField Ts(this->Ts());
    const volScalarField L2("v2f:L2", sqr(Ls()));
    const volScalarField v2fAlpha
    (
        "v2f:alpha",
        1.0/Ts*((C1_ - N)*v2_ - 2.0/3.0*k_*(C1_ - 1.0))
    );

    const volScalarField Ceps1
    (
        "Ceps1",
        1.4*(1.0 + 0.05*min(sqrt(k_/v2_), scalar(100)))
    );

    // Update epsilon (and possibly G) at the wall
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
        Ceps1*phase*rho*G/Ts
      - fvm::SuSp(((2.0/3.0)*Ceps1 + Ceps3_)*phase*rho*divU, epsilon_)
      - fvm::Sp(Ceps2_*phase*rho/Ts, epsilon_)
    );

    epsEqn.ref().relax();
    epsEqn.ref().boundaryManipulate(epsilon_.boundaryFieldRef());

    // reset the corrected epsilon near wall cell to its perturbed value
    this->setEpsilonNearWall();

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
        phase*rho*G
      - fvm::SuSp((2.0/3.0)*phase*rho*divU, k_)
      - fvm::Sp(phase*rho*epsilon_/k_, k_)
    );

    kEqn.ref().relax();

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


    // we need to bound before computing residuals
    // this will avoid having NaN residuals
    this->boundTurbVars(f_, fMin_);

    // Relaxation function equation
    tmp<fvScalarMatrix> fEqn
    (
       - fvm::laplacian(f_)
      ==
       - fvm::Sp(1.0/L2, f_)
       - 1.0/L2/k_*(v2fAlpha - C2_*G)
    );

    fEqn.ref().relax();

    if(fvMatrixName=="fEqn") 
    {
        fvMatrixDiag.clear();
        fvMatrixLower.clear();
        fvMatrixUpper.clear();
        fvMatrixDiag = fEqn.ref().diag();
        fvMatrixLower = fEqn.ref().lower();
        fvMatrixUpper = fEqn.ref().upper();
    }

    // calculate residuals
    if(isRef)
    {
        fResRef_ = fEqn() & f_;
        // need to normalize Res. NOTE: fRes is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("fRes",adjIO_.normalizeResiduals)) 
        {
            forAll(fResRef_,cellI)
            {
                fResRef_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }
    else
    {
        fRes_ = fEqn() & f_;
        // need to normalize Res. NOTE: fRes is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("fRes",adjIO_.normalizeResiduals)) 
        {
            forAll(fRes_,cellI)
            {
                fRes_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }

    // we need to bound before computing residuals
    // this will avoid having NaN residuals
    this->boundTurbVars(v2_, v2Min_);

    // Relaxation function equation
    tmp<fvScalarMatrix> v2Eqn
    (
        fvm::ddt(phase, rho, v2_)
      + fvm::div(phaseRhoPhi, v2_,divV2Scheme)
      - fvm::laplacian(phase*rho*DkEff(), v2_)
      ==
        phase*rho*min(k_*f_, C2_*G - v2fAlpha)
      - fvm::Sp(N*phase*rho*epsilon_/k_, v2_)
    );

    v2Eqn.ref().relax();

    if(fvMatrixName=="v2Eqn") 
    {
        fvMatrixDiag.clear();
        fvMatrixLower.clear();
        fvMatrixUpper.clear();
        fvMatrixDiag = v2Eqn.ref().diag();
        fvMatrixLower = v2Eqn.ref().lower();
        fvMatrixUpper = v2Eqn.ref().upper();
    }

    // calculate residuals
    if(isRef)
    {
        v2ResRef_ = v2Eqn() & v2_;
        // need to normalize Res. NOTE: v2Res is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("v2Res",adjIO_.normalizeResiduals)) 
        {
            forAll(v2ResRef_,cellI)
            {
                v2ResRef_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }
    else
    {
        v2Res_ = v2Eqn() & v2_;
        // need to normalize Res. NOTE: v2Res is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("v2Res",adjIO_.normalizeResiduals)) 
        {
            forAll(v2Res_,cellI)
            {
                v2Res_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }


    
    return;
}

void Adjointv2f::saveEpsilonNearWall()
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

void Adjointv2f::setEpsilonNearWall()
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


void Adjointv2f::correctAdjStateResidualTurbCon
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
                adjStateResidualConInfo[idxI].append("v2");  // also add k for that level
                adjStateResidualConInfo[idxI].append("f");  // also add k for that level
            }
        }
        
    }
    
    return; 

}

void Adjointv2f::setAdjStateResidualTurbCon
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
    adjStateResidualConInfo.set
    (
        "fRes",
        {
            {"U","f","v2","epsilon","k","phi"}, // lv0
            {"U","f","v2","epsilon","k"},       // lv1
            {"U","f","v2","epsilon","k"}        // lv2
        }
    );
    adjStateResidualConInfo.set
    (
        "v2Res",
        {
            {"U","v2","f","epsilon","k","phi"}, // lv0
            {"U","v2","f","epsilon","k"},       // lv1
            {"U","v2","f","epsilon","k"}        // lv2
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
    adjStateResidualConInfo.set
    (
        "fRes",
        {
            {"U","T",pName,"f","v2","epsilon","k","phi"}, // lv0
            {"U","T",pName,"f","v2","epsilon","k"},       // lv1
            {"U","T",pName,"f","v2","epsilon","k"}        // lv2
        }
    );
    adjStateResidualConInfo.set
    (
        "v2Res",
        {
            {"U","T",pName,"f","v2","epsilon","k","phi"}, // lv0
            {"U","T",pName,"f","v2","epsilon","k"},       // lv1
            {"U","T",pName,"f","v2","epsilon","k"}        // lv2
        }
    );
#endif

}

void Adjointv2f::clearTurbVars()
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

    fRes_.clear();
    fResRef_.clear();
    fResPartDeriv_.clear();


    v2Res_.clear();
    v2ResRef_.clear();
    v2ResPartDeriv_.clear();

}

void Adjointv2f::writeTurbStates()
{
    nut_.write();
    k_.write();
    epsilon_.write();
    f_.write();
    v2_.write();
}


} // End namespace Foam

// ************************************************************************* //
