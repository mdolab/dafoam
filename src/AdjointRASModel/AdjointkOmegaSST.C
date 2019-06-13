/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0
    
\*---------------------------------------------------------------------------*/

#include "AdjointkOmegaSST.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointkOmegaSST, 0);
addToRunTimeSelectionTable(AdjointRASModel, AdjointkOmegaSST, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointkOmegaSST::AdjointkOmegaSST
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
    y_(db_.lookupObject<volScalarField>("yWall"))
    
{
    // add the turbulence state variables, this will be used in AdjointIndexing
    this->turbStates.append("k"); 
    this->turbStates.append("omega");
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
tmp<volScalarField> AdjointkOmegaSST::F1
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


tmp<volScalarField> AdjointkOmegaSST::F2() const
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



tmp<volScalarField> AdjointkOmegaSST::F3() const
{
     
    tmp<volScalarField> arg3 = min
    (
        150*(this->getNu())/(omega_*sqr(y_)),
        scalar(10)
    );

    return 1 - tanh(pow4(arg3));
}

tmp<volScalarField> AdjointkOmegaSST::F23() const
{
    tmp<volScalarField> f23(F2());

    if (F3_)
    {
        f23.ref() *= F3();
    }

    return f23;
}


tmp<volScalarField::Internal> AdjointkOmegaSST::GbyNu
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


tmp<volScalarField::Internal> AdjointkOmegaSST::Pk
(
    const volScalarField::Internal& G
) const
{
    return min(G, (c1_*betaStar_)*k_()*omega_());
}

tmp<volScalarField::Internal> AdjointkOmegaSST::epsilonByk
(
    const volScalarField& F1,
    const volTensorField& gradU
) const
{
    return betaStar_*omega_();
}


tmp<fvScalarMatrix> AdjointkOmegaSST::kSource() const
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


tmp<fvScalarMatrix> AdjointkOmegaSST::omegaSource() const
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

tmp<fvScalarMatrix> AdjointkOmegaSST::Qsas
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


// Augmented functions

void AdjointkOmegaSST::updateNut()
{
    const volVectorField U=db_.lookupObject<volVectorField>("U");
    tmp<volTensorField> tgradU = fvc::grad(U);
    volScalarField S2(2*magSqr(symm(tgradU())));

    nut_ = a1_*k_/max(a1_*omega_, b1_*F23()*sqrt(S2));
    
    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k
    
    return;
}


void AdjointkOmegaSST::copyTurbStates(const word option)
{
    // "Ref2Var", assign ref states to states
    // "Var2Ref", assign states to ref states
    if(option=="Ref2Var")
    {
        omega_=omegaRef_;
        k_=kRef_;
        this->correctTurbBoundaryConditions();
    }
    else if (option=="Var2Ref")
    {
        omegaRef_ = omega_;
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


void AdjointkOmegaSST::correctTurbBoundaryConditions()
{

    k_.correctBoundaryConditions(); // kqWallFunction is a zero-gradient BC
    //this->correctOmegaBoundaryConditions(); // special treatment for omega

    // Note: we need to update nut since we may have perturbed other turbulence vars
    // that affect the nut values
    this->updateNut(); 
    
    return;
}

void AdjointkOmegaSST::correctOmegaBoundaryConditions()
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


void AdjointkOmegaSST::calcTurbResiduals
(  
    const label isRef,
    const label isPC,
    const word fvMatrixName
)
{    

    // Copied and modified based on the "correct" function
    
    word divKScheme="div(phi,k)";
    word divOmegaScheme="div(phi,omega)";
    if(isPC) 
    {
        divKScheme="div(pc)";
        divOmegaScheme="div(pc)";
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
    volScalarField::Internal G("kOmegaSST:G", nut_*GbyNu0);

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
    
    return;
}

void AdjointkOmegaSST::saveOmegaNearWall()
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

void AdjointkOmegaSST::setOmegaNearWall()
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


void AdjointkOmegaSST::correctAdjStateResidualTurbCon
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

void AdjointkOmegaSST::setAdjStateResidualTurbCon
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
            {"U","omega","k","phi"}, // lv0
            {"U","omega","k"},       // lv1
            {"U","omega","k"}        // lv2
        }
    );
    
    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","omega","k","phi"}, // lv0
            {"U","omega","k"},       // lv1
            {"U","omega","k"}        // lv2
        }
    );
#endif 

#ifdef CompressibleFlow
    adjStateResidualConInfo.set
    (
        "omegaRes",
        {
            {"U","T",pName,"omega","k","phi"}, // lv0
            {"U","T",pName,"omega","k"},       // lv1
            {"U","T",pName,"omega","k"}        // lv2
        }
    );
    
    adjStateResidualConInfo.set
    (
        "kRes",
        {
            {"U","T",pName,"omega","k","phi"}, // lv0
            {"U","T",pName,"omega","k"},       // lv1
            {"U","T",pName,"omega","k"}        // lv2
        }
    );
#endif
}

void AdjointkOmegaSST::clearTurbVars()
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
}

void AdjointkOmegaSST::writeTurbStates()
{
    nut_.write();
    k_.write();
    omega_.write();
}

} // End namespace Foam

// ************************************************************************* //
