/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointkOmega.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointkOmega, 0);
addToRunTimeSelectionTable(AdjointRASModel, AdjointkOmega, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointkOmega::AdjointkOmega
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
    
    // kOmega parameters
    Cmu_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "betaStar",
            this->coeffDict_,
            0.09
        )
    ),
    beta_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "beta",
            this->coeffDict_,
            0.072
        )
    ),
    gamma_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "gamma",
            this->coeffDict_,
            0.52
        )
    ),
    alphaK_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaK",
            this->coeffDict_,
            0.5
        )
    ),
    alphaOmega_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "alphaOmega",
            this->coeffDict_,
            0.5
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
    )
    
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



// Augmented functions

void AdjointkOmega::updateNut()
{
    
    nut_ = k_/omega_;
    nut_.correctBoundaryConditions(); // nutkWallFunction: update wall face nut based on k
    
    return;
}


void AdjointkOmega::copyTurbStates(const word option)
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


void AdjointkOmega::correctTurbBoundaryConditions()
{

    k_.correctBoundaryConditions(); // kqWallFunction is a zero-gradient BC
    //this->correctOmegaBoundaryConditions(); // special treatment for omega

    // Note: we need to update nut since we may have perturbed other turbulence vars
    // that affect the nut values
    this->updateNut(); 
    
    return;
}

void AdjointkOmega::correctOmegaBoundaryConditions()
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


void AdjointkOmega::calcTurbResiduals
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
    volScalarField divU(fvc::div(fvc::absolute(phi_/fvc::interpolate(rho_), U_)));

    tmp<volTensorField> tgradU = fvc::grad(U_);
    volScalarField G
    (
        "kOmega:G",
        nut_*(tgradU() && dev(twoSymm(tgradU())))
    );
    tgradU.clear();

    // Update omega and G at the wall. NOTE: this will not update the inter-proc face values  
    //omega_.boundaryFieldRef().updateCoeffs();
    
    this->correctOmegaBoundaryConditions();

    // Turbulent frequency equation
    tmp<fvScalarMatrix> omegaEqn
    (
        fvm::ddt(phase, rho, omega_)
      + fvm::div(phaseRhoPhi, omega_, divOmegaScheme)
      - fvm::laplacian(phase*rho*DomegaEff(), omega_)
     ==
        gamma_*phase*rho*G*omega_/k_
      - fvm::SuSp(((2.0/3.0)*gamma_)*phase*rho*divU, omega_)
      - fvm::Sp(beta_*phase*rho*omega_, omega_)
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


    // we need to bound before computing residuals
    // this will avoid having NaN residuals
    this->boundTurbVars(k_, kMin_);

    // Turbulent kinetic energy equation
    tmp<fvScalarMatrix> kEqn
    (
        fvm::ddt(phase, rho, k_)
      + fvm::div(phaseRhoPhi, k_, divKScheme)
      - fvm::laplacian(phase*rho*DkEff(), k_)
     ==
        phase*rho*G
      - fvm::SuSp((2.0/3.0)*phase*rho*divU, k_)
      - fvm::Sp(Cmu_*phase*rho*omega_, k_)
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

void AdjointkOmega::saveOmegaNearWall()
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

void AdjointkOmega::setOmegaNearWall()
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


void AdjointkOmega::correctAdjStateResidualTurbCon
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

void AdjointkOmega::setAdjStateResidualTurbCon
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

void AdjointkOmega::clearTurbVars()
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

void AdjointkOmega::writeTurbStates()
{
    nut_.write();
    k_.write();
    omega_.write();
}

} // End namespace Foam

// ************************************************************************* //
