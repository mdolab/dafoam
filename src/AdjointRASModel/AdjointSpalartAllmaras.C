/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointSpalartAllmaras.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointSpalartAllmaras, 0);
addToRunTimeSelectionTable(AdjointRASModel, AdjointSpalartAllmaras, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointSpalartAllmaras::AdjointSpalartAllmaras
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
    // SA parameters
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
    Cs_
    (
        dimensioned<scalar>::lookupOrAddToDict
        (
            "Cs",
            this->coeffDict_,
            0.3
        )
    ),

    // Augmented variables
    nuTilda_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("nuTilda")
        )
    ),
    nuTildaRes_
    (
        IOobject
        (
            "nuTildaRes",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
#ifdef CompressibleFlow
        dimensionedScalar("nuTildaRes",dimensionSet(1,-1,-2,0,0,0,0),0.0),
#endif
#ifdef IncompressibleFlow
        dimensionedScalar("nuTildaRes",dimensionSet(0,2,-2,0,0,0,0),0.0),
#endif
        zeroGradientFvPatchScalarField::typeName
    ),
    nuTildaResRef_
    (
        IOobject
        (
            "nuTildaResRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        nuTildaRes_
    ),
    nuTildaResPartDeriv_
    (
        IOobject
        (
            "nuTildaResPartDeriv",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        nuTildaRes_
    ),
    nuTildaRef_
    (
        IOobject
        (
            "nuTildaRef",
            mesh.time().timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        nuTilda_
    ),
    y_(db_.lookupObject<volScalarField>("yWall"))
    
{
    // add the turbulence state variables, this will be used in AdjointIndexing
    this->turbStates.append("nuTilda"); 
    this->copyTurbStates("Var2Ref"); // copy turbVars to turbVarsRef
    //this->calcTurbResiduals(1,0); 
    //Info<<nuTildaResidualRef_<<endl;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


// SA member functions
tmp<volScalarField> AdjointSpalartAllmaras::chi() const
{
    return nuTilda_/this->getNu();
}

tmp<volScalarField> AdjointSpalartAllmaras::fv1
(
    const volScalarField& chi
) const
{
    const volScalarField chi3(pow3(chi));
    return chi3/(chi3 + pow3(Cv1_));
}

tmp<volScalarField> AdjointSpalartAllmaras::fv2
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    return 1.0 - chi/(1.0 + chi*fv1);
}

tmp<volScalarField> AdjointSpalartAllmaras::Stilda
(
    const volScalarField& chi,
    const volScalarField& fv1
) const
{
    volScalarField Omega(::sqrt(2.0)*mag(skew(fvc::grad(U_))));
    
    return
    (
        max
        (
            Omega
          + fv2(chi, fv1)*nuTilda_/sqr(kappa_*y_),
            Cs_*Omega
        )
    );
}

tmp<volScalarField> AdjointSpalartAllmaras::fw
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

tmp<volScalarField> AdjointSpalartAllmaras::DnuTildaEff() const
{
    return tmp<volScalarField>
    (
        new volScalarField("DnuTildaEff", (nuTilda_ + this->getNu() )/sigmaNut_)
    );
}


// Augmented functions
void AdjointSpalartAllmaras::updateNut()
{
    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));
    nut_ = nuTilda_ * fv1;
    
    nut_.correctBoundaryConditions();
    
    return;
}


void AdjointSpalartAllmaras::copyTurbStates(const word option)
{
    // "Ref2Var", assign ref states to states
    // "Var2Ref", assign states to ref states
    if(option=="Ref2Var")
    {
        nuTilda_=nuTildaRef_;
        this->correctTurbBoundaryConditions();
    }
    else if (option=="Var2Ref")
    {
        nuTildaRef_ = nuTilda_;
    }
    else
    {
        FatalErrorIn("option not valid! Should be either Var2Ref or Ref2Var")
        << abort(FatalError);
    }

    this->updateNut();
    
    return;
    
}


void AdjointSpalartAllmaras::correctTurbBoundaryConditions()
{
    // correct the BCs for the perturbed fields
    nuTilda_.correctBoundaryConditions();
    
    // Note: we need to update nut and its BC since we may have perturbed other turbulence vars
    // that affect the nut values
    this->updateNut(); 

    return;
}


void AdjointSpalartAllmaras::calcTurbResiduals
(  
    const label isRef,
    const label isPC,
    const word fvMatrixName
)
{    
    // Copy and modify based on the "correct" function
    
    word divNuTildaScheme="div(phi,nuTilda)";
    if(isPC) divNuTildaScheme="div(pc)";
    
    // this portion is copied from the correct() function
    //if (!this->turbulence_)
    //{
    //    return;
    //}

    // Local references
    //const alphaField& alpha = this->alpha_;
    //const rhoField& rho = this->rho_;
    //const surfaceScalarField& alphaRhoPhi = this->alphaRhoPhi_;
    //fv::options& fvOptions(fv::options::New(mesh_));

    // we need to bound nuTilda before computing residuals
    // this will avoid having NaN residuals
    this->boundTurbVars(nuTilda_, dimensionedScalar("0", nuTilda_.dimensions(), 0.0));
    
    volScalarField& phase = phase_;
    volScalarField& rho = rho_;
    surfaceScalarField& phaseRhoPhi = phaseRhoPhi_;
    
    //eddyViscosity<RASModelAugmented<BasicTurbulenceModel> >::correct();

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    scalar prodSwitch=1.0;
    if(adjIO_.delTurbProd4PCMat and isPC) prodSwitch=0.0;

    tmp<fvScalarMatrix> nuTildaEqn
    (
        fvm::ddt(phase, rho, nuTilda_)
      + fvm::div(phaseRhoPhi, nuTilda_, divNuTildaScheme)
      - fvm::laplacian(phase*rho*DnuTildaEff(), nuTilda_)
      - Cb2_/sigmaNut_*phase*rho*magSqr(fvc::grad(nuTilda_))
     ==
        prodSwitch*Cb1_*phase*rho*Stilda*nuTilda_
      - fvm::Sp(Cw1_*phase*rho*fw(Stilda)*nuTilda_/sqr(y_), nuTilda_)
    //  + fvOptions(phase, rho, nuTilda_)
    );

    nuTildaEqn.ref().relax();
    //fvOptions.constrain(nuTildaEqn.ref());
    //solve(nuTildaEqn);
    //fvOptions.correct(nuTilda_);
    // ************ Note **************
    // SA model in OF-v1812 does not support read nuTildaMin from dict
    // it always bounds nuTilda to 0
    //bound(nuTilda_, dimensionedScalar("0", nuTilda_.dimensions(), 0.0));
    //nuTilda_.correctBoundaryConditions();

    //correctNut(fv1);

    if(fvMatrixName=="nuTildaEqn") 
    {
        fvMatrixDiag.clear();
        fvMatrixLower.clear();
        fvMatrixUpper.clear();
        fvMatrixDiag = nuTildaEqn.ref().diag();
        fvMatrixLower = nuTildaEqn.ref().lower();
        fvMatrixUpper = nuTildaEqn.ref().upper();
    }
    
    // calculate residuals
    if(isRef)
    {
        nuTildaResRef_ = nuTildaEqn.ref()&nuTilda_;
        // need to normalize Res. NOTE: nuTildaRes is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("nuTildaRes",adjIO_.normalizeResiduals)) 
        {
            forAll(nuTildaResRef_,cellI)
            {
                nuTildaResRef_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }
    else
    {
        nuTildaRes_ = nuTildaEqn.ref()&nuTilda_;
        // need to normalize Res. NOTE: nuTildaRes is normalized by its cell volume by default!
        if(!adjIO_.isInList<word>("nuTildaRes",adjIO_.normalizeResiduals)) 
        {
            forAll(nuTildaRes_,cellI)
            {
                nuTildaRes_[cellI] *= mesh_.V()[cellI];  
            }
        }
    }

    return;
}


void AdjointSpalartAllmaras::correctAdjStateResidualTurbCon
(
    List< List<word> >& adjStateResidualConInfo
)
{
    // For SA model just replace nut with nuTilda
    forAll(adjStateResidualConInfo,idxI)
    {
        forAll(adjStateResidualConInfo[idxI],idxJ)
        {
            word conStateName = adjStateResidualConInfo[idxI][idxJ];
            if( conStateName == "nut" ) adjStateResidualConInfo[idxI][idxJ]="nuTilda";
        }
    }
}

void AdjointSpalartAllmaras::setAdjStateResidualTurbCon
(
    HashTable< List< List<word> > >& adjStateResidualConInfo
)
{

    word pName=adjIO_.getPName();

#ifdef IncompressibleFlow
    adjStateResidualConInfo.set
    (
        "nuTildaRes",
        {
            {"U","nuTilda","phi"}, // lv0
            {"U","nuTilda"},       // lv1
            {"nuTilda"}            // lv2
        }
    );
#endif

#ifdef CompressibleFlow
    adjStateResidualConInfo.set
    (
        "nuTildaRes",
        {
            {"U","T",pName,"nuTilda","phi"}, // lv0
            {"U","T",pName,"nuTilda"},       // lv1
            {"T",pName,"nuTilda"}            // lv2
        }
    );
#endif 
}

void AdjointSpalartAllmaras::clearTurbVars()
{
    // clear any variables that are not needed for solveAdjoint
    //nuTildaRef_.clear();
    nuTildaRes_.clear();
    nuTildaResRef_.clear();
    nuTildaResPartDeriv_.clear();
}

void AdjointSpalartAllmaras::writeTurbStates()
{
    nut_.write();
    nuTilda_.write();
}

} // End namespace Foam

// ************************************************************************* //
