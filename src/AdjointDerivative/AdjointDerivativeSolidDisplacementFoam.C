/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointDerivativeSolidDisplacementFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDerivativeSolidDisplacementFoam, 0);
addToRunTimeSelectionTable(AdjointDerivative, AdjointDerivativeSolidDisplacementFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDerivativeSolidDisplacementFoam::AdjointDerivativeSolidDisplacementFoam
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
    setResidualClassMemberVector(D,dimensionSet(0,1,-2,0,0,0,0)),
    // these are intermediate variables or objects
    gradD_
    (
        const_cast<volTensorField&>
        (
            db_.lookupObject<volTensorField>("gradD")
        )
    ),
    sigmaD_
    (
        const_cast<volSymmTensorField&>
        (
            db_.lookupObject<volSymmTensorField>("sigmaD")
        )
    ),
    divSigmaExp_
    (
        const_cast<volVectorField&>
        (
            db_.lookupObject<volVectorField>("divSigmaExp")
        )
    ),
    lambda_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("solid:lambda")
        )
    ),
    mu_
    (
        const_cast<volScalarField&>
        (
            db_.lookupObject<volScalarField>("solid:mu")
        )
    ),
    centriF_
    (
        IOobject
        (
            "centriF",
            mesh_.time().timeName(),
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedVector("centriF",dimensionSet(0,1,-2,0,0,0,0),vector::zero),
        zeroGradientFvPatchVectorField::typeName
    )

{
    this->copyStates("Var2Ref"); // copy states to statesRef

    IOdictionary thermalProperties
    (
        IOobject
        (
            "thermalProperties",
            mesh_.time().constant(),
            mesh_,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    Switch thermalStress( thermalProperties.lookup("thermalStress"));
    if(thermalStress) 
    {
        FatalErrorIn("")<<"thermalStress=true not supported"<< abort(FatalError);
    }

    const dictionary& stressControl = mesh_.solutionDict().subDict("stressAnalysis");

    Switch compactNormalStress(stressControl.lookup("compactNormalStress"));

    if(!compactNormalStress) 
    {
        FatalErrorIn("")<<"compactNormalStress=false not supported"<< abort(FatalError);
    }

    isTractionDisplacementBC_=0;
    forAll(D_.boundaryField(),patchI)
    {
        if (D_.boundaryField()[patchI].type()=="tractionDisplacement")
        {
            isTractionDisplacementBC_=1;
        }
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void AdjointDerivativeSolidDisplacementFoam::calcResiduals
(
    const label isRef,
    const label isPC,
    const word fvMatrixName,
    const label updatePhi
)
{
    
    fvVectorMatrix DEqn
    (
        fvm::d2dt2(D_)
      ==
        fvm::laplacian(2*mu_ + lambda_, D_, "laplacian(DD,D)")
      + divSigmaExp_
      + centriF_
    );

    if(isRef) DResRef_  = DEqn&D_;
    else DRes_  = DEqn&D_;
    // need to normalize Res. 
    normalizeResiduals(DRes);   
    
    return;

}

void AdjointDerivativeSolidDisplacementFoam::updateDAndGradD()
{
    // NOTE: we need to update D boundary conditions iteratively if tractionDisplacement BC is used
    // this is because tractionDisplacement BC is dependent on gradD, while gradD
    // is dependent on the D bc values.
    if(adjIO_.solveAdjoint) // this will be called after doing perturbStates
    {
        if (isTractionDisplacementBC_)
        {
            for(label i=0;i<adjIO_.tractionBCMaxIter;i++)
            {
                D_.correctBoundaryConditions();
                gradD_=fvc::grad(D_);
            }
        }
        else
        {
            D_.correctBoundaryConditions();
            gradD_=fvc::grad(D_);
        }
    }
    else
    {
        D_.correctBoundaryConditions();
        gradD_=fvc::grad(D_);
    }
}

void AdjointDerivativeSolidDisplacementFoam::updateIntermediateVariables()
{
    
    this->updateDAndGradD();

    sigmaD_ = mu_*twoSymm(gradD_) + (lambda_*I)*tr(gradD_);

    divSigmaExp_ = fvc::div(sigmaD_ - (2*mu_ + lambda_)*gradD_,"div(sigmaD)");

    // update centrifugal force
    forAll(centriF_,cellI)
    {
        centriF_[cellI] = ( adjIO_.rotRad ^ (adjIO_.rotRad ^ (mesh_.C()[cellI] - adjIO_.CofR)) );
    }
}


} // End namespace Foam

// ************************************************************************* //