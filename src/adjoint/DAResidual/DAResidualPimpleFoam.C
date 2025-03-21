/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

#include "DAResidualPimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualPimpleFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualPimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualPimpleFoam::DAResidualPimpleFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberVector(U, dimensionSet(0, 1, -2, 0, 0, 0, 0)),
      setResidualClassMemberScalar(p, dimensionSet(0, 0, -1, 0, 0, 0, 0)),
      setResidualClassMemberPhi(phi),
      TResPtr_(nullptr),
      fvSource_(const_cast<volVectorField&>(
          mesh_.thisDb().lookupObject<volVectorField>("fvSource"))),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      // create simpleControl
      pimple_(const_cast<fvMesh&>(mesh))
{
    // initialize fvSource
    const dictionary& allOptions = daOption.getAllOptions();
    if (allOptions.subDict("fvSource").toc().size() != 0)
    {
        hasFvSource_ = 1;
    }

    // check whether to include the temperature field
    hasTField_ = DAUtility::isFieldReadable(mesh, "T", "volScalarField");
    if (hasTField_)
    {

        TResPtr_.reset(new volScalarField(
            IOobject(
                "TRes",
                mesh.time().timeName(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE),
            mesh,
            dimensionedScalar("TRes", dimensionSet(0, 0, -1, 1, 0, 0, 0), 0.0),
            zeroGradientFvPatchField<scalar>::typeName));

        // initialize the Prandtl number from transportProperties
        IOdictionary transportProperties(
            IOobject(
                "transportProperties",
                mesh.time().constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false));
        Pr_ = readScalar(transportProperties.lookup("Pr"));
        Prt_ = readScalar(transportProperties.lookup("Prt"));
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualPimpleFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    URes_.clear();
    pRes_.clear();
    phiRes_.clear();

    if (hasTField_)
    {
        TResPtr_->clear();
    }
}

void DAResidualPimpleFoam::calcResiduals(const dictionary& options)
{
    /*
    Description:
        This is the function to compute residuals.
    
    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,U)

        p_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        URes_, pRes_, phiRes_: residual field variables
    */

    // We dont support MRF and fvOptions so all the related lines are commented
    // out for now

    // ******** U Residuals **********
    // copied and modified from UEqn.H

    word divUScheme = "div(phi,U)";

    label isPC = options.getLabel("isPC");

    if (isPC)
    {
        divUScheme = "div(pc)";
    }

    if (hasFvSource_)
    {
        DAFvSource& daFvSource(const_cast<DAFvSource&>(
            mesh_.thisDb().lookupObject<DAFvSource>("DAFvSource")));
        // update the actuator source term
        daFvSource.calcFvSource(fvSource_);
    }

    fvVectorMatrix UEqn(
        fvm::ddt(U_)
        + fvm::div(phi_, U_, divUScheme)
        + daTurb_.divDevReff(U_)
        - fvSource_);

    // NOTE: we need to call UEqn.relax here because it does some BC treatment, but we need to
    // force the relaxation factor to be 1.0 because the last pimple loop does not use relaxation
    UEqn.relax(1.0);

    URes_ = (UEqn & U_) + fvc::grad(p_);
    normalizeResiduals(URes);

    // ******** p Residuals **********
    // copied and modified from pEqn.H
    // NOTE manually set pRefCell and pRefValue
    label pRefCell = 0;
    scalar pRefValue = 0.0;

    volScalarField rAU(1.0 / UEqn.A());
    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, it may degrade the accuracy of derivatives
    // because constraining variables will create discontinuity. Here we have a option to use the old
    // implementation in OpenFOAM-3.0+ and before (no constraint for HbyA)
    autoPtr<volVectorField> HbyAPtr = nullptr;
    label useConstrainHbyA = daOption_.getOption<label>("useConstrainHbyA");
    if (useConstrainHbyA)
    {
        HbyAPtr.reset(new volVectorField(constrainHbyA(rAU * UEqn.H(), U_, p_)));
    }
    else
    {
        HbyAPtr.reset(new volVectorField("HbyA", U_));
        HbyAPtr() = rAU * UEqn.H();
    }
    volVectorField& HbyA = HbyAPtr();

    surfaceScalarField phiHbyA(
        "phiHbyA",
        fvc::flux(HbyA));

    if (p_.needReference())
    {
        adjustPhi(phiHbyA, U_, p_);
    }

    tmp<volScalarField> rAtU(rAU);

    if (pimple_.consistent())
    {
        rAtU = 1.0 / max(1.0 / rAU - UEqn.H1(), 0.1 / rAU);
        phiHbyA +=
            fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p_) * mesh_.magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(p_);
    }

    fvScalarMatrix pEqn(
        fvm::laplacian(rAtU(), p_)
        == fvc::div(phiHbyA));

    pEqn.setReference(pRefCell, pRefValue);

    pRes_ = pEqn & p_;
    normalizeResiduals(pRes);

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    phiRes_ = phiHbyA - pEqn.flux() - phi_;
    // need to normalize phiRes
    normalizePhiResiduals(phiRes);

    if (hasTField_)
    {
        volScalarField& alphat = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("alphat"));

        volScalarField& T = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("T"));

        volScalarField& TRes_ = TResPtr_();

        // ******** T Residuals **************
        volScalarField alphaEff("alphaEff", daTurb_.nu() / Pr_ + alphat);

        fvScalarMatrix TEqn(
            fvm::ddt(T)
            + fvm::div(phi_, T)
            - fvm::laplacian(alphaEff, T));

        TEqn.relax(1.0);

        TRes_ = TEqn & T;
        normalizeResiduals(TRes);
    }
}

void DAResidualPimpleFoam::calcPCMatWithFvMatrix(Mat PCMat)
{
    /* 
    Description:
        Calculate the diagonal block of the preconditioner matrix dRdWTPC using the fvMatrix
    */

    const labelUList& owner = mesh_.owner();
    const labelUList& neighbour = mesh_.neighbour();

    PetscScalar val;

    dictionary normStateDict = daOption_.getAllOptions().subDict("normalizeStates");
    wordList normResDict = daOption_.getOption<wordList>("normalizeResiduals");

    scalar UScaling = 1.0;
    if (normStateDict.found("U"))
    {
        UScaling = normStateDict.getScalar("U");
    }
    scalar UResScaling = 1.0;

    fvVectorMatrix UEqn(
        fvm::ddt(U_)
        + fvm::div(phi_, U_, "div(pc)")
        + daTurb_.divDevReff(U_)
        - fvSource_);

    UEqn.relax(1.0);

    // set diag
    forAll(U_, cellI)
    {
        if (normResDict.found("URes"))
        {
            UResScaling = mesh_.V()[cellI];
        }
        for (label i = 0; i < 3; i++)
        {
            PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("U", cellI, i);
            PetscInt colI = rowI;
            scalarField D = UEqn.D();
            scalar val1 = D[cellI] * UScaling / UResScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &rowI, 1, &colI, &val, INSERT_VALUES);
        }
    }

    // set lower/owner
    for (label faceI = 0; faceI < daIndex_.nLocalInternalFaces; faceI++)
    {
        label ownerCellI = owner[faceI];
        label neighbourCellI = neighbour[faceI];

        if (normResDict.found("URes"))
        {
            UResScaling = mesh_.V()[neighbourCellI];
        }

        for (label i = 0; i < 3; i++)
        {
            PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("U", neighbourCellI, i);
            PetscInt colI = daIndex_.getGlobalAdjointStateIndex("U", ownerCellI, i);
            scalar val1 = UEqn.lower()[faceI] * UScaling / UResScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
        }
    }

    // set upper/neighbour
    for (label faceI = 0; faceI < daIndex_.nLocalInternalFaces; faceI++)
    {
        label ownerCellI = owner[faceI];
        label neighbourCellI = neighbour[faceI];

        if (normResDict.found("URes"))
        {
            UResScaling = mesh_.V()[ownerCellI];
        }

        for (label i = 0; i < 3; i++)
        {
            PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("U", ownerCellI, i);
            PetscInt colI = daIndex_.getGlobalAdjointStateIndex("U", neighbourCellI, i);
            scalar val1 = UEqn.upper()[faceI] * UScaling / UResScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
        }
    }

    label pRefCell = 0;
    scalar pRefValue = 0.0;

    volScalarField rAU(1.0 / UEqn.A());
    autoPtr<volVectorField> HbyAPtr = nullptr;
    label useConstrainHbyA = daOption_.getOption<label>("useConstrainHbyA");
    if (useConstrainHbyA)
    {
        HbyAPtr.reset(new volVectorField(constrainHbyA(rAU * UEqn.H(), U_, p_)));
    }
    else
    {
        HbyAPtr.reset(new volVectorField("HbyA", U_));
        HbyAPtr() = rAU * UEqn.H();
    }
    volVectorField& HbyA = HbyAPtr();

    surfaceScalarField phiHbyA(
        "phiHbyA",
        fvc::flux(HbyA));

    if (p_.needReference())
    {
        adjustPhi(phiHbyA, U_, p_);
    }

    tmp<volScalarField> rAtU(rAU);

    if (pimple_.consistent())
    {
        rAtU = 1.0 / max(1.0 / rAU - UEqn.H1(), 0.1 / rAU);
        phiHbyA +=
            fvc::interpolate(rAtU() - rAU) * fvc::snGrad(p_) * mesh_.magSf();
        HbyA -= (rAU - rAtU()) * fvc::grad(p_);
    }

    fvScalarMatrix pEqn(
        fvm::laplacian(rAtU(), p_)
        == fvc::div(phiHbyA));

    pEqn.setReference(pRefCell, pRefValue);

    // ********* p
    scalar pScaling = 1.0;
    if (normStateDict.found("p"))
    {
        pScaling = normStateDict.getScalar("p");
    }
    scalar pResScaling = 1.0;
    // set diag
    forAll(p_, cellI)
    {
        if (normResDict.found("pRes"))
        {
            pResScaling = mesh_.V()[cellI];
        }

        PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("p", cellI);
        PetscInt colI = rowI;
        scalarField D = pEqn.D();
        scalar val1 = D[cellI] * pScaling / pResScaling;
        assignValueCheckAD(val, val1);
        MatSetValues(PCMat, 1, &rowI, 1, &colI, &val, INSERT_VALUES);
    }

    // set lower/owner
    for (label faceI = 0; faceI < daIndex_.nLocalInternalFaces; faceI++)
    {
        label ownerCellI = owner[faceI];
        label neighbourCellI = neighbour[faceI];

        if (normResDict.found("pRes"))
        {
            pResScaling = mesh_.V()[neighbourCellI];
        }

        PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("p", neighbourCellI);
        PetscInt colI = daIndex_.getGlobalAdjointStateIndex("p", ownerCellI);
        scalar val1 = pEqn.lower()[faceI] * pScaling / pResScaling;
        assignValueCheckAD(val, val1);
        MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
    }

    // set upper/neighbour
    for (label faceI = 0; faceI < daIndex_.nLocalInternalFaces; faceI++)
    {
        label ownerCellI = owner[faceI];
        label neighbourCellI = neighbour[faceI];

        if (normResDict.found("pRes"))
        {
            pResScaling = mesh_.V()[ownerCellI];
        }

        PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("p", ownerCellI);
        PetscInt colI = daIndex_.getGlobalAdjointStateIndex("p", neighbourCellI);
        scalar val1 = pEqn.upper()[faceI] * pScaling / pResScaling;
        assignValueCheckAD(val, val1);
        MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
    }

    if (hasTField_)
    {
        volScalarField& alphat = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("alphat"));

        volScalarField& T = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("T"));

        volScalarField alphaEff("alphaEff", daTurb_.nu() / Pr_ + alphat);

        fvScalarMatrix TEqn(
            fvm::ddt(T)
            + fvm::div(phi_, T)
            - fvm::laplacian(alphaEff, T));

        TEqn.relax(1.0);

        scalar TScaling = 1.0;
        if (normStateDict.found("T"))
        {
            TScaling = normStateDict.getScalar("T");
        }
        scalar TResScaling = 1.0;
        // set diag
        forAll(T, cellI)
        {
            if (normResDict.found("TRes"))
            {
                TResScaling = mesh_.V()[cellI];
            }

            PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("T", cellI);
            PetscInt colI = rowI;
            scalarField D = TEqn.D();
            scalar val1 = D[cellI] * TScaling / TResScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &rowI, 1, &colI, &val, INSERT_VALUES);
        }

        // set lower/owner
        for (label faceI = 0; faceI < daIndex_.nLocalInternalFaces; faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (normResDict.found("TRes"))
            {
                TResScaling = mesh_.V()[neighbourCellI];
            }

            PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("T", neighbourCellI);
            PetscInt colI = daIndex_.getGlobalAdjointStateIndex("T", ownerCellI);
            scalar val1 = TEqn.lower()[faceI] * TScaling / TResScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
        }

        // set upper/neighbour
        for (label faceI = 0; faceI < daIndex_.nLocalInternalFaces; faceI++)
        {
            label ownerCellI = owner[faceI];
            label neighbourCellI = neighbour[faceI];

            if (normResDict.found("TRes"))
            {
                TResScaling = mesh_.V()[ownerCellI];
            }

            PetscInt rowI = daIndex_.getGlobalAdjointStateIndex("T", ownerCellI);
            PetscInt colI = daIndex_.getGlobalAdjointStateIndex("T", neighbourCellI);
            scalar val1 = TEqn.upper()[faceI] * TScaling / TResScaling;
            assignValueCheckAD(val, val1);
            MatSetValues(PCMat, 1, &colI, 1, &rowI, &val, INSERT_VALUES);
        }
    }
}

void DAResidualPimpleFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables
    */

    // nothing to update for DAPimpleFoam
}

void DAResidualPimpleFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    U_.correctBoundaryConditions();
    p_.correctBoundaryConditions();
    if (hasTField_)
    {
        volScalarField& T = const_cast<volScalarField&>(
            mesh_.thisDb().lookupObject<volScalarField>("T"));
        T.correctBoundaryConditions();
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
