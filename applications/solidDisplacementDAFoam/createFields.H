#include "readMechanicalProperties.H"
#include "readThermalProperties.H"

Info<< "Reading field D\n" << endl;
volVectorField D
(
    IOobject
    (
        "D",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


autoPtr<volScalarField> Tptr(nullptr);

if (thermalStress)
{
    Info<< "Reading field T\n" << endl;
    Tptr.reset
    (
        new volScalarField
        (
            IOobject
            (
                "T",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        )
    );
}


Info<< "Calculating stress field sigmaD\n" << endl;
volSymmTensorField sigmaD
(
    IOobject
    (
        "sigmaD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mu*twoSymm(fvc::grad(D)) + lambda*(I*tr(fvc::grad(D)))
);

// gradD is used in the tractionDisplacement BC
volTensorField gradD
(
    IOobject
    (
        "gradD",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    fvc::grad(D)
);

Info<< "Calculating explicit part of div(sigma) divSigmaExp\n" << endl;
volVectorField divSigmaExp
(
    IOobject
    (
        "divSigmaExp",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    fvc::div(sigmaD)
);

if (compactNormalStress)
{
    divSigmaExp -= fvc::laplacian(2*mu + lambda, D, "laplacian(DD,D)");
}
else
{
    divSigmaExp -= fvc::div((2*mu + lambda)*fvc::grad(D), "div(sigmaD)");
}

mesh.setFluxRequired(D.name());


// adjoint
AdjointIO adjIO(mesh);
// Set the BC based on the flowBCs key in adjointDict
// Since this will changed the field BC, we should only call it for flow simulation 
if (!adjIO.solveAdjoint and adjIO.setFlowBCs) adjIO.setFlowBoundaryConditions(); 

// NOTE: we need to update D boundary conditions iteratively
// this is because tractionDisplacement BC is dependent on gradD, while gradD
// is dependent on the D bc values.
if(adjIO.solveAdjoint)
{
    for(label i=0;i<adjIO.tractionBCMaxIter;i++)
    {
        D.correctBoundaryConditions();
        gradD=fvc::grad(D);
    }
}

autoPtr<AdjointSolverRegistry> adjReg
(
    AdjointSolverRegistry::New(mesh)
);

volVectorField centrifugalForce
(
    IOobject
    (
        "centrifugalForce",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("centrifugalForce",dimensionSet(0,1,-2,0,0,0,0),vector::zero),
    zeroGradientFvPatchVectorField::typeName
);
forAll(centrifugalForce,cellI)
{
    centrifugalForce[cellI] = ( adjIO.rotRad ^ (adjIO.rotRad ^ (mesh.C()[cellI] - adjIO.CofR)) );
}

// add dummy U, phi, nut, and laminarTransport
volVectorField U
(
    IOobject
    (
        "U",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("U",dimensionSet(0,1,-1,0,0,0,0),vector::zero),
    zeroGradientFvPatchVectorField::typeName
);
#include "createPhi.H"
// add dummy nut 
volScalarField nut
(
    IOobject
    (
        "nut",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar("nut",dimensionSet(0,2,-1,0,0,0,0),0.0),
    zeroGradientFvPatchScalarField::typeName
);

singlePhaseTransportModel laminarTransport(U, phi);

nearWallDist d(mesh);
autoPtr<AdjointRASModel> adjRAS
(
    AdjointRASModel::New(mesh,adjIO,d,laminarTransport)
);

AdjointIndexing adjIdx(mesh,adjIO,adjReg(),adjRAS());

autoPtr<AdjointJacobianConnectivity> adjCon
(
    AdjointJacobianConnectivity::New(mesh,adjIO,adjReg(),adjRAS(),adjIdx)
);

AdjointObjectiveFunction adjObj(mesh,adjIO,adjRAS(),adjIdx,adjCon());
