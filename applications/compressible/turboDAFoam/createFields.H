Info<< "Reading thermophysical properties\n" << endl;

autoPtr<fluidThermo> pThermo
(
    fluidThermo::New(mesh)
);
fluidThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

volScalarField& p = thermo.p();

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    thermo.rho()
);

volScalarField T
(
    IOobject
    (
        "T",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    thermo.T()
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

volVectorField URel
(
    IOobject
    (
        "URel",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    U
);

#include "compressibleCreatePhi.H"

pressureControl pressureControl(p, rho, simple.dict());

mesh.setFluxRequired(p.name());

Info<< "Creating turbulence model\n" << endl;
autoPtr<compressible::turbulenceModel> turbulence
(
    compressible::turbulenceModel::New
    (
        rho,
        U,
        phi,
        thermo
    )
);

dimensionedScalar initialMass = fvc::domainIntegrate(rho);

#include "createMRF.H"

// adjoint
AdjointIO adjIO(mesh);
// Set the BC based on the flowBCs key in adjointDict
// Since this will changed the field BC, we should only call it for flow simulation 
if (!adjIO.solveAdjoint and adjIO.setFlowBCs) adjIO.setFlowBoundaryConditions(); 

autoPtr<AdjointSolverRegistry> adjReg
(
    AdjointSolverRegistry::New(mesh)
);

nearWallDist& d = const_cast<nearWallDist&>(turbulence->y());
autoPtr<AdjointRASModel> adjRAS
(
    AdjointRASModel::New(mesh,adjIO,d,thermo)
);

AdjointIndexing adjIdx(mesh,adjIO,adjReg(),adjRAS());

autoPtr<AdjointJacobianConnectivity> adjCon
(
    AdjointJacobianConnectivity::New(mesh,adjIO,adjReg(),adjRAS(),adjIdx)
);

AdjointObjectiveFunction adjObj(mesh,adjIO,adjRAS(),adjIdx,adjCon());
