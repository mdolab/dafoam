Info<< "Reading thermophysical properties\n" << endl;

autoPtr<fluidThermo> pThermo(fluidThermo::New(mesh));
fluidThermo& thermo = pThermo();
thermo.validate(args.executable(), "h", "e");

volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    thermo.rho()
);

volScalarField& p = thermo.p();

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

#include "compressibleCreatePhi.H"

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


#include "readGravitationalAcceleration.H"
#include "readhRef.H"
#include "gh.H"


Info<< "Reading field p_rgh\n" << endl;
volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

// Force p_rgh to be consistent with p
p_rgh = p - rho*gh;

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell
(
    p,
    p_rgh,
    simple.dict(),
    pRefCell,
    pRefValue
);

mesh.setFluxRequired(p_rgh.name());

dimensionedScalar initialMass = fvc::domainIntegrate(rho);
dimensionedScalar totalVolume = sum(mesh.V());

#include "createMRF.H"
#include "createRadiationModel.H"

dimensionedScalar rhoMax
(
    dimensionedScalar::lookupOrDefault
    (
        "rhoMax",
        simple.dict(),
        dimDensity,
        GREAT
    )
);

dimensionedScalar rhoMin
(
    dimensionedScalar::lookupOrDefault
    (
        "rhoMin",
        simple.dict(),
        dimDensity,
        0
    )
);

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

