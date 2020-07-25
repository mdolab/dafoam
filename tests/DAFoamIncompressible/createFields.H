Info << "Reading field p\n"
     << endl;
volScalarField p(
    IOobject(
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

Info << "Reading field U\n"
     << endl;
volVectorField U(
    IOobject(
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE),
    mesh);

#include "createPhi.H"

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence(
    incompressible::turbulenceModel::New(U, phi, laminarTransport));

// need to register laminarTransport and turbulence to mesh
DARegDbSinglePhaseTransportModel regDbTransport(mesh, laminarTransport);
DARegDbTurbulenceModelIncompressible regDbTurbulence(mesh, turbulence());

DAOption daOption(mesh, pyDict);

word turbModelName = daOption.getOption<word>("turbulenceModel");
autoPtr<DATurbulenceModel> daTurbulenceModel(DATurbulenceModel::New(turbModelName, mesh, daOption));

DAModel daModel(mesh, daOption);

word solverName = daOption.getOption<word>("solverName");
autoPtr<DAStateInfo> daStateInfo(
    DAStateInfo::New(solverName, mesh, daOption, daModel));

DAIndex daIndex(mesh, daOption, daModel);

DAField daField(mesh, daOption, daModel, daIndex);