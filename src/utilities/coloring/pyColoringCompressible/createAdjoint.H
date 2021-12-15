// adjoint

DAUtility daUtil;

DAOption daOption(mesh, pyOptions_);

// need to register thermo and turbulence to mesh 
// before initializing DATurbulenceModel
DARegDbFluidThermo regDbThermo(mesh, thermo);
DARegDbTurbulenceModelCompressible regDbTurbulence(mesh, turbulence());

// read the RAS model from constant/turbulenceProperties
const word turbModelName(
    IOdictionary(
        IOobject(
            "turbulenceProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false))
        .subDict("RAS")
        .lookup("RASModel"));
autoPtr<DATurbulenceModel> daTurbulenceModel(DATurbulenceModel::New(turbModelName, mesh, daOption));

DAModel daModel(mesh, daOption);

DAIndex daIndex(mesh, daOption, daModel);
