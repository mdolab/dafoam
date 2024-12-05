// adjoint

DAOption daOption(mesh, pyOptions_);

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
