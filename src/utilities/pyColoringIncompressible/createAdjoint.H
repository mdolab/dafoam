// adjoint

DAUtility daUtil;

DAOption daOption(mesh, pyOptions_);

// need to register turbulence to mesh
DARegDbSinglePhaseTransportModel regDbTransport(mesh, laminarTransport);
DARegDbTurbulenceModelIncompressible regDbTurbulence(mesh, turbulence());

word turbModelName = daOption.getOption<word>("turbulenceModel");
autoPtr<DATurbulenceModel> daTurbulenceModel(DATurbulenceModel::New(turbModelName, mesh, daOption));

DAModel daModel(mesh, daOption);

DAIndex daIndex(mesh, daOption, daModel);
