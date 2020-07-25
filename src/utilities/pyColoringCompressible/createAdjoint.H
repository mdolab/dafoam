// adjoint

DAUtility daUtil;

DAOption daOption(mesh, pyOptions_);

// need to register thermo and turbulence to mesh 
// before initializing DATurbulenceModel
DARegDbFluidThermo regDbThermo(mesh, thermo);
DARegDbTurbulenceModelCompressible regDbTurbulence(mesh, turbulence());

word turbModelName = daOption.getOption<word>("turbulenceModel");
autoPtr<DATurbulenceModel> daTurbulenceModel(DATurbulenceModel::New(turbModelName, mesh, daOption));

DAModel daModel(mesh, daOption);

DAIndex daIndex(mesh, daOption, daModel);
