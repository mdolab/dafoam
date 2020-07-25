// adjoint

daOptionPtr_.reset(new DAOption(mesh, pyOptions_));

// need to register thermo and turbulence to mesh 
// before initializing DATurbulenceModel
DARegDbFluidThermo regDbThermo(mesh, thermo);
DARegDbTurbulenceModelCompressible regDbTurbulence(mesh, turbulencePtr_());

word turbModelName = daOptionPtr_->getOption<word>("turbulenceModel");
daTurbulenceModelPtr_.reset(DATurbulenceModel::New(turbModelName, mesh, daOptionPtr_()));

daModelPtr_.reset(new DAModel(mesh, daOptionPtr_()));

word solverName = daOptionPtr_->getOption<word>("solverName");
daStateInfoPtr_.reset(DAStateInfo::New(solverName, mesh, daOptionPtr_(), daModelPtr_()));
stateInfo_ = daStateInfoPtr_->getStateInfo();

daIndexPtr_.reset(new DAIndex(mesh, daOptionPtr_(), daModelPtr_()));

daIndexPtr_->printIndices();

daFieldPtr_.reset(new DAField(mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));

daResidualPtr_.reset(DAResidual::New(solverName, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));
