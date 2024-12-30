/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

// adjoint

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
daTurbulenceModelPtr_.reset(DATurbulenceModel::New(turbModelName, mesh, daOptionPtr_()));

daModelPtr_.reset(new DAModel(mesh, daOptionPtr_()));

daRegressionPtr_.reset(new DARegression(mesh, daOptionPtr_(), daModelPtr_()));

word solverName = daOptionPtr_->getOption<word>("solverName");
daStateInfoPtr_.reset(DAStateInfo::New(solverName, mesh, daOptionPtr_(), daModelPtr_()));
stateInfo_ = daStateInfoPtr_->getStateInfo();

daIndexPtr_.reset(new DAIndex(mesh, daOptionPtr_(), daModelPtr_()));

daIndexPtr_->printIndices();

daFieldPtr_.reset(new DAField(mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));

daResidualPtr_.reset(DAResidual::New(solverName, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));

daGlobalVarPtr_.reset(new DAGlobalVar(mesh));

// calculate the initial (default) value for validate state func
this->getInitStateVals(initStateVals_);

// initialize the mean states if needed
this->initMeanStates();
