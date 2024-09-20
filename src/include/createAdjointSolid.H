/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

\*---------------------------------------------------------------------------*/

// adjoint

daModelPtr_.reset(new DAModel(mesh, daOptionPtr_()));

daRegressionPtr_.reset(new DARegression(mesh, daOptionPtr_(), daModelPtr_()));

word solverName = daOptionPtr_->getOption<word>("solverName");
daStateInfoPtr_.reset(DAStateInfo::New(solverName, mesh, daOptionPtr_(), daModelPtr_()));
stateInfo_ = daStateInfoPtr_->getStateInfo();

daIndexPtr_.reset(new DAIndex(mesh, daOptionPtr_(), daModelPtr_()));

daIndexPtr_->printIndices();

daFieldPtr_.reset(new DAField(mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));

daResidualPtr_.reset(DAResidual::New(solverName, mesh, daOptionPtr_(), daModelPtr_(), daIndexPtr_()));

// calculate the initial (default) value for validate state func
this->getInitStateVals(initStateVals_);
