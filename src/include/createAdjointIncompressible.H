/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

// adjoint

// need to register laminarTransport and turbulence to mesh
DARegDbSinglePhaseTransportModel regDbTransport(mesh, laminarTransport);
DARegDbTurbulenceModelIncompressible regDbTurbulence(mesh, turbulencePtr_());

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

// calculate the initial (default) value for validate state func
this->getInitStateVals(initStateVals_);

// Uttam 20251107
if (daOptionPtr_().getOption<word>("multiMesh") == "yes")
{
    Info << "Multi-mesh mode: Creating COARSE adjoint infrastructure..." << endl;
    
    fvMesh& coarseMesh = targetPtr_();
    
    /// creating the MRF object
    this->MRFPtr_Coarse_.reset(new IOMRFZoneListDF(coarseMesh));
    
    // Create the new, coarse DAOption
    this->daOptionPtr_Coarse_.reset(new DAOption(coarseMesh, pyOptions_));
    
    // --- Create Coarse Transport/Turbulence Models ---
    const volVectorField& U_target = coarseMesh.lookupObject<volVectorField>("U");
    const surfaceScalarField& phi_target = coarseMesh.lookupObject<surfaceScalarField>("phi");
    
    
    laminarTransportPtr_Coarse_.reset(new singlePhaseTransportModel(U_target, phi_target));
    singlePhaseTransportModel& laminarTransport_Coarse = laminarTransportPtr_Coarse_();

    turbulencePtr_Coarse_.reset(incompressible::turbulenceModel::New(U_target, phi_target, laminarTransport_Coarse));
    
        
    // 2. Read the turbulence model name
    const word turbModelName(
        IOdictionary(
            IOobject(
                "turbulenceProperties",
                coarseMesh.time().constant(), // Use coarse mesh
                coarseMesh,                   // Use coarse mesh
                IOobject::MUST_READ,
                IOobject::NO_WRITE,
                false))
            .subDict("RAS")
            .lookup("RASModel"));
    
    // 4. Register the *new* coarse models with the coarse mesh
    DARegDbSinglePhaseTransportModel regDbTransport_Coarse(coarseMesh, laminarTransportPtr_Coarse_);
    DARegDbTurbulenceModelIncompressible regDbTurbulence_Coarse(coarseMesh, turbulencePtr_Coarse_());
    // Note: The DATurbulenceModel::New will handle its own registration
    
    // --- Create and assign to the COARSE pointers ---
    this->daTurbulenceModelPtr_Coarse_.reset(DATurbulenceModel::New(turbModelName, coarseMesh, this->daOptionPtr_Coarse_()));
    this->daModelPtr_Coarse_.reset(new DAModel(coarseMesh, this->daOptionPtr_Coarse_()));
    this->daStateInfoPtr_Coarse_.reset(DAStateInfo::New(solverName, coarseMesh, this->daOptionPtr_Coarse_(), this->daModelPtr_Coarse_()));
    Info <<" Getting state information... " << endl;
    stateInfo_Coarse = daStateInfoPtr_Coarse_->getStateInfo();
    this->daIndexPtr_Coarse_.reset(new DAIndex(coarseMesh, this->daOptionPtr_Coarse_(), this->daModelPtr_Coarse_()));
    this->daFieldPtr_Coarse_.reset(new DAField(coarseMesh, this->daOptionPtr_Coarse_(), this->daModelPtr_Coarse_(), this->daIndexPtr_Coarse_()));
    this->daResidualPtr_Coarse_.reset(DAResidual::New(solverName, coarseMesh, this->daOptionPtr_Coarse_(), this->daModelPtr_Coarse_(), this->daIndexPtr_Coarse_()));
    Info << "Defining the linear equation pointer for the COARSE mesh..." << endl ;
    this->daLinearEqnPtr_Coarse_.reset(new DALinearEqn(coarseMesh, daOptionPtr_()));
    
    
    Info << "COARSE Adjoint Infrastructure Created. Printing stats:" << endl;
    this->daIndexPtr_Coarse_->printIndices(); // <-- This prints the COARSE MESH stats
}
