Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
simpleControl& simple = simplePtr_();
fluidThermo& thermo = pThermoPtr_();
volScalarField& p = thermo.p();
volScalarField& rho = rhoPtr_();
volVectorField& U = UPtr_();
volVectorField& URel = URelPtr_();
surfaceScalarField& phi = phiPtr_();
scalar& cumulativeContErr = cumulativeContErr_;
pressureControl& pressureControl = pressureControlPtr_();
initialMass_ = fvc::domainIntegrate(rho);
const volScalarField& psi = thermo.psi();
const dictionary& allOptions = daOptionPtr_->getAllOptions();
IOMRFZoneListDF& MRF = MRFPtr_();
