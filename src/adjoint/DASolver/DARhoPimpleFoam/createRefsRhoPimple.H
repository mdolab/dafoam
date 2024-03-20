Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
pimpleControlDF& pimple = pimplePtr_();
fluidThermo& thermo = pThermoPtr_();
volScalarField& p = thermo.p();
volScalarField& rho = rhoPtr_();
volVectorField& U = UPtr_();
surfaceScalarField& phi = phiPtr_();
volScalarField& dpdt = dpdtPtr_();
volScalarField& K = KPtr_();
scalar& cumulativeContErr = cumulativeContErr_;
const volScalarField& psi = thermo.psi();
const dictionary& allOptions = daOptionPtr_->getAllOptions();
volVectorField& fvSource = fvSourcePtr_();
volScalarField& fvSourceEnergy = fvSourceEnergyPtr_();
