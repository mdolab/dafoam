Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
simpleControl& simple = simplePtr_();
volScalarField& p = pPtr_();
volVectorField& U = UPtr_();
surfaceScalarField& phi = phiPtr_();
volScalarField& alphaPorosity = alphaPorosityPtr_();
singlePhaseTransportModel& laminarTransport = laminarTransportPtr_();
scalar& cumulativeContErr = cumulativeContErr_;
label& pRefCell = pRefCell_;
scalar& pRefValue = pRefValue_;
volVectorField& fvSource = fvSourcePtr_();
IOMRFZoneListDF& MRF = MRFPtr_();
const dictionary& allOptions = daOptionPtr_->getAllOptions();
