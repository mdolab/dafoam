Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
volScalarField& T = TPtr_();
volScalarField& kappa = kappaPtr_();
this->correctKappa();
volScalarField& fvSource = fvSourcePtr_();
