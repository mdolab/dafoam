Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
volScalarField& rho = rhoPtr_();
volScalarField& mu = muPtr_();
volScalarField& lambda = lambdaPtr_();
volVectorField& D = DPtr_();
volTensorField& gradD = gradDPtr_();
