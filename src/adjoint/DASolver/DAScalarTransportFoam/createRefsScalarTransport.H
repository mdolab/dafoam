Time& runTime = runTimePtr_();
// reset time to 0
runTime.setTime(0.0, 0);
fvMesh& mesh = meshPtr_();
volScalarField& T = TPtr_();
surfaceScalarField& phi = phiPtr_();
dimensionedScalar& DT = DTPtr_();

// create a file to store the objective values
this->initializeObjFuncHistFilePtr("objFuncTimeSeries");

this->setDAObjFuncList();
