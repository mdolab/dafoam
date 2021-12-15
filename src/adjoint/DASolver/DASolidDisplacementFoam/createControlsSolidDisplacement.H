const dictionary& stressControl = mesh.solutionDict().subDict("stressAnalysis");

stressControl.readEntry<Switch>("compactNormalStress", compactNormalStress_);

nCorr_ = stressControl.lookupOrDefault<label>("nCorrectors", 1);

convergenceTolerance_ = readScalar(stressControl.lookup("D"));
