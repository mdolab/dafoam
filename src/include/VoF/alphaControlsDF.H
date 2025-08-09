const dictionary& alphaControls = mesh.solverDict(alpha1.name());

label nAlphaCorr(alphaControls.get<label>("nAlphaCorr"));

label nAlphaSubCycles(alphaControls.get<label>("nAlphaSubCycles"));

// Isotropic compression coefficient
scalar icAlpha(
    alphaControls.lookupOrDefault<scalar>("icAlpha", 0));

// Shear compression coefficient
scalar scAlpha(
    alphaControls.lookupOrDefault<scalar>("scAlpha", 0));
