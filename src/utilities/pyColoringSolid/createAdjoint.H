// adjoint

fvMesh& mesh = meshPtr_();

DAUtility daUtil;

DAOption daOption(mesh, pyOptions_);

DAModel daModel(mesh, daOption);

DAIndex daIndex(mesh, daOption, daModel);
