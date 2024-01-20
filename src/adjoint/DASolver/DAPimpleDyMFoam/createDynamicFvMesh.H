Info<< "Create mesh for time = "
    << runTime.timeName() << nl << endl;

autoPtr<dynamicFvMesh> meshPtr(dynamicFvMesh::New(args, runTime));

dynamicFvMesh& mesh = meshPtr();
