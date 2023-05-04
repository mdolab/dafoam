Time& runTime = runTimePtr_();
const word regionName1 = "mesh1";

// Create the fvMesh object using the updated path to the polyMesh directory

meshPtr1_.reset(
    new fvMesh(
    IOobject(
            regionName1,
            runTime.timeName(),
            runTimePtr_(),
            IOobject::MUST_READ)));


