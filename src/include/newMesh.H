Time& runTime = runTimePtr_();
const word newRegion = "target";

// Create the fvMesh object using the updated path to the polyMesh directory
targetPtr_.reset(
    new fvMesh(
        IOobject(
            newRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    )
);
