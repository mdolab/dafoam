/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

Info << "Create mesh for time = " << runTimePtr_().timeName() << endl;

word regionName = fvMesh::defaultRegion;
meshPtr_.reset(
    new fvMesh(
        IOobject(
            regionName,
            runTimePtr_().timeName(),
            runTimePtr_(),
            IOobject::MUST_READ)));

