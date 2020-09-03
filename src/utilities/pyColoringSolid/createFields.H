// add dummy U, phi, and laminarTransport

fvMesh& mesh = meshPtr_();

volVectorField D(
    IOobject(
        "D",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedVector("D", dimensionSet(0, 1, 0, 0, 0, 0, 0), vector::zero),
    zeroGradientFvPatchField<vector>::typeName);
