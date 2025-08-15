
scalar alphaCoNum = 0.0;
scalar meanAlphaCoNum = 0.0;

if (mesh.nInternalFaces())
{
    scalarField sumPhi(
        mixture.nearInterface()().primitiveField()
        * fvc::surfaceSum(mag(phi))().primitiveField());

    alphaCoNum = 0.5 * gMax(sumPhi / mesh.V().field()) * runTime.deltaTValue();

    meanAlphaCoNum =
        0.5 * (gSum(sumPhi) / gSum(mesh.V().field())) * runTime.deltaTValue();
}

Info << "Interface Courant Number mean: " << meanAlphaCoNum
     << " max: " << alphaCoNum << endl;

// ************************************************************************* //
