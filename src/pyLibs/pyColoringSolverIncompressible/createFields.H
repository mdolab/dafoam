// add dummy U, phi, and laminarTransport
volVectorField U
(
    IOobject
    (
        "U",
        mesh.time().timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedVector("U",dimensionSet(0,1,-1,0,0,0,0),vector::zero),
    zeroGradientFvPatchVectorField::typeName
);
#include "createPhi.H"

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

