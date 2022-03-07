/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAMotion.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAMotion, 0);
defineRunTimeSelectionTable(DAMotion, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAMotion::DAMotion(
    const dynamicFvMesh& mesh,
    const DAOption& daOption)
    : mesh_(mesh),
      daOption_(daOption)
{
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAMotion> DAMotion::New(
    const dynamicFvMesh& mesh,
    const DAOption& daOption)
{
    // standard setup for runtime selectable classes

    // look up the solver name
    word modelType = daOption.getAllOptions().subDict("rigidBodyMotion").getWord("mode");

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting type: " << modelType << " for DAMotion. " << endl;
    }

    dictionaryConstructorTable::iterator cstrIter =
        dictionaryConstructorTablePtr_->find(modelType);

    // if the solver name is not found in any child class, print an error
    if (cstrIter == dictionaryConstructorTablePtr_->end())
    {
        FatalErrorIn(
            "DAMotion::New"
            "("
            "    const dynamicFvMesh&,"
            "    const DAOption&,"
            ")")
            << "Unknown DAMotion type "
            << modelType << nl << nl
            << "Valid DAMotion types:" << endl
            << dictionaryConstructorTablePtr_->sortedToc()
            << exit(FatalError);
    }

    // child class found
    return autoPtr<DAMotion>(
        cstrIter()(mesh,
                   daOption));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //
vector DAMotion::getForce(const dynamicFvMesh& mesh)
{

    const objectRegistry& db = mesh.thisDb();
    const volScalarField& p = db.lookupObject<volScalarField>("p");
    const volScalarField& nut = db.lookupObject<volScalarField>("nut");
    const volScalarField& nu = db.lookupObject<volScalarField>("nu");
    const volVectorField& U = db.lookupObject<volVectorField>("U");

    volSymmTensorField tdevRhoReff("devRhoReff", -(nu + nut) * dev(twoSymm(fvc::grad(U))));

    const volSymmTensorField::Boundary& devRhoReffb = tdevRhoReff.boundaryField();

    // iterate over patches and extract boundary surface forces
    vector force(vector::zero);
    forAll(patchNames_, idxI)
    {
        // get the patch id label
        label patchI = mesh.boundaryMesh().findPatchID(patchNames_[idxI]);
        // normal force
        vectorField fN(mesh.Sf().boundaryField()[patchI] * p.boundaryField()[patchI]);
        // tangential force
        vectorField fT(mesh.Sf().boundaryField()[patchI] & devRhoReffb[patchI]);
        // sum them up
        forAll(mesh.boundaryMesh()[patchI], faceI)
        {
            force += fN[faceI] + fT[faceI];
        }
    }

    reduce(force[0], sumOp<scalar>());
    reduce(force[1], sumOp<scalar>());
    reduce(force[2], sumOp<scalar>());

    return force;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
