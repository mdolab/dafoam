/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAMotionTranslation.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAMotionTranslation, 0);
addToRunTimeSelectionTable(DAMotion, DAMotionTranslation, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAMotionTranslation::DAMotionTranslation(
    const dynamicFvMesh& mesh,
    const DAOption& daOption)
    : DAMotion(
        mesh,
        daOption)
{
    amplitude_ = daOption_.getAllOptions().subDict("rigidBodyMotion").getScalar("amplitude");
    frequency_ = daOption_.getAllOptions().subDict("rigidBodyMotion").getScalar("frequency");
    phase_ = daOption_.getAllOptions().subDict("rigidBodyMotion").getScalar("phase");
    scalarList dirList;
    daOption_.getAllOptions().subDict("rigidBodyMotion").readEntry<scalarList>("direction", dirList);
    direction_[0] = dirList[0];
    direction_[1] = dirList[1];
    direction_[2] = dirList[2];
    daOption_.getAllOptions().subDict("rigidBodyMotion").readEntry<wordList>("patchNames", patchNames_);
}

void DAMotionTranslation::correct()
{
    volVectorField& cellDisp =
        const_cast<volVectorField&>(mesh_.thisDb().lookupObject<volVectorField>("cellDisplacement"));

    scalar t = mesh_.time().value();

    const scalar& pi = Foam::constant::mathematical::pi;

    scalar y = amplitude_ * sin(2.0 * pi * frequency_ * t + phase_);

    forAll(patchNames_, idxI)
    {
        const word& patchName = patchNames_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);

        forAll(cellDisp.boundaryField()[patchI], faceI)
        {
            cellDisp.boundaryFieldRef()[patchI][faceI] = y * direction_;
        }
    }

    // print information
    Info << "Time: " << t << "  y: " << y << endl;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
