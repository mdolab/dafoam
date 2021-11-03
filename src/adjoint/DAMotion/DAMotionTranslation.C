/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

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

    scalar currentTime = mesh_.time().value();

    const scalar& pi = Foam::constant::mathematical::pi;

    vector dy = amplitude_ * sin(2.0 * pi * frequency_ * currentTime + phase_) * direction_;

    forAll(patchNames_, idxI)
    {
        const word& patchName = patchNames_[idxI];
        label patchI = mesh_.boundaryMesh().findPatchID(patchName);

        forAll(cellDisp.boundaryField()[patchI], faceI)
        {
            cellDisp.boundaryFieldRef()[patchI][faceI] = dy;
        }
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
