/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1812

\*---------------------------------------------------------------------------*/

#include "AdjointSolverRegistryRhoSimpleCFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointSolverRegistryRhoSimpleCFoam, 0);
addToRunTimeSelectionTable(AdjointSolverRegistry, AdjointSolverRegistryRhoSimpleCFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointSolverRegistryRhoSimpleCFoam::AdjointSolverRegistryRhoSimpleCFoam
(
    const fvMesh& mesh
)
    :
    AdjointSolverRegistry(mesh)
{
    // Register state variables
    // NOTE: do not include any turbulence state variables since they will be added 
    // in the AdjointRASModel class!
    volScalarStates.append("p");
    volScalarStates.append("T");
    volVectorStates.append("U");
    surfaceScalarStates.append("phi");
    // append here if you have more state variables

    // for debugging
    //Info<<"volScalarStates: "<<volScalarStates<<endl;
    //Info<<"volVectorStates: "<<volVectorStates<<endl;
    //Info<<"surfaceScalarStates: "<<surfaceScalarStates<<endl;


    // Need to call this to check if the registered states are actually created in db
    //this->validate();
    
    // setup the derived state info
    this->setDerivedInfo();
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
