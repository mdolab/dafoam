/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAFPAdjSimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAFPAdjSimpleFoam, 0);
addToRunTimeSelectionTable(DAFPAdj, DAFPAdjSimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAFPAdjSimpleFoam::DAFPAdjSimpleFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAFPAdj(
        modelType,
        mesh,
        daOption,
        daModel,
        daIndex)
{
}

label DAFPAdjSimpleFoam::run(
    Vec dFdW,
    Vec psi)
{
#ifdef CODI_AD_REVERSE
    /*
    Description:
        Solve the adjoint using the fixed-point iteration method
    
    dFdW:
        The dF/dW vector 

    psi:
        The adjoint solution vector
    */

    Info << "Solving the adjoint using fixed-point iteration method..." << endl;
#endif
    return 0;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
