/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

Info << "Create simpleControl." << endl;

simplePtr_.reset(
    new simpleControl(
        mesh));

simpleControl& simple = simplePtr_();
