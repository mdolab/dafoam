/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

Info << "Create pimpleControl." << endl;

pimplePtr_.reset(
    new pimpleControl(
        mesh));

pimpleControl& pimple = pimplePtr_();
