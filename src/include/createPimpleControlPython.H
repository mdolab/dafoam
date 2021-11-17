/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

Info << "Create pimpleControlDF." << endl;

pimplePtr_.reset(
    new pimpleControlDF(
        mesh));

pimpleControlDF& pimple = pimplePtr_();
