/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

Info << "Create pisoControl." << endl;

pisoPtr_.reset(
    new pisoControl(
        mesh));

pisoControl& piso = pisoPtr_();
