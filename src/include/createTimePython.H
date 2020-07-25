/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

Info << "Create time\n"
     << endl;
runTimePtr_.reset(
    new Time(
        Time::controlDictName,
        args));

