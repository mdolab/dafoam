/*---------------------------------------------------------------------------*\

    pyOFM  : Python interface for OpenFOAM mesh
    Version : v1.1


    Description:
        This is similar to setRootCase.H except we set the last argument
        initialise=false because we don't want to print the long OpenFOAM header
        We also add a -python bool option to tell UPstream.C not to call
        MPI_Finalize because we want mpi4py to call it instead

\*---------------------------------------------------------------------------*/

Foam::argList::addBoolOption("python","Use Python/Cython wrapping such that the UPstream function \
    in OpenFOAM will not initialize or finalize MPI, instead, mpi4py will do that.");

Foam::argList args(argc, argv, true, true, false);
if (!args.checkRootCase())
{
    Foam::FatalError.exit();
}


