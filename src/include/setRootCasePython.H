/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4


    Description:
        This is similar to setRootCase.H except we set the last argument
        initialise=false because we don't want to print the long OpenFOAM header
        We also add a -python bool option to tell UPstream.C not to call
        MPI_Finalize because we want mpi4py to call it instead
    
    This file is modified from OpenFOAM's source code
    src/OpenFOAM/include/setRootCase.H

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

Foam::argList::addBoolOption("python", "Use Python/Cython wrapping such that the UPstream function \
    in OpenFOAM will not initialize or finalize MPI, instead, mpi4py will do that.");

argsPtr_.reset(new Foam::argList(argc, argv, true, true, false));

Foam::argList& args = argsPtr_();

if (!args.checkRootCase())
{
    Foam::FatalError.exit();
}
