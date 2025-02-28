/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAIrkPimpleFoam
    
    This class is modified from OpenFOAM's source code
    applications/solvers/incompressible/pimpleFoam

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

#ifndef DAIrkPimpleFoam_H
#define DAIrkPimpleFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
//#include "pimpleControlDF.H"    // A modified version of pimpleControl, Basically, we disable the output to the screen
#include "pimpleControl.H"      // origional pimpleControl

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAIrkPimpleFoam Declaration
\*---------------------------------------------------------------------------*/

class DAIrkPimpleFoam
    : public DASolver
{

protected:
public:
    TypeName("DAIrkPimpleFoam");
    // Constructors

    //- Construct from components
    DAIrkPimpleFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAIrkPimpleFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
