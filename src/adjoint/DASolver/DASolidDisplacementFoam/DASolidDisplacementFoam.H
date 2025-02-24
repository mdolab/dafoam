/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DASolidDisplacementFoam
    
    This class is modified from OpenFOAM's source code
    applications/solvers/stressAnalysis/solidDisplacementFoam

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

#ifndef DASolidDisplacementFoam_H
#define DASolidDisplacementFoam_H

#include "Switch.H"
#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DASolidDisplacementFoam Declaration
\*---------------------------------------------------------------------------*/

class DASolidDisplacementFoam
    : public DASolver
{

protected:
    /// density field pointer
    autoPtr<volScalarField> rhoPtr_;

    /// mu field pointer
    autoPtr<volScalarField> muPtr_;

    /// lambda field pointer
    autoPtr<volScalarField> lambdaPtr_;

    /// E field pointer
    autoPtr<volScalarField> EPtr_;

    /// nu field pointer
    autoPtr<volScalarField> nuPtr_;

    /// displacement field pointer
    autoPtr<volVectorField> DPtr_;

    /// displacement gradient pointer
    autoPtr<volTensorField> gradDPtr_;

public:
    TypeName("DASolidDisplacementFoam");
    // Constructors

    //- Construct from components
    DASolidDisplacementFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DASolidDisplacementFoam()
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
