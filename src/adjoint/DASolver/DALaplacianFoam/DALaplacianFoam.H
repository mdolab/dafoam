/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DALaplacianFoam
    
    This class is modified from OpenFOAM's source code
    applications/solvers/basic/laplacianFoam

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

#ifndef DALaplacianFoam_H
#define DALaplacianFoam_H

#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DALaplacianFoam Declaration
\*---------------------------------------------------------------------------*/

class DALaplacianFoam
    : public DASolver
{

protected:
 
    /// temperature field pointer
    autoPtr<volScalarField> TPtr_;

    /// DT pointer
    autoPtr<dimensionedScalar> DTPtr_;

    /// unsteady mode
    word mode_ = "None";


public:
    TypeName("DALaplacianFoam");
    // Constructors

    //- Construct from components
    DALaplacianFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DALaplacianFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal(
        const Vec xvVec,
        Vec wVec);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
