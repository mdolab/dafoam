/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DAScalarTransportFoam
    
    This class is modified from OpenFOAM's source code
    applications/solvers/basic/scalarTransportFoam

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

#ifndef DAScalarTransportFoam_H
#define DAScalarTransportFoam_H

#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAScalarTransportFoam Declaration
\*---------------------------------------------------------------------------*/

class DAScalarTransportFoam
    : public DASolver
{

protected:
 
    /// temperature field pointer
    autoPtr<volScalarField> TPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// DT pointer
    autoPtr<dimensionedScalar> DTPtr_;

    /// unsteady mode
    word mode_ = "None";


public:
    TypeName("DAScalarTransportFoam");
    // Constructors

    //- Construct from components
    DAScalarTransportFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAScalarTransportFoam()
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
