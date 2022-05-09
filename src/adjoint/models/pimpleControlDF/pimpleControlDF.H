/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        A modified version of pimpleControl
        Basically, we disable the output to the screen
    
    This file is modified from OpenFOAM's source code
    src/finiteVolume/cfdTools/general/solutionControl/pimpleControl/pimpleControl.H

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

#ifndef pimpleControlDF_H
#define pimpleControlDF_H

#include "solutionControl.H"

//- Declare that pimpleControlDF will be used
#define PIMPLE_CONTROL

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                        Class pimpleControlDF Declaration
\*---------------------------------------------------------------------------*/

class pimpleControlDF
    : public solutionControl
{
    // Private member functions

    //- No copy construct
    pimpleControlDF(const pimpleControlDF&) = delete;

    //- No copy assignment
    void operator=(const pimpleControlDF&) = delete;

protected:
    // Protected data

    // Solution controls

    //- Flag to indicate whether to solve for the flow
    bool solveFlow_;

    //- Maximum number of PIMPLE correctors
    label nCorrPIMPLE_;

    //- Maximum number of PISO correctors
    label nCorrPISO_;

    //- Current PISO corrector
    label corrPISO_;

    //- Flag to indicate whether to update density in SIMPLE
    //  rather than PISO mode
    bool SIMPLErho_;

    //- Flag to indicate whether to only solve turbulence on final iter
    bool turbOnFinalIterOnly_;

    //- Converged flag
    bool converged_;

    // Protected Member Functions

    //- Read controls from fvSolution dictionary
    virtual void read();

    //- Return true if all convergence checks are satisfied
    virtual bool criteriaSatisfied();

    //- Set the firstIteration flag on the mesh data dictionary
    virtual void setFirstIterFlag(
        const bool check = true,
        const bool force = false);

public:
    // Static Data Members

    //- Run-time type information
    TypeName("pimpleControlDF");

    // Constructors

    //- Construct from mesh and the name of control sub-dictionary
    pimpleControlDF(fvMesh& mesh, const word& dictName = "PIMPLE");

    //- Destructor
    virtual ~pimpleControlDF() = default;

    // Member Functions

    // Access

    //- Maximum number of PIMPLE correctors
    inline label nCorrPIMPLE() const;

    //- Maximum number of PISO correctors
    inline label nCorrPISO() const;

    //- Current PISO corrector index
    inline label corrPISO() const;

    //- Flag to indicate whether to update density in SIMPLE
    //  rather than PISO mode
    inline bool SIMPLErho() const;

    // Solution control

    //- PIMPLE loop
    virtual bool loop();

    //- Pressure corrector loop control
    inline bool correct();

    //- Return true to store the intial residuals
    inline bool storeInitialResiduals() const;

    //- Return true for first PIMPLE (outer) iteration
    inline bool firstIter() const;

    //- Return true for final PIMPLE (outer) iteration
    inline bool finalIter() const;

    //- Return true for final inner iteration
    inline bool finalInnerIter() const;

    //- Return true to solve for flow
    inline bool solveFlow() const;

    //- Return true to solve for turbulence
    inline bool turbCorr();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "pimpleControlIDF.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
