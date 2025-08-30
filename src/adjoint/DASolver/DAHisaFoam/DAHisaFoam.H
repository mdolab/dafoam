/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAHisaFoam
    
    This class is modified from the High Speed Aerodynamic (HiSA) solver
    
    Website: https://hisa.gitlab.io/index.html
    Repo: https://gitlab.com/hisa/hisa

\*---------------------------------------------------------------------------*/

#ifndef DAHisaFoam_H
#define DAHisaFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "fvCFD.H"
#include "pimpleControl.H"
#include "pseudotimeControl.H"
#include "solverModule.H"
#include "fluxScheme.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAHisaFoam Declaration
\*---------------------------------------------------------------------------*/

class DAHisaFoam
    : public DASolver
{

protected:
    
    autoPtr<solverModule> hisaSolverPtr_;

    /// turbulence pointer
    autoPtr<compressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

public:
    TypeName("DAHisaFoam");
    // Constructors

    //- Construct from components
    DAHisaFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAHisaFoam()
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
