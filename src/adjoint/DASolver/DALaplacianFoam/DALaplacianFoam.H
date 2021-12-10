/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DALaplacianFoam

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
