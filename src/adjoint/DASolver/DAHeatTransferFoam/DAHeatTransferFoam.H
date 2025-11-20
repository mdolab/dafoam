/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        Child class for DAHeatTransferFoam

\*---------------------------------------------------------------------------*/

#ifndef DAHeatTransferFoam_H
#define DAHeatTransferFoam_H

#include "DASolver.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAHeatTransferFoam Declaration
\*---------------------------------------------------------------------------*/

class DAHeatTransferFoam
    : public DASolver
{

protected:
 
    /// temperature field pointer
    autoPtr<volScalarField> TPtr_;

    /// heat source
    autoPtr<volScalarField> fvSourcePtr_;

    /// h pointer
    autoPtr<dimensionedScalar> kPtr_;

public:
    TypeName("DAHeatTransferFoam");
    // Constructors

    //- Construct from components
    DAHeatTransferFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAHeatTransferFoam()
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
