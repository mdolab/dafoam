/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DAHeatTransferFoam

\*---------------------------------------------------------------------------*/

#ifndef DAHeatTransferFoam_H
#define DAHeatTransferFoam_H

#include "DASolver.H"
#include "DAFvSource.H"

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

    /// DASource pointer
    autoPtr<DAFvSource> daFvSourcePtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

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
    virtual label solvePrimal();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
