/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAScalarTransportFoam

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
