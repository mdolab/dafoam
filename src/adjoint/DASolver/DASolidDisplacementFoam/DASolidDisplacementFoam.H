/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DASolidDisplacementFoam

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

    /// whether to use compact normal stress
    Switch compactNormalStress_;

    /// number of correctors
    label nCorr_;

    /// convergence tolerance for D
    scalar convergenceTolerance_;

    /// mechanicalProperties pointer
    autoPtr<IOdictionary> mechanicalPropertiesPtr_;

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
    
    /// stress field pointer
    autoPtr<volSymmTensorField> sigmaDPtr_;

    /// displacement gradient pointer
    autoPtr<volTensorField> gradDPtr_;

    /// explicit part of div(sigma) pointer
    autoPtr<volVectorField> divSigmaExpPtr_;

    
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
    virtual label solvePrimal(
        const Vec xvVec,
        Vec wVec);

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
