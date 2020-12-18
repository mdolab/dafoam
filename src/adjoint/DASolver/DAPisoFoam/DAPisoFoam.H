/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for DAPisoFoam

\*---------------------------------------------------------------------------*/

#ifndef DAPisoFoam_H
#define DAPisoFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pisoControl.H"
#include "DARegDbSinglePhaseTransportModel.H"
#include "DARegDbTurbulenceModelIncompressible.H"
#include "DAFvSource.H"
#include "DAIntmdVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAPisoFoam Declaration
\*---------------------------------------------------------------------------*/

class DAPisoFoam
    : public DASolver
{

protected:
    /// piso pointer
    autoPtr<pisoControl> pisoPtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// laminar transport properties pointer
    autoPtr<singlePhaseTransportModel> laminarTransportPtr_;

    /// turbulence pointer
    autoPtr<incompressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// DAIntmVar pointer
    autoPtr<DAIntmdVar> daIntmdVarPtr_;

    /// DASource pointer
    autoPtr<DAFvSource> daFvSourcePtr_;

    /// fvSource term 
    autoPtr<volVectorField> fvSourcePtr_;

    /// whether to have fvSource term
    label hasFvSource_ = 0;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

    /// pressure referefence cell id
    label pRefCell_ = 0;

    /// pressure reference value
    scalar pRefValue_ = 0.0;

    /// state variable list for all instances
    List<List<scalar>> stateAllInstances_;

    /// state boundary variable list for all instances
    List<List<scalar>> stateBounaryAllInstances_;

    /// objective function for all instances
    List<dictionary> objFuncsAllInstances_;

    /// the runtime value for all instances
    List<scalar> runTimeAllInstances_;

    /// the runtime index for all instances
    List<label> runTimeIndexAllInstances_;

    /// number of time instances for hybird adjoint
    label nTimeInstances_;

    /// periodicity of oscillating flow variables
    scalar periodicity_;

    /// save primal variable to time instance list for unsteady adjoint
    void saveTimeInstanceField(label& timeInstanceI);

public:
    TypeName("DAPisoFoam");
    // Constructors

    //- Construct from components
    DAPisoFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DAPisoFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal(
        const Vec xvVec,
        Vec wVec);

    /// assign primal variables based on the current time instance
    virtual void setTimeInstanceField(const label instanceI);

    /// return the value of objective function at the given time instance and name
    virtual scalar getTimeInstanceObjFunc(
        const label instanceI,
        const word objFuncName);
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
