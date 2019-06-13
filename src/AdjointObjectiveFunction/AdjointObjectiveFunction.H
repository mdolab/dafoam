/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Adjoint objective function

    Steps to add new objective functions:

    1. Register the objective function name to AdjointObjectiveFunction::objFuncsReg_

    2. Add a new function to compute this new objective function. Take calcCD() as reference.
       Basically, we need to compute the objective function value (objVal) as well as its discrete
       values on objective function patches or volumes (objDiscreteVal) 

    3. Call this new function in calcObjFuncs(). Follow the format from other objective functions.

    4. Add connectivity information in the AdjointJacobianConnectivity::setupObjFuncCon() function 
       in AdjointJacobianConnectivity.C, follow the format from other objective functions.

    5. Add this new objective function to self.possibleObjectives in pyDAFoam.py

\*---------------------------------------------------------------------------*/


#ifndef AdjointObjectiveFunction_H
#define AdjointObjectiveFunction_H

#include "fvOptions.H"
#include "surfaceFields.H"
#include "AdjointIO.H"
#include "AdjointRASModel.H"
#include "AdjointJacobianConnectivity.H"
#include "AdjointIndexing.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class AdjointObjectiveFunction Declaration
\*---------------------------------------------------------------------------*/

class AdjointObjectiveFunction
{

protected:

    /// list of objective function names
    List<word> objFuncs_;

    /// list of objective function geometry information
    List< List<word> > objFuncGeoInfo_;

    const fvMesh& mesh_;
    const AdjointIO& adjIO_;
    AdjointRASModel& adjRAS_;
    AdjointIndexing& adjIdx_;
    AdjointJacobianConnectivity& adjCon_;
    
    /// list of registered objective functions
    List<word> objFuncsReg_;

    /// ID of the registered objective functions
    HashTable<label> objFuncsRegID_;
    
    /// patch names for the force-related objFuncs
    List<word> forcePatches_;
    
    /// force for each discrete surface
    List<vector> surfForces_;

    /// calculate AdjointObjectiveFunction::surfForces_
    vector calcForces();

    /// calculate drag coefficient
    void calcCD(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate lift coefficient
    void calcCL(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate drag coefficient
    void calcCPL(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate Nusselt number
    void calcNUS(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate moment coefficient in x direction
    void calcCMX(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate moment coefficient in y direction
    void calcCMY(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate moment coefficient in z direction
    void calcCMZ(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate KS-aggregated von Mises stress
    void calcVMS(scalar& objVal, scalarList& objDiscreteVal,scalar& KSExpSum);

    /// calculate averaged state variable in a user-defined volume in AdjointIO::userDefinedVolumeInfo
    void calcAVGV(scalar& objVal, scalarList& objDiscreteVal, wordList geoName);

    /// calculate variance of state variable in a user-defined volume in AdjointIO::userDefinedVolumeInfo
    void calcVARV(scalar& objVal, scalarList& objDiscreteVal, wordList geoName);

    /// calculate averaged state variable over a user-defined patch in AdjointIO::userDefinedPatchInfo
    void calcAVGS(scalar& objVal, scalarList& objDiscreteVal, wordList geoName);

    /// calculate mass flow rate
    void calcMFR(scalar& objVal, scalarList& objDiscreteVal);

    /// calculate total pressure ratio 
    void calcTPR(scalar& objVal, scalarList& objDiscreteVal, label isRef);

    /// calculate total temperature ratio
    void calcTTR(scalar& objVal, scalarList& objDiscreteVal, label isRef);

    /// sum of obj function value
    scalarList objFuncsVal_;

    /// reference value for the sum of objFunc value
    scalarList objFuncsValRef_;

    /// partial derivative of objFunc
    scalarList objFuncsPartialDeriv_;

    /// discrete obj fun values for each surface
    List< List<scalar> > objFuncsDiscreteVal_;

    /// discrete reference obj fun values for each surface
    List< List<scalar> > objFuncsDiscreteValRef_;

    /// discrete obj fun partial derivative values for each surface
    List< List<scalar> > objFuncsDiscretePartialDeriv_;

    /// force related objective function
    List<word> forceRelatedObjFuncs_;
    
    /// 3D Nutsselt field
    volScalarField NUSField_;

    /// bulk temperature for Nus
    volScalarField TBulk_;

    /// Prandlt number
    scalar Pr_;

    /// reference averaged total pressure and total temperature at inlet and outlet
    scalar TPInRef_=0.0, TPOutRef_=0.0, TTInRef_=0.0, TTOutRef_=0.0;

    
public:

    // Constructors

    //- Construct from components
    AdjointObjectiveFunction
    (
        const fvMesh& mesh,
        const AdjointIO& adjIO,
        AdjointRASModel& adjRAS,
        AdjointIndexing& adjIdx,
        AdjointJacobianConnectivity& adjCon
    );

    //- Destructor
    virtual ~AdjointObjectiveFunction(); 
    
    /// calculate objective function
    void calcObjFuncs(word objFunc,label isRef);

    /// calculate partial derivatives for an given objective function
    void calcObjFuncPartDerivs(const scalar eps,const word objFunc);
    
    /// initialize AdjointObjectiveFunction::surfForces_
    void initializeSurfForces();

    /// initialize AdjointObjectiveFunction::forcePatches_
    void initializeForcePatches();
    
    /// print objective function values to screen
    void printObjFuncValues();

    /// write objective function values to disk
    void writeObjFuncValues();
    
    /// get the partial derivatives for a given objFunc
    scalar getObjFuncPartDeriv(const word objFunc);
    
    /// get the discrete partial derivatives for a given objFunc and a surface index 
    scalar getObjFuncDiscretePartDeriv(word objFunc,label idxI);

    /// calculate actuator disk source term and assign it to sourceU
    void calcActuatorDiskSource(vectorField& sourceU);

    /// return the objective function value
    scalar getObjFunc(const word objFunc)
    {
        const label& id=objFuncsRegID_[objFunc];
        return objFuncsVal_[id];
    }

    /// KSExpSumRef stores sum[exp(KSCoeff*x_i)], it will be used to scale dFdW
    scalar KSExpSumRef;

    Switch thermalStress;
    
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
