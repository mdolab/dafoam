/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

\*---------------------------------------------------------------------------*/

#include "AdjointDummyTurbulenceModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(AdjointDummyTurbulenceModel, 0);
addToRunTimeSelectionTable(AdjointRASModel, AdjointDummyTurbulenceModel, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

AdjointDummyTurbulenceModel::AdjointDummyTurbulenceModel
(
    const fvMesh& mesh,
    const AdjointIO& adjIO,
    nearWallDist& d,
#ifdef IncompressibleFlow
    const singlePhaseTransportModel& laminarTransport
#endif
#ifdef CompressibleFlow
    const fluidThermo& thermo
#endif
)
    :
#ifdef IncompressibleFlow
    AdjointRASModel(mesh,adjIO,d,laminarTransport)
#endif
#ifdef CompressibleFlow
    AdjointRASModel(mesh,adjIO,d,thermo)
#endif
    
{
    turbStates.setSize(0); 
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //



// Augmented functions
void AdjointDummyTurbulenceModel::updateNut()
{
}


void AdjointDummyTurbulenceModel::copyTurbStates(const word option)
{
}


void AdjointDummyTurbulenceModel::correctTurbBoundaryConditions()
{
}


void AdjointDummyTurbulenceModel::calcTurbResiduals
(  
    const label isRef,
    const label isPC,
    const word fvMatrixName
)
{    
}


void AdjointDummyTurbulenceModel::correctAdjStateResidualTurbCon
(
    List< List<word> >& adjStateResidualConInfo
)
{
}

void AdjointDummyTurbulenceModel::setAdjStateResidualTurbCon
(
    HashTable< List< List<word> > >& adjStateResidualConInfo
)
{

}

void AdjointDummyTurbulenceModel::clearTurbVars()
{

}

void AdjointDummyTurbulenceModel::writeTurbStates()
{

}

} // End namespace Foam

// ************************************************************************* //
