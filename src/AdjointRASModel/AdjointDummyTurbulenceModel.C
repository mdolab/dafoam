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
    // set turbulence variable to zero
    forAll(nut_,idxI) nut_[idxI]=0.0;
    forAll(nut_.boundaryField(),patchI)
    {
        forAll(nut_.boundaryField()[patchI],faceI)
        {
            nut_.boundaryFieldRef()[patchI][faceI]=0.0;
        }
    }
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
    // we need to remove nut from the conList
    List< List<word> > tmp;
    tmp.setSize(adjStateResidualConInfo.size());

    // For SA model just replace nut with nuTilda
    forAll(adjStateResidualConInfo,idxI)
    {
        forAll(adjStateResidualConInfo[idxI],idxJ)
        {
            word conStateName = adjStateResidualConInfo[idxI][idxJ];
            if( conStateName != "nut" ) tmp[idxI].append(adjStateResidualConInfo[idxI][idxJ]);
        }
    }
    adjStateResidualConInfo.clear();

    adjStateResidualConInfo=tmp;
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
