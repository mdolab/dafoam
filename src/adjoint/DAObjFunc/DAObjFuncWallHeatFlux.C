/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAObjFuncWallHeatFlux.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAObjFuncWallHeatFlux, 0);
addToRunTimeSelectionTable(DAObjFunc, DAObjFuncWallHeatFlux, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAObjFuncWallHeatFlux::DAObjFuncWallHeatFlux(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex,
    const DAResidual& daResidual,
    const word objFuncName,
    const word objFuncPart,
    const dictionary& objFuncDict)
    : DAObjFunc(
        mesh,
        daOption,
        daModel,
        daIndex,
        daResidual,
        objFuncName,
        objFuncPart,
        objFuncDict),
#ifdef CompressibleFlow
      thermo_(const_cast<fluidThermo&>(daModel.getThermo())),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      wallHeatFlux_(
          IOobject(
              "wallHeatFlux",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("wallHeatFlux", dimensionSet(1, 0, -3, 0, 0, 0, 0), 0.0),
          "calculated")
#endif
#ifdef IncompressibleFlow
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      wallHeatFlux_(
          IOobject(
              "wallHeatFlux",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::AUTO_WRITE),
          mesh,
          dimensionedScalar("wallHeatFlux", dimensionSet(1, -2, 1, 1, 0, 0, 0), 0.0),
          "calculated")
#endif
#ifdef SolidDASolver
       wallHeatFlux_(
           IOobject(
               "wallHeatFlux",
               mesh.time().timeName(),
               mesh,
               IOobject::NO_READ,
               IOobject::AUTO_WRITE),
           mesh,
           dimensionedScalar("wallHeatFlux", dimensionSet(1, -2, 1, 1, 0, 0, 0), 0.0),
           "calculated")
#endif
{
    // Assign type, this is common for all objectives
    objFuncDict_.readEntry<word>("type", objFuncType_);

    objFuncDict_.readEntry<scalar>("scale", scale_);

#ifdef CompressibleFlow

    // setup the connectivity for heat flux, this is needed in Foam::DAJacCondFdW
    objFuncConInfo_ = {
        {"nut", "T"}, // level 0
        {"T"}}; // level 1

    // now replace nut with the corrected name for the selected turbulence model
    daModel.correctModelStates(objFuncConInfo_[0]);

#endif

#ifdef IncompressibleFlow

    // setup the connectivity for heat flux, this is needed in Foam::DAJacCondFdW
    objFuncConInfo_ = {
        {"nut", "T"}, // level 0
        {"T"}}; // level 1

    // now replace nut with the corrected name for the selected turbulence model
    daModel.correctModelStates(objFuncConInfo_[0]);

    // initialize the Prandtl number from transportProperties
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    // for incompressible flow, we need to read Cp from transportProperties
    if (Cp_ < 0)
    {
        Cp_ = readScalar(transportProperties.lookup("Cp"));
    }
    rho_ = 1.0;
#endif

#ifdef SolidDASolver
    IOdictionary transportProperties(
        IOobject(
            "transportProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    // for incompressible flow, we need to read Cp from transportProperties
    if (Cp_ < 0)
    {
        Cp_ = readScalar(transportProperties.lookup("Cp"));
    }
    if (rho_ < 0)
    {
        rho_ = readScalar(transportProperties.lookup("rho"));
    }
    if (DT_ < 0)
    {
        DT_ = readScalar(transportProperties.lookup("DT"));
    }
#endif
}

/// calculate the value of objective function
void DAObjFuncWallHeatFlux::calcObjFunc(
    const labelList& objFuncFaceSources,
    const labelList& objFuncCellSources,
    scalarList& objFuncFaceValues,
    scalarList& objFuncCellValues,
    scalar& objFuncValue)
{
    /*
    Description:
        Calculate the heat flux F=k*dT/dz from the first cell. Modified based on
        OpenFOAM/OpenFOAM-v1812/src/functionObjects/field/wallWallHeatFlux/wallWallHeatFlux.C

    Input:
        objFuncFaceSources: List of face source (index) for this objective
    
        objFuncCellSources: List of cell source (index) for this objective

    Output:
        objFuncFaceValues: the discrete value of objective for each face source (index). 
        This  will be used for computing df/dw in the adjoint.
    
        objFuncCellValues: the discrete value of objective on each cell source (index). 
        This will be used for computing df/dw in the adjoint.
    
        objFuncValue: the sum of objective, reduced across all processors and scaled by "scale"
    */

    // calculate the area of all the heat flux patches
    if (areaSum_ < 0.0)
    {
        areaSum_ = 0.0;
        forAll(objFuncFaceSources, idxI)
        {
            const label& objFuncFaceI = objFuncFaceSources[idxI];
            label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
            const label patchI = daIndex_.bFacePatchI[bFaceI];
            const label faceI = daIndex_.bFaceFaceI[bFaceI];
            areaSum_ += mesh_.magSf().boundaryField()[patchI][faceI];
        }
        reduce(areaSum_, sumOp<scalar>());
    }

    // initialize faceValues to zero
    forAll(objFuncFaceValues, idxI)
    {
        objFuncFaceValues[idxI] = 0.0;
    }
    // initialize objFunValue
    objFuncValue = 0.0;

    volScalarField::Boundary& wallHeatFluxBf = wallHeatFlux_.boundaryFieldRef();

#ifdef IncompressibleFlow
    // incompressible flow does not have he, so we do H = Cp * alphaEff * dT/dz
    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    volScalarField alphaEff = daTurb_.alphaEff();
    const volScalarField::Boundary& TBf = T.boundaryField();
    const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();
    forAll(wallHeatFluxBf, patchI)
    {
        if (!wallHeatFluxBf[patchI].coupled())
        {
            wallHeatFluxBf[patchI] = rho_ * Cp_ * alphaEffBf[patchI] * TBf[patchI].snGrad();
        }
    }
#endif

#ifdef CompressibleFlow
    // compressible flow, H = alphaEff * dHE/dz
    volScalarField& he = thermo_.he();
    const volScalarField::Boundary& heBf = he.boundaryField();
    volScalarField alphaEff = daTurb_.alphaEff();
    const volScalarField::Boundary& alphaEffBf = alphaEff.boundaryField();
    forAll(wallHeatFluxBf, patchI)
    {
        if (!wallHeatFluxBf[patchI].coupled())
        {
            wallHeatFluxBf[patchI] = alphaEffBf[patchI] * heBf[patchI].snGrad();
        }
    }
#endif

#ifdef SolidDASolver
    // solid. H = rho * Cp * DT * dT/dz
    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    const volScalarField::Boundary& TBf = T.boundaryField();
    forAll(wallHeatFluxBf, patchI)
    {
        if (!wallHeatFluxBf[patchI].coupled())
        {
            wallHeatFluxBf[patchI] = rho_ * Cp_ * DT_ * TBf[patchI].snGrad();
        }
    }
#endif

    // calculate area weighted heat flux
    forAll(objFuncFaceSources, idxI)
    {
        const label& objFuncFaceI = objFuncFaceSources[idxI];
        label bFaceI = objFuncFaceI - daIndex_.nLocalInternalFaces;
        const label patchI = daIndex_.bFacePatchI[bFaceI];
        const label faceI = daIndex_.bFaceFaceI[bFaceI];

        scalar area = mesh_.magSf().boundaryField()[patchI][faceI];
        objFuncFaceValues[idxI] = scale_ * wallHeatFluxBf[patchI][faceI] * area / areaSum_;

        objFuncValue += objFuncFaceValues[idxI];
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
