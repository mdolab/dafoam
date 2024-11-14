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

    // Heat flux can be calculated by either per unit area or over entire surface. Default value is byUnitArea
    if (objFuncDict_.found("scheme"))
    {
        objFuncDict_.readEntry<word>("scheme", calcMode_);
    }
    else
    {
        calcMode_ = "byUnitArea";
    }


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
    // for solid, we need to read k from transportProperties
    if (k_ < 0)
    {
        k_ = readScalar(transportProperties.lookup("k"));
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

    // if calcMode is per unit area then calculate the area of all the heat flux patches
    if (calcMode_ == "byUnitArea")
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
            wallHeatFluxBf[patchI] = Cp_ * alphaEffBf[patchI] * TBf[patchI].snGrad();
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
    // solid. H = k * dT/dz, where k = DT / rho / Cp
    const objectRegistry& db = mesh_.thisDb();
    const volScalarField& T = db.lookupObject<volScalarField>("T");
    const volScalarField::Boundary& TBf = T.boundaryField();
    forAll(wallHeatFluxBf, patchI)
    {
        if (!wallHeatFluxBf[patchI].coupled())
        {
            wallHeatFluxBf[patchI] = k_ * TBf[patchI].snGrad();
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

        if (calcMode_ == "byUnitArea")
        {
            objFuncFaceValues[idxI] = scale_ * wallHeatFluxBf[patchI][faceI] * area / areaSum_;
        }
        else if (calcMode_ == "total")
        {
            objFuncFaceValues[idxI] = scale_ * wallHeatFluxBf[patchI][faceI] * area;
        }
        else
        {
           FatalErrorIn(" ") << "mode for "
                            << objFuncName_ << " " << objFuncPart_ << " not valid!"
                            << "Options: byUnitArea (default value), total."
                            << abort(FatalError);
        }

        objFuncValue += objFuncFaceValues[idxI];
    }

    // need to reduce the sum of force across all processors
    reduce(objFuncValue, sumOp<scalar>());

    // check if we need to calculate refDiff.
    this->calcRefVar(objFuncValue);

    return;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
