/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v5

\*---------------------------------------------------------------------------*/

#include "DAResidual.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defineTypeNameAndDebug(DAResidual, 0);
defineRunTimeSelectionTable(DAResidual, dictionary);

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidual::DAResidual(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : regIOobject(
        IOobject(
            "DAResidual", // always use DAResidual for the db name
            mesh.time().timeName(),
            mesh, // register to mesh
            IOobject::NO_READ,
            IOobject::NO_WRITE,
            true // always register object
            )),
      mesh_(mesh),
      daOption_(daOption),
      daModel_(daModel),
      daIndex_(daIndex),
      daField_(mesh, daOption, daModel, daIndex)
{
    // get molWeight and Cp from thermophysicalProperties
    if (mesh.thisDb().foundObject<IOdictionary>("thermophysicalProperties"))
    {
        const IOdictionary& thermoDict = mesh.thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
        dictionary mixSubDict = thermoDict.subDict("mixture");
        dictionary specieSubDict = mixSubDict.subDict("specie");
        molWeight_ = specieSubDict.getScalar("molWeight");
        dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
        Cp_ = thermodynamicsSubDict.getScalar("Cp");
        transportType_ = thermoDict.subDict("thermoType").getWord("transport");
        if (transportType_ == "sutherland")
        {
            As_ = mixSubDict.subDict("transport").getScalar("As");
            Ts_ = mixSubDict.subDict("transport").getScalar("Ts");
        }

        if (daOption_.getOption<label>("debug"))
        {
            Info << "molWeight " << molWeight_ << endl;
            Info << "Cp " << Cp_ << endl;
        }
    }
}

// * * * * * * * * * * * * * * * * * Selectors * * * * * * * * * * * * * * * //

autoPtr<DAResidual> DAResidual::New(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
{
    // standard setup for runtime selectable classes

    if (daOption.getAllOptions().lookupOrDefault<label>("debug", 0))
    {
        Info << "Selecting " << modelType << " for DAResidual" << endl;
    }

    auto* ctorPtr = dictionaryConstructorTable(modelType);

    if (!ctorPtr)
    {
        FatalErrorInLookup(
            "DAResidual",
            modelType,
            *dictionaryConstructorTablePtr_)
            << exit(FatalError);
    }

    return autoPtr<DAResidual>(
        ctorPtr(modelType, mesh, daOption, daModel, daIndex));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAResidual::masterFunction(
    const dictionary& options,
    const Vec xvVec,
    const Vec wVec,
    Vec resVec)
{
    /*
    Description:
        A master function that takes the volume mesh points and state variable vecs
        as input, and compute the residual vector.
    
    Input:
        options.updateState: whether to assign the values in wVec to the state
        variables of the OpenFOAM fields (e.g., U, p). This will also update boundary conditions
        and update all intermediate variables that are dependent on the state 
        variables. 

        options.updateMesh: whether to assign the values in xvVec to the OpenFOAM mesh 
        coordinates in Foam::fvMesh. This will also call mesh.movePoints() to update
        all the mesh metrics such as mesh volume, cell centers, mesh surface area, etc.

        options.setResVec: whether to assign the residuals (e.g., URes_) from the OpenFOAM fields
        to the resVec. If set to 0, nothing will be written to resVec

        options.isPC: whether to compute residual for constructing PC matrix

        xvVec: the volume coordinates vector (flatten)

        wVec: the state variable vector
    
    Output:
        resVec: the residual vector

    NOTE1: the ordering of the xvVec, wVec, and resVec depends on adjStateOrdering, see 
    Foam::DAIndex for details

    NOTE2: the calcResiduals function will be implemented in the child classes
    */

    VecZeroEntries(resVec);

    DAModel& daModel = const_cast<DAModel&>(daModel_);

    label updateState = options.getLabel("updateState");

    label updateMesh = options.getLabel("updateMesh");

    label setResVec = options.getLabel("setResVec");

    if (updateMesh)
    {
        FatalErrorIn("DAResidual::masterFunction")
            << "updateMesh=true not supported!"
            << abort(FatalError);
    }

    if (updateState)
    {
        daField_.stateVec2OFField(wVec);

        // now update intermediate states and boundry conditions
        this->correctBoundaryConditions();
        this->updateIntermediateVariables();
        daModel.correctBoundaryConditions();
        daModel.updateIntermediateVariables();
        // if there are special boundary conditions, apply special treatment
        daField_.specialBCTreatment();
    }

    this->calcResiduals(options);
    daModel.calcResiduals(options);

    if (setResVec)
    {
        // asssign the openfoam residual field to resVec
        daField_.ofResField2ResVec(resVec);
    }
}

void DAResidual::updateThermoVars()
{
    /* 
    Description:
        Update the thermo variables for compressible flow cases

        ********************** NOTE *****************
        we assume hePsiThermo, pureMixture, perfectGas, hConst, and const/sutherland transport
        TODO: need to do this using built-in openfoam functions.
    
        we need to:
        1, update psi based on T, psi=1/(R*T)
        2, update rho based on p and psi, rho=psi*p
        3, update E/H based on T, p and rho, E=Cp*T-p/rho, H=Cp*T
        4, update mu and alpha if the sutherland transport is used.
    */

    volScalarField& T = mesh_.thisDb().lookupObjectRef<volScalarField>("T");
    volScalarField& psi = mesh_.thisDb().lookupObjectRef<volScalarField>("thermo:psi");
    volScalarField& rho = mesh_.thisDb().lookupObjectRef<volScalarField>("rho");
    volScalarField& p = mesh_.thisDb().lookupObjectRef<volScalarField>("p");
    fluidThermo& thermo = mesh_.thisDb().lookupObjectRef<fluidThermo>("thermophysicalProperties");
    volScalarField& he = thermo.he();

    // 8314.4700665  gas constant in OpenFOAM
    // src/OpenFOAM/global/constants/thermodynamic/thermodynamicConstants.H
    scalar RR = Foam::constant::thermodynamic::RR;

    // R = RR/molWeight
    // Foam::specie::R() function in src/thermophysicalModels/specie/specie/specieI.H
    dimensionedScalar R(
        "R1",
        dimensionSet(0, 2, -2, -1, 0, 0, 0),
        RR / molWeight_);

    // psi = 1/T/R
    // see src/thermophysicalModels/specie/equationOfState/perfectGas/perfectGasI.H
    psi = 1.0 / T / R;

    // rho = psi*p
    // see src/thermophysicalModels/basic/psiThermo/psiThermo.C
    rho = psi * p;

    // **************** NOTE ****************
    // need to relax rho to be consistent with the primal solver
    // However, the rho.relax() will mess up perturbation
    // That being said, we comment out the rho.relax() call to
    // get the correct perturbed rho; however, the E residual will
    // be a bit off compared with the ERes at the converged state
    // from the primal solver. TODO. Need to figure out how to improve this
    // **************** NOTE ****************
    // rho_.relax();

    dimensionedScalar Cp(
        "Cp1",
        dimensionSet(0, 2, -2, -1, 0, 0, 0),
        Cp_);

    // Hs = Cp*T
    // see Hs() in src/thermophysicalModels/specie/thermo/hConst/hConstThermoI.H
    // here the H departure EquationOfState::H(p, T) will be zero for perfectGas
    // Es = Hs - p/rho = Hs - T * R;
    // see Es() in src/thermophysicalModels/specie/thermo/thermo/thermoI.H
    // **************** NOTE ****************
    // See the comment from the rho.relax() call, if we write he_=Cp*T-p/rho, the
    // accuracy of he_ may be impact by the inaccurate rho. So here we want to
    // rewrite he_ as he_ = Cp * T_ - T_ * R instead, such that we dont include rho
    // **************** NOTE ****************
    if (he.name() == "e")
    {
        he = Cp * T - T * R;
    }
    else
    {
        he = Cp * T;
    }
    he.correctBoundaryConditions();

    // update mu and alpha if the transportType is sutherland
    // codes adjusted from src/thermophysicalModels/specie/transport/sutherland/sutherlandTransportI.H
    if (transportType_ == "sutherland")
    {
        dimensionedScalar Cv(
            "Cv1",
            dimensionSet(0, 2, -2, -1, 0, 0, 0),
            0.0);
        Cv = Cp - R;
        volScalarField& mu = mesh_.thisDb().lookupObjectRef<volScalarField>("thermo:mu");
        volScalarField& alpha = mesh_.thisDb().lookupObjectRef<volScalarField>("thermo:alpha");
        forAll(mu, cellI)
        {
            mu[cellI] = As_ * sqrt(T[cellI]) / (1.0 + Ts_ / T[cellI]);
            alpha[cellI] = mu[cellI] * Cv.value() * (1.32 + 1.77 * R.value() / Cv.value()) / Cp.value();
        }
        mu.correctBoundaryConditions();
        alpha.correctBoundaryConditions();
    }

    // NOTE: alphat is updated in the correctNut function in DATurbulenceModel child classes
}

void DAResidual::calcPCMatWithFvMatrix(Mat PCMat)
{
    FatalErrorIn("DAResidual::calcPCMatWithFvMatrix")
        << "Child class not implemented!"
        << abort(FatalError);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
