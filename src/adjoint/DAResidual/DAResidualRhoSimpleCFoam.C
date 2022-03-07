/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

\*---------------------------------------------------------------------------*/

#include "DAResidualRhoSimpleCFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAResidualRhoSimpleCFoam, 0);
addToRunTimeSelectionTable(DAResidual, DAResidualRhoSimpleCFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAResidualRhoSimpleCFoam::DAResidualRhoSimpleCFoam(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : DAResidual(modelType, mesh, daOption, daModel, daIndex),
      // initialize and register state variables and their residuals, we use macros defined in macroFunctions.H
      setResidualClassMemberVector(U, dimensionSet(1, -2, -2, 0, 0, 0, 0)),
      setResidualClassMemberScalar(p, dimensionSet(1, -3, -1, 0, 0, 0, 0)),
      setResidualClassMemberScalar(T, dimensionSet(1, -1, -3, 0, 0, 0, 0)),
      setResidualClassMemberPhi(phi),
      thermo_(const_cast<fluidThermo&>(daModel.getThermo())),
      he_(thermo_.he()),
      rho_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("rho"))),
      alphat_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("alphat"))),
      psi_(const_cast<volScalarField&>(
          mesh_.thisDb().lookupObject<volScalarField>("thermo:psi"))),
      daTurb_(const_cast<DATurbulenceModel&>(daModel.getDATurbulenceModel())),
      // create simpleControl
      simple_(const_cast<fvMesh&>(mesh)),
      pressureControl_(p_, rho_, simple_.dict())
{

    // get molWeight and Cp from thermophysicalProperties
    const IOdictionary& thermoDict = mesh.thisDb().lookupObject<IOdictionary>("thermophysicalProperties");
    dictionary mixSubDict = thermoDict.subDict("mixture");
    dictionary specieSubDict = mixSubDict.subDict("specie");
    molWeight_ = specieSubDict.getScalar("molWeight");
    dictionary thermodynamicsSubDict = mixSubDict.subDict("thermodynamics");
    Cp_ = thermodynamicsSubDict.getScalar("Cp");

    if (daOption_.getOption<label>("debug"))
    {
        Info << "molWeight " << molWeight_ << endl;
        Info << "Cp " << Cp_ << endl;
    }
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void DAResidualRhoSimpleCFoam::clear()
{
    /*
    Description:
        Clear all members to avoid memory leak because we will initalize 
        multiple objects of DAResidual. Here we need to delete all members
        in the parent and child classes
    */
    URes_.clear();
    pRes_.clear();
    TRes_.clear();
    phiRes_.clear();
}

void DAResidualRhoSimpleCFoam::calcResiduals(const dictionary& options)
{
    /*
    Description:
        This is the function to compute residuals.
    
    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,U), div(phi,e)

        p_, T_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        URes_, pRes_, TRes_, phiRes_: residual field variables
    */

    // We dont support MRF and fvOptions so all the related lines are commented
    // out for now

    label isPC = options.getLabel("isPC");

    word divUScheme = "div(phi,U)";
    word divHEScheme = "div(phi,e)";
    word divPhidPScheme = "div(phid,p)";

    if (he_.name() == "h")
    {
        divHEScheme = "div(phi,h)";
    }

    if (isPC)
    {
        divUScheme = "div(pc)";
        divHEScheme = "div(pc)";
        divPhidPScheme = "div(pc)";
    }

    // ******** U Residuals **********
    // copied and modified from UEqn.H

    tmp<fvVectorMatrix> tUEqn(
        fvm::div(phi_, U_, divUScheme)
        + daTurb_.divDevRhoReff(U_));
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    URes_ = (UEqn & U_) + fvc::grad(p_);
    normalizeResiduals(URes);

    // ******** e Residuals **********
    // copied and modified from EEqn.H
    volScalarField alphaEff("alphaEff", thermo_.alphaEff(alphat_));

    fvScalarMatrix EEqn(
        fvm::div(phi_, he_, divHEScheme)
        + (he_.name() == "e"
               ? fvc::div(phi_, volScalarField("Ekp", 0.5 * magSqr(U_) + p_ / rho_))
               : fvc::div(phi_, volScalarField("K", 0.5 * magSqr(U_))))
        - fvm::laplacian(alphaEff, he_));

    EEqn.relax();

    TRes_ = EEqn & he_;
    normalizeResiduals(TRes);

    // ******** p and phi Residuals **********
    // copied and modified from pEqn.H
    volScalarField rAU(1.0 / UEqn.A());
    volScalarField rAtU(1.0 / (1.0 / rAU - UEqn.H1()));
    //volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
    //***************** NOTE *******************
    // constrainHbyA has been used since OpenFOAM-v1606; however, We do NOT use the constrainHbyA
    // function in DAFoam because we found it significantly degrades the accuracy of shape derivatives.
    // Basically, we should not constrain any variable because it will create discontinuity.
    // Instead, we use the old implementation used in OpenFOAM-3.0+ and before
    volVectorField HbyA("HbyA", U_);
    HbyA = rAU * UEqn.H();
    tUEqn.clear();

    surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho_) * fvc::flux(HbyA));

    volScalarField rhorAtU("rhorAtU", rho_ * rAtU);

    // NOTE: we don't support transonic = false

    surfaceScalarField phid(
        "phid",
        (fvc::interpolate(psi_) / fvc::interpolate(rho_)) * phiHbyA);

    phiHbyA +=
        fvc::interpolate(rho_ * (rAtU - rAU)) * fvc::snGrad(p_) * mesh_.magSf()
        - fvc::interpolate(psi_ * p_) * phiHbyA / fvc::interpolate(rho_);

    HbyA -= (rAU - rAtU) * fvc::grad(p_);

    fvScalarMatrix pEqn(
        fvc::div(phiHbyA)
        + fvm::div(phid, p_, divPhidPScheme)
        - fvm::laplacian(rhorAtU, p_));

    // for PC we do not include the div(phid, p) term, this improves the convergence
    if (isPC && daOption_.getOption<label>("transonicPCOption") == 1)
    {
        pEqn -= fvm::div(phid, p_, divPhidPScheme);
    }

    // Relax the pressure equation to maintain diagonal dominance
    pEqn.relax();

    pEqn.setReference(pressureControl_.refCell(), pressureControl_.refValue());

    pRes_ = pEqn & p_;
    normalizeResiduals(pRes);

    // ******** phi Residuals **********
    // copied and modified from pEqn.H
    phiRes_ = phiHbyA + pEqn.flux() - phi_;
    normalizePhiResiduals(phiRes);

}

void DAResidualRhoSimpleCFoam::updateIntermediateVariables()
{
    /* 
    Description:
        Update the intermediate variables that depend on the state variables

        ********************** NOTE *****************
        we assume hePsiThermo, pureMixture, perfectGas, hConst, and const transport
        TODO: need to do this using built-in openfoam functions.
    
        we need to:
        1, update psi based on T, psi=1/(R*T)
        2, update rho based on p and psi, rho=psi*p
        3, update E based on T, p and rho, E=Cp*T-p/rho
        4, update velocity boundary based on MRF (not yet implemented)
    */

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
    psi_ = 1.0 / T_ / R;

    // rho = psi*p
    // see src/thermophysicalModels/basic/psiThermo/psiThermo.C
    rho_ = psi_ * p_;

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
    if (he_.name() == "e")
    {
        he_ = Cp * T_ - T_ * R;
    }
    else
    {
        he_ = Cp * T_;
    }
    he_.correctBoundaryConditions();

    // NOTE: alphat is updated in the correctNut function in DATurbulenceModel child classes
}

void DAResidualRhoSimpleCFoam::correctBoundaryConditions()
{
    /* 
    Description:
        Update the boundary condition for all the states in the selected solver
    */

    U_.correctBoundaryConditions();
    p_.correctBoundaryConditions();
    T_.correctBoundaryConditions();
}

} // End namespace Foam

// ************************************************************************* //
