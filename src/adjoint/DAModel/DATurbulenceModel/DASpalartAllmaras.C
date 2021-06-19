/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DASpalartAllmaras.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DASpalartAllmaras, 0);
addToRunTimeSelectionTable(DATurbulenceModel, DASpalartAllmaras, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DASpalartAllmaras::DASpalartAllmaras(
    const word modelType,
    const fvMesh& mesh,
    const DAOption& daOption)
    : DATurbulenceModel(modelType, mesh, daOption),
      // SA parameters
      sigmaNut_(dimensioned<scalar>::lookupOrAddToDict(
          "sigmaNut",
          this->coeffDict_,
          0.66666)),
      kappa_(dimensioned<scalar>::lookupOrAddToDict(
          "kappa",
          this->coeffDict_,
          0.41)),

      Cb1_(dimensioned<scalar>::lookupOrAddToDict(
          "Cb1",
          this->coeffDict_,
          0.1355)),
      Cb2_(dimensioned<scalar>::lookupOrAddToDict(
          "Cb2",
          this->coeffDict_,
          0.622)),
      Cw1_(Cb1_ / sqr(kappa_) + (1.0 + Cb2_) / sigmaNut_),
      Cw2_(dimensioned<scalar>::lookupOrAddToDict(
          "Cw2",
          this->coeffDict_,
          0.3)),
      Cw3_(dimensioned<scalar>::lookupOrAddToDict(
          "Cw3",
          this->coeffDict_,
          2.0)),
      Cv1_(dimensioned<scalar>::lookupOrAddToDict(
          "Cv1",
          this->coeffDict_,
          7.1)),
      Cs_(dimensioned<scalar>::lookupOrAddToDict(
          "Cs",
          this->coeffDict_,
          0.3)),
      // Augmented variables
      nuTilda_(const_cast<volScalarField&>(
          mesh.thisDb().lookupObject<volScalarField>("nuTilda"))),
      nuTildaRes_(
          IOobject(
              "nuTildaRes",
              mesh.time().timeName(),
              mesh,
              IOobject::NO_READ,
              IOobject::NO_WRITE),
          mesh,
#ifdef CompressibleFlow
          dimensionedScalar("nuTildaRes", dimensionSet(1, -1, -2, 0, 0, 0, 0), 0.0),
#endif
#ifdef IncompressibleFlow
          dimensionedScalar("nuTildaRes", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),
#endif
          zeroGradientFvPatchField<scalar>::typeName),
      y_(mesh.thisDb().lookupObject<volScalarField>("yWall"))
{

    // initialize printInterval_ we need to check whether it is a steady state
    // or unsteady primal solver
    IOdictionary fvSchemes(
        IOobject(
            "fvSchemes",
            mesh.time().system(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE,
            false));
    word ddtScheme = word(fvSchemes.subDict("ddtSchemes").lookup("default"));
    if (ddtScheme == "steadyState")
    {
        printInterval_ =
            daOption.getAllOptions().lookupOrDefault<label>("printInterval", 100);
    }
    else
    {
        printInterval_ =
            daOption.getAllOptions().lookupOrDefault<label>("printIntervalUnsteady", 500);
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// SA member functions. these functions are copied from
// src/TurbulenceModels/turbulenceModels/RAS/SpalartAllmaras/SpalartAllmaras.C
tmp<volScalarField> DASpalartAllmaras::chi() const
{
    return nuTilda_ / this->nu();
}

tmp<volScalarField> DASpalartAllmaras::fv1(
    const volScalarField& chi) const
{
    const volScalarField chi3(pow3(chi));
    return chi3 / (chi3 + pow3(Cv1_));
}

tmp<volScalarField> DASpalartAllmaras::fv2(
    const volScalarField& chi,
    const volScalarField& fv1) const
{
    return 1.0 - chi / (1.0 + chi * fv1);
}

tmp<volScalarField> DASpalartAllmaras::Stilda(
    const volScalarField& chi,
    const volScalarField& fv1) const
{
    volScalarField Omega(::sqrt(2.0) * mag(skew(fvc::grad(U_))));

    return (
        max(
            Omega
                + fv2(chi, fv1) * nuTilda_ / sqr(kappa_ * y_),
            Cs_ * Omega));
}

tmp<volScalarField> DASpalartAllmaras::fw(
    const volScalarField& Stilda) const
{
    volScalarField r(
        min(
            nuTilda_
                / (max(
                       Stilda,
                       dimensionedScalar("SMALL", Stilda.dimensions(), SMALL))
                   * sqr(kappa_ * y_)),
            scalar(10.0)));
    r.boundaryFieldRef() == 0.0;

    const volScalarField g(r + Cw2_ * (pow6(r) - r));

    return g * pow((1.0 + pow6(Cw3_)) / (pow6(g) + pow6(Cw3_)), 1.0 / 6.0);
}

tmp<volScalarField> DASpalartAllmaras::DnuTildaEff() const
{
    return tmp<volScalarField>(
        new volScalarField("DnuTildaEff", (nuTilda_ + this->nu()) / sigmaNut_));
}

// Augmented functions
void DASpalartAllmaras::correctModelStates(wordList& modelStates) const
{
    /*
    Description:
        Update the name in modelStates based on the selected physical model at runtime

    Example:
        In DAStateInfo, if the modelStates reads:
        
        modelStates = {"nut"}
        
        then for the SA model, calling correctModelStates(modelStates) will give:
    
        modelStates={"nuTilda"}
        
        while calling correctModelStates(modelStates) for the SST model will give 
        
        modelStates={"k","omega"}
        
        We don't udpate the names for the radiation model becasue users are 
        supposed to set modelStates={"G"}
    */

    // replace nut with nuTilda
    forAll(modelStates, idxI)
    {
        word stateName = modelStates[idxI];
        if (stateName == "nut")
        {
            modelStates[idxI] = "nuTilda";
        }
    }
}

void DASpalartAllmaras::correctNut()
{
    /*
    Description:
        Update nut based on other turbulence variables and update the BCs
        Also update alphat if is present
    */

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));
    nut_ = nuTilda_ * fv1;

    nut_.correctBoundaryConditions();

    // this is basically BasicTurbulenceModel::correctNut();
    this->correctAlphat();

    return;
}

void DASpalartAllmaras::correctBoundaryConditions()
{
    /*
    Description:
        Update turbulence variable boundary values
    */

    // correct the BCs for the perturbed fields
    nuTilda_.correctBoundaryConditions();
}

void DASpalartAllmaras::updateIntermediateVariables()
{
    /*
    Description:
        Update nut based on nuTilda. Note: we need to update nut and its BC since we 
        may have perturbed other turbulence vars that affect the nut values
    */

    this->correctNut();
}

void DASpalartAllmaras::correctStateResidualModelCon(List<List<word>>& stateCon) const
{
    /*
    Description:
        Update the original variable connectivity for the adjoint state 
        residuals in stateCon. Basically, we modify/add state variables based on the
        original model variables defined in stateCon.

    Input:
    
        stateResCon: the connectivity levels for a state residual, defined in Foam::DAJacCon

    Example:
        If stateCon reads:
        stateCon=
        {
            {"U", "p", "nut"},
            {"p"}
        }
    
        For the SA turbulence model, calling this function for will get a new stateCon
        stateCon=
        {
            {"U", "p", "nuTilda"},
            {"p"}
        }
    
        For the SST turbulence model, calling this function will give
        stateCon=
        {
            {"U", "p", "k", "omega"},
            {"p", "U"}
        }
        ***NOTE***: we add a extra level of U connectivity because nut is 
        related to grad(U), k, and omega in SST!
    */

    forAll(stateCon, idxI)
    {
        forAll(stateCon[idxI], idxJ)
        {
            word conStateName = stateCon[idxI][idxJ];
            if (conStateName == "nut")
            {
                stateCon[idxI][idxJ] = "nuTilda";
            }
        }
    }
}

void DASpalartAllmaras::addModelResidualCon(HashTable<List<List<word>>>& allCon) const
{
    /*
    Description:
        Add the connectivity levels for all physical model residuals to allCon

    Input:
        allCon: the connectivity levels for all state residual, defined in DAJacCon

    Example:
        If stateCon reads:
        allCon=
        {
            "URes":
            {
               {"U", "p", "nut"},
               {"p"}
            }
        }
    
        For the SA turbulence model, calling this function for will get a new stateCon,
        something like this:
        allCon=
        {
            "URes":
            {
               {"U", "p", "nuTilda"},
               {"p"}
            },
            "nuTildaRes": 
            {
                {"U", "phi", "nuTilda"},
                {"U"}
            }
        }

    */

    word pName;

    if (mesh_.thisDb().foundObject<volScalarField>("p"))
    {
        pName = "p";
    }
    else if (mesh_.thisDb().foundObject<volScalarField>("p_rgh"))
    {
        pName = "p_rgh";
    }
    else
    {
        FatalErrorIn(
            "Neither p nor p_rgh was found in mesh.thisDb()!"
            "addModelResidualCon failed to setup turbulence residuals!")
            << exit(FatalError);
    }

    // NOTE: for compressible flow, it depends on rho so we need to add T and p
#ifdef IncompressibleFlow
    allCon.set(
        "nuTildaRes",
        {
            {"U", "nuTilda", "phi"}, // lv0
            {"U", "nuTilda"}, // lv1
            {"nuTilda"} // lv2
        });
#endif

#ifdef CompressibleFlow
    allCon.set(
        "nuTildaRes",
        {
            {"U", "T", pName, "nuTilda", "phi"}, // lv0
            {"U", "T", pName, "nuTilda"}, // lv1
            {"T", pName, "nuTilda"} // lv2
        });
#endif
}

void DASpalartAllmaras::correct()
{
    /*
    Descroption:
        Solve the residual equations and update the state. This function will be called 
        by the DASolver. It is needed because we want to control the output frequency
        of the residual convergence every 100 steps. If using the correct from turbulence
        it will output residual every step which will be too much of information.
    */

    // We set the flag solveTurbState_ to 1 such that in the calcResiduals function
    // we will solve and update nuTilda
    solveTurbState_ = 1;
    dictionary dummyOptions;
    this->calcResiduals(dummyOptions);
    // after it, we reset solveTurbState_ = 0 such that calcResiduals will not
    // update nuTilda when calling from the adjoint class, i.e., solveAdjoint from DASolver.
    solveTurbState_ = 0;
}

void DASpalartAllmaras::calcResiduals(const dictionary& options)
{
    /*
    Descroption:
        If solveTurbState_ == 1, this function solve and update nuTilda, and 
        is the same as calling turbulence.correct(). If solveTurbState_ == 0,
        this function compute residuals for turbulence variables, e.g., nuTildaRes_

    Input:
        options.isPC: 1 means computing residuals for preconditioner matrix.
        This essentially use the first order scheme for div(phi,nuTilda)

        p_, U_, phi_, etc: State variables in OpenFOAM
    
    Output:
        nuTildaRes_: If solveTurbState_ == 0, update the residual field variable

        nuTilda_: If solveTurbState_ == 1, update nuTilda
    */

    // Copy and modify based on the "correct" function

    label printToScreen = this->isPrintTime(mesh_.time(), printInterval_);

    word divNuTildaScheme = "div(phi,nuTilda)";

    label isPC = 0;

    if (!solveTurbState_)
    {
        isPC = options.getLabel("isPC");

        if (isPC)
        {
            divNuTildaScheme = "div(pc)";
        }
    }

    const volScalarField chi(this->chi());
    const volScalarField fv1(this->fv1(chi));

    const volScalarField Stilda(this->Stilda(chi, fv1));

    tmp<fvScalarMatrix> nuTildaEqn(
        fvm::ddt(phase_, rho_, nuTilda_)
            + fvm::div(phaseRhoPhi_, nuTilda_, divNuTildaScheme)
            - fvm::laplacian(phase_ * rho_ * DnuTildaEff(), nuTilda_)
            - Cb2_ / sigmaNut_ * phase_ * rho_ * magSqr(fvc::grad(nuTilda_))
        == Cb1_ * phase_ * rho_ * Stilda * nuTilda_
            - fvm::Sp(Cw1_ * phase_ * rho_ * fw(Stilda) * nuTilda_ / sqr(y_), nuTilda_));

    nuTildaEqn.ref().relax();

    if (solveTurbState_)
    {

        // get the solver performance info such as initial
        // and final residuals
        SolverPerformance<scalar> solverNuTilda = solve(nuTildaEqn);
        if (printToScreen)
        {
            Info << "nuTilda Initial residual: " << solverNuTilda.initialResidual() << endl
                 << "          Final residual: " << solverNuTilda.finalResidual() << endl;
        }

        DAUtility::boundVar(allOptions_, nuTilda_, printToScreen);
        nuTilda_.correctBoundaryConditions();

        // ***************** NOTE*****************
        // In the original SA, it is correctNut(fv1) and fv1 is not
        // updated based on the latest nuTilda. We use correctNut which
        // recompute fv1 with the latest nuTilda
        this->correctNut();
    }
    else
    {
        // calculate residuals
        nuTildaRes_ = nuTildaEqn.ref() & nuTilda_;
        // need to normalize residuals
        normalizeResiduals(nuTildaRes);
    }

    return;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
