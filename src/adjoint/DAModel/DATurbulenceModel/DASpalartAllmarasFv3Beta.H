/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
    Child class for the SpalartAllmarasFv3 model with a betaSA field multiplying 
    to the production term. This betaSA term can then be trained to improve the
    SA turbulence model.

\*---------------------------------------------------------------------------*/

#ifndef DASpalartAllmarasFv3Beta_H
#define DASpalartAllmarasFv3Beta_H

#include "DATurbulenceModel.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DASpalartAllmarasFv3Beta Declaration
\*---------------------------------------------------------------------------*/

class DASpalartAllmarasFv3Beta
    : public DATurbulenceModel
{

protected:
    /// \name SA parameters
    //@{
    dimensionedScalar sigmaNut_;
    dimensionedScalar kappa_;
    dimensionedScalar Cb1_;
    dimensionedScalar Cb2_;
    dimensionedScalar Cw1_;
    dimensionedScalar Cw2_;
    dimensionedScalar Cw3_;
    dimensionedScalar Cv1_;
    dimensionedScalar Cv2_;
    //@}

    /// \name SA functions
    //@{
    tmp<volScalarField> chi() const;

    tmp<volScalarField> fv1(const volScalarField& chi) const;

    tmp<volScalarField> fv2(
        const volScalarField& chi,
        const volScalarField& fv1) const;

    tmp<volScalarField> fv3(
        const volScalarField& chi,
        const volScalarField& fv1) const;

    tmp<volScalarField> fw(const volScalarField& Stilda) const;

    //@}

    /// \name Augmented variables for residual computation
    //@{
    volScalarField& nuTilda_;
    volScalarField nuTildaRes_;
    //@}

    /// A beta field multiplying to the production term
    volScalarField& betaSA_; 

    /// a reference beta field
    volScalarField& betaSATrue_; 

    /// reference velocity
    volVectorField& UTrue_; 

    /// 3D wall distance
    const volScalarField& y_;

    /// whether to solve for turb states
    label solveTurbState_ = 0;

    /// time step interval to print residual
    label printInterval_;

public:
    TypeName("SpalartAllmarasFv3Beta");
    // Constructors

    //- Construct from components
    DASpalartAllmarasFv3Beta(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DASpalartAllmarasFv3Beta()
    {
    }

    // Member functions

    /// Return the effective diffusivity for nuTilda
    tmp<volScalarField> DnuTildaEff() const;

    /// update the turbulence state for DAStateInfo::regStates_
    virtual void correctModelStates(wordList& modelStates) const;

    /// update nut based on other turbulence variables and update the BCs
    virtual void correctNut();

    /// update turbulence variable boundary values
    virtual void correctBoundaryConditions();

    /// update any intermediate variables that are dependent on state variables and are used in calcResiduals
    virtual void updateIntermediateVariables();

    /// update the original variable connectivity for the adjoint state residuals in stateCon
    virtual void correctStateResidualModelCon(List<List<word>>& stateCon) const;

    /// add the model residual connectivity to stateCon
    virtual void addModelResidualCon(HashTable<List<List<word>>>& allCon) const;

    /// compute the turbulence residuals
    virtual void calcResiduals(const dictionary& options);

    /// solve the residual equations and update the state
    virtual void correct();

    /// return the value of the production term from the turbulence model 
    virtual void getTurbProdTerm(scalarList& prodTerm) const;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
