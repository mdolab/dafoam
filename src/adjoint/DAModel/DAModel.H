/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
        DAModel is a general class for handling physical models in DAFoam. 
        The constructor for DAModel does not explicitly contain objects of 
        physical models. Instead, it will look up any model class, e.g., 
        DATurbulenceModel, DARaditionModel, that has been registred in fvMesh. 
        This enables the flexibility such that we can simultaneously have 
        multiple models for adjoint, e.g., turbulence model + radiation model.
        
        NOTE 1: in adjoint functions, we should call functions in DAModel to handle
        general model operation, e.g., call daModel.correctBoundaryCondition(), 
        instead of calling daTurbulenceModel.correctBoundaryCondition(). Because
        daModel.correctBoundaryCondition() will call the correctBoundaryCondition()
        in daTurbulenceModel and daRadiationModel, if these two models are used.

        NOTE 2: Because of the above feature, we need to initialize any physical
        model BEFORE initializing the DAModel, e.g.,

        // initialize DATurbulence and automatically register it to fvMesh
        autoPtr<DATurbulenceModel> daTurb(DATurbulenceModel::New(mesh));
        
        ............ // codes to initialize any other physical models

        // now we can initialize DAModel and lookup DATurbulenceModel in mesh
        DAModel daModel(mesh);

        Otherwise, the physical models will NOT be picked up by the DAModel

\*---------------------------------------------------------------------------*/

#ifndef DAModel_H
#define DAModel_H

#include "fvOptions.H"
#include "DAOption.H"
#include "fvc.H"
#include "fvm.H"
#include "surfaceFields.H"
#include "geometricOneField.H"
#include "zeroGradientFvPatchField.H"

#ifndef SolidDASolver
#include "DATurbulenceModel.H"
#include "DARadiationModel.H"
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class DAModel Declaration
\*---------------------------------------------------------------------------*/

class DAModel
{

private:
    /// Disallow default bitwise copy construct
    DAModel(const DAModel&);

    /// Disallow default bitwise assignment
    void operator=(const DAModel&);

    /// Foam::fvMesh object
    const fvMesh& mesh_;

    /// DAOptoin object
    const DAOption& daOption_;

    /// have turbulence models for the flow and adjoint?
    label hasTurbulenceModel_ = 0;

    /// have radiation models for the flow and adjoint?
    label hasRadiationModel_ = 0;

public:
    /// Constructors
    DAModel(
        const fvMesh& mesh,
        const DAOption& daOption);

    /// Destructor
    virtual ~DAModel();

    /// update the name in modelStates based on the selected physical model at runtime
    void correctModelStates(wordList& modelStates) const;

    /// update the original variable connectivity for the adjoint state residuals in stateCon
    void correctStateResidualModelCon(List<List<word>>& stateCon) const;

    /// add the model residual connectivity to stateCon
    void addModelResidualCon(HashTable<List<List<word>>>& allCon) const;

    /// calculate the residuals for model state variables
    void calcResiduals(const dictionary& options);

    /// correct boundary conditions for model states
    void correctBoundaryConditions();

    /// update intermediate variables that are dependent on the model states
    void updateIntermediateVariables();

    /// return the value of the production term from the turbulence model 
    void getTurbProdTerm(scalarList& prodTerm) const;

#ifndef SolidDASolver
    /// get a reference to DATurbulenceModel
    const DATurbulenceModel& getDATurbulenceModel() const;
#endif

#ifdef CompressibleFlow
    /// return thermo_ from DATurbulenceModel
    const fluidThermo& getThermo() const;
#endif

};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
