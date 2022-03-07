/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v3

    Description:
    Augmented radiation model for the adjoint method, including residual 
    calculation functions, etc

\*---------------------------------------------------------------------------*/

#ifndef DARadiationModel_H
#define DARadiationModel_H

#include "runTimeSelectionTables.H"
#include "DAOption.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class DARadiationModel Declaration
\*---------------------------------------------------------------------------*/

class DARadiationModel
    : public regIOobject
{

private:
    //- Disallow default bitwise copy construct
    DARadiationModel(const DARadiationModel&);

    //- Disallow default bitwise assignment
    void operator=(const DARadiationModel&);

protected:
    const fvMesh& mesh_;

    const DAOption& daOption_;

public:
    //- Runtime type information
    TypeName("DARadiationModel");

    // Declare run-time constructor selection table
    declareRunTimeSelectionTable(
        autoPtr,
        DARadiationModel,
        dictionary,
        (const word modelType, 
         const fvMesh& mesh, 
         const DAOption& daOption),
        (modelType, mesh, daOption));

    // Constructors

    //- Construct from components
    DARadiationModel(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    // Selectors

    //- Return a reference to the selected model
    static autoPtr<DARadiationModel> New(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption);

    //- Destructor
    virtual ~DARadiationModel()
    {
    }

    // Members

    /// this is a virtual function for regIOobject
    bool writeData(Ostream& os) const;

    virtual void addModelResidualCon(HashTable<List<List<word>>>& allCon) const = 0;
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
