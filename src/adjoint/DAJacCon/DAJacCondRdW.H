/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Child class for dRdW

\*---------------------------------------------------------------------------*/

#ifndef DAJacCondRdW_H
#define DAJacCondRdW_H

#include "DAJacCon.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAJacCondRdW Declaration
\*---------------------------------------------------------------------------*/

class DAJacCondRdW
    : public DAJacCon
{

protected:
    /// \name on- and off-diagonal preallocation sizes for each row of dRdW, dRdWT, and dRdXv
    //@{
    Vec dRdWTPreallocOn_;
    Vec dRdWTPreallocOff_;
    Vec dRdWPreallocOn_;
    Vec dRdWPreallocOff_;
    //@}

    /// initialize petsc vectors
    void initializePetscVecs();

    /// compute preallocation vectors
    void allocateJacobianConnections(
        Vec preallocOnProc,
        Vec preallocOffProc,
        Vec preallocOnProcT,
        Vec preallocOffProcT,
        Mat connections,
        const label row);

    /// compute preallocation vectors
    void preallocateJacobianMatrix(
        Mat dRMat,
        const Vec preallocOnProc,
        const Vec preallocOffProc) const;

    void setupdRdWCon(
        const HashTable<List<List<word>>>& stateResConInfo,
        const label isPrealloc);

public:
    TypeName("dRdW");
    // Constructors

    //- Construct from components
    DAJacCondRdW(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAJacCondRdW()
    {
    }

    // Member functions

    /// calculate the
    virtual void setupJacConPreallocation(const dictionary& options);

    /// initialize the state Jacobian connectivity matrix
    virtual void initializeJacCon(const dictionary& options);

    /// assign 1 to all non-zero elements for the Jacobian connecitivyt matrix
    virtual void setupJacCon(const dictionary& options);

    /// preallocate memory for dRdW using the computed preallocation vectors
    virtual void preallocatedRdW(
        Mat dRMat,
        const label transposed) const;
    
    /// clear members in parent and child objects
    virtual void clear();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
