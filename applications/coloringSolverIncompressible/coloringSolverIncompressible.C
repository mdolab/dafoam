/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.0

    Description:
    Parallel distance 2 graph coloring solver for Incompressible adjoint solvers

\*---------------------------------------------------------------------------*/

static char help[] = "Solves a linear system in parallel with KSP in OpenFOAM.\n\n";

#include <petscksp.h>
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "AdjointIO.H"
#include "AdjointSolverRegistry.H"
#include "AdjointRASModel.H"
#include "AdjointIndexing.H"
#include "AdjointJacobianConnectivity.H"
#include "nearWallDist.H"
#include "argList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addOption
    (
        "onlyColor",
        "(dRdW ... dFdW)",
        "run coloring for selected partial derivatives only"
    );

    #include "postProcess.H"
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // Initialize the petsc solver. This needs to be called after the case
    // setup so that petsc uses the OpenFOAM MPI_COMM
    PetscInitialize(&argc,&argv,(char*)0,help);

    #include "createFields.H"

    // read onlyColor option
    List<word> onlyColor;
    if (args.optionFound("onlyColor"))
    {
        onlyColor = wordList(args.optionLookup("onlyColor")());
    }

    // dRdW
    if (!args.optionFound("onlyColor") || adjIO.isInList<word>("dRdW",onlyColor))
    {
        adjCon->setupdRdWCon(1);
        adjCon->initializedRdWCon();
        adjCon->setupdRdWCon(0);
        Info<<"dRdWCon Created. " <<mesh.time().elapsedClockTime()<<" s"<<endl;
        Info<<"Calculating dRdW Coloring... " <<mesh.time().elapsedClockTime()<<" s"<<endl;
        adjCon->calcdRdWColoring(); 
        Info<<"Calculating dRdW Coloring... Completed! " <<mesh.time().elapsedClockTime()<<" s"<<endl;  
        adjCon->deletedRdWCon();
    }
    
    // dFdW
    if (!args.optionFound("onlyColor") || adjIO.isInList<word>("dFdW",onlyColor))
    {
        forAll(adjIO.objFuncs,idxI)
        {
            word objFunc = adjIO.objFuncs[idxI];
            Info<<"Calculating dFdW Coloring for "+objFunc<<endl; 
            adjCon->initializedFdWCon(objFunc);
            adjCon->setupObjFuncCon(objFunc,"dFdW");
            adjCon->calcdFdWColoring(objFunc); 
            adjCon->deletedFdWCon();
        }
    }

    if(adjIO.isInList<word>("Xv",adjIO.adjDVTypes))
    {
        // dRdXv
        if (!args.optionFound("onlyColor") || adjIO.isInList<word>("dRdXv",onlyColor))
        {
            adjCon->setupdRdXvCon(1);
            adjCon->initializedRdXvCon();
            adjCon->setupdRdXvCon(0);
            Info<<"dRdXvCon Created. " <<mesh.time().elapsedClockTime()<<" s"<<endl;
            Info<<"Calculating dRdXv Coloring... " <<mesh.time().elapsedClockTime()<<" s"<<endl;
            adjCon->calcdRdXvColoring(); 
            Info<<"Calculating dRdXv Coloring... Completed! " <<mesh.time().elapsedClockTime()<<" s"<<endl;  
            adjCon->deletedRdXvCon();
        }

        // dFdXv
        if (!args.optionFound("onlyColor") || adjIO.isInList<word>("dFdXv",onlyColor))
        {
             forAll(adjIO.objFuncs,idxI)
             {
                 word objFunc = adjIO.objFuncs[idxI];
                 Info<<"Calculating dFdXv Coloring for "+objFunc<<endl;
                 adjCon->initializedFdXvCon(objFunc);  
                 adjCon->setupObjFuncCon(objFunc,"dFdXv");
                 adjCon->calcdFdXvColoring(objFunc); 
                 adjCon->deletedFdXvCon();
             }
        }
    }
 
    return 0;
}


// ************************************************************************* //
