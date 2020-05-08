/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v1.1

\*---------------------------------------------------------------------------*/
#include "ColoringSolverCompressible.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Constructors
ColoringSolverCompressible::ColoringSolverCompressible(char* argsAll)
{
    argsAll_ = argsAll;
}

ColoringSolverCompressible::~ColoringSolverCompressible()
{
}

void ColoringSolverCompressible::run()
{

    argList::addOption
    (
        "onlyColor",
        "(dRdW ... dFdW)",
        "run coloring for selected partial derivatives only"
    );

    #include "setArgs.H"
    #include "setRootCasePython.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"
    #include "createAdjoint.H"

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
   

    return;

}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
