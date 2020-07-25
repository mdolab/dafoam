/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

    Description:
        Set the arguments for wrapping OpenFOAM with Cython.
        For an OpenFOAM lib, we only pass the argsAll parameter, so here we need
        to convert argsAll to argv and argc for the main function

\*---------------------------------------------------------------------------*/

// create argc and argv, given argsAll
List<word> argvList;

char* pch;
pch = strtok(argsAll_, " ");
int argc = 0;
while (pch != NULL)
{
    argc++;
    argvList.append(pch);
    pch = strtok(NULL, " ");
}

char** argv = new char*[argc];
for (int i = 0; i < argc; i++)
{
    argv[i] = const_cast<char*>(argvList[i].c_str());
}
//argv = n_argv;
