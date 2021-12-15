/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v2

\*---------------------------------------------------------------------------*/

#include "DAColoring.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAColoring::DAColoring(
    const fvMesh& mesh,
    const DAOption& daOption,
    const DAModel& daModel,
    const DAIndex& daIndex)
    : mesh_(mesh),
      daOption_(daOption),
      daIndex_(daIndex)
{
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void DAColoring::parallelD2Coloring(
    const Mat conMat,
    Vec colors,
    label& nColors) const
{
    /*
    Description:
        A general function to compute coloring for a Jacobian matrix using a 
        paralel heuristic distance 2 algorithm

    Input:
        conMat: a Petsc matrix that have the connectivity pattern (value one for 
        all nonzero elements)

    Output:
        colors: the coloring vector to store the coloring indices, starting with 0
        
        nColors: the number of colors
    
        Example:
        If the conMat reads,
    
               color0  color1
                 |     |
                 1  0  0  0
        conMat = 0  1  1  0
                 0  0  1  0
                 0  0  0  1
                    |     | 
                color0   color0
    
        Then, calling this function gives colors = {0, 0, 1, 0}.
        This can be done for parallel conMat
    */

    // if we end up having more than 10000 colors, something must be wrong
    label maxColors = 10000;

    PetscInt nCols, nCols2;
    const PetscInt* cols;
    const PetscScalar* vals;
    const PetscScalar* vals2;

    PetscInt colorStart, colorEnd;
    VecScatter colorScatter;

    label Istart, Iend;
    label currColor;
    label notColored = 1;
    IS globalIS;
    label maxCols = 750;
    scalar allNonZeros;
    Vec globalVec;
    PetscInt nRowG, nColG;

    Info << "Parallel Distance 2 Graph Coloring...." << endl;

    // initialize the number of colors to zero
    nColors = 0;

    // get the range of colors owned by the local prock
    VecGetOwnershipRange(colors, &colorStart, &colorEnd);

    // Set the entire color vector to -1
    VecSet(colors, -1);

    // Determine which rows are on the current processor
    MatGetOwnershipRange(conMat, &Istart, &Iend);

    //then get the global number of rows and columns
    MatGetSize(conMat, &nRowG, &nColG);
    label nRowL = Iend - Istart;
    label nColL = colorEnd - colorStart;

    /* 
    Start by looping over the rows to determine the largest
    number of non-zeros per row. This will determine maxCols
    and the minumum bound for the number of colors.
    */
    this->getMatNonZeros(conMat, maxCols, allNonZeros);
    Info << "MaxCols: " << maxCols << endl;
    Info << "AllNonZeros: " << allNonZeros << endl;

    // Create a local sparse matrix with a single row to use as a sparse vector
    Mat localCols;
    MatCreateSeqAIJ(
        PETSC_COMM_SELF,
        1,
        daIndex_.nGlobalAdjointStates,
        daIndex_.nLocalAdjointStates,
        NULL,
        &localCols);
    MatSetOption(localCols, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetUp(localCols);
    MatZeroEntries(localCols);

    //Now loop over the owned rows and set the value in any occupied col to 1.
    PetscInt idxI = 0;
    PetscScalar v = 1;
    for (label i = Istart; i < Iend; i++)
    {
        MatGetRow(conMat, i, &nCols, &cols, &vals);
        // set any columns that have a nonzero entry into localCols
        for (label j = 0; j < nCols; j++)
        {
            if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
            {
                PetscInt idx = cols[j];
                MatSetValues(localCols, 1, &idxI, 1, &idx, &v, INSERT_VALUES);
            }
        }
        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(localCols, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(localCols, MAT_FINAL_ASSEMBLY);

    // now localCols contains the unique set of local columns on each processor
    label nUniqueCols = 0;
    MatGetRow(localCols, 0, &nCols, &cols, &vals);
    for (label j = 0; j < nCols; j++)
    {
        if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
        {
            nUniqueCols++;
        }
    }
    Info << "nUniqueCols: " << nUniqueCols << endl;

    //Loop over the local vectors and set nonzero entries in a global vector
    // This lets us determine which columns are strictly local and which have
    // interproccessor overlap.
    VecCreate(PETSC_COMM_WORLD, &globalVec);
    VecSetSizes(globalVec, nColL, PETSC_DECIDE);
    VecSetFromOptions(globalVec);
    VecSet(globalVec, 0);

    for (label j = 0; j < nCols; j++)
    {
        PetscInt idx = cols[j];
        if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
        {
            VecSetValue(globalVec, idx, vals[j], ADD_VALUES);
        }
    }
    VecAssemblyBegin(globalVec);
    VecAssemblyEnd(globalVec);

    MatRestoreRow(localCols, 0, &nCols, &cols, &vals);

    // now create an index set of the strictly local columns
    label* localColumnStat = new label[nUniqueCols];
    label* globalIndexList = new label[nUniqueCols];

    PetscScalar* globalVecArray;
    VecGetArray(globalVec, &globalVecArray);

    label colCounter = 0;
    MatGetRow(localCols, 0, &nCols, &cols, &vals);
    for (label j = 0; j < nCols; j++)
    {
        label col = cols[j];
        if (DAUtility::isValueCloseToRef(vals[j], 1.0))
        {
            globalIndexList[colCounter] = col;
            localColumnStat[colCounter] = 1;
            if (col >= colorStart && col < colorEnd)
            {
                if (DAUtility::isValueCloseToRef(globalVecArray[col - colorStart], 1.0))
                {
                    localColumnStat[colCounter] = 2; // 2: strictly local
                }
            }
            colCounter++;
        }
    }
    MatRestoreRow(localCols, 0, &nCols, &cols, &vals);
    MatDestroy(&localCols);

    // create a list of the rows that have any of the strictly local columns included

    label* localRowList = new label[nRowL];
    for (label i = Istart; i < Iend; i++)
    {
        label idx = i - Istart;
        MatGetRow(conMat, i, &nCols, &cols, &vals);
        //check if this row has any strictly local columns

        /*we know that our global index lists are stored sequentially, so we
          don't need to start every index search at zero, we can start at the
          last entry found in this row */
        label kLast = -1;
        for (label j = 0; j < nCols; j++)
        {
            label matCol = cols[j];
            if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
            {
                //Info<<"j: "<<j<<vals[j]<<endl;
                kLast = this->find_index(matCol, kLast + 1, nUniqueCols, globalIndexList);

                if (kLast >= 0)
                {
                    //k was found
                    label localVal = localColumnStat[kLast];
                    if (localVal == 2)
                    {
                        localRowList[idx] = 1;
                        break;
                    }
                    else
                    {
                        localRowList[idx] = 0;
                    }
                }
                else
                {
                    localRowList[idx] = 0;
                }
            }
        }
        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }
    VecRestoreArray(globalVec, &globalVecArray);

    /* Create the scatter context for the remainder of the function */
    Vec colorsLocal;
    // create a scatter context for these colors
    VecCreateSeq(PETSC_COMM_SELF, nUniqueCols, &colorsLocal);
    VecSet(colorsLocal, -1);

    // now create the Index sets
    ISCreateGeneral(PETSC_COMM_WORLD, nUniqueCols, globalIndexList, PETSC_COPY_VALUES, &globalIS);
    // Create the scatter
    VecScatterCreate(colors, globalIS, colorsLocal, NULL, &colorScatter);

    /* Create the conflict resolution scheme*/
    // create tiebreakers locally
    Vec globalTiebreaker;
    VecDuplicate(globalVec, &globalTiebreaker);
    for (label i = colorStart; i < colorEnd; i++)
    {
        srand(i);
        PetscScalar val = rand() % nColG;
        VecSetValue(globalTiebreaker, i, val, INSERT_VALUES);
    }

    // and scatter the random values
    Vec localTiebreaker;
    VecDuplicate(colorsLocal, &localTiebreaker);
    VecScatterBegin(colorScatter, globalTiebreaker, localTiebreaker, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(colorScatter, globalTiebreaker, localTiebreaker, INSERT_VALUES, SCATTER_FORWARD);

    //initialize conflict columns
    label* conflictCols = new label[maxCols];
    label* conflictLocalColIdx = new label[maxCols];
    for (label j = 0; j < maxCols; j++)
    {
        conflictCols[j] = -1;
        conflictLocalColIdx[j] = -1;
    }

    // Create a global distrbuted vector of the only the local portion of
    // localcolumnsstatus
    Vec globalColumnStat;
    VecDuplicate(colors, &globalColumnStat);
    VecSet(globalColumnStat, 0.0);
    for (label k = 0; k < nUniqueCols; k++)
    {
        label localCol = globalIndexList[k];
        PetscScalar localVal = localColumnStat[k];
        if (localCol >= colorStart && localCol < colorEnd)
        {
            VecSetValue(globalColumnStat, localCol, localVal, INSERT_VALUES);
        }
    }
    VecAssemblyBegin(globalColumnStat);
    VecAssemblyEnd(globalColumnStat);

    /*
      create a duplicate matrix for conMat that contains its index into the
      local arrays
    */
    Mat conIndMat;
    MatDuplicate(conMat, MAT_SHARE_NONZERO_PATTERN, &conIndMat);

    // now loop over conMat locally, find the index in the local array
    // and store that value in conIndMat
    for (label i = Istart; i < Iend; i++)
    {
        MatGetRow(conMat, i, &nCols, &cols, &vals);
        label kLast = -1;
        for (label j = 0; j < nCols; j++)
        {
            label matCol = cols[j];
            if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
            {
                kLast = this->find_index(matCol, kLast + 1, nUniqueCols, globalIndexList);
                PetscScalar val = kLast;
                MatSetValue(conIndMat, i, matCol, val, INSERT_VALUES);
            }
        }
        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }
    MatAssemblyBegin(conIndMat, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(conIndMat, MAT_FINAL_ASSEMBLY);

    /*Now color the locally independent columns using the only the rows
      that contain entries from those columns.*/

    // Retrieve the local portion of the color vector
    PetscScalar *colColor, *tbkrLocal, *tbkrGlobal, *globalStat;
    VecGetArray(colors, &colColor);
    VecGetArray(localTiebreaker, &tbkrLocal);
    VecGetArray(globalTiebreaker, &tbkrGlobal);
    VecGetArray(globalColumnStat, &globalStat);

    // Loop over the maximum number of colors
    label printInterval = daOption_.getOption<label>("printInterval");
    for (label n = 0; n < maxColors; n++)
    {
        if (n % printInterval == 0)
        {
            Info << "ColorSweep: " << n << "   " << mesh_.time().elapsedClockTime() << " s" << endl;
        }

        /* Set all entries for strictly local columns that are currently -1
           to the current color */
        for (label k = 0; k < nUniqueCols; k++)
        {
            label localCol = globalIndexList[k];
            label localVal = localColumnStat[k];
            if (localCol >= colorStart && localCol < colorEnd && localVal == 2)
            {
                // this is a strictly local column;
                label idx = localCol - colorStart;
                if (DAUtility::isValueCloseToRef(colColor[idx], -1.0))
                {
                    colColor[idx] = n;
                }
            }
        }

        //Now loop over the rows and resolve conflicts
        for (label i = Istart; i < Iend; i++)
        {

            // Get the row local row index
            label idx = i - Istart;

            //create the variables for later sorting
            label smallest = nColG;
            label idxKeep = -1;

            //First check if this is a row that contains strictly local columns
            if (localRowList[idx] > 0)
            {

                /* this is a row that contains strictly local columns,get the
                   row information */
                MatGetRow(conMat, i, &nCols, &cols, &vals);

                // set any columns with the current color into conflictCols
                for (label j = 0; j < nCols; j++)
                {
                    if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
                    {
                        label colIdx = cols[j];

                        // Check that this is a local column
                        if (colIdx >= colorStart && colIdx < colorEnd)
                        {
                            //now check if it is a strictly local column
                            label localVal = globalStat[colIdx - colorStart];
                            if (localVal == 2)
                            {
                                // check if the color in this column is from the
                                // current set
                                if (DAUtility::isValueCloseToRef(colColor[colIdx - colorStart], n * 1.0))
                                {
                                    /* This is a potentially conflicting column
                                       store it */
                                    conflictCols[j] = colIdx;

                                    // now check whether this is the one we keep
                                    label tbkr = tbkrGlobal[colIdx - colorStart];
                                    if (tbkr < smallest)
                                    {
                                        smallest = tbkr;
                                        idxKeep = colIdx;
                                    }
                                }
                            }
                        }
                    }
                }

                // Now reset all columns but the one that wins the tiebreak
                for (label j = 0; j < nCols; j++)
                {

                    //check if this is a conflicting column
                    label colIdx = conflictCols[j];
                    if (colIdx >= 0)
                    {
                        // Check that this is also a local column
                        if (colIdx >= colorStart && colIdx < colorEnd)
                        {
                            // and now if it is a strictly local column
                            label localVal = globalStat[colIdx - colorStart];
                            if (localVal == 2)
                            {
                                // now reset the column
                                if (colIdx >= 0 && (colIdx != idxKeep))
                                {
                                    colColor[colIdx - colorStart] = -1;
                                }
                            }
                        }
                    }
                }
                // reset the changed values in conflictCols
                for (label j = 0; j < nCols; j++)
                {
                    if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
                    {
                        //reset all values related to this row in conflictCols
                        conflictCols[j] = -1;
                    }
                }
                MatRestoreRow(conMat, i, &nCols, &cols, &vals);
            }
        }

        // now we want to check the coloring on the strictly local columns
        notColored = 0;

        //loop over the columns and check if there are any uncolored rows
        label colorCounter = 0;
        for (label k = 0; k < nUniqueCols; k++)
        {
            // get the column info
            label localVal = localColumnStat[k];
            label localCol = globalIndexList[k];
            // check if it is strictly local, if so it should be colored
            if (localVal == 2)
            {
                // confirm that it is a local column (is this redundant?
                if (localCol >= colorStart && localCol < colorEnd)
                {
                    label idx = localCol - colorStart;
                    label color = colColor[idx];
                    // now check that it has been colored
                    if (not(color >= 0))
                    {
                        // this column is not colored and coloring is not complete
                        notColored = 1;
                        colorCounter++;
                        //break;
                    }
                }
            }
        }

        // reduce the logical so that we know that all of the processors are
        // ok
        reduce(notColored, sumOp<label>());
        reduce(colorCounter, sumOp<label>());

        if (n % printInterval == 0)
        {
            Info << "number of uncolored: " << colorCounter << " " << notColored << endl;
        }

        if (notColored == 0)
        {
            Info << "ColorSweep: " << n << "   " << mesh_.time().elapsedClockTime() << " s" << endl;
            Info << "number of uncolored: " << colorCounter << " " << notColored << endl;
            break;
        }
    }
    VecRestoreArray(colors, &colColor);
    /***** end of local coloring ******/

    // now redo the local row list to handle the global columns
    // create a list of the rows that have any of the global columns included

    for (label i = Istart; i < Iend; i++)
    {
        label idx = i - Istart;
        // get the row information
        MatGetRow(conMat, i, &nCols, &cols, &vals);

        //check if this row has any non-local columns

        /* We know that our localColumnStat is stored sequentially, so we don't
           need to start every index search at zero, we can start at the last
           entry found in this row.*/
        label kLast = -1;

        for (label j = 0; j < nCols; j++)
        {
            // get the column of interest
            label matCol = cols[j];
            // confirm that it has an entry
            if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
            {
                // find the index into the local arrays for this column
                kLast = this->find_index(matCol, kLast + 1, nUniqueCols, globalIndexList);
                // if this column is present (should always be true?) process the row
                if (kLast >= 0)
                {
                    // get the local column type
                    label localVal = localColumnStat[kLast];
                    /* If this is a global column, add the row and move to the next
                       row, otherwise check the next column in the row */
                    if (localVal == 1)
                    {
                        localRowList[idx] = 1;
                        break;
                    }
                    else
                    {
                        localRowList[idx] = 0;
                    }
                }
                else
                {
                    // if this column wasn't found, move to the next column
                    localRowList[idx] = 0;
                }
            }
        }
        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }

    // now that we know the set of global rows, complete the coloring

    // Loop over the maximum number of colors
    for (label n = 0; n < maxColors; n++)
    {
        if (n % printInterval == 0)
        {
            Info << "Global ColorSweep: " << n << "   " << mesh_.time().elapsedClockTime() << " s" << endl;
        }

        // Retrieve the local portion of the color vector
        // and set all entries that are currently -1 to the current color
        PetscScalar* colColor;
        VecGetArray(colors, &colColor);

        for (label i = colorStart; i < colorEnd; i++)
        {
            label idx = i - colorStart;
            if (DAUtility::isValueCloseToRef(colColor[idx], -1.0))
            {
                colColor[idx] = n;
            }
        }
        VecRestoreArray(colors, &colColor);

        /* We will do the confilct resolution in two passes. On the first pass
           we will keep the value with the smallest random index on the local
           column set. We will not touch the off processor columns. On the second
           pass we will keep the value with the smallest random index, regardless
           of processor. This is to prevent deadlocks in the conflict resolution.*/
        for (label conPass = 0; conPass < 2; conPass++)
        {

            // Scatter the global colors to each processor
            VecScatterBegin(colorScatter, colors, colorsLocal, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(colorScatter, colors, colorsLocal, INSERT_VALUES, SCATTER_FORWARD);

            // compute the number of local rows
            //int nRows = Iend_-Istart_;

            //Allocate a Scalar array to recieve colColors.
            PetscScalar* colColorLocal;
            VecGetArray(colorsLocal, &colColorLocal);
            VecGetArray(colors, &colColor);

            // set the iteration limits based on conPass
            label start, end;
            if (conPass == 0)
            {
                start = colorStart;
                end = colorEnd;
            }
            else
            {
                start = 0;
                end = nColG;
            }

            //Now loop over the rows and resolve conflicts
            for (label i = Istart; i < Iend; i++)
            {
                label idx = i - Istart;
                if (localRowList[idx] == 1) //this row includes at least 1 global col.
                {
                    /* Get the connectivity row as well as its index into the
                       local indices. */
                    MatGetRow(conMat, i, &nCols, &cols, &vals);
                    MatGetRow(conIndMat, i, &nCols2, NULL, &vals2);

                    // initialize the sorting variables
                    label smallest = nColG;
                    label idxKeep = -1;

                    //int localColIdx;
                    // set any columns with the current color into conflictCols
                    for (label j = 0; j < nCols; j++)
                    {
                        if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
                        {
                            label colIdx = cols[j];
                            label localColIdx = round(vals2[j]);

                            // check if the color in this column is from the
                            // current set
                            if (DAUtility::isValueCloseToRef(colColorLocal[localColIdx], n * 1.0))
                            {
                                /* This matches the current color, so set as a
                                   potential conflict */
                                conflictCols[j] = colIdx;
                                conflictLocalColIdx[j] = localColIdx;
                                /* this is one of the conflicting columns.
                                   If this is a strictly local column, keep it.
                                   Otherwise, compare its random number to the
                                   current smallest one, keep the smaller one and
                                   its index find the index of the smallest
                                   tiebreaker. On the first pass this is only
                                   for the the local columns. On pass two it is
                                   for all columns.*/
                                if (localColumnStat[localColIdx] == 2)
                                {
                                    smallest = -1;
                                    idxKeep = colIdx;
                                }
                                else if (tbkrLocal[localColIdx] < smallest and colIdx >= start and colIdx < end)
                                {
                                    smallest = tbkrLocal[localColIdx];
                                    idxKeep = cols[j];
                                }
                            }
                        }
                    }

                    // Now reset all the conflicting rows
                    for (label j = 0; j < nCols; j++)
                    {
                        label colIdx = conflictCols[j];
                        label localColIdx = conflictLocalColIdx[j];
                        // check that the column is in the range for this conPass.
                        if (colIdx >= start && colIdx < end)
                        {
                            /*this column is local. If this isn't the
                            smallest, reset it.*/
                            if (colIdx != idxKeep)
                            {
                                if (localColIdx >= 0)
                                {
                                    if (localColumnStat[localColIdx] == 2)
                                    {
                                        Pout << "local Array Index: " << colIdx << endl;
                                        Info << "Error, setting a local column!" << endl;
                                        return;
                                    }
                                    PetscScalar valIn = -1;
                                    VecSetValue(colors, colIdx, valIn, INSERT_VALUES);
                                    colColorLocal[localColIdx] = -1;
                                }
                            }
                        }
                    }

                    /* reset any columns that have been changed in conflictCols
                    and conflictLocalColIdx */
                    for (label j = 0; j < nCols; j++)
                    {
                        if (!DAUtility::isValueCloseToRef(vals[j], 0.0))
                        {
                            //reset all values related to this row in conflictCols
                            conflictCols[j] = -1;
                            conflictLocalColIdx[j] = -1;
                        }
                    }

                    // Restore the row information
                    MatRestoreRow(conIndMat, i, &nCols2, NULL, &vals2);
                    MatRestoreRow(conMat, i, &nCols, &cols, &vals);
                }
            }

            VecRestoreArray(colors, &colColor);
            VecRestoreArray(colorsLocal, &colColorLocal);
            VecAssemblyBegin(colors);
            VecAssemblyEnd(colors);
        }

        //check the coloring for completeness
        label colorCounter = 0;
        this->coloringComplete(colors, colorCounter, notColored);
        if (n % printInterval == 0)
        {
            Info << "Number of Uncolored: " << colorCounter << " " << notColored << endl;
        }
        if (notColored == 0)
        {
            Info << "Global ColorSweep: " << n << "   " << mesh_.time().elapsedClockTime() << " s" << endl;
            Info << "Number of Uncolored: " << colorCounter << " " << notColored << endl;
            break;
        }
    }
    VecRestoreArray(globalTiebreaker, &tbkrGlobal);
    VecRestoreArray(globalColumnStat, &globalStat);

    // count the current colors and aggregate
    currColor = 0;
    PetscScalar color;
    for (label i = colorStart; i < colorEnd; i++)
    {
        VecGetValues(colors, 1, &i, &color);
        //Pout<<"Color: "<<i<<" "<<color<<endl;
        if (color > currColor)
        {
            currColor = color;
        }
    }

    reduce(currColor, maxOp<label>());

    nColors = currColor + 1;

    Info << "Ncolors: " << nColors << endl;

    //check the initial coloring for completeness
    //this->coloringComplete(colors, colorCounter, notColored);

    // clean up the unused memory
    VecRestoreArray(localTiebreaker, &tbkrLocal);
    delete[] conflictCols;
    delete[] conflictLocalColIdx;
    delete[] globalIndexList;
    delete[] localColumnStat;
    ISDestroy(&globalIS);
    VecScatterDestroy(&colorScatter);
    VecDestroy(&colorsLocal);
    VecDestroy(&globalTiebreaker);
    VecDestroy(&globalColumnStat);
    VecDestroy(&localTiebreaker);
    delete[] localRowList;
    VecDestroy(&globalVec);
    MatDestroy(&conIndMat);
}

void DAColoring::getMatNonZeros(
    const Mat conMat,
    label& maxCols,
    scalar& allNonZeros) const
{
    /*
    Description:
        Get the max nonzeros per row, and all the nonzeros for this matrix
        This will be used in computing coloring

    Input:
        conMat: the matrix to compute nonzeros

    Output:
        maxCols: max nonzeros per row among all rows
    
        allNonZeros: all non zero elements in conMat
    */

    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    label Istart, Iend;

    // set the counter
    maxCols = 0;
    allNonZeros = 0.0;

    // Determine which rows are on the current processor
    MatGetOwnershipRange(conMat, &Istart, &Iend);

    // loop over the matrix and find the largest number of cols
    for (label i = Istart; i < Iend; i++)
    {
        MatGetRow(conMat, i, &nCols, &cols, &vals);
        if (nCols < 0)
        {
            std::cout << "Warning! procI: " << Pstream::myProcNo() << " nCols <0 at rowI: " << i << std::endl;
            std::cout << "Set nCols to zero " << std::endl;
            nCols = 0;
        }
        if (nCols > maxCols) // perhaps actually check vals?
        {
            maxCols = nCols;
        }
        allNonZeros += nCols;
        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }

    //reduce the maxcols value so that all procs have the same size
    reduce(maxCols, maxOp<label>());

    reduce(allNonZeros, sumOp<scalar>());

    return;
}

label DAColoring::find_index(
    const label target,
    const label start,
    const label size,
    const label* valArray) const
{
    /*
    Description:
        Find the index of a value in an array
    
    Input:
        target: the target value to find

        start: the start index in valArray

        size: the size of valArray array

        valArray: the array to check the target value

    Output:
        k: the index of the value in the array, if the value is 
        not found, return -1
    */

    // loop over the valArray from start until target is found
    for (label k = start; k < size; k++)
    {
        if (valArray[k] == target)
        {
            //Info<<"Start: "<<start<<" "<<k<<endl;
            //this is the k of interest
            return k;
        }
    }
    return -1;
}

void DAColoring::coloringComplete(
    const Vec colors,
    label& colorCounter,
    label& notColored) const
{
    /*
    Description:
        Check if the coloring process is finished and return
        the number of uncolored columns

    Input:
        colors: the current coloring vector

    Output:
        notColored: the number of uncolored columns

    */

    PetscScalar color;
    PetscInt colorStart, colorEnd;

    notColored = 0;
    // get the range of colors owned by the local prock
    VecGetOwnershipRange(colors, &colorStart, &colorEnd);
    //loop over the columns and check if there are any uncolored rows
    const PetscScalar* colColor;
    VecGetArrayRead(colors, &colColor);
    colorCounter = 0;
    for (label i = colorStart; i < colorEnd; i++)
    {
        color = colColor[i - colorStart];
        //VecGetValues(colors,1,&i,&color);
        if (not(color >= 0))
        {
            // this columns not colored and coloring is not complete
            //Pout<<"coloring incomplete...: "<<color<<" "<<i<<endl;
            //VecView(colors,PETSC_VIEWER_STDOUT_WORLD);
            notColored = 1;
            colorCounter++;
            //break;
        }
    }
    VecRestoreArrayRead(colors, &colColor);
    // reduce the logical so that we know that all of the processors are
    // ok
    //Pout<<"local number of uncolored: "<<colorCounter<<" "<<notColored<<endl;
    reduce(notColored, sumOp<label>());
    reduce(colorCounter, sumOp<label>());
}

void DAColoring::validateColoring(
    Mat conMat,
    Vec colors) const
{
    /*
    Description:
        Loop over the rows and verify that no row has two columns with the same color

    Input:
        conMat: connectivity mat for check coloring

        colors: the coloring vector

    Example:
        If the conMat reads, its coloring for each column can be
    
               color0  color1
                 |     |
                 1  0  0  0
        conMat = 0  1  1  0
                 0  0  1  0
                 0  0  0  1
                    |     | 
                color0   color0
    
        Then, if colors = {0, 0, 1, 0}-> no coloring conflict
        if colors = {0, 1, 0, 0}-> coloring conclict
    */

    Info << "Validating Coloring..." << endl;

    PetscInt nCols;
    const PetscInt* cols;
    const PetscScalar* vals;

    label Istart, Iend;

    // scatter colors to local array for all procs
    Vec vout;
    VecScatter ctx;
    VecScatterCreateToAll(colors, &ctx, &vout);
    VecScatterBegin(ctx, colors, vout, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(ctx, colors, vout, INSERT_VALUES, SCATTER_FORWARD);

    PetscScalar* colorsArray;
    VecGetArray(vout, &colorsArray);

    // Determine which rows are on the current processor
    MatGetOwnershipRange(conMat, &Istart, &Iend);

    // first calc the largest nCols in conMat
    label colMax = 0;
    for (label i = Istart; i < Iend; i++)
    {
        MatGetRow(conMat, i, &nCols, &cols, &vals);
        if (nCols > colMax)
        {
            colMax = nCols;
        }
        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }

    // now check if conMat has conflicting rows
    labelList rowColors(colMax);
    for (label i = Istart; i < Iend; i++)
    {
        MatGetRow(conMat, i, &nCols, &cols, &vals);

        // initialize rowColors with -1
        for (label nn = 0; nn < colMax; nn++)
        {
            rowColors[nn] = -1;
        }

        // set rowColors for this row
        for (label j = 0; j < nCols; j++)
        {
            if (DAUtility::isValueCloseToRef(vals[j], 1.0))
            {
                rowColors[j] = round(colorsArray[cols[j]]);
            }
        }

        // check if rowColors has duplicated colors
        for (label nn = 0; nn < nCols; nn++)
        {
            for (label mm = nn + 1; mm < nCols; mm++)
            {
                if (rowColors[nn] != -1 && rowColors[nn] == rowColors[mm])
                {
                    FatalErrorIn("Conflicting Colors Found!")
                        << " row: " << i << " col1: " << cols[nn] << " col2: " << cols[mm]
                        << " color: " << rowColors[nn] << abort(FatalError);
                }
            }
        }

        MatRestoreRow(conMat, i, &nCols, &cols, &vals);
    }

    VecRestoreArray(vout, &colorsArray);
    VecScatterDestroy(&ctx);
    VecDestroy(&vout);

    Info << "No Conflicting Colors Found!" << endl;

    return;
}
} // End namespace Foam

// ************************************************************************* //
