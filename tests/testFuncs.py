# This file contains two functions to help regression testing. The
# first is used to format float values with a specified absolute and
# relative tolerance. This information is used by the second function
# when it takes in two such formatted strings and decides if they are
# sufficiently close to be considered equal.

import numpy
import os
import sys
from mpi4py import MPI

REG_FILES_MATCH = 0
REG_FILES_DO_NOT_MATCH = 1
REG_ERROR = -1


def run_tests(om, Top, comm, daOptions, funcNames, dvNames, dvIndices, funcDict, derivDict):

    # adjoint-deriv
    prob = om.Problem()
    prob.model = Top()
    prob.setup(mode="rev")
    om.n2(prob, show_browser=False, outfile="mphys_aero.html")
    prob.run_model()
    totals = prob.compute_totals(of=funcNames)

    if comm.rank == 0:
        print(totals)
        for funcName in funcNames:
            funcDict[funcName] = prob.get_val(funcName)
            derivDict[funcName] = {}

    # forwardAD-deriv
    daOptions["useAD"]["mode"] = "forward"
    for i, dvName in enumerate(dvNames):
        tempI = 0
        for index in dvIndices[i]:
            daOptions["useAD"]["dvName"] = dvName
            daOptions["useAD"]["seedIndex"] = index
            prob = om.Problem()
            prob.model = Top()
            prob.setup(mode="rev")
            prob.run_model()

            if comm.rank == 0:
                for funcName in funcNames:
                    derivDict[funcName]["%s%i-Adjoint" % (dvName, index)] = [
                        totals[(funcName, "dvs.%s" % dvName)][0][tempI]
                    ]
                    derivDict[funcName]["%s%i-ForwardAD" % (dvName, index)] = prob.get_val(funcName)[0]

            tempI += 1


def reg_write(values, rel_tol=1e-12, abs_tol=1e-12):
    """
    Write values in special value format
    """
    values = numpy.atleast_1d(values)
    values = values.flatten()
    for val in values:
        s = "@value %26.16f %g %g" % (val, rel_tol, abs_tol)
        print(s)

    return


def reg_par_write(values, rel_tol=1e-12, abs_tol=1e-12):
    """
    Write value(values) from parallel process in sorted order
    """
    values = MPI.COMM_WORLD.gather(values)
    if MPI.COMM_WORLD.rank == 0:
        for i in range(len(values)):
            print("Value(s) on processor: %d" % i)
            reg_write(values[i], rel_tol, abs_tol)


def reg_write_dict(d, rel_tol=1e-12, abs_tol=1e-12):
    """
    Write all values in a dictionary in sorted key order
    """
    for key in sorted(d.keys()):
        print("Dictionary Key: %s" % key)
        if type(d[key]) == dict:
            reg_write_dict(d[key], rel_tol, abs_tol)
        elif type(d[key]) == bool:
            reg_write(int(d[key]), rel_tol, abs_tol)
        else:
            reg_write(d[key], rel_tol, abs_tol)


def _reg_str_comp(str1, str2):
    """
    Compare the float values in str1 and str2 and determine if they
    are equal. Returns True if they are the "same", False if different
    """

    aux1 = str1.split()
    aux2 = str2.split()

    if not aux1[0] == aux2[0] == "@value":
        # This line does not need to be compared
        return True

    # Extract required tolerances and values
    rel_tol = float(aux1[2])
    abs_tol = float(aux1[3])
    val1 = float(aux1[1])
    val2 = float(aux2[1])

    rel_err = 0
    if val2 != 0:
        rel_err = abs((val1 - val2) / val2)
    else:
        rel_err = abs((val1 - val2) / (val2 + 1e-16))

    abs_err = abs(val1 - val2)

    if abs_err < abs_tol or rel_err < rel_tol:
        return True
    else:
        return False


def reg_file_comp(ref_file, comp_file):
    """
    Compare the reference file 'ref_file' with 'comp_file'. The
    order of these two files matter. The ref_file MUST be given
    first. Only values specified by reg_write() are compared.  All
    other lines are ignored. Floating point values are compared based
    on rel_tol and abs_tol
    """

    all_ref_lines = []
    ref_values = []
    comp_values = []
    try:
        f = open(ref_file, "r")
    except IOError:
        print("File %s was not found. Cannot do comparison." % ref_file)
        return REG_ERROR
    for line in f.readlines():
        all_ref_lines.append(line)
        if line[0:6] == "@value":
            ref_values.append(line)

    f.close()

    try:
        f = open(comp_file, "r")
    except IOError:
        print("File %s was not found. Cannot do comparison." % comp_file)
        return REG_ERROR

    for line in f.readlines():
        if line[0:6] == "@value":
            comp_values.append(line)

    f.close()

    # Copy the comp_file to compe_file.orig
    # os.system("cp %s %s.orig" % (comp_file, comp_file))

    # We must check that we have the same number of @value's to compare:
    if len(ref_values) != len(comp_values):
        print("Error: number of @value lines in file not the same!")
        return REG_FILES_DO_NOT_MATCH

    # Open the (new) comp_file:
    f = open(comp_file, "w")

    # Loop over all the ref_lines, for value lines, do the
    # comparison. If comparison is ok, write the ref line, otherwise
    # write orig line.

    j = 0
    res = REG_FILES_MATCH
    for i in range(len(all_ref_lines)):
        line = all_ref_lines[i]
        if line[0:6] == "@value":
            if _reg_str_comp(line, comp_values[j]) is False:
                f.write(comp_values[j])
                res = REG_FILES_DO_NOT_MATCH
            else:
                f.write(line)

            j += 1
        else:
            f.write(line)

    f.close()

    return res


def replace_text_in_file(ref_file, old_text, new_text):
    """
    Replace the text in a file
    """
    if MPI.COMM_WORLD.rank == 0:
        f = open(ref_file, "r")
        fData = f.read()
        f.close()
        fDataNew = fData.replace(old_text, new_text)
        f = open(ref_file, "w")
        f.write(fDataNew)
        f.close()


if __name__ == "__main__":

    testFileRef = sys.argv[1]
    testFile = sys.argv[2]
    res = reg_file_comp(testFileRef, testFile)

    if res == 0:
        exit(0)
    else:
        exit(1)
