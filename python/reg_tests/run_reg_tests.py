# =============================================================================
# Standard Python modules                                           
# =============================================================================
import os, sys, argparse, glob

# =============================================================================
# Extension modules
# =============================================================================
import regression_helper as reg

# define scripts to run:
module_name = 'dafoam'

# Get the optional commandline arguments:
parser = argparse.ArgumentParser()
parser.add_argument("-mode",default='test',choices=['test','compare'],
                    help='Test the regression or compare it with FD results')
parser.add_argument("-diff_cmd",default='meld',
                    help='Command to run for displaying diff. Default: meld')
parser.add_argument("-nodiff", default=True,
                    help='Suppress displaying the comparison if not successful')
parser.add_argument("-mpiexec",default='mpirun',
                    help='Command to use for running mpi')
parser.add_argument("-test", default='all',
                    help='Run individual test')
args = parser.parse_args()

mode= args.mode
diff_cmd = args.diff_cmd
nodiff = args.nodiff
mpiexec= args.mpiexec

testFiles= [f for f in os.listdir('tests/') if (os.path.isfile(os.path.join('tests/', f))) and ('test' in f) ]
testFiles.sort()
testNames=[]
for testFile in testFiles:
    testNames.append(os.path.splitext(testFile)[0].split('_')[1])

# remove existing
try:
    os.remove('%s_reg'%(module_name))
except OSError:
    pass

masterRes = 0

# Run each script
for i in range(len(testFiles)):

    testFile=testFiles[i]
    testName=testNames[i]

    if (args.test=='all') or (args.test==testName):
        os.system('%s -np 4 python tests/%s %s > %s_reg_%s 2>&1'%(mpiexec,testFile,mode,module_name,testName) )
    
        regFile    = '%s_reg_%s'%(module_name,testName)
        regRefFile = 'refs/%s_reg_%s.ref'%(module_name,testName)
        # Do the comparison (reference file must be first)
        res = reg.reg_file_comp(regRefFile,regFile)
    
        # Set the proper return codes for the script running this:
        if res == 0: #reg.REG_FILES_MATCH
            print '%s %s: Success!'%(module_name,testName)
        
        elif res == 1: #reg.REG_FILES_DO_NOT_MATCH
            print '%s %s: Failure!'%(module_name,testName)
            if not nodiff:
                os.system('%s refs/%s_reg_%s.ref refs/%s_reg_%s'%(diff_cmd, module_name,testName,module_name,testName))
            masterRes += 1
            
        elif res == -1: #reg.REG_ERROR
            print '%s: Error in regression. Missing files.'%(module_name)
            masterRes += 1
        
        # Concentenate outputs for reference if it failed:
        if res == 1 or res == -1:
            os.system('cat %s >> dafoam_reg.ref'%(regRefFile))
            os.system('cat %s >> dafoam_reg'%(regFile))
            os.system('cat %s.orig >> dafoam_reg.orig'%(regFile))

# Exit with code from reg_file_comp:
sys.exit(masterRes)

