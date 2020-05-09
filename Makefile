# make file

default:
	@make opt

clean:
	@echo "********************Cleaning up********************"
# compile src/ofLibs/boundaryConditions
	cd src/ofLibs/boundaryConditions && ./Allclean
# compile src/ofLibs/models
	cd src/ofLibs/models && ./Allclean
# compile src/ofLibs/adjoint with incompressible
	cd src/ofLibs/adjoint && ./Allclean
# compile applications/incompressible
	cd applications/incompressible && ./Allclean
# compile applications/compressible
	cd applications/compressible && ./Allclean
# compile applications/utilities
	cd applications/utilities && ./Allclean
# compile srs/pyLibs
	cd src/pyLibs && ./Allclean

opt:
	@echo "******************Compiling Opt Mode******************"
# compile src/ofLibs/boundaryConditions
	cd src/ofLibs/boundaryConditions && ./Allmake_Opt
# compile src/ofLibs/models
	cd src/ofLibs/models && ./Allmake_Opt
# compile src/ofLibs/adjoint with incompressible
	cd src/ofLibs/adjoint && ./Allmake_Incompressible_Opt
# compile applications/incompressible
	cd applications/incompressible && ./Allmake_Opt
# compile src/ofLibs/adjoint with compressible
	cd src/ofLibs/adjoint && ./Allmake_Compressible_Opt
# compile applications/compressible
	cd applications/compressible && ./Allmake_Opt
# compile applications/utilities
	cd applications/utilities && ./Allmake
# compile srs/pyLibs
	cd src/pyLibs && ./Allmake_Opt
	
debug:
	@echo "******************Compiling Debug Mode******************"
	@exit 1
