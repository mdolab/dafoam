# make file

default:
	@make opt

clean:
	@echo "********************Cleaning up********************"
# clean src/ofLibs/myLibs
	cd src/ofLibs/myLibs && ./Allclean
# clean src/ofLibs with incompressible
	cd src/ofLibs && ./Allclean
# clean applications/incompressible
	cd applications/incompressible && ./Allclean
# clean applications/compressible
	cd applications/compressible && ./Allclean
# clean applications/utilities
	cd applications/utilities && ./Allclean
# clean srs/pyLibs
	cd src/pyLibs && ./Allclean

opt:
	@echo "******************Compiling Opt Mode******************"
# compile src/ofLibs/myLibs
	cd src/ofLibs/myLibs && ./Allmake_Opt
# compile src/ofLibs with incompressible
	cd src/ofLibs && ./Allmake_Incompressible_Opt
# compile applications/incompressible
	cd applications/incompressible && ./Allmake_Opt
# compile src/ofLibs with compressible
	cd src/ofLibs && ./Allmake_Compressible_Opt
# compile applications/compressible
	cd applications/compressible && ./Allmake_Opt
# compile applications/utilities
	cd applications/utilities && ./Allmake
# compile srs/pyLibs
	cd src/pyLibs && ./Allmake_Opt
	
debug:
	@echo "******************Compiling Debug Mode******************"
	@exit 1
