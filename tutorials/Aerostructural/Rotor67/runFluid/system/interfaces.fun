/*--------------------------------*- C++ -*--------------------------------------------------------------*\
| =========                 |                                                                             |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox                                       |
|  \\    /   O peration     | Version:  dev                                                               |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                                                  |
|    \\/     M anipulation  |                                                                             |
|*-------------------------------------------------------------------------------------------------------*|
|  File created by CFD support s.r.o.  on   Tue Nov 13 20:37:43 2018                                      |
|                    http://www.cdfsupport.com                                                            |
|*-------------------------------------------------------------------------------------------------------*|
|  TCFD 18.10 licensed to:                                                                                |
|     drpinghe@umich.edu                                                                                  |
|     drpinghe@umich.edu                                                                                  |
|     TCFD 18.10                                                                                          |
|     Limited                                                                                             |
\*-------------------------------------------------------------------------------------------------------*/
impeller-in
{
    type interfaces;
    functionObjectLibs
    (
         "libTCFDinterfaces.so"
    );
    writeControl timeStep;
    writeInterval 1;
    averagingWindow 1000;
    finalWriteTimes (10000 20000 35000 55000 75000);
    fields ( pTot p phi magU rho T TTot );
    calculateAngles on;
    rhoInf 1.2;
    log debug;
    pRef 101325;
    weightField phi;
    axis (0 0 1);
    omega 1680.02;
    patches
    (
         inlet
    );
    multiplier 22;
    orientation axial;
}

impeller-out
{
    type interfaces;
    functionObjectLibs
    (
         "libTCFDinterfaces.so"
    );
    writeControl timeStep;
    writeInterval 1;
    averagingWindow 1000;
    finalWriteTimes (10000 20000 35000 55000 75000);
    fields ( pTot p phi magU rho T TTot );
    calculateAngles on;
    rhoInf 1.2;
    log debug;
    pRef 101325;
    weightField phi;
    axis (0 0 1);
    omega 1680.02;
    patches
    (
         outlet
    );
    multiplier 22;
    orientation axial;
}

// ************************************************************************* //
