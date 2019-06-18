.. _Aerodynamics:

Aerodynamics
------------

**NACA0012_Airfoil_Incompressible**

    | Case: Airfoil aerodynamic optimization 
    | Geometry: NACA0012  
    | Objective function: Drag coefficient
    | Design variables: 40 FFD points moving in the y direction, one angle of attack
    | Constraints: Symmetry, volume, thickness, and lift constraints (total number: 81)
    | Mach number: 0.1
    | Reynolds number: 2.3 million
    | Mesh cells: 8.6K
    | Adjoint solver: simpleDAFoam

|

**NACA0012_Airfoil_Compressible**

    | Case: Airfoil aerodynamic optimization
    | Geometry: NACA0012 
    | Objective function: Drag coefficient
    | Design variables: 40 FFD points moving in the y direction, one angle of attack
    | Constraints: Symmetry, volume, thickness, and lift constraints (total number: 81)
    | Mach number: 0.7
    | Reynolds number: 2.3 million
    | Mesh cells: 8.6K
    | Adjoint solver: rhoSimpleCDAFoam

|

**Odyssey_Wing**


    | Case: UAV wing multipoint aerodynamic optimization
    | Geometry: Rectangular wing with the Eppler214 profile 
    | Objective function: Weighted drag coefficient at CL=0.6 and 0.75
    | Design variables: 120 FFD points moving in the y direction, 6 twists, two angle of attack
    | Constraints: Volume, thickness, LE/TE, and lift constraints (total number: 414)
    | Mach number: 0.07
    | Reynolds number: 0.9 million
    | Mesh cells: 25K
    | Adjoint solver: simpleDAFoam

|

**CRM_Wing_Body_Tail**


    | Case: Aircraft aerodynamic optimization
    | Geometry: CRM wing, body, and tail
    | Objective function: Drag coefficient
    | Design variables: 216 FFD points moving in the z direction, 9 wing twists, one tail twist, one angle of attack
    | Constraints: Volume, thickness, LE/TE, and lift constraints (total number: 771)
    | Mach number: 0.85
    | Reynolds number: 5 million
    | Mesh cells: 100K
    | Adjoint solver: rhoSimpleCDAFoam
