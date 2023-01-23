// gridsize = 0.8;
// cons = 1;
cons = 2.2;

Point(1) = {-3,-1, 0};
Point(2) = {-1,-1, 0};
Point(3) = { 1,-1, 0};
Point(4) = { 3,-1, 0};
Point(5) = { 3, 1, 0};
Point(6) = { 1, 1, 0};
Point(7) = {-1, 1, 0};
Point(8) = {-3, 1, 0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};

Line(4) = {8,7};
Line(5) = {7,6};
Line(6) = {6,5};

Line(7) = {1,8};
Line(8) = {2,7};
Line(9) = {3,6};
Line(10) = {4,5};

Line Loop(1) = {1,8,-4,-7};
Line Loop(2) = {2,9,-5,-8};
Line Loop(3) = {3,10,-6,-9};

Plane Surface(1) = 1;
Plane Surface(2) = 2;
Plane Surface(3) = 3;

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3};

Physical Line(1)= {7};
Physical Line(2)= {4};
Physical Line(3)= {5};
Physical Line(4)= {6};
Physical Line(5)= {10};
Physical Line(6)= {3};
Physical Line(7)= {2};
Physical Line(9)= {1};

Transfinite Line{1,8,-4,-7} = cons/(Mesh.MeshSizeMin*Mesh.MeshSizeMin); // LINKS
Transfinite Surface{1};
Recombine Surface{1};

Transfinite Line{3,10,-6,-9} = cons/(Mesh.MeshSizeMin*Mesh.MeshSizeMin); // RECHTS
Transfinite Surface{3};
Recombine Surface{3};

// Transfinite Line{2,9,-5,-8} = cons/(Mesh.MeshSizeMin*Mesh.MeshSizeMin); // MITTE
// Transfinite Surface{2};
// Recombine Surface{2};

// Physical Line("Line1",25) = {9};

Mesh.Algorithm = 1;


