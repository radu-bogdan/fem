
gridsize = 0.4;

Point(1) = {-3,-1, 0};
Point(2) = {-1,-1, 0};
Point(3) = { 1,-1, 0};
Point(4) = { 3,-1, 0};
Point(5) = { 3, 1, 0};
Point(6) = { 1, 1, 0};
Point(7) = {-1, 1, 0};
Point(8) = {-3, 1, 0};

Line(9) = {1,2};
Line(10)= {2,3};
Line(11)= {3,4};

Line(12)= {8,7};
Line(13)= {7,6};
Line(14)= {6,5};

Line(15)= {1,8};
Line(16)= {2,7};
Line(17)= {3,6};
Line(18)= {4,5};

//+
SetFactory("OpenCASCADE");
Circle(25) = {0, 0, 0, 0.3, 0, 2*Pi};
SetFactory("Built-In");

Curve Loop(26) = {25};

Line Loop(19) = {9,16,-12,-15};
Line Loop(20) = {10,17,-13,-16};
Line Loop(21) = {11,18,-14,-17};

Plane Surface(22) = 19;
Plane Surface(23) = {20,26};
Plane Surface(24) = 21;

Physical Surface("domain", 23) = {20};

Transfinite Line{9,16,-12,-15} = 2/(gridsize*gridsize);
Transfinite Surface{22};
Recombine Surface{22};

Transfinite Line{11,18,-14,-17} = 2/(gridsize*gridsize);
Transfinite Surface{24};
Recombine Surface{24};

// Physical Line("Line1",25) = {9};

Mesh.Algorithm = 1;

Mesh.MeshSizeFactor = gridsize;
Mesh.MeshSizeMin = gridsize;
Mesh.MeshSizeMax = gridsize;

