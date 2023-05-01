// Gmsh project created on Fri Apr 21 16:21:20 2023
SetFactory("OpenCASCADE");



//+
Rectangle(1) = {-1, -1, 0, 2, 2, 0};
//+
Circle(5) = {-0.25, 0, 0, 0.1, 0, 2*Pi};
//+
Circle(6) = {0.25, 0.0, 0, 0.1, 0, 2*Pi};
//+

//+
Curve Loop(2) = {5};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {6};
//+
Plane Surface(3) = {3};

Coherence;

// 2d 2 - i1
// 2d 3 - i2
// 2d 4 - air

// 1d 1, 2, 3, 4 - 0 DC boundary

// 1d 6 - displacement boundary



//+
Transfinite Curve {5} = 50 Using Progression 1;
//+
//Transfinite Curve {6} = 200 Using Bump 1;


//+
Field[1] = Ball;
//+q
Field[1].Radius = 0.11;
//+
Field[1].Thickness = 0.0;
//+
Field[1].XCenter = 0.25;
//+
Field[1].VOut = 0.05;
//+
Field[1].VIn = 0.01;
//+

Background Field = 1;
//+
//+

