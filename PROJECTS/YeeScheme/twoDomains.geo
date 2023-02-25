//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {1, 1, 0, 1.0};
//+
Point(3) = {1, -1, 0, 1.0};
//+
Point(4) = {-1, 1, 0, 1.0};
//+
Line(1) = {4, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 1};
//+
Line(4) = {1, 4};
//+
SetFactory("OpenCASCADE");
Circle(5) = {-0, 0, 0, 0.3, 0, 2*Pi};
SetFactory("Built-in");
//+
Physical Curve(1) = {1};
//+
Physical Curve(2) = {2};
//+
Physical Curve(3) = {3};
//+
Physical Curve(4) = {4};
//+
Physical Curve(5) = {5};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Curve Loop(2) = {5};
//+
Plane Surface(1) = {1, 2};
//+
Plane Surface(2) = {2};
