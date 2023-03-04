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
Physical Curve(1) = {1};
//+
Physical Curve(2) = {2};
//+
Physical Curve(3) = {3};
//+
Physical Curve(4) = {4};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
