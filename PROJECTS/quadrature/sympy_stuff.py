import sympy as sym
from sympy.integrals.intpoly import polytope_integrate


p1 = sym.Point2D(0, 0)
p2 = sym.Point2D(0, 1)
p3 = sym.Point2D(1, 0) # Reference triangle
# p3 = sym.Point2D(1/2, sym.sqrt(2)/2) # Equilateral unit triangle

m1 = (p2+p3)/2
m2 = (p1+p3)/2
m3 = (p1+p2)/2
bb = (p1+p2+p3)/3


trig = sym.Polygon(p1,p2,p3)
x, y = sym.symbols('x y')

a = sym.symbols('a')
w = sym.symbols('w')


a1 = sym.Point2D(1-2*a, a)
a2 = sym.Point2D(a, 1-2*a)
a3 = sym.Point2D(a, a)

axis1 = x
axis2 = y
axis3 = 1-x-y

saxis1 = x-a     # x = a
saxis2 = y-a     # y = a
saxis3 = 1-x-y-a # 1-x-y = a
saxis4 = 1-x-y

polynom1 = axis1 * axis2 * axis3 * saxis1; polynom1_eval = lambda a : polynom1.subs(dict(zip([x, y], a)))
polynom2 = axis1 * axis2 * axis3 * saxis2; polynom2_eval = lambda a : polynom2.subs(dict(zip([x, y], a)))
polynom3 = axis1 * axis2 * axis3 * saxis3; polynom3_eval = lambda a : polynom3.subs(dict(zip([x, y], a)))
polynom4 = axis1 * axis2 * axis3;          polynom4_eval = lambda a : polynom4.subs(dict(zip([x, y], a)))
polynom5 = x*y*(x-y)*saxis2;          polynom5_eval = lambda a : polynom5.subs(dict(zip([x, y], a)))


result1 = sym.integrate(sym.integrate(polynom1,(y,0,1-x)),  (x, 0, 1)); print(result1)
result2 = sym.integrate(sym.integrate(polynom2,(y,0,1-x)),  (x, 0, 1)); print(result2)
result3 = sym.integrate(sym.integrate(polynom3,(y,0,1-x)),  (x, 0, 1)); print(result3)
result4 = sym.integrate(sym.integrate(polynom4,(y,0,1-x)),  (x, 0, 1)); print(result4)
result5 = sym.integrate(sym.integrate(polynom5,(y,0,1-x)),  (x, 0, 1)); print(result5)

# weight = 2*1/polynom1_eval(a1)*result

print(polynom1_eval(a1),polynom1_eval(a2),polynom1_eval(a3))
print(polynom2_eval(a1),polynom2_eval(a2),polynom2_eval(a3))
print(polynom3_eval(a1),polynom3_eval(a2),polynom3_eval(a3))
print(polynom5_eval(a1),polynom5_eval(a2),polynom5_eval(a3))




### Bedingungen : 
# 
# w*polynom1_eval(a3) = result1
# w*(polynom1_eval(a3) + polynom1_eval(a3) + polynom1_eval(a3)) = result4 !!!!
#

# print(poly_eval)
# print(weight)

val = (7-sym.sqrt(13))/18
wei = (29+17*sym.sqrt(13))/360

pol1evala = polynom1_eval(a1).subs(a,val)
pol2evala = polynom2_eval(a2).subs(a,val)
pol3evala = polynom3_eval(a3).subs(a,val)
pol4evala = (polynom4_eval(a1) + polynom4_eval(a2) + polynom4_eval(a3)).subs(a,val)
pol5evala = (polynom5_eval(a1) + polynom5_eval(a2) + polynom5_eval(a3)).subs(a,val)

result1a = result1.subs(a,val)
result2a = result2.subs(a,val)
result3a = result3.subs(a,val)
result4a = result4.subs(a,val)

print('\n')
# print(polynom1_eval(a3).subs(a,val), " ", float(polynom1_eval(a3).subs(a,val)))
# print(result1.subs(a,val), " ", float(result1.subs(a,val)))

print(float(1/2*wei*pol1evala),float(result1a),float(1/2*wei*pol1evala)/float(result1a))
print(float(1/2*wei*pol2evala),float(result2a),float(1/2*wei*pol2evala)/float(result2a))
print(float(1/2*wei*pol3evala),float(result3a),float(1/2*wei*pol3evala)/float(result3a))
print(float(1/2*wei*pol4evala),float(result4a),float(1/2*wei*pol4evala)/float(result4a))

print(polynom1_eval(a1),' ## ', result1)
print(polynom4_eval(a1),' ## ', result4)

zz = sym.solve([w*(polynom1_eval(a1))-result1,
                w*(polynom2_eval(a2))-result2,
                w*(polynom4_eval(a1) + polynom4_eval(a2) + polynom4_eval(a3))-result4,
                w*(polynom5_eval(a1) + polynom5_eval(a2) + polynom5_eval(a3))-result5], [w,a], dict=True)
print(zz)
