include("MySimplexQuad.jl")
using .MySimplexQuad

include("OrthoPols.jl")
using .OrthoPols

using Printf

# BLAS.set_num_threads(1)

using LinearAlgebra
BLAS.set_num_threads(Sys.CPU_THREADS)

use_bigfloat = true

##############################################################################################################################

function rse(vec::Vector{T}) where T
    threshold = if T == BigFloat
        BigFloat("1e-60")
    else
        T(1e-14)
    end
    
    map(x -> abs(x) < threshold ? zero(T) : x, vec)
end

##############################################################################################################################


function Trans(point::Vector{T}) where T
    @assert length(point) == 2 "Input point must be a 2-element vector"

    # Define the vertices of the target triangle
    v1 = T[-1, -1 / sqrt(T(3))]   # (0,0) maps to this
    v2 = T[ 1, -1 / sqrt(T(3))]   # (1,0) maps to this
    v3 = T[ 0,  2 / sqrt(T(3))]   # (0,1) maps to this
    
    # Extract x and y from the input point
    x, y = point

    # Compute the affine transformation
    new_x = (v2[1] - v1[1]) * x + (v3[1] - v1[1]) * y + v1[1]
    new_y = (v2[2] - v1[2]) * x + (v3[2] - v1[2]) * y + v1[2]

    return T[new_x, new_y]
end

function TransJ()
    # Determine the type based on the global variable
    T = use_bigfloat ? BigFloat : Float64

    # Define the vertices of the target triangle
    v1 = T[-1, -1 / sqrt(T(3))]
    v2 = T[ 1, -1 / sqrt(T(3))]
    v3 = T[ 0,  2 / sqrt(T(3))]
    
    # Compute the Jacobian matrix
    J = T[
        v2[1] - v1[1]  v3[1] - v1[1];
        v2[2] - v1[2]  v3[2] - v1[2]
    ]
    
    return J
end

function TransBack(point::Vector{T}) where T
    @assert length(point) == 2 "Input point must be a 2-element vector"

    # Define the vertices of the target triangle
    v1 = T[-1, -1 / sqrt(T(3))]   # Maps to (0,0)
    v2 = T[ 1, -1 / sqrt(T(3))]   # Maps to (1,0)
    v3 = T[ 0,  2 / sqrt(T(3))]   # Maps to (0,1)
    
    # Extract x and y from the input point
    x, y = point

    # Compute the inverse affine transformation
    denominator = (v1[1] * (v2[2] - v3[2]) + v2[1] * (v3[2] - v1[2]) + v3[1] * (v1[2] - v2[2]))
    
    new_x = ((v2[2] - v3[2]) * (x - v1[1]) + (v3[1] - v2[1]) * (y - v1[2])) / denominator
    new_y = ((v3[2] - v1[2]) * (x - v2[1]) + (v1[1] - v3[1]) * (y - v2[2])) / denominator

    return T[new_x, new_y]
end

function TransBackJ()
    # Determine the type based on the global variable
    T = use_bigfloat ? BigFloat : Float64

    # Define the vertices of the target triangle
    v1 = T[-1, -1 / sqrt(T(3))]   # Maps to (0,0)
    v2 = T[ 1, -1 / sqrt(T(3))]   # Maps to (1,0)
    v3 = T[ 0,  2 / sqrt(T(3))]   # Maps to (0,1)
    
    # Compute the denominator
    denominator = (v1[1] * (v2[2] - v3[2]) + v2[1] * (v3[2] - v1[2]) + v3[1] * (v1[2] - v2[2]))
    
    # Compute the Jacobian matrix
    J = T[(v2[2] - v3[2]) / denominator  (v3[1] - v2[1]) / denominator;
          (v3[2] - v1[2]) / denominator  (v1[1] - v3[1]) / denominator]
    
    return J
end

##############################################################################################################################

using LinearAlgebra
using Optim
using Plots
using Optim
# using LineSearches

BLAS.set_num_threads(Sys.CPU_THREADS)


using Random
current_time = time()
time_ns = floor(Int, current_time * 1e9)
seed = hash(time_ns)
Random.seed!(seed)

# order = 8

# specs = [
#     (1, 0), # Vertices
#     (2, 0), # Edge midpoints
#     (4, 1), # Edge class
#     (3, 0), # Trig midpoint
#     (5, 1), # Interior class, type 1
#     (6, 2)  # Interior class, type 2
# ]

order = 16

specs = [
    (1, 0), # Vertices

    (4, 1), # Edge class
    (4, 1), # Edge class
    (4, 1), # Edge class    
    (2, 0), # Edge Midpoints (T2)

    (5, 1), # Interior class, type 1
    (5, 1), # Interior class, type 1 
    (5, 1), # Interior class, type 1
    (5, 1), # Interior class, type 1
    (5, 1), # Interior class, type 1

    (6, 2), # Interior class, type 2
    (6, 2), # Interior class, type 2
    (6, 2), # Interior class, type 2
    (6, 2), # Interior class, type 2
]

freeparam = sum(x[2] for x in specs)
# indices = 1:(Int((order+1)*(order+2)/2))
indices = [1, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 137, 139, 141, 143, 145, 147, 149, 151, 153]
##############################################################################################################################

function (vec::Vector{T}) where T <: AbstractFloat
    threshold = if T == BigFloat
        BigFloat("1e-60")
    else
        T(1e-14)
    end
    
    map(x -> abs(x) < threshold ? zero(T) : x, vec)
end

##############################################################################################################################
p1 = use_bigfloat ? BigFloat[0.0, 0.0] : [0.0, 0.0]
p2 = use_bigfloat ? BigFloat[1.0, 0.0] : [1.0, 0.0]
p3 = use_bigfloat ? BigFloat[0.0, 1.0] : [0.0, 1.0]

m1 = (p2 .+ p3) ./ (use_bigfloat ? BigFloat(2.0) : 2.0)
m2 = (p1 .+ p3) ./ (use_bigfloat ? BigFloat(2.0) : 2.0)
m3 = (p1 .+ p2) ./ (use_bigfloat ? BigFloat(2.0) : 2.0)

b = (p1 .+ p2 .+ p3) ./ (use_bigfloat ? BigFloat(3.0) : 3.0)
##############################################################################################################################


# vertices
T1 = hcat(p1, p2, p3)
eval_T1(order) = (ortho2eva3(order, Trans(T1[:,1]))[1])+
                 (ortho2eva3(order, Trans(T1[:,2]))[1])+
                 (ortho2eva3(order, Trans(T1[:,3]))[1])

# edge midpoints
T2 = hcat(m1, m2, m3)
eval_T2(order) = (ortho2eva3(order, Trans(T2[:,1]))[1])+
                 (ortho2eva3(order, Trans(T2[:,2]))[1])+
                 (ortho2eva3(order, Trans(T2[:,3]))[1])

# midpoint
T3 = hcat(b)
eval_T3(order) = (ortho2eva3(order, Trans(T3[:,1]))[1])


# edge class
T4(a) = hcat(a * p1 .+ (1 .- a) * p2,
             a * p2 .+ (1 .- a) * p1,
             a * p3 .+ (1 .- a) * p1,
             a * p1 .+ (1 .- a) * p3,
             a * p3 .+ (1 .- a) * p2,
             a * p2 .+ (1 .- a) * p3)

dT4 = hcat(p1.-p2,
           p2.-p1,
           p3.-p1,
           p1.-p3,
           p3.-p2,
           p2.-p3)

eval_T4(order,a) = (ortho2eva3(order, Trans(T4(a)[:,1]))[1])+
                   (ortho2eva3(order, Trans(T4(a)[:,2]))[1])+
                   (ortho2eva3(order, Trans(T4(a)[:,3]))[1])+
                   (ortho2eva3(order, Trans(T4(a)[:,4]))[1])+
                   (ortho2eva3(order, Trans(T4(a)[:,5]))[1])+
                   (ortho2eva3(order, Trans(T4(a)[:,6]))[1])

eval_dT4(order,a) = ([ortho2eva3(order, Trans(T4(a)[:,1]))[2], ortho2eva3(order, Trans(T4(a)[:,1]))[3]]'*TransJ()*(dT4[:,1])+
                     [ortho2eva3(order, Trans(T4(a)[:,1]))[2], ortho2eva3(order, Trans(T4(a)[:,1]))[3]]'*TransJ()*(dT4[:,1])+
                     [ortho2eva3(order, Trans(T4(a)[:,1]))[2], ortho2eva3(order, Trans(T4(a)[:,1]))[3]]'*TransJ()*(dT4[:,1])+
                     [ortho2eva3(order, Trans(T4(a)[:,1]))[2], ortho2eva3(order, Trans(T4(a)[:,1]))[3]]'*TransJ()*(dT4[:,1])+
                     [ortho2eva3(order, Trans(T4(a)[:,1]))[2], ortho2eva3(order, Trans(T4(a)[:,1]))[3]]'*TransJ()*(dT4[:,1])+
                     [ortho2eva3(order, Trans(T4(a)[:,1]))[2], ortho2eva3(order, Trans(T4(a)[:,1]))[3]]'*TransJ()*(dT4[:,1]))'

# inner class, type 1
T5(a) = hcat(a * m1 .+ (1 .- a) * p1,
             a * m2 .+ (1 .- a) * p2,
             a * m3 .+ (1 .- a) * p3)

dT5 = hcat(m1-p1,
           m2-p2,
           m3-p3)

eval_T5(order,a) = ortho2eva3(order, Trans(T5(a)[:,1]))[1]+
                   ortho2eva3(order, Trans(T5(a)[:,2]))[1]+
                   ortho2eva3(order, Trans(T5(a)[:,3]))[1]

eval_dT5(order,a) = ([ortho2eva3(order, Trans(T5(a)[:,1]))[2], ortho2eva3(order, Trans(T5(a)[:,1]))[3]]'*TransJ()*(dT5[:,1])+
                     [ortho2eva3(order, Trans(T5(a)[:,2]))[2], ortho2eva3(order, Trans(T5(a)[:,2]))[3]]'*TransJ()*(dT5[:,2])+
                     [ortho2eva3(order, Trans(T5(a)[:,3]))[2], ortho2eva3(order, Trans(T5(a)[:,3]))[3]]'*TransJ()*(dT5[:,3]))'


# inner class, type 2
T6(a, b) =  hcat(b * (a * m1 .+ (1 - a) .* p1) .+ (1 - b) * (a * m2 .+ (1 - a) .* p2),
                 b * (a * m1 .+ (1 - a) .* p1) .+ (1 - b) * (a * m3 .+ (1 - a) .* p3),
                 b * (a * m3 .+ (1 - a) .* p3) .+ (1 - b) * (a * m2 .+ (1 - a) .* p2),
                 b * (a * m2 .+ (1 - a) .* p2) .+ (1 - b) * (a * m1 .+ (1 - a) .* p1),
                 b * (a * m3 .+ (1 - a) .* p3) .+ (1 - b) * (a * m1 .+ (1 - a) .* p1),
                 b * (a * m2 .+ (1 - a) .* p2) .+ (1 - b) * (a * m3 .+ (1 - a) .* p3))

daT6(b) = hcat(b * (m1 - p1) + (1 - b) * (m2 - p2),
               b * (m1 - p1) + (1 - b) * (m3 - p3),
               b * (m3 - p3) + (1 - b) * (m2 - p2),
               b * (m2 - p2) + (1 - b) * (m1 - p1),
               b * (m3 - p3) + (1 - b) * (m1 - p1),
               b * (m2 - p2) + (1 - b) * (m3 - p3))

dbT6(a) = hcat((a * m1 + (1 - a) * p1) - (a * m2 + (1 - a) * p2),
               (a * m1 + (1 - a) * p1) - (a * m3 + (1 - a) * p3),
               (a * m3 + (1 - a) * p3) - (a * m2 + (1 - a) * p2),
               (a * m2 + (1 - a) * p2) - (a * m1 + (1 - a) * p1),
               (a * m3 + (1 - a) * p3) - (a * m1 + (1 - a) * p1),
               (a * m2 + (1 - a) * p2) - (a * m3 + (1 - a) * p3))

eval_T6(order,a,b) = ortho2eva3(order, Trans(T6(a, b)[:,1]))[1]+
                     ortho2eva3(order, Trans(T6(a, b)[:,2]))[1]+
                     ortho2eva3(order, Trans(T6(a, b)[:,3]))[1]+
                     ortho2eva3(order, Trans(T6(a, b)[:,4]))[1]+
                     ortho2eva3(order, Trans(T6(a, b)[:,5]))[1]+
                     ortho2eva3(order, Trans(T6(a, b)[:,6]))[1]

eval_daT6(order,a,b) = ([ortho2eva3(order, Trans(T6(a, b)[:,1]))[2], ortho2eva3(order, Trans(T6(a, b)[:,1]))[3]]'*TransJ()*(daT6(b)[:,1])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,2]))[2], ortho2eva3(order, Trans(T6(a, b)[:,2]))[3]]'*TransJ()*(daT6(b)[:,2])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,3]))[2], ortho2eva3(order, Trans(T6(a, b)[:,3]))[3]]'*TransJ()*(daT6(b)[:,3])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,4]))[2], ortho2eva3(order, Trans(T6(a, b)[:,4]))[3]]'*TransJ()*(daT6(b)[:,4])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,5]))[2], ortho2eva3(order, Trans(T6(a, b)[:,5]))[3]]'*TransJ()*(daT6(b)[:,5])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,6]))[2], ortho2eva3(order, Trans(T6(a, b)[:,6]))[3]]'*TransJ()*(daT6(b)[:,6]))'

eval_dbT6(order,a,b) = ([ortho2eva3(order, Trans(T6(a, b)[:,1]))[2], ortho2eva3(order, Trans(T6(a, b)[:,1]))[3]]'*TransJ()*(dbT6(a)[:,1])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,2]))[2], ortho2eva3(order, Trans(T6(a, b)[:,2]))[3]]'*TransJ()*(dbT6(a)[:,2])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,3]))[2], ortho2eva3(order, Trans(T6(a, b)[:,3]))[3]]'*TransJ()*(dbT6(a)[:,3])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,4]))[2], ortho2eva3(order, Trans(T6(a, b)[:,4]))[3]]'*TransJ()*(dbT6(a)[:,4])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,5]))[2], ortho2eva3(order, Trans(T6(a, b)[:,5]))[3]]'*TransJ()*(dbT6(a)[:,5])+
                        [ortho2eva3(order, Trans(T6(a, b)[:,6]))[2], ortho2eva3(order, Trans(T6(a, b)[:,6]))[3]]'*TransJ()*(dbT6(a)[:,6]))'

##############################################################################################################################

function generate_A(specs)
    expr = quote
        function A(a)
            hcat(
                $(generate_exprs(specs)...)
            )
        end
    end
    return eval(expr)
end

function generate_exprs(specs)
    a_index = 1
    exprs = []
    for (T, param_count) in specs
        if param_count == 0
            push!(exprs, :(eval_T$T(order)[indices]))
        else
            params = [:(a[$i]) for i in a_index:(a_index+param_count-1)]
            push!(exprs, :(eval_T$T(order, $(params...))[indices]))
            a_index += param_count
        end
    end
    return exprs
end

##############################################################################################################################

function generate_A_and_dA(specs)
    A_expr = generate_A_expr(specs)
    dA_exprs = generate_dA_exprs(specs)
    combined_dA_expr = generate_combined_dA_expr(dA_exprs)
    
    return eval(quote
        $A_expr
        $(dA_exprs...)
        $combined_dA_expr
        (A = A, dA = dA)
    end)
end

function generate_A_expr(specs)
    quote
        function A(a)
            hcat($(generate_columns(specs, :eval_T)...))
        end
    end
end

function generate_dA_exprs(specs)
    param_count = sum(spec[2] for spec in specs)
    [generate_dA_expr(specs, i) for i in 1:param_count]
end

function generate_dA_expr(specs, param_index)
    columns = generate_columns(specs, :eval_T, param_index)
    quote
        function $(Symbol("dA$param_index"))(a)
            hcat($(columns...))
        end
    end
end

function generate_combined_dA_expr(dA_exprs)
    quote
        function dA(a)
            vcat($([:($(Symbol("dA$i"))(a)) for i in 1:length(dA_exprs)]...))
        end
    end
end

function generate_columns(specs, base_func, active_param=nothing)
    a_index = 1
    columns = []
    for (T, param_count) in specs
        if param_count == 0
            push!(columns, :($(Symbol("$(base_func)$T"))(order)[indices]))
        else
            params = [:(a[$i]) for i in a_index:(a_index+param_count-1)]
            if active_param !== nothing && a_index <= active_param < a_index + param_count
                if T == 6
                    func = active_param == a_index ? :eval_daT6 : :eval_dbT6
                else
                    func = Symbol("eval_dT$T")
                end
                push!(columns, :($func(order, $(params...))[indices]))
            else
                expr = :($(Symbol("$(base_func)$T"))(order, $(params...))[indices])
                expr = active_param === nothing ? expr : :(0 * $expr)
                push!(columns, expr)
            end
            a_index += param_count
        end
    end
    return columns
end

functions = generate_A_and_dA(specs)

##############################################################################################################################


function rhs()
    T = use_bigfloat ? BigFloat : Float64
    X, W = mysimplexquad(T, order, 2)
    w = zeros(T, div((order + 1) * (order + 2), 2))[indices]  # Initialize w with the correct type and size
    
    for k = 1:length(W)
        pols = (ortho2eva3(order, Trans(T.(X[k,:])))[1])[indices]
        w .+= 2*W[k] .* pols
    end
    return w
end

funs = generate_A_and_dA(specs)

A = funs.A
dA = funs.dA

f(a) = norm(A(a)*((A(a)' * A(a))\(A(a)' * rhs()))-rhs())
g(a) = A(a)*((A(a)' * A(a))\(A(a)' * rhs()))-rhs()

##############################################################################################################################


B(a) = A(a)'*A(a)
C(a) = inv(B(a))

J1(a) = reshape(dA(a)*C(a)*A(a)'*rhs(),:,freeparam)

function m_new(a)
    A_a = A(a)
    dA_a = dA(a)
    
    # Determine the number of blocks dynamically
    n_rows_A, n_cols_A = size(A_a)
    n_rows_dA, n_cols_dA = size(dA_a)
    n_blocks = n_rows_dA ÷ n_rows_A

    # Split dA_a into blocks
    dA_blocks = [dA_a[i:i+n_rows_A-1, :] for i in 1:n_rows_A:n_rows_dA]
    
    return vcat([dAi' * A_a + A_a' * dAi for dAi in dA_blocks]...)
end

J2(a) = -A(a)*C(a)*reshape(m_new(a)*C(a)*A(a)'*rhs(),:,freeparam)
J(a) = J1(a) + J2(a)

up(a) = inv(J(a)'*J(a))*J(a)'

##############################################################################################################################

weight(a) = ((A(a)' * A(a))\(A(a)' * rhs()))

function run(A,up,g,freeparam)
    weight(a) = ((A(a)' * A(a))\(A(a)' * rhs()))

    for k=1:10
        min_val = 0.01
        max_val = 0.99
        a = min_val .+ (max_val - min_val) .*(rand(freeparam))
        bad = false

        print("starting with a new config: ")
        print("f(a) is about $(@sprintf("%.3g", f(a))). ")

        for i=1:1000
            try
                res = up(a)*g(a)
                a = a-res

                println("$(@sprintf("%.3g", f(a))). ", norm(res))
                if norm(res)<1e-15
                    print(".. res looking good!.")
                    bad = false
                    break
                end

                # println(a)
                
                if any(x -> (x<0), weight(a)) && i>50
                    bad = true
                    # print(".. neg lams.\n")
                    break
                elseif norm(res)>2  && i>10
                    # print(".. res norm too big, breaking.\n")
                    bad = true
                    break
                elseif f(a)>0.01  && i>10
                    # print(".. Zielfunktion not decreasing.\n")
                    display(a)
                    bad = true
                    break
                elseif any(x-> (x>2), a)  && i>10
                    # print(".. big weights.\n")
                    bad = true
                    break
                end
            catch e
                bad = true
                print(".. nans... breaking\n")
                break
            end
        end

        if !bad
            print(".. found something promising...")
            weights = ((A(a)' * A(a))\(A(a)' * rhs()))
            display(a)

            if all(x -> (x>0), weights)
                print(" WEIGHTS POSITIVE! \n")
                break
            else
                print("..weights negative, continuing ..")
                print("f(a)=", f(a), "\n")
                # if abs(f(a))<1e-10
                #     break
                # end
            end
        end
    end
end
##############################################################################################################################

# a = BigFloat.(["0.38795567023711125",
# "0.19919036675837604",
# "0.04228598931612859",
# "0.30329531785740743",
# "0.9447829689502107",
# "0.08023462740680715",
# "1.3419989636655103",
# "0.6111585613042461",
# "0.08142499585711943",
# "0.09489106159252968",
# "0.3396898680218051",
# "0.6965597558127528"])

a = BigFloat.([0.8425091314698989197751188839902170002460479736328125
0.694917680152862704545668748323805630207061767578125
0.060145512851151154698481349214489455334842205047607421875
0.76086928139311160723678995054797269403934478759765625
0.48047935053790935544526519151986576616764068603515625
0.051281620477828672444164936905508511699736118316650390625
0.269284989653191531022713434140314348042011260986328125
0.88285106285853043317501942510716617107391357421875
0.08139286964794946455103996640900732018053531646728515625
0.2112496012057298588882048306913929991424083709716796875
0.2466319822858454580138953815549029968678951263427734375
0.7541945351195431523905199355795048177242279052734375
0.07262588913486554831333563697626232169568538665771484375
0.602075485976834468004881273373030126094818115234375
0.2079834518061185189008455154180410318076610565185546875
-0.0939200446763669705152466349318274296820163726806640625])

function deeper(a)
    upa = up(a)
    for i in 1:300
        res = upa * g(a)
        a = a .- res  # Element-wise subtraction
        # println(a, " and ", norm(res), " and ", f(a))
        println(a)
        println(norm(res))
        # println(f(a),"\n\n")
    end
    return a
end