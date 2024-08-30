include("MySimplexQuad.jl")
using .MySimplexQuad

include("OrthoPols.jl")
using .OrthoPols

using Printf, LinearAlgebra, Optim, Plots, Random

# BLAS.set_num_threads(1)
BLAS.set_num_threads(Sys.CPU_THREADS)

Random.seed!(hash(floor(Int, time() * 1e9)))

order = 8


specs = [
    (1, 0), # Vertices
    (2, 0), # Edge midpoints
    (4, 1), # Edge class
    (3, 0), # Trig midpoint
    (5, 1), # Interior class, type 1
    (6, 2)  # Interior class, type 2
]

# order = 14

# specs = [
#     (1, 0), # Vertices

#     (4, 1), # Edge class
#     (4, 1), # Edge class
#     (4, 1), # Edge class
#     (3, 0), # Trig midpoint    

#     (5, 1), # Interior class, type 1
#     (5, 1), # Interior class, type 1 
#     (5, 1), # Interior class, type 1
#     # (5, 1), # Interior class, type 1 # no midpoint, additional 5-class, so 2 additional points...

#     (6, 2), # Interior class, type 2
#     (6, 2), # Interior class, type 2
#     (6, 2), # Interior class, type 2
# ]


freeparam = sum(x[2] for x in specs)
indices = 1:(Int((order+1)*(order+2)/2))

# findall(abs.(g(a)).>1e-10)

# indices14 = [1, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 106, 108, 110, 112, 114, 116, 118, 120]
# indices16 = [1, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 137, 139, 141, 143, 145, 147, 149, 151, 153]
# indices18 = [1, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190]
# indices20 = [1, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231]
# indices = indices18

function Trans(point)
    @assert length(point) == 2 "Input point must be a 2-element vector"
    v1 = [-1.0, -1.0 / sqrt(3.0)]
    v2 = [ 1.0, -1.0 / sqrt(3.0)]
    v3 = [ 0.0,  2.0 / sqrt(3.0)]
    x, y = point
    new_x = (v2[1] - v1[1]) * x + (v3[1] - v1[1]) * y + v1[1]
    new_y = (v2[2] - v1[2]) * x + (v3[2] - v1[2]) * y + v1[2]
    return [new_x, new_y]
end

function TransJ()
    v1 = [-1.0, -1.0 / sqrt(3.0)]
    v2 = [ 1.0, -1.0 / sqrt(3.0)]
    v3 = [ 0.0,  2.0 / sqrt(3.0)]
    return [
        v2[1] - v1[1]  v3[1] - v1[1];
        v2[2] - v1[2]  v3[2] - v1[2]
    ]
end

function TransBack(point::Vector{Float64})
    @assert length(point) == 2 "Input point must be a 2-element vector"
    v1 = [-1.0, -1.0 / sqrt(3.0)]
    v2 = [ 1.0, -1.0 / sqrt(3.0)]
    v3 = [ 0.0,  2.0 / sqrt(3.0)]
    x, y = point
    denominator = (v1[1] * (v2[2] - v3[2]) + v2[1] * (v3[2] - v1[2]) + v3[1] * (v1[2] - v2[2]))
    new_x = ((v2[2] - v3[2]) * (x - v1[1]) + (v3[1] - v2[1]) * (y - v1[2])) / denominator
    new_y = ((v3[2] - v1[2]) * (x - v2[1]) + (v1[1] - v3[1]) * (y - v2[2])) / denominator
    return [new_x, new_y]
end

function TransBackJ()
    v1 = [-1.0, -1.0 / sqrt(3.0)]
    v2 = [ 1.0, -1.0 / sqrt(3.0)]
    v3 = [ 0.0,  2.0 / sqrt(3.0)]
    denominator = (v1[1] * (v2[2] - v3[2]) + v2[1] * (v3[2] - v1[2]) + v3[1] * (v1[2] - v2[2]))
    return [
        (v2[2] - v3[2]) / denominator  (v3[1] - v2[1]) / denominator;
        (v3[2] - v1[2]) / denominator  (v1[1] - v3[1]) / denominator
    ]
end

p1 = [0.0, 0.0]
p2 = [1.0, 0.0]
p3 = [0.0, 1.0]

m1 = (p2 .+ p3) ./ 2.0
m2 = (p1 .+ p3) ./ 2.0
m3 = (p1 .+ p2) ./ 2.0

b = (p1 .+ p2 .+ p3) ./ 3.0

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


function rhs()
    w = zeros(div((order + 1) * (order + 2), 2))[indices]
    w[1] = 1/3^(1/4)
    return w
end




function check_points_in_triangle(specs, a)
    # Helper function to check if a point is inside the triangle
    function is_inside_triangle(point)
        x, y = point
        return x >= 0 && y >= 0 && x + y <= 1
    end

    # Generate points based on specs and parameters
    points = []
    a_index = 1

    for (T, param_count) in specs
        if T == 1
            append!(points, [T1[:, i] for i in 1:size(T1, 2)])
        elseif T == 2
            append!(points, [T2[:, i] for i in 1:size(T2, 2)])
        elseif T == 3
            append!(points, [T3[:, i] for i in 1:size(T3, 2)])
        elseif T == 4
            for _ in 1:param_count
                append!(points, [T4(a[a_index])[:, i] for i in 1:size(T4(a[a_index]), 2)])
                a_index += 1
            end
        elseif T == 5
            for _ in 1:param_count
                append!(points, [T5(a[a_index])[:, i] for i in 1:size(T5(a[a_index]), 2)])
                a_index += 1
            end
        elseif T == 6
            append!(points, [T6(a[a_index], a[a_index+1])[:, i] for i in 1:size(T6(a[a_index], a[a_index+1]), 2)])
            a_index += 2
        end
    end

    # Check if all points are inside the triangle
    all_inside = all(is_inside_triangle, points)

    # Check if parameters are within the allowed ranges
    params_in_range = all(0 .<= a[1:end-2] .<= 1) && all(-1 .<= a[end-1:end] .<= 1)

    return all_inside && params_in_range
end

# Function to generate valid parameters
function generate_valid_parameters(specs)
    freeparam = sum(x[2] for x in specs)
    
    while true
        a = rand(freeparam)
        
        # Adjust the range for T6 parameters
        for i in 1:length(specs)
            if specs[i][1] == 6
                start_index = sum(x[2] for x in specs[1:i-1]) + 1
                a[start_index:start_index+1] = 2 .* a[start_index:start_index+1] .- 1
            end
        end
        
        if check_points_in_triangle(specs, a)
            return a
        end
    end
end




A = functions.A
dA = functions.dA

f(a) = norm(A(a)*((A(a)' * A(a))\(A(a)' * rhs()))-rhs())

g(a) = A(a)*((A(a)' * A(a))\(A(a)' * rhs()))-rhs()
w(a) = ((A(a)' * A(a))\(A(a)' * rhs()))

a = generate_valid_parameters(specs)
n = length(a)





function g_new(x::T...) where {T<:Real}
    return g([x...])
end

function w_new(x::T...) where {T<:Real}
    return w([x...])
end



using JuMP
using Ipopt

# Define the model
model = Model(Ipopt.Optimizer)
register(model, :.^, 2, .^; autodiff = true)
register(model, :g_new, n, g_new; autodiff = true)
register(model, :w_new, n, w_new; autodiff = true)


# Assuming x is a vector of decision variables
@variable(model, x[1:n])

# Objective function: 1/2 * ||g(x)||_2^2
@NLobjective(model, Min, 0.5 * sum(g_new(x...).^2))

# Constraint: w(x) > 0
@NLconstraint(model, w_new(x...) >= 1e-12)

# Initial guess for the decision variables
initial_x = a  # Adjust according to your problem
set_start_value.(x, a)

# Solve the optimization problem
optimize!(model)

# Extract the solution
solution = value.(x)
println("Optimal solution: ", solution)