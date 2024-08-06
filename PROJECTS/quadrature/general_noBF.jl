include("MySimplexQuad.jl")
using .MySimplexQuad

include("OrthoPols.jl")
using .OrthoPols

using Printf, LinearAlgebra, Optim, Plots, Random

# BLAS.set_num_threads(1)
BLAS.set_num_threads(Sys.CPU_THREADS)

Random.seed!(hash(floor(Int, time() * 1e9)))

order = 16

specs = [
    # (1, 0), # Vertices (T1)
    # (4, 1), # Edge class (T4)
    # (4, 1), # Edge class (T4)
    # (4, 1), # Edge class (T4)
    # (2, 0), # Edge Midpoints (T2)
    # (3, 0), # Trig midpoint
    # (5, 1), # Interior class, type 1 (T5)
    # (5, 1), # Interior class, type 1 (T5)
    # (5, 1), # Interior class, type 1 (T5)
    # (5, 1), # Interior class, type 1 (T5)
    # (5, 1), # Interior class, type 1 (T5)
    # (5, 1), # Interior class, type 1 (T5)
    # (6, 2), # Interior class, type 2 (T6)
    # (6, 2), # Interior class, type 2 (T6)
    # (6, 2), # Interior class, type 2 (T6)
    # (6, 2), # Interior class, type 2 (T6)
    (6, 2), # Interior class, type 2 (T6)
]

freeparam = sum(x[2] for x in specs)
# indices = 1:(Int((order+1)*(order+2)/2))
indices = [1, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28, 30, 32, 34, 36, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 137, 139, 141, 143, 145, 147, 149, 151, 153]

function Trans(point::Vector{Float64})
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

# T6(a, b) = hcat(
#     (1-a)*(1-b) * p1 + a*(1-b) * p2 + b * m1,
#     (1-a)*(1-b) * p1 + a*(1-b) * p3 + b * m1,
#     (1-a)*(1-b) * p2 + a*(1-b) * p3 + b * m3,
#     (1-a)*(1-b) * p2 + a*(1-b) * p1 + b * m2,
#     (1-a)*(1-b) * p3 + a*(1-b) * p1 + b * m3,
#     (1-a)*(1-b) * p3 + a*(1-b) * p2 + b * m2
# )

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

# function rhs_slow()
#     X, W = mysimplexquad(order, 2)
#     w = zeros(div((order + 1) * (order + 2), 2))[indices]
    
#     for k = 1:length(W)
#         pols = (ortho2eva3(order, Trans(X[k,:]))[1])[indices]
#         w .+= 2*W[k] .* pols
#     end
#     return w
# end

function rhs()
    w = zeros(div((order + 1) * (order + 2), 2))[indices]
    w[1] = 1/3^(1/4)
    return w
end

A = functions.A
dA = functions.dA

f(a) = norm(A(a)*((A(a)' * A(a))\(A(a)' * rhs()))-rhs())
g(a) = A(a)*((A(a)' * A(a))\(A(a)' * rhs()))-rhs()

B(a) = A(a)'*A(a)
C(a) = inv(B(a))

J1(a) = reshape(dA(a)*C(a)*A(a)'*rhs(),:,freeparam)

function m_new(a)
    A_a = A(a)
    dA_a = dA(a)
    
    n_rows_A, n_cols_A = size(A_a)
    n_rows_dA, n_cols_dA = size(dA_a)
    n_blocks = n_rows_dA ÷ n_rows_A

    dA_blocks = [dA_a[i:i+n_rows_A-1, :] for i in 1:n_rows_A:n_rows_dA]
    
    return vcat([dAi' * A_a + A_a' * dAi for dAi in dA_blocks]...)
end

J2(a) = -A(a)*C(a)*reshape(m_new(a)*C(a)*A(a)'*rhs(),:,freeparam)
J(a) = J1(a) + J2(a)

up(a) = inv(J(a)'*J(a))*J(a)'

function run(A, up, g, freeparam)
    weight(a) = ((A(a)' * A(a))\(A(a)' * rhs()))

    for k=1:10
        min_val = 0.01
        max_val = 0.99
        a = min_val .+ (max_val - min_val) .*(rand(freeparam))
        bad = false

        print("starting with a new config: ")
        print("f(a) is about $(@sprintf("%.3g", f(a))). ")

        for i=1:3000
            try
                res = up(a)*g(a)
                a = a-res

                if norm(res)<1e-15
                    print(".. res looking good!.")
                    bad = false
                    break
                end
                
                if any(x -> (x<0), weight(a)) && i>50
                    bad = true
                    print(".. neg lams.\n")
                    break
                elseif norm(res)>2  && i>50
                    print(".. res norm too big, breaking.\n")
                    bad = true
                    break
                elseif f(a)>0.01  && i>200
                    print(".. Zielfunktion not decreasing.\n")
                    display(a)
                    bad = true
                    break
                elseif any(x-> (x>5), a)  && i>50
                    print(".. big weights.\n")
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
            print(".. found something promising... \n")
            weights = ((A(a)' * A(a))\(A(a)' * rhs()))
            display(a)
            display(f(a))

            if all(x -> (x>0), weights) && f(a)<1e-1
                print(" WEIGHTS POSITIVE! \n")
                break
            else
                print("..weights negative, continuing ..")
                print("f(a)=", f(a), "\n")
            end
        end
    end
end


weight(a) = ((A(a)' * A(a))\(A(a)' * rhs()))


function run_parallel(max_attempts = 2000000, target_f = 1e-5, target_res = 1e-12)
    rhs_val = rhs()  # Precompute rhs
    
    weight(a) = ((A(a)' * A(a))\(A(a)' * rhs_val))
    
    best_result = nothing
    best_f = Inf
    result_lock = ReentrantLock()
    
    attempts = Threads.Atomic{Int}(0)
    solution_found = Threads.Atomic{Bool}(false)
    
    function worker()
        while Threads.atomic_add!(attempts, 1) <= max_attempts && !solution_found[]
            # min_val, max_val = -1.99, 1.99
            min_val, max_val = 0.09, 0.99
            a = min_val .+ (max_val - min_val) .* rand(freeparam)
            
            cids = calculate_inverse_distance_sum(specs, a)

            if cids>11000
                break
            end

            try
                current_f = f(a)
                @printf("Thread %d starting with a new config: f(a) is about %.3g and cids is %2d. \n", Threads.threadid(), current_f, cids)
                
                for i in 1:20000
                    res = up(a) * g(a)
                    a = a .- res  # Element-wise subtraction
                    
                    current_f = f(a)
                    
                    w = weight(a)
                    
                    # if all(x -> x > 0, w) && all(x -> x > 0, a)
                    if all(x -> x > 0, w) && i>100
                        lock(result_lock) do
                            if current_f < best_f
                                best_result = a
                                best_f = current_f
                                println("Thread $(Threads.threadid()): New best result! f(a) = $current_f and res = $(norm(res))\n")
                            end
                        end
                        
                        if current_f < target_f
                            println("Thread $(Threads.threadid()): Solution found! f(a) = $current_f")
                            Threads.atomic_xchg!(solution_found, true)
                            return
                        end
                        
                        if norm(res) < target_res
                            println("Thread $(Threads.threadid()): Small res found! f(a) = $current_f\n")
                            Threads.atomic_xchg!(solution_found, true)
                            return
                        end
                    end            
                    
                    if i > 200
                        if any(x -> x < 0, w)
                            println("Thread $(Threads.threadid()): .. weights negative.")
                            break
                        elseif norm(res) > 2
                            println("Thread $(Threads.threadid()): .. res norm too big, breaking.")
                            break
                        elseif any(x -> x > 5, a)
                            println("Thread $(Threads.threadid()): .. big lams.")
                            break
                        end
                    end
                    
                    if i > 1000 && current_f > 0.9
                        println("Thread $(Threads.threadid()): .. Zielfunktion not decreasing.")
                        break
                    end
                end
            catch e
                # println("Thread $(Threads.threadid()): Error occurred.")
                # println("Error: ", e)
            end
        end
    end

    # Start initial batch of workers
    tasks = [Threads.@spawn worker() for _ in 1:Threads.nthreads()]

    # As tasks finish, start new ones until max_attempts is reached or solution is found
    while !isempty(tasks) && attempts[] <= max_attempts && !solution_found[]
        finished = findfirst(istaskdone, tasks)
        if finished !== nothing
            wait(tasks[finished])
            if attempts[] < max_attempts && !solution_found[]
                tasks[finished] = Threads.@spawn worker()
            else
                deleteat!(tasks, finished)
            end
        else
            sleep(0.001)  # Short sleep to prevent busy-waiting
        end
    end

    # Wait for all remaining tasks to finish
    for task in tasks
        wait(task)
    end

    if solution_found[]
        println("Solution found with positive weights with either f(a) < $target_f: or small res, u gotta check!")
        display(best_result)
        println("f(a) = ", best_f)
    else
        println("No solution found meeting the criteria after $(attempts[]) attempts.")
        if best_result !== nothing
            println("Best result found (positive weights and positive a):")
            display(best_result)
            println("f(a) = ", best_f)
        else
            println("No valid results found with positive weights and positive a.")
        end
    end
    
    return best_result, best_f
end




function calculate_inverse_distance_sum(specs, a)
    points = []
    a_index = 1
    
    for (T, param_count) in specs
        if T == 1
            append!(points, [Trans(T1[:, i]) for i in 1:size(T1, 2)])
        elseif T == 2
            append!(points, [Trans(T2[:, i]) for i in 1:size(T2, 2)])
        elseif T == 3
            append!(points, [Trans(T3[:, i]) for i in 1:size(T3, 2)])
        elseif T == 4
            for _ in 1:param_count
                append!(points, [Trans(T4(a[a_index])[:, i]) for i in 1:size(T4(a[a_index]), 2)])
                a_index += 1
            end
        elseif T == 5
            for _ in 1:param_count
                append!(points, [Trans(T5(a[a_index])[:, i]) for i in 1:size(T5(a[a_index]), 2)])
                a_index += 1
            end
        elseif T == 6
            append!(points, [Trans(T6(a[a_index], a[a_index+1])[:, i]) for i in 1:size(T6(a[a_index], a[a_index+1]), 2)])
            a_index += 2
        end
    end
    
    n = length(points)
    sum_inverse_distances = 0.0
    
    for i in 1:n
        for j in (i+1):n
            distance_squared = sum((points[i] - points[j]).^2)
            sum_inverse_distances += 1 / distance_squared
        end
    end
    
    return sum_inverse_distances
end






function deeper(a)
    for i in 1:100
        res = up(a) * g(a)
        a = a .- res  # Element-wise subtraction
        println(a, " and ", norm(res), " and ", f(a))
    end
    return a
end



weight(a) = ((A(a)' * A(a))\(A(a)' * rhs()))









function plot_configuration(specs, a; equi=false)
    # Create a new plot
    p = plot()

    # Define the vertices of the triangle
    p1, p2, p3 = [0,0], [0,1], [1,0]
    
    # Define the vertices of the equilateral triangle
    eq_p1, eq_p2, eq_p3 = [0, 0], [1, 0], [0.5, sqrt(3)/2]
    
    # Function to map points to the equilateral triangle
    function map_to_equilateral(point)
        # Barycentric coordinates
        λ1 = 1 - point[1] - point[2]
        λ2 = point[1]
        λ3 = point[2]
        
        # Map to equilateral triangle
        return λ1 * eq_p1 + λ2 * eq_p2 + λ3 * eq_p3
    end
    
    # Apply mapping if equi is true
    if equi
        p1, p2, p3 = map_to_equilateral(p1), map_to_equilateral(p2), map_to_equilateral(p3)
    end
    
    x = [p1[1], p2[1], p3[1], p1[1]]
    y = [p1[2], p2[2], p3[2], p1[2]]

    # Plot the triangle
    plot!(p, x, y, seriestype = :shape, fillalpha = 0.3, linecolor = :blue, legend = false, aspect_ratio = :equal)

    a_index = 1
    for (T, param_count) in specs
        if T == 1
            points = equi ? map(map_to_equilateral, eachcol(T1)) : eachcol(T1)
            scatter!(p, first.(points), last.(points), markersize = 3, markercolor = :blue)
        elseif T == 2
            points = equi ? map(map_to_equilateral, eachcol(T2)) : eachcol(T2)
            scatter!(p, first.(points), last.(points), markersize = 3, markercolor = :blue)
        elseif T == 3
            points = equi ? map(map_to_equilateral, eachcol(T3)) : eachcol(T3)
            scatter!(p, first.(points), last.(points), markersize = 3, markercolor = :blue)
        elseif T == 4
            for i in 1:param_count
                points = equi ? map(map_to_equilateral, eachcol(T4(a[a_index]))) : eachcol(T4(a[a_index]))
                scatter!(p, first.(points), last.(points), markersize = 3, markercolor = :blue)
                a_index += 1
            end
        elseif T == 5
            for i in 1:param_count
                points = equi ? map(map_to_equilateral, eachcol(T5(a[a_index]))) : eachcol(T5(a[a_index]))
                scatter!(p, first.(points), last.(points), markersize = 3, markercolor = :blue)
                a_index += 1
            end
        elseif T == 6
            points = equi ? map(map_to_equilateral, eachcol(T6(a[a_index], a[a_index+1]))) : eachcol(T6(a[a_index], a[a_index+1]))
            scatter!(p, first.(points), last.(points), markersize = 3, markercolor = :blue)
            a_index += 2
        end
    end

    # Add plot title and labels
    xlabel!(p, "x")
    ylabel!(p, "y")

    return p  # Return the plot object
end

function draw()
    p = plot_configuration(specs, a, equi=true)
    display(p)
    gui()
end





using Statistics: mean, median, std

function seeALL(num_samples = 10000)
    min_val, max_val = 0, 1
    vals = zeros(num_samples)
    best_a = nothing
    best_val = Inf
    
    for i in 1:num_samples
        a = min_val .+ (max_val - min_val) .* rand(freeparam)
        val = calculate_inverse_distance_sum(specs, a)
        vals[i] = val
        
        if val < best_val
            best_val = val
            best_a = copy(a)  # Make a copy to ensure we keep the best configuration
        end
    end
    
    p = plot(vals, title="Distribution of Inverse Distance Sum",
             xlabel="Sample", ylabel="Sum of Inverse Squared Distances",
             legend=false, linewidth=2)
    
    # Add histogram as an inset
    histogram!(twinx(), vals, orientation=:h, ylabel="Frequency",
               alpha=0.3, bins=50, legend=false)
    
    display(p)
    
    # Return statistics and best configuration
    return (
        mean = mean(vals),
        median = median(vals),
        min = minimum(vals),
        max = maximum(vals),
        std = std(vals),
        best_a = best_a,
        best_val = best_val
    )
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