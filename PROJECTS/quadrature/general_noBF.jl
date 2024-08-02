include("MySimplexQuad.jl")
using .MySimplexQuad

include("OrthoPols.jl")
using .OrthoPols

using Printf, LinearAlgebra, Optim, Plots, Random

# BLAS.set_num_threads(1)
BLAS.set_num_threads(Sys.CPU_THREADS)

Random.seed!(hash(floor(Int, time() * 1e9)))

order = 14

# specs = [
#     (1, 0), # Vertices
#     (4, 1), # Edge class
#     (4, 1), # Edge class
#     (5, 1), # Interior class, type 1
#     (5, 1), # Interior class, type 1
#     (5, 1), # Interior class, type 1
#     (6, 2)  # Interior class, type 2
# ]

specs = [
    (1, 0), # Vertices
    (4, 1), # Edge class
    (4, 1), # Edge class
    (4, 1), # Edge class
    (3, 0), # Trig Midpoint
    (5, 1), # Interior class, type 1
    (5, 1), # Interior class, type 1
    (5, 1), # Interior class, type 1
    (6, 2), # Interior class, type 2
    (6, 2), # Interior class, type 2
    (6, 2), # Interior class, type 2
]

# specs = [
#     (1, 0), # Vertices
#     (2, 0), # Edge Midpoints
#     (4, 1), # Edge class
#     (4, 1), # Edge class
#     (5, 1), # Interior class, type 1
#     (5, 1), # Interior class, type 1
#     (5, 1), # Interior class, type 1
#     (6, 2), # Interior class, type 2
#     (6, 2)  # Interior class, type 2
# ]

freeparam = sum(x[2] for x in specs)
indices = 1:(Int((order+1)*(order+2)/2))

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

function rhs_slow()
    X, W = mysimplexquad(order, 2)
    w = zeros(div((order + 1) * (order + 2), 2))[indices]
    
    for k = 1:length(W)
        pols = (ortho2eva3(order, Trans(X[k,:]))[1])[indices]
        w .+= 2*W[k] .* pols
    end
    return w
end

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


function run_parallel(max_attempts = 500, target_f = 1e-10, target_res = 1e-15)
    rhs_val = rhs()  # Precompute rhs
    
    weight(a) = ((A(a)' * A(a))\(A(a)' * rhs_val))
    
    best_result = nothing
    best_f = Inf
    result_lock = ReentrantLock()
    
    attempts = Threads.Atomic{Int}(0)
    solution_found = Threads.Atomic{Bool}(false)
    
    function worker()
        while Threads.atomic_add!(attempts, 1) <= max_attempts && !solution_found[]
            min_val, max_val = 0.01, 0.99
            a = min_val .+ (max_val - min_val) .* rand(freeparam)
            
            try
                current_f = f(a)
                @printf("Thread %d starting with a new config: f(a) is about %.3g. \n", Threads.threadid(), current_f)
                
                for i in 1:10000
                    res = up(a) * g(a)
                    a = a .- res  # Element-wise subtraction
                    
                    current_f = f(a)
                    
                    w = weight(a)
                    
                    # if all(x -> x > 0, w) && all(x -> x > 0, a)
                    if all(x -> x > 0, w)
                        lock(result_lock) do
                            if current_f < best_f
                                best_result = a
                                best_f = current_f
                                println("Thread $(Threads.threadid()): New best result! f(a) = $current_f and res = $(norm(res))")
                            end
                        end
                        
                        if current_f < target_f
                            println("Thread $(Threads.threadid()): Solution found! f(a) = $current_f")
                            Threads.atomic_xchg!(solution_found, true)
                            return
                        end
                        
                        if norm(res) < target_res
                            println("Thread $(Threads.threadid()): Small res found! f(a) = $current_f")
                            Threads.atomic_xchg!(solution_found, true)
                            return
                        end
                    end            
                    
                    if i > 30
                        if any(x -> x > 0, w)
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
                    
                    if i > 500 && current_f > 0.01
                        println("Thread $(Threads.threadid()): .. Zielfunktion not decreasing.")
                        break
                    end
                end
            catch e
                println("Thread $(Threads.threadid()): Error occurred.")
                println("Error: ", e)
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



# a = BigFloat.(["0.08241681507823011945294286373531904405845952423134596282614025628517770576804981",
#                 "0.7336586356282496528158148279730068963729017833255884881520737441032971857121487", 
#                 "0.3968992432090585282173714149068856270044609413319084636886713959115893061823328", 
#                 "0.08483206729954031152189230763403608792291905858396488199050836173441803150625181",
#                 "0.7897555797440240196725215804242898435608124756653441874097675727603200540085861", 
#                 "0.1324011112438714654919235821499105342443585799348871082110686313515405366807619", 
#                 "0.3618924416944691121874987235030598557249083247797107107788101611925815667687246", 
#                 "0.1147700956805489009249444304666597162954323750142438600326002054576767855213305", 
#                 "0.127964059325616321114149468299952478585934750904417616264837824112179310167174"])




function deeper(a)
    for i in 1:1000
        res = up(a) * g(a)
        a = a .- res  # Element-wise subtraction
        println(a, " and ", norm(res), " and ", f(a))
    end
    return a
end



weight(a) = ((A(a)' * A(a))\(A(a)' * rhs()))









function plot_configuration(specs, a)
    # Create a new plot
    p = plot()

    # Define the vertices of the triangle
    p1, p2, p3 = [0,0], [0,1], [1,0]
    x = [p1[1], p2[1], p3[1], p1[1]]
    y = [p1[2], p2[2], p3[2], p1[2]]

    # Plot the triangle
    plot!(p, x, y, seriestype = :shape, fillalpha = 0.3, linecolor = :blue, legend = false, aspect_ratio = :equal)

    a_index = 1
    for (T, param_count) in specs
        if T == 1
            scatter!(p, T1[1, :], T1[2, :], markersize = 3, markercolor = :blue)
        elseif T == 2
            scatter!(p, T2[1, :], T2[2, :], markersize = 3, markercolor = :blue)
        elseif T == 3
            scatter!(p, T3[1, :], T3[2, :], markersize = 3, markercolor = :blue)
        elseif T == 4
            for i in 1:param_count
                scatter!(p, T4(a[a_index])[1, :], T4(a[a_index])[2, :], markersize = 3, markercolor = :blue)
                a_index += 1
            end
        elseif T == 5
            for i in 1:param_count
                scatter!(p, T5(a[a_index])[1, :], T5(a[a_index])[2, :], markersize = 3, markercolor = :blue)
                a_index += 1
            end
        elseif T == 6
            scatter!(p, T6(a[a_index], a[a_index+1])[1, :], T6(a[a_index], a[a_index+1])[2, :], markersize = 3, markercolor = :blue)
            a_index += 2
        end
    end

    # Add plot title and labels
    xlabel!(p, "x")
    ylabel!(p, "y")

    return p  # Return the plot object
end


# p = plot_configuration(specs, a)
# display(p)