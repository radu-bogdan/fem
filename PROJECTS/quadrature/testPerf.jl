using LinearAlgebra
using BenchmarkTools

function blas_test(n)
    A = rand(n, n)
    B = rand(n, n)
    C = zeros(n, n)
    
    BLAS.set_num_threads(1)
    t1 = @belapsed BLAS.gemm!('N', 'N', 1.0, $A, $B, 0.0, $C)
    
    BLAS.set_num_threads(Sys.CPU_THREADS)
    t2 = @belapsed BLAS.gemm!('N', 'N', 1.0, $A, $B, 0.0, $C)
    
    speedup = t1 / t2
    println("Single-threaded time: $(round(t1, digits=6)) seconds")
    println("Multi-threaded time:  $(round(t2, digits=6)) seconds")
    println("Speedup: $(round(speedup, digits=2))x")
    println("Efficiency: $(round(100 * speedup / Sys.CPU_THREADS, digits=1))%")
end

# Run the test with a 5000x5000 matrix
blas_test(5000)