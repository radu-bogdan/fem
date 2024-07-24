# using Makie
include("MySimplexQuad.jl")
using .MySimplexQuad

using Plots
plot()

vertices = [
    0.0 0.0
    1.0 0.0
    0.0 1.0
]

# Need at least 4 vertices for plotting?
vertices1 = [vertices; vertices[1:1, :]]

for npoints in [1]
    X, W = mysimplexquad(BigFloat, npoints, 2)

    # scene = Scene()
    # lines!(scene, vertices1; color = :black, linewidth = 2)
    # scatter!(scene, X[:, 1], X[:, 2]; color = :red, markersize = 0.3 * sqrt.(W))
    # text!(scene, "N=$npoints"; fontsize = 0.15, position = (0.5, 0.9))
    # scale!(scene, 1, 1)

    # Makie.save("figures/gauß-points-$npoints.png", scene; resolution = (300, 300))

    # x = [p1[1], p2[1], p3[1], p1[1]]
    # y = [p1[2], p2[2], p3[2], p1[2]]


    # Plot the triangle
    plot!(vertices1[1,:], vertices1[2,:], seriestype = :shape, fillalpha = 0.3, linecolor = :blue, legend = false, aspect_ratio = :equal)
    scatter!(X[:, 1], X[:, 2], markersize = 8, markercolor = :red, legend = false, aspect_ratio = :equal)

    # Add plot title and labels
    xlabel!("x")
    ylabel!("y")
end