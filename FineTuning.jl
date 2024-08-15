### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ faefa212-5a79-11ef-3c63-2bd1c9ddec53
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Pipe", "Flux", "JLD2"])
	Pkg.add(["Images",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, Flux, JLD2, Images
	using Random, Statistics
	using Pipe: @pipe

	# Uncomment to Add GPU Support
	Pkg.add(["CUDA", "cuDNN"])
	import CUDA, cuDNN
end

# ╔═╡ 0f7d3f8f-4ed2-4a9f-b5ed-8b1b1beea689
const A = Apollo;

# ╔═╡ 7e08d864-d1ab-467e-855d-30876e65490d
const R = Rasters;

# ╔═╡ b622f36b-17fc-4b5b-91ea-93ff9fdf29e1
md"""
# Data Pipeline
"""

# ╔═╡ 852eedbc-17d7-4f20-9b3b-20bc66c0337d
labels = R.Raster("data/RiceSC_South_S2/water.tif");

# ╔═╡ e02f9f7d-741c-416d-9955-39c7e2e8d513
features = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_10m.tif");

# ╔═╡ de0d18eb-ec03-4d91-9dff-f7e704e8991b
μ = Apollo.folddims(mean, features, dims=Rasters.Band);

# ╔═╡ 42f75456-798b-43a9-b21b-cb3c8f12f189
σ = Apollo.folddims(std, features, dims=Rasters.Band);

# ╔═╡ e680082b-a81c-4000-8a3c-78ac7d29e9ad
begin
	# Sample 256x256 Tiles
	xsampler = A.TileView(features, 256, stride=64)
	ysampler = A.TileView(labels, 256, stride=64)
	#weights = A.mapobs(x -> ifelse.(x .== 1, 10.0f0, 1.0f0), ysampler)
	samplers = A.zipobs(xsampler, ysampler)

	# Remove Samples With Less Than 1% Water
	tiles = A.filterobs(samplers) do (x, y)
		return (sum(y) / length(y)) > 0.01
	end

	# Define Data Pipeline
	dtypes = (A.Image(), A.Mask())
	t = A.Tensor() |> A.Normalize(μ, σ)
	pipeline = A.transform(t, dtypes, tiles)

	# Split Into Train and Test
	data = Flux.DataLoader(pipeline, batchsize=8, shuffle=true)
end;

# ╔═╡ e0fdbca0-43df-4215-9238-19cbb74f4f6a
md"""
# Train Model
"""

# ╔═╡ 1655b2ea-8de2-4584-bf22-d51c2b508e63
loss = A.DiceLoss();

# ╔═╡ c9d1dd9b-40bc-494c-aaf6-2da57ba2ed1e
_model = let
	input = A.Single(channels=4)
	encoder = A.ResNet18(weights=:ImageNet)
	model = A.UNet(input=input, encoder=encoder, activation=Flux.sigmoid)
	
	model_state = JLD2.load("models/water_resnet18_imagenet.jld2", "model_state")
	
	Flux.loadmodel!(model, model_state)
end;

# ╔═╡ e5315b1d-2c4b-4202-a187-64ac9a31f263
model = let
	# Initialize Model
	model = deepcopy(_model) |> Flux.gpu
	
	# Initialize Optimizer
	opt = Flux.Optimisers.Adam(5e-4)
	opt_state = Flux.Optimisers.setup(opt, model)
	
	# Freeze Layers
	Flux.freeze!(opt_state.encoder)
	Flux.freeze!(opt_state.decoder)

	# Iterate Over Train Data
	A.train!(loss, model, opt_state, Flux.gpu(data))

	# Return Model
	model
end;

# ╔═╡ 3aeec7e5-b25e-4aff-b6f4-1c53f4553903
eval_metrics = [A.MIoU([0,1]), A.Accuracy(), A.Loss(A.DiceLoss())];

# ╔═╡ 6ac20764-a8fd-409d-8738-67a33a13dd08
initial_results = let
	model = Flux.gpu(_model)
	A.evaluate(data, eval_metrics...) do (x, y)
		ŷ = Flux.gpu(x) |> model |> Flux.cpu
		return (ŷ, y)
	end
end;

# ╔═╡ 6a321c6c-ef15-470d-9523-7f05a2f47199
tuned_results = A.evaluate(data, eval_metrics...) do (x, y)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ 49e420b6-db78-48e6-a333-346ab6cf375f
md"""
# Quantitative Analysis

**Initial MIoU:** $(round(Float64(initial_results.MIoU), digits=4))

**Initial Accuracy:** $(round(Float64(initial_results.accuracy), digits=4))

**Tuned MIoU:** $(round(Float64(tuned_results.MIoU), digits=4))

**Tuned Accuracy:** $(round(Float64(tuned_results.accuracy), digits=4))
"""

# ╔═╡ f272b74f-4276-4105-ab7d-65357ccb1938
md"""
# Qualitative Analysis
"""

# ╔═╡ 854ec47d-7fcd-4e31-8eb0-5b2fb47fb863
function merge_tiles(to, tiles...)
	n = Rasters.Raster(zeros(Float32, size(to)), to)
	dst = Rasters.Raster(zeros(Float32, size(to)), to)
	for tile in tiles
		extended = Rasters.extend(tile, to=to, missingval=0.0f0)
		n .+= ifelse.(extended .== 0.0f0, 0.0f0, 1.0f0)
		dst .+= extended
	end
	return dst ./ clamp!(n, 1.0f0, Inf32)
end;

# ╔═╡ 1f148a0a-94a2-4a0b-bc74-a53a133d165d
preds = map(A.TileView(features, 1024)) do x
		@pipe A.tensor(x) |> 
		A.normalize(_, μ, σ) |>
		Flux.gpu |>
		model |>
		Flux.cpu |>
		UInt8.(_ .> 0.5) |>
		Rasters.Raster(_[:,:,1,1], Rasters.dims(x)[1:2])
end;

# ╔═╡ 5e902774-b7c1-41f4-a2ce-a7c69f076b60
mosaic = merge_tiles(Rasters.dims(features)[1:2], preds...);

# ╔═╡ 5916a273-7682-46ec-9bbe-95bf7dc9317b
lb = Apollo.folddims(x -> quantile(x, 0.02), features, dims=Rasters.Band);

# ╔═╡ 606ab315-7458-4a2a-a861-f20847bdc1f2
ub = Apollo.folddims(x -> quantile(x, 0.98), features, dims=Rasters.Band);

# ╔═╡ f3c0de29-4ac3-4245-9233-b531729eb3b8
let
	rgb = A.rgb(features, lb, ub, bands=[3, 2, 1])
	water_mask = A.binmask(mosaic)
	A.mosaicview((1,2), rgb, water_mask)
end

# ╔═╡ b2e83ec8-71e7-4104-aff7-4f706b8dc1a9
let
	tiles = map(A.sampleobs(pipeline, 3)) do (x, y)
		rgb = A.rgb(A.denormalize(x, μ, σ), lb, ub, bands=[3,2,1])
		infrared = A.rgb(A.denormalize(x, μ, σ), lb, ub, bands=[4,3,2])
		label = A.binmask(y)
		pred = @pipe Flux.gpu(x) |> model |> Flux.cpu |> (_ .> 0.5) |> A.binmask
		return A.mosaicview((1,4), rgb, infrared, label, pred)
	end
	A.mosaicview((3,1), tiles...)
end

# ╔═╡ Cell order:
# ╠═faefa212-5a79-11ef-3c63-2bd1c9ddec53
# ╠═0f7d3f8f-4ed2-4a9f-b5ed-8b1b1beea689
# ╠═7e08d864-d1ab-467e-855d-30876e65490d
# ╟─b622f36b-17fc-4b5b-91ea-93ff9fdf29e1
# ╠═852eedbc-17d7-4f20-9b3b-20bc66c0337d
# ╠═e02f9f7d-741c-416d-9955-39c7e2e8d513
# ╠═de0d18eb-ec03-4d91-9dff-f7e704e8991b
# ╠═42f75456-798b-43a9-b21b-cb3c8f12f189
# ╠═e680082b-a81c-4000-8a3c-78ac7d29e9ad
# ╟─e0fdbca0-43df-4215-9238-19cbb74f4f6a
# ╠═1655b2ea-8de2-4584-bf22-d51c2b508e63
# ╠═c9d1dd9b-40bc-494c-aaf6-2da57ba2ed1e
# ╠═e5315b1d-2c4b-4202-a187-64ac9a31f263
# ╟─49e420b6-db78-48e6-a333-346ab6cf375f
# ╠═3aeec7e5-b25e-4aff-b6f4-1c53f4553903
# ╠═6ac20764-a8fd-409d-8738-67a33a13dd08
# ╠═6a321c6c-ef15-470d-9523-7f05a2f47199
# ╟─f272b74f-4276-4105-ab7d-65357ccb1938
# ╠═854ec47d-7fcd-4e31-8eb0-5b2fb47fb863
# ╠═1f148a0a-94a2-4a0b-bc74-a53a133d165d
# ╠═5e902774-b7c1-41f4-a2ce-a7c69f076b60
# ╠═5916a273-7682-46ec-9bbe-95bf7dc9317b
# ╠═606ab315-7458-4a2a-a861-f20847bdc1f2
# ╠═f3c0de29-4ac3-4245-9233-b531729eb3b8
# ╠═b2e83ec8-71e7-4104-aff7-4f706b8dc1a9
