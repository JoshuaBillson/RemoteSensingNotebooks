### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 40c186f4-5b3d-11ef-2a2e-99823024c7fe
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Flux", "JLD2", "Images"])
	Pkg.add(["Pipe",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, Flux, JLD2, Images
	using Random, Statistics
	using Pipe: @pipe

	# Uncomment to Add GPU Support
	Pkg.add(["CUDA", "cuDNN"])
	import CUDA, cuDNN
end

# ╔═╡ bbbd8c59-027b-4770-8334-f5cb814fbebf
const A = Apollo;

# ╔═╡ d23808f0-f9b3-40c8-89f7-7a8b11a49f79
const R = Rasters;

# ╔═╡ c033b6d5-9811-4ca3-993a-9a7b9b537075
md"""
# Define Data Pipeline
"""

# ╔═╡ 8bf5bd76-f283-4b82-a78f-a5d9d9bbe221
features = @pipe readdir("data/RiceSC_South_S1/", join=true) |> 
filter(x -> contains(x, r"\d\d\d\d-\d\d-\d\d"), _) |>
map(x -> R.Raster(x, name=split(basename(x), ".")[1], lazy=false), _) |>
R.RasterStack(_...);

# ╔═╡ 65fab3fc-ad45-41ce-a363-defacabc3981
labels = R.Raster("data/RiceSC_South_S1/labels.tif");

# ╔═╡ ebf8c65d-c349-4bef-ae2e-5524f0c8f773
μ = mean(A.tensor(Float64, features, layerdim=R.Ti), dims=(1,2,4,5)) |> vec;

# ╔═╡ 957ca4c5-c962-4b71-8a36-c05f27d257b0
σ = std(A.tensor(Float64, features, layerdim=R.Ti), dims=(1,2,4,5)) |> vec;

# ╔═╡ 059904ca-21ac-47ba-bc60-eeeabfaa7382
begin
	xsampler = A.TileView(features, 256)
	ysampler = A.TileView(labels, 256)
	tiles = A.zipobs(xsampler, ysampler)

	t = A.Tensor(layerdim=R.Ti) |> A.Normalize(μ, σ)
	pipeline = A.transform(t, (A.Image(), A.Mask()), tiles)

	train, test = A.splitobs(pipeline, at=0.7)
	traindata = Flux.DataLoader(train, batchsize=2, shuffle=true)
	testdata = Flux.DataLoader(test, batchsize=2)
end;

# ╔═╡ 5f35182e-2328-4f57-9972-f1f32b8c560f
md"""
# Train Model
"""

# ╔═╡ 7bbb7a92-3de5-47f9-8eaf-44e314f1a3cd
loss = Apollo.BinaryCrossEntropy();

# ╔═╡ 2c8f74a6-1652-41cc-a7e0-46ba8b58b68c
model = let
	# Build Model
	input = Apollo.Series(channels=2)
	encoder = Apollo.ResNet18(weights=:ImageNet)
	model = Apollo.UNet(input=input, encoder=encoder, activation=Flux.sigmoid)

	# Move Model to GPU
	model = Flux.gpu(model)
	
	# Prepare Optimiser
	opt = Flux.Optimisers.Adam(5e-4)
	opt_state = Flux.Optimisers.setup(opt, model)
	Flux.Optimisers.freeze!(opt_state.encoder)

	# Track Performance
	tracker = A.Tracker(
		"train_loss" => A.Loss(loss), 
		"train_MIoU" => A.MIoU([0,1]), 
		"test_loss" => A.Loss(loss), 
		"test_MIoU" => A.MIoU([0,1]),
	)

	# Train for 15 Epochs
 	for epoch in 1:30

		# Update Model
		Apollo.train!(loss, model, opt_state, Flux.gpu(traindata))

		# Evaluate Train Performance
		for (x, y) in traindata
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			Apollo.step!(tracker, r"train_", ŷ, y)
		end

		# Evaluate Validation Performance
		for (x, y) in testdata
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			Apollo.step!(tracker, r"test_", ŷ, y)
		end

		# End Epoch
		@info Apollo.printscores(tracker, metrics=r"_loss")
		Apollo.epoch!(tracker)
	end

	# Return Best Model
	model
end;

# ╔═╡ f7873f37-16c6-4f7d-9ca5-18a83f734ee6
train_metrics = Apollo.evaluate(traindata, A.Accuracy(), A.MIoU([0,1])) do (x, y)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ f20c0da0-2fbd-43a9-b119-544a4fbe974d
test_metrics = Apollo.evaluate(testdata, A.Accuracy(), A.MIoU([0,1])) do (x, y)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ 3cff2e1b-2152-4ef0-9532-ef4fad961f1c
md"""
# Quantiative Analysis

**Train Accuracy:** $(round(Float64(train_metrics.accuracy), digits=4))

**Train MIoU:** $(round(Float64(train_metrics.MIoU), digits=4))

**Test Accuracy:** $(round(Float64(test_metrics.accuracy), digits=4))

**Test MIoU:** $(round(Float64(test_metrics.MIoU), digits=4))
"""

# ╔═╡ 4cf5a2ba-601b-4804-a294-5290275ae618
md"""
# Qualitative Analysis
"""

# ╔═╡ 348d90c3-66ca-415d-9c50-013b5c7225bc
let
	s2_10m = R.Raster("data/RiceSC_South_S2/2018-01-05_10m.tif")

	lb = A.folddims(x -> quantile(x, 0.02), s2_10m, dims=R.Band)
	ub = A.folddims(x -> quantile(x, 0.98), s2_10m, dims=R.Band)
	
	s2_sampler = A.TileView(s2_10m, 1024)
	xsampler = A.TileView(features, 1024)
	ysampler = A.TileView(labels, 1024)
	tiles = A.zipobs(xsampler, ysampler, s2_sampler)

	plots = map(A.sampleobs(tiles, 3)) do (x, y, s2)
		rgb = A.rgb(s2, lb, ub, bands=[3, 2, 1])
		label = A.binmask(y)
		x = @pipe A.tensor(x, layerdim=R.Ti) |> A.normalize(_, μ, σ)
		pred = @pipe Flux.gpu(x) |> model |> Flux.cpu |> (_ .> 0.5) |> A.binmask
		return A.mosaicview((1,3), rgb, label, pred)
	end

	A.mosaicview((3,1), plots...)
end

# ╔═╡ Cell order:
# ╠═40c186f4-5b3d-11ef-2a2e-99823024c7fe
# ╠═bbbd8c59-027b-4770-8334-f5cb814fbebf
# ╠═d23808f0-f9b3-40c8-89f7-7a8b11a49f79
# ╟─c033b6d5-9811-4ca3-993a-9a7b9b537075
# ╠═8bf5bd76-f283-4b82-a78f-a5d9d9bbe221
# ╠═65fab3fc-ad45-41ce-a363-defacabc3981
# ╠═ebf8c65d-c349-4bef-ae2e-5524f0c8f773
# ╠═957ca4c5-c962-4b71-8a36-c05f27d257b0
# ╠═059904ca-21ac-47ba-bc60-eeeabfaa7382
# ╟─5f35182e-2328-4f57-9972-f1f32b8c560f
# ╠═7bbb7a92-3de5-47f9-8eaf-44e314f1a3cd
# ╠═2c8f74a6-1652-41cc-a7e0-46ba8b58b68c
# ╟─3cff2e1b-2152-4ef0-9532-ef4fad961f1c
# ╠═f7873f37-16c6-4f7d-9ca5-18a83f734ee6
# ╠═f20c0da0-2fbd-43a9-b119-544a4fbe974d
# ╟─4cf5a2ba-601b-4804-a294-5290275ae618
# ╠═348d90c3-66ca-415d-9c50-013b5c7225bc
