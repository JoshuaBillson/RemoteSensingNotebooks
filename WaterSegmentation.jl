### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 2e1a738a-4ab3-11ef-2d6a-6de98b3c0476
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

# ╔═╡ 66fe9f19-b3c7-4e56-97f6-7d73fdcce33a
const A = Apollo;

# ╔═╡ 7c9833c6-2d34-4c9a-a04a-1eb9365eedd5
md"""
# Load Data
"""

# ╔═╡ d16152c6-96bd-4477-b6cf-9fba99b25fff
labels = Float32.(Rasters.Raster("data/YuanWater/label.tif", lazy=false) .== 0xff);

# ╔═╡ 16a9c258-c382-47d0-b4a4-cc4f81a46e78
features = Rasters.Raster("data/YuanWater/rgb_nir.tif", lazy=true)[:,:,[3,2,1,4]];

# ╔═╡ ce393bf6-f4f4-46bb-8ac3-1cbffe54a417
md"""
# Define Data Pipeline
"""

# ╔═╡ d64dc24e-9335-4fcd-b1c5-3aa6acadb32b
μ = A.folddims(mean, features, dims=Rasters.Band);

# ╔═╡ cc3a0a9f-d6f0-4c10-bf2a-b0a1e3f53561
σ = A.folddims(std, features, dims=Rasters.Band);

# ╔═╡ 2060c5ed-a0db-43e3-a860-9ffb22ec7241
begin
	# Sample 256x256 Tiles
	xsampler = A.TileView(features, 256)
	ysampler = A.TileView(labels, 256)
	samplers = A.zipobs(xsampler, ysampler)

	# Remove Samples With Less Than 1% Water
	tiles = A.filterobs(samplers) do (x, y)
		return (sum(y) / length(y)) > 0.01
	end

	# Define Data Pipeline
	t = A.Tensor() |> A.Normalize(μ, σ)
	pipeline = A.transform(t, (A.Image(), A.Mask()), tiles)

	# Split Into Train and Test
	traindata, testdata = A.splitobs(pipeline, at=0.7)
	train = Flux.DataLoader(traindata, batchsize=8, shuffle=true)
	test = Flux.DataLoader(testdata, batchsize=8)
end;

# ╔═╡ da83b16e-9262-4a27-a076-44d335313aae
md"""
# Visualize Data
"""

# ╔═╡ a5aa99ca-23ec-46fd-b781-838e58bb69d2
lb = A.folddims(x -> quantile(x, 0.02), features, dims=Rasters.Band);

# ╔═╡ 6f32d16b-a9c8-48bb-8d2e-c115b86e447e
ub = A.folddims(x -> quantile(x, 0.98), features, dims=Rasters.Band);

# ╔═╡ 2517afec-4100-4a78-a941-bf68f9d90191
let
	plots = map(A.sampleobs(traindata, 9)) do (x, y)
		rgb = @pipe A.denormalize(x, μ, σ) |> A.rgb(_, lb, ub, bands=[3,2,1])
		label = A.binmask(y)
		Images.mosaicview(rgb, label, nrow=1, ncol=2, npad=5, fillvalue=1.0)
	end
	Images.mosaicview(plots..., nrow=3, ncol=3, npad=5, fillvalue=1.0)
end

# ╔═╡ 537be333-41de-407f-9e61-dcba0b7941a2
md"""
# Train Model
"""

# ╔═╡ 42bb31a4-178f-4be4-aa5f-5d4023c910b0
loss = A.DiceLoss();

# ╔═╡ 1132d5c1-9efc-4f65-bf84-33abc6c10927
tracker = Apollo.Tracker(
	"train_loss" => A.Loss(loss), 
	"test_loss" => A.Loss(loss), 
);

# ╔═╡ bc3ea369-ea8c-4edc-a359-b133e17a398f
# ╠═╡ show_logs = false
model = let
	# Initialize Model
	input = A.Single(channels=4)
	encoder = A.ResNet18(weights=:ImageNet)
	model = A.UNet(encoder=encoder, input=input, activation=Flux.sigmoid) |> Flux.gpu

	# Save Fit Result
	fitresult = model |> Flux.cpu

	# Initialize Optimizer
	opt = Flux.Optimisers.Adam(5e-4)
	opt_state = Flux.Optimisers.setup(opt, model)
	
	# Freeze Encoder
	Flux.freeze!(opt_state.encoder)

	# Train for 20 Epochs
	for epoch in 1:20

		# Iterate Over Train Data
		A.train!(loss, model, opt_state, Flux.gpu(train))
		
		# Evaluate Train Performance
		for (x, y) in train
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			A.step!(tracker, "train_loss", ŷ, y)
		end

		# Evaluate Test Performance
		for (x, y) in test
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			A.step!(tracker, "test_loss", ŷ, y)
		end

		# End Epoch
		A.epoch!(tracker)
		@info A.printscores(tracker, epoch=epoch)

		# Update Fit Result
		if A.best_epoch(tracker, A.Min("test_loss")) == epoch
			fitresult = model |> Flux.cpu
		end
	end

	# Return Best Model
	fitresult |> Flux.gpu
end

# ╔═╡ e6958d4d-7441-4b56-a89e-3d724a225e41
eval_metrics = [A.MIoU([0,1]), A.Accuracy(), A.Loss(loss)];

# ╔═╡ 2270d5d5-da8f-4af3-8567-7bd17652b694
results = A.evaluate(test, eval_metrics...) do (x, y)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ c8b37a14-e923-4df2-bd7d-1debf4f06658
md"""
# Quantitative Analysis

**Test Loss:** $(round(results.loss, digits=4))

**Test Accuracy:** $(round(results.accuracy, digits=4))

**Test MIoU:** $(round(Float64(results.MIoU), digits=4))
"""

# ╔═╡ 66816fc3-eb62-47c4-bda6-76acd4142c7c
md"""
# Qualitative Analysis
"""

# ╔═╡ 14c004ff-2fb9-4c33-8595-f06f2894560d
let
	plots = map(A.sampleobs(testdata, 6)) do (x, y)
		ŷ = @pipe Flux.gpu(x) |> model |> Flux.cpu |> (_ .> 0.5)
		rgb = @pipe A.denormalize(x, μ, σ) |> A.rgb(_, lb, ub, bands=[3,2,1])
		pred, label = map(A.binmask, (ŷ, y))
		A.mosaicview((1,3), rgb, pred, label)
	end
	A.mosaicview((3,2), plots...)
end

# ╔═╡ b3bbfc28-8444-4ee5-ba7a-b4f1157f9917
md"""
# Save Model
"""

# ╔═╡ 6acf269b-29b3-4ee3-9681-7430015e7dd7
false && let
	model_state = model |> Flux.cpu |> Flux.state
	JLD2.jldsave("models/water_resnet18_imagenet.jld2"; model_state)
end;

# ╔═╡ 92acf6ce-7fec-41de-8de8-281af1a412c7
md"""
# Model Generalization
"""

# ╔═╡ d9286bfb-b3a8-46ab-9362-e7286c6c3334
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

# ╔═╡ e46d6703-62e0-4f2e-8017-2784c6a69567
brazil_features = Rasters.Raster("data/RiceSC_South_S2/2018-01-05_10m.tif");

# ╔═╡ 4a7f7517-249a-4fad-90df-541253a950bd
brazil_μ = A.folddims(mean, Float64.(brazil_features), dims=Rasters.Band);

# ╔═╡ f7f308ea-6f00-4987-810d-a14ad1ef6b70
brazil_σ = A.folddims(std, Float64.(brazil_features), dims=Rasters.Band);

# ╔═╡ 50efc360-e8bc-4a79-b9d3-9bcb409ad94e
preds = map(A.TileView(brazil_features, 1024)) do x
		@pipe A.tensor(x) |> 
		A.normalize(_, brazil_μ, brazil_σ) |>
		Flux.gpu |>
		model |>
		Flux.cpu |>
		UInt8.(_ .> 0.5) |>
		Rasters.Raster(_[:,:,1,1], Rasters.dims(x)[1:2])
end;

# ╔═╡ 0ce60e0f-4ab5-4775-bd5f-a5f794e9f9c6
mosaic = merge_tiles(Rasters.dims(brazil_features)[1:2], preds...);

# ╔═╡ e4aef0fd-73f0-4db0-a546-d9872fb8ba47
let
	lb = Apollo.folddims(x -> quantile(x, 0.02), brazil_features, dims=Rasters.Band)
	ub = Apollo.folddims(x -> quantile(x, 0.98), brazil_features, dims=Rasters.Band)
	rgb = A.rgb(brazil_features, lb, ub, bands=[3, 2, 1])
	water_mask = A.binmask(mosaic)
	A.mosaicview((1,2), rgb, water_mask)
end

# ╔═╡ Cell order:
# ╠═2e1a738a-4ab3-11ef-2d6a-6de98b3c0476
# ╠═66fe9f19-b3c7-4e56-97f6-7d73fdcce33a
# ╟─7c9833c6-2d34-4c9a-a04a-1eb9365eedd5
# ╠═d16152c6-96bd-4477-b6cf-9fba99b25fff
# ╠═16a9c258-c382-47d0-b4a4-cc4f81a46e78
# ╟─ce393bf6-f4f4-46bb-8ac3-1cbffe54a417
# ╠═d64dc24e-9335-4fcd-b1c5-3aa6acadb32b
# ╠═cc3a0a9f-d6f0-4c10-bf2a-b0a1e3f53561
# ╠═2060c5ed-a0db-43e3-a860-9ffb22ec7241
# ╟─da83b16e-9262-4a27-a076-44d335313aae
# ╠═a5aa99ca-23ec-46fd-b781-838e58bb69d2
# ╠═6f32d16b-a9c8-48bb-8d2e-c115b86e447e
# ╠═2517afec-4100-4a78-a941-bf68f9d90191
# ╟─537be333-41de-407f-9e61-dcba0b7941a2
# ╠═42bb31a4-178f-4be4-aa5f-5d4023c910b0
# ╠═1132d5c1-9efc-4f65-bf84-33abc6c10927
# ╠═bc3ea369-ea8c-4edc-a359-b133e17a398f
# ╟─c8b37a14-e923-4df2-bd7d-1debf4f06658
# ╠═e6958d4d-7441-4b56-a89e-3d724a225e41
# ╠═2270d5d5-da8f-4af3-8567-7bd17652b694
# ╟─66816fc3-eb62-47c4-bda6-76acd4142c7c
# ╠═14c004ff-2fb9-4c33-8595-f06f2894560d
# ╟─b3bbfc28-8444-4ee5-ba7a-b4f1157f9917
# ╠═6acf269b-29b3-4ee3-9681-7430015e7dd7
# ╟─92acf6ce-7fec-41de-8de8-281af1a412c7
# ╠═d9286bfb-b3a8-46ab-9362-e7286c6c3334
# ╠═e46d6703-62e0-4f2e-8017-2784c6a69567
# ╠═4a7f7517-249a-4fad-90df-541253a950bd
# ╠═f7f308ea-6f00-4987-810d-a14ad1ef6b70
# ╠═50efc360-e8bc-4a79-b9d3-9bcb409ad94e
# ╠═0ce60e0f-4ab5-4775-bd5f-a5f794e9f9c6
# ╠═e4aef0fd-73f0-4db0-a546-d9872fb8ba47
