### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 94f376d0-522e-11ef-36b2-91ade41b6743
# ╠═╡ show_logs = false
begin
	import Pkg
	Pkg.add(url="https://github.com/JoshuaBillson/Apollo.jl")
	Pkg.add(["Rasters", "ArchGDAL", "Flux", "JLD2", "ImageShow"])
	Pkg.add(["Pipe",  "Statistics", "Random"])

	import Apollo, Rasters, ArchGDAL, Flux, JLD2, ImageShow
	using Random, Statistics
	using Pipe: @pipe

	# Uncomment to Add GPU Support
	Pkg.add(["CUDA", "cuDNN"])
	import CUDA, cuDNN
end

# ╔═╡ 8abfc331-beb4-433b-b966-173da123af87
md"""
# Import Packages
"""

# ╔═╡ 4b30620a-1906-4d21-ae22-7bd94628f901
const A = Apollo;

# ╔═╡ a2d26e2a-7635-4a6d-9cba-d7e53f83e63a
md"""
# Data Pipeline
"""

# ╔═╡ ea30afdc-41c6-4338-9d70-877b4c58e158
labels = Rasters.Raster("data/rice/labels_2018_2019_roi.tif", lazy=false);

# ╔═╡ 43ed91d8-d1c7-4253-9a46-4f338e3318fc
features_10m = Rasters.Raster("data/rice/2018-01-05_10m.tif", lazy=true);

# ╔═╡ 88bec67c-1a13-4efc-9d85-8fd99935e944
features_20m = Rasters.Raster("data/rice/2018-01-05_20m_sharpened.tif", lazy=true);

# ╔═╡ 93a3d1c9-0a34-4805-9fe3-ea0287b4c10f
features = cat(features_10m[:,:,:], features_20m[:,:,:], dims=Rasters.Band);

# ╔═╡ 0774f553-03a8-4f73-a090-020f74e0315a
μ = Apollo.folddims(mean, Float64.(features), dims=Rasters.Band);

# ╔═╡ bc1f54ca-0311-41f1-9d4d-29999557b104
σ = Apollo.folddims(std, Float64.(features), dims=Rasters.Band);

# ╔═╡ c196fb1a-42f7-4377-b28a-e9973c4d6b5c
begin
	# Sample 400 256x256 Tiles
	xsampler = Apollo.TileSampler(features, 256)
	ysampler = Apollo.TileSampler(labels, 256)

	# Generate Masks
	masks = Apollo.map(ysampler) do y
		# Initialize Mask
		mask = y .* 0.0f0

		# Add Labelled Rice Pixels
		if sum(y) >= 20
			indices = @pipe findall(==(1), y) |> Random.shuffle(_)[1:20]
			mask[indices] .= 1.0f0
		end

		# Add Labelled Background Pixels
		indices = @pipe findall(==(0), y) |> Random.shuffle(_)[1:20]
		mask[indices] .= 1.0f0

		# Return Mask
		return mask
	end

	# Pipeline
	t = A.Tensor() |> A.Normalize(μ, σ)
	dtypes = (A.Image(:x), A.Mask(:y), A.Mask(:m))
	transformed = A.transform(t, dtypes, A.zipobs(xsampler, ysampler, masks))

	# Remove Tiles With Less Than 1% Rice
	data = A.filterobs(transformed) do (x, y, m)
		return (sum(y) / length(y)) >= 0.01
	end

	# Create DataLoaders
	train, test = A.splitobs(data)
	traindata = Flux.DataLoader(train, batchsize=8, shuffle=true)
	testdata = Flux.DataLoader(test, batchsize=8)
end;

# ╔═╡ e7af3527-6e90-4ca5-bae0-9ea127c19afa
md"""
# Visualize Data
"""

# ╔═╡ e805f1f2-c567-4863-bf1c-03ca8d2d7d51
lb = Apollo.folddims(x -> quantile(x, 0.02), features, dims=Rasters.Band);

# ╔═╡ cf818c47-584c-419a-a90e-8bfb470fe42d
ub = Apollo.folddims(x -> quantile(x, 0.98), features, dims=Rasters.Band);

# ╔═╡ ac3bfdd5-e518-4acd-9746-06333da668ae
let
	plots = map(Apollo.sampleobs(data, 3)) do (x, y, m)
		rgb = Apollo.rgb(Apollo.denormalize(x, μ, σ), lb, ub, bands=[3,2,1])
		labels = Apollo.binmask(y)
		mask = ifelse.(m .== 1, Float32.(y), 0.35f0)
		mask = Apollo.binmask(mask)
		Apollo.mosaicview((1,3), rgb, labels, mask)
	end
	Apollo.mosaicview((3,1), plots...)
end

# ╔═╡ 1fae2ecd-ffab-4c2b-ac46-bc1ada70a910
md"""
# Train Model
"""

# ╔═╡ d2b79c82-6165-472d-b57e-41613fcd57fe
loss = A.MaskedLoss(A.BinaryCrossEntropy());

# ╔═╡ 9c8a1169-63f8-4863-8400-c0fd852defc1
tracker = Apollo.Tracker(
	"train_loss" => Apollo.Loss(A.BinaryCrossEntropy()), 
	"train_acc" => Apollo.Accuracy(), 
	"train_MIoU" => Apollo.MIoU([0,1]), 
	"test_loss" => Apollo.Loss(A.BinaryCrossEntropy()), 
	"test_acc" => Apollo.Accuracy(), 
	"test_MIoU" => Apollo.MIoU([0,1]),
);

# ╔═╡ f08df72c-7bea-42d1-b4ac-843d9f26ba92
# ╠═╡ show_logs = false
model = let
	# Init Model
	encoder = Apollo.ResNet18(weights=:ImageNet)
	unet = Apollo.UNet(encoder=encoder, channels=10, activation=Flux.sigmoid)
	model = Flux.gpu(unet)
	
	# Freeze Backbone
	opt = Flux.Optimisers.Adam(5e-4)
	opt_state = Flux.Optimisers.setup(opt, model)
	Flux.Optimisers.freeze!(opt_state.encoder)

	# Store Fit Result
	best_loss = Inf32
	fitresult = model |> Flux.cpu

	# Train for 20 Epochs
 	for epoch in 1:30

		# Update Model
 		for (x, y, mask) in Flux.gpu(traindata)
			grads = Flux.gradient(m -> loss(m(x), y, mask), model)
 			Flux.update!(opt_state, model, grads[1])
 		end

		# Evaluate Train Performance
		for (x, y, mask) in traindata
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			Apollo.step!(tracker, r"train_", ŷ, y)
		end

		# Evaluate Validation Performance
		masked_loss = 0.0f0
		for (x, y, mask) in testdata
			ŷ = Flux.gpu(x) |> model |> Flux.cpu
			Apollo.step!(tracker, r"test_", ŷ, y)
			masked_loss += loss(ŷ, y, mask)
		end
		masked_loss /= length(testdata)

		# End Epoch
		scores = Apollo.printscores(tracker, metrics=r"_loss")
		@info scores * "  masked_loss: $masked_loss"
		Apollo.epoch!(tracker)

		# Update Fitresult
		if masked_loss < best_loss
			best_loss = masked_loss
			fitresult = model |> Flux.cpu
		end
	end

	# Return Best Model
	fitresult |> Flux.gpu
end;

# ╔═╡ 49cc0ee1-6485-4058-b841-49eb8cdfba96
eval_metrics = [A.Loss(A.BinaryCrossEntropy()), A.Accuracy(), A.MIoU([0,1])];

# ╔═╡ 45fdedd2-7b46-4cdd-914d-2ea756befe6b
train_eval = A.evaluate(traindata, eval_metrics...) do (x, y, m)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ dc1af415-907f-4df3-a2b6-abade43f7ba4
test_eval = A.evaluate(testdata, eval_metrics...) do (x, y, m)
	ŷ = Flux.gpu(x) |> model |> Flux.cpu
	return (ŷ, y)
end;

# ╔═╡ 2a501be0-1c37-4397-98bb-30fb867b99e0
md"""
# Quantitative Analysis

**Train Loss:** $(round(train_eval.loss, digits=4))

**Train Accuracy:** $(round(train_eval.accuracy, digits=4))

**Train MIoU:** $(round(Float64(train_eval.MIoU), digits=4))

**Test Loss:** $(round(test_eval.loss, digits=4))

**Test Accuracy:** $(round(test_eval.accuracy, digits=4))

**Test MIoU:** $(round(Float64(test_eval.MIoU), digits=4))
"""

# ╔═╡ 83579d51-9854-4202-96fe-74c445c315ac
md"""
# Qualitative Analysis
"""

# ╔═╡ b3122b60-75c7-4bf5-a7a2-8f24598e5fe5
let
	# Plot Label/Prediction Pairs
	plots = map(Apollo.sampleobs(test, 3)) do (x, y, mask)
		# Plot RGB
		rgb = Apollo.rgb(Apollo.denormalize(x, μ, σ), lb, ub, bands=[3,2,1])

		# Plot label
		label = Apollo.binmask(y)

		# Plot Prediction
		ŷ = @pipe Flux.gpu(x) |> model |> Flux.cpu |> round.(Int, _)
		pred = Apollo.binmask(ŷ)
		
		# Combine RGB, Label, and Prediction
		Apollo.mosaicview((1,3), rgb, label, pred)
	end

	# Plot Mosaic
	Apollo.mosaicview((3,1), plots...)
end

# ╔═╡ Cell order:
# ╟─8abfc331-beb4-433b-b966-173da123af87
# ╠═94f376d0-522e-11ef-36b2-91ade41b6743
# ╠═4b30620a-1906-4d21-ae22-7bd94628f901
# ╟─a2d26e2a-7635-4a6d-9cba-d7e53f83e63a
# ╠═ea30afdc-41c6-4338-9d70-877b4c58e158
# ╠═43ed91d8-d1c7-4253-9a46-4f338e3318fc
# ╠═88bec67c-1a13-4efc-9d85-8fd99935e944
# ╠═93a3d1c9-0a34-4805-9fe3-ea0287b4c10f
# ╠═0774f553-03a8-4f73-a090-020f74e0315a
# ╠═bc1f54ca-0311-41f1-9d4d-29999557b104
# ╠═c196fb1a-42f7-4377-b28a-e9973c4d6b5c
# ╟─e7af3527-6e90-4ca5-bae0-9ea127c19afa
# ╠═e805f1f2-c567-4863-bf1c-03ca8d2d7d51
# ╠═cf818c47-584c-419a-a90e-8bfb470fe42d
# ╠═ac3bfdd5-e518-4acd-9746-06333da668ae
# ╟─1fae2ecd-ffab-4c2b-ac46-bc1ada70a910
# ╠═d2b79c82-6165-472d-b57e-41613fcd57fe
# ╠═9c8a1169-63f8-4863-8400-c0fd852defc1
# ╠═f08df72c-7bea-42d1-b4ac-843d9f26ba92
# ╟─2a501be0-1c37-4397-98bb-30fb867b99e0
# ╠═49cc0ee1-6485-4058-b841-49eb8cdfba96
# ╠═45fdedd2-7b46-4cdd-914d-2ea756befe6b
# ╠═dc1af415-907f-4df3-a2b6-abade43f7ba4
# ╟─83579d51-9854-4202-96fe-74c445c315ac
# ╠═b3122b60-75c7-4bf5-a7a2-8f24598e5fe5
