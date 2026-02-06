# %%
#!/usr/bin/env python3
"""
run_diffusion.py

Command-line interface for training and comparing diffusion flow and score models.
"""

import argparse
import torch
import diffusion_utils as du


def main():
    parser = argparse.ArgumentParser(description="Train and visualize diffusion flow/score models.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_flow", action="store_true", help="Train the flow model")
    parser.add_argument("--train_score", action="store_true", help="Train the score model")
    parser.add_argument("--compare", action="store_true", help="Run comparison plots between models")
    parser.add_argument("--plot", action="store_true", help="Plot flow and score trajectories")
    parser.add_argument("--simulate", action="store_true", help="Simulate flow and score trajectories")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")
    nmodes = 5
    scale = 10.0
    angles = torch.linspace(0, 2 * 3.14159, nmodes + 1)[:nmodes]
    means = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
    
    # 2. SQUASH the Y-axis to create spectral disparity
    # X-axis (Dim 0) stays large (scale 10). Y-axis (Dim 1) becomes small (scale 1).
    squash_factor = torch.tensor([1.0, 0.1]).to(device)
    means = means * squash_factor
    
    # 3. Create anisotropic covariance for clusters too
    covs = torch.diag_embed(torch.ones(nmodes, 2)).to(device)
    covs = covs * (torch.tensor([1.0, 0.1]).to(device)**2) # Small variance in Y
    
    weights = torch.ones(nmodes).to(device) / nmodes
    
    #p_data = du.GaussianMixture.symmetric_4D(nmodes=5, std=PARAMS["target_std"]).to(device)
    p_data = du.GaussianMixture(means, covs, weights).to(device)

    print("Computing dataset stats for Equal SNR Approach")

    stat_samples = p_data.sample(10000)
    data_std = torch.std(stat_samples, dim=0)
    print(f"Data Std (Signal Variance C^1/2): {data_std}")
    # Prepare conditional probability path
    path = du.GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=du.LinearAlpha(),
        beta=du.SquareRootBeta(),
        data_std=data_std
    ).to(device)

    # Shared model hyperparameters
    dim = p_data.dim
    hiddens = [64,64,64,64]

    flow_model, score_model = None, None

    # -------------------------------
    # Train Flow Matching Model
    # -------------------------------
    if args.train_flow:
        print("\n=== Training Flow Matching Model ===")
        flow_model = du.MLPVectorField(dim=dim, hiddens=hiddens)
        flow_trainer = du.ConditionalFlowMatchingTrainer(path, flow_model)
        flow_trainer.train(
            num_epochs=args.epochs,
            device=device,
            lr=1e-3,
            batch_size=args.batch_size,
        )
        torch.save(flow_model.state_dict(), "flow_model.pt")
        print("Saved flow model to flow_model.pt")

    # -------------------------------
    # Train Score Matching Model
    # -------------------------------
    if args.train_score:
        print("\n=== Training Score Matching Model ===")
        score_model = du.MLPScore(dim=dim, hiddens=hiddens)
        score_trainer = du.ConditionalScoreMatchingTrainer(path, score_model)
        score_trainer.train(
            num_epochs=args.epochs,
            device=device,
            lr=1e-3,
            batch_size=args.batch_size,
        )
        torch.save(score_model.state_dict(), "score_model.pt")
        print("Saved score model to score_model.pt")

    # -------------------------------
    # Load models (if needed)
    # -------------------------------
    if flow_model is None:
        try:
            flow_model = du.MLPVectorField(dim=dim, hiddens=hiddens)
            flow_model.load_state_dict(torch.load("flow_model.pt"))
            flow_model = flow_model.to(device)
            print("Loaded pretrained flow_model.pt")
        except FileNotFoundError:
            pass

    if score_model is None:
        try:
            score_model = du.MLPScore(dim=dim, hiddens=hiddens)
            score_model.load_state_dict(torch.load("score_model.pt"))
            score_model = score_model.to(device)
            print("Loaded pretrained score_model.pt")
        except FileNotFoundError:
            pass

    # -------------------------------
    # Comparisons
    # -------------------------------
    if args.compare and flow_model and score_model:
        print("\n=== Comparing Vector Fields and Scores ===")
        du.compare_vector_fields(path, flow_model, score_model)
        du.compare_scores(path, flow_model, score_model)

    # -------------------------------
    # Plot Trajectories
    # -------------------------------
    if args.plot:
        print("\n=== Plotting Flow and Score Simulations ===")
        if flow_model:
            flow_score_model = du.ScoreFromVectorField(flow_model,path.alpha,path.beta)
            du.plot_flow(path, flow_model, 1000, output_file="flow_trajectory.pdf")
            du.plot_score(path, flow_model, flow_score_model, 300, output_file="flow_stochastic_trajectory.pdf")
        if score_model:
            score_flow_model = du.VectorFieldFromScore(score_model,path.alpha,path.beta) 
            du.plot_flow(path, score_flow_model, 1000, output_file="score_deterministic_trajectory.pdf") 
            du.plot_score(path, score_flow_model, score_model, 300, output_file="score_trajectory.pdf")
            du.plot_score(path, flow_model, score_model, 300, output_file="score_flow_trajectory.pdf")

    # -------------------------------
    # Simulate
    # -------------------------------
    if args.simulate:
        num_samples = 10000
        timestep_list = torch.tensor([4,8,16,32,64,128,256])  # sweep values

        results = {}
        if flow_model and False:
            flow_score_model = du.ScoreFromVectorField(flow_model, path.alpha, path.beta)
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_flow(path, flow_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Flow] Steps={num_steps:<4d} W={W:.4f}")
            results["flow_deterministic"] = (timestep_list, W_list)

            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_score(path, flow_model, flow_score_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Flow Stochastic] Steps={num_steps:<4d} W={W:.4f}")
            results["flow_stochastic"] = (timestep_list, W_list)
        if score_model:
            score_flow_model = du.VectorFieldFromScore(score_model, path.alpha, path.beta)
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_score(path, score_flow_model, score_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                #target_samples = path.p_data.sample(num_samples) 
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Score] Steps={num_steps:<4d} W={W:.4f}")
            results["score_stochastic"] = (timestep_list, W_list)

            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_flow(path, score_flow_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                #target_samples = path.p_data.sample(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Score Deterministic] Steps={num_steps:<4d} W={W:.4f}")
            results["score_deterministic"] = (timestep_list, W_list)
        du.plot_results(results)
if __name__ == "__main__":
    main()
