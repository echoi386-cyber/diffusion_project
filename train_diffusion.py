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

    # FIX 1: Handle Colab arguments by passing empty list
    args = parser.parse_args(args=[])
    
    # Enable flags manually for notebook execution
    args.train_score = True
    args.train_flow = True  # Added back in case you want to train flow too
    args.compare = True
    args.plot = True
    args.simulate = True

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    PARAMS = {
        "scale": 15.0,
        "target_scale": 10.0,
        "target_std": 1.0,
    }

    # === DATA SETUP ===
    # Using your original symmetric_2D format
    p_data = du.GaussianMixture.symmetric_2D(
        nmodes=5, 
        std=PARAMS["target_std"], 
        scale=PARAMS["target_scale"]
    ).to(device)

    # === EQUAL SNR STATS ===
    print("Compute data stats for Equal SNR")
    stat_samples = p_data.sample(10000)
    data_std = torch.std(stat_samples, dim=0)
    print(f"Data Std: {data_std}")
    
    # Prepare conditional probability path with data_std
    path = du.GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=du.LinearAlpha(),
        beta=du.SquareRootBeta(),
        data_std=data_std
    ).to(device)

    dim = p_data.dim
    hiddens = [64,64,64,64]

    flow_model, score_model = None, None

    # -------------------------------
    # Train Flow Matching Model
    # -------------------------------
    if args.train_flow:
        print("\n=== Training Flow Matching Model ===")
        flow_model = du.MLPVectorField(dim=dim, hiddens=hiddens).to(device)
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
        score_model = du.MLPScore(dim=dim, hiddens=hiddens).to(device)
        score_trainer = du.ConditionalScoreMatchingTrainer(path, score_model)
        
        # FIX 2: Lower Learning Rate for Weighted Loss (Prevents exploding/divergence)
        score_trainer.train(
            num_epochs=args.epochs,
            device=device,
            lr=2e-4, 
            batch_size=args.batch_size,
        )
        torch.save(score_model.state_dict(), "score_model.pt")
        print("Saved score model to score_model.pt")

    # -------------------------------
    # Load models (if needed)
    # -------------------------------
    if flow_model is None:
        try:
            flow_model = du.MLPVectorField(dim=dim, hiddens=hiddens).to(device)
            flow_model.load_state_dict(torch.load("flow_model.pt", map_location=device))
            print("Loaded pretrained flow_model.pt")
        except FileNotFoundError:
            pass

    if score_model is None:
        try:
            score_model = du.MLPScore(dim=dim, hiddens=hiddens).to(device)
            score_model.load_state_dict(torch.load("score_model.pt", map_location=device))
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
    # Simulate (Full Original Logic)
    # -------------------------------
    if args.simulate:
        num_samples = 10000
        timestep_list = torch.tensor([4,8,16,32,64,128,256])  # sweep values

        results = {}
        
        # 1. Flow Model (Deterministic)
        if flow_model:
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_flow(path, flow_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Flow] Steps={num_steps:<4d} W={W:.4f}")
            results["flow_deterministic"] = (timestep_list, W_list)

            # 2. Flow Model (Stochastic)
            flow_score_model = du.ScoreFromVectorField(flow_model, path.alpha, path.beta)
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_score(path, flow_model, flow_score_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Flow Stochastic] Steps={num_steps:<4d} W={W:.4f}")
            results["flow_stochastic"] = (timestep_list, W_list)
        
        # 3. Score Model (Stochastic & Deterministic)
        if score_model:
            score_flow_model = du.VectorFieldFromScore(score_model, path.alpha, path.beta)
            
            # Stochastic Score
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_score(path, score_flow_model, score_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Score Stochastic] Steps={num_steps:<4d} W={W:.4f}")
            results["score_stochastic"] = (timestep_list, W_list)

            # Deterministic Score
            W_list = []
            for num_steps in timestep_list:
                samples = du.simulate_flow(path, score_flow_model, num_samples, num_steps)
                target_samples = path.p_data.sample_projected(num_samples)
                W = du.wasserstein_distance(samples, target_samples)
                W_list.append(W.cpu())
                print(f"[Score Deterministic] Steps={num_steps:<4d} W={W:.4f}")
            results["score_deterministic"] = (timestep_list, W_list)
            
        du.plot_results(results)

if __name__ == "__main__":
    main()