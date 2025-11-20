"""
Example script showing how to use the centralized output_paths configuration.

This demonstrates the new recommended way to handle file paths in the project.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs.output_paths import (
    # Dataset paths
    get_dataset_dir,
    get_model_dataset_dir,
    get_layer_file_path,
    
    # Probe paths
    get_probe_base_dir,
    get_probe_architecture_dir,
    get_probe_file_path,
    get_probe_evaluation_summary_path,
    
    # Visualization paths
    get_plots_base_dir,
    get_evaluation_plots_dir,
    get_dashboard_path,
    
    # Utility functions
    ensure_directory_exists,
    get_all_model_names,
    get_all_architecture_names,
    print_path_structure
)


def example_1_save_embeddings():
    """Example: Saving model embeddings for a new dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Saving Model Embeddings")
    print("="*70)
    
    dataset_name = "TIMIT"
    model_name = "HUBERT_BASE"
    layer_num = 5
    
    # Get paths for train and test splits
    train_path = get_layer_file_path(model_name, layer_num, "train", dataset_name)
    test_path = get_layer_file_path(model_name, layer_num, "test", dataset_name)
    
    # Ensure directories exist
    ensure_directory_exists(train_path.parent)
    
    print(f"\nSave embeddings to:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    
    # Your code to save embeddings would go here:
    # with open(train_path, 'wb') as f:
    #     pickle.dump(train_embeddings, f)


def example_2_train_probes():
    """Example: Training probes with proper path management."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Training Probes")
    print("="*70)
    
    architecture = "linear"
    dataset_name = "TIMIT"
    
    # Get architecture directory
    arch_dir = get_probe_architecture_dir(architecture, dataset_name)
    ensure_directory_exists(arch_dir)
    
    print(f"\nArchitecture directory: {arch_dir}")
    
    # Loop through models and layers
    models = ["HUBERT_BASE", "WAV2VEC2_BASE"]
    features = ["voiced", "fricative", "nasal"]
    
    for model in models:
        for layer in range(3):  # First 3 layers for example
            for feature in features:
                # Get probe save path
                probe_path = get_probe_file_path(
                    architecture, model, layer, feature, dataset_name
                )
                ensure_directory_exists(probe_path.parent)
                
                print(f"  Train probe: {model}/layer_{layer:02d}/{feature}")
                print(f"    Save to: {probe_path}")
                
                # Your probe training code would go here:
                # probe = train_probe(...)
                # torch.save(probe.state_dict(), probe_path)
    
    # Save evaluation summary
    summary_path = get_probe_evaluation_summary_path(architecture, dataset_name)
    print(f"\n  Save evaluation summary to: {summary_path}")


def example_3_generate_visualizations():
    """Example: Generating plots and dashboard."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Generating Visualizations")
    print("="*70)
    
    dataset_name = "TIMIT"
    
    # Get evaluation plots directory
    eval_plots_dir = get_evaluation_plots_dir(dataset_name)
    ensure_directory_exists(eval_plots_dir)
    
    print(f"\nEvaluation plots directory: {eval_plots_dir}")
    
    # Example: Save accuracy plots
    accuracy_dir = eval_plots_dir / "01_Accuracy" / "linear"
    ensure_directory_exists(accuracy_dir)
    
    models = ["HUBERT_BASE", "HUBERT_LARGE"]
    for model in models:
        plot_path = accuracy_dir / f"{model}.png"
        print(f"  Save plot: {plot_path}")
        
        # Your plotting code would go here:
        # plt.savefig(plot_path)
    
    # Get dashboard path
    dashboard_path = get_dashboard_path(dataset_name)
    print(f"\n  Save dashboard to: {dashboard_path}")


def example_4_new_dataset():
    """Example: Working with a completely new dataset."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Working with New Dataset (BUCKEYE)")
    print("="*70)
    
    dataset_name = "BUCKEYE"
    
    # All paths automatically adapt to new dataset
    dataset_dir = get_dataset_dir(dataset_name)
    probe_dir = get_probe_base_dir(dataset_name)
    plots_dir = get_plots_base_dir(dataset_name)
    dashboard_path = get_dashboard_path(dataset_name)
    
    print(f"\nBUCKEYE dataset structure:")
    print(f"  Datasets:  {dataset_dir}")
    print(f"  Probes:    {probe_dir}")
    print(f"  Plots:     {plots_dir}")
    print(f"  Dashboard: {dashboard_path}")
    
    # Create directories
    for directory in [dataset_dir, probe_dir, plots_dir]:
        ensure_directory_exists(directory)
        print(f"  ✓ Created: {directory}")


def example_5_discovery():
    """Example: Discovering existing models and architectures."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Discovery - Find Existing Data")
    print("="*70)
    
    dataset_name = "TIMIT"
    
    # Find all models with embeddings
    models = get_all_model_names(dataset_name)
    print(f"\nModels with embeddings in {dataset_name}:")
    for model in models:
        print(f"  • {model}")
    
    # Find all trained architectures
    architectures = get_all_architecture_names(dataset_name)
    print(f"\nTrained probe architectures:")
    for arch in architectures:
        print(f"  • {arch}")


def example_6_complete_workflow():
    """Example: Complete workflow from embeddings to visualization."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Complete Workflow")
    print("="*70)
    
    dataset_name = "TIMIT"
    model_name = "HUBERT_BASE"
    architecture = "linear"
    
    print(f"\nWorkflow for {model_name} on {dataset_name} with {architecture} probes:")
    
    # Step 1: Embeddings
    print("\n1. Load/Save Embeddings:")
    for layer in [0, 6, 12]:
        train_path = get_layer_file_path(model_name, layer, "train", dataset_name)
        test_path = get_layer_file_path(model_name, layer, "test", dataset_name)
        print(f"   Layer {layer:2d}: {train_path.name}, {test_path.name}")
    
    # Step 2: Train probes
    print("\n2. Train Probes:")
    for feature in ["voiced", "fricative", "nasal"]:
        probe_path = get_probe_file_path(architecture, model_name, 6, feature, dataset_name)
        print(f"   {feature:10s}: {probe_path.relative_to(probe_path.parents[4])}")
    
    # Step 3: Save evaluation
    print("\n3. Save Evaluation:")
    summary_path = get_probe_evaluation_summary_path(architecture, dataset_name)
    print(f"   Summary: {summary_path.name}")
    
    # Step 4: Generate plots
    print("\n4. Generate Plots:")
    eval_plots_dir = get_evaluation_plots_dir(dataset_name)
    print(f"   Location: {eval_plots_dir.relative_to(eval_plots_dir.parents[2])}")
    
    # Step 5: Create dashboard
    print("\n5. Create Dashboard:")
    dashboard_path = get_dashboard_path(dataset_name)
    print(f"   Dashboard: {dashboard_path.name}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("OUTPUT PATHS CONFIGURATION - USAGE EXAMPLES")
    print("="*70)
    
    # Show current structure
    print_path_structure("TIMIT")
    
    # Run examples
    example_1_save_embeddings()
    example_2_train_probes()
    example_3_generate_visualizations()
    example_4_new_dataset()
    example_5_discovery()
    example_6_complete_workflow()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nFor more details, see:")
    print("  - SUMMARY_FILES/OUTPUT_PATH_GUIDE.md")
    print("  - configs/output_paths.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
