# run_imitation_pipeline.py
"""
Complete pipeline for imitation learning approach.
Runs expert demonstration extraction, behavioral cloning, and RL fine-tuning.
"""

import sys
from pathlib import Path
import time
import traceback

def run_expert_extraction():
    """Step 1: Extract expert demonstrations from trail data"""
    print("\n🎯 STEP 1: EXTRACTING EXPERT DEMONSTRATIONS")
    print("=" * 60)
    
    try:
        from trail_expert import main as expert_main
        expert_main()
        return True
    except Exception as e:
        print(f"❌ Expert extraction failed: {e}")
        traceback.print_exc()
        return False

def run_behavioral_cloning():
    """Step 2: Train behavioral cloning policy"""
    print("\n🎯 STEP 2: BEHAVIORAL CLONING TRAINING") 
    print("=" * 60)
    
    try:
        from behavioral_cloning import main as bc_main
        bc_main()
        return True
    except Exception as e:
        print(f"❌ Behavioral cloning failed: {e}")
        traceback.print_exc()
        return False

def run_rl_finetuning():
    """Step 3: Fine-tune with RL"""
    print("\n🎯 STEP 3: RL FINE-TUNING")
    print("=" * 60)
    
    try:
        from rl_finetune import main as rl_main
        rl_main()
        return True
    except Exception as e:
        print(f"❌ RL fine-tuning failed: {e}")
        traceback.print_exc()
        return False

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("🔍 CHECKING REQUIREMENTS")
    print("=" * 30)
    
    required_files = [
        "../data/processed/trail_coordinates.npy",
        "../physics_hiking_env.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    # Check Python packages
    required_packages = ['torch', 'sklearn', 'matplotlib', 'stable_baselines3', 'gymnasium']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All requirements satisfied")
    return True

def main():
    """Run complete imitation learning pipeline"""
    
    print("🚀 IMITATION LEARNING PIPELINE")
    print("=" * 50)
    print("This pipeline will:")
    print("1. Extract expert demonstrations from trail data")
    print("2. Train behavioral cloning policy to mimic expert")
    print("3. Fine-tune policy with reinforcement learning")
    print("=" * 50)
    
    start_time = time.time()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements not met. Please install missing dependencies.")
        return False
    
    # Step 1: Expert demonstration extraction
    if not run_expert_extraction():
        print("\n💥 Pipeline failed at expert extraction step")
        return False
    
    print("✅ Expert demonstrations extracted successfully")
    time.sleep(2)  # Brief pause between steps
    
    # Step 2: Behavioral cloning
    if not run_behavioral_cloning():
        print("\n💥 Pipeline failed at behavioral cloning step")
        return False
    
    print("✅ Behavioral cloning completed successfully")
    time.sleep(2)
    
    # Step 3: RL fine-tuning
    if not run_rl_finetuning():
        print("\n💥 Pipeline failed at RL fine-tuning step")
        return False
    
    print("✅ RL fine-tuning completed successfully")
    
    # Pipeline complete
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n🎉 IMITATION LEARNING PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"⏱️  Total time: {hours}h {minutes}m")
    print("\n📁 Generated files:")
    print("   - trail_demonstrations.pkl (expert demonstrations)")
    print("   - best_navigation_policy.pth (behavioral cloning model)")
    print("   - best_finetuned_policy.zip (RL fine-tuned model)")
    print("   - training_history.png (training curves)")
    
    print("\n🎯 USAGE:")
    print("   - Use 'best_finetuned_policy.zip' for best performance")
    print("   - Compare with main RL approach from parent directory")
    print("   - Test with test_agent.py using the fine-tuned model")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
