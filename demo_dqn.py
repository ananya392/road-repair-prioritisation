#!/usr/bin/env python3
"""
Demo script for DQN Road Repair Automation
This script provides a quick demonstration of the DQN system
"""

import os
import sys
import subprocess
import time

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("DQN ROAD REPAIR AUTOMATION DEMO")
    print("=" * 60)
    print("This demo will show you how to automate road repair")
    print("decisions using Deep Q-Network (DQN) reinforcement learning.")
    print("=" * 60)

def check_requirements():
    """Check if required files exist"""
    print("\nChecking requirements...")
    
    required_files = [
        "coimbatore_road_preprocessed.geojson",
        "road_repair_env.py",
        "dqn_agent.py",
        "train_dqn.py",
        "main_dqn.py",
        "evaluate_dqn.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(" Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("All required files found!")
        return True

def run_preprocessing():
    """Run data preprocessing if needed"""
    if not os.path.exists("coimbatore_road_preprocessed.geojson"):
        print("\n Running data preprocessing...")
        try:
            subprocess.run([sys.executable, "preprocess_road_data.py"], check=True)
            print("Data preprocessing completed!")
        except subprocess.CalledProcessError:
            print("Data preprocessing failed!")
            return False
    else:
        print(" Preprocessed data already exists!")
    return True

def show_menu():
    """Show demo menu"""
    print("\n DEMO OPTIONS:")
    print("1.  Quick Training (100 episodes)")
    print("2.  Full Training (1000 episodes)")
    print("3.  Visualize Existing Model")
    print("4.  Evaluate Strategies")
    print("5.  Comprehensive Analysis")
    print("6.  Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("Please enter a number between 1 and 6")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def run_quick_training():
    """Run quick training demo"""
    print("\n🏃 Running quick training (100 episodes)...")
    print("This will take a few minutes...")
    
    try:
        # Modify train_dqn.py to run fewer episodes
        with open("train_dqn.py", "r") as f:
            content = f.read()
        
        # Replace episodes=1000 with episodes=100
        modified_content = content.replace("episodes=1000", "episodes=100")
        
        with open("train_dqn_quick.py", "w") as f:
            f.write(modified_content)
        
        subprocess.run([sys.executable, "train_dqn_quick.py"], check=True)
        
        # Clean up
        os.remove("train_dqn_quick.py")
        
        print("Quick training completed!")
        print("Model saved as 'dqn_model_final.pth'")
        
    except subprocess.CalledProcessError:
        print("Training failed!")
        return False
    
    return True

def run_full_training():
    """Run full training"""
    print("\n Running full training (1000 episodes)...")
    print("This will take 15-30 minutes depending on your system...")
    
    try:
        subprocess.run([sys.executable, "train_dqn.py"], check=True)
        print("Full training completed!")
        print("Model saved as 'dqn_model_final.pth'")
    except subprocess.CalledProcessError:
        print("Training failed!")
        return False
    
    return True

def run_visualization():
    """Run visualization"""
    print("\n Running visualization...")
    
    try:
        subprocess.run([sys.executable, "main_dqn.py"], check=True)
        print("Visualization completed!")
    except subprocess.CalledProcessError:
        print("Visualization failed!")
        return False
    
    return True

def run_evaluation():
    """Run strategy evaluation"""
    print("\n Running strategy evaluation...")
    
    try:
        subprocess.run([sys.executable, "evaluate_dqn.py"], check=True)
        print("Evaluation completed!")
    except subprocess.CalledProcessError:
        print("Evaluation failed!")
        return False
    
    return True

def run_analysis():
    """Run comprehensive analysis"""
    print("\n Running comprehensive analysis...")
    print("This includes detailed performance metrics and pattern analysis...")
    
    try:
        subprocess.run([sys.executable, "evaluate_dqn.py"], check=True)
        print("Analysis completed!")
        print("Check the generated plots for detailed insights!")
    except subprocess.CalledProcessError:
        print("Analysis failed!")
        return False
    
    return True

def main():
    """Main demo function"""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n Please ensure all required files are present before running the demo.")
        return
    
    # Run preprocessing if needed
    if not run_preprocessing():
        print("\n Demo cannot continue without preprocessed data.")
        return
    
    print("\n Setup complete! Ready to start the demo.")
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            if run_quick_training():
                print("\n Quick training completed! You can now run visualization (option 3)")
        elif choice == '2':
            if run_full_training():
                print("\n Full training completed! You can now run visualization (option 3)")
        elif choice == '3':
            if not os.path.exists("dqn_model_final.pth"):
                print("\n No trained model found! Please run training first (option 1 or 2)")
            else:
                run_visualization()
        elif choice == '4':
            run_evaluation()
        elif choice == '5':
            run_analysis()
        elif choice == '6':
            print("\n Thanks for trying the DQN Road Repair Demo!")
            print("Check out the generated files and plots for more insights!")
            break
        
        if choice in ['1', '2', '3', '4', '5']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Demo interrupted. Goodbye!")
        sys.exit(0)
