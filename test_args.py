import sys
import os
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), 'event'))
import train

def run_test():
    # We will pass the arguments exactly as train.sh would do
    test_args = ["--config", "event/cifar10dvs_rebuttal_ablate.yaml"]
    
    # We temporarily replace sys.argv so train.parse_args parses our test config
    original_argv = sys.argv
    sys.argv = ['train.py'] + test_args

    try:
        args = train.parse_args()
        
        print("\n--- YAML Extraction Verification ---")
        print(f"Loaded config: {test_args[1]}")
        
        # Read the yaml to get expected values
        with open('event/cifar10dvs_rebuttal_ablate.yaml', 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        
        params_to_inspect = ['snnbp-epsilon', 'snnbp-alpha', 'snnbp-beta', 'snnbp-intervention', 'snnbp-eta', 'snnbp-blend', 'snnbp-p']
        all_match = True
        
        for param in params_to_inspect:
            if param in yaml_cfg:
                yaml_val = yaml_cfg[param]
                # Argparse will store it with underscores
                attr_name = param.replace('-', '_')
                actual_val = getattr(args, attr_name, None)
                
                if actual_val == yaml_val:
                    print(f"[SUCCESS] {param} -> args.{attr_name} = {actual_val} (Matches YAML)")
                else:
                    print(f"[FAIL] {param} -> args.{attr_name} = {actual_val} (EXPECTED {yaml_val} from YAML)")
                    all_match = False
                    
        if all_match:
            print("\nVERIFIED: The hyphenated parameters mapped flawlessly to argparse underscores!")
            print("The model is now receiving the correct hyperparameters.")
        else:
            print("\nERROR: Parameter mapping failed.")
            
    finally:
        sys.argv = original_argv

if __name__ == '__main__':
    run_test()
