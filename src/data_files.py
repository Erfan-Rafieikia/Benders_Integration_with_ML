import os
import pandas as pd

def parse_ufl_instance_sizes(directory):
    instance_data = []

    for filename in os.listdir(directory):
        # Skip .opt, .bub, and README files
        if filename.endswith((".opt", ".bub")) or filename.startswith("README"):
            continue
        
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, "r") as file:
                first_line = file.readline().strip()
                if not first_line:
                    continue  # skip empty files
                
                parts = first_line.split()
                if len(parts) < 2:
                    continue  # not a valid instance file
                
                n_facilities = int(parts[0])
                n_customers = int(parts[1])

                instance_data.append({
                    "file_name": filename,
                    "n_facilities": n_facilities,
                    "n_customers": n_customers
                })

        except Exception as e:
            print(f"Error reading file {filename}: {e}")

    # Save to Excel
    df = pd.DataFrame(instance_data)
    output_path = os.path.join(directory, "ufl_instance_summary.xlsx")
    df.to_excel(output_path, index=False)
    print(f"Saved summary to {output_path}")

# Example usage
directory_path = "G:/My Drive/Programming/Research Projects/Benders_Integration_with_ML/data/UFL"
parse_ufl_instance_sizes(directory_path)
