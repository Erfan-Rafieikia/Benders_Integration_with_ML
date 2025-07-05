def truncate_ufl_instance(input_file_path, output_file_path, n_keep_facilities=20, n_keep_customers=20):
    with open(input_file_path, "r") as infile:
        lines = infile.readlines()

    # Read number of facilities and customers
    n_facilities, n_customers = map(int, lines[0].strip().split())

    # Safety checks
    if n_keep_facilities > n_facilities or n_keep_customers > n_customers:
        raise ValueError("Requested number of facilities/customers exceeds original file.")

    # Get facility lines
    facility_lines = lines[1:1 + n_facilities]
    truncated_facilities = facility_lines[:n_keep_facilities]

    # Get customer lines (2 lines per customer: demand line, cost line)
    customer_lines = lines[1 + n_facilities:]
    truncated_customers = []

    for i in range(n_keep_customers):
        demand_line = customer_lines[2 * i].strip()
        cost_line = customer_lines[2 * i + 1].strip()

        # Keep only first `n_keep_facilities` costs
        costs = list(map(float, cost_line.split()))
        truncated_costs = costs[:n_keep_facilities]

        truncated_customers.append(f"{demand_line}\n")
        truncated_customers.append(" ".join(map(str, truncated_costs)) + "\n")

    # Write to output file
    with open(output_file_path, "w") as outfile:
        outfile.write(f"{n_keep_facilities} {n_keep_customers}\n")
        outfile.writelines(truncated_facilities)
        outfile.writelines(truncated_customers)

    print(f"Truncated instance written to {output_file_path}")

# Example usage
truncate_ufl_instance(
    input_file_path="G:/My Drive/Programming/Research Projects/Benders_Integration_with_ML/data/UFL/MO1",
    output_file_path="G:/My Drive/Programming/Research Projects/Benders_Integration_with_ML/data/UFL/MO0",
    n_keep_facilities=20,
    n_keep_customers=20
)
