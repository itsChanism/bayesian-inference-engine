#Jiamu Tang
#jtang41
import itertools
import random
import sys


class Variable:
    def __init__(self, name, domain):
        self.name = name #name for vars like rain, x
        self.domain = domain #possible values
        self.parents = [] #parent vars
        self.children = [] #children cars
        self.cpt = []
#entire class
class BayesianNetwork:
    def __init__(self):
        self.variables = {}  #dictionary for vars
        self.topological_order = [] #sorted topologically

    def add_variable(self, name, domain):
        self.variables[name] = Variable(name, domain)

    def get_variable(self, name):
        return self.variables.get(name, None)
    #index in the cpt
    def compute_cpt_index(self, var_name, parent_values):
        var = self.get_variable(var_name)
        if not var.parents:
            return 0
        index = 0
        parent_vars = [self.get_variable(p) for p in var.parents]
        for i, parent in enumerate(var.parents):
            value = parent_values[i]
            value_index = self.get_variable(parent).domain.index(value)
            product = 1
            for j in range(i + 1, len(var.parents)):
                product *= len(parent_vars[j].domain)
            index += value_index * product
        return index
#get network from file
def parse_network(filename):
    network = BayesianNetwork()
    with open(filename, 'r') as f:
        # Remove blank lines and comment
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        ptr = 0

        # Variable Descriptor
        num_vars = int(lines[ptr])
        ptr += 1
        for _ in range(num_vars):
            parts = lines[ptr].split()
            name = parts[0]
            domain = parts[1:]
            network.add_variable(name, domain)
            ptr += 1

        #CPT Descriptor
        num_cpts = int(lines[ptr])
        ptr += 1
        for _ in range(num_cpts):
            # Skip extra blank lines
            while ptr < len(lines) and not lines[ptr]:
                ptr += 1
            if ptr >= len(lines):
                break

            header_line = lines[ptr]
            # If the header contains '|', split on it otherwise, split by whitespace
            if '|' in header_line:
                header = header_line.split('|')
                var_name = header[0].strip().rstrip(':')
                parents_str = header[1].strip()
                parents = parents_str.split() if parents_str else []
            else:
                tokens = header_line.split()
                var_name = tokens[0]
                parents = tokens[1:]
            ptr += 1

            var = network.get_variable(var_name)
            if var is None:
                raise ValueError(f"Variable '{var_name}' not defined in network.")
            var.parents = parents
            for p in parents:
                parent_var = network.get_variable(p)
                if parent_var is None:
                    raise ValueError(f"Variable '{p}' not defined in network.")
                parent_var.children.append(var_name)

            #number of lines for the CPT
            cpt = []
            num_parent_combos = 1
            for p in parents:
                num_parent_combos *= len(network.get_variable(p).domain)
            for _ in range(num_parent_combos):
                probs = list(map(float, lines[ptr].split()))
                domain = var.domain
                if len(probs) != len(domain):
                    raise ValueError(f"Number of probabilities ({len(probs)}) does not match domain size ({len(domain)}) for variable {var_name}.")
                dist = {domain[i]: probs[i] for i in range(len(domain))}
                cpt.append(dist)
                ptr += 1
            var.cpt = cpt

    # topological Sort
    order = []
    visited = set()
    def topological_sort(var_name):
        if var_name in visited:
            return
        var = network.get_variable(var_name)
        for parent in var.parents:
            topological_sort(parent)
        visited.add(var_name)
        order.append(var_name)
    for var_name in network.variables:
        if var_name not in visited:
            topological_sort(var_name)
    network.topological_order = order
    return network
#sample a value from a given probability distribution
def sample_from_distribution(dist):
    r = random.random()
    cumulative = 0.0
    for value, prob in dist.items():
        cumulative += prob
        if r <= cumulative:
            return value
    return list(dist.keys())[-1]## In case of rounding errors, return the last key

#exact inference by summing over joint probabilities
def exact_inference(network, query_var, evidence):
    query_domain = network.get_variable(query_var).domain
    non_evidence = [v for v in network.variables if v != query_var and v not in evidence]
    prob_dist = {val: 0.0 for val in query_domain}
    all_vars = non_evidence + [query_var]
    domains = [network.get_variable(var).domain for var in all_vars]
    total_joint = 0.0
    for assignment_values in itertools.product(*domains):
        current_assignment = {}
        for i, var in enumerate(all_vars):
            current_assignment[var] = assignment_values[i]
        current_assignment.update(evidence)
        joint = 1.0
        ## Multiply probabilities along the order
        for var_name in network.topological_order:
            var = network.get_variable(var_name)
            value = current_assignment.get(var_name, None)
            if value is None:
                continue
            parent_values = [current_assignment[p] for p in var.parents]
            cpt_index = network.compute_cpt_index(var.name, parent_values) if var.parents else 0
            dist = var.cpt[cpt_index]
            joint *= dist.get(value, 0.0)
        prob_dist[current_assignment[query_var]] += joint
        total_joint += joint
    if total_joint == 0:
        return None
    normalized = {k: v / total_joint for k, v in prob_dist.items()}
    return [normalized[val] for val in query_domain]
#rejection sampling for approximate inference
# def rejection_sampling(network, query_var, evidence, num_samples=5000):
#     query_domain = network.get_variable(query_var).domain
#     accepted = [] # samples that match the evidence
#     trials = 0
#     max_trials = 1000000  # avoid endless loop
#     while len(accepted) < num_samples and trials < max_trials:
#         sample = {}
#         for var_name in network.topological_order:
#             var = network.get_variable(var_name)
#             if var_name in evidence:
#                 sample[var_name] = evidence[var_name]
#             else:
#                 parent_values = [sample[p] for p in var.parents]
#                 cpt_index = network.compute_cpt_index(var.name, parent_values) if var.parents else 0
#                 dist = var.cpt[cpt_index]
#                 sample[var_name] = sample_from_distribution(dist)
#         trials += 1
#         # accept if it is consistent with the evidence
#         if all(sample.get(var) == evidence[var] for var in evidence):
#             accepted.append(sample)
#     counts = {val: 0 for val in query_domain}
#     for s in accepted:
#         counts[s[query_var]] += 1
#     total = len(accepted)
#     if total == 0:
#         return None
#     return [counts[val] / total for val in query_domain]
def rejection_sampling(network, query_var, evidence, num_samples=50000):
    query_domain = network.get_variable(query_var).domain
    accepted = []  # samples that match the evidence
    trials = 0
    max_trials = 1000000  # avoid endless loop

    while len(accepted) < num_samples and trials < max_trials:
        sample = {}
        consistent = True
        for var_name in network.topological_order:
            var = network.get_variable(var_name)
            # Compute parent values
            parent_values = [sample[p] for p in var.parents]
            cpt_index = network.compute_cpt_index(var.name, parent_values) if var.parents else 0
            # Sample from the distribution
            sample[var_name] = sample_from_distribution(var.cpt[cpt_index])
            # Early rejection: check evidence immediately
            if var_name in evidence and sample[var_name] != evidence[var_name]:
                consistent = False
                break  # Stop this trial early if evidence doesn't match
        trials += 1
        if consistent:
            accepted.append(sample)

    counts = {val: 0 for val in query_domain}
    for s in accepted:
        counts[s[query_var]] += 1

    total = len(accepted)
    if total == 0:
        return None
    return [counts[val] / total for val in query_domain]
# MCMC method for inference
def gibbs_sampling(network, query_var, evidence, num_samples=10000, burn_in=10000):
    query_domain = network.get_variable(query_var).domain
    state = evidence.copy() # Start with the evidence
    non_evidence = [var for var in network.variables if var not in evidence]
    # initialize non-evidence vars randomly
    for var in non_evidence:
        var_obj = network.get_variable(var)
        parent_values = [state[p] for p in var_obj.parents] if var_obj.parents else []
        idx = network.compute_cpt_index(var_obj.name, parent_values) if var_obj.parents else 0
        state[var] = sample_from_distribution(var_obj.cpt[idx])
     # Burn-in period to let the Markov chain stabilize
    for _ in range(burn_in):
        for var in non_evidence:
            var_obj = network.get_variable(var)
            parent_values = [state[p] for p in var_obj.parents] if var_obj.parents else []
            idx = network.compute_cpt_index(var_obj.name, parent_values) if var_obj.parents else 0
            prior_dist = var_obj.cpt[idx]
            likelihood = {}
             # likelihood based on the children
            for val in var_obj.domain:
                likelihood[val] = 1.0
                for child in var_obj.children:
                    child_obj = network.get_variable(child)
                    new_child_parents = []
                    for p in child_obj.parents:
                        if p == var:
                            new_child_parents.append(val)
                        else:
                            new_child_parents.append(state[p])
                    idx_child = network.compute_cpt_index(child, new_child_parents) if child_obj.parents else 0
                    likelihood[val] *= child_obj.cpt[idx_child].get(state[child], 0.0)
            unnormalized = {val: prior_dist.get(val, 0) * likelihood[val] for val in var_obj.domain}
            total = sum(unnormalized.values())
            if total == 0:
                conditional = {val: 1.0 / len(var_obj.domain) for val in var_obj.domain}
            else:
                conditional = {val: unnormalized[val] / total for val in var_obj.domain}
            state[var] = sample_from_distribution(conditional)
    samples = []
    # collect samples after burn-in
    for _ in range(num_samples):
        for var in non_evidence:
            var_obj = network.get_variable(var)
            parent_values = [state[p] for p in var_obj.parents] if var_obj.parents else []
            idx = network.compute_cpt_index(var_obj.name, parent_values) if var_obj.parents else 0
            prior_dist = var_obj.cpt[idx]
            likelihood = {}
            for val in var_obj.domain:
                likelihood[val] = 1.0
                for child in var_obj.children:
                    child_obj = network.get_variable(child)
                    new_child_parents = []
                    for p in child_obj.parents:
                        if p == var:
                            new_child_parents.append(val)
                        else:
                            new_child_parents.append(state[p])
                    idx_child = network.compute_cpt_index(child, new_child_parents) if child_obj.parents else 0
                    likelihood[val] *= child_obj.cpt[idx_child].get(state[child], 0.0)
            unnormalized = {val: prior_dist.get(val, 0) * likelihood[val] for val in var_obj.domain}
            total = sum(unnormalized.values())
            if total == 0:
                conditional = {val: 1.0 / len(var_obj.domain) for val in var_obj.domain}
            else:
                conditional = {val: unnormalized[val] / total for val in var_obj.domain}
            state[var] = sample_from_distribution(conditional)
        samples.append(state.copy())
    counts = {val: 0 for val in query_domain}
    #frequency of each query value in the collected samples
    for s in samples:
        counts[s[query_var]] += 1
    total_samples = len(samples)
    return [counts[val] / total_samples for val in query_domain]
def main():
    # check if input is interactive
    if sys.stdin.isatty():
        network = None
        while True:
            try:
                cmd = input().strip()  # No prompt text
            except EOFError:
                break
            if not cmd:
                continue
            if cmd.startswith("load"):
                try:
                    _, filename = cmd.split(maxsplit=1)
                    network = parse_network(filename)
                    # print(f"Loaded network from {filename}")
                except Exception as e:
                    print("Error parsing network:", e)
            elif cmd.startswith("xquery"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Invalid xquery command. Usage: xquery Q | X1=x1 ...")
                    continue
                query_var = parts[1]
                evidence = {}
                if '|' in parts:
                    idx = parts.index('|')
                    for part in parts[idx+1:]:
                        if '=' in part:
                            var, val = part.split('=')
                            evidence[var] = val
                if network is None:
                    print("No network loaded")
                    continue
                result = exact_inference(network, query_var, evidence)
                if result is None:
                    print("Invalid query or evidence")
                else:
                    print(" ".join(map(str, result)))
            elif cmd.startswith("rquery"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Invalid rquery command. Usage: rquery Q | X1=x1 ...")
                    continue
                query_var = parts[1]
                evidence = {}
                if '|' in parts:
                    idx = parts.index('|')
                    for part in parts[idx+1:]:
                        if '=' in part:
                            var, val = part.split('=')
                            evidence[var] = val
                if network is None:
                    print("No network loaded")
                    continue
                result = rejection_sampling(network, query_var, evidence)
                if result is None:
                    print("No samples accepted")
                else:
                    print(" ".join(map(str, result)))
            elif cmd.startswith("gquery"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Invalid gquery command. Usage: gquery Q | X1=x1 ...")
                    continue
                query_var = parts[1]
                evidence = {}
                if '|' in parts:
                    idx = parts.index('|')
                    for part in parts[idx+1:]:
                        if '=' in part:
                            var, val = part.split('=')
                            evidence[var] = val
                if network is None:
                    print("No network loaded")
                    continue
                result = gibbs_sampling(network, query_var, evidence)
                print(" ".join(map(str, result)))
            elif cmd == "quit":
                break
            else:
                print("Unknown command")
    else:
        # Non-interactive mode: read all lines from standard input.
        input_lines = sys.stdin.read().splitlines()
        network = None
        for line in input_lines:
            cmd = line.strip()
            if not cmd:
                continue
            if cmd.startswith("load"):
                try:
                    _, filename = cmd.split(maxsplit=1)
                    network = parse_network(filename)
                    # print(f"Loaded network from {filename}")
                except Exception as e:
                    print("Error parsing network:", e)
            elif cmd.startswith("xquery"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Invalid xquery command. Usage: xquery Q | X1=x1 ...")
                    continue
                query_var = parts[1]
                evidence = {}
                if '|' in parts:
                    idx = parts.index('|')
                    for part in parts[idx+1:]:
                        if '=' in part:
                            var, val = part.split('=')
                            evidence[var] = val
                if network is None:
                    print("No network loaded")
                    continue
                result = exact_inference(network, query_var, evidence)
                if result is None:
                    print("Invalid query or evidence")
                else:
                    print(" ".join(map(str, result)))
            elif cmd.startswith("rquery"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Invalid rquery command. Usage: rquery Q | X1=x1 ...")
                    continue
                query_var = parts[1]
                evidence = {}
                if '|' in parts:
                    idx = parts.index('|')
                    for part in parts[idx+1:]:
                        if '=' in part:
                            var, val = part.split('=')
                            evidence[var] = val
                if network is None:
                    print("No network loaded")
                    continue
                result = rejection_sampling(network, query_var, evidence)
                if result is None:
                    print("No samples accepted")
                else:
                    print(" ".join(map(str, result)))
            elif cmd.startswith("gquery"):
                parts = cmd.split()
                if len(parts) < 2:
                    print("Invalid gquery command. Usage: gquery Q | X1=x1 ...")
                    continue
                query_var = parts[1]
                evidence = {}
                if '|' in parts:
                    idx = parts.index('|')
                    for part in parts[idx+1:]:
                        if '=' in part:
                            var, val = part.split('=')
                            evidence[var] = val
                if network is None:
                    print("No network loaded")
                    continue
                result = gibbs_sampling(network, query_var, evidence)
                print(" ".join(map(str, result)))
            elif cmd == "quit":
                break
            else:
                print("Unknown command")

if __name__ == '__main__':
    main()

# import itertools
# import random
# import sys


# class Variable:
#     def __init__(self, name, domain):
#         self.name = name
#         self.domain = domain
#         self.parents = []
#         self.children = []
#         self.cpt = []

# class BayesianNetwork:
#     def __init__(self):
#         self.variables = {}
#         self.topological_order = []

#     def add_variable(self, name, domain):
#         self.variables[name] = Variable(name, domain)

#     def get_variable(self, name):
#         return self.variables.get(name, None)

#     def compute_cpt_index(self, var_name, parent_values):
#         var = self.get_variable(var_name)
#         if not var.parents:
#             return 0
#         index = 0
#         parent_vars = [self.get_variable(p) for p in var.parents]
#         for i, parent in enumerate(var.parents):
#             value = parent_values[i]
#             value_index = self.get_variable(parent).domain.index(value)
#             product = 1
#             for j in range(i + 1, len(var.parents)):
#                 product *= len(parent_vars[j].domain)
#             index += value_index * product
#         return index

# def parse_network(filename):
#     network = BayesianNetwork()
#     with open(filename, 'r') as f:
#         # Remove blank lines and comments.
#         lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
#         ptr = 0

#         # --- Variable Descriptor Segment ---
#         num_vars = int(lines[ptr])
#         ptr += 1
#         for _ in range(num_vars):
#             parts = lines[ptr].split()
#             name = parts[0]
#             domain = parts[1:]
#             network.add_variable(name, domain)
#             ptr += 1

#         # --- CPT Descriptor Segment ---
#         num_cpts = int(lines[ptr])
#         ptr += 1
#         for _ in range(num_cpts):
#             # Skip any extra blank lines.
#             while ptr < len(lines) and not lines[ptr]:
#                 ptr += 1
#             if ptr >= len(lines):
#                 break

#             header_line = lines[ptr]
#             # If the header contains '|', split on it; otherwise, split by whitespace.
#             if '|' in header_line:
#                 header = header_line.split('|')
#                 var_name = header[0].strip().rstrip(':')
#                 parents_str = header[1].strip()
#                 parents = parents_str.split() if parents_str else []
#             else:
#                 tokens = header_line.split()
#                 var_name = tokens[0]
#                 parents = tokens[1:]
#             ptr += 1

#             var = network.get_variable(var_name)
#             if var is None:
#                 raise ValueError(f"Variable '{var_name}' not defined in network.")
#             var.parents = parents
#             for p in parents:
#                 parent_var = network.get_variable(p)
#                 if parent_var is None:
#                     raise ValueError(f"Variable '{p}' not defined in network.")
#                 parent_var.children.append(var_name)

#             # Determine number of lines for the CPT.
#             cpt = []
#             num_parent_combos = 1
#             for p in parents:
#                 num_parent_combos *= len(network.get_variable(p).domain)
#             for _ in range(num_parent_combos):
#                 probs = list(map(float, lines[ptr].split()))
#                 domain = var.domain
#                 if len(probs) != len(domain):
#                     raise ValueError(f"Number of probabilities ({len(probs)}) does not match domain size ({len(domain)}) for variable {var_name}.")
#                 dist = {domain[i]: probs[i] for i in range(len(domain))}
#                 cpt.append(dist)
#                 ptr += 1
#             var.cpt = cpt

#     # --- Topological Sorting ---
#     order = []
#     visited = set()
#     def topological_sort(var_name):
#         if var_name in visited:
#             return
#         var = network.get_variable(var_name)
#         for parent in var.parents:
#             topological_sort(parent)
#         visited.add(var_name)
#         order.append(var_name)
#     for var_name in network.variables:
#         if var_name not in visited:
#             topological_sort(var_name)
#     network.topological_order = order
#     return network

# def sample_from_distribution(dist):
#     r = random.random()
#     cumulative = 0.0
#     for value, prob in dist.items():
#         cumulative += prob
#         if r <= cumulative:
#             return value
#     return list(dist.keys())[-1]

# def exact_inference(network, query_var, evidence):
#     query_domain = network.get_variable(query_var).domain
#     # Build list of variables to enumerate over (non-evidence plus the query variable).
#     non_evidence = [v for v in network.variables if v != query_var and v not in evidence]
#     prob_dist = {val: 0.0 for val in query_domain}
#     all_vars = non_evidence + [query_var]
#     domains = [network.get_variable(var).domain for var in all_vars]
#     total_joint = 0.0
#     for assignment_values in itertools.product(*domains):
#         current_assignment = {}
#         for i, var in enumerate(all_vars):
#             current_assignment[var] = assignment_values[i]
#         # Merge in evidence.
#         current_assignment.update(evidence)
#         joint = 1.0
#         for var_name in network.topological_order:
#             var = network.get_variable(var_name)
#             value = current_assignment.get(var_name, None)
#             if value is None:
#                 continue
#             parent_values = [current_assignment[p] for p in var.parents]
#             cpt_index = network.compute_cpt_index(var.name, parent_values) if var.parents else 0
#             dist = var.cpt[cpt_index]
#             joint *= dist.get(value, 0.0)
#         prob_dist[current_assignment[query_var]] += joint
#         total_joint += joint
#     if total_joint == 0:
#         return None
#     normalized = {k: v / total_joint for k, v in prob_dist.items()}
#     return [normalized[val] for val in query_domain]

# def rejection_sampling(network, query_var, evidence, num_samples=10000):
#     query_domain = network.get_variable(query_var).domain
#     accepted = []
#     trials = 0
#     max_trials = 1000000
#     while len(accepted) < num_samples and trials < max_trials:
#         sample = {}
#         for var_name in network.topological_order:
#             var = network.get_variable(var_name)
#             if var_name in evidence:
#                 sample[var_name] = evidence[var_name]
#             else:
#                 parent_values = [sample[p] for p in var.parents]
#                 cpt_index = network.compute_cpt_index(var.name, parent_values) if var.parents else 0
#                 dist = var.cpt[cpt_index]
#                 sample[var_name] = sample_from_distribution(dist)
#         trials += 1
#         if all(sample.get(var) == evidence[var] for var in evidence):
#             accepted.append(sample)
#     counts = {val: 0 for val in query_domain}
#     for s in accepted:
#         counts[s[query_var]] += 1
#     total = len(accepted)
#     if total == 0:
#         return None
#     return [counts[val] / total for val in query_domain]

# def gibbs_sampling(network, query_var, evidence, num_samples=10000, burn_in=10000):
#     """
#     Perform Gibbs sampling with a full sweep update (in fixed order) over non-evidence variables.
#     """
#     query_domain = network.get_variable(query_var).domain
#     state = evidence.copy()
#     non_evidence = [var for var in network.variables if var not in evidence]
    
#     # Initialization: assign an initial value for each non-evidence variable.
#     for var in non_evidence:
#         var_obj = network.get_variable(var)
#         parent_values = [state[p] for p in var_obj.parents] if var_obj.parents else []
#         idx = network.compute_cpt_index(var_obj.name, parent_values) if var_obj.parents else 0
#         state[var] = sample_from_distribution(var_obj.cpt[idx])
    
#     # Burn-in period: perform full sweeps over all non-evidence variables in fixed order.
#     for _ in range(burn_in):
#         for var in non_evidence:
#             var_obj = network.get_variable(var)
#             parent_values = [state[p] for p in var_obj.parents] if var_obj.parents else []
#             idx = network.compute_cpt_index(var_obj.name, parent_values) if var_obj.parents else 0
#             prior_dist = var_obj.cpt[idx]
#             likelihood = {}
#             for val in var_obj.domain:
#                 likelihood[val] = 1.0
#                 for child in var_obj.children:
#                     child_obj = network.get_variable(child)
#                     new_child_parents = []
#                     for p in child_obj.parents:
#                         if p == var:
#                             new_child_parents.append(val)
#                         else:
#                             new_child_parents.append(state[p])
#                     idx_child = network.compute_cpt_index(child, new_child_parents) if child_obj.parents else 0
#                     likelihood[val] *= child_obj.cpt[idx_child].get(state[child], 0.0)
#             unnormalized = {val: prior_dist.get(val, 0) * likelihood[val] for val in var_obj.domain}
#             total = sum(unnormalized.values())
#             if total == 0:
#                 conditional = {val: 1.0 / len(var_obj.domain) for val in var_obj.domain}
#             else:
#                 conditional = {val: unnormalized[val] / total for val in var_obj.domain}
#             state[var] = sample_from_distribution(conditional)
    
#     # Sampling period: after each full sweep, record a sample.
#     samples = []
#     for _ in range(num_samples):
#         for var in non_evidence:
#             var_obj = network.get_variable(var)
#             parent_values = [state[p] for p in var_obj.parents] if var_obj.parents else []
#             idx = network.compute_cpt_index(var_obj.name, parent_values) if var_obj.parents else 0
#             prior_dist = var_obj.cpt[idx]
#             likelihood = {}
#             for val in var_obj.domain:
#                 likelihood[val] = 1.0
#                 for child in var_obj.children:
#                     child_obj = network.get_variable(child)
#                     new_child_parents = []
#                     for p in child_obj.parents:
#                         if p == var:
#                             new_child_parents.append(val)
#                         else:
#                             new_child_parents.append(state[p])
#                     idx_child = network.compute_cpt_index(child, new_child_parents) if child_obj.parents else 0
#                     likelihood[val] *= child_obj.cpt[idx_child].get(state[child], 0.0)
#             unnormalized = {val: prior_dist.get(val, 0) * likelihood[val] for val in var_obj.domain}
#             total = sum(unnormalized.values())
#             if total == 0:
#                 conditional = {val: 1.0 / len(var_obj.domain) for val in var_obj.domain}
#             else:
#                 conditional = {val: unnormalized[val] / total for val in var_obj.domain}
#             state[var] = sample_from_distribution(conditional)
#         samples.append(state.copy())
    
#     counts = {val: 0 for val in query_domain}
#     for s in samples:
#         counts[s[query_var]] += 1
#     total_samples = len(samples)
#     return [counts[val] / total_samples for val in query_domain]

# def main():
#     network = None
#     while True:
#         # cmd = input("Enter command: ").strip()
#         cmd=sys.stdin.read().splitlines()
#         if not cmd:
#             continue
#         if cmd.startswith("load"):
#             _, filename = cmd.split(maxsplit=1)
#             network = parse_network(filename)
#             # print(f"Loaded network from {filename}")
#         elif cmd.startswith("xquery"):
#             parts = cmd.split()
#             query_var = parts[1]
#             evidence = {}
#             if '|' in parts:
#                 idx = parts.index('|')
#                 for part in parts[idx+1:]:
#                     if '=' in part:
#                         var, val = part.split('=')
#                         evidence[var] = val
#             if network is None:
#                 print("No network loaded")
#                 continue
#             result = exact_inference(network, query_var, evidence)
#             if result is None:
#                 print("Invalid query or evidence")
#             else:
#                 print(" ".join(map(str, result)))
#         elif cmd.startswith("rquery"):
#             parts = cmd.split()
#             query_var = parts[1]
#             evidence = {}
#             if '|' in parts:
#                 idx = parts.index('|')
#                 for part in parts[idx+1:]:
#                     if '=' in part:
#                         var, val = part.split('=')
#                         evidence[var] = val
#             if network is None:
#                 print("No network loaded")
#                 continue
#             result = rejection_sampling(network, query_var, evidence)
#             if result is None:
#                 print("No samples accepted")
#             else:
#                 print(" ".join(map(str, result)))
#         elif cmd.startswith("gquery"):
#             parts = cmd.split()
#             query_var = parts[1]
#             evidence = {}
#             if '|' in parts:
#                 idx = parts.index('|')
#                 for part in parts[idx+1:]:
#                     if '=' in part:
#                         var, val = part.split('=')
#                         evidence[var] = val
#             if network is None:
#                 print("No network loaded")
#                 continue
#             result = gibbs_sampling(network, query_var, evidence)
#             print(" ".join(map(str, result)))
#         elif cmd == "quit":
#             break
#         else:
#             print("Unknown command")

# if __name__ == "__main__":
#     main()
