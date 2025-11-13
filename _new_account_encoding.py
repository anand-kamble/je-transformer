#%%
import json
import torch
import torch.nn as nn
from geoopt import ManifoldParameter
from geoopt.manifolds import Lorentz
from geoopt.optim import RiemannianAdam
import numpy as np

#%%
accounts_data = None
with open('out_small_ingest/20251110-223021/artifacts/accounts_20251110-223029.json', 'r', encoding='utf-8') as f:
    accounts_data = json.load(f)

    
    
#%%

class Account:
    def __init__(self, number, name, nature, parent_number):
        self.number = number
        self.name = name
        self.nature = nature
        self.parent_number = parent_number
        self.children = []
        
    def __repr__(self):
        return f"Account(number={self.number}, name={self.name}, nature={self.nature}, parent_number={self.parent_number})"

#%%
# Build a tree of Account objects according to the split number hierarchy
accounts_dict = {}  # number -> Account instance
root_accounts = []

for account in accounts_data["accounts"]:
    segments = account['number'].split('-')
    parent = None
    number_path = []
    for level, segment in enumerate(segments):
        number_path.append(segment)
        account_num = '-'.join(number_path)
        if account_num not in accounts_dict:
            # Create node (use current name/nature for the leaf, else placeholders)
            if level == len(segments) - 1:
                name = account['name']
                nature = account['nature']
            else:
                # Use generic name/nature for intermediate (parent) accounts; can be refined if info available
                name = f'Level {level + 1} of {account_num}'
                nature = None
            parent_number = '-'.join(number_path[:-1]) if number_path[:-1] else None
            node = Account(account_num, name, nature, parent_number)
            accounts_dict[account_num] = node
            if parent_number:
                parent_node = accounts_dict[parent_number]
                parent_node.children.append(node)
            else:
                root_accounts.append(node)
        parent = accounts_dict[account_num]


#%%
# Build a list of (parent_number, child_number) tuples representing edges in the account hierarchy.
def collect_edges(nodes):
    edges = []
    for node in nodes:
        for child in node.children:
            edges.append((node.number, child.number))
            edges.extend(collect_edges([child]))
    return edges

account_edges = collect_edges(root_accounts)
 


# Build a mapping from account number strings to integer indices.
account_number_to_idx = {}
idx_to_account_number = {}

for idx, acct_num in enumerate(sorted(accounts_dict.keys())):
    account_number_to_idx[acct_num] = idx
    idx_to_account_number[idx] = acct_num

#%%

# Update account_edges to use integer indices rather than strings
account_edges = [
    (account_number_to_idx[parent], account_number_to_idx[child])
    for parent, child in account_edges
]
account_edges

#%%
# Build set of all nodes dynamically from account_edges
all_nodes = set()
for edge in account_edges:
    all_nodes.update(edge)

# Build set of positive pairs (directed and undirected if needed)
positive_set = set()
for parent, child in account_edges:
    positive_set.add((parent, child))
    positive_set.add((child, parent))  # treat as undirected for strict negative mining

# Build negative pairs: sample one negative per edge (negative set matches len(account_edges))
import random

all_nodes_list = sorted(list(all_nodes))  # deterministic ordering for reproducibility
negative_set = set()
num_negatives_needed = len(account_edges)
attempts = 0
max_attempts = num_negatives_needed * 20  # Safety to avoid infinite loop

while len(negative_set) < num_negatives_needed and attempts < max_attempts:
    node = random.choice(all_nodes_list)
    possible_negatives = [
        other for other in all_nodes_list
        if other != node and (node, other) not in positive_set and (node, other) not in negative_set
    ]
    if possible_negatives:
        negative = random.choice(possible_negatives)
        negative_set.add((node, negative))
    attempts += 1
if len(negative_set) < num_negatives_needed:
    print(f"Warning: Only generated {len(negative_set)} negatives, requested {num_negatives_needed}.")


len(negative_set)
#%%


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Use double precision for Lorentz manifold (recommended)
torch.set_default_dtype(torch.float64)


class LorentzEmbedding(nn.Module):
    """
    Lorentz embeddings for hierarchical data.
    
    Based on "Learning Continuous Hierarchies in the Lorentz Model 
    of Hyperbolic Geometry" (Nickel & Kiela, 2018)
    """
    def __init__(self, num_nodes, embedding_dim, curvature=1.0):
        """
        Args:
            num_nodes: Number of nodes/entities to embed
            embedding_dim: Dimension of embeddings (ambient dimension)
            curvature: Negative curvature of the hyperbolic space (k)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Initialize Lorentz manifold with negative curvature
        self.manifold = Lorentz(k=curvature)
        
        # Initialize embeddings on the manifold
        # Using random normal initialization from the origin
        embeddings = self.manifold.random_normal(
            num_nodes, embedding_dim, std=0.1
        )
        
        # Create ManifoldParameter for Riemannian optimization
        self.embeddings = ManifoldParameter(embeddings, manifold=self.manifold)
    
    def forward(self, indices):
        """Get embeddings for given indices."""
        return self.embeddings[indices]
    
    def distance(self, idx1, idx2):
        """Compute Lorentzian distance between embeddings."""
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]
        return self.manifold.dist(emb1, emb2)


def lorentz_loss(embeddings, pos_pairs, neg_pairs, manifold, margin=0.1):
    """
    Ranking loss for hierarchical structure learning.
    
    Args:
        embeddings: Embedding tensor
        pos_pairs: Positive pairs (parent-child relationships)
        neg_pairs: Negative pairs (non-hierarchical relationships)
        manifold: Lorentz manifold
        margin: Margin for ranking loss
    """
    # Positive distances (should be small)
    pos_u = embeddings[pos_pairs[:, 0]]
    pos_v = embeddings[pos_pairs[:, 1]]
    pos_dist = manifold.dist(pos_u, pos_v)
    
    # Negative distances (should be large)
    neg_u = embeddings[neg_pairs[:, 0]]
    neg_v = embeddings[neg_pairs[:, 1]]
    neg_dist = manifold.dist(neg_u, neg_v)
    
    # Ranking loss: max(0, pos_dist - neg_dist + margin)
    loss = torch.relu(pos_dist - neg_dist + margin).mean()
    
    return loss


def generate_toy_hierarchy():
    """
    Generate toy hierarchical data (tree structure).
    Returns positive pairs (edges in tree) and negative pairs.
    """
    # Simple tree: 0 -> {1, 2}, 1 -> {3, 4}, 2 -> {5, 6}
    edges = account_edges
    
    pos_pairs = torch.tensor(edges, dtype=torch.long)
    
    # Generate negative pairs (non-adjacent nodes)
    
    # Optionally subsample for balance
    # E.g., neg_pairs = neg_pairs[:len(edges) * 2]
    
    neg_pairs = list(negative_set)
    
    neg_pairs = torch.tensor(neg_pairs, dtype=torch.long)
    
    return pos_pairs, neg_pairs


def train_lorentz_embeddings():
    """Train Lorentz embeddings on hierarchical data."""
    
    # Hyperparameters
    num_nodes = len(accounts_dict)
    embedding_dim = 5  # Ambient dimension (n+1 for n-dimensional hyperbolic)
    curvature = 1.0
    learning_rate = 0.01
    num_epochs = 500
    
    # Generate toy hierarchy
    pos_pairs, neg_pairs = generate_toy_hierarchy()
    
    # Initialize model
    model = LorentzEmbedding(num_nodes, embedding_dim, curvature)
    
    # Use Riemannian Adam optimizer from geoopt
    optimizer = RiemannianAdam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Training Lorentz embeddings...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Compute loss
        loss = lorentz_loss(
            model.embeddings, 
            pos_pairs, 
            neg_pairs, 
            model.manifold
        )
        
        # Backward pass
        loss.backward()
        
        # Riemannian optimization step
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Evaluate final embeddings
    print("\n=== Final Results ===")
    print(f"Final Loss: {loss.item():.4f}")
    
    # Check some distances
    print("\n=== Hierarchical Distances ===")
    with torch.no_grad():
        # Parent-child distances (should be small)
        print("Parent-Child distances:")
        for parent, child in pos_pairs[:3]:
            dist = model.distance(parent, child)
            print(f"  Node {parent} -> Node {child}: {dist.item():.4f}")
        
        # Cross-subtree distances (should be large)
        print("\nCross-subtree distances:")
        for n1, n2 in neg_pairs[:3]:
            dist = model.distance(n1, n2)
            print(f"  Node {n1} <-> Node {n2}: {dist.item():.4f}")
    
    return model


def visualize_embedding_norms(model):
    """Analyze the structure of learned embeddings."""
    print("\n=== Embedding Analysis ===")
    with torch.no_grad():
        embeddings = model.embeddings
        
        # Distance from origin (depth in hierarchy)
        print("Distances from origin (hierarchy depth):")
        for i in range(model.num_nodes):
            dist_origin = model.manifold.dist0(embeddings[i:i+1])
            print(f"  Node {i}: {dist_origin.item():.4f}")
        
        # Verify Lorentzian constraint: -x₀² + x₁² + ... + xₙ² = -k
        print("\nLorentzian constraints (should be close to -1.0):")
        for i in range(min(3, model.num_nodes)):
            x = embeddings[i]
            constraint = -x[0]**2 + torch.sum(x[1:]**2)
            print(f"  Node {i}: {constraint.item():.4f}")


# Advanced example with custom loss
class HierarchicalLorentzModel(nn.Module):
    """
    More sophisticated model with learnable curvature.
    """
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        
        # Learnable curvature
        self.manifold = Lorentz(k=1.0, learnable=True)
        
        # Initialize embeddings
        embeddings = self.manifold.random_normal(
            num_nodes, embedding_dim, std=0.1
        )
        self.embeddings = ManifoldParameter(embeddings, manifold=self.manifold)
    
    def entailment_cone_loss(self, parent_idx, child_idx):
        """
        Entailment cone loss for hierarchy.
        Children should be in the future light cone of parents.
        """
        parent = self.embeddings[parent_idx]
        child = self.embeddings[child_idx]
        
        # Project onto Lorentz manifold and compute distance
        dist = self.manifold.dist(parent, child)
        
        return dist


# Run basic training
model = train_lorentz_embeddings()

# Analyze embeddings
visualize_embedding_norms(model)

print("\n=== Training Complete ===")
print("The Lorentz embeddings capture hierarchical structure where:")
print("- Distance from origin represents depth in hierarchy")
print("- Parent-child pairs have smaller distances")
print("- Non-related nodes have larger distances")

# %%

def test_account_proximity(account_number_str, model, accounts_dict, account_number_to_idx, idx_to_account_number, topk=3):
    """
    Given an account number (as string), print:
      - its parent (if any) and distance
      - its children (if any) and distances
      - the closest other account numbers in embedding space (excluding itself)
      
    Args:
        account_number_str (str): Account number, e.g. '101-01'
        model: LorentzEmbedding instance (trained)
        accounts_dict: mapping of account number str -> Account object
        account_number_to_idx: mapping str -> int index
        idx_to_account_number: mapping int -> str
        topk: number of closest accounts to show

    Returns:
        None (prints information)
    """
    import torch

    if account_number_str not in account_number_to_idx:
        print(f"Account number '{account_number_str}' not found.")
        return

    idx = account_number_to_idx[account_number_str]
    embedding = model.embeddings[idx].detach()

    print(f"\n==== Analysis for Account '{account_number_str}' (idx={idx}) ====")
    account_obj = accounts_dict[account_number_str]

    # Parent
    if account_obj.parent_number is not None:
        parent_str = account_obj.parent_number
        parent_idx = account_number_to_idx.get(parent_str)
        if parent_idx is not None:
            parent_dist = model.distance(idx, parent_idx).item()
            print(f"Parent: {parent_str} (idx={parent_idx}), Distance: {parent_dist:.4f}")
        else:
            print("Parent not found in mapping.")
    else:
        print("No parent (is root node).")

    # Children
    if account_obj.children:
        print(f"Children ({len(account_obj.children)}):")
        for child in account_obj.children[:topk]:
            child_str = child.number
            child_idx = account_number_to_idx.get(child_str)
            if child_idx is not None:
                child_dist = model.distance(idx, child_idx).item()
                print(f"  {child_str} (idx={child_idx}), Distance: {child_dist:.4f}")
            else:
                print(f"  Child {child_str} not found in mapping.")
        if len(account_obj.children) > topk:
            print(f"  ... ({len(account_obj.children) - topk} more not shown)")
    else:
        print("No children.")

    # Find closest other accounts in embedding space (excluding self)
    all_indices = [i for i in range(model.num_nodes) if i != idx]
    distances = []
    for i in all_indices:
        d = model.distance(idx, i).item()
        distances.append((d, i))
    distances.sort()
    print(f"\nClosest {topk} accounts by embedding distance:")
    for d, i in distances[:topk]:
        print(f"  {idx_to_account_number[i]} (idx={i}) - Distance: {d:.4f}")

    return {
        "account_number": account_number_str,
        "parent": account_obj.parent_number,
        "children": [child.number for child in account_obj.children],
        "closest": [idx_to_account_number[i] for _, i in distances[:topk]]
    }

test_account_proximity('105', model, accounts_dict, account_number_to_idx, idx_to_account_number, topk=3)
