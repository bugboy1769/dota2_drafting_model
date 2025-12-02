"""
Standalone embedding visualization script.
Runs without Streamlit to isolate Plotly/UMAP issues.
Saves an interactive HTML file you can open in your browser.
"""

import torch
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import umap.umap_ as umap
from sklearn.decomposition import PCA
import sys

# Add src to path
sys.path.append(".")
from src.model import DraftModel

# Configuration
MODEL_PATH = "models/best_model.pt"
HEROES_PATH = "data/raw/heroes.json"
OUTPUT_HTML = "embedding_visualization.html"

def load_model_and_data():
    """Load model and hero metadata."""
    print("Loading hero data...")
    with open(HEROES_PATH, 'r') as f:
        heroes_list = json.load(f)
    
    hero_map = {}
    for h in heroes_list:
        hero_map[h['id']] = {
            'name': h['localized_name'],
            'attr': h['primary_attr'],
            'roles': ", ".join(h['roles'])
        }
    
    print("Loading model...")
    model = DraftModel(num_heroes=150, embedding_dim=128)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, hero_map

def get_embeddings(model):
    """Extract hero embeddings."""
    weights = model.hero_embedding.weight.detach().numpy()
    return weights[1:]  # Skip padding token

def create_visualization(algorithm='umap'):
    """Create the 3D visualization."""
    
    # 1. Load everything
    model, hero_map = load_model_and_data()
    
    # 2. Get vectors
    print("Extracting embeddings...")
    vectors = get_embeddings(model)
    print(f"Vector shape: {vectors.shape}")
    print(f"Min: {vectors.min()}, Max: {vectors.max()}")
    
    # 3. Prepare metadata
    print("Preparing metadata...")
    plot_data = []
    for i in range(len(vectors)):
        h_id = i + 1
        if h_id in hero_map:
            info = hero_map[h_id]
            plot_data.append({
                'name': info['name'],
                'attr': info['attr'],
                'roles': info['roles'],
                'id': h_id
            })
        else:
            plot_data.append({
                'name': f"Unknown-{h_id}",
                'attr': 'unknown',
                'roles': ''
            })
    
    df = pd.DataFrame(plot_data)
    print(f"DataFrame shape: {df.shape}")
    
    # 4. Dimensionality reduction
    print(f"Running {algorithm.upper()}...")
    if algorithm == 'pca':
        reducer = PCA(n_components=3)
        projections = reducer.fit_transform(vectors)
    else:  # UMAP
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine'
        )
        projections = reducer.fit_transform(vectors)
    
    print(f"Projections shape: {projections.shape}")
    print(f"Projections range: [{projections.min()}, {projections.max()}]")
    
    # 5. Add coordinates to DataFrame
    df['x'] = projections[:, 0]
    df['y'] = projections[:, 1]
    df['z'] = projections[:, 2]
    
    print("\nFirst 5 heroes:")
    print(df[['name', 'attr', 'x', 'y', 'z']].head())
    
    # 6. Create plot
    print("\nCreating Plotly figure...")
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='attr',
        hover_name='name',
        hover_data=['roles'],
        title=f"Hero Embeddings ({algorithm.upper()})",
        height=800,
        template='plotly_dark'
    )
    
    # Enhance markers
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    
    # Better labels
    fig.update_layout(
        scene=dict(
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            zaxis_title='Component 3'
        )
    )
    
    # 7. Save to HTML
    print(f"\nSaving to {OUTPUT_HTML}...")
    fig.write_html(OUTPUT_HTML)
    print(f"✅ Done! Open '{OUTPUT_HTML}' in your browser to view.")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize hero embeddings')
    parser.add_argument('--algo', choices=['umap', 'pca'], default='umap',
                        help='Dimensionality reduction algorithm')
    args = parser.parse_args()
    
    create_visualization(algorithm=args.algo)
