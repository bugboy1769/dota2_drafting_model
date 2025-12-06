import streamlit as st
import torch
import json
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import umap.umap_ as umap
from sklearn.decomposition import PCA

# -- Config -- #
st.set_page_config(page_title="DOTA2 AI Drafter", layout="wide")
MODEL_PATH="models/best_model.pt"
HEROES_PATH="data/raw/heroes.json"

# -- Helper Functions -- #
def load_model_and_data():
    """Loads the model and hero data once and caches it"""

    #1. Load Hero Data
    with open(HEROES_PATH, 'r') as f:
        heroes_list=json.load(f)
    
    #Create a mapping: ID -> Name, ID -> Attributes
    hero_map={}
    for h in heroes_list:
        hero_map[h['id']]={
            'name':h['localized_name'],
            'attr':h['primary_attr'],
            'roles':",".join(h['roles'])
        }
    
    #2. Load Model
    import sys
    sys.path.append('src')
    from src.model import DraftModel

    #Initialise Model
    model=DraftModel(num_heroes=150, embedding_dim=128)

    try:
        checkpoint=torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None
    
    return model, hero_map

def get_embeddings(model):
    """Get embedding weights for all heroes from the model"""
    weights=model.hero_embedding.weight.detach().numpy()

    return weights[1:]

# -- Main App -- #
st.title("DOTA2 AI Neural Internals")

model, hero_map=load_model_and_data()
if model and hero_map:
    st.sidebar.success("Model and data loaded successfully")
    tab1, tab2, tab3=st.tabs(["Embedding Visualisation", "Draft Assistant", "Draft Analysis"])
    with tab1:
        st.header("Hero Embedding Landscape")
        st.markdown("""
        This Visualisation flattends the model's internal 128 dimensional representation into 3D.
        * **Closer Points**: The model thinks these heroes are similar.
        * **Clusters**: Roles or Archetypes (e.g., "Hard Carries, "Save Heroes)
        """)
        
        #Controls
        col1, col2=st.columns(2)
        with col1:
            algo=st.selectbox("Algorithm", ["UMAP (Recommended)", "PCA (Linear)"])
        with col2:
            color_by=st.selectbox("Color By", ["Primary Attribute", "Role (Carry/Support)"])
        
        if st.button("Generate Visualisation"):
            with st.spinner("Projecting Dimensions .."):
                #1. Get Vectors
                vectors=get_embeddings(model)
                
                #2. Prepare metadata for Plotting
                plot_data=[]
                for i in range(len(vectors)):
                    h_id=i+1 #HeroIDs are 1-indexed
                    if h_id in hero_map:
                        info=hero_map[h_id]
                        plot_data.append({
                            'name': info['name'],
                            'attr': info['attr'],
                            'roles': info['roles'],
                            'id': h_id,
                        })
                    else:
                        #Placeholder for missing IDs (if any)
                        plot_data.append({
                            'name': f"Unknown-{h_id}",
                            'attr': 'Unknown',
                            'roles': 'Unknown',
                        })
                df=pd.DataFrame(plot_data)
                
                #3. Dimensionality Reduction
                try:
                    if algo == "UMAP (Recommended)":
                        reducer = umap.UMAP(
                            n_components=3,
                            n_neighbors=3,
                            min_dist=0.1,
                            metric='cosine',
                            random_state=42
                        )
                        projections = reducer.fit_transform(vectors)
                    else: # PCA (Linear)
                        reducer = PCA(n_components=3)
                        projections = reducer.fit_transform(vectors)
                except Exception as e:
                    st.error(f"Dimensionality Reduction Failed: {e}")
                    st.info("Tip: If UMAP failed, try selecting PCA. UMAP can have issues on M1 Macs with Numba.")
                    st.stop()
                
                #Add coords to DF
                df['x']=projections[:, 0]
                df['y']=projections[:, 1]
                df['z']=projections[:, 2]

                #4. Plot
                try:
                    fig=px.scatter_3d(
                        df, x='x', y='y', z='z',
                        color='attr',
                        hover_name='name',
                        hover_data=['roles'],
                        title=f"Hero Embeddings ({algo})",
                        height=700,
                        template='plotly_dark'  # Try different template
                    )
                    
                    # Update marker size and opacity for better visibility
                    fig.update_traces(marker=dict(size=8, opacity=0.8))
                    
                    # Update layout for better 3D visualization
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='Component 1',
                            yaxis_title='Component 2',
                            zaxis_title='Component 3'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Plotting Error: {e}")
                    st.write("Try switching to PCA if UMAP isn't working.")
    with tab2:
        st.write("Coming Up: Draft Assistant Interface")

    with tab3:
        st.header("Draft Probability Space Analysis")
        st.markdown("""
        Visualize the **Draft Trajectory** through the model's 128D state space.
        * **Win Probability**: The "elevation" of the current state.
        * **Entropy**: The "uncertainty" (fog) at the current state.
        """)
        
        from src.draft_analyzer import DraftAnalyzer, load_sample_draft
        analyzer = DraftAnalyzer(model)
        
        # UI: Draft Builder
        if 'draft_seq' not in st.session_state:
            st.session_state.draft_seq = load_sample_draft()
            
        if st.button("Analyze Sample Draft"):
            # Run Analysis
            results = analyzer.analyze_draft_sequence(st.session_state.draft_seq)
            key_moments = analyzer.find_turning_points(
                results['win_probs'], results['entropies'], results['turns']
            )
            
            # Plot 1: Win Probability
            st.subheader("1. Win Probability Trajectory")
            import plotly.graph_objects as go
            fig_win = go.Figure()
            fig_win.add_trace(go.Scatter(
                x=results['turns'], y=results['win_probs'],
                mode='lines+markers', name='Win %',
                line=dict(color='#00ff00', width=3)
            ))
            fig_win.add_hline(y=0.5, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_win, use_container_width=True)
            
            # Plot 2: Entropy
            st.subheader("2. Certainty (Entropy) Evolution")
            fig_ent = go.Figure()
            fig_ent.add_trace(go.Scatter(
                x=results['turns'], y=results['entropies'],
                mode='lines+markers', name='Entropy',
                line=dict(color='#ff00ff', width=3),
                fill='tozeroy'
            ))
            st.plotly_chart(fig_ent, use_container_width=True)
            
            # Key Moments
            st.subheader("3. Key Moments")
            st.json(key_moments)

                