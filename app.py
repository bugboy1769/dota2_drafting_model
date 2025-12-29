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
    import yaml
    sys.path.append('src')
    from src.model import DraftModel

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    emb_dim = config['model']['embedding_dim']

    #Initialise Model
    model=DraftModel(num_heroes=150, embedding_dim=emb_dim)

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
        This Visualisation flattends the model's internal 256 dimensional representation into 3D.
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
                    color_map = {'str': 'red', 'agi': 'green', 'int': 'blue', 'all':'white', 'unknown':'yellow'}
                    fig=px.scatter_3d(
                        df, x='x', y='y', z='z',
                        color='attr',
                        color_discrete_map=color_map,
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
        st.header("Draft Assistant")
        
        # Initialize Assistant
        from src.draft_assistant import DraftAssistant
        assistant = DraftAssistant(model, hero_map, device='cpu')
        
        # Session State for Live Draft
        if 'live_draft' not in st.session_state:
            st.session_state.live_draft = []
        if 'win_prob_history' not in st.session_state:
            st.session_state.win_prob_history = []
        
        # Validate Session State (Fix for corrupted state)
        if st.session_state.live_draft and not isinstance(st.session_state.live_draft[0], int):
            try:
                # Try to convert if they are string digits
                st.session_state.live_draft = [int(x) for x in st.session_state.live_draft]
            except:
                st.warning("Detected corrupted draft state. Resetting...")
                st.session_state.live_draft = []
                st.session_state.win_prob_history = []
            
        # --- Layout ---
        
        # Top Bar: Controls
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.subheader("Current Turn")
            turn_idx = len(st.session_state.live_draft)
            if turn_idx < 24:
                team, action = assistant.get_turn_info(turn_idx)
                color = "green" if team == "Radiant" else "red"
                st.markdown(f"### :{color}[{team} {action}]")
            else:
                st.success("Draft Complete!")
                
        with col2:
            st.subheader("Thinking Mode")
            ruthless_mode = st.checkbox("Enable MCTS", value=False, help="Runs 50 internal simulations per pick to find the best outcome. Slower but smarter.")

        with col3:
            if st.button("Reset Draft"):
                st.session_state.live_draft = []
                st.session_state.win_prob_history = []
                st.rerun()
            if st.button("Undo Last"):
                if st.session_state.live_draft:
                    st.session_state.live_draft.pop()
                    if st.session_state.win_prob_history:
                        st.session_state.win_prob_history.pop()
                    st.rerun()
        
        # VISUAL DRAFT BOARD
        st.divider()
        st.subheader("📋 Draft Board")
        
        # Helper function to render a slot
        def render_slot(slot_idx, team_name):
            is_completed = slot_idx < len(st.session_state.live_draft)
            is_current = slot_idx == len(st.session_state.live_draft)
            is_ban = slot_idx < 8
            
            # Determine content
            if is_completed:
                h_id = st.session_state.live_draft[slot_idx]
                h_data = hero_map.get(h_id)
                hero_name = h_data['name'] if h_data else f"ID{h_id}"
                content = f"#{slot_idx+1} {hero_name}"
            elif is_current:
                content = f"➤ #{slot_idx+1}"
            else:
                content = f"#{slot_idx+1}"
            
            # Styling
            if is_current:
                bg_color = "#FFD700" if team_name == "Radiant" else "#FF4500"
                text_color = "black"
                border = "3px solid black"
            elif is_completed:
                bg_color = "#228B22" if team_name == "Radiant" else "#8B0000"
                text_color = "white"
                border = "1px solid #444"
            else:
                bg_color = "#2a2a2a"
                text_color = "#666"
                border = "1px dashed #444"
            
            # Size
            height = "40px" if is_ban else "60px"
            font_size = "10px" if is_ban else "12px"
            
            return f"""
            <div style="
                background-color: {bg_color};
                color: {text_color};
                border: {border};
                border-radius: 4px;
                padding: 4px;
                margin: 2px;
                text-align: center;
                height: {height};
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: {font_size};
                font-weight: bold;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            ">
                {content}
            </div>
            """
        
        # Render Draft Grid
        # Radiant: Slots 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 (even)
        # Dire:    Slots 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 (odd)
        
        col_r, col_d = st.columns(2)
        
        with col_r:
            st.markdown("#### 🟢 Radiant")
            st.markdown("**Bans:**")
            ban_cols = st.columns(4)
            radiant_ban_slots = [0, 2, 4, 6]
            for i, slot in enumerate(radiant_ban_slots):
                with ban_cols[i]:
                    st.markdown(render_slot(slot, "Radiant"), unsafe_allow_html=True)
            
            st.markdown("**Picks:**")
            pick_cols = st.columns(5)
            radiant_pick_slots = [8, 10, 12, 14, 16]
            for i, slot in enumerate(radiant_pick_slots):
                with pick_cols[i]:
                    st.markdown(render_slot(slot, "Radiant"), unsafe_allow_html=True)
        
        with col_d:
            st.markdown("#### 🔴 Dire")
            st.markdown("**Bans:**")
            ban_cols = st.columns(4)
            dire_ban_slots = [1, 3, 5, 7]
            for i, slot in enumerate(dire_ban_slots):
                with ban_cols[i]:
                    st.markdown(render_slot(slot, "Dire"), unsafe_allow_html=True)
            
            st.markdown("**Picks:**")
            pick_cols = st.columns(5)
            dire_pick_slots = [9, 11, 13, 15, 17]
            for i, slot in enumerate(dire_pick_slots):
                with pick_cols[i]:
                    st.markdown(render_slot(slot, "Dire"), unsafe_allow_html=True)
        
        st.divider()

        # Main Area: Draft Board & Predictions
        col_board, col_pred = st.columns([1, 1])
        
        with col_board:
            st.subheader("Draft Board")
            # Separate picks by team
            radiant_heroes = []
            dire_heroes = []
            
            for i, h_id in enumerate(st.session_state.live_draft):
                # hero_map[id] is a dict {'name': ..., ...}
                h_data = hero_map.get(h_id)
                h_name = h_data['name'] if h_data else f"Unknown ({h_id})"
                
                # Logic: Even indices = Radiant, Odd = Dire
                if i % 2 == 0:
                    radiant_heroes.append(f"{i+1}. {h_name}")
                else:
                    dire_heroes.append(f"{i+1}. {h_name}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Radiant**")
                for h in radiant_heroes:
                    st.write(h)
            with c2:
                st.markdown("**Dire**")
                for h in dire_heroes:
                    st.write(h)
                    
            # Hero Selector (Only if draft not full)
            if turn_idx < 24:
                st.divider()
                st.write("Make Selection:")
                # Filter out taken heroes
                # name_to_id is {name: id}, so unpack as (name, hid)
                available_heroes = {name: hid for name, hid in assistant.name_to_id.items() 
                                  if hid not in st.session_state.live_draft}
                
                selected_name = st.selectbox("Choose Hero", options=sorted(available_heroes.keys()))
                
                if st.button(f"Confirm {action} {selected_name}", type="primary"):
                    hid = available_heroes[selected_name]
                    st.session_state.live_draft.append(hid)
                    
                    # Calculate win prob for this new state and store it
                    temp_preds = assistant.predict_next_step(st.session_state.live_draft, use_mcts=ruthless_mode)
                    st.session_state.win_prob_history.append(temp_preds['win_prob'])
                    
                    st.rerun()

        with col_pred:
            st.subheader("Model Insights")
            
            if st.session_state.live_draft:
                # Get Predictions
                with st.spinner("Consulting Oracle..."):
                    preds = assistant.predict_next_step(st.session_state.live_draft, use_mcts=ruthless_mode)
                
                # 1. Win Probability
                win_p = preds['win_prob']
                delta = 0 # Could track delta from previous turn if we stored it
                st.metric("Radiant Win Probability", f"{win_p:.1%}")
                
                # 2. Lane Synergy
                st.write("**Predicted Lane Outcomes (Radiant):**")
                lanes = preds['lanes']
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Safe", f"{lanes['Safe']:.0%}")
                lc2.metric("Mid", f"{lanes['Mid']:.0%}")
                lc3.metric("Off", f"{lanes['Off']:.0%}")
                
                # 3. Suggestions
                if turn_idx < 24:
                    st.write(f"**Top Suggestions for {team} {action}:**")
                    suggestions = preds['suggestions']
                    for s in suggestions:
                        st.write(f"- **{s['name']}** (Score: {s['logit']:.2f})")
                
                # 4. Download Draft as JSON
                st.divider()
                st.write("**Export Draft:**")
                
                if st.session_state.live_draft:
                    import json
                    from datetime import datetime
                    
                    # Build export data
                    export_data = {
                        "timestamp": datetime.now().isoformat(),
                        "draft_sequence": st.session_state.live_draft,
                        "hero_names": [
                            hero_map.get(hid, {}).get('name', f'Unknown_{hid}') 
                            for hid in st.session_state.live_draft
                        ],
                        "win_prob_history": st.session_state.win_prob_history,
                        "final_win_prob": win_p,
                        "num_picks": len(st.session_state.live_draft)
                    }
                    
                    json_str = json.dumps(export_data, indent=2)
                    
                    st.download_button(
                        label="📥 Download Draft JSON",
                        data=json_str,
                        file_name=f"draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            else:
                st.info("Start the draft to see predictions.")

    with tab3:
        st.header("Draft Probability Space Analysis")
        st.markdown("""
        Visualize the **Draft Trajectory** through the model's 256D state space.
        * **Win Probability**: The "elevation" of the current state.
        * **Entropy**: The "uncertainty" (fog) at the current state.
        """)
        
        from src.draft_analyzer import DraftAnalyzer, load_sample_draft
        import json
        analyzer = DraftAnalyzer(model)
        
        # UI: Draft Source Selection
        st.subheader("1. Select Draft Source")
        
        draft_source = st.radio(
            "Choose draft to analyze:",
            ["Current Draft (from Assistant)", "Upload JSON", "Sample Draft"],
            horizontal=True
        )
        
        draft_to_analyze = None
        win_prob_data = None
        
        if draft_source == "Current Draft (from Assistant)":
            if st.session_state.get('live_draft'):
                draft_to_analyze = st.session_state.live_draft
                win_prob_data = st.session_state.get('win_prob_history', [])
                st.success(f"Using current draft ({len(draft_to_analyze)} picks)")
            else:
                st.warning("No current draft available. Go to Draft Assistant tab to create one.")
        
        elif draft_source == "Upload JSON":
            uploaded_file = st.file_uploader("Upload a draft JSON file", type=['json'])
            if uploaded_file is not None:
                try:
                    json_data = json.load(uploaded_file)
                    draft_to_analyze = json_data.get('draft_sequence', [])
                    win_prob_data = json_data.get('win_prob_history', [])
                    
                    st.success(f"Loaded draft: {len(draft_to_analyze)} picks")
                    st.write(f"**Heroes:** {', '.join(json_data.get('hero_names', []))}")
                except Exception as e:
                    st.error(f"Error loading JSON: {e}")
        
        else:  # Sample Draft
            draft_to_analyze = load_sample_draft()
            st.info(f"Using sample draft ({len(draft_to_analyze)} picks)")
        
        # Store in session state for analysis button
        if 'draft_seq' not in st.session_state:
            st.session_state.draft_seq = load_sample_draft()
        
        if draft_to_analyze:
            st.session_state.draft_seq = draft_to_analyze
            
        if st.button("📊 Analyze Draft"):
            # Run Analysis
            results = analyzer.analyze_draft_sequence(st.session_state.draft_seq)
            key_moments = analyzer.find_turning_points(
                results['win_probs'], results['entropies'], results['turns']
            )
            
            # Map IDs to Names for Visualization
            draft_names = []
            for h_id in st.session_state.draft_seq:
                h_data = hero_map.get(h_id)
                draft_names.append(h_data['name'] if h_data else f"ID {h_id}")
            
            # Plot 1: Win Probability
            st.subheader("2. Win Probability Trajectory")
            import plotly.graph_objects as go
            fig_win = go.Figure()
            
            # Main trace (calculated from model)
            fig_win.add_trace(go.Scatter(
                x=results['turns'], 
                y=results['win_probs'],
                mode='lines+markers', 
                name='Win % (Calculated)',
                text=draft_names, # Hover text
                hovertemplate='<b>%{text}</b><br>Turn %{x}<br>Win: %{y:.1%}<extra></extra>',
                line=dict(color='#00ff00', width=3)
            ))
            
            # Overlay historical data if available from uploaded JSON
            if win_prob_data and len(win_prob_data) > 0:
                fig_win.add_trace(go.Scatter(
                    x=list(range(1, len(win_prob_data) + 1)),
                    y=win_prob_data,
                    mode='lines+markers',
                    name='Win % (Historical)',
                    line=dict(color='#FFD700', width=2, dash='dot'),
                    marker=dict(size=6, symbol='star')
                ))
            
            fig_win.add_hline(y=0.5, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_win, use_container_width=True)
            
            # Plot 2: Entropy
            st.subheader("2. Certainty (Entropy) Evolution")
            fig_ent = go.Figure()
            fig_ent.add_trace(go.Scatter(
                x=results['turns'], 
                y=results['entropies'],
                mode='lines+markers', 
                name='Entropy',
                text=draft_names, # Hover text
                hovertemplate='<b>%{text}</b><br>Turn %{x}<br>Entropy: %{y:.2f}<extra></extra>',
                line=dict(color='#ff00ff', width=3),
                fill='tozeroy'
            ))
            st.plotly_chart(fig_ent, use_container_width=True)
            
            # Key Moments
            st.subheader("3. Key Moments")
            # Enrich key moments with names
            for k, v in key_moments.items():
                turn_idx = v['turn'] - 1 # 0-indexed
                if turn_idx < len(draft_names):
                    v['hero'] = draft_names[turn_idx]
                else:
                    v['hero'] = "Unknown"
            
            st.json(key_moments)

                