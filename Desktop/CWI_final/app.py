import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dash_bootstrap_components as dbc
from sklearn.metrics import cohen_kappa_score
import os
from datetime import datetime

# Load and prepare data
final_sampled_df = pd.read_csv('final_sampled_df.csv').reset_index(drop=True)
filtered_train_df = pd.read_csv('filtered_train.csv').reset_index(drop=True)
final_sampled_rows = pd.read_csv('final_sampled_rows.csv').reset_index(drop=True)

# Configuration
FEATURES = ['frequency', 'length', 'syllable_count',
           'familiarity', 'concreteness', 'imaginability', 'aoa_gpt']
CLUSTER_COL = 'cluster'
TARGET_COL = 'binary_complex'
WORD_COL = 'Complex_Phrase'

# Initialize components
scaler = StandardScaler()
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Prepare data pools
full_data = pd.concat([final_sampled_df, filtered_train_df])
scaler.fit(full_data[FEATURES])

# Initialize model with baseline data
X_baseline = final_sampled_df[FEATURES]
model.fit(scaler.transform(X_baseline), final_sampled_df[TARGET_COL])

# Create active learning pool
active_learning_pool = filtered_train_df[
    ~filtered_train_df.index.isin(final_sampled_df.index)
].reset_index(drop=True)

active_learning_pool[TARGET_COL] = np.nan

# Prepare test data
test_data = final_sampled_rows.reset_index(drop=True)
test_indices = test_data.index.tolist()

def select_uncertain_word(candidate_indices):
    if not candidate_indices:
        return None
    X = scaler.transform(active_learning_pool.loc[candidate_indices][FEATURES])
    probas = model.predict_proba(X)
    entropy = -np.sum(probas * np.log(probas + 1e-10), axis=1)
    return candidate_indices[np.argmax(entropy)]

def propagate_labels(word_idx, label):
    cluster_value = active_learning_pool.loc[word_idx, CLUSTER_COL]
    cluster_mask = active_learning_pool[CLUSTER_COL] == cluster_value
    cluster_members = active_learning_pool[cluster_mask & (active_learning_pool.index != word_idx)]
    
    propagated_indices = []
    if not cluster_members.empty:
        word_features = active_learning_pool.loc[[word_idx], FEATURES]
        distances = np.linalg.norm(
            scaler.transform(cluster_members[FEATURES]) - scaler.transform(word_features),
            axis=1
        )
        nearest_indices = cluster_members.iloc[np.argsort(distances)[:150]].index.tolist()
        active_learning_pool.loc[nearest_indices, TARGET_COL] = label
        propagated_indices = nearest_indices
    
    return propagated_indices

def retrain_model():
    train_data = pd.concat([
        final_sampled_df,
        active_learning_pool[active_learning_pool[TARGET_COL].notna()]
    ])
    counts = active_learning_pool[TARGET_COL].value_counts()
    print("Complex words (1):", counts.get(1, 0))
    print("Simple words (0):", counts.get(0, 0))
    print("Unlabeled words:", active_learning_pool[TARGET_COL].isna().sum())
      
    X_train = scaler.transform(train_data[FEATURES])
    y_train = train_data[TARGET_COL]
    model.fit(X_train, y_train)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize first word before creating layout
initial_remaining = active_learning_pool.index.tolist()
initial_word_idx = select_uncertain_word(initial_remaining) if initial_remaining else None

app.layout = dbc.Container([
    dcc.Store(id='store-state', data={
        'training_annotations_made': 0,
        'test_annotations_made': 0,
        'remaining_train_indices': initial_remaining,
        'remaining_test_indices': test_indices[:22].copy(),
        'current_word_idx': initial_word_idx,
        'test_labels': {},
        'phase': 'train',
        'user_data': None,
        'submitted': False
    }),
    
    html.H1("Active Learning Annotation", className="mb-4"),
    
    # User input form
    html.Div(id='user-form', children=[
        html.H3("User Information"),
        dbc.Input(id='user-name', placeholder="Enter full name", className="mb-2"),
        dcc.Dropdown(
            id='english-level',
            options=[
                {'label': 'Beginner', 'value': 'beginner'},
                {'label': 'Intermediate', 'value': 'intermediate'},
                {'label': 'Advanced', 'value': 'advanced'},
                {'label': 'Native', 'value': 'native'}
            ],
            placeholder="Select English proficiency level",
            className="mb-3"
        ),
        dbc.Button("Submit", id='submit-btn', color="primary"),
        html.Div(id='form-status', className="mt-2")
    ]),
    
    # Annotation interface (hidden until submission)
    html.Div(id='annotation-interface', style={'display': 'none'}, children=[
        html.H3(id='current-word'),
        dbc.Row([
            dbc.Col(dbc.Button("Simple (0)", color="success", id='btn-simple')),
            dbc.Col(dbc.Button("Complex (1)", color="danger", id='btn-complex')),
        ], className="mb-3"),
        dbc.Progress(id='progress', value=0),
        html.Div(id='progress-text', className="mt-2")
    ]),
    
    html.Div(id='results', className="mt-4")
])
@callback(
    [Output('annotation-interface', 'style'),
     Output('form-status', 'children'),
     Output('store-state', 'data', allow_duplicate=True)],  # Add allow_duplicate here
    [Input('submit-btn', 'n_clicks')],
    [State('user-name', 'value'),
     State('english-level', 'value'),
     State('store-state', 'data')],
    prevent_initial_call=True
)

def handle_submission(n_clicks, name, level, state):
    if n_clicks is None or not name or not level:
        return {'display': 'none'}, "Please fill all fields", state
    
    # Create user entry
    user_data = {
        'name': name,
        'english_level': level,
        'timestamp': datetime.now().isoformat(),
        'f1_score': None,
        'kappa_score': None
    }
    
    # Save initial entry to CSV
    df = pd.DataFrame([user_data])
    if not os.path.exists('user_data.csv'):
        df.to_csv('user_data.csv', index=False)
    else:
        df.to_csv('user_data.csv', mode='a', header=False, index=False)
    
    # Update state
    state['user_data'] = user_data
    state['submitted'] = True
    
    return {'display': 'block'}, "Submission successful! Starting annotation...", state

@callback(
    [Output('current-word', 'children'),
     Output('progress', 'value'),
     Output('progress-text', 'children'),
     Output('store-state', 'data', allow_duplicate=True),  # Add allow_duplicate here
     Output('results', 'children')],
    [Input('btn-simple', 'n_clicks'),
     Input('btn-complex', 'n_clicks'),
     Input('store-state', 'data')],
    [State('store-state', 'data')],
    prevent_initial_call=True
)
def handle_annotation(simple_clicks, complex_clicks, state_input, state):
    ctx = dash.callback_context

    if not ctx.triggered:
        if state['current_word_idx'] is None and state['phase'] == 'train':
            if state['remaining_train_indices']:
                state['current_word_idx'] = select_uncertain_word(state['remaining_train_indices'])
        return update_progress(state)

    if ctx.triggered[0]['prop_id'] in ('btn-simple.n_clicks', 'btn-complex.n_clicks'):
        if state['phase'] == 'train':
            return handle_training_phase(ctx, state)
        elif state['phase'] == 'test':
            return handle_test_phase(ctx, state)
    
    return update_progress(state)

def handle_training_phase(ctx, state):
    if state['training_annotations_made'] >= 22:
        state['phase'] = 'test'
        return get_next_test_word(state)
    
    label = 0 if 'btn-simple' in ctx.triggered[0]['prop_id'] else 1
    
    if state['current_word_idx'] is not None:
        propagated_indices = propagate_labels(state['current_word_idx'], label)
        
        all_indices = [state['current_word_idx']] + propagated_indices
        state['remaining_train_indices'] = [
            idx for idx in state['remaining_train_indices']
            if idx not in all_indices
        ]
        
        retrain_model()
        state['training_annotations_made'] += 1

        if state['remaining_train_indices']:
            state['current_word_idx'] = select_uncertain_word(state['remaining_train_indices'])
        else:
            state['current_word_idx'] = None
    
    return update_progress(state)

def handle_test_phase(ctx, state):
    if not state['remaining_test_indices']:
        return final_results(state)
    
    label = 0 if 'btn-simple' in ctx.triggered[0]['prop_id'] else 1
    current_idx = state['remaining_test_indices'][0]
    
    state['test_labels'][str(current_idx)] = label
    state['test_annotations_made'] += 1
    state['remaining_test_indices'].pop(0)
    
    return update_progress(state)

def update_progress(state):
    total = 22 + 23
    done = state['training_annotations_made'] + state['test_annotations_made']
    progress = (done / total) * 100
    
    if state['phase'] == 'train' and state['current_word_idx'] is not None:
        word = active_learning_pool.loc[state['current_word_idx'], WORD_COL]
        text = f"Training Word {state['training_annotations_made']+1}/23: {word}"
    elif state['phase'] == 'test' and state['remaining_test_indices']:
        current_idx = state['remaining_test_indices'][0]
        word = test_data.loc[current_idx, WORD_COL]
        text = f"Test Word {state['test_annotations_made']+1}/22: {word}"
    else:
        return final_results(state)
    
    return (
        text,
        progress,
        f"Overall Progress: {done}/45",
        state,
        dash.no_update
    )

def final_results(state):
    X_test = scaler.transform(test_data[FEATURES])
    y_pred = model.predict(X_test)
    y_true = [state['test_labels'].get(str(i), 0) for i in range(len(test_data))]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred)
    }
    
    # Update CSV with metrics
    if state['user_data']:
        user_data = state['user_data']
        df = pd.read_csv('user_data.csv')
        
        mask = (df['name'] == user_data['name']) & (df['timestamp'] == user_data['timestamp'])
        df.loc[mask, 'f1_score'] = metrics['f1']
        df.loc[mask, 'kappa_score'] = metrics['kappa']
        df.to_csv('user_data.csv', index=False)
    
    return (
        "Annotation Complete!",
        100,
        "",
        state,
        html.Div([
            html.H3("Final Results"),
            html.Div([
                html.H4("Your Model"),
                html.P(f"Accuracy: {metrics['accuracy']:.2%}"),
                html.P(f"F1 Score: {metrics['f1']:.2f}"),
                html.P(f"Precision: {metrics['precision']:.2f}"),
                html.P(f"Recall: {metrics['recall']:.2f}"),
                html.P(f"Kappa: {metrics['kappa']:.2f}")
            ], className='mb-4'),
            html.H4("Baseline Comparisons"),
            html.Div([
                html.Div([
                    html.H5("Always Predict 0"),
                    html.P(f"Accuracy: {accuracy_score(y_true, [0]*len(y_true)):.2%}")
                ], className='border p-3 m-2'),
                html.Div([
                    html.H5("Always Predict 1"),
                    html.P(f"Accuracy: {accuracy_score(y_true, [1]*len(y_true)):.2%}")
                ], className='border p-3 m-2')
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ])
    )

def get_next_test_word(state):
    if not state['remaining_test_indices']:
        return final_results(state)
    
    current_idx = state['remaining_test_indices'][0]
    word = test_data.loc[current_idx, WORD_COL]
    progress = ((state['training_annotations_made'] + 1) / (22 + 23)) * 100
    
    return (
        f"Test Word 1/22: {word}",
        progress,
        f"Overall Progress: {state['training_annotations_made'] + 1}/45",
        state,
        dash.no_update
    )


app = dash.Dash(__name__)
server = app.server 
if __name__ == '__main__':
    app.run(debug=True)