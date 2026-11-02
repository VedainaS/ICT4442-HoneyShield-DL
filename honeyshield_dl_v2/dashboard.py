"""HoneyShield-DL presentation dashboard. Run: python -m streamlit run dashboard.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix
from models import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
st.set_page_config(page_title='HoneyShield-DL', page_icon='🛡️', layout='wide')
st.title('🛡️ HoneyShield-DL')
st.caption('Enterprise login anomaly detection • Synthetic data • Four autoencoder architectures')

@st.cache_data
def load_data():
    X=np.load(ROOT/'data/sequences_X.npy')
    y=np.load(ROOT/'data/sequences_y.npy')
    attack=np.load(ROOT/'data/sequences_attack.npy',allow_pickle=False)
    split=np.load(ROOT/'data/sequences_split.npy')
    if not (len(X)==len(y)==len(attack)==len(split)):
        raise ValueError('Sequence files have inconsistent lengths. Rebuild them together.')
    mask=split=='test'
    return X[mask],y[mask],attack[mask],np.flatnonzero(mask)

@st.cache_resource
def load_model(name):
    saved=torch.load(ROOT/f'checkpoints/{name}.pt',map_location='cpu',weights_only=True)
    model=MODEL_REGISTRY[name](saved['seq_len'],saved['num_features'])
    model.load_state_dict(saved['state_dict']);model.eval()
    return model,float(saved['threshold'])

def score(name,X):
    model,threshold=load_model(name)
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(X),128):
            batch=torch.tensor(X[start:start+128],dtype=torch.float32)
            chunks.append(((model(batch)-batch)**2).mean(dim=(1,2)).numpy())
    return np.concatenate(chunks),threshold

def metrics(y,scores,threshold):
    p=(scores>threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(Accuracy=accuracy_score(y,p),Precision=precision_score(y,p,zero_division=0),
                Recall=recall_score(y,p,zero_division=0),F1=f1_score(y,p,zero_division=0),
                AP=average_precision_score(y,scores),FPR=fp/(fp+tn),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))

try:
    X,y,attacks,indices=load_data()
    recorded=json.loads((ROOT/'results/final_run_25ep.json').read_text())
    baseline=json.loads((ROOT/'results/isolation_forest_result.json').read_text())
except (OSError,ValueError) as exc:
    st.error(f'Project data unavailable: {exc}');st.stop()

with st.sidebar:
    st.header('Demo controls')
    selected=st.selectbox('Model',list(MODEL_REGISTRY),index=1,format_func=str.upper,key="model")
    run=st.button('Run selected model',type='primary')
    run_all=st.button('Run all four models')
    st.caption('Buttons load saved weights and execute inference. No training and no result files are overwritten.')
    st.divider()
    st.write('**Evaluation protocol**')
    st.write('Disjoint users; training-only preprocessing; entirely normal train/validation windows.')
    st.write('Threshold: 99th percentile of clean normal validation error.')
    st.warning('A flagged event is a prediction, not a confirmed attack.')

for col,label,value in zip(st.columns(4),['Test windows','Actual attacks','Normal targets','Always-normal accuracy'],[f'{len(y):,}',f'{int(y.sum()):,}',f'{int((y==0).sum()):,}',f'{(y==0).mean():.2%}']):
    col.metric(label,value)

if 'runs' not in st.session_state:st.session_state.runs={}
if run or run_all:
    names=list(MODEL_REGISTRY) if run_all else [selected]
    with st.spinner('Running checkpoint inference on held-out users...'):
        try:
            for name in names:
                scores,threshold=score(name,X)
                st.session_state.runs[name]=(scores,threshold)
        except (OSError,RuntimeError,ValueError) as exc:
            st.error(f'Prediction failed: {exc}')
    if all(n in st.session_state.runs for n in names):st.success('Inference completed on the test set.')

tab1,tab2,tab3,tab4=st.tabs(['Model comparison','Live predictions','Attack analysis','How to improve'])
with tab1:
    st.subheader('Common test-set comparison')
    rows=[]
    for item in recorded:
        name=item['model']
        if name in st.session_state.runs:
            m=metrics(y,*st.session_state.runs[name]);source='Live checkpoint run'
        else:
            n=item['n_test']; tp=item['true_positives']; fp=item['false_positives']; normal=n-item['n_test_anomalous']
            m=dict(Accuracy=(tp+normal-fp)/n,Precision=item['precision'],Recall=item['recall'],F1=item['f1_score'],AP=item['pr_auc'],FPR=item['false_positive_rate'])
            source='Saved verified run'
        rows.append({'Model':name.upper(),**{k:m[k] for k in ['Accuracy','Precision','Recall','F1','AP','FPR']},'Source':source})
    # Baseline artifact does not store confusion counts; only show measured fields.
    rows.append(dict(Model='Isolation Forest',Accuracy=np.nan,Precision=baseline['precision'],Recall=baseline['recall'],F1=baseline['f1_score'],AP=baseline['pr_auc'],FPR=np.nan,Source='Saved baseline run'))
    comparison=pd.DataFrame(rows).set_index('Model')
    st.dataframe(comparison.style.format({k:'{:.4f}' for k in ['Accuracy','Precision','Recall','F1','AP','FPR']},na_rep='—'),width='stretch')
    st.bar_chart(comparison[['Precision','Recall','F1','AP']])
    st.caption('AP = average precision; FPR = false-positive rate. Missing baseline fields were not saved. Baseline uses the last event only.')
    st.info('CNN leads on F1, but the neural scores are close. Isolation Forest has the highest AP. High accuracy can hide missed attacks in this imbalanced dataset.')
    st.download_button('Download comparison CSV',comparison.to_csv(),file_name='dashboard_comparison.csv',mime='text/csv')
with tab2:
    st.subheader(f'{selected.upper()} • checkpoint predictions')
    if selected not in st.session_state.runs:
        st.info('Click “Run selected model” to compute predictions live.')
    else:
        scores,threshold=st.session_state.runs[selected];p=(scores>threshold).astype(int);m=metrics(y,scores,threshold)
        for col,label,value in zip(st.columns(4),['Flagged targets','True positives','False positives','Missed attacks'],[int(p.sum()),m['TP'],m['FP'],m['FN']]):col.metric(label,value)
        st.caption(f'Fixed validation threshold: {threshold:.8g}. Scores are reconstruction errors, not probabilities.')
        left,right=st.columns(2)
        with left:
            st.write('**Confusion matrix**')
            st.dataframe(pd.DataFrame([[m['TN'],m['FP']],[m['FN'],m['TP']]],index=['Actual normal','Actual attack'],columns=['Predicted normal','Predicted attack']),width='stretch')
        with right:
            st.write('**Score distribution**')
            counts,edges=np.histogram(np.log10(np.maximum(scores,1e-12)),bins=25)
            st.bar_chart(pd.DataFrame({'Windows':counts},index=np.round((edges[:-1]+edges[1:])/2,2)))
            st.caption('Horizontal axis: log10 reconstruction error.')
        frame=pd.DataFrame({'Sequence index':indices,'Attack category':attacks,'Actual anomaly':y.astype(int),'Score':scores,'Predicted anomaly':p})
        view=st.selectbox('Show rows',['All test windows','Flagged only','False positives','Missed attacks'],key='rows')
        masks={'All test windows':np.ones(len(y),dtype=bool),'Flagged only':p==1,'False positives':(p==1)&(y==0),'Missed attacks':(p==0)&(y==1)}
        st.dataframe(frame[masks[view]],width='stretch',hide_index=True)
        st.download_button('Download test predictions',frame.to_csv(index=False),file_name=f'{selected}_test_predictions.csv',mime='text/csv')
with tab3:
    st.subheader('Recall by attack category')
    categories=list(recorded[0]['recall_by_attack'])
    attack_rows=[]
    for category in categories:
        row={'Attack':category,'Test count':int((attacks==category).sum())}
        for item in recorded:
            name=item['model']
            if name in st.session_state.runs:
                scores,threshold=st.session_state.runs[name]
                row[name.upper()]=float((scores[attacks==category]>threshold).mean())
            else:row[name.upper()]=item['recall_by_attack'][category]
        attack_rows.append(row)
    by_attack=pd.DataFrame(attack_rows).set_index('Attack')
    st.dataframe(by_attack,width='stretch')
    st.bar_chart(by_attack.drop(columns='Test count'))
    st.warning('Read recall together with false alarms. Whole-window errors may flag a normal target because its history contains an attack.')
with tab4:
    st.subheader('Improve the detector, not just the displayed score')
    st.markdown('''1. **Align score and label.** Evaluate last-event reconstruction error for the last-event target; compare against the current whole-window score.
2. **Build representative validation windows.** Current validation has entirely clean histories. Deployment-like validation should include normal targets with anomalous history. Use a separate labelled validation set if tuning F1.
3. **Try an ensemble after calibration.** Raw MSE scales differ across models. Normalize scores using validation distributions before averaging, or test majority voting. Combination can also combine false alarms.
4. **Improve representations.** Add new-device indicators, user-relative behavior and actual distance/time features; mask padding and consider a per-step attention decoder.
5. **Evaluate honestly.** Select configurations on validation only, freeze them, then use a fresh untouched holdout. The current test set has already been inspected repeatedly. Report multiple seeds and false-positive costs.''')
    st.caption('These are proposed experiments, not measured improvements. Existing results and report numbers remain unchanged.')
