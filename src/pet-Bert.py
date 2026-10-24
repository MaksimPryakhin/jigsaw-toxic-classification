from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
os.makedirs('reports/data/processed', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
df = pd.read_csv("/kaggle/input/datasets/maksimpryakhin/trains/train.csv")
pd.set_option('display.max_columns', None)
numeric = df.select_dtypes(include=np.number).columns
for label in numeric:
    print(f"{label}: {df[label].sum()} ({df[label].mean() * 100:.2f}%)")

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df['length'] = df['comment_text'].str.len()
per = df['length'].quantile([0.5, 0.9, 0.95, 0.99])
print(per)
for label in label_cols:
    sr_0 = df[df[label] == 0]['length'].mean()
    sr_1 = df[df[label] == 1]['length'].mean()
    sr = sr_0 - sr_1
    print(f'Метка {label}: средние: {sr_0}, {sr_1}, разница = {sr}, процент: {sr / sr_0 * 100}')

fig, ax = plt.subplots(figsize=(8, 5))
counts = df[label_cols].sum().sort_values()
bars = ax.barh(counts.index, counts.values, color='steelblue')
ax.set_title('Распределение классов')
ax.set_xlabel('Количество комментариев')
pct = counts / len(df) * 100
for bar, count, p in zip(bars, counts.values, pct):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{count:,} ({p:.2f}%)')
ax.set_xlim(0, counts.max() * 1.3)
plt.tight_layout()
plt.savefig('reports/figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df['length'].clip(upper=2000), bins=50)
ax.axvline(1355, color='red', ls='--', label='p95 = 1355')
ax.set_title('Распределение длин комментариев')
ax.set_xlabel('Длина комментария')
ax.set_ylabel('Количество')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/comment_length.png', dpi=150, bbox_inches='tight')
plt.show()

print((df['length'] < 10).sum())
print((df['length'] == 0).sum())
print(df[df['length'] < 10][label_cols].sum())
print(df[label_cols].corr().round(2))

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
n_train = int(0.8 * n)
n_val = int(0.1 * n)
df.loc[:n_train-1, 'split'] = 'train'
df.loc[n_train:n_train + n_val - 1, 'split'] = 'val'
df.loc[n_train + n_val:, 'split'] = 'test'
df_prov = pd.DataFrame({
    label: df.groupby('split')[label].mean()
    for label in label_cols
})
df.to_csv('reports/data/processed/jigsaw_split.csv', index=False)

df_split = pd.read_csv('reports/data/processed/jigsaw_split.csv')

train = df_split[df_split['split'] == 'train']
val = df_split[df_split['split'] == 'val']

X_train = train['comment_text']
y_train = train[label_cols]
X_val = val['comment_text']
y_val = val[label_cols]

vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = vec.fit_transform(X_train)
X_val_tfidf = vec.transform(X_val)

baseline_model = OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'))
baseline_model.fit(X_train_tfidf, y_train.values)
baseline_proba = baseline_model.predict_proba(X_val_tfidf)
print(f"macro ROC-AUC: {roc_auc_score(y_val.values, baseline_proba, average='macro'):.4f}")
per_class = roc_auc_score(y_val.values, baseline_proba, average=None)
for label, score in zip(label_cols, per_class):
    print(f"{label}: {score:.4f}")

val_ids = val['id'].values
baseline_df = pd.DataFrame(baseline_proba, columns=[f'{c}_proba' for c in label_cols])
baseline_df.insert(0, 'id', val_ids)
baseline_df.to_csv('reports/baseline_val_proba.csv', index=False)


class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


local_model_path = "/kaggle/input/datasets/maksimpryakhin/distilbert-base/distilbert-base-uncased"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    local_model_path,
    num_labels=6,
    problem_type='multi_label_classification',
    local_files_only=True
).to(device)
train_dataset = ToxicDataset(X_train.values, y_train.values, tokenizer, max_length=512)
val_dataset = ToxicDataset(X_val.values, y_val.values, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


pos_weight = torch.tensor(
    ((len(y_train) - y_train.sum()) / y_train.sum()).values,
    dtype=torch.float
).to(device)

num_epochs = 1
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * num_epochs
warmup_step = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_step,
    num_training_steps=total_steps
)


def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, 'Training')
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
        )
        loss = criterion(outputs.logits, batch['labels'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}'
        })
    return total_loss / len(loader)


def val_one_epoch(model, loader, device):
    model.eval()
    all_log = []
    all_labels = []
    pbar = tqdm(loader, 'Validation')
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            preds = outputs.logits
            all_log.append(preds.cpu())
            all_labels.append(batch['labels'].cpu())
    all_log = torch.cat(all_log).numpy()
    all_labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(all_labels, all_log, average='macro')
    return auc


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs):
    history = {'train_loss': [], 'val_roc_auc': []}
    best_roc_auc = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_roc_auc = val_one_epoch(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_roc_auc'].append(val_roc_auc)

        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_roc_auc={val_roc_auc:.4f}")

        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            torch.save(model.state_dict(), 'best_bert.pt')
            print(f"  → Сохранено (best_roc_auc={best_roc_auc:.4f})")

    return history


history = train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs)

model.eval()
all_logits = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device)
        )
        preds = outputs.logits
        all_logits.append(preds.cpu())
        all_labels.append(batch['labels'].cpu())
all_logits = torch.cat(all_logits).numpy()
all_labels = torch.cat(all_labels).numpy()
bert_proba = torch.sigmoid(torch.tensor(all_logits)).numpy()
bert_df = pd.DataFrame(bert_proba, columns=[f'{c}_proba' for c in label_cols])
bert_df.insert(0, 'id', val_ids)
bert_df.to_csv('reports/bert_val_proba.csv', index=False)


def macro_f1(proba):
    thresholds = {}
    f1_scores = {}
    for i, label in enumerate(label_cols):
        best_f1 = 0
        best_t = 0.5
        for t in np.arange(0.05, 0.96, 0.05):
            pred = (proba[:, i] > t).astype(int)
            f1 = f1_score(y_val[label].values, pred)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        thresholds[label] = best_t
        f1_scores[label] = best_f1
    return thresholds, f1_scores


baseline_thresholds, baseline_f1_scores = macro_f1(baseline_proba)
bert_thresholds, bert_f1_scores = macro_f1(bert_proba)

print(np.mean(list(baseline_f1_scores.values())))
print(np.mean(list(bert_f1_scores.values())))

for label in label_cols:
    print(f'{label}: baseline={baseline_f1_scores[label]:.4f}, transformer={bert_f1_scores[label]:.4f}, diff={bert_f1_scores[label] - baseline_f1_scores[label]:+.4f}')

fp_mask = (bert_proba[:, 5] > 0.7) & (y_val['identity_hate'].values == 0)
fp_idx = np.where(fp_mask)[0]
print('false positive identity_hate')
for i in fp_idx[:3]:
    print(f'idx={i}, proba={bert_proba[i, 5]:.3f}, label={y_val["identity_hate"].values[i]}')
    print(val.iloc[i]['comment_text'][:300].replace('\n', ' '))
    print('---')

fn_mask = (bert_proba[:, 3] < 0.3) & (y_val['threat'].values == 1)
fn_idx = np.where(fn_mask)[0]
print('\nfalse negative threat')
for i in fn_idx[:3]:
    print(f'idx={i}, proba={bert_proba[i, 3]:.3f}, label={y_val["threat"].values[i]}')
    print(val.iloc[i]['comment_text'][:300].replace('\n', ' '))
    print('---')

tp_mask = (bert_proba[:, 3] > 0.9) & (y_val['threat'].values == 1)
tp_idx = np.where(tp_mask)[0]
print('\ntrue positive threat')
for i in tp_idx[:3]:
    print(f'idx={i}, proba={bert_proba[i, 3]:.3f}, label={y_val["threat"].values[i]}')
    print(val.iloc[i]['comment_text'][:300].replace('\n', ' '))
    print('---')


def predict_fn(texts):
    model.eval()
    with torch.no_grad():
        if isinstance(texts, str):
            texts = [texts]
        else:
            try:
                texts = [' '.join(t) if isinstance(t, (list, tuple)) else str(t) for t in texts]
            except TypeError:
                texts = [str(texts)]
        encoding = tokenizer(texts, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        logits = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        ).logits
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()


import shap
masker = shap.maskers.Text(r'\W')
explainer = shap.Explainer(predict_fn, masker, output_names=label_cols)
examples = {
    'tp_threat': val.iloc[21]['comment_text'],
    'fp_identity_hate': val.iloc[80]['comment_text'],
    'fn_threat_implicit': val.iloc[7841]['comment_text'],
    'fn_threat_explicit': val.iloc[1029]['comment_text'],
}

for name, text in examples.items():
    print(f'\n=== {name} ===')
    shap_values = explainer([text])
    class_idx = 3 if 'threat' in name else 5
    class_name = label_cols[class_idx]
    print(f'class: {class_name}')
    shap.plots.text(shap_values[0, :, class_name])


examples = {
    'tp_threat': (val.iloc[21]['comment_text'], 'threat'),
    'fp_identity_hate': (val.iloc[80]['comment_text'], 'identity_hate'),
    'fn_threat_implicit': (val.iloc[7841]['comment_text'], 'threat'),
    'fn_threat_explicit': (val.iloc[1029]['comment_text'], 'threat'),
}

for name, (text, class_name) in examples.items():
    print(f'=== {name} ===')
    shap_values = explainer([text])
    plt.figure()
    shap.plots.bar(shap_values[0, :, class_name], max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(f'reports/figures/shap_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved reports/figures/shap_{name}.png')

test = df_split[df_split['split'] == 'test'].reset_index(drop=True)
test_dataset = ToxicDataset(test['comment_text'].values, test[label_cols].values, tokenizer, max_length=512)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.load_state_dict(torch.load('best_bert.pt'))
model.eval()
all_logits = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        all_logits.append(outputs.logits.cpu())
all_logits = torch.cat(all_logits).numpy()
bert_test_proba = torch.sigmoid(torch.tensor(all_logits)).numpy()

X_test_tfidf = vec.transform(test['comment_text'])
baseline_test_proba = baseline_model.predict_proba(X_test_tfidf)

y_test = test[label_cols].values

def apply_thresholds(proba, thresholds):
    preds = np.zeros_like(proba)
    for i, label in enumerate(label_cols):
        preds[:, i] = (proba[:, i] > thresholds[label]).astype(int)
    return preds

baseline_test_pred = apply_thresholds(baseline_test_proba, baseline_thresholds)
bert_test_pred = apply_thresholds(bert_test_proba, bert_thresholds)

baseline_test_auc = roc_auc_score(y_test, baseline_test_proba, average='macro')
bert_test_auc = roc_auc_score(y_test, bert_test_proba, average='macro')
baseline_test_f1 = f1_score(y_test, baseline_test_pred, average='macro')
bert_test_f1 = f1_score(y_test, bert_test_pred, average='macro')

print(f'baseline test ROC-AUC: {baseline_test_auc:.4f}, F1: {baseline_test_f1:.4f}')
print(f'bert test ROC-AUC: {bert_test_auc:.4f}, F1: {bert_test_f1:.4f}')

results = pd.DataFrame({
    'model': ['TF-IDF + LogReg', 'DistilBERT'],
    'val_roc_auc': [0.9822, 0.9913],
    'val_f1': [0.6365, 0.7148],
    'test_roc_auc': [baseline_test_auc, bert_test_auc],
    'test_f1': [baseline_test_f1, bert_test_f1]
})
results.to_csv('reports/final_results.csv', index=False)
print(results.round(4))