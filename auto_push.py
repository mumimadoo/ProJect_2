import pandas as pd
import numpy as np
import random
import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. คลาส BoardOptimizer (วิธี khongrau - ต้นฉบับ)
# ==========================================
class BoardOptimizer:
    def __init__(self, fitness_func, n_agents=15, n_dim=30, max_iter=50):
        self.fitness_func = fitness_func
        self.n_agents = n_agents
        self.n_dim = n_dim
        self.max_iter = max_iter

        # เริ่มต้นด้วยการสุ่มยีน 5%
        self.X = np.random.choice([0, 1], size=(n_agents, n_dim), p=[0.95, 0.05])
        self.X_best = np.zeros(n_dim)
        self.best_score = -float('inf')

    def optimize(self):
        # ประเมินรอบแรก
        for i in range(self.n_agents):
            score = self.fitness_func(self.X[i])
            if score > self.best_score:
                self.best_score = score
                self.X_best = self.X[i].copy()

        for t in range(self.max_iter):
            start_time = time.time()
            for i in range(self.n_agents):
                if np.array_equal(self.X[i], self.X_best): continue

                p = np.random.rand()
                # เคสที่ 1: ปรับตามตัวที่ต่างจาก X_best
                if p < 0.5:
                    differ_idx = np.where(self.X[i] != self.X_best)[0]
                    for idx in differ_idx:
                        if np.random.rand() < 0.7:
                            self.X[i, idx] = 1 - self.X[i, idx]
                # เคสที่ 2: สุ่มเปลี่ยนทุกตำแหน่ง 50/50
                else:
                    for idx in range(self.n_dim):
                        if np.random.rand() < 0.5:
                            self.X[i, idx] = 1 - self.X[i, idx]

            # ประเมินคะแนนหลัง Update
            for i in range(self.n_agents):
                score = self.fitness_func(self.X[i])
                if score > self.best_score:
                    self.best_score = score
                    self.X_best = self.X[i].copy()

            print(f"   -> รอบที่ {t+1:03d}/{self.max_iter} | Best Fitness: {self.best_score:.4f} | เวลา: {time.time()-start_time:.2f} วิ")

        return self.X_best

# --- 1. Setup Data ---
data = load_breast_cancer()
X_full = data.data
y_full = data.target
total_genes = X_full.shape[1]
SEED = 42

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=SEED)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

def get_models():
    return {
        'Decision Tree': DecisionTreeClassifier(random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Perceptron': Perceptron(random_state=SEED),
        'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='ovr')
    }

metrics = ['accuracy', 'recall', 'f1_score', 'precision']

# --- 3. Helper Logic ---
def calculate_custom_score(base_score, num_selected):
    penalty = 1.0 - (num_selected / total_genes)
    return (0.85 * base_score) + (0.15 * penalty)

def get_base_score(model, X_tr, X_te, metric_type):
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    if metric_type == 'accuracy': return accuracy_score(y_test, y_pred)
    elif metric_type == 'recall': return recall_score(y_test, y_pred, average='macro', zero_division=0)
    elif metric_type == 'f1_score': return f1_score(y_test, y_pred, average='macro', zero_division=0)
    elif metric_type == 'precision': return precision_score(y_test, y_pred, average='macro', zero_division=0)
    return 0

# ใช้ KNN เป็นตัวประเมินใน Fitness Function ตามต้นฉบับ
def fitness_factory(metric_type):
    def fitness(pos):
        idx = np.where(pos == 1)[0]
        if len(idx) == 0: return 0.0
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_s[:, idx], y_train)
        y_pred = model.predict(X_test_s[:, idx])
        
        if metric_type == 'accuracy': score = accuracy_score(y_test, y_pred)
        elif metric_type == 'recall': score = recall_score(y_test, y_pred, average='macro', zero_division=0)
        elif metric_type == 'f1_score': score = f1_score(y_test, y_pred, average='macro', zero_division=0)
        elif metric_type == 'precision': score = precision_score(y_test, y_pred, average='macro', zero_division=0)
        
        penalty = 1.0 - (len(idx) / total_genes)
        return (0.85 * score) + (0.15 * penalty)
    return fitness

# --- 4. Main Execution ---
results_list = []

# --- [METHOD: khongrau (ตาม Logic ของคุณ)] ---
print("\n[METHOD: khongrau (ของพวกเรา)]")
for met in metrics:
    print(f"\n🚀 กำลังหาชุดฟีเจอร์ที่ดีที่สุดสำหรับค่า: [ {met.upper()} ]")
    opt = BoardOptimizer(fitness_factory(met), n_agents=15, n_dim=total_genes, max_iter=50)
    best_vector = opt.optimize()
    final_idx = np.where(best_vector == 1)[0]
    
    models = get_models()
    for m_name, model in models.items():
        base = get_base_score(model, X_train_s[:, final_idx], X_test_s[:, final_idx], met)
        final_score = calculate_custom_score(base, len(final_idx))
        print(f"\t- {m_name:22} | Final: {final_score:.4f} | ใช้ฟีเจอร์ ({len(final_idx)} ตัว)")
        results_list.append({'Method': 'khongrau', 'Metric': met, 'Model': m_name, 'Score': final_score})

# (ส่วนของ Method อื่นๆ Correlation, GA, Forward, PCA ใส่ต่อท้ายที่นี่ได้ตามปกติ)
# --- [METHOD: Correlation] ---
print("\n[METHOD: Correlation]")
# ตัดฟีเจอร์ที่ Correlation > 0.9 แล้วให้แต่ละโมเดลประเมินผล
df_temp = pd.DataFrame(X_train_s)
corr_matrix = df_temp.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.70)]
corr_idx = [i for i in range(total_genes) if i not in to_drop]

for met in metrics:
    print(f"\t{met}")
    models = get_models()
    for m_name, model in models.items():
        base = get_base_score(model, X_train_s[:, corr_idx], X_test_s[:, corr_idx], met)
        final = calculate_custom_score(base, len(corr_idx))
        print(f"\t\t- {m_name:22} | Final: {final:.4f} | ใช้ฟีเจอร์ ({len(corr_idx)} ตัว)")
        results_list.append({'Method': 'Correlation', 'Metric': met, 'Model': m_name, 'Score': final})

# --- [METHOD: Genetic Algorithm] [cite: 2] ---
print("\n[METHOD: Genetic Algorithm]")
def run_ga_for_model(model_obj, metric_name):
    POP_SIZE = 10
    GENS = 20
    pop = [[random.randint(0, 1) for _ in range(total_genes)] for _ in range(POP_SIZE)]
    best_ind = pop[0]
    
    for _ in range(GENS):
        scores = []
        for ind in pop:
            idx = [i for i, bit in enumerate(ind) if bit == 1]
            if not idx: 
                scores.append(0)
                continue
            base = get_base_score(model_obj, X_train_s[:, idx], X_test_s[:, idx], metric_name)
            scores.append(calculate_custom_score(base, len(idx)))
        
        best_ind = pop[np.argmax(scores)]
        # Selection & Crossover
        new_pop = [best_ind]
        while len(new_pop) < POP_SIZE:
            p1, p2 = random.choices(pop, k=2)
            cut = random.randint(1, total_genes-1)
            child = p1[:cut] + p2[cut:]
            new_pop.append(child)
        pop = new_pop
    return [i for i, bit in enumerate(best_ind) if bit == 1]

for met in metrics:
    print(f"\t{met}")
    models = get_models()
    for m_name, model in models.items():
        best_idx = run_ga_for_model(model, met)
        base = get_base_score(model, X_train_s[:, best_idx], X_test_s[:, best_idx], met)
        final = calculate_custom_score(base, len(best_idx))
        print(f"\t\t- {m_name:22} | Final: {final:.4f} | ใช้ฟีเจอร์ ({len(best_idx)} ตัว)")
        results_list.append({'Method': 'Genetic', 'Metric': met, 'Model': m_name, 'Score': final})


# --- [METHOD: Forward Selection] ---
print("\n[METHOD: Forward Selection]")
for met in metrics:
    print(f"\t{met}")
    models = get_models()
    for m_name, model in models.items():
        current_features = []
        best_score_for_model = 0
        remaining_features = list(range(total_genes))
        
        # วนลูปเพิ่มฟีเจอร์ทีละตัวตามหลัก Forward Selection
        while len(remaining_features) > 0:
            scores_with_new_feature = []
            for f in remaining_features:
                temp_features = current_features + [f]
                score = get_base_score(model, X_train_s[:, temp_features], X_test_s[:, temp_features], met)
                scores_with_new_feature.append((score, f))
            
            # หาฟีเจอร์ที่เพิ่มเข้ามาแล้วทำให้คะแนนดีที่สุดในรอบนั้น
            scores_with_new_feature.sort(reverse=True)
            best_new_score, best_feature = scores_with_new_feature[0]
            
            # เงื่อนไขการหยุด: 
            # 1. ถ้าคะแนนถึง 0.85 ตามที่กำหนด
            # 2. หรือถ้าเพิ่มแล้วคะแนนไม่ดีขึ้นกว่าเดิม (ป้องกัน Loop ไม่สิ้นสุด)
            if best_new_score >= 0.97 or best_new_score <= best_score_for_model:
                if best_new_score >= 0.97:
                    current_features.append(best_feature)
                    best_score_for_model = best_new_score
                break
            
            current_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_score_for_model = best_new_score

        # แสดงผลลัพธ์โดยใช้คะแนนดิบ (No Penalty)
        print(f"\t\t- {m_name:22} | Score: {best_score_for_model:.4f} | ใช้ฟีเจอร์ ({len(current_features)} ตัว)")
        results_list.append({'Method': 'Forward', 'Metric': met, 'Model': m_name, 'Score': best_score_for_model})
# --- [METHOD: PCA]  ---
print("\n[METHOD: PCA]")
for met in metrics:
    print(f"\t{met}")
    models = get_models()
    for m_name, model in models.items():
        best_pca_score = -1
        best_comp_count = 0
        # ลองหลายค่า Variance เพื่อให้จำนวน Component ของแต่ละโมเดลไม่เท่ากัน
        for var in [0.90]:
            pca = PCA(n_components=var)
            X_tr_p = pca.fit_transform(X_train_s)
            X_te_p = pca.transform(X_test_s)
            n_comp = X_tr_p.shape[1]
            
            base = get_base_score(model, X_tr_p, X_te_p, met)
            final = calculate_custom_score(base, n_comp)
            
            if final > best_pca_score:
                best_pca_score = final
                best_comp_count = n_comp
        
        print(f"\t\t- {m_name:22} | Final: {best_pca_score:.4f} | ใช้ฟีเจอร์ ({best_comp_count} Components)")
        results_list.append({'Method': 'PCA', 'Metric': met, 'Model': m_name, 'Score': best_pca_score})

# --- Summary ---
print(f"\n{'='*30}\n    Best Models per Metric\n{'='*30}")
final_df = pd.DataFrame(results_list)
for met in metrics:
    best = final_df[final_df['Metric'] == met].sort_values(by='Score', ascending=False).iloc[0]
    print(f"BEST {met.upper():14} Model: {best['Model']:18} | Method: {best['Method']:12}  ได้ค่า : {best['Score']:.4f}")