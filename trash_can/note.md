## 數學式
### 1. 定義符號
* $\theta$ : CLIP pre-trained model
* $\delta$ : Universal Adversarial Perturbation (UAP)
* $\mathcal{L}$ : loss function (cross-entropy)
* $\mathcal{D}_o$ : data owner's online dataset, 被 model owner 學走
* $\mathcal{D}_l$ : data owner's local dataset, 還沒被 model owner 學走, 和 $\mathcal{D}_o$ 相同 class
* $\mathcal{D}_{s1}$ : data owner's surrogate dataset
* $\mathcal{D}_{s2}$ : data owner's surrogate dataset, 和 $\mathcal{D}_{s1}$ 相同 classes
* $\mathcal{D}_{other1}$ : model owner's dataset
* $\mathcal{D}_{other2}$ : model owner's dataset, 和 $\mathcal{D}_{other1}$ 相同 classes

### 2. 模型定義
* $\theta_{s1}$=$\theta(\mathcal{D}_o \cup \mathcal{D}_{s1})$
* $\theta_{s2}$=$\theta_{s1}(\mathcal{D}_l + \delta \cup \mathcal{D}_{s2})$
* $\theta_1$=$\theta(\mathcal{D}_o \cup \mathcal{D}_{other1})$
* $\theta_2$=$\theta_1(\mathcal{D}_l + \delta \cup \mathcal{D}_{other2})$

### 3. 雙層優化
$$
\delta^{\star} = \max_{\delta} L(\mathcal{D}_{o} \cup \mathcal{D}_{l} + \delta, \theta^{\star}) \quad s.t. \| \delta^{\star} \|_p \le \epsilon
$$
$$
\theta^{\star} = \argmin_{\theta} L(\mathcal{D}_{s2} \cup  \mathcal{D}_{l} +  \delta, \theta)
$$