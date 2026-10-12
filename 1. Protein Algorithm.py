"""CBAT-Net: Protein Secondary Structure Prediction algorithm viewer."""

import tkinter as tk
from tkinter import ttk

class CBATNetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CBAT-Net: Protein Secondary Structure Prediction - Analysis Dashboard")
        self.root.geometry("1400x900")
        self.root.state('zoomed')  # Maximize window
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#3498db',
            'secondary': '#e74c3c',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'purple': '#9b59b6',
            'dark': '#2c3e50'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup main UI components."""
        title_frame = tk.Frame(self.root, bg=self.colors['dark'], height=80)
        title_frame.pack(fill='x', padx=0, pady=0)
        title_frame.pack_propagate(False)

        title_label = tk.Label(
            title_frame,
            text="CBAT-Net: CNN-BiLSTM-Transformer Architecture",
            font=('Arial', 24, 'bold'),
            bg=self.colors['dark'],
            fg='white'
        )
        title_label.pack(pady=15)

        subtitle_label = tk.Label(
            title_frame,
            text="Joint Q3/Q8 Protein Secondary Structure Prediction",
            font=('Arial', 12),
            bg=self.colors['dark'],
            fg='#ecf0f1'
        )
        subtitle_label.pack()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        self.create_algorithm_tab()
        
    def create_algorithm_tab(self):
        """Create tab with mathematical algorithms"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='📐 Mathematical Algorithms')
        
        # Create scrolled text widget
        text_frame = tk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        text_widget = tk.Text(
            text_frame,
            wrap='word',
            font=('Courier New', 10),
            yscrollcommand=scrollbar.set,
            bg='#f8f9fa',
            padx=15,
            pady=15
        )
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insert algorithms
        algorithms = self.get_algorithm_text()
        text_widget.insert('1.0', algorithms)
        text_widget.config(state='disabled')  # Make read-only
        
        # Configure tags for formatting
        text_widget.tag_config('title', font=('Arial', 14, 'bold'), foreground=self.colors['dark'])
        text_widget.tag_config('subtitle', font=('Arial', 12, 'bold'), foreground=self.colors['primary'])
        text_widget.tag_config('equation', font=('Courier New', 11), background='#e8f4f8')
        
    def get_algorithm_text(self):
        """Return formatted algorithm text"""
        return """
═══════════════════════════════════════════════════════════════════════════════
                    CBAT-NET MATHEMATICAL FORMULATIONS
═══════════════════════════════════════════════════════════════════════════════

1. CONVOLUTIONAL NEURAL NETWORK (CNN) COMPONENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Multi-scale convolution for local feature extraction:

    H^CNN_k = ReLU(Conv1D_k(X) + b_k),  k ∈ {3, 5, 7, 9}

Detailed computation:

    h^CNN_{i,j,k} = σ(Σ_{m=0}^{k-1} Σ_{d=0}^{D_in-1} W^CNN_{k,j,d,m} · X_{i+m-⌊k/2⌋,d} + b^CNN_{k,j})

Where:
    - i: position index (1 to L, sequence length)
    - j: filter index
    - k: kernel size
    - d: input dimension index
    - σ: ReLU activation function
    - W^CNN: convolutional weights
    - X: input features (L × 77)

Concatenation:

    H^CNN = Concat[H^CNN_3, H^CNN_5, H^CNN_7, H^CNN_9] ∈ ℝ^{L×256}

Parameter count:

    Θ_CNN = Σ_k (k × D_in × 64 + 64) = (3+5+7+9) × 77 × 64 + 256 = 118,912


2. BIDIRECTIONAL LSTM (BiLSTM) COMPONENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Forward LSTM:

    h⃗_t = LSTM(H^CNN_t, h⃗_{t-1}; θ_fwd)

Backward LSTM:

    h⃖_t = LSTM(H^CNN_t, h⃖_{t+1}; θ_bwd)

LSTM Cell Operations (forward direction):

    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)           [Forget Gate]
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)           [Input Gate]
    c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)        [Candidate Cell]
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t               [Cell State Update]
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)           [Output Gate]
    h_t = o_t ⊙ tanh(c_t)                         [Hidden State]

Bidirectional concatenation:

    H^BiLSTM_t = [h⃗_t ; h⃖_t] ∈ ℝ^512

Parameter count:

    Θ_BiLSTM = 2 × 4 × (256 × 512 + 256 × 256 + 256) = 1,576,960


3. MULTI-HEAD SELF-ATTENTION TRANSFORMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Multi-head attention with N_h = 8 heads:

    MultiHead(Q, K, V) = Concat(head_1, ..., head_8)W^O

Individual attention head:

    head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)

Scaled Dot-Product Attention:

    Attention(Q, K, V) = softmax(QK^T / √d_k)V

Where d_k = 512/8 = 64 (dimension per head)

Attention weight computation:

    α_{ij} = exp(q_i^T k_j / √d_k) / Σ_{j'=1}^L exp(q_i^T k_{j'} / √d_k)

Context vector:

    z_i = Σ_{j=1}^L α_{ij} v_j

Feed-Forward Network:

    FFN(x) = ReLU(xW_1 + b_1)W_2 + b_2

With residual connections and layer normalization:

    H^Attn = LayerNorm(H^BiLSTM + MultiHead(H^BiLSTM, H^BiLSTM, H^BiLSTM))
    H^Trans = LayerNorm(H^Attn + FFN(H^Attn))

Parameter count:

    Θ_Trans = 3 × (512 × 512) + 512 × 2048 + 2048 × 512 = 2,885,632


4. PREDICTION HEADS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q3 Prediction (3 classes: Helix, Strand, Coil):

    P^Q3_i = softmax(W^Q3 H^Trans_i + b^Q3) ∈ ℝ^3

Q8 Prediction (8 classes: H, G, I, E, B, T, S, C):

    P^Q8_i = softmax(W^Q8 H^Trans_i + b^Q8) ∈ ℝ^8

Softmax computation:

    P^Q3_{i,c} = exp(z^Q3_{i,c}) / Σ_{c'=1}^3 exp(z^Q3_{i,c'})

    P^Q8_{i,c} = exp(z^Q8_{i,c}) / Σ_{c'=1}^8 exp(z^Q8_{i,c'})


5. ADAPTIVE DUAL-TASK LOSS FUNCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total loss with uncertainty weighting:

    L_total = (1/σ²_Q3)L_Q3 + (1/σ²_Q8)L_Q8 + log(σ²_Q3) + log(σ²_Q8) + λL_reg

Cross-entropy losses:

    L_Q3 = -(1/L) Σ_{i=1}^L Σ_{c=1}^3 y^Q3_{i,c} log(P^Q3_{i,c})

    L_Q8 = -(1/L) Σ_{i=1}^L Σ_{c=1}^8 y^Q8_{i,c} log(P^Q8_{i,c})

Focal loss variant (for class imbalance):

    L_focal = -(1/L) Σ_{i=1}^L Σ_{c=1}^C α_c(1-P_{i,c})^γ y_{i,c} log(P_{i,c})

Where:
    - α_c = 1/f_c (inverse frequency weight)
    - γ = 2 (focusing parameter)

Regularization:

    L_reg = (λ_1/2)||W||²_2 + λ_2 Σ_l KL(P^Q3_l || P^Q8_{mapped,l})


6. GRADIENT FLOW (BACKPROPAGATION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Output layer gradient:

    ∂L/∂z^Q3_{i,c} = P^Q3_{i,c} - y^Q3_{i,c}

Weight gradient:

    ∂L/∂W^Q3 = Σ_{i=1}^L (P^Q3_i - y^Q3_i) ⊗ H^Trans_i

Attention gradient:

    ∂L/∂Q = (∂L/∂Z_attn)(∂Z_attn/∂A)(∂A/∂Q)

    ∂A/∂Q = (1/√d_k)(A ⊙ (1-A))K^T

BiLSTM gradient (Backpropagation Through Time):

    ∂L/∂h_t = (∂L/∂h_{t+1})(∂h_{t+1}/∂h_t) + ∂L/∂o_t

    ∂L/∂c_t = (∂L/∂h_t) ⊙ o_t ⊙ (1-tanh²(c_t)) + (∂L/∂c_{t+1}) ⊙ f_{t+1}

CNN gradient:

    ∂L/∂W^CNN_{k,j,d,m} = Σ_{i=1}^L (∂L/∂h^CNN_{i,j,k}) · σ'(·) · X_{i+m-⌊k/2⌋,d}


7. ADAM OPTIMIZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Learning rate schedule with warmup:

    η_t = η_base × min(1, t/t_warmup) × 1/√max(t - t_warmup, 1)

Parameter updates:

    m_t = β_1 m_{t-1} + (1-β_1)g_t                [First moment]
    v_t = β_2 v_{t-1} + (1-β_2)g²_t               [Second moment]

    m̂_t = m_t / (1-β₁^t)                          [Bias correction]
    v̂_t = v_t / (1-β₂^t)

    θ_{t+1} = θ_t - η_t m̂_t / (√v̂_t + ε)        [Update rule]

Hyperparameters:
    - β_1 = 0.9 (exponential decay for first moment)
    - β_2 = 0.999 (exponential decay for second moment)
    - ε = 10^{-8} (numerical stability)
    - η_base = 0.001 (base learning rate)


8. EVALUATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q_n Accuracy (n ∈ {3,8}):

    Q_n = (1/N_test) Σ_{i=1}^{N_test} (Σ_{j=1}^{L_i} 𝟙[y_{ij} = ŷ_{ij}] / L_i) × 100%

Matthews Correlation Coefficient:

    MCC_c = (TP_c·TN_c - FP_c·FN_c) / √((TP_c+FP_c)(TP_c+FN_c)(TN_c+FP_c)(TN_c+FN_c))

Segment Overlap (SOV):

    SOV = (100/N_obs) Σ_{s∈S} [min(len(s_1),len(s_2)) + δ(s)] / max(len(s_1),len(s_2)) × len(s_1)


9. MODEL COMPLEXITY
━━━━━════════════════════════════════════════════════════════════════════

Total Parameters:

    Θ_total = Θ_CNN + Θ_BiLSTM + Θ_Trans + Θ_heads
            = 118,912 + 1,576,960 + 2,885,632 + 5,651
            = 4,587,155 ≈ 4.6M parameters

FLOPs (per sequence of length L):

    FLOPs ≈ 2L(Θ_CNN) + 8LΘ_BiLSTM + L²d_model + 2Ld²_FFN
         ≈ 2L(119K) + 8L(1.58M) + L²(512) + 2L(2048²)
         ≈ O(L²) for attention, O(L) otherwise


10. CONVERGENCE GUARANTEE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Under assumptions:
    1. L is L-smooth: ||∇L(θ₁) - ∇L(θ₂)|| ≤ L||θ₁ - θ₂||
    2. Bounded gradients: E[||g_t||²] ≤ G²
    3. Unbiased gradient estimates: E[g_t] = ∇L(θ_t)

Adam converges at rate:

    E[||∇L(θ_T)||²] ≤ O(1/√T)

Where T is the number of iterations.

═══════════════════════════════════════════════════════════════════════════════
                            END OF MATHEMATICAL FORMULATIONS
═══════════════════════════════════════════════════════════════════════════════
"""


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = CBATNetGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()