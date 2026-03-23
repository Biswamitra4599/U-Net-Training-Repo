----------------------------------------------------------------------

_main.png — Per-Branch Processing and Fusion

This visualization provides a complete view of how each domain (Raw, Fourier, Wavelet) is processed and contributes to the final prediction. For each branch, it shows the original input, the encoder plus self-attention output, the spatial self-attention map (indicating where the model focuses), the learned gate values (indicating how important that domain is relative to others), and the gated output (the actual contribution that flows into the final fused representation). Together, this figure explains both where the model looks within each domain and how strongly each domain influences the final decision, making it the central interpretability visualization of the architecture.

----------------------------------------------------------------------

_main.png — Per-Branch Processing and Fusion

This visualization provides a complete view of how each domain (Raw, Fourier, Wavelet) is processed and contributes to the final prediction. For each branch, it shows the original input, the encoder plus self-attention output, the spatial self-attention map (indicating where the model focuses), the learned gate values (indicating how important that domain is relative to others), and the gated output (the actual contribution that flows into the final fused representation). Together, this figure explains both where the model looks within each domain and how strongly each domain influences the final decision, making it the central interpretability visualization of the architecture.

----------------------------------------------------------------------

_overlay.png — Spatial Attention Overlay

This visualization overlays the self-attention map on top of the original input image for each domain. The grayscale image shows the original SAR scene, while the colored heatmap highlights regions receiving high attention from the model. This provides an intuitive and spatially grounded explanation of the model’s focus, allowing one to visually identify the parts of the image that most strongly influence the representation learned by the attention mechanism.

----------------------------------------------------------------------

_rgb.png — Cross-Domain Contribution Composite

This visualization combines the outputs of the three domains into a single RGB image where each channel represents one domain (Red = Raw, Green = Fourier, Blue = Wavelet). The resulting composite shows which domain contributes most strongly at each spatial location: pure colors indicate dominance of a single domain, while mixed colors indicate joint contribution. This plot provides a spatial view of domain dominance and complementarity, illustrating how different signal representations contribute across the scene.

----------------------------------------------------------------------

summary/ outputs — Class-wise Domain Importance

The summary plots aggregate information across all samples to provide dataset-level insights. The per-class bar plots show the average gate values for each domain within each class, while the class-domain heatmap visualizes the relative importance of Raw, Fourier, and Wavelet representations across all classes. These summaries reveal systematic patterns in domain usage (e.g., certain classes relying more heavily on a particular domain), providing a high-level understanding of how the model leverages multi-domain information for classification.

----------------------------------------------------------------------