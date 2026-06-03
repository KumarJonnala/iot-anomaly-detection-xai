ZSCORE_THRESHOLD = 3.0
IF_CONTAMINATION = 0.034    # matches known AI4I failure rate
AE_PERCENTILE    = 95       # p95 of normal-row reconstruction errors → flag threshold
AE_LATENT_DIM    = 4
ROLLING_WINDOW   = 50       # readings for dynamic Z-score baseline
SENSOR_WINDOW    = 10       # ±N readings for context assembly
FUSION_THRESHOLD = 0.5
FUSION_WEIGHTS   = (0.3, 0.4, 0.3)   # z-score, isolation forest, autoencoder
