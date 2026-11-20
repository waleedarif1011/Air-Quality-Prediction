# Hugging Face Space deploy

The Gradio app is Space-ready. A live Space was **not** created from this agent run because no Hugging Face write token (`HF_TOKEN`) was available in the environment.

## Exact steps (Waleed)

1. Create a token at https://huggingface.co/settings/tokens with **Write** + Spaces access.
2. Locally (or in CI):

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login   # paste the token
huggingface-cli repo create air-quality-prediction --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/<YOUR_HF_USER>/air-quality-prediction
git push space main
```

3. Confirm the Space builds and the three presets behave as expected:
   - Clean coastal day → Good / Moderate
   - Rush-hour smog → elevated AQI
   - Wildfire smoke event → Unhealthy+ category
4. Update the Live Demo badge URL in `README.md` if the Space slug differs from `waleedarif1011/air-quality-prediction`.

## Files the Space needs

- `app.py`, `requirements.txt`, `README.md` (YAML frontmatter)
- `src/`
- `artifacts/` (`aqi_model.joblib`, `metrics.json`)
- `assets/` (feature importance plot used in the UI)
