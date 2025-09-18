# botwenger

![botwenger logo](https://github.com/diegitpush/botwenger/blob/main/botwenger.jpg?raw=true)

**botwenger** is a Python repository implementing an end-to-end machine learning workflow to make recommendations for the football fantasy game [Biwenger](https://biwenger.as.com/). It includes:

- **Preprocessing:** raw data loading, cleaning, interpolating/filling missing data and preprocessing. The raw data has been sourced using the [pybiwenger](https://github.com/pablominue/pybiwenger) library, which is also used to get data for daily inference

- **Feature engineering:** extracting and constructing informative features from the preprocessed data. There are different pipelines for training and inference, as the source data varies slightly for them

- **Model training:** training and optimizing different XGBoost models to predict future points for a player (as an average for match sequences of different lengths).

- **Prediction:** generating outputs from this trained model. The used one predicts the points average for the next three matches

- **Post-processing with PuLP:** using a Mixed Integer Linear Programming (MILP) solver to get the final output, solving the problem of maximizing points with a given budget and set of players, or minimizing budget with a given minimum expected points constraint

- **Posting output:** using a Telegram Bot to post the recommendations as daily chat messages

---

## How to use it

You need to have filled these five environment variables:

- **TELEGRAM_BOT_TOKEN:** A bot token generated on Telegram (using BotFather)
- **TELEGRAM_CHAT_ID:** The chat ID of your conversation with the bot (can be obtained from Telegram API, check their documentation)
- **BIWENGER_USERNAME:** Your Biwenger username
- **BIWENGER_PASSWORD:** Your Biwenger password
- **BIWENGER_LEAGUE:** Your Biwenger league name

After setting up and activating the conda environment, call the endpoint:

```python botwenger/modeling/predict.py
```
A message (in Spanish, check code) should appear in the Telegram Bot chat with the info and recommendations

---

## Model info (for avg of 3 future matches)

- **Test Set RMSE:** 2.0179
- **Test Set R2:** 0.3647

- **SHAP Features Summary:**

![SHAP Fetaures Summary](https://github.com/diegitpush/botwenger/blob/main/reports/figures/model_3_shap_summary.png?raw=true)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This library is not affiliated with Biwenger or any of its parent companies. Use at your own risk and respect Biwenger's terms of service. Also, I made this for fun