# services/ai_models.py
"""
Handles AI/ML model integration for signal filtering and market regime detection.
Includes functions for model training (placeholders), prediction, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter # To check class counts for stratification
# import xgboost as xgb # Uncomment when XGBoost is fully implemented

from utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

class AIModelService:
    """
    A service class to encapsulate AI/ML model functionalities.
    """
    def __init__(self):
        self.signal_filter_model = None
        self.regime_detection_model = None # Placeholder for future regime model
        self.feature_scaler = StandardScaler() # For scaling features before model prediction
        self.trained_feature_names = None # To store feature names used during training

    def _prepare_features_for_signal_filtering(self, price_data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares features from price data and raw signals for the signal filtering model.
        This is a crucial step and will likely need extensive customization.
        """
        if signals.empty or price_data.empty:
            logger.warning("AIModelService: Empty signals or price data provided for feature preparation.")
            return pd.DataFrame()

        # Ensure signals DataFrame has 'SignalTime' as a DatetimeIndex or a column that can be converted
        if not isinstance(signals.index, pd.DatetimeIndex):
            if 'SignalTime' in signals.columns:
                try:
                    signals['SignalTime'] = pd.to_datetime(signals['SignalTime'])
                    signals = signals.set_index('SignalTime', drop=False) # Keep SignalTime as column too
                except Exception as e:
                    logger.error(f"AIModelService: Could not convert 'SignalTime' column to DatetimeIndex: {e}", exc_info=True)
                    return pd.DataFrame()
            else:
                logger.error("AIModelService: Signals DataFrame must have 'SignalTime' as index or column.")
                return pd.DataFrame()
        
        # Ensure price_data has a DatetimeIndex
        if not isinstance(price_data.index, pd.DatetimeIndex):
            logger.error("AIModelService: price_data must have a DatetimeIndex.")
            return pd.DataFrame()
            
        # Ensure timezones are consistent or handled (assuming NYT for both as per project standard)
        # This should ideally be enforced before this function.
        # For robustness, let's check and attempt conversion if naive.
        try:
            if price_data.index.tz is None:
                price_data.index = price_data.index.tz_localize(settings.NY_TIMEZONE_STR)
            elif str(price_data.index.tz) != settings.NY_TIMEZONE_STR:
                price_data.index = price_data.index.tz_convert(settings.NY_TIMEZONE_STR)

            if signals.index.tz is None:
                signals.index = signals.index.tz_localize(settings.NY_TIMEZONE_STR)
            elif str(signals.index.tz) != settings.NY_TIMEZONE_STR:
                signals.index = signals.index.tz_convert(settings.NY_TIMEZONE_STR)
        except Exception as tz_err:
            logger.error(f"AIModelService: Timezone conversion/localization error: {tz_err}", exc_info=True)
            return pd.DataFrame()


        features_list = []
        # Placeholder: Example features. This needs significant expansion.
        # Pre-calculate indicators on price_data if they are complex, for efficiency.
        # Example: price_data['SMA_50'] = price_data['Close'].rolling(window=50).mean()

        for signal_time, signal_row in signals.iterrows():
            try:
                # Find the bar in price_data corresponding to the signal time
                # Use asof for exact or preceding match if price_data is high-frequency
                bar_data_at_signal_time_index = price_data.index.asof(signal_time)
                if pd.isna(bar_data_at_signal_time_index):
                    logger.debug(f"No matching bar in price_data for signal at {signal_time}. Skipping.")
                    features_list.append({'SignalTime': signal_time}) # Add placeholder to maintain index alignment
                    continue
                
                bar_data_at_signal = price_data.loc[bar_data_at_signal_time_index]

                if bar_data_at_signal.isnull().any(): # Check for NaNs in the specific bar
                    logger.debug(f"NaNs found in bar data for signal at {signal_time}. Bar: {bar_data_at_signal_time_index}. Skipping feature gen for this signal.")
                    features_list.append({'SignalTime': signal_time})
                    continue

                # Example features (very basic, expand significantly):
                bar_range = bar_data_at_signal['High'] - bar_data_at_signal['Low']
                close_position_in_range = (bar_data_at_signal['Close'] - bar_data_at_signal['Low']) / bar_range if bar_range > 0 else 0.5
                
                # Add more features:
                # - Volatility (e.g., ATR at signal time)
                # - Momentum (e.g., RSI, MACD at signal time)
                # - Price relative to MAs (e.g., Close - SMA_50)
                # - Time-based features (e.g., hour of day, day of week)
                # - Signal properties (e.g., if signal type is long/short, this could be a feature if model is generic)

                features_list.append({
                    'SignalTime': signal_time, # Keep for merging/alignment
                    'bar_range': bar_range,
                    'close_position_in_range': close_position_in_range,
                    # 'ATR_14': price_data.loc[bar_data_at_signal_time_index, 'ATR_14'], # Example if ATR is precalculated
                })
            except KeyError as ke: # Specific error for missing columns in price_data
                logger.warning(f"AIModelService: KeyError generating features for signal at {signal_time}: {ke}. Check precalculated indicators.")
                features_list.append({'SignalTime': signal_time})
            except Exception as e:
                logger.error(f"AIModelService: Error generating features for signal at {signal_time}: {e}", exc_info=True)
                features_list.append({'SignalTime': signal_time}) # Add placeholder for alignment

        if not features_list:
            logger.warning("AIModelService: No features were generated.")
            return pd.DataFrame()

        features_df = pd.DataFrame(features_list)
        if 'SignalTime' not in features_df.columns: # Should not happen if logic above is correct
            logger.error("AIModelService: 'SignalTime' column missing from features_df after generation.")
            return pd.DataFrame()
            
        features_df['SignalTime'] = pd.to_datetime(features_df['SignalTime'])
        features_df = features_df.set_index('SignalTime')
        
        # Drop rows that might be all NaN (if placeholders were added and no actual features)
        features_df.dropna(how='all', subset=[col for col in features_df.columns if col != 'SignalTime'], inplace=True)

        # Impute NaNs - simple ffill then 0 for remaining. Consider more sophisticated imputation.
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(0, inplace=True)
        
        logger.info(f"AIModelService: Prepared {len(features_df)} feature sets for signal filtering.")
        return features_df

    def train_signal_filter_model(self, features: pd.DataFrame, labels: pd.Series, model_type: str = "RandomForest"):
        """
        Trains a signal filtering model (placeholder logic).
        """
        if features.empty or labels.empty:
            logger.error("AIModelService: Features or labels are empty. Cannot train model.")
            return
        if len(features) != len(labels):
            logger.error(f"AIModelService: Mismatch in length of features ({len(features)}) and labels ({len(labels)}). Cannot train.")
            return

        # Check class distribution for stratification
        label_counts = Counter(labels)
        min_class_count = min(label_counts.values()) if label_counts else 0
        
        stratify_param = labels
        if min_class_count < 2: # n_splits for train_test_split is effectively 2 for a single split
            logger.warning(
                f"AIModelService: The least populated class in labels has only {min_class_count} member(s), "
                f"which is too few for stratification. Performing train/test split without stratification. "
                f"Label counts: {label_counts}"
            )
            stratify_param = None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=stratify_param
            )
        except ValueError as e:
            logger.error(f"AIModelService: Error during train_test_split, possibly due to insufficient samples after all checks: {e}. "
                         f"Features shape: {features.shape}, Labels shape: {labels.shape}, Label counts: {Counter(labels)}. "
                         f"Attempting split without stratification if not already tried.")
            if stratify_param is not None: # If stratification was attempted and failed
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        features, labels, test_size=0.2, random_state=42, stratify=None
                    )
                    logger.info("AIModelService: Successfully split data without stratification after initial error.")
                except Exception as e_nostrat:
                    logger.error(f"AIModelService: Failed to split data even without stratification: {e_nostrat}. Cannot train model.")
                    return
            else: # Stratification was already None, or some other ValueError
                logger.error("AIModelService: Cannot train model due to train_test_split error.")
                return


        # Scale features
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        # X_test_scaled = self.feature_scaler.transform(X_test) # For evaluation

        self.trained_feature_names = X_train.columns.tolist() # Store feature names

        if model_type == "RandomForest":
            self.signal_filter_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        # elif model_type == "XGBoost":
        #     self.signal_filter_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        else:
            logger.error(f"AIModelService: Unsupported model type '{model_type}'. Using RandomForest as default.")
            self.signal_filter_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

        try:
            self.signal_filter_model.fit(X_train_scaled, y_train)
            logger.info(f"AIModelService: Signal filter model ({model_type}) trained successfully.")
            # Placeholder: Evaluate model on X_test_scaled, y_test
            # accuracy = self.signal_filter_model.score(X_test_scaled, y_test)
            # logger.info(f"AIModelService: Model accuracy on test set: {accuracy:.2f}")
        except Exception as e:
            logger.error(f"AIModelService: Error fitting the model {model_type}: {e}", exc_info=True)
            self.signal_filter_model = None # Reset model if fitting fails

    def filter_signals(self, price_data: pd.DataFrame, raw_signals: pd.DataFrame, model_type_used_for_training: str) -> pd.DataFrame:
        """
        Filters raw signals using the trained AI model.
        """
        if self.signal_filter_model is None:
            logger.warning("AIModelService: Signal filter model is not trained. Returning raw signals.")
            return raw_signals
        if raw_signals.empty:
            logger.info("AIModelService: No raw signals to filter.")
            return pd.DataFrame()
        if self.trained_feature_names is None:
            logger.error("AIModelService: Model was trained, but feature names were not stored. Cannot filter.")
            return raw_signals

        logger.info(f"AIModelService: Filtering signals using {model_type_used_for_training} model...")
        
        features_for_prediction = self._prepare_features_for_signal_filtering(price_data, raw_signals.copy())
        
        if features_for_prediction.empty:
            logger.warning("AIModelService: No features generated for the provided raw signals. Returning raw signals.")
            return raw_signals

        # Align columns with training features
        try:
            # Ensure all trained feature names are present, fill missing with 0 (or a more sophisticated imputation)
            for col in self.trained_feature_names:
                if col not in features_for_prediction.columns:
                    features_for_prediction[col] = 0 
                    logger.warning(f"AIModelService: Feature '{col}' (used in training) not found in prediction features. Added with value 0.")
            
            features_for_prediction_aligned = features_for_prediction[self.trained_feature_names]
        except KeyError as e:
            logger.error(f"AIModelService: Error aligning features for prediction. Missing feature: {e}. Trained features: {self.trained_feature_names}. Prediction features: {features_for_prediction.columns.tolist()}", exc_info=True)
            return raw_signals # Or handle more gracefully

        try:
            scaled_features = self.feature_scaler.transform(features_for_prediction_aligned)
            predictions = self.signal_filter_model.predict(scaled_features) # Returns 0 or 1
            
            # Create a Series for predictions with the same index as features_for_prediction_aligned
            predictions_series = pd.Series(predictions, index=features_for_prediction_aligned.index, name='prediction')
            
            # Filter signals: Keep signals where prediction is 1 (or your "good signal" class)
            # Need to merge predictions back to raw_signals. Both should have SignalTime index.
            
            # Ensure raw_signals has 'SignalTime' as index if it's not already
            if not isinstance(raw_signals.index, pd.DatetimeIndex) and 'SignalTime' in raw_signals.columns:
                raw_signals_indexed = raw_signals.set_index(pd.to_datetime(raw_signals['SignalTime']))
            elif isinstance(raw_signals.index, pd.DatetimeIndex):
                raw_signals_indexed = raw_signals
            else:
                logger.error("AIModelService: Raw signals for filtering do not have a usable DatetimeIndex or 'SignalTime' column.")
                return raw_signals # Return raw if cannot merge

            # Join predictions with the original signals
            # The index of predictions_series is SignalTime from features_for_prediction_aligned
            merged_signals = raw_signals_indexed.join(predictions_series, how='left')
            
            # If a signal didn't have features (e.g., due to data issues), its prediction might be NaN.
            # Decide how to handle these: exclude them or include them. For now, exclude.
            passing_signals = merged_signals[merged_signals['prediction'] == 1].copy()
            passing_signals.drop(columns=['prediction'], inplace=True, errors='ignore') # Clean up

            logger.info(f"AIModelService: Original signals: {len(raw_signals)}, Filtered signals: {len(passing_signals)}.")
            return passing_signals

        except Exception as e:
            logger.error(f"AIModelService: Error during signal filtering prediction: {e}", exc_info=True)
            return raw_signals # Fallback to raw signals on error

    # --- Placeholder for Regime Detection ---
    def _prepare_features_for_regime_detection(self, price_data: pd.DataFrame) -> pd.DataFrame:
        if price_data.empty: return pd.DataFrame()
        features = pd.DataFrame(index=price_data.index)
        # Example: ATR for volatility regime
        # features['ATR_14'] = ta.atr(price_data['High'], price_data['Low'], price_data['Close'], length=14)
        features['Close_Diff_Std'] = price_data['Close'].diff().rolling(window=20).std() # Example
        features.fillna(method='ffill', inplace=True)
        features.fillna(0, inplace=True)
        return features

    def train_regime_detection_model(self, features: pd.DataFrame, labels: pd.Series):
        logger.warning("AIModelService: Regime detection model training is a placeholder and not fully implemented.")
        # Placeholder: Train a clustering (e.g., KMeans) or classification model
        # self.regime_detection_model = SomeRegimeModel()
        # self.regime_detection_model.fit(features, labels) # Or just features for unsupervised
        pass

    def detect_regime(self, current_price_data_segment: pd.DataFrame) -> str:
        if self.regime_detection_model is None:
            # logger.debug("AIModelService: Regime detection model not trained.")
            return "Unknown" # Default if no model
        
        features = self._prepare_features_for_regime_detection(current_price_data_segment)
        if features.empty:
            return "Unknown"
        
        # Use the latest features for prediction
        latest_features = features.iloc[-1:] 
        # scaled_latest_features = self.feature_scaler_regime.transform(latest_features) # Separate scaler if needed
        # regime_prediction = self.regime_detection_model.predict(scaled_latest_features)
        # return str(regime_prediction[0]) # Convert to string or map to label
        logger.warning("AIModelService: Regime detection prediction is a placeholder.")
        return "Trending (Placeholder)"


# --- Main execution block for testing ---
if __name__ == '__main__':
    logger.info("Testing AIModelService...")
    ai_service = AIModelService()

    # Create dummy price data
    date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='H', tz=settings.NY_TIMEZONE_STR)
    dummy_price_data = pd.DataFrame(date_rng, columns=['SignalTime'])
    dummy_price_data['Open'] = np.random.rand(len(date_rng)) * 100 + 100
    dummy_price_data['High'] = dummy_price_data['Open'] + np.random.rand(len(date_rng)) * 10
    dummy_price_data['Low'] = dummy_price_data['Open'] - np.random.rand(len(date_rng)) * 10
    dummy_price_data['Close'] = dummy_price_data['Low'] + np.random.rand(len(date_rng)) * (dummy_price_data['High'] - dummy_price_data['Low'])
    dummy_price_data['Volume'] = np.random.randint(1000, 10000, size=len(date_rng))
    dummy_price_data = dummy_price_data.set_index('SignalTime')

    # Create dummy signals
    dummy_signals_list = []
    for i in range(0, len(dummy_price_data), 5): # Create a signal every 5 bars
        ts = dummy_price_data.index[i]
        dummy_signals_list.append({
            'SignalTime': ts,
            'SignalType': 'Long' if i % 2 == 0 else 'Short',
            'EntryPrice': dummy_price_data['Close'].iloc[i],
            'SL': dummy_price_data['Close'].iloc[i] - 5,
            'TP': dummy_price_data['Close'].iloc[i] + 15
        })
    dummy_signals_for_filter = pd.DataFrame(dummy_signals_list)
    if not dummy_signals_for_filter.empty:
        dummy_signals_for_filter['SignalTime'] = pd.to_datetime(dummy_signals_for_filter['SignalTime'])
        dummy_signals_for_filter = dummy_signals_for_filter.set_index('SignalTime')


    logger.info(f"Dummy price data (first 5 rows):\\n{dummy_price_data.head()}")
    if not dummy_signals_for_filter.empty:
        logger.info(f"Dummy signals for filter (first 5 rows):\\n{dummy_signals_for_filter.head()}")
    else:
        logger.info("No dummy signals generated for filter test.")


    # Test feature preparation for signal filtering
    logger.info("\\nTesting feature preparation for signal filtering...")
    if not dummy_signals_for_filter.empty:
        features_sig = ai_service._prepare_features_for_signal_filtering(dummy_price_data, dummy_signals_for_filter.copy()) # Pass copy
        if not features_sig.empty:
            logger.info(f"Signal features prepared (first 5 rows):\\n{features_sig.head()}")

            # Test signal filter model training (placeholder)
            # Generate more varied dummy labels for testing stratification
            if len(features_sig) >= 5: # Ensure enough samples for varied labels
                 # Make one class have only 1 sample to test the fix
                labels_list = [0] * (len(features_sig) -1) + [1] 
                np.random.shuffle(labels_list) # Shuffle to make it less predictable
                dummy_labels_sig = pd.Series(labels_list, index=features_sig.index)

                logger.info(f"Dummy labels for signal filter training (counts): {Counter(dummy_labels_sig)}")
                logger.info("\\nTesting signal filter model training...")
                ai_service.train_signal_filter_model(features_sig, dummy_labels_sig, model_type="RandomForest")
            else:
                logger.info("Not enough features generated to robustly test training with varied labels.")
            
            # Test signal filtering
            if ai_service.signal_filter_model:
                logger.info("\\nTesting signal filtering...")
                # For filter_signals, raw_signals is expected to have 'SignalTime' as a column or index
                # If it was set as index above, reset it for this call if filter_signals expects a column
                # Or ensure filter_signals can handle index. Current filter_signals logic tries to handle both.
                
                # Reset index for dummy_signals_for_filter to pass it as expected by some parts of filter_signals
                # or ensure filter_signals internally handles it if it's already indexed.
                # The current filter_signals tries to set index if not already, so passing indexed should be fine.
                
                filtered_sigs = ai_service.filter_signals(dummy_price_data, dummy_signals_for_filter.copy(), "RandomForest") # Pass copy
                logger.info(f"Filtered signals (first 5 rows):\\n{filtered_sigs.head()}")
                logger.info(f"Original signals: {len(dummy_signals_for_filter)}, Filtered signals: {len(filtered_sigs)}\n")
        else:
            logger.info("Signal features DataFrame is empty after preparation.")
    else:
        logger.info("Skipping signal filter tests as no dummy signals were generated.")
    # ... (rest of the test code for regime detection) ...
