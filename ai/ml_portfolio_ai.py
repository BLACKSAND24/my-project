"""ML-Based Portfolio Allocator.

This module provides a more sophisticated portfolio allocation using
machine learning instead of the simple correlation penalty in PortfolioAI.

Features used:
- Historical returns (momentum, mean reversion signals)
- Macro inputs (sentiment, FRED data)
- Volume features (volatility, market activity)
- Regime detection (low_vol, high_vol, crisis)
"""
import numpy as np
from typing import Dict, List, Optional
import json
import os
from collections import deque

# Try to import ML libraries - fall back to simple model if not available
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("ML_PORTFOLIO_AI")


class FeatureEngineer:
    """Extract features from market data for ML model."""
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.feature_history = deque(maxlen=1000)
    
    def extract_features(self, market_data: dict, raw_scores: Dict[str, float], 
                        regime: str = None, macro_sentiment: float = 0.0) -> np.ndarray:
        """Extract features from market data.
        
        Args:
            market_data: Dict with returns, volatility, prices, etc.
            raw_scores: Strategy raw scores from StrategyEnsemble
            regime: Current market regime
            macro_sentiment: RSS sentiment score
            
        Returns:
            numpy array of features
        """
        features = []
        
        # 1. Return features (momentum, volatility)
        returns = market_data.get("returns", [])
        if len(returns) >= 5:
            # Short-term momentum (5-day)
            features.append(np.mean(returns[-5:]) * 100)
            features.append(np.std(returns[-5:]) * 100)
        else:
            features.append(0.0)
            features.append(0.0)
        
        if len(returns) >= 20:
            # Long-term momentum (20-day)
            features.append(np.mean(returns[-20:]) * 100)
            features.append(np.std(returns[-20:]) * 100)
        else:
            features.append(0.0)
            features.append(0.0)
        
        # 2. Volatility features
        vol = market_data.get("volatility", 0.0)
        features.append(vol * 100)  # Annualized vol
        
        # 3. Macro sentiment
        features.append(macro_sentiment * 10)
        
        # 4. Regime encoding
        regime_map = {"low_vol": 0, "normal": 1, "high_vol": 2, "crisis": 3}
        features.append(regime_map.get(regime, 1))
        
        # 5. Strategy raw scores
        strategy_names = ["momentum", "mean_revert", "vol_breakout"]
        for name in strategy_names:
            features.append(raw_scores.get(name, 0.0))
        
        # 6. Cross-strategy features
        # Average score
        if raw_scores:
            features.append(np.mean(list(raw_scores.values())))
            features.append(np.max(list(raw_scores.values())))
            features.append(np.min(list(raw_scores.values())))
        else:
            features.append(0.0)
            features.append(0.0)
            features.append(0.0)
        
        # 7. Market activity (price changes)
        prices = market_data.get("prices", {})
        if prices:
            price_vals = list(prices.values())
            if len(price_vals) >= 2:
                features.append((price_vals[-1] - price_vals[0]) / price_vals[0] * 100)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        self.feature_history.append(features)
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names for debugging."""
        return [
            "momentum_5d", "volatility_5d",
            "momentum_20d", "volatility_20d",
            "annualized_vol", "macro_sentiment", "regime",
            "score_momentum", "score_mean_revert", "score_vol_breakout",
            "avg_score", "max_score", "min_score", "price_change"
        ]


class SimpleMLPortfolioModel:
    """A simple but effective ML model for portfolio allocation.
    
    Uses gradient boosting if available, otherwise falls back to 
    a rule-based model.
    """
    
    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_trained = False
        self.feature_engineer = FeatureEngineer()
        
        # Model parameters
        self.lookback = 20
        self.min_confidence = 0.1
        
        # Initialize model
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using rule-based model")
            return
        
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model on historical data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target weights (n_samples, n_assets)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train - scikit-learn not available")
            return
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
            logger.info(f"ML model trained on {len(X)} samples")
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict portfolio weights.
        
        Args:
            features: Feature vector
            
        Returns:
            Predicted weights for each asset
        """
        if not self.is_trained or not SKLEARN_AVAILABLE:
            # Fall back to rule-based prediction
            return self._rule_based_predict(features)
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            weights = self.model.predict(features_scaled)
            
            # Normalize to sum to 1
            weights = np.array(weights)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(weights)) / len(weights)
            
            # Apply confidence threshold
            max_weight = np.max(weights)
            if max_weight < self.min_confidence:
                # Too uncertain, use equal weights
                weights = np.ones(len(weights)) / len(weights)
            
            return weights
        except Exception as e:
            logger.error(f"Prediction failed: {e}, using rule-based")
            return self._rule_based_predict(features)
    
    def _rule_based_predict(self, features: np.ndarray) -> np.ndarray:
        """Rule-based prediction when ML is not available.
        
        Simple heuristic:
        - High vol -> reduce exposure
        - Crisis regime -> reduce exposure
        - High sentiment -> increase exposure
        - Use raw scores from middle of feature vector
        """
        n_assets = 3  # momentum, mean_revert, vol_breakout
        
        # Extract relevant features
        vol = features[4] if len(features) > 4 else 0  # annualized vol
        sentiment = features[5] if len(features) > 5 else 0  # macro sentiment
        regime = int(features[6]) if len(features) > 6 else 1  # regime
        
        # Strategy scores are at indices 7, 8, 9
        scores = features[7:10] if len(features) > 9 else np.ones(3)
        
        # Adjust based on regime and volatility
        regime_multiplier = 1.0
        if regime == 3:  # crisis
            regime_multiplier = 0.3
        elif regime == 2:  # high_vol
            regime_multiplier = 0.6
        
        # Sentiment adjustment
        sentiment_multiplier = 1.0 + sentiment * 0.2
        
        # Apply adjustments
        adjusted_scores = scores * regime_multiplier * sentiment_multiplier
        
        # Normalize
        if adjusted_scores.sum() > 0:
            weights = adjusted_scores / adjusted_scores.sum()
        else:
            weights = np.ones(n_assets) / n_assets
        
        return weights
    
    def save(self, path: str):
        """Save model to disk."""
        if not self.is_trained:
            logger.warning("Model not trained, nothing to save")
            return
        
        # Save model parameters
        model_data = {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "lookback": self.lookback,
            "min_confidence": self.min_confidence
        }
        
        with open(path + "_config.json", "w") as f:
            json.dump(model_data, f)
        
        logger.info(f"Model config saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        config_path = path + "_config.json"
        
        if not os.path.exists(config_path):
            logger.warning(f"Model config not found at {config_path}")
            return
        
        with open(config_path, "r") as f:
            model_data = json.load(f)
        
        self.model_type = model_data.get("model_type", "gradient_boosting")
        self.is_trained = model_data.get("is_trained", False)
        self.lookback = model_data.get("lookback", 20)
        self.min_confidence = model_data.get("min_confidence", 0.1)
        
        self._init_model()
        logger.info(f"Model loaded from {path}")


class MLPortfolioAI:
    """ML-enhanced portfolio allocator.
    
    This wraps the simple PortfolioAI and adds ML-based allocation
    as an option. It can fall back to the simple correlation penalty
    if ML is not available or fails.
    """
    
    def __init__(self, risk_brain, use_ml: bool = True, model_path: str = None):
        """Initialize ML Portfolio AI.
        
        Args:
            risk_brain: GlobalRiskController for correlation data
            use_ml: Whether to use ML model (default True)
            model_path: Path to save/load model (optional)
        """
        self.risk_brain = risk_brain
        self.use_ml = use_ml and SKLEARN_AVAILABLE
        self.model_path = model_path
        
        # Initialize ML model
        self.ml_model = SimpleMLPortfolioModel()
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Load existing model if available
        if model_path and os.path.exists(model_path + "_config.json"):
            self.ml_model.load(model_path)
        
        # Fallback to simple correlation penalty
        self._simple_portfolio_ai = None  # Will be set if needed
        
        logger.info(f"MLPortfolioAI initialized (ML={'enabled' if self.use_ml else 'disabled'})")
    
    def get_weights(self, raw_scores: Dict[str, float], 
                   market_data: dict = None,
                   regime: str = None,
                   macro_sentiment: float = 0.0) -> Dict[str, float]:
        """Return ML-enhanced portfolio weights.
        
        Args:
            raw_scores: Strategy scores from StrategyEnsemble
            market_data: Market data for feature extraction
            regime: Current market regime
            macro_sentiment: RSS sentiment score
            
        Returns:
            dict mapping strategy name to weight
        """
        if not raw_scores:
            return {}
        
        # If ML is disabled or market_data not available, use simple approach
        if not self.use_ml or market_data is None:
            return self._simple_correlation_penalty(raw_scores)
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(
                market_data, raw_scores, regime, macro_sentiment
            )
            
            # Get ML predictions
            ml_weights = self.ml_model.predict(features)
            
            # Map to strategy names
            strategy_names = ["momentum", "mean_revert", "vol_breakout"]
            ml_weight_dict = {
                strategy_names[i]: float(ml_weights[i]) 
                for i in range(min(len(ml_weights), len(strategy_names)))
            }
            
            # Also get simple correlation weights
            simple_weights = self._simple_correlation_penalty(raw_scores)
            
            # Blend ML and simple weights (70% ML, 30% simple)
            blend_factor = float(CONFIG.get("ML_WEIGHT_BLEND", 0.7))
            
            # Normalize simple weights to ensure they sum to 1
            total_simple = sum(simple_weights.values())
            if total_simple > 0:
                simple_weights = {k: v/total_simple for k, v in simple_weights.items()}
            
            # Merge: use ML weights but ensure all strategies are represented
            result = {}
            all_strategies = set(ml_weight_dict.keys()) | set(simple_weights.keys())
            
            for strat in all_strategies:
                ml_w = ml_weight_dict.get(strat, 0.0)
                simple_w = simple_weights.get(strat, 0.0)
                result[strat] = blend_factor * ml_w + (1 - blend_factor) * simple_w
            
            # Final normalization
            total = sum(result.values())
            if total > 0:
                result = {k: v/total for k, v in result.items()}
            else:
                return {k: 0.0 for k in raw_scores}
            
            return result
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}, using simple correlation penalty")
            return self._simple_correlation_penalty(raw_scores)
    
    def _simple_correlation_penalty(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """Simple correlation penalty (same as original PortfolioAI)."""
        # Compute average correlation for each strategy
        corr_info = self.risk_brain.correlation_matrix()
        names = corr_info.get("names", [])
        mat = corr_info.get("matrix")
        avg_corr = {}
        
        if mat is not None and mat.size > 1:
            for i, name in enumerate(names):
                coeffs = [abs(v) for j, v in enumerate(mat[i]) if j != i]
                avg_corr[name] = sum(coeffs) / len(coeffs) if coeffs else 0.0
        else:
            for name in raw_scores:
                avg_corr[name] = 0.0
        
        # Adjust scores by correlation penalty
        adjusted = {}
        for name, score in raw_scores.items():
            penalty = avg_corr.get(name, 0.0)
            adjusted[name] = float(score) * (1.0 - penalty)
        
        # Normalize
        total = sum(adjusted.values())
        if total <= 0:
            return {k: 0.0 for k in adjusted}
        return {k: v / total for k, v in adjusted.items()}
    
    def train(self, historical_data: List[dict]):
        """Train the ML model on historical data.
        
        Args:
            historical_data: List of dicts with keys:
                - market_data: dict with returns, volatility, etc.
                - regime: str
                - macro_sentiment: float
                - target_weights: dict of strategy -> weight
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train - scikit-learn not available")
            return
        
        X = []
        y = []
        
        for data in historical_data:
            market_data = data.get("market_data", {})
            raw_scores = data.get("raw_scores", {})
            regime = data.get("regime")
            macro_sentiment = data.get("macro_sentiment", 0.0)
            target_weights = data.get("target_weights", {})
            
            if not target_weights:
                continue
            
            # Extract features
            features = self.feature_engineer.extract_features(
                market_data, raw_scores, regime, macro_sentiment
            )
            
            # Extract targets
            strategy_names = ["momentum", "mean_revert", "vol_breakout"]
            targets = [target_weights.get(name, 0.0) for name in strategy_names]
            
            # Normalize targets
            total = sum(targets)
            if total > 0:
                targets = [t/total for t in targets]
            else:
                targets = [1.0/len(strategy_names)] * len(strategy_names)
            
            X.append(features)
            y.append(targets)
        
        if len(X) < 10:
            logger.warning(f"Not enough training data ({len(X)} samples), need at least 10")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        self.ml_model.train(X, y)
        
        # Save model
        if self.model_path:
            self.ml_model.save(self.model_path)
        
        logger.info(f"Training complete: {len(X)} samples")
    
    def save_model(self, path: str = None):
        """Save the trained model."""
        save_path = path or self.model_path
        if save_path:
            self.ml_model.save(save_path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        self.ml_model.load(path)


def create_ml_portfolio_ai(risk_brain, config: dict = None) -> MLPortfolioAI:
    """Factory function to create MLPortfolioAI with config.
    
    Args:
        risk_brain: GlobalRiskController
        config: Optional config dict
        
    Returns:
        MLPortfolioAI instance
    """
    config = config or {}
    
    use_ml = config.get("USE_ML", True)
    model_path = config.get("MODEL_PATH", "models/ml_portfolio_model")
    blend = config.get("ML_WEIGHT_BLEND", 0.7)
    
    # Update config globally
    if blend:
        CONFIG["ML_WEIGHT_BLEND"] = blend
    
    return MLPortfolioAI(risk_brain, use_ml=use_ml, model_path=model_path)
