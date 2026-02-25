"""
Online ML Policy Module for Financial Organism.

This module provides:
- Feature building from historical returns + macro + volume data
- Regime classification using ML
- Online learning updates
- Integration into the strategy ensemble

The ML policy contributes to live signal generation and diagnostics.
"""
import numpy as np
import numpy.random as nrandom
from collections import deque
from typing import Dict, List, Optional, Tuple

from financial_organism.config import CONFIG
from financial_organism.utils.logger import get_logger

logger = get_logger("ML_POLICY")


class FeatureBuilder:
    """Builds features from market data for ML model."""
    
    def __init__(self, lookback_window: int = 60):
        """Initialize feature builder.
        
        Args:
            lookback_window: Number of periods to look back for features
        """
        self.lookback_window = lookback_window
        self.returns_history = deque(maxlen=lookback_window)
        self.volume_history = deque(maxlen=lookback_window)
        self.price_history = deque(maxlen=lookback_window)
    
    def update(self, returns: float, volume: float = 1.0, price: float = None):
        """Update history with new data point.
        
        Args:
            returns: Current period return
            volume: Current period volume
            price: Current price (optional)
        """
        self.returns_history.append(returns)
        self.volume_history.append(volume)
        if price is not None:
            self.price_history.append(price)
    
    def build_features(self) -> np.ndarray:
        """Build feature vector from historical data.
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Returns-based features
        returns = list(self.returns_history)
        if len(returns) >= 5:
            # Momentum features
            features.append(np.mean(returns[-5:]))  # Short-term mean
            features.append(np.std(returns[-5:]))   # Short-term volatility
        else:
            features.extend([0.0, 0.0])
        
        if len(returns) >= 20:
            # Medium-term features
            features.append(np.mean(returns[-20:]))
            features.append(np.std(returns[-20:]))
            # Trend
            features.append(returns[-1] - returns[-20])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        if len(returns) >= self.lookback_window:
            # Long-term features
            long_returns = list(self.returns_history)
            features.append(np.mean(long_returns))
            features.append(np.std(long_returns))
            # Skewness approximation
            mean_ret = np.mean(long_returns)
            std_ret = np.std(long_returns) if np.std(long_returns) > 1e-8 else 1.0
            skew = np.mean([((r - mean_ret) / std_ret) ** 3 for r in long_returns])
            features.append(skew)
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Volume-based features
        volumes = list(self.volume_history)
        if len(volumes) >= 5:
            features.append(np.mean(volumes[-5:]))
            features.append(np.std(volumes[-5:]) / max(np.mean(volumes[-5:]), 1e-8))
        else:
            features.extend([1.0, 0.0])
        
        if len(volumes) >= 20:
            # Volume trend
            features.append(volumes[-1] / max(np.mean(volumes[-20:]), 1e-8))
        else:
            features.append(1.0)
        
        # Price-based features (if available)
        prices = list(self.price_history)
        if len(prices) >= 5:
            # High-low range
            features.append((max(prices[-5:]) - min(prices[-5:])) / max(prices[-1], 1e-8))
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of features for debugging/interpretation."""
        return [
            "returns_short_mean",
            "returns_short_std",
            "returns_medium_mean",
            "returns_medium_std",
            "returns_trend",
            "returns_long_mean",
            "returns_long_std",
            "returns_skewness",
            "volume_short_mean",
            "volume_short_cv",
            "volume_trend",
            "price_range"
        ]


class RegimeClassifier:
    """Classifies market regime using simple ML model."""
    
    # Regime constants
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    HIGH_VOL = "high_vol"
    CRISIS = "crisis"
    
    def __init__(self):
        """Initialize regime classifier."""
        self.low_vol_threshold = CONFIG.get("LOW_VOL_THRESHOLD", 0.10)
        self.high_vol_threshold = CONFIG.get("HIGH_VOL_THRESHOLD", 0.30)
        self.crisis_threshold = CONFIG.get("CRISIS_VOL_THRESHOLD", 0.35)
        
        # Simple threshold-based classifier (could be replaced with ML)
        self.current_regime = self.NORMAL
        self.regime_history = deque(maxlen=100)
    
    def classify(self, features: np.ndarray, volatility: float = None) -> str:
        """Classify current market regime.
        
        Args:
            features: Feature vector
            volatility: Optional explicit volatility (annualized)
            
        Returns:
            Regime string
        """
        # Use explicit volatility if provided, otherwise estimate from features
        if volatility is None:
            # Extract volatility from features (index 1 is short-term std)
            if len(features) > 1:
                volatility = features[1] * np.sqrt(252)  # Annualize
            else:
                volatility = 0.15  # Default
        
        # Classify based on volatility
        if volatility < self.low_vol_threshold:
            regime = self.LOW_VOL
        elif volatility < self.high_vol_threshold:
            regime = self.NORMAL
        elif volatility < self.crisis_threshold:
            regime = self.HIGH_VOL
        else:
            regime = self.CRISIS
        
        self.current_regime = regime
        self.regime_history.append(regime)
        
        return regime
    
    def get_current_regime(self) -> str:
        """Get current regime."""
        return self.current_regime
    
    def get_regime_probabilities(self) -> Dict[str, float]:
        """Get probability distribution over regimes based on history."""
        if not self.regime_history:
            return {self.LOW_VOL: 0.0, self.NORMAL: 1.0, self.HIGH_VOL: 0.0, self.CRISIS: 0.0}
        
        counts = {
            self.LOW_VOL: 0,
            self.NORMAL: 0,
            self.HIGH_VOL: 0,
            self.CRISIS: 0
        }
        
        for regime in self.regime_history:
            counts[regime] = counts.get(regime, 0) + 1
        
        total = len(self.regime_history)
        return {k: v / total for k, v in counts.items()}


class OnlineLearner:
    """Performs online learning updates to the ML policy."""
    
    def __init__(self, learning_rate: float = 0.01):
        """Initialize online learner.
        
        Args:
            learning_rate: Learning rate for updates
        """
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = 0.0
        self.feature_importance = None
        self.update_count = 0
    
    def initialize_weights(self, feature_dim: int):
        """Initialize weights for linear model.
        
        Args:
            feature_dim: Dimension of feature vector
        """
        # Xavier initialization
        self.weights = nrandom.randn(feature_dim) * np.sqrt(2.0 / feature_dim)
        self.feature_importance = np.zeros(feature_dim)
    
    def predict(self, features: np.ndarray) -> float:
        """Make prediction using current model.
        
        Args:
            features: Feature vector
            
        Returns:
            Predicted score
        """
        if self.weights is None:
            self.initialize_weights(len(features))
        
        return np.dot(features, self.weights) + self.bias
    
    def update(self, features: np.ndarray, target: float, reward: float = None):
        """Update model based on observed outcome.
        
        Args:
            features: Feature vector
            target: Target value (actual return)
            reward: Optional reward signal for reinforcement learning
        """
        if self.weights is None:
            self.initialize_weights(len(features))
        
        # Simple online learning (could be replaced with more sophisticated method)
        prediction = self.predict(features)
        error = target - prediction
        
        # Update weights
        self.weights += self.learning_rate * error * features
        self.bias += self.learning_rate * error
        
        # Update feature importance (running average of absolute weights)
        self.feature_importance = (
            0.99 * self.feature_importance + 
            0.01 * np.abs(features * error)
        )
        
        self.update_count += 1
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_importance is None:
            return {}
        
        # Normalize
        total = np.sum(self.feature_importance)
        if total > 0:
            normalized = self.feature_importance / total
        else:
            normalized = self.feature_importance
        
        return {f"feature_{i}": float(v) for i, v in enumerate(normalized)}


class MLPolicy:
    """Main ML Policy class that combines feature building, regime classification, and online learning."""
    
    def __init__(self, name: str = "ml_policy"):
        """Initialize ML Policy.
        
        Args:
            name: Policy name
        """
        self.name = name
        self.feature_builder = FeatureBuilder()
        self.regime_classifier = RegimeClassifier()
        self.online_learner = OnlineLearner()
        
        # State
        self.is_enabled = True
        self.last_score = 0.0
        self.last_regime = self.regime_classifier.NORMAL
        
        # Performance tracking
        self.prediction_history = deque(maxlen=100)
        self.actual_returns = deque(maxlen=100)
        
        logger.info(f"MLPolicy initialized: {name}")
    
    def update(self, returns: float, volume: float = 1.0, price: float = None, 
               macro_data: Dict = None, liquidity: float = None):
        """Update policy with new market data.
        
        Args:
            returns: Current period return
            volume: Current period volume
            price: Current price
            macro_data: Optional macro data (sentiment, FRED series, etc.)
            liquidity: Optional liquidity metric
        """
        # Update feature builder
        self.feature_builder.update(returns, volume, price)
        
        # Build features
        features = self.feature_builder.build_features()
        
        # Add macro features if available
        if macro_data:
            macro_features = self._extract_macro_features(macro_data)
            features = np.concatenate([features, macro_features])
        
        # Add liquidity features if available
        if liquidity is not None:
            features = np.append(features, liquidity)
        
        # Classify regime
        volatility = None
        if macro_data and 'volatility' in macro_data:
            volatility = macro_data['volatility']
        
        regime = self.regime_classifier.classify(features, volatility)
        self.last_regime = regime
        
        # Make prediction
        score = self.online_learner.predict(features)
        self.last_score = score
        
        # Store for later training
        self.prediction_history.append((features, score, regime))
    
    def _extract_macro_features(self, macro_data: Dict) -> np.ndarray:
        """Extract features from macro data.
        
        Args:
            macro_data: Macro data dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Sentiment
        sentiment = macro_data.get('macro_sentiment', 0.0)
        features.append(sentiment)
        
        # FRED data (if available)
        fred = macro_data.get('fred', [])
        if fred and len(fred) > 0:
            # Use latest value
            try:
                fred_value = float(fred[-1].get('value', 0))
                # Normalize (rough approximation)
                fred_normalized = fred_value / 1000000  # Rough normalization for GDP
                features.append(fred_normalized)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def score(self, market_data: Dict) -> float:
        """Calculate score for current market conditions.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Score (0.0 to 1.0, higher = more bullish)
        """
        if not self.is_enabled:
            return 0.0
        
        # Get returns from market data
        returns = market_data.get("returns", [])
        if not returns:
            return 0.0
        
        # Use latest return
        latest_return = returns[-1] if returns else 0.0
        
        # Get volume (default to 1.0 if not available)
        volume = market_data.get("volume", 1.0)
        
        # Get price
        prices = market_data.get("prices", {})
        price = None
        if prices:
            price = list(prices.values())[0]
        
        # Get macro data
        macro_data = {}
        if "macro_sentiment" in market_data:
            macro_data['macro_sentiment'] = market_data["macro_sentiment"]
        if "fred" in market_data:
            macro_data['fred'] = market_data["fred"]
        if "volatility" in market_data:
            macro_data['volatility'] = market_data["volatility"]
        
        # Get liquidity
        liquidity = market_data.get("liquidity")
        
        # Update policy
        self.update(latest_return, volume, price, macro_data, liquidity)
        
        # Return normalized score (sigmoid-like transformation)
        # Map from raw prediction to 0-1 range
        normalized_score = 1.0 / (1.0 + np.exp(-self.last_score))
        
        # Adjust based on regime
        regime = self.last_regime
        if regime == RegimeClassifier.CRISIS:
            normalized_score *= 0.3  # Reduce exposure in crisis
        elif regime == RegimeClassifier.HIGH_VOL:
            normalized_score *= 0.6  # Reduce exposure in high vol
        elif regime == RegimeClassifier.LOW_VOL:
            normalized_score *= 1.2  # Increase exposure in low vol
        
        return max(0.0, min(1.0, normalized_score))
    
    def train_on_outcome(self, actual_return: float):
        """Update model based on actual outcome.
        
        Args:
            actual_return: Actual return observed
        """
        if not self.prediction_history:
            return
        
        # Get last features
        features, predicted, regime = self.prediction_history[-1]
        
        # Update online learner
        self.online_learner.update(features, actual_return)
        
        # Store actual return
        self.actual_returns.append(actual_return)
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information from the ML policy.
        
        Returns:
            Dictionary with diagnostics
        """
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "current_regime": self.last_regime,
            "last_score": float(self.last_score),
            "regime_probabilities": self.regime_classifier.get_regime_probabilities(),
            "feature_importance": self.online_learner.get_feature_importance(),
            "update_count": self.update_count,
            "prediction_history_size": len(self.prediction_history)
        }
    
    def enable(self):
        """Enable the ML policy."""
        self.is_enabled = True
        logger.info(f"MLPolicy {self.name} enabled")
    
    def disable(self):
        """Disable the ML policy."""
        self.is_enabled = False
        logger.info(f"MLPolicy {self.name} disabled")


def create_ml_policy(config: dict = None) -> MLPolicy:
    """Factory function to create ML policy.
    
    Args:
        config: Optional config dict
        
    Returns:
        MLPolicy instance
    """
    config = config or {}
    name = config.get("ML_POLICY_NAME", "ml_policy")
    return MLPolicy(name=name)
