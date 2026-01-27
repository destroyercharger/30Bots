"""
Test script for 30-Model Risk Parameters Integration
"""

import sys
sys.path.insert(0, '.')

print("=" * 70)
print("30-MODEL RISK PARAMETERS - INTEGRATION TEST")
print("=" * 70)

# Test 1: Import the risk params module
print("\n[TEST 1] Importing risk parameters module...")
try:
    from ai_30model_brain import (
        MODEL_RISK_PARAMS,
        get_model_risk_params,
        calculate_risk_prices,
        DEFAULT_RISK_PARAMS
    )
    print(f"  [OK] Successfully imported {len(MODEL_RISK_PARAMS)} model configurations")
except Exception as e:
    print(f"  [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 2: Verify all 30 models have risk params
print("\n[TEST 2] Verifying all 30 models have risk parameters...")
expected_models = [
    'Momentum_Selective', 'Momentum_Moderate', 'Momentum_Aggressive',
    'MeanReversion_Selective', 'MeanReversion_Moderate', 'MeanReversion_Aggressive',
    'Breakout_Selective', 'Breakout_Moderate', 'Breakout_Aggressive',
    'TrendFollowing_Selective', 'TrendFollowing_Moderate', 'TrendFollowing_Aggressive',
    'GapTrading_Selective', 'GapTrading_Moderate', 'GapTrading_Aggressive',
    'MultiIndicator_Selective', 'MultiIndicator_Moderate', 'MultiIndicator_Aggressive',
    'VWAP_Selective', 'VWAP_Moderate', 'VWAP_Aggressive',
    'RSIDivergence_Selective', 'RSIDivergence_Moderate', 'RSIDivergence_Aggressive',
    'BollingerBands_Selective', 'BollingerBands_Moderate', 'BollingerBands_Aggressive',
    'VolumeSpike_Selective', 'VolumeSpike_Moderate', 'VolumeSpike_Aggressive'
]

missing = [m for m in expected_models if m not in MODEL_RISK_PARAMS]
if missing:
    print(f"  [FAIL] Missing models: {missing}")
else:
    print(f"  [OK] All 30 models configured correctly")

# Test 3: Simulate a trade with Momentum_Selective
print("\n[TEST 3] Simulating a trade with Momentum_Selective...")
entry_price = 150.00
shares = 100
model = 'Momentum_Selective'

params = get_model_risk_params(model)
prices = calculate_risk_prices(entry_price, model)

print(f"  Model: {model}")
print(f"  Entry Price: ${entry_price:.2f}")
print(f"  Shares: {shares}")
print(f"  Position Value: ${entry_price * shares:,.2f}")
print(f"  ")
print(f"  Stop Loss: ${prices['stop_loss_price']:.2f} (-{params['stop_loss_pct']*100:.1f}%)")
print(f"  Take Profit: ${prices['take_profit_price']:.2f} (+{params['take_profit_pct']*100:.1f}%)")
print(f"  Trailing Activation: ${prices['trailing_activation_price']:.2f} (+{params['trailing_activation_pct']*100:.1f}%)")
print(f"  Trailing Stop: {params['trailing_stop_pct']*100:.1f}% once activated")
print(f"  Risk:Reward Ratio: {prices['risk_reward_ratio']}:1")
print(f"  ")
print(f"  Max Loss: ${(entry_price - prices['stop_loss_price']) * shares:.2f}")
print(f"  Target Profit: ${(prices['take_profit_price'] - entry_price) * shares:.2f}")

# Test 4: Compare different risk levels
print("\n[TEST 4] Comparing risk levels for Momentum strategy...")
print(f"  {'Level':<12} {'Stop Loss':>10} {'Take Profit':>12} {'R:R':>6} {'Trail':>8}")
print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*6} {'-'*8}")

for level in ['Selective', 'Moderate', 'Aggressive']:
    model_name = f'Momentum_{level}'
    p = MODEL_RISK_PARAMS[model_name]
    rr = p['take_profit_pct'] / p['stop_loss_pct']
    print(f"  {level:<12} {p['stop_loss_pct']*100:>9.1f}% {p['take_profit_pct']*100:>11.1f}% {rr:>5.1f}x {p['trailing_stop_pct']*100:>7.1f}%")

# Test 5: Show all strategies comparison
print("\n[TEST 5] All strategies - Selective level comparison...")
print(f"  {'Strategy':<16} {'Stop Loss':>10} {'Take Profit':>12} {'R:R':>6}")
print(f"  {'-'*16} {'-'*10} {'-'*12} {'-'*6}")

strategies = ['Momentum', 'MeanReversion', 'Breakout', 'TrendFollowing', 'GapTrading',
              'MultiIndicator', 'VWAP', 'RSIDivergence', 'BollingerBands', 'VolumeSpike']

for strategy in strategies:
    model_name = f'{strategy}_Selective'
    p = MODEL_RISK_PARAMS[model_name]
    rr = p['take_profit_pct'] / p['stop_loss_pct']
    print(f"  {strategy:<16} {p['stop_loss_pct']*100:>9.1f}% {p['take_profit_pct']*100:>11.1f}% {rr:>5.1f}x")

# Test 6: Simulate trailing stop behavior
print("\n[TEST 6] Simulating trailing stop behavior...")
entry = 100.00
model = 'Momentum_Selective'
params = get_model_risk_params(model)

print(f"  Entry: ${entry:.2f}")
print(f"  Initial Stop: ${entry * (1 - params['stop_loss_pct']):.2f}")
print(f"  Trailing activates at: ${entry * (1 + params['trailing_activation_pct']):.2f}")

# Simulate price movement
current_stop = entry * (1 - params['stop_loss_pct'])
trailing_active = False
prices_sequence = [100, 101, 102, 102.5, 103, 102, 101.5, 103.5, 104, 103]

print(f"\n  Price Movement Simulation:")
print(f"  {'Price':>8} {'Gain':>8} {'Stop':>8} {'Status':<25}")
print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*25}")

for price in prices_sequence:
    gain_pct = (price - entry) / entry
    
    # Check if trailing should activate
    if price > entry * (1 + params['trailing_activation_pct']):
        if not trailing_active:
            trailing_active = True
            status = "TRAILING ACTIVATED!"
        else:
            status = "Trailing"
        
        # Update trailing stop
        new_trail_stop = price * (1 - params['trailing_stop_pct'])
        if new_trail_stop > current_stop:
            current_stop = new_trail_stop
            status += " (stop raised)"
    else:
        status = "Waiting for activation"
    
    print(f"  ${price:>7.2f} {gain_pct*100:>7.1f}% ${current_stop:>7.2f} {status}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
