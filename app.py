# app.py - MULTI-COLOR PREDICTION VERSION WITH 3 BOMBS ONLY
from flask import Flask, render_template, request, jsonify
import os
from groq import Groq
import random
import re
from datetime import datetime
from collections import Counter

app = Flask(__name__)

# 🔑 Get API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = None
active_model = None

if GROQ_API_KEY and GROQ_API_KEY != "gsk_YOUR_API_KEY_HERE":
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Test models in order of preference
        test_models = [
            "llama-3.2-90b-text-preview",
            "llama-3.1-70b-versatile", 
            "mixtral-8x7b-32768",
            "llama-3.1-8b-instant",
        ]
        
        print("🧪 Testing available models...")
        for model in test_models:
            try:
                test = client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=model,
                    max_tokens=5
                )
                active_model = model
                print(f"✅ Using model: {model}")
                break
            except Exception as e:
                print(f"❌ {model} failed: {str(e)[:50]}")
                continue
        
        if not active_model:
            print("❌ No working models found!")
            client = None
            
    except Exception as e:
        print(f"❌ Groq initialization failed: {e}")
        client = None
else:
    print("⚠️ WARNING: Set GROQ_API_KEY environment variable!")


def analyze_color_pattern(history):
    """
    Analyze color history to find patterns
    """
    if not history or len(history) < 3:
        return {
            'most_common': None,
            'least_common': None,
            'streak': None,
            'alternating': False,
            'frequency': {},
            'top_3': [],
            'bottom_3': []
        }
    
    # Count frequencies
    frequency = Counter(history)
    
    # Find most and least common
    most_common_list = frequency.most_common()
    most_common = most_common_list[0][0] if most_common_list else None
    least_common = most_common_list[-1][0] if most_common_list else None
    
    # Get top 3 and bottom 3
    top_3 = [color for color, _ in most_common_list[:3]]
    bottom_3 = [color for color, _ in most_common_list[-3:]]
    
    # Check for streaks
    current_streak = 1
    streak_color = history[-1] if history else None
    for i in range(len(history) - 1, 0, -1):
        if history[i] == history[i-1]:
            current_streak += 1
        else:
            break
    
    # Check for alternating pattern
    alternating = False
    if len(history) >= 4:
        alternating = all(history[i] != history[i-1] for i in range(len(history)-3, len(history)))
    
    # Recent patterns
    recent_5 = history[-5:] if len(history) >= 5 else history
    recent_10 = history[-10:] if len(history) >= 10 else history
    
    return {
        'most_common': most_common,
        'least_common': least_common,
        'streak': current_streak if current_streak > 1 else None,
        'streak_color': streak_color,
        'alternating': alternating,
        'frequency': dict(frequency),
        'last_3': history[-3:] if len(history) >= 3 else history,
        'top_3': top_3,
        'bottom_3': bottom_3,
        'recent_5_freq': dict(Counter(recent_5)),
        'recent_10_freq': dict(Counter(recent_10))
    }


def extract_multiple_colors_from_response(response_text, available_colors):
    """
    Extract 3 colors from AI response with rankings
    """
    response_lower = response_text.lower()
    predictions = []
    
    # Method 1: Look for numbered predictions
    patterns = [
        r'(?:1st|first|#1|1\.|prediction 1)[:\s]+(\w+)',
        r'(?:2nd|second|#2|2\.|prediction 2)[:\s]+(\w+)',
        r'(?:3rd|third|#3|3\.|prediction 3)[:\s]+(\w+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            color = match.group(1).strip()
            if color in available_colors and color not in predictions:
                predictions.append(color)
    
    # Method 2: Look for listed format
    if len(predictions) < 3:
        list_match = re.search(r'predictions?[:\s]+([^.]+)', response_lower)
        if list_match:
            text = list_match.group(1)
            for color in available_colors:
                if color in text and color not in predictions:
                    predictions.append(color)
    
    # Method 3: Look for any mentions of colors in order
    if len(predictions) < 3:
        for color in available_colors:
            if color in response_lower and color not in predictions:
                predictions.append(color)
                if len(predictions) == 3:
                    break
    
    # Ensure we have exactly 3 predictions
    while len(predictions) < 3:
        # Add colors that haven't been selected yet
        remaining = [c for c in available_colors if c not in predictions]
        if remaining:
            predictions.append(random.choice(remaining))
        else:
            break
    
    return predictions[:3]


def extract_confidence_levels(response_text):
    """
    Extract confidence levels for each prediction
    """
    # Look for multiple confidence values
    confidence_pattern = r'(\d+)%'
    matches = re.findall(confidence_pattern, response_text)
    
    confidences = []
    for match in matches:
        conf = int(match)
        if 50 <= conf <= 100:
            confidences.append(min(95, max(60, conf)))
    
    # If we don't have 3 confidence levels, generate them
    while len(confidences) < 3:
        if len(confidences) == 0:
            confidences.append(random.randint(75, 90))  # High confidence for first
        elif len(confidences) == 1:
            confidences.append(random.randint(65, 80))  # Medium for second
        else:
            confidences.append(random.randint(55, 70))  # Lower for third
    
    return confidences[:3]


def extract_reasoning(response_text):
    """
    Extract reasoning from response
    """
    reason_patterns = [
        r'reason(?:ing)?[:\s]+(.+?)(?:\n|$)',
        r'because[:\s]+(.+?)(?:\n|\.)',
        r'strategy[:\s]+(.+?)(?:\n|\.)',
    ]
    
    for pattern in reason_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:200]
    
    sentences = [s.strip() for s in response_text.split('.') if len(s.strip()) > 20]
    if sentences:
        return sentences[0][:200]
    
    return response_text[:200]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/color', methods=['POST'])
def predict_color():
    print("\n" + "="*60)
    print("🎨 MULTI-COLOR PREDICTION REQUEST (3 COLORS)")
    print("="*60)
    
    colors = ['red', 'green', 'blue', 'yellow', 'pink', 'purple']
    
    if not client or not active_model:
        # Fallback predictions
        random_colors = random.sample(colors, 3)
        return jsonify({
            'error': False,
            'predictions': [
                {'color': random_colors[0], 'confidence': 75, 'rank': 1},
                {'color': random_colors[1], 'confidence': 65, 'rank': 2},
                {'color': random_colors[2], 'confidence': 55, 'rank': 3}
            ],
            'pattern': 'Random Strategy',
            'reasoning': 'API not configured - using random selection',
            'timestamp': datetime.now().isoformat(),
            'ai_powered': False
        })
    
    try:
        # Get request data
        data = request.get_json() or {}
        history = data.get('history', [])
        
        print(f"📊 History length: {len(history)}")
        print(f"📊 Recent history: {history[-10:] if history else 'None'}")
        
        # Analyze patterns
        pattern_analysis = analyze_color_pattern(history)
        print(f"📈 Pattern analysis: {pattern_analysis}")
        
        # Build intelligent prompt for 3 predictions
        prompt = f"""You are an expert color game predictor analyzing patterns and probabilities.
Your task is to predict the TOP 3 MOST LIKELY colors for the next round.

AVAILABLE COLORS: {', '.join(colors)}

GAME HISTORY (last {len(history)} rounds):
{history[-20:] if history else 'No history yet - first prediction'}

PATTERN ANALYSIS:
- Most frequent overall: {pattern_analysis['most_common'] or 'N/A'}
- Least frequent overall: {pattern_analysis['least_common'] or 'N/A'}
- Current streak: {pattern_analysis['streak'] or 'None'} {f"({pattern_analysis['streak_color']})" if pattern_analysis.get('streak_color') else ''}
- Alternating pattern: {'Yes' if pattern_analysis['alternating'] else 'No'}
- Overall frequency: {pattern_analysis['frequency']}
- Recent 5 rounds frequency: {pattern_analysis.get('recent_5_freq', {})}
- Recent 10 rounds frequency: {pattern_analysis.get('recent_10_freq', {})}

MULTI-COLOR PREDICTION STRATEGY:
1. PRIMARY CHOICE: Use strongest pattern indicator (hot/cold/due colors)
2. SECONDARY CHOICE: Use contrarian indicator (opposite of primary logic)
3. TERTIARY CHOICE: Use balanced probability approach

Consider:
- Hot colors (frequently appearing) might continue or cool down
- Cold colors (rarely appearing) might be statistically due
- Streak breakers vs streak continuers
- Recent trends vs overall patterns

RESPOND IN THIS EXACT FORMAT:
1ST: [color] (confidence%)
2ND: [color] (confidence%)
3RD: [color] (confidence%)
STRATEGY: [Brief explanation of your prediction logic]

Example:
1ST: blue (85%)
2ND: red (72%)
3RD: yellow (60%)
STRATEGY: Blue is underrepresented and due for regression, red shows recent momentum, yellow balances the selection."""

        print(f"📤 Calling AI ({active_model}) for 3-color prediction...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a statistical analysis AI specializing in multi-outcome prediction using pattern recognition, probability theory, and game theory. Always predict exactly 3 different colors."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=active_model,
            temperature=0.7,
            max_tokens=400,
            top_p=0.9
        )
        
        ai_response = chat_completion.choices[0].message.content
        print(f"📥 AI Response:\n{ai_response}\n")
        
        # Parse response for 3 colors
        predictions = extract_multiple_colors_from_response(ai_response, colors)
        confidences = extract_confidence_levels(ai_response)
        reasoning = extract_reasoning(ai_response)
        
        print(f"🔍 Parsed predictions: {predictions}")
        print(f"🔍 Confidences: {confidences}")
        
        # Determine pattern type based on predictions
        pattern_types = []
        for pred in predictions:
            if pattern_analysis.get('top_3') and pred in pattern_analysis['top_3']:
                pattern_types.append("Hot")
            elif pattern_analysis.get('bottom_3') and pred in pattern_analysis['bottom_3']:
                pattern_types.append("Cold")
            else:
                pattern_types.append("Balanced")
        
        pattern = f"{'/'.join(pattern_types)} Strategy"
        
        # Format results
        result = {
            'error': False,
            'predictions': [
                {
                    'color': predictions[0],
                    'confidence': confidences[0],
                    'rank': 1,
                    'strategy': pattern_types[0] if pattern_types else 'Primary'
                },
                {
                    'color': predictions[1],
                    'confidence': confidences[1],
                    'rank': 2,
                    'strategy': pattern_types[1] if len(pattern_types) > 1 else 'Secondary'
                },
                {
                    'color': predictions[2],
                    'confidence': confidences[2],
                    'rank': 3,
                    'strategy': pattern_types[2] if len(pattern_types) > 2 else 'Tertiary'
                }
            ],
            'pattern': pattern,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat(),
            'ai_powered': True,
            'model_used': active_model,
            'analysis': {
                'history_length': len(history),
                'most_common': pattern_analysis.get('most_common'),
                'least_common': pattern_analysis.get('least_common'),
                'current_streak': pattern_analysis.get('streak')
            }
        }
        
        print(f"✅ SUCCESS: Predicted {predictions[0]} ({confidences[0]}%), {predictions[1]} ({confidences[1]}%), {predictions[2]} ({confidences[2]}%)")
        print("="*60 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        
        # Fallback to random selection
        random_colors = random.sample(colors, 3)
        return jsonify({
            'error': False,
            'predictions': [
                {'color': random_colors[0], 'confidence': 70, 'rank': 1},
                {'color': random_colors[1], 'confidence': 60, 'rank': 2},
                {'color': random_colors[2], 'confidence': 50, 'rank': 3}
            ],
            'pattern': 'Fallback Strategy',
            'reasoning': f'Error occurred: {str(e)[:50]} - Using strategic random selection',
            'timestamp': datetime.now().isoformat(),
            'ai_powered': False
        })


def get_strategic_positions():
    """
    Return strategic positions for minesweeper with EXACTLY 3 BOMBS
    """
    # All positions (0-24)
    all_positions = list(range(25))
    
    # Strategic categories
    corners = [0, 4, 20, 24]
    edges = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]
    center = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    
    return {
        'all': all_positions,
        'corners': corners,
        'edges': edges,
        'center': center
    }


def extract_bomb_positions(response_text):
    """
    Extract exactly 3 bomb positions from AI response
    """
    # Look for bomb/danger positions
    patterns = [
        r'bombs?[:\s]+([0-9,\s]+)',
        r'danger[:\s]+([0-9,\s]+)',
        r'mines?[:\s]+([0-9,\s]+)',
    ]
    
    bomb_positions = []
    for pattern in patterns:
        match = re.search(pattern, response_text.lower())
        if match:
            text = match.group(1)
            positions = [int(x.strip()) for x in re.findall(r'\d+', text)
                        if 0 <= int(x.strip()) < 25]
            bomb_positions.extend(positions)
            if bomb_positions:
                break
    
    # Ensure exactly 3 unique bomb positions
    bomb_positions = list(set(bomb_positions))[:3]
    
    # If we don't have exactly 3, use strategic placement
    if len(bomb_positions) < 3:
        strategic = get_strategic_positions()
        # Default strategic bomb placement: 1 center, 1 edge, 1 corner
        default_bombs = []
        if strategic['center']:
            default_bombs.append(random.choice(strategic['center']))
        if strategic['edges'] and len(default_bombs) < 3:
            default_bombs.append(random.choice(strategic['edges']))
        if strategic['corners'] and len(default_bombs) < 3:
            default_bombs.append(random.choice(strategic['corners']))
        
        # Fill remaining with random positions
        while len(default_bombs) < 3:
            pos = random.randint(0, 24)
            if pos not in default_bombs:
                default_bombs.append(pos)
        
        bomb_positions = default_bombs[:3]
    
    return bomb_positions[:3]


@app.route('/predict/minesweeper', methods=['POST'])
def predict_minesweeper():
    print("\n" + "="*60)
    print("💣 MINESWEEPER PREDICTION REQUEST (EXACTLY 3 BOMBS)")
    print("="*60)
    
    if not client or not active_model:
        # Generate random 3 bomb positions
        bomb_positions = random.sample(range(25), 3)
        safe_positions = [i for i in range(25) if i not in bomb_positions]
        
        return jsonify({
            'error': False,
            'safe_spots': safe_positions,
            'danger_spots': bomb_positions,
            'total_bombs': 3,
            'confidence': 70,
            'reasoning': 'Random placement - API not configured',
            'timestamp': datetime.now().isoformat(),
            'ai_powered': False,
            'model_used': None,
        })
    
    try:
        strategic_pos = get_strategic_positions()
        
        prompt = f"""You are a minesweeper expert analyzing a 5x5 grid (25 cells, numbered 0-24) with EXACTLY 3 BOMBS.

GRID LAYOUT:
 0  1  2  3  4
 5  6  7  8  9
10 11 12 13 14
15 16 17 18 19
20 21 22 23 24

IMPORTANT: There are EXACTLY 3 BOMBS in the grid.

STRATEGIC ANALYSIS:
- Corners ({strategic_pos['corners']}): Lower probability (only 3 neighbors each)
- Edges: Medium probability (5 neighbors each)
- Center: Higher probability (8 neighbors each)

Your task: Predict where the 3 BOMBS are most likely located.

Consider:
1. Statistical distribution patterns
2. Common bomb placement strategies
3. Avoid clustering all bombs together
4. Balance between predictable and random placement

RESPOND IN EXACT FORMAT:
BOMBS: [exactly 3 cell numbers, comma-separated]
CONFIDENCE: [60-90]
STRATEGY: [Brief explanation of bomb placement logic]

Example:
BOMBS: 7, 16, 23
CONFIDENCE: 75
STRATEGY: Distributed placement - one center (high risk), one edge, one corner for balanced coverage."""

        print(f"📤 Calling AI ({active_model}) for 3-bomb prediction...")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a minesweeper AI expert. ALWAYS predict EXACTLY 3 bomb positions (no more, no less) for a 5x5 grid."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=active_model,
            temperature=0.6,
            max_tokens=300,
        )
        
        ai_response = chat_completion.choices[0].message.content
        print(f"📥 AI Response:\n{ai_response}\n")
        
        # Extract exactly 3 bomb positions
        bomb_positions = extract_bomb_positions(ai_response)
        
        # Ensure we have exactly 3 bombs
        if len(bomb_positions) != 3:
            # Fallback to strategic placement
            bomb_positions = []
            # Place one bomb in center area
            bomb_positions.append(random.choice(strategic_pos['center']))
            # Place one bomb on edge
            bomb_positions.append(random.choice(strategic_pos['edges']))
            # Place one more randomly
            remaining = [i for i in range(25) if i not in bomb_positions]
            bomb_positions.append(random.choice(remaining))
        
        # All other positions are safe
        safe_positions = [i for i in range(25) if i not in bomb_positions]
        
        confidence = extract_confidence_levels(ai_response)[0]
        reasoning = extract_reasoning(ai_response)
        
        if not reasoning:
            reasoning = f"Strategic placement of 3 bombs: positions {', '.join(map(str, bomb_positions))}"
        
        print(f"💣 Bomb positions: {bomb_positions}")
        print(f"✅ Safe positions: {len(safe_positions)} cells")
        print(f"📊 Confidence: {confidence}%")
        
        result = {
            'error': False,
            'safe_spots': safe_positions,
            'danger_spots': bomb_positions,
            'total_bombs': 3,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat(),
            'ai_powered': True,
            'model_used': active_model,
        }
        
        print(f"✅ SUCCESS: 3 bombs at positions {bomb_positions}")
        print("="*60 + "\n")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        
        # Fallback to random 3 bombs
        bomb_positions = random.sample(range(25), 3)
        safe_positions = [i for i in range(25) if i not in bomb_positions]
        
        return jsonify({
            'error': False,
            'safe_spots': safe_positions,
            'danger_spots': bomb_positions,
            'total_bombs': 3,
            'confidence': 65,
            'reasoning': f'Fallback placement due to error - 3 random bombs',
            'timestamp': datetime.now().isoformat(),
            'ai_powered': False,
            'model_used': None,
        })


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════╗
║  🎮 AI GAME PREDICTOR - 3-COLOR & 3-BOMB MODE ║
╚════════════════════════════════════════════════╝

🎯 Features:
   • Predicts TOP 3 COLORS with confidence levels
   • Minesweeper with EXACTLY 3 BOMBS
   • Pattern Recognition & Probability Theory
   
📡 Server: http://localhost:5000
🤖 AI Client: {}
🧠 Active Model: {}

{}Press CTRL+C to stop
    """.format(
        "✅ READY" if client else "❌ NOT CONFIGURED",
        active_model if active_model else "None",
        "⚠️  Set GROQ_API_KEY environment variable!\n\n" if not client else ""
    ))
    
    app.run(debug=True, host='0.0.0.0', port=5000)