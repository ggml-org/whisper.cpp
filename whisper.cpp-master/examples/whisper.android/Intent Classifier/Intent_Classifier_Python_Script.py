%%writefile enhanced_nlu_pipeline.py
# Enhanced Precise Slot Extractor with Improved Pattern Recognition
# Better handling of complex phrases, synonyms, and contextual clues

import json
import re
from collections import defaultdict
import spacy
from datetime import datetime

class EnhancedSlotExtractor:
    """Enhanced slot extractor with better pattern recognition and context awareness"""

    def __init__(self, training_data=None):
        self.intent_slot_templates = {}
        self.slot_value_examples = defaultdict(set)
        self.synonym_mappings = self._build_synonym_mappings()
        self.context_patterns = self._build_context_patterns()

        # Try to load spacy model for better NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.use_spacy = True
            print("✓ SpaCy model loaded for enhanced NLP")
        except:
            self.nlp = None
            self.use_spacy = False
            print("⚠️ SpaCy not available, using regex-only approach")

        if training_data:
            self.analyze_intent_slot_patterns(training_data)

    def _build_synonym_mappings(self):
        """Build comprehensive synonym mappings for better slot recognition"""
        return {
            'metric_synonyms': {
                # Steps synonyms
                'steps': ['steps', 'step', 'walk', 'walked', 'walking', 'footsteps', 'pace'],
                'distance': ['distance', 'walked', 'walk', 'miles', 'kilometers', 'km', 'far'],
                'calories': ['calories', 'calorie', 'kcal', 'energy', 'burned', 'burn'],
                'heart rate': [
                    'heart rate', 'heartrate', 'hr', 'pulse', 'bpm',
                    'heart beat', 'heartbeat', 'cardiac'
                ],
                'sleep': ['sleep', 'slept', 'sleeping', 'rest', 'rested'],
                'sleep score': ['sleep score', 'sleep quality', 'sleep rating'],
                'spo2': ['spo2', 'oxygen', 'blood oxygen', 'o2', 'saturation'],
                'weight': ['weight', 'weigh', 'kg', 'pounds', 'lbs'],
                'stress': ['stress', 'stressed', 'anxiety', 'tension'],
                'activity': ['activity', 'exercise', 'workout', 'training']
            },

            'time_synonyms': {
                'today': ['today', 'now', 'currently', 'this day', 'present'],
                'yesterday': ['yesterday', 'last day'],
                'last night': ['last night', 'night', 'overnight', 'during sleep'],
                'this morning': ['this morning', 'morning', 'am'],
                'this week': ['this week', 'current week', 'weekly'],
                'last week': ['last week', 'past week', 'previous week'],
                'this month': ['this month', 'current month', 'monthly'],
                'recent': ['recent', 'recently', 'latest', 'current']
            },

            'qualifier_synonyms': {
                'minimum': ['minimum', 'min', 'lowest', 'least', 'bottom'],
                'maximum': ['maximum', 'max', 'highest', 'most', 'peak', 'top'],
                'average': ['average', 'avg', 'mean', 'typical', 'normal'],
                'total': ['total', 'sum', 'overall', 'complete', 'entire']
            },

            'action_synonyms': {
                'how much': ['how much', 'how many', 'what amount', 'quantity'],
                'show me': ['show', 'display', 'view', 'see', 'check'],
                'what is': ['what is', 'what was', 'what are', 'tell me']
            }
        }

    def _build_context_patterns(self):
        """Build context-aware patterns for better extraction"""
        return {
            'walking_patterns': [
                r'\b(?:how\s+(?:much|far|many))\s+(?:did\s+i\s+)?(?:walk|walked|walking|steps?)\b',
                r'\b(?:distance|steps?|walking)\s+(?:yesterday|today|last\s+night)\b',
                r'\b(?:my|i)\s+(?:walk|walked|walking|steps?)\b'
            ],

            'heart_rate_patterns': [
                r'\b(?:heart\s+rate|heartrate|pulse|hr)\s+(?:was|is)?\b',
                r'\b(?:minimum|maximum|average|lowest|highest)\s+(?:heart\s+rate|pulse|hr)\b',
                r'\b(?:my|what)\s+(?:heart\s+rate|pulse)\b'
            ],

            'time_context_patterns': [
                r'\blast\s+night\b',
                r'\byesterday\s+(?:morning|afternoon|evening)\b',
                r'\bthis\s+(?:morning|afternoon|evening|week|month)\b',
                r'\b(?:during|while)\s+(?:sleep|sleeping)\b'
            ]
        }

    def analyze_intent_slot_patterns(self, data):
        """Enhanced analysis with better pattern recognition"""
        intent_slots = defaultdict(set)
        slot_examples = defaultdict(list)

        for item in data:
            intent = item['intent']
            slots = item.get('slots', {})
            text = item['text']

            for slot_name in slots.keys():
                intent_slots[intent].add(slot_name)

            slot_examples[intent].append({
                'text': text,
                'slots': slots
            })

            for slot_name, slot_value in slots.items():
                self.slot_value_examples[slot_name].add(str(slot_value))

        for intent, slot_names in intent_slots.items():
            self.intent_slot_templates[intent] = {
                'required_slots': list(slot_names),
                'examples': slot_examples[intent]
            }

        print(f"📋 Enhanced Intent-Slot Analysis:")
        for intent, info in self.intent_slot_templates.items():
            print(f"  {intent}: {info['required_slots']}")

    def extract_slots_for_intent(self, text, intent):
        """Enhanced slot extraction with better pattern matching"""
        text_lower = text.lower()
        slots = {}

        if intent not in self.intent_slot_templates:
            return slots

        required_slots = self.intent_slot_templates[intent]['required_slots']

        # Pre-process text for better matching
        processed_text = self._preprocess_text(text_lower)

        # Extract each required slot with enhanced methods
        for slot_name in required_slots:
            value = self._extract_single_slot_enhanced(processed_text, text_lower, slot_name, intent)
            if value is not None:
                slots[slot_name] = value

        # Post-process to add missing contextual slots
        slots = self._add_contextual_slots(text_lower, intent, slots)

        return slots

    def _preprocess_text(self, text):
        """Preprocess text for better pattern matching"""
        # Normalize common variations
        text = re.sub(r'\bhow\s+much\s+did\s+i\s+walk', 'walking distance', text)
        text = re.sub(r'\bhow\s+many\s+steps', 'steps', text)
        text = re.sub(r'\bhow\s+far\s+did\s+i\s+walk', 'walking distance', text)
        text = re.sub(r'\bwhat\s+is\s+my', 'my', text)
        text = re.sub(r'\bshow\s+me\s+my', 'my', text)

        return text

    def _add_contextual_slots(self, text, intent, existing_slots):
        """Add missing slots based on context clues"""
        # For QueryPoint intent, try to infer missing metric
        if intent == 'QueryPoint' and 'metric' not in existing_slots:
            inferred_metric = self._infer_metric_from_context(text)
            if inferred_metric:
                existing_slots['metric'] = inferred_metric

        # Add qualifier if detected but not extracted
        if intent == 'QueryPoint' and 'qualifier' not in existing_slots:
            qualifier = self._extract_qualifier(text)
            if qualifier:
                existing_slots['qualifier'] = qualifier

        return existing_slots

    def _infer_metric_from_context(self, text):
        """Infer metric from context when not explicitly stated"""
        inference_patterns = {
            'steps': [
                r'\b(?:walk|walked|walking)\b(?!\s+distance)',
                r'\bhow\s+much.*(?:walk|walked)\b',
                r'\bsteps?\b'
            ],
            'distance': [
                r'\bhow\s+far\b',
                r'\b(?:walk|walked|walking)\s+(?:distance|far)\b',
                r'\bdistance.*(?:walk|walked)\b',
                r'\bkilometers?\b|\bmiles?\b|\bkm\b'
            ],
            'heart rate': [
                r'\bheart\s+rate\b|\bheartrate\b|\bpulse\b|\bhr\b|\bbpm\b'
            ],
            'calories': [
                r'\bcalories?\b|\bkcal\b|\benergy\b|\bburn\b'
            ]
        }

        for metric, patterns in inference_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return metric

        return None

    def _extract_single_slot_enhanced(self, processed_text, original_text, slot_name, intent):
        """Enhanced single slot extraction with better patterns"""

        extraction_methods = {
            'metric': lambda: self._extract_metric_enhanced(processed_text, original_text),
            'time_ref': lambda: self._extract_time_ref_enhanced(original_text),
            'unit': lambda: self._extract_unit_enhanced(original_text),
            'threshold': lambda: self._extract_threshold_enhanced(original_text),
            'target': lambda: self._extract_target_enhanced(original_text),
            'value': lambda: self._extract_value_enhanced(original_text, intent),
            'qualifier': lambda: self._extract_qualifier(original_text),
            'feature': lambda: self._extract_feature_enhanced(original_text),
            'state': lambda: self._extract_state_enhanced(original_text),
            'action': lambda: self._extract_action_enhanced(original_text),
            'tool': lambda: self._extract_tool_enhanced(original_text),
            'activity_type': lambda: self._extract_activity_type_enhanced(original_text),
            'app': lambda: self._extract_app_enhanced(original_text),
            'contact': lambda: self._extract_contact_enhanced(original_text),
            'location': lambda: self._extract_location_enhanced(original_text),
            'attribute': lambda: self._extract_attribute_enhanced(original_text),
            'type': lambda: self._extract_type_enhanced(original_text),
            'period': lambda: self._extract_period_enhanced(original_text),
            'event_type': lambda: self._extract_event_type_enhanced(original_text)
        }

        if slot_name in extraction_methods:
            return extraction_methods[slot_name]()

        return None

    def _extract_metric_enhanced(self, processed_text, original_text):
        """Enhanced metric extraction with synonyms and context"""
        # Direct synonym matching
        for metric, synonyms in self.synonym_mappings['metric_synonyms'].items():
            for synonym in synonyms:
                if re.search(rf'\b{re.escape(synonym)}\b', original_text):
                    return metric

        # Context-based inference
        if re.search(r'\b(?:walk|walked|walking)\b', original_text):
            if re.search(r'\b(?:far|distance|km|miles?)\b', original_text):
                return 'distance'
            else:
                return 'steps'

        return None

    def _extract_time_ref_enhanced(self, text):
        """Enhanced time reference extraction"""
        # More specific time patterns
        time_patterns = {
            'last night': r'\blast\s+night\b|\bduring\s+(?:the\s+)?night\b|\bovernight\b',
            'yesterday': r'\byesterday\b(?!\s+night)',
            'yesterday morning': r'\byesterday\s+morning\b',
            'yesterday afternoon': r'\byesterday\s+afternoon\b',
            'yesterday evening': r'\byesterday\s+evening\b',
            'today': r'\btoday\b|\bnow\b|\bcurrently\b|\bthis\s+day\b',
            'this morning': r'\bthis\s+morning\b|\bmorning\b',
            'this afternoon': r'\bthis\s+afternoon\b|\bafternoon\b',
            'this evening': r'\bthis\s+evening\b|\bevening\b',
            'this week': r'\bthis\s+week\b|\bcurrent\s+week\b|\bweekly\b',
            'last week': r'\blast\s+week\b|\bpast\s+week\b|\bprevious\s+week\b',
            'this month': r'\bthis\s+month\b|\bcurrent\s+month\b|\bmonthly\b'
        }

        # Check most specific patterns first
        for time_ref, pattern in time_patterns.items():
            if re.search(pattern, text):
                return time_ref

        return None

    def _extract_qualifier(self, text):
        """Extract qualifiers like minimum, maximum, average"""
        qualifier_patterns = {
            'minimum': r'\b(?:minimum|min|lowest|least|bottom|smallest)\b',
            'maximum': r'\b(?:maximum|max|highest|most|peak|top|largest)\b',
            'average': r'\b(?:average|avg|mean|typical|normal)\b',
            'total': r'\b(?:total|sum|overall|complete|entire|all)\b'
        }

        for qualifier, pattern in qualifier_patterns.items():
            if re.search(pattern, text):
                return qualifier

        return None

    def _extract_unit_enhanced(self, text):
        """Enhanced unit extraction with better context awareness"""
        unit_patterns = {
            'bpm': r'\b(?:bpm|beats?\s+per\s+minute)\b',
            'kg': r'\b(?:kg|kilogram|kgs)\b',
            'pounds': r'\b(?:pounds?|lbs?|lb)\b',
            'km': r'\b(?:km|kilometer|kilometres?)\b',
            'miles': r'\b(?:miles?|mi)\b',
            'kcal': r'\b(?:kcal|calories?)\b',
            'hours': r'\b(?:hours?|hrs?|h)\b',
            'minutes': r'\b(?:min|minutes?|mins)\b',
            'percent': r'\b(?:percent|%)\b',
            'steps': r'\bsteps?\b'
        }

        for unit, pattern in unit_patterns.items():
            if re.search(pattern, text):
                return unit

        # Context-based unit inference
        if re.search(r'\b(?:heart\s+rate|pulse|hr)\b', text):
            return 'bpm'
        elif re.search(r'\b(?:weight|weigh)\b', text):
            return 'kg'
        elif re.search(r'\bsteps?\b', text):
            return 'count'

        return None

    def _extract_threshold_enhanced(self, text):
        """Enhanced threshold extraction"""
        # Look for numbers in context
        number_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(?:bpm|kg|km|miles?|percent|%|hours?|minutes?)\b',
            r'\b(?:above|over|exceeds?|higher\s+than)\s+(\d+(?:\.\d+)?)\b',
            r'\b(?:below|under|less\s+than|lower\s+than)\s+(\d+(?:\.\d+)?)\b'
        ]

        for pattern in number_patterns:
            match = re.search(pattern, text)
            if match:
                return int(float(match.group(1)))

        # Fallback to any number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        return int(float(numbers[0])) if numbers else None

    def _extract_target_enhanced(self, text):
        """Enhanced target extraction for goals"""
        # Look for goal-setting patterns
        goal_patterns = [
            r'\b(?:goal|target|aim).*?(\d+(?:\.\d+)?)\b',
            r'\b(?:set|change|update).*?(?:to|at)\s*(\d+(?:\.\d+)?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:steps?|kg|km|hours?|minutes?|calories?)\b'
        ]

        for pattern in goal_patterns:
            match = re.search(pattern, text)
            if match:
                return int(float(match.group(1)))

        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        return int(float(numbers[0])) if numbers else None

    def _extract_value_enhanced(self, text, intent):
        """Enhanced value extraction based on context"""
        if intent == 'LogEvent':
            weight_pattern = r'\b(\d+(?:\.\d+)?)\s*(?:kg|pounds?|lbs?)\b'
            match = re.search(weight_pattern, text)
            if match:
                return float(match.group(1))

        elif intent == 'TimerStopwatch':
            time_patterns = [
                r'\b(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\b',
                r'\b(\d+)\s*(?:min|minutes?)\b',
                r'\b(\d+)\s*(?:hours?|hrs?)\b'
            ]

            for pattern in time_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)

        # General number extraction
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        return int(float(numbers[0])) if numbers else None

    # Enhanced versions of other extraction methods
    def _extract_feature_enhanced(self, text):
        """Enhanced feature extraction"""
        features = {
            'do not disturb': r'\b(?:do\s+not\s+disturb|dnd|silent\s+mode)\b',
            'AOD': r'\b(?:AOD|always\s+on\s+display|always-on)\b',
            'raise to wake': r'\b(?:raise\s+to\s+wake|lift\s+to\s+wake|tap\s+to\s+wake)\b',
            'vibration': r'\b(?:vibration|vibrate|haptic)\b',
            'brightness': r'\b(?:brightness|screen\s+brightness)\b',
            'volume': r'\b(?:volume|sound\s+level)\b'
        }

        for feature, pattern in features.items():
            if re.search(pattern, text):
                return feature
        return None

    def _extract_state_enhanced(self, text):
        """Enhanced state extraction"""
        if re.search(r'\b(?:turn\s+on|enable|activate|switch\s+on|start)\b', text):
            return 'on'
        elif re.search(r'\b(?:turn\s+off|disable|deactivate|switch\s+off|stop)\b', text):
            return 'off'
        elif re.search(r'\b(?:increase|up|higher|raise)\b', text):
            return 'increase'
        elif re.search(r'\b(?:decrease|down|lower|reduce)\b', text):
            return 'decrease'
        return None

    def _extract_action_enhanced(self, text):
        """Enhanced action extraction"""
        actions = {
            'set': r'\b(?:set|setup|configure)\b',
            'start': r'\b(?:start|begin|initiate|launch)\b',
            'stop': r'\b(?:stop|end|finish|terminate)\b',
            'call': r'\b(?:call|phone|dial)\b',
            'message': r'\b(?:message|text|sms|send)\b',
            'open': r'\b(?:open|launch|start|show)\b',
            'check': r'\b(?:check|verify|examine|look)\b',
            'measure': r'\b(?:measure|test|record)\b'
        }

        for action, pattern in actions.items():
            if re.search(pattern, text):
                return action
        return None

    def _extract_tool_enhanced(self, text):
        """Enhanced timer tool extraction"""
        if re.search(r'\b(?:alarm|wake\s+up|wake\s+me)\b', text):
            return 'alarm'
        elif re.search(r'\b(?:timer|countdown)\b', text):
            return 'timer'
        elif re.search(r'\b(?:stopwatch|chronometer)\b', text):
            return 'stopwatch'
        return None

    def _extract_activity_type_enhanced(self, text):
        """Enhanced activity type extraction"""
        activities = {
            'outdoor run': r'\b(?:outdoor\s+)?(?:run|running|jog|jogging)\b',
            'indoor cycling': r'\b(?:indoor\s+)?(?:cycling|bike|biking)\b',
            'swimming': r'\b(?:swim|swimming|pool)\b',
            'yoga': r'\b(?:yoga|meditation|stretch)\b',
            'walking': r'\b(?:walk|walking|hike|hiking)\b',
            'workout': r'\b(?:workout|exercise|training|gym)\b'
        }

        for activity, pattern in activities.items():
            if re.search(pattern, text):
                return activity
        return None

    def _extract_app_enhanced(self, text):
        """Enhanced app extraction"""
        apps = {
            'weather': r'\b(?:weather|forecast|temperature|rain|snow)\b',
            'settings': r'\b(?:settings?|preferences|config)\b',
            'health': r'\b(?:health|fitness|medical)\b',
            'calendar': r'\b(?:calendar|schedule|appointment)\b'
        }

        for app, pattern in apps.items():
            if re.search(pattern, text):
                return app
        return None

    def _extract_contact_enhanced(self, text):
        """Enhanced contact extraction"""
        contacts = {
            'mom': r'\b(?:mom|mother|mama|mum)\b',
            'dad': r'\b(?:dad|father|papa|pop)\b',
            'sister': r'\b(?:sister|sis)\b',
            'brother': r'\b(?:brother|bro)\b'
        }

        for contact, pattern in contacts.items():
            if re.search(pattern, text):
                return contact
        return None

    def _extract_location_enhanced(self, text):
        """Enhanced location extraction"""
        locations = {
            'london': r'\b(?:london|uk|england)\b',
            'bangalore': r'\b(?:bangalore|bengaluru|blr)\b',
            'mumbai': r'\b(?:mumbai|bombay)\b',
            'delhi': r'\b(?:delhi|new\s+delhi)\b'
        }

        for location, pattern in locations.items():
            if re.search(pattern, text):
                return location

        return 'current location'  # Default

    def _extract_attribute_enhanced(self, text):
        """Enhanced weather attribute extraction"""
        attributes = {
            'forecast': r'\b(?:forecast|prediction|outlook)\b',
            'temperature': r'\b(?:temperature|temp|hot|cold|warm|cool)\b',
            'rain': r'\b(?:rain|raining|shower|umbrella|wet)\b',
            'humidity': r'\b(?:humidity|humid|moisture)\b',
            'air quality': r'\b(?:air\s+quality|aqi|pollution|smog)\b'
        }

        for attr, pattern in attributes.items():
            if re.search(pattern, text):
                return attr
        return None

    def _extract_type_enhanced(self, text):
        """Enhanced threshold type extraction"""
        if re.search(r'\b(?:above|over|exceed|higher|more\s+than|greater)\b', text):
            return 'high'
        elif re.search(r'\b(?:below|under|less\s+than|lower|drops?)\b', text):
            return 'low'
        return None

    def _extract_period_enhanced(self, text):
        """Enhanced period extraction"""
        periods = {
            'daily': r'\b(?:daily|every\s+day|each\s+day)\b',
            'weekly': r'\b(?:weekly|every\s+week|each\s+week)\b',
            'monthly': r'\b(?:monthly|every\s+month|each\s+month)\b'
        }

        for period, pattern in periods.items():
            if re.search(pattern, text):
                return period
        return None

    def _extract_event_type_enhanced(self, text):
        """Enhanced event type extraction"""
        if re.search(r'\b(?:weight|weigh|kg|pounds)\b', text):
            return 'weight'
        elif re.search(r'\b(?:menstrual|period|cycle)\b', text):
            return 'menstrual cycle'
        return None

    def evaluate_enhanced_accuracy(self, training_data):
        """Evaluate the enhanced extractor accuracy"""
        total_slots = 0
        correct_slots = 0
        slot_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

        print("\nEvaluating enhanced extractor on training data...")

        for item in training_data:
            text = item['text']
            intent = item['intent']
            true_slots = item.get('slots', {})

            predicted_slots = self.extract_slots_for_intent(text, intent)

            for slot_name, true_value in true_slots.items():
                total_slots += 1
                slot_accuracy[slot_name]['total'] += 1

                if slot_name in predicted_slots:
                    pred_value = predicted_slots[slot_name]

                    if self._values_match(true_value, pred_value):
                        correct_slots += 1
                        slot_accuracy[slot_name]['correct'] += 1

        overall_accuracy = correct_slots / total_slots if total_slots > 0 else 0

        print(f"Enhanced slot accuracy: {overall_accuracy:.1%} ({correct_slots}/{total_slots})")
        print("\nAccuracy by slot type:")

        for slot_name, stats in sorted(slot_accuracy.items()):
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {slot_name:15}: {acc:.1%} ({stats['correct']:2d}/{stats['total']:2d})")

        return overall_accuracy, slot_accuracy

    def _values_match(self, true_value, pred_value):
        """Check if predicted and true values match"""
        if isinstance(true_value, (int, float)) and isinstance(pred_value, (int, float)):
            return abs(float(true_value) - float(pred_value)) < 0.01
        return str(true_value).lower() == str(pred_value).lower()


def test_enhanced_extractor():
    """Test the enhanced slot extractor with your problem cases"""
    print("🚀 TESTING ENHANCED SLOT EXTRACTOR")
    print("=" * 60)

    # Load training data and create enhanced extractor
    with open('noise_ai_training_dataset_cleaned_1.json', 'r') as f:
        training_data = json.load(f)

    extractor = EnhancedSlotExtractor(training_data)

    # Test the problematic cases
    test_cases = [
        ("How much did I walked yesterday?", "QueryPoint"),
        ("What is my minimum heart rate last night?", "QueryPoint"),
        ("How far did I walk today?", "QueryPoint"),
        ("Show me my total steps for this week", "QueryPoint"),
        ("What was my average sleep score last night?", "QueryPoint"),
        ("How many calories did I burn yesterday?", "QueryPoint"),
        ("What is my maximum heart rate this morning?", "QueryPoint"),
        ("Set my daily step goal to 15000", "SetGoal"),
        ("Alert me when heart rate goes above 180 bpm", "SetThreshold")
    ]

    print("Testing enhanced extraction on problem cases:")
    print("-" * 60)

    for text, intent in test_cases:
        slots = extractor.extract_slots_for_intent(text, intent)
        print(f"Text: '{text}'")
        print(f"Intent: {intent}")
        print(f"Enhanced slots: {slots}")
        print("-" * 40)

    # Evaluate on full training data
    accuracy, slot_stats = extractor.evaluate_enhanced_accuracy(training_data)

    return extractor, accuracy


def integrate_enhanced_extractor_into_pipeline():
    """Show how to integrate the enhanced extractor into the main pipeline"""
    print("🔧 INTEGRATING ENHANCED EXTRACTOR")
    print("=" * 50)

    print("""
To integrate the enhanced extractor into your main pipeline:

1. Replace PreciseSlotExtractor with EnhancedSlotExtractor in CompleteNLUPipeline
2. Update the train_slot_extractor method
3. The rest of the pipeline remains the same

Here's the updated method for CompleteNLUPipeline class:
""")

    return EnhancedSlotExtractor


# Updated CompleteNLUPipeline class with enhanced extractor
class CompleteNLUPipelineV2:
    """Updated NLU Pipeline with Enhanced Slot Extractor"""

    def __init__(self, json_file_path=None):
        self.json_file_path = json_file_path
        self.training_data = None
        self.sentence_model = None
        self.intent_classifier = None
        self.intent_encoder = None
        self.slot_extractor = None
        self.training_metrics = {}

    def load_and_prepare_data(self):
        """Load and prepare training data - same as before"""
        print("📊 Loading training data...")

        with open(self.json_file_path, 'r') as f:
            self.training_data = json.load(f)

        print(f"✓ Loaded {len(self.training_data)} training examples")
        return self.training_data

    def train_intent_classifier(self):
        """Train intent classifier - same as before"""
        print("\n🎯 Training Intent Classifier...")

        from sentence_transformers import SentenceTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split, cross_val_score

        # Load sentence transformer
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Prepare data
        texts = [item['text'] for item in self.training_data]
        intents = [item['intent'] for item in self.training_data]

        # Encode texts
        X = self.sentence_model.encode(texts, show_progress_bar=True)

        # Encode labels
        self.intent_encoder = LabelEncoder()
        y = self.intent_encoder.fit_transform(intents)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train best model
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_score = cv_scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_model = model

        self.intent_classifier = best_model
        self.intent_classifier.fit(X_train, y_train)

        test_score = self.intent_classifier.score(X_test, y_test)
        print(f"✓ Intent classifier trained - Test accuracy: {test_score:.3f}")

        self.training_metrics['intent_test_accuracy'] = test_score
        return self.intent_classifier

    def train_slot_extractor(self):
        """Train the ENHANCED precise slot extractor"""
        print("\n🏷️  Training Enhanced Slot Extractor...")

        # Use Enhanced version instead of basic version
        self.slot_extractor = EnhancedSlotExtractor(self.training_data)

        # Evaluate enhanced accuracy
        accuracy, slot_stats = self.slot_extractor.evaluate_enhanced_accuracy(self.training_data)

        print(f"✓ Enhanced slot extraction accuracy: {accuracy:.3f}")
        self.training_metrics['enhanced_slot_accuracy'] = accuracy
        self.training_metrics['slot_breakdown'] = slot_stats

        return self.slot_extractor

    def predict(self, text):
        """Make complete NLU prediction with enhanced slot extraction"""
        if not all([self.sentence_model, self.intent_classifier, self.slot_extractor]):
            raise ValueError("Model not trained. Call train_complete_pipeline() first.")

        # Encode text
        text_embedding = self.sentence_model.encode([text])

        # Predict intent
        intent_pred = self.intent_classifier.predict(text_embedding)[0]
        intent_proba = self.intent_classifier.predict_proba(text_embedding)[0]

        intent_name = self.intent_encoder.inverse_transform([intent_pred])[0]
        confidence = float(intent_proba.max())

        # Extract slots using ENHANCED extractor
        slots = self.slot_extractor.extract_slots_for_intent(text, intent_name)

        return {
            'text': text,
            'intent': intent_name,
            'confidence': confidence,
            'slots': slots
        }

    def train_complete_pipeline(self):
        """Train the complete enhanced NLU pipeline"""
        print("🚀 TRAINING ENHANCED NLU PIPELINE")
        print("=" * 60)

        start_time = datetime.now()

        # Load data
        self.load_and_prepare_data()

        # Train intent classifier
        self.train_intent_classifier()

        # Train enhanced slot extractor
        self.train_slot_extractor()

        training_time = datetime.now() - start_time

        print(f"\n🎉 ENHANCED TRAINING COMPLETED!")
        print(f"⏱️  Total training time: {training_time}")
        print(f"📊 Enhanced metrics: {self.training_metrics}")

        return self


def train_enhanced_nlu_system(json_file_path):
    """Train the enhanced NLU system with better slot extraction"""
    print("🚀 TRAINING ENHANCED NLU SYSTEM")
    print("=" * 70)

    # Initialize enhanced pipeline
    pipeline = CompleteNLUPipelineV2(json_file_path)

    # Train complete system
    trained_pipeline = pipeline.train_complete_pipeline()

    # Test the problematic cases
    print("\n🧪 Testing Enhanced System on Problem Cases:")
    print("-" * 50)

    problem_cases = [
        "How much did I walked yesterday?",
        "What is my minimum heart rate last night?",
        "How far did I walk today?",
        "Show me my total steps for this week",
        "What was my average sleep score last night?",
        "How many calories did I burn yesterday?"
    ]

    for text in problem_cases:
        result = trained_pipeline.predict(text)
        print(f"'{text}'")
        print(f"  → {result['intent']} ({result['confidence']:.1%})")
        print(f"  → Slots: {result['slots']}")
        print()

    return trained_pipeline


class EnhancedModelTester:
    """Enhanced testing class for the improved pipeline"""

    def __init__(self, enhanced_pipeline):
        self.pipeline = enhanced_pipeline
        self.test_results = []

    def run_comprehensive_slot_tests(self):
        """Run comprehensive tests focusing on slot extraction improvements"""
        print("🔍 COMPREHENSIVE SLOT EXTRACTION TESTING")
        print("=" * 70)

        test_categories = {
            'Walking/Steps Queries': [
                "How much did I walked yesterday?",
                "How many steps did I take today?",
                "What's my step count for this week?",
                "How far did I walk last night?",
                "What is the average steps for last week?"
            ],

            'Heart Rate with Qualifiers': [
                "What is my minimum heart rate last night?",
                "Show me my maximum heart rate today",
                "What was my average heart rate yesterday?",
                "My lowest heart rate this morning",
                "Peak heart rate during workout"
            ],

            'Complex Time References': [
                "My sleep score during last night",
                "Heart rate yesterday morning",
                "Steps count this afternoon",
                "Calories burned yesterday evening",
                "Stress level this week"
            ],

            'Goal Setting with Units': [
                "Set my daily step goal to 15000 steps",
                "Change my weight target to 75 kg",
                "Update my calorie goal to 2500 kcal",
                "Set walking distance goal to 5 km"
            ],

            'Thresholds with Qualifiers': [
                "Alert me if heart rate goes above 180 bpm",
                "Notify when my weight drops below 65 kg",
                "Warn if stress exceeds maximum level",
                "Tell me when spo2 falls under 95 percent"
            ]
        }

        category_scores = {}

        for category, test_cases in test_categories.items():
            print(f"\n🎯 Testing: {category}")
            print("-" * 40)

            correct_extractions = 0
            high_confidence = 0
            total_tests = len(test_cases)

            for text in test_cases:
                try:
                    result = self.pipeline.predict(text)

                    # Count successful extractions
                    if result['slots']:
                        correct_extractions += 1

                    if result['confidence'] > 0.7:
                        high_confidence += 1

                    print(f"  '{text}'")
                    print(f"    Intent: {result['intent']} ({result['confidence']:.1%})")
                    print(f"    Slots: {result['slots']}")
                    print()

                    self.test_results.append({
                        'category': category,
                        'text': text,
                        'result': result
                    })

                except Exception as e:
                    print(f"  ERROR: {text} - {e}")

            extraction_rate = correct_extractions / total_tests
            confidence_rate = high_confidence / total_tests

            category_scores[category] = {
                'extraction_rate': extraction_rate,
                'confidence_rate': confidence_rate,
                'total_tests': total_tests
            }

            print(f"  📊 Slot Extraction Rate: {extraction_rate:.1%}")
            print(f"  📊 High Confidence Rate: {confidence_rate:.1%}")

        # Overall summary
        print("\n" + "=" * 70)
        print("📊 ENHANCED TESTING SUMMARY")
        print("=" * 70)

        overall_extraction = sum(s['extraction_rate'] * s['total_tests']
                               for s in category_scores.values()) / sum(s['total_tests']
                               for s in category_scores.values())

        overall_confidence = sum(s['confidence_rate'] * s['total_tests']
                               for s in category_scores.values()) / sum(s['total_tests']
                               for s in category_scores.values())

        print(f"Overall Slot Extraction Rate: {overall_extraction:.1%}")
        print(f"Overall High Confidence Rate: {overall_confidence:.1%}")

        print(f"\nCategory Breakdown:")
        for category, scores in category_scores.items():
            print(f"  {category:25}: Extract={scores['extraction_rate']:.1%}, Conf={scores['confidence_rate']:.1%}")

        return category_scores

    def compare_with_basic_extractor(self, basic_pipeline):
        """Compare enhanced vs basic extractor performance"""
        print("\n⚖️  ENHANCED vs BASIC EXTRACTOR COMPARISON")
        print("=" * 60)

        comparison_tests = [
            "How much did I walked yesterday?",
            "What is my minimum heart rate last night?",
            "Show me my total calories burned today",
            "Set my daily step goal to 12000",
            "Alert me if heart rate exceeds 150 bpm",
        ]

        enhanced_better = 0
        basic_better = 0

        print("Test Results Comparison:")
        print("-" * 60)

        for text in comparison_tests:
            enhanced_result = self.pipeline.predict(text)
            basic_result = basic_pipeline.predict(text)

            print(f"Text: '{text}'")
            print(f"Enhanced: {enhanced_result['intent']} ({enhanced_result['confidence']:.1%}) - {enhanced_result['slots']}")
            print(f"Basic:    {basic_result['intent']} ({basic_result['confidence']:.1%}) - {basic_result['slots']}")

            # Compare slot completeness
            enhanced_slots = len(enhanced_result['slots'])
            basic_slots = len(basic_result['slots'])

            if enhanced_slots > basic_slots:
                enhanced_better += 1
                print("  ✓ Enhanced wins (more slots)")
            elif basic_slots > enhanced_slots:
                basic_better += 1
                print("  ✗ Basic wins (more slots)")
            else:
                print("  = Tie (same slot count)")

            print("-" * 40)

        print(f"\n📊 Comparison Summary:")
        print(f"  Enhanced better: {enhanced_better}/{len(comparison_tests)}")
        print(f"  Basic better: {basic_better}/{len(comparison_tests)}")

        return enhanced_better, basic_better


# Quick setup and testing functions
def quick_test_enhanced_system():
    """Quick function to test the enhanced system"""
    print("🚀 QUICK TEST OF ENHANCED SYSTEM")
    print("=" * 50)

    # Train enhanced system
    enhanced_pipeline = train_enhanced_nlu_system('noise_ai_training_dataset_cleaned_1.json')

    # Run comprehensive tests
    tester = EnhancedModelTester(enhanced_pipeline)
    results = tester.run_comprehensive_slot_tests()

    return enhanced_pipeline, results

def interactive_testing(self):
        """Interactive testing mode"""
        print("\n🎮 INTERACTIVE TESTING MODE")
        print("=" * 70)
        print("Enter your voice commands to test the NLU pipeline.")
        print("Type 'quit' to exit.")
        print("=" * 70)

        while True:
            user_input = input("\nEnter command: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            try:
                result = self.nlu_pipeline.predict(user_input)

                print(f"\n📝 Results for: '{user_input}'")
                print(f"🎯 Intent: {result['intent']}")
                print(f"📊 Confidence: {result['confidence']:.1%}")

                if result['slots']:
                    print(f"🏷️  Slots:")
                    for slot_name, slot_value in result['slots'].items():
                        print(f"   {slot_name}: {slot_value}")
                else:
                    print(f"🏷️  Slots: (none detected)")

            except Exception as e:
                print(f"❌ Error: {e}")

def save_enhanced_model(enhanced_pipeline, model_name="enhanced_nlu_model"):
    """Save the enhanced model with better slot extraction"""
    print(f"\n💾 SAVING ENHANCED MODEL: {model_name}")
    print("=" * 50)

    from datetime import datetime
    import pickle
    import json
    import os

    # Create enhanced model manager
    class EnhancedModelManager:
        def __init__(self):
            self.model_dir = "./saved_enhanced_models"
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        def save_enhanced_model(self, pipeline, name):
            model_path = os.path.join(self.model_dir, name)
            os.makedirs(model_path, exist_ok=True)

            try:
                # Save all components
                if pipeline.sentence_model:
                    sentence_path = os.path.join(model_path, "sentence_model")
                    pipeline.sentence_model.save(sentence_path)

                if pipeline.intent_classifier:
                    classifier_path = os.path.join(model_path, "intent_classifier.pkl")
                    with open(classifier_path, 'wb') as f:
                        pickle.dump(pipeline.intent_classifier, f)

                if pipeline.intent_encoder:
                    encoder_path = os.path.join(model_path, "intent_encoder.pkl")
                    with open(encoder_path, 'wb') as f:
                        pickle.dump(pipeline.intent_encoder, f)

                # Save enhanced slot extractor
                if pipeline.slot_extractor:
                    extractor_path = os.path.join(model_path, "enhanced_slot_extractor.pkl")
                    with open(extractor_path, 'wb') as f:
                        pickle.dump(pipeline.slot_extractor, f)

                # Save metadata
                metadata = {
                    'model_type': 'enhanced_nlu_pipeline',
                    'created_date': datetime.now().isoformat(),
                    'training_metrics': pipeline.training_metrics,
                    'extractor_type': 'EnhancedSlotExtractor'
                }

                metadata_path = os.path.join(model_path, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)

                print(f"✅ Enhanced model saved successfully!")
                print(f"📁 Location: {model_path}")
                return model_path

            except Exception as e:
                print(f"❌ Error saving enhanced model: {e}")
                return None

    manager = EnhancedModelManager()
    return manager.save_enhanced_model(enhanced_pipeline, model_name)


# Instructions for using the enhanced system
print("\n🎯 ENHANCED SLOT EXTRACTION SYSTEM READY!")
print("=" * 70)
print("Key Improvements:")
print("✓ Better synonym recognition (walk/walked/walking → steps/distance)")
print("✓ Enhanced time reference extraction (last night vs yesterday)")
print("✓ Qualifier extraction (minimum/maximum/average/total)")
print("✓ Context-aware metric inference")
print("✓ Improved pattern matching with preprocessing")
print("✓ Better handling of complex phrases")
print("")
print("Usage:")
print("1. enhanced_pipeline = train_enhanced_nlu_system('your_data.json')")
print("2. tester = EnhancedModelTester(enhanced_pipeline)")
print("3. results = tester.run_comprehensive_slot_tests()")
print("4. save_enhanced_model(enhanced_pipeline, 'my_enhanced_model')")
print("")
print("Or run quick test: quick_test_enhanced_system()")
print("=" * 70)
