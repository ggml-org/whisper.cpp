package com.whispercppdemo.intent

import android.util.Log
import org.json.JSONObject
import java.util.*
import java.util.regex.Pattern
import kotlin.text.RegexOption

private const val LOG_TAG = "SlotExtractor"

data class SlotExtractionResult(
    val slots: Map<String, Any>,
    val confidence: Float
)

class SlotExtractor {
    
    // Activity type to number mapping based on SportActivityName enum
    private val activityNumberMap = mapOf(
        "indoor running" to 3,
        "treadmill" to 66,
        "outdoor walking" to 2,
        "outdoor running" to 1,
        "trekking" to 4,
        "trail running" to 5,
        "hunting" to 15,
        "fishing" to 36,
        "skateboarding" to 17,
        "fencing" to 102,
        "boxing" to 56,
        "tai chi" to 59,
        "outdoor cycling" to 6,
        "indoor cycling" to 7,
        "bmx" to 14,
        "curling" to 37,
        "outdoor skating" to 19,
        "indoor skating" to 38,
        "archery" to 65,
        "equestrian" to 20,
        "cricket" to 217,
        "basketball" to 9,
        "badminton" to 12,
        "outdoor hiking" to 13,
        "golf" to 134,
        "football" to 10,
        "ballet" to 47,
        "square dance" to 49,
        "zumba" to 53,
        "mixed aerobics" to 24,
        "strength training" to 25,
        "stretching" to 26,
        "indoor fitness" to 30,
        "elliptical machine" to 34,
        "yoga" to 35,
        "climbing machine" to 27,
        "flexibility training" to 29,
        "stepper" to 31,
        "step training" to 32,
        "gymnastics" to 33,
        "freestyle" to 8,
        "core training" to 23,
        "sailing" to 16,
        "roller skating" to 18,
        "baseball" to 40,
        "bowling" to 41,
        "squash" to 42,
        "softball" to 43,
        "croquet" to 44,
        "volleyball" to 45,
        "handball" to 46,
        "pingpong" to 11,
        "belly dance" to 48,
        "street dance" to 50,
        "ballroom dancing" to 51,
        "dance" to 52,
        "cross training crossfit" to 84,
        "karate" to 55,
        "judo" to 57,
        "wrestling" to 58,
        "muay thai" to 60,
        "taekwondo" to 61,
        "martial arts" to 62,
        "free sparring" to 63,
        "pool swimming" to 200,
        "rope skipping" to 122,
        "rowing machine" to 121,
        "open water" to 201,
        "triathlon" to 123,
        "kendo" to 54,
        "pilates" to 28,
        "functional training" to 94,
        "sit ups" to 93,
        "dumbbell training" to 88,
        "barbell training" to 89,
        "weightlifting" to 90,
        "hiit" to 64,
        "deadlift" to 91,
        "darts" to 114,
        "frisbee" to 118,
        "kite flying" to 117,
        "tug of war" to 115,
        "paddleboard surfing" to 132,
        "double board skiing" to 130,
        "paddle board" to 67,
        "water polo" to 68,
        "water sports" to 69,
        "water skiing" to 70,
        "kayaking" to 71,
        "kayak rafting" to 72,
        "motorboat" to 73,
        "fin swimming" to 74,
        "diving" to 75,
        "synchronized swimming" to 76,
        "snorkeling" to 77,
        "kite surfing" to 78,
        "rock climbing" to 79,
        "parkour" to 80,
        "atv" to 81,
        "paraglider" to 82,
        "climb the stairs" to 83,
        "aerobics" to 85,
        "physical training" to 86,
        "wall ball" to 87,
        "bobby jump" to 92,
        "upper limb training" to 95,
        "lower limb training" to 96,
        "waist and abdomen training" to 97,
        "back training" to 98,
        "national dance" to 99,
        "jazz dance" to 100,
        "latin dance" to 101,
        "rugby" to 103,
        "hockey" to 104,
        "tennis" to 105,
        "billiards" to 106,
        "sepak takraw" to 108,
        "snow sports" to 109,
        "snowmobile" to 110,
        "puck" to 111,
        "snow car" to 112,
        "sled" to 113,
        "hula hoop" to 116,
        "track and field" to 119,
        "racing car" to 120,
        "mountain cycling" to 124,
        "kickboxing" to 125,
        "skiing" to 126,
        "cross country skiing" to 127,
        "snowboarding" to 128,
        "alpine skiing" to 129,
        "free exercise" to 131,
        "kabaddi" to 133,
        "indoor walking" to 135,
        "table football" to 136,
        "seven stones" to 137,
        "kho kho" to 138
    )
    
    private val intentSlotTemplates = mapOf(
        "QueryPoint" to listOf("metric"),  // time_ref, unit, and identifier can have defaults
        "SetGoal" to listOf("metric", "target"),  // unit can be inferred
        "SetThreshold" to listOf("metric", "threshold"),  // unit can be inferred
        "TimerStopwatch" to listOf("tool", "action"),  // value only required for "set timer X min"
        "ToggleFeature" to listOf("feature", "state"),
        "LogEvent" to listOf("event_type"),  // value, unit, time_ref can have defaults
        "StartActivity" to listOf("activity_type"),
        "StopActivity" to listOf("activity_type"),
        "OpenApp" to listOf("app"),  // action and target can have defaults
        "PhoneAction" to listOf("contact"), //no need of action
        "MediaAction" to listOf("action"),  // target can be optional for some actions
        "WeatherQuery" to listOf("location"),  // attribute can default to general weather
        "QueryTrend" to listOf("metric"),  // period and unit can have defaults
        "IrrelevantInput" to emptyList()  // No required slots for irrelevant input
    )
    
    fun getRequiredSlots(intent: String): List<String> {
        return intentSlotTemplates[intent] ?: emptyList()
    }
    
    // Special method to check if slots are adequate for the specific action within an intent
    fun areRequiredSlotsSatisfied(intent: String, slots: Map<String, Any>, text: String): Boolean {
        val baseRequiredSlots = getRequiredSlots(intent)
        
        // Check if all base required slots are present
        if (!baseRequiredSlots.all { slots.containsKey(it) }) {
            return false
        }
        
        // Special case handling for TimerStopwatch - value required only for "set timer" 
        if (intent == "TimerStopwatch") {
            val action = slots["action"] as? String
            val tool = slots["tool"] as? String
            
            // If setting a timer, value is required
            if ((action == "set" || action == "start") && tool == "timer") {
                if (!slots.containsKey("value")) {
                    return false
                }
            }
        }
        
        // Special case for SetGoal and SetThreshold - target/threshold must be meaningful
        if (intent == "SetGoal" && slots["target"] == null) {
            return false
        }
        
        if (intent == "SetThreshold" && slots["threshold"] == null) {
            return false
        }
        
        return true
    }
    
    // Synonym mappings based on Python implementation
    private val metricSynonyms = mapOf(
        "steps" to listOf(
            "steps", "step", "walk", "walked", "walking", "footsteps", "pace", "stride", 
            "tread", "footfall", "gait", "paces", "stroll", "strolling", "strolled", 
            "amble", "ambling", "saunter", "march", "marching", "trudge", "hike", 
            "hiking", "trek", "wander", "movement", "activity", "moves"
        ),
        "distance" to listOf(
            "distance", "walked", "walk", "miles", "kilometers", "km", "far", "meter", 
            "metres", "meters", "travelled", "traveled", "covered", "journey", "range", 
            "length", "span", "route", "path", "mileage", "odometer", "how far", 
            "feet", "yards"
        ),
        "calories" to listOf(
            "calories", "calorie", "kcal", "energy", "burned", "burn", "burning", 
            "burnt", "expended", "consumed", "expenditure", "kilojoules", "kj", 
            "nutrition", "intake", "food energy", "metabolic", "metabolism", 
            "fat burn", "fat burning", "cal"
        ),
        "heart rate" to listOf(
            "heart rate", "heartrate", "hr", "pulse", "bpm", "heart beat", "heartbeat", 
            "cardiac", "beats per minute", "resting heart rate", "rhr", "heart rhythm", 
            "cardiovascular", "cardio", "heart health", "ticker", "heart pulse", 
            "pulse rate", "heart monitor", "heart"
        ),
        "sleep" to listOf(
            "sleep", "slept", "sleeping", "rest", "rested", "resting", "nap", 
            "napping", "napped", "slumber", "snooze", "snoozed", "doze", "dozed", 
            "asleep", "bedtime", "night sleep", "shuteye", "zzz", "sleep time", 
            "sleep duration", "hours slept", "sleep hours"
        ),
        "sleep score" to listOf(
            "sleep score", "sleep quality", "sleep rating", "sleep performance", 
            "sleep analysis", "sleep grade", "sleep rank", "sleep level", 
            "sleep efficiency", "sleep assessment", "how well slept", "sleep health", 
            "sleep metric", "sleep stats", "sleep report", "sleep evaluation"
        ),
        "spo2" to listOf(
            "spo2", "oxygen", "blood oxygen", "o2", "saturation", "oxygen saturation", 
            "oxygen level", "oxygen levels", "o2 sat", "blood o2", "oxygen sat", 
            "pulse ox", "pulse oximetry", "oximeter", "oxygen reading", "o2 level", 
            "respiratory", "breathing", "blood oxygen level"
        ),
        "weight" to listOf(
            "weight", "weigh", "kg", "pounds", "lbs", "kilogram", "kilograms", "lb", 
            "body weight", "mass", "scale", "weighed", "weighing", "bmi", 
            "body mass", "how much weigh", "weight reading", "weighted", "grams", 
            "stone", "ounces", "oz"
        ),
        "stress" to listOf(
            "stress", "stressed", "anxiety", "tension", "anxious", "worried", 
            "worry", "pressure", "strain", "overwhelmed", "nervous", "nervousness", 
            "stress level", "mental stress", "emotional stress", "burnout", 
            "stress score", "relaxation", "calm", "mental health", "wellbeing"
        )
    )
    
    private val timeSynonyms = mapOf(
        "today" to listOf(
            "today", "now", "currently", "this day", "present", "right now", 
            "at present", "so far today", "today's", "current day", "as of today", 
            "till now", "up to now", "at the moment", "presently", "at this time", 
            "this very day", "the present day", "nowadays", "in the present", 
            "for today", "on this day", "todays", "since midnight"
        ),
        "yesterday" to listOf(
            "yesterday", "last day", "previous day", "day before", "1 day ago", 
            "one day ago", "a day ago", "the day before", "prior day", 
            "the previous day", "24 hours ago", "yesterdays", "yesterday's", 
            "the other day", "day prior", "the last day", "past day", 
            "the day that was", "the preceding day", "most recent day", 
            "the latest day", "just yesterday", "only yesterday", "back yesterday"
        ),
        "last night" to listOf(
            "last night", "night", "overnight", "during sleep", "while sleeping", 
            "nighttime", "night time", "at night", "during the night", 
            "throughout the night", "all night", "past night", "previous night", 
            "the other night", "last evening", "yesterday night", "yesterday evening", 
            "late yesterday", "after dark", "hours of sleep", "sleeping hours", 
            "bedtime", "sleep time", "in bed", "whilst asleep", "sleep period"
        ),
        "this morning" to listOf(
            "this morning", "morning", "am", "early today", "earlier today", 
            "this am", "today morning", "in the morning", "morning time", 
            "mornings", "early hours", "before noon", "dawn", "daybreak", 
            "sunrise", "first thing", "early on", "at dawn", "morning hours", 
            "start of day", "beginning of day", "waking hours", "after waking",
            "upon waking", "since waking"
        ),
        "this week" to listOf(
            "this week", "current week", "weekly", "so far this week", 
            "week to date", "wtd", "the week", "present week", "the current week", 
            "in this week", "for the week", "throughout the week", "during the week", 
            "over the week", "7 days", "past 7 days", "last 7 days", 
            "these 7 days", "this weeks", "this week's", "since monday", 
            "week so far", "till now this week", "up to now this week", "weekly total"
        ),
        "last week" to listOf(
            "last week", "past week", "previous week", "the week before", 
            "prior week", "1 week ago", "one week ago", "a week ago", 
            "week prior", "the last week", "the past week", "the previous week", 
            "7 days ago", "last weeks", "last week's", "the preceding week", 
            "most recent week", "latest week", "former week", "earlier week", 
            "the other week", "back last week", "during last week", "throughout last week", 
            "over last week"
        ),
        "this month" to listOf(
            "this month", "current month", "monthly", "so far this month", 
            "month to date", "mtd", "the month", "present month", 
            "the current month", "in this month", "for the month", 
            "throughout the month", "during the month", "over the month", 
            "30 days", "past 30 days", "last 30 days", "these 30 days", 
            "this months", "this month's", "since the 1st", "month so far", 
            "till now this month", "up to now this month", "monthly total"
        )
    )
    
    private val qualifierSynonyms = mapOf(
        "minimum" to listOf(
            "minimum", "min", "lowest", "least", "bottom", "bare minimum", 
            "minimal", "minimally", "rock bottom", "floor", "base", "baseline", 
            "low point", "lower", "smallest", "tiniest", "fewest", "less", 
            "lesser", "reduced", "at least", "no less than", "starting from", 
            "beginning at", "from", "low", "lows", "worst", "slowest"
        ),
        "maximum" to listOf(
            "maximum", "max", "highest", "most", "peak", "top", "maximal", 
            "maximally", "ceiling", "upper limit", "high point", "higher", 
            "greatest", "largest", "biggest", "best", "record", "all time high", 
            "at most", "no more than", "up to", "limit", "cap", "high", "highs", 
            "fastest", "extreme", "topmost", "ultimate"
        ),
        "average" to listOf(
            "average", "avg", "mean", "typical", "normal", "averaged", "averaging", 
            "median", "mid", "middle", "midpoint", "central", "moderate", 
            "standard", "regular", "usual", "common", "ordinary", "per day", 
            "daily average", "on average", "typically", "normally", "generally", 
            "approximately", "around", "about", "roughly"
        ),
        "total" to listOf(
            "total", "sum", "overall", "complete", "entire", "totaled", "totaling", 
            "all", "all time", "full", "whole", "combined", "cumulative", 
            "aggregate", "collectively", "together", "in total", "altogether", 
            "grand total", "summation", "net", "gross", "comprehensive", 
            "accumulated", "compilation", "tally", "count", "running total"
        )
    )
    
    // Pre-compiled regex patterns for better performance
    private val walkingMovementRegex = Regex("\\b(?:walk|walked|walking|stroll|strolled|strolling|hike|hiked|hiking|trek|trekked|trekking|march|marched|marching|wander|wandered|wandering|roam|roamed|roaming|amble|ambled|ambling|saunter|sauntered|sauntering|trudge|trudged|trudging|move|moved|moving|movement)\\b", RegexOption.IGNORE_CASE)
    private val distanceRegex = Regex("\\b(?:far|distance|km|kilometers|kilometre|kilometres|mile|miles|meter|meters|metre|metres|feet|ft|yard|yards|yd|long|length|covered|travelled|traveled|route|path|journey|span|range|how far|mileage)\\b", RegexOption.IGNORE_CASE)
    private val sleepRegex = Regex("\\b(?:sleep|slept|sleeping|asleep|nap|napped|napping|rest|rested|resting|snooze|snoozed|snoozing|doze|dozed|dozing|slumber|bedtime|night|overnight|bed|zzz)\\b", RegexOption.IGNORE_CASE)
    private val sleepQualityRegex = Regex("\\b(?:quality|score|rating|rate|well|badly|good|bad|poor|deep|light|efficiency|grade|rank|analysis|performance|how well)\\b", RegexOption.IGNORE_CASE)
    private val heartRegex = Regex("\\b(?:heart|cardiac|cardio|cardiovascular|pulse|beat|beats|beating|bpm|rhythm|ticker)\\b", RegexOption.IGNORE_CASE)
    private val caloriesRegex = Regex("\\b(?:calorie|calories|kcal|energy|burn|burned|burnt|burning|expend|expended|consume|consumed|intake|kilojoule|kilojoules|kj|food energy|metabolic|metabolism|fat)\\b", RegexOption.IGNORE_CASE)
    private val oxygenRegex = Regex("\\b(?:oxygen|o2|spo2|saturation|sat|blood oxygen|pulse ox|oximeter|oximetry|breathing|respiratory|respiration|air|breathe)\\b", RegexOption.IGNORE_CASE)
    private val weightRegex = Regex("\\b(?:weight|weigh|weighing|weighed|kg|kilogram|kilograms|pound|pounds|lbs|lb|body mass|bmi|body weight|mass|scale|heavy|light|stone|gram|grams|ounce|ounces|oz)\\b", RegexOption.IGNORE_CASE)
    private val stressRegex = Regex("\\b(?:stress|stressed|stressful|anxiety|anxious|tension|tense|worried|worry|worrying|pressure|pressured|strain|strained|overwhelm|overwhelmed|nervous|nervousness|burnout|mental health|relaxation|relax|calm|peace|peaceful)\\b", RegexOption.IGNORE_CASE)
    private val heartRateUnitRegex = Regex("\\b(?:heart\\s+rate|pulse|hr)\\b", RegexOption.IGNORE_CASE)
    private val weightUnitRegex = Regex("\\b(?:weight|weigh)\\b", RegexOption.IGNORE_CASE)
    private val stepsUnitRegex = Regex("\\bsteps?\\b", RegexOption.IGNORE_CASE)
    private val numberRegex = Regex("\\b(\\d+(?:\\.\\d+)?)\\b")
    private val numberSequenceRegex = Regex("\\b(\\d+(?:\\.\\d+)?)\\b")
    
    // Pre-compiled synonym patterns for better performance
    private val synonymPatterns = mutableMapOf<String, Regex>()
    
    init {
        // Pre-compile synonym patterns
        for ((metric, synonyms) in metricSynonyms) {
            val escapedSynonyms = synonyms.map { Regex.escape(it) }
            val pattern = "\\b(?:${escapedSynonyms.joinToString("|")})\\b"
            synonymPatterns[metric] = Regex(pattern, RegexOption.IGNORE_CASE)
        }
    }
    
    // Pre-compiled unit regex patterns
    private val unitPatterns = mapOf(
        "bpm" to Regex("\\b(?:bpm|beats?\\s+per\\s+minute|heart\\s+rate|pulse\\s+rate|hr|heartbeat|heart\\s+beat|pulse|cardiac\\s+rate|beat\\s+rate|rhythm|heart\\s+rhythm|cardiac\\s+rhythm)\\b", RegexOption.IGNORE_CASE),
        "kg" to Regex("\\b(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|k\\.?g\\.?)\\b", RegexOption.IGNORE_CASE),
        "pounds" to Regex("\\b(?:pounds?|lbs?|lb|pound\\s+weight|#|lbs\\s+weight|lb\\s+weight)\\b", RegexOption.IGNORE_CASE),
        "km" to Regex("\\b(?:km|kms|kilometer|kilometers|kilometre|kilometres|k\\.?m\\.?|klick|klicks)\\b", RegexOption.IGNORE_CASE),
        "miles" to Regex("\\b(?:miles?|mi|mile\\s+distance|mi\\.?|statute\\s+miles?)\\b", RegexOption.IGNORE_CASE),
        "kcal" to Regex("\\b(?:kcal|calories?|calorie|cal|cals|kilocalories?|kilocalorie|food\\s+calories?|dietary\\s+calories?|energy|k\\.?cal)\\b", RegexOption.IGNORE_CASE),
        "hours" to Regex("\\b(?:hours?|hrs?|hr|h|hour\\s+duration|h\\.?r\\.?s?\\.?)\\b", RegexOption.IGNORE_CASE),
        "minutes" to Regex("\\b(?:min|mins|minute|minutes|minute\\s+duration|min\\.?s?\\.?)\\b", RegexOption.IGNORE_CASE),
        "percent" to Regex("\\b(?:percent|%|percentage|pct|pc|per\\s+cent|percentile|blood oxygen|spo2|sp2|spO2)\\b", RegexOption.IGNORE_CASE),
        "count" to Regex("\\b(?:steps?|step\\s+count|footsteps?|foot\\s+steps?|paces?|strides?|walk\\s+count|walking\\s+count|number\\s+of\\s+steps?|total\\s+steps?|step\\s+total)\\b", RegexOption.IGNORE_CASE),
        "meters" to Regex("\\b(?:meters?|metres?|m|meter\\s+distance|metre\\s+distance|m\\.?)\\b", RegexOption.IGNORE_CASE),
        "feet" to Regex("\\b(?:feet|foot|ft|f\\.?t\\.?|')\\b", RegexOption.IGNORE_CASE),
        "seconds" to Regex("\\b(?:seconds?|secs?|sec|s|second\\s+duration|s\\.?e?c?\\.?s?\\.?)\\b", RegexOption.IGNORE_CASE),
        "grams" to Regex("\\b(?:grams?|grammes?|g|gm|gms|g\\.?m?\\.?s?\\.?)\\b", RegexOption.IGNORE_CASE),
        "liters" to Regex("\\b(?:liters?|litres?|l|ltr|ltrs|l\\.?t?r?\\.?s?\\.?)\\b", RegexOption.IGNORE_CASE),
        "degrees" to Regex("\\b(?:degrees?|deg|°|degree\\s+celsius|degree\\s+fahrenheit|celsius|fahrenheit)\\b", RegexOption.IGNORE_CASE),
        "score" to Regex("\\b(?:score|rating|grade|rank|level|point|points|pts?|value|number)\\b", RegexOption.IGNORE_CASE),
        "distance" to Regex("\\b(?:distance|length|span|range|mileage|how\\s+far|travelled|traveled|covered)\\b", RegexOption.IGNORE_CASE)
    )
    
    fun extractSlots(text: String, intent: String): SlotExtractionResult {
        Log.d(LOG_TAG, "🏷️ Extracting slots for intent: $intent, text: '$text'")
        
        val slots = mutableMapOf<String, Any>()
        val textLower = text.lowercase(Locale.getDefault())
        
        // Get required slots for this intent
        val requiredSlots = intentSlotTemplates[intent] ?: emptyList()
        
        // Pre-process text
        val processedText = preprocessText(textLower)
        
        // Extract each required slot
        for (slotName in requiredSlots) {
            val value = extractSingleSlot(processedText, textLower, slotName, intent)
            if (value != null) {
                slots[slotName] = value
                Log.d(LOG_TAG, "  ✓ Extracted $slotName: $value")
            }
        }
        
        // Special handling for SetThreshold: combine type into metric
        if (intent == "SetThreshold") {
            val type = extractSingleSlot(processedText, textLower, "type", intent) as? String
            val metric = slots["metric"] as? String
            if (type != null && metric != null) {
                // Set unit based on base metric before combining
                if (!slots.containsKey("unit")) {
                    when (metric) {
                        "steps" -> slots["unit"] = "count"
                        "distance" -> slots["unit"] = "km"
                        "calories" -> slots["unit"] = "kcal"
                        "heart rate" -> slots["unit"] = "bpm"
                        "sleep" -> slots["unit"] = "hours"
                        "weight" -> slots["unit"] = "kg"
                        "spo2" -> slots["unit"] = "percent"
                        "stress" -> slots["unit"] = "score"
                    }
                }
                slots["metric"] = "$type $metric"
                Log.d(LOG_TAG, "  ✓ Combined metric: ${slots["metric"]}")
            }
        }
        
        // Add contextual slots
        addContextualSlots(textLower, intent, slots)
        
        // Calculate confidence based on how many slots were extracted
        val confidence = if (requiredSlots.isNotEmpty()) {
            slots.size.toFloat() / requiredSlots.size.toFloat()
        } else {
            1.0f
        }
        
        Log.d(LOG_TAG, "🎯 Final slots: $slots (confidence: ${"%.2f".format(confidence)})")
        
        return SlotExtractionResult(slots.toMap(), confidence)
    }
    
    private fun preprocessText(text: String): String {
        var processed = text
        
        // Normalize common variations (from Python implementation)
        processed = processed.replace(Regex("\\bhow\\s+much\\s+did\\s+i\\s+walk"), "walking distance")
        processed = processed.replace(Regex("\\bhow\\s+many\\s+steps"), "steps")
        processed = processed.replace(Regex("\\bhow\\s+far\\s+did\\s+i\\s+walk"), "walking distance")
        processed = processed.replace(Regex("\\bwhat\\s+is\\s+my"), "my")
        processed = processed.replace(Regex("\\bshow\\s+me\\s+my"), "my")
        
        return processed
    }
    
    private fun extractSingleSlot(processedText: String, originalText: String, slotName: String, intent: String): Any? {
        return when (slotName) {
            "metric" -> extractMetric(processedText, originalText)
            "time_ref" -> extractTimeRef(originalText)
            "unit" -> extractUnit(originalText)
            "qualifier" -> extractQualifier(originalText)
            "identifier" -> extractIdentifier(originalText)
            "threshold" -> extractThreshold(originalText)
            "target" -> extractTarget(originalText)
            "value" -> extractValue(originalText, intent)
            "feature" -> extractFeature(originalText)
            "state" -> extractState(originalText)
            "action" -> extractAction(originalText)
            "tool" -> extractTool(originalText)
            "activity_type" -> extractActivityType(originalText)
            "activity_number" -> extractActivityNumber(originalText)
            "app" -> extractApp(originalText)
            "contact" -> extractContact(originalText)
            "location" -> extractLocation(originalText)
            "attribute" -> extractAttribute(originalText)
            "type" -> extractType(originalText)
            "period" -> extractPeriod(originalText)
            "event_type" -> extractEventType(originalText)
            "message" -> "Sorry, please say again"
            else -> null
        }
    }
    
    private fun extractMetric(processedText: String, originalText: String): String? {
        // Direct synonym matching on processed text first
        for ((metric, pattern) in synonymPatterns) {
            if (pattern.containsMatchIn(processedText)) {
                return metric
            }
        }
        
        // Fallback to original text
        for ((metric, pattern) in synonymPatterns) {
            if (pattern.containsMatchIn(originalText)) {
                return metric
            }
        }
        
        // Context-based inference with expanded patterns
        
        // Walking/Movement context
        if (walkingMovementRegex.containsMatchIn(originalText)) {
            return if (distanceRegex.containsMatchIn(originalText)) {
                "distance"
            } else {
                "steps"
            }
        }
        
        // Sleep context
        if (sleepRegex.containsMatchIn(originalText)) {
            return if (sleepQualityRegex.containsMatchIn(originalText)) {
                "sleep score"
            } else {
                "sleep"
            }
        }
        
        // Heart/Cardio context
        if (heartRegex.containsMatchIn(originalText)) {
            return "heart rate"
        }
        
        // Calories/Energy context
        if (caloriesRegex.containsMatchIn(originalText)) {
            return "calories"
        }
        
        // Oxygen/Breathing context
        if (oxygenRegex.containsMatchIn(originalText)) {
            return "spo2"
        }
        
        // Weight/Body Mass context
        if (weightRegex.containsMatchIn(originalText)) {
            return "weight"
        }
        
        // Stress/Mental Health context
        if (stressRegex.containsMatchIn(originalText)) {
            return "stress"
        }
        
        return null
    }
    
    private fun extractTimeRef(text: String): String? {
        val timePatterns = mapOf(
            "last night" to "\\blast\\s+night\\b|\\bduring\\s+(?:the\\s+)?night\\b|\\bovernight\\b|\\bnight\\s+time\\b|\\bnighttime\\b|\\bat\\s+night\\b|\\bthrough(?:out)?\\s+(?:the\\s+)?night\\b|\\ball\\s+night\\b|\\bpast\\s+night\\b|\\bprevious\\s+night\\b|\\bthe\\s+other\\s+night\\b|\\blast\\s+evening\\b|\\byesterday\\s+night\\b|\\byesterday\\s+evening\\b|\\blate\\s+yesterday\\b|\\bafter\\s+dark\\b|\\bwhile\\s+(?:I\\s+)?sleep(?:ing)?\\b|\\bduring\\s+(?:my\\s+)?sleep\\b|\\bsleep(?:ing)?\\s+(?:hours|time|period)\\b|\\bin\\s+bed\\b|\\bwhilst\\s+asleep\\b|\\bwhen\\s+(?:I\\s+)?(?:was\\s+)?asleep\\b|\\bbedtime\\b",

            "yesterday" to "\\byesterday\\b(?!\\s+(?:night|evening|morning|afternoon))|\\blast\\s+day\\b|\\bprevious\\s+day\\b|\\bday\\s+before\\b|\\b1\\s+day\\s+ago\\b|\\bone\\s+day\\s+ago\\b|\\ba\\s+day\\s+ago\\b|\\bthe\\s+day\\s+before\\b|\\bprior\\s+day\\b|\\bthe\\s+previous\\s+day\\b|\\b24\\s+hours\\s+ago\\b|\\byesterdays?\\b|\\bthe\\s+other\\s+day\\b|\\bday\\s+prior\\b|\\bthe\\s+last\\s+day\\b|\\bpast\\s+day\\b|\\bthe\\s+preceding\\s+day\\b|\\bmost\\s+recent\\s+day\\b|\\bjust\\s+yesterday\\b|\\bonly\\s+yesterday\\b|\\bback\\s+yesterday\\b|\\byesterday\\s+morning\\b|\\byesterday\\s+am\\b|\\bmorning\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+morning\\b|\\byesterday\\s+early\\b|\\byesterday\\s+at\\s+dawn\\b|\\bearly\\s+yesterday\\b|\\byesterday\\s+daybreak\\b|\\byesterday\\s+sunrise\\b|\\byesterday\\s+first\\s+thing\\b|\\byesterday\\s+afternoon\\b|\\byesterday\\s+pm\\b|\\bafternoon\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+afternoon\\b|\\byesterday\\s+midday\\b|\\byesterday\\s+noon\\b|\\byesterday\\s+lunchtime\\b|\\byesterday\\s+mid[\\s-]?day\\b|\\byesterday\\s+evening\\b|\\bevening\\s+yesterday\\b|\\byesterday\\s+in\\s+the\\s+evening\\b|\\byesterday\\s+at\\s+night\\b|\\blate\\s+yesterday\\b|\\byesterday\\s+dusk\\b|\\byesterday\\s+twilight\\b|\\byesterday\\s+sundown\\b|\\byesterday\\s+nightfall\\b",

            "now" to "\\bnow\\b|\\bright\\s+now\\b|\\bat\\s+(?:this\\s+)?moment\\b|\\bthis\\s+instant\\b|\\bimmediately\\b|\\binstantly\\b|\\bright\\s+away\\b|\\bright\\s+at\\s+this\\s+moment\\b|\\bat\\s+the\\s+current\\s+time\\b|\\bthis\\s+very\\s+moment\\b|\\bcurrent\\s+time\\b|\\bthe\\s+present\\s+moment\\b|\\bright\\s+here\\b|\\bright\\s+at\\s+this\\s+instant\\b",

            "today" to "\\btoday\\b(?!\\s+(?:morning|afternoon|evening|night))|\\bcurrently\\b|\\bthis\\s+day\\b|\\bat\\s+present\\b|\\bso\\s+far\\s+today\\b|\\btodays?\\b|\\bcurrent\\s+day\\b|\\bas\\s+of\\s+today\\b|\\btill\\s+now\\b|\\bup\\s+to\\s+now\\b|\\bpresently\\b|\\bat\\s+this\\s+time\\b|\\bthis\\s+very\\s+day\\b|\\bthe\\s+present\\s+day\\b|\\bfor\\s+today\\b|\\bon\\s+this\\s+day\\b|\\bsince\\s+midnight\\b|\\bso\\s+far\\b|\\buntil\\s+now\\b|\\bas\\s+of\\s+now\\b|\\blater\\s+today\\b|\\bend\\s+of\\s+(?:the\\s+)?day\\b",

            "last week" to "\\blast\\s+week\\b|\\bpast\\s+week\\b|\\bprevious\\s+week\\b|\\bthe\\s+week\\s+before\\b|\\bprior\\s+week\\b|\\b1\\s+week\\s+ago\\b|\\bone\\s+week\\s+ago\\b|\\ba\\s+week\\s+ago\\b|\\bweek\\s+prior\\b|\\bthe\\s+last\\s+week\\b|\\bthe\\s+past\\s+week\\b|\\bthe\\s+previous\\s+week\\b|\\b7\\s+days\\s+ago\\b|\\blast\\s+weeks?\\b|\\bthe\\s+preceding\\s+week\\b|\\bmost\\s+recent\\s+week\\b|\\blatest\\s+week\\b|\\bformer\\s+week\\b|\\bearlier\\s+week\\b|\\bthe\\s+other\\s+week\\b|\\bduring\\s+last\\s+week\\b|\\bthroughout\\s+last\\s+week\\b|\\bover\\s+last\\s+week\\b|\\bback\\s+last\\s+week\\b",

            "tomorrow" to "\\btomorrow\\b|\\bnext\\s+day\\b|\\bthe\\s+day\\s+after\\b|\\bday\\s+after\\b|\\btomorrow\\s+morning\\b|\\btomorrow\\s+afternoon\\b|\\btomorrow\\s+evening\\b|\\btomorrow\\s+night\\b|\\bcoming\\s+day\\b|\\bupcoming\\s+day\\b|\\bfuture\\s+day\\b|\\bthe\\s+following\\s+day\\b|\\b24\\s+hours\\s+from\\s+now\\b|\\bin\\s+24\\s+hours\\b|\\bby\\s+tomorrow\\b|\\btill\\s+tomorrow\\b|\\buntil\\s+tomorrow\\b",

            "this morning" to "\\bthis\\s+morning\\b|\\bmorning\\b(?!\\s+(?:yesterday|tomorrow|next|last|this\\s+(?:afternoon|evening|week|month|year)))|\\bearly\\s+today\\b|\\btoday\\s+morning\\b|\\bin\\s+the\\s+morning\\b|\\bthis\\s+am\\b|\\bearly\\s+hours\\b|\\bdawn\\b|\\bsunrise\\b|\\bdaybreak\\b|\\bfirst\\s+thing\\b|\\bright\\s+after\\s+waking\\b|\\bupon\\s+waking\\b|\\bsince\\s+waking\\b",

            "this afternoon" to "\\bthis\\s+afternoon\\b|\\bafternoon\\b(?!\\s+(?:yesterday|tomorrow|next|last|this\\s+(?:morning|evening|week|month|year)))|\\btoday\\s+afternoon\\b|\\bin\\s+the\\s+afternoon\\b|\\bthis\\s+pm\\b|\\bafter\\s+noon\\b|\\bmidday\\b|\\bmid[\\s-]?day\\b|\\bnoon\\b|\\blunchtime\\b|\\blate\\s+morning\\b|\\bearly\\s+afternoon\\b",

            "this evening" to "\\bthis\\s+evening\\b|\\bevening\\b(?!\\s+(?:yesterday|tomorrow|next|last|this\\s+(?:morning|afternoon|week|month|year)))|\\btonight\\b|\\btoday\\s+evening\\b|\\bin\\s+the\\s+evening\\b|\\blater\\s+today\\b|\\bend\\s+of\\s+day\\b|\\bafter\\s+work\\b|\\bdusk\\b|\\btwilight\\b|\\bsundown\\b|\\bsunset\\b|\\bnightfall\\b|\\bafter\\s+dark\\b",

            "this week" to "\\bthis\\s+week\\b|\\bcurrent\\s+week\\b|\\bweek\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+week\\b|\\bup\\s+to\\s+now\\s+this\\s+week\\b|\\bweekly\\s+total\\b|\\bweek\\s+to\\s+date\\b|\\bthis\\s+weeks?\\b|\\bongoing\\s+week\\b|\\bpresent\\s+week\\b|\\bwithin\\s+this\\s+week\\b|\\bduring\\s+this\\s+week\\b|\\bthroughout\\s+this\\s+week\\b|\\bover\\s+this\\s+week\\b",

            "this month" to "\\bthis\\s+month\\b|\\bcurrent\\s+month\\b|\\bmonth\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+month\\b|\\bup\\s+to\\s+now\\s+this\\s+month\\b|\\bmonthly\\s+total\\b|\\bmonth\\s+to\\s+date\\b|\\bthis\\s+months?\\b|\\bongoing\\s+month\\b|\\bpresent\\s+month\\b|\\bwithin\\s+this\\s+month\\b|\\bduring\\s+this\\s+month\\b|\\bthroughout\\s+this\\s+month\\b|\\bover\\s+this\\s+month\\b",

            "last month" to "\\blast\\s+month\\b|\\bpast\\s+month\\b|\\bprevious\\s+month\\b|\\bthe\\s+month\\s+before\\b|\\bprior\\s+month\\b|\\b1\\s+month\\s+ago\\b|\\bone\\s+month\\s+ago\\b|\\ba\\s+month\\s+ago\\b|\\bmonth\\s+prior\\b|\\bthe\\s+last\\s+month\\b|\\bthe\\s+past\\s+month\\b|\\bthe\\s+previous\\s+month\\b|\\blast\\s+months?\\b|\\bthe\\s+preceding\\s+month\\b|\\bmost\\s+recent\\s+month\\b|\\blatest\\s+month\\b|\\bformer\\s+month\\b|\\bearlier\\s+month\\b|\\bthe\\s+other\\s+month\\b|\\bduring\\s+last\\s+month\\b|\\bthroughout\\s+last\\s+month\\b|\\bover\\s+last\\s+month\\b|\\bback\\s+last\\s+month\\b",

            "next week" to "\\bnext\\s+week\\b|\\bcoming\\s+week\\b|\\bupcoming\\s+week\\b|\\bfollowing\\s+week\\b|\\bweek\\s+ahead\\b|\\bthe\\s+next\\s+7\\s+days\\b|\\bin\\s+a\\s+week\\b|\\ba\\s+week\\s+from\\s+now\\b|\\b7\\s+days\\s+from\\s+now\\b|\\bnext\\s+weeks?\\b|\\bfuture\\s+week\\b|\\bforthcoming\\s+week\\b",

            "next month" to "\\bnext\\s+month\\b|\\bcoming\\s+month\\b|\\bupcoming\\s+month\\b|\\bfollowing\\s+month\\b|\\bmonth\\s+ahead\\b|\\bthe\\s+next\\s+30\\s+days\\b|\\bin\\s+a\\s+month\\b|\\ba\\s+month\\s+from\\s+now\\b|\\b30\\s+days\\s+from\\s+now\\b|\\bnext\\s+months?\\b|\\bfuture\\s+month\\b|\\bforthcoming\\s+month\\b",

            "recently" to "\\brecently\\b|\\blately\\b|\\bof\\s+late\\b|\\bin\\s+recent\\s+times\\b|\\bthese\\s+days\\b|\\bthe\\s+past\\s+few\\s+days\\b|\\bthe\\s+last\\s+few\\s+days\\b|\\brecent\\s+days\\b|\\bjust\\s+recently\\b|\\bnot\\s+long\\s+ago\\b|\\ba\\s+short\\s+while\\s+ago\\b|\\bwithin\\s+the\\s+past\\s+few\\s+days\\b|\\bover\\s+the\\s+past\\s+few\\s+days\\b",

            "all time" to "\\ball\\s+time\\b|\\ball-time\\b|\\bever\\b|\\ball\\s+history\\b|\\bsince\\s+(?:the\\s+)?beginning\\b|\\bfrom\\s+(?:the\\s+)?start\\b|\\bthroughout\\s+history\\b|\\blifetime\\b|\\ball\\s+my\\s+life\\b|\\bsince\\s+I\\s+(?:started|begun|began)\\b|\\bfrom\\s+day\\s+one\\b|\\bsince\\s+inception\\b|\\ball\\s+records\\b|\\ball\\s+data\\b|\\bcomplete\\s+history\\b|\\bfull\\s+history\\b",

            "this year" to "\\bthis\\s+year\\b|\\bcurrent\\s+year\\b|\\byear\\s+so\\s+far\\b|\\btill\\s+now\\s+this\\s+year\\b|\\bup\\s+to\\s+now\\s+this\\s+year\\b|\\byearly\\s+total\\b|\\byear\\s+to\\s+date\\b|\\bthis\\s+years?\\b|\\bongoing\\s+year\\b|\\bpresent\\s+year\\b|\\bwithin\\s+this\\s+year\\b|\\bduring\\s+this\\s+year\\b|\\bthroughout\\s+this\\s+year\\b|\\bover\\s+this\\s+year\\b",

            "last year" to "\\blast\\s+year\\b|\\bpast\\s+year\\b|\\bprevious\\s+year\\b|\\bthe\\s+year\\s+before\\b|\\bprior\\s+year\\b|\\b1\\s+year\\s+ago\\b|\\bone\\s+year\\s+ago\\b|\\ba\\s+year\\s+ago\\b|\\byear\\s+prior\\b|\\bthe\\s+last\\s+year\\b|\\bthe\\s+past\\s+year\\b|\\bthe\\s+previous\\s+year\\b|\\blast\\s+years?\\b|\\bthe\\s+preceding\\s+year\\b|\\bmost\\s+recent\\s+year\\b|\\blatest\\s+year\\b|\\bformer\\s+year\\b|\\bearlier\\s+year\\b|\\bthe\\s+other\\s+year\\b|\\bduring\\s+last\\s+year\\b|\\bthroughout\\s+last\\s+year\\b|\\bover\\s+last\\s+year\\b|\\bback\\s+last\\s+year\\b"
        )
        for ((timeRef, pattern) in timePatterns) {
            if (text.contains(pattern.toRegex())) {
                return timeRef
            }
        }
        
        return null
    }
    
    private fun extractQualifier(text: String): String? {
        val qualifierPatterns = mapOf(
            "minimum" to "\\b(?:minimum|min|lowest|least|bottom|smallest|bare minimum|minimal|minimally|rock bottom|floor|base|baseline|low point|lower|tiniest|fewest|less|lesser|reduced|at least|no less than|starting from|beginning at|from|low|lows|worst|slowest|minimum value|min value|floor value|bottom line|rock-bottom|absolute minimum|very least|bare min)\\b",

            "maximum" to "\\b(?:maximum|max|highest|most|peak|top|largest|biggest|maximal|maximally|ceiling|upper limit|high point|higher|greatest|best|record|all time high|at most|no more than|up to|limit|cap|high|highs|fastest|extreme|topmost|ultimate|max value|maximum value|ceiling value|top line|all-time high|absolute maximum|very most|max out)\\b",

            "average" to "\\b(?:average|avg|mean|typical|normal|averaged|averaging|median|mid|middle|midpoint|central|moderate|standard|regular|usual|common|ordinary|per day|daily average|on average|typically|normally|generally|approximately|around|about|roughly|average value|mean value|avg value|in average|on avg|medium|middling|fair)\\b",

            "total" to "\\b(?:total|sum|overall|complete|entire|all|totaled|totaling|all time|full|whole|combined|cumulative|aggregate|collectively|together|in total|altogether|grand total|summation|net|gross|comprehensive|accumulated|compilation|tally|count|running total|total value|sum total|total amount|cumulative total|overall total|combined total|full total|net total)\\b"
        )
        
        for ((qualifier, pattern) in qualifierPatterns) {
            if (text.contains(pattern.toRegex())) {
                return qualifier
            }
        }
        
        return "today"
    }
    
    private fun extractIdentifier(text: String): String? {
        val identifierPatterns = mapOf(
            "minimum" to "\\b(?:minimum|min|lowest|least|bottom|smallest|bare minimum|minimal|minimally|rock bottom|floor|base|baseline|low point|lower|tiniest|fewest|less|lesser|reduced|at least|no less than|starting from|beginning at|from|low|lows|worst|slowest|minimum value|min value|floor value|bottom line|rock-bottom|absolute minimum|very least|bare min)\\b",
            
            "maximum" to "\\b(?:maximum|max|highest|most|peak|top|largest|biggest|maximal|maximally|ceiling|upper limit|high point|higher|greatest|best|record|all time high|at most|no more than|up to|limit|cap|high|highs|fastest|extreme|topmost|ultimate|max value|maximum value|ceiling value|top line|all-time high|absolute maximum|very most|max out)\\b",
            
            "average" to "\\b(?:average|avg|mean|typical|normal|averaged|averaging|median|mid|middle|midpoint|central|moderate|standard|regular|usual|common|ordinary|per day|daily average|on average|typically|normally|generally|approximately|around|about|roughly|average value|mean value|avg value|in average|on avg|medium|middling|fair)\\b"
        )
        
        for ((identifier, pattern) in identifierPatterns) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return identifier
            }
        }
        
        // Default to average if no specific identifier is found
        return "average"
    }
    
    private fun extractUnit(text: String): String? {
        // Try context-based unit inference first
        when {
            heartRateUnitRegex.containsMatchIn(text) -> return "bpm"
            stressRegex.containsMatchIn(text) -> return "score"
            oxygenRegex.containsMatchIn(text) -> return "percent"
            sleepRegex.containsMatchIn(text) -> return "hours"
            sleepQualityRegex.containsMatchIn(text) -> return "score"
            distanceRegex.containsMatchIn(text) -> return "km"
            caloriesRegex.containsMatchIn(text) -> return "kcal"
            walkingMovementRegex.containsMatchIn(text) -> return "distance"
            stepsUnitRegex.containsMatchIn(text) -> return "count"
        }
        
        // Check pre-compiled unit patterns
        for ((unit, pattern) in unitPatterns) {
            if (pattern.containsMatchIn(text)) {
                return unit
            }
        }
        
        return null
    }
    
    private fun extractThreshold(text: String): Int? {
        // Look for numbers in context
        val numberPatterns = listOf(
            // Direct number with unit pattern - expanded units
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:bpm|beats?\\s+per\\s+minute|kg|kgs|kilogram|kilograms|pounds?|lbs?|lb|km|kms|kilometer|kilometers|kilometre|kilometres|miles?|mi|meter|meters|metre|metres|m|feet|foot|ft|percent|%|percentage|hours?|hrs?|hr|h|minutes?|mins?|min|seconds?|secs?|sec|s|kcal|calories?|cal|cals|steps?|grams?|g|liters?|litres?|l|ltr)\\b",
            
            // Above/Over/Exceeds pattern - expanded
            "\\b(?:above|over|exceeds?|exceeded|exceeding|higher\\s+than|more\\s+than|greater\\s+than|beyond|past|upwards?\\s+of|in\\s+excess\\s+of|surpass(?:es|ed|ing)?|top(?:s|ped|ping)?|beat(?:s|ing)?|outperform(?:s|ed|ing)?|at\\s+least|minimum\\s+of|no\\s+less\\s+than|starting\\s+from|from|upward\\s+of)\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Below/Under/Less than pattern - expanded
            "\\b(?:below|under|less\\s+than|lower\\s+than|fewer\\s+than|beneath|short\\s+of|shy\\s+of|down\\s+to|up\\s+to|no\\s+more\\s+than|at\\s+most|maximum\\s+of|capped\\s+at|limited\\s+to|within|not\\s+exceeding|doesn'?t\\s+exceed|under\\s+the|below\\s+the|inferior\\s+to)\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Around/Approximately pattern - NEW
            "\\b(?:around|about|approximately|roughly|nearly|close\\s+to|near|almost|circa|approx\\.?|~|somewhere\\s+around|in\\s+the\\s+region\\s+of|in\\s+the\\s+ballpark\\s+of|give\\s+or\\s+take|or\\s+so|ish)\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Between/Range pattern - NEW
            "\\b(?:between|from)\\s+(\\d+(?:\\.\\d+)?)\\s+(?:to|and|through|-|–)\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Exactly/Precisely pattern - NEW
            "\\b(?:exactly|precisely|just|only|specifically|right\\s+at|dead\\s+on|on\\s+the\\s+dot|bang\\s+on|spot\\s+on)\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Number followed by qualifier - NEW
            "\\b(\\d+(?:\\.\\d+)?)\\s+(?:or\\s+(?:more|less|above|below|over|under|higher|lower|greater|fewer)?)\\b",
            
            // Comparative with number - NEW
            "\\b(?:increased?|decreased?|dropped?|rose|raised?|fell|climbed?|went\\s+up|went\\s+down|gained?|lost)\\s+(?:by|to)?\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Target/Goal pattern - NEW
            "\\b(?:target|goal|aim|objective|hit|reach(?:ed)?|achieve(?:d)?|attain(?:ed)?|get\\s+to|make\\s+it\\s+to)\\s+(\\d+(?:\\.\\d+)?)\\b",
            
            // Standalone number with context - NEW
            "\\b(\\d+(?:\\.\\d+)?)\\s+(?:steps?|calories?|hours?|minutes?|kg|pounds?|km|miles?|bpm|percent)\\b"
        )
        
        for (pattern in numberPatterns) {
            val match = pattern.toRegex().find(text)
            if (match != null) {
                return match.groupValues[1].toDoubleOrNull()?.toInt()
            }
        }
        
        // Fallback to any number
        val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
        return numbers.firstOrNull()?.groupValues?.get(1)?.toDoubleOrNull()?.toInt()
    }
    
    private fun extractTarget(text: String): Int? {
        val goalPatterns = listOf(
            // Goal/Target/Aim patterns - expanded (now supports comma-separated numbers)
            "\\b(?:goal|target|aim|objective|plan|intention|aspiration|ambition|desire|want|wish|hope)\\s*(?:is|of|to|for|at)?\\s*(?:be|reach|hit|achieve|get|make|do)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Set/Change/Update patterns - expanded (now supports comma-separated numbers)
            "\\b(?:set|change|update|modify|adjust|edit|configure|make|establish|create|define|specify)\\s*(?:my|the)?\\s*(?:goal|target|aim|objective)?\\s*(?:to|at|as|for)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Action + number + unit pattern - expanded (now supports comma-separated numbers)
            "\\b(?:reach|hit|achieve|attain|get|get\\s+to|make|do|complete|finish|accomplish|meet)\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|kilogram|kilograms|pounds?|lbs?|km|kms|kilometer|kilometers|miles?|hours?|hrs?|minutes?|mins?|calories?|kcal|bpm)\\b",
            
            // Number + unit pattern - expanded (now supports comma-separated numbers)
            "\\b(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|kilogram|kilograms|pounds?|lbs?|lb|km|kms|kilometer|kilometers|kilometre|kilometres|miles?|mi|hours?|hrs?|hr|h|minutes?|mins?|min|m|calories?|kcal|cal|cals|bpm|beats?|meter|meters|metre|metres|feet|foot|ft)\\b",
            
            // "I want to" patterns - NEW (now supports comma-separated numbers)
            "\\b(?:I|i)\\s+(?:want|wanna|need|must|should|have\\s+to|got\\s+to|gotta)\\s+(?:to\\s+)?(?:reach|hit|get|do|achieve|make|walk|run|burn|lose|gain|sleep)\\s*(?:to|at)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // "Trying to" patterns - NEW (now supports comma-separated numbers)
            "\\b(?:trying|attempting|aiming|working|striving|shooting|going)\\s+(?:to|for)\\s+(?:reach|hit|get|do|achieve|make)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Daily/Weekly goal patterns - NEW (now supports comma-separated numbers)
            "\\b(?:daily|weekly|monthly|per\\s+day|each\\s+day|every\\s+day)\\s+(?:goal|target|aim|objective)?\\s*(?:is|of|to)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Increase/Decrease to patterns - NEW (now supports comma-separated numbers)
            "\\b(?:increase|raise|boost|bump|up|improve|decrease|reduce|lower|drop|cut|bring\\s+down)\\s+(?:to|by)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Minimum/Maximum patterns - NEW (now supports comma-separated numbers)
            "\\b(?:at\\s+least|minimum\\s+of|no\\s+less\\s+than|minimum|min)\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // "Need to be" patterns - NEW (now supports comma-separated numbers)
            "\\b(?:need|needs)\\s+to\\s+(?:be|reach|hit|get)\\s*(?:at|to)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Suggestion patterns - NEW (now supports comma-separated numbers)
            "\\b(?:suggest|recommend|advise|tell\\s+me|remind\\s+me|notify\\s+me).*?(?:when|if|at)\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Limit patterns - NEW (now supports comma-separated numbers)
            "\\b(?:limit|cap|max|maximum|ceiling|upper\\s+limit)\\s*(?:of|to|at|is)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Challenge patterns - NEW (now supports comma-separated numbers)
            "\\b(?:challenge|dare|bet|see\\s+if).*?(?:to\\s+)?(?:reach|hit|do|get|make)\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // "X or more/less" patterns - NEW (now supports comma-separated numbers)
            "\\b(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\s+(?:or\\s+(?:more|above|over|higher|greater|less|below|under|lower|fewer))\\b",
            
            // Alert/Notify patterns - NEW (now supports comma-separated numbers)
            "\\b(?:alert|notify|tell|remind|ping|warn|let\\s+me\\s+know).*?(?:when|if|at|after|once).*?(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Threshold patterns - NEW (now supports comma-separated numbers)
            "\\b(?:threshold|cutoff|mark|milestone|benchmark)\\s*(?:of|at|is)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // "Should be" patterns - NEW (now supports comma-separated numbers)
            "\\b(?:should|must|ought\\s+to|supposed\\s+to)\\s+(?:be|reach|hit|get)\\s*(?:at|to)?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b",
            
            // Track patterns - NEW (now supports comma-separated numbers)
            "\\b(?:track|monitor|watch|follow|check).*?(?:until|till|to|up\\s+to)\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b"
        )
        
        // Try each pattern
        for (pattern in goalPatterns) {
            val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
            if (match != null && match.groupValues.size > 1) {
                // Remove commas before parsing
                val value = match.groupValues[1].replace(",", "").toDoubleOrNull()?.toInt()
                if (value != null && value > 0) {
                    return value
                }
            }
        }
    
        // Fallback: extract any number from the text (including comma-separated)
        val numbers = "\\b(\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?)\\b".toRegex().findAll(text)
        return numbers.firstOrNull()?.groupValues?.get(1)?.replace(",", "")?.toDoubleOrNull()?.toInt()
    }
    
    /**
     * Converts word numbers to digits
     * Examples: "three" -> "3", "fifteen" -> "15", "twenty five" -> "25"
     */
    private fun convertWordToNumber(text: String): String {
        val wordToDigit = mapOf(
            "zero" to "0", "one" to "1", "two" to "2", "three" to "3", "four" to "4",
            "five" to "5", "six" to "6", "seven" to "7", "eight" to "8", "nine" to "9",
            "ten" to "10", "eleven" to "11", "twelve" to "12", "thirteen" to "13",
            "fourteen" to "14", "fifteen" to "15", "sixteen" to "16", "seventeen" to "17",
            "eighteen" to "18", "nineteen" to "19", "twenty" to "20", "thirty" to "30",
            "forty" to "40", "fifty" to "50", "sixty" to "60", "seventy" to "70",
            "eighty" to "80", "ninety" to "90",
            "a" to "1", "an" to "1", "half" to "0.5", "quarter" to "0.25"
        )
        
        var result = text.lowercase()
        
        // Handle compound numbers like "twenty five" -> "25"
        val compoundPattern = "\\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[\\s-]+(one|two|three|four|five|six|seven|eight|nine)\\b".toRegex()
        compoundPattern.findAll(result).forEach { match ->
            val tens = wordToDigit[match.groupValues[1]] ?: "0"
            val ones = wordToDigit[match.groupValues[2]] ?: "0"
            val sum = tens.toInt() + ones.toInt()
            result = result.replace(match.value, sum.toString())
        }
        
        // Replace simple word numbers
        wordToDigit.forEach { (word, digit) ->
            result = result.replace("\\b$word\\b".toRegex(), digit)
        }
        
        return result
    }
    
    /**
     * Normalizes various time formats to 24-hour HH:MM format
     * Examples:
     * - "730" -> "07:30"
     * - "730 pm" -> "19:30"
     * - "730 p.m." -> "19:30"
     * - "730 p m" -> "19:30"
     * - "1030 pm" -> "22:30"
     * - "2230" -> "22:30"
     * - "7:30 am" -> "07:30"
     * - "7:30 a.m." -> "07:30"
     * - "5 30 pm" -> "17:30"
     * - "5.30 a.m" -> "05:30"
     * - "12:00 pm" -> "12:00"
     * - "12:00 am" -> "00:00"
     * Supports AM/PM formats: am, pm, a.m., p.m., a.m, p.m, a m, p m
     */
    private fun normalizeTimeFormat(timeString: String): String {
        val cleanTime = timeString.trim().lowercase()
        
        // Pattern for times like "730", "1030", "2230" (3-4 digits) without AM/PM
        val fourDigitPattern = "^(\\d{3,4})$".toRegex()
        fourDigitPattern.find(cleanTime)?.let { match ->
            val digits = match.groupValues[1]
            return when (digits.length) {
                3 -> {
                    // e.g., "730" -> "07:30"
                    val hour = digits.substring(0, 1)
                    val minute = digits.substring(1, 3)
                    String.format("%02d:%s", hour.toInt(), minute)
                }
                4 -> {
                    // e.g., "2230" -> "22:30"
                    val hour = digits.substring(0, 2)
                    val minute = digits.substring(2, 4)
                    "$hour:$minute"
                }
                else -> timeString
            }
        }
        
        // Pattern for times like "730 pm", "5pm", "6 p.m.", "6.pm" (digits + optional space/dot + am/pm)
        // Supports: am, pm, a.m., p.m., a.m, p.m, a m, p m, 6.pm, 5.am
        val amPmPattern = "^(\\d{1,4})[\\.\\s]*([ap])(?:\\.?\\s*m(?:\\.?)?)?$".toRegex()
        amPmPattern.find(cleanTime)?.let { match ->
            val timeDigits = match.groupValues[1]
            val amPm = if (match.groupValues[2] == "a") "am" else "pm"
            
            var hour = 0
            var minute = 0
            
            when (timeDigits.length) {
                1, 2 -> {
                    // e.g., "7 pm" -> hour = 7, minute = 0
                    hour = timeDigits.toInt()
                    minute = 0
                }
                3 -> {
                    // e.g., "730 pm" -> hour = 7, minute = 30
                    hour = timeDigits.substring(0, 1).toInt()
                    minute = timeDigits.substring(1, 3).toInt()
                }
                4 -> {
                    // e.g., "1030 pm" -> hour = 10, minute = 30
                    hour = timeDigits.substring(0, 2).toInt()
                    minute = timeDigits.substring(2, 4).toInt()
                }
            }
            
            // Convert to 24-hour format
            when {
                amPm == "pm" && hour != 12 -> hour += 12
                amPm == "am" && hour == 12 -> hour = 0
            }
            
            return String.format("%02d:%02d", hour, minute)
        }
        
        // Pattern for "5 30 pm" or "5.30 am" format (hour [space|dot] minute am/pm)
        // Supports: am, pm, a.m., p.m., a.m, p.m, a m, p m
        val hourMinuteAmPmPattern = "^(\\d{1,2})[\\s.]+?(\\d{1,2})[\\.\\s]*([ap])(?:\\.?\\s*m(?:\\.?)?)?$".toRegex()
        hourMinuteAmPmPattern.find(cleanTime)?.let { match ->
            var hour = match.groupValues[1].toInt()
            val minute = match.groupValues[2].toInt()
            val amPm = if (match.groupValues[3] == "a") "am" else "pm"
            
            // Validate hour and minute ranges
            if (hour > 12 || minute >= 60) {
                return timeString // Return original if invalid
            }
            
            // Convert to 24-hour format
            when {
                amPm == "pm" && hour != 12 -> hour += 12
                amPm == "am" && hour == 12 -> hour = 0
            }
            
            return String.format("%02d:%02d", hour, minute)
        }
        
        // Pattern for times with colon like "7:30 pm", "10:30 am"
        // Supports: am, pm, a.m., p.m., a.m, p.m, a m, p m
        val colonAmPmPattern = "^(\\d{1,2}):(\\d{2})[\\.\\s]*([ap])(?:\\.?\\s*m(?:\\.?)?)?$".toRegex()
        colonAmPmPattern.find(cleanTime)?.let { match ->
            var hour = match.groupValues[1].toInt()
            val minute = match.groupValues[2].toInt()
            val amPm = if (match.groupValues[3] == "a") "am" else "pm"
            
            // Convert to 24-hour format
            when {
                amPm == "pm" && hour != 12 -> hour += 12
                amPm == "am" && hour == 12 -> hour = 0
            }
            
            return String.format("%02d:%02d", hour, minute)
        }
        
        // Pattern for 24-hour format like "22:30", "07:30"
        val twentyFourHourPattern = "^(\\d{1,2}):(\\d{2})$".toRegex()
        twentyFourHourPattern.find(cleanTime)?.let { match ->
            val hour = match.groupValues[1].toInt()
            val minute = match.groupValues[2].toInt()
            return String.format("%02d:%02d", hour, minute)
        }
        
        // Return original string if no pattern matches
        return timeString
    }
    
    /**
     * Normalizes timer duration to seconds
     * Examples:
     * - "5 min" -> 300 (seconds)
     * - "2 hr" -> 7200 (seconds)
     * - "1 hour 30 minutes" -> 5400 (seconds)
     * - "45 sec" -> 45 (seconds)
     * - "1:30" -> 90 (seconds, interpreted as MM:SS)
     * - "2:30:45" -> 9045 (seconds, HH:MM:SS)
     */
    private fun normalizeTimerDuration(text: String, matchedValue: String): Int {
        // Convert word numbers to digits first
        val cleanText = convertWordToNumber(text.lowercase())
        val cleanMatchedValue = convertWordToNumber(matchedValue.lowercase())
        
        // Check for combined duration patterns first
        
        // "1 hour 30 minutes 45 seconds" pattern
        val fullPattern = "(\\d+)\\s*(?:hours?|hrs?|hr|h)\\s*(?:and\\s+)?(\\d+)\\s*(?:minutes?|mins?|min|m)\\s*(?:and\\s+)?(\\d+)\\s*(?:seconds?|secs?|sec|s)".toRegex(RegexOption.IGNORE_CASE)
        fullPattern.find(cleanText)?.let { match ->
            val hours = match.groupValues[1].toIntOrNull() ?: 0
            val minutes = match.groupValues[2].toIntOrNull() ?: 0
            val seconds = match.groupValues[3].toIntOrNull() ?: 0
            return hours * 3600 + minutes * 60 + seconds
        }
        
        // "1 hour 30 minutes" or "2h 45m" pattern
        val hourMinutePattern = "(\\d+)\\s*(?:hours?|hrs?|hr|h)\\s*(?:and\\s+)?(\\d+)\\s*(?:minutes?|mins?|min|m)".toRegex(RegexOption.IGNORE_CASE)
        hourMinutePattern.find(cleanText)?.let { match ->
            val hours = match.groupValues[1].toIntOrNull() ?: 0
            val minutes = match.groupValues[2].toIntOrNull() ?: 0
            return hours * 3600 + minutes * 60
        }
        
        // "30 minutes 45 seconds" or "30m 45s" pattern
        val minuteSecondPattern = "(\\d+)\\s*(?:minutes?|mins?|min|m)\\s*(?:and\\s+)?(\\d+)\\s*(?:seconds?|secs?|sec|s)".toRegex(RegexOption.IGNORE_CASE)
        minuteSecondPattern.find(cleanText)?.let { match ->
            val minutes = match.groupValues[1].toIntOrNull() ?: 0
            val seconds = match.groupValues[2].toIntOrNull() ?: 0
            return minutes * 60 + seconds
        }
        
        // Check for time format patterns (MM:SS or HH:MM:SS)
        
        // HH:MM:SS format
        val hhMmSsPattern = "^(\\d+):(\\d+):(\\d+)$".toRegex()
        hhMmSsPattern.find(cleanMatchedValue.trim())?.let { match ->
            val hours = match.groupValues[1].toIntOrNull() ?: 0
            val minutes = match.groupValues[2].toIntOrNull() ?: 0
            val seconds = match.groupValues[3].toIntOrNull() ?: 0
            return hours * 3600 + minutes * 60 + seconds
        }
        
        // MM:SS format (assuming minutes:seconds for timer)
        val mmSsPattern = "^(\\d+):(\\d+)$".toRegex()
        mmSsPattern.find(cleanMatchedValue.trim())?.let { match ->
            val minutes = match.groupValues[1].toIntOrNull() ?: 0
            val seconds = match.groupValues[2].toIntOrNull() ?: 0
            return minutes * 60 + seconds
        }
        
        // Check for single unit patterns
        val value = cleanMatchedValue.replace(Regex("[^\\d.]"), "").toDoubleOrNull() ?: return 0
        
        return when {
            cleanText.contains(Regex("\\b(?:hours?|hrs?|hr|h)\\b")) -> (value * 3600).toInt()
            cleanText.contains(Regex("\\b(?:minutes?|mins?|min|m)\\b")) -> (value * 60).toInt()
            cleanText.contains(Regex("\\b(?:seconds?|secs?|sec|s)\\b")) -> value.toInt()
            else -> value.toInt() // Default to treating as seconds if no unit specified
        }
    }
    
    private fun extractValue(text: String, intent: String): Any? {
        return when (intent) {
            "LogEvent" -> {
                // Expanded weight pattern with 20+ variations
                val weightPattern = "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|pounds?|lbs?|lb|#|stone|st|grams?|grammes?|g|gm|ounces?|oz)\\b".toRegex(RegexOption.IGNORE_CASE)
                val match = weightPattern.find(text)
                match?.groupValues?.get(1)?.toDoubleOrNull()
            }
            
            "TimerStopwatch" -> {
                // Determine if this is a timer (duration) or alarm (specific time) context
                val isTimerContext = text.contains(Regex("\\b(?:timer|stopwatch|countdown|count\\s+down)\\b", RegexOption.IGNORE_CASE))
                val isAlarmContext = text.contains(Regex("\\b(?:alarm|wake|remind|alert)\\b", RegexOption.IGNORE_CASE))
                
                val timePatterns = listOf(
                    // Space/dot separated time with AM/PM - highest priority for "5 30 pm", "10.30 am" format
                    "\\b(\\d{1,2}[\\s.]+?\\d{1,2}\\s*[ap](?:\\.?\\s*m(?:\\.?)?)?(?!\\w))\\b",
                    
                    // Time with AM/PM - capture full time including AM/PM for "5pm", "6 p.m." format
                    "\\b(\\d{1,2}(?::\\d{2})?\\s*[ap](?:\\.?\\s*m(?:\\.?)?)?(?!\\w))\\b",
                    "\\b(\\d{3,4}\\s*[ap](?:\\.?\\s*m(?:\\.?)?)?(?!\\w))\\b", // For "1030 pm" format  
                    
                    // Duration patterns for timers (with word number support)
                    "\\b(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:hours?|hrs?|hr|h)\\b",
                    "\\b(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:minutes?|mins?|min|m)\\b",
                    "\\b(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:seconds?|secs?|sec|s)\\b",
                    
                    // Combined time patterns
                    "\\b(\\d+)\\s*(?:h|hr|hours?)\\s*(?:and\\s+)?(\\d+)\\s*(?:m|min|minutes?)\\b",
                    "\\b(\\d+)\\s*(?:m|min|minutes?)\\s*(?:and\\s+)?(\\d+)\\s*(?:s|sec|seconds?)\\b",
                    "\\b(\\d+):(\\d+):(\\d+)\\b", // HH:MM:SS format
                    "\\b(\\d+):(\\d+)\\b", // MM:SS or HH:MM format
                    
                    // Duration keywords (with word number support)
                    "\\b(?:for|during|lasting|takes?)\\s+(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Timer-specific patterns (with word number support)
                    "\\b(?:set|start|begin|run|timer|stopwatch)\\s*(?:for|to|at)?\\s*(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Alarm-specific patterns - capture full time with AM/PM (prioritize space-separated)
                    "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{1,2}[\\s.]+?\\d{1,2}[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{1,2}(?::\\d{2})?[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{3,4}[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    
                    // "In X time" patterns (with word number support)
                    "\\b(?:in|after|within)\\s+(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Countdown patterns (with word number support)
                    "\\b(?:countdown|count\\s+down)\\s*(?:from|for)?\\s*(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|a|an|half|quarter)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Direct time patterns for alarm setting - capture full time with AM/PM (prioritize space-separated)
                    "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}[\\s.]+?\\d{1,2}[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4}[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:set|create|make).*?(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}[\\s.]+?\\d{1,2}[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4}[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)\\b",
                    
                    // Standalone numeric patterns for alarm context (no AM/PM)
                    "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{3,4})\\b",
                    "\\b(?:alarm|wake).*?(?:for|at)\\s*(\\d{1,2}(?::\\d{2})?)\\b"
                )
                
                for (pattern in timePatterns) {
                    val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
                    if (match != null && match.groupValues.size > 1) {
                        
                        // Handle combined time formats for duration (e.g., "1h 30m" or "1:30")
                        if (match.groupValues.size > 2 && match.groupValues[2].isNotEmpty() && 
                            !text.contains(Regex("\\b(?:am|pm)\\b", RegexOption.IGNORE_CASE))) {
                            
                            if (isTimerContext) {
                                // For timers, convert combined duration to seconds
                                return normalizeTimerDuration(text, match.value)
                            } else {
                                // For other contexts, return total minutes as before
                                val hours = match.groupValues[1].toIntOrNull() ?: 0
                                val minutes = match.groupValues[2].toIntOrNull() ?: 0
                                return (hours * 60 + minutes).toString()
                            }
                        }
                        
                        val timeValue = match.groupValues[1]
                        
                        // Check if this is a time format (contains AM/PM or looks like time in alarm context)
                        val containsAmPm = timeValue.contains(Regex("(?:am|pm|a\\.?\\s*m\\.?|p\\.?\\s*m\\.?|\\.am|\\.pm)", RegexOption.IGNORE_CASE))
                        val containsSpaceOrDot = timeValue.contains(Regex("[\\s.]\\d"))
                        val looksLikeTime = timeValue.matches(Regex("\\d{3,4}")) || timeValue.contains(":") || containsSpaceOrDot
                        
                        if (containsAmPm || (isAlarmContext && looksLikeTime)) {
                            // This is an alarm time, normalize to HH:MM format
                            return normalizeTimeFormat(timeValue)
                        } else if (isTimerContext) {
                            // This is a timer duration, convert to seconds
                            return normalizeTimerDuration(text, match.value)
                        }
                        
                        // Return the time value for other cases
                        return timeValue
                    }
                }
                
                // Fallback to any number
                val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
                numbers.firstOrNull()?.groupValues?.get(1)?.toIntOrNull()
            }
            
            "SetGoal" -> {
                // Goal value patterns - NEW intent support
                val goalPatterns = listOf(
                    // Number with unit
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|pounds?|lbs?|km|kms|miles?|calories?|kcal|hours?|hrs?|minutes?|mins?|bpm)\\b",
                    
                    // Goal keywords with number
                    "\\b(?:goal|target|aim|objective)\\s*(?:of|to|is|at)?\\s*(\\d+(?:\\.\\d+)?)\\b",
                    
                    // Action + number
                    "\\b(?:reach|hit|achieve|get|make|do)\\s*(\\d+(?:\\.\\d+)?)\\b",
                    
                    // "X per day" patterns
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:per|each|every|a)\\s+day\\b"
                )
                
                for (pattern in goalPatterns) {
                    val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
                    if (match != null && match.groupValues.size > 1) {
                        return match.groupValues[1].toDoubleOrNull()?.toInt()
                    }
                }
                
                // Fallback
                val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
                numbers.firstOrNull()?.groupValues?.get(1)?.toIntOrNull()
            }
            
            "QueryMetrics" -> {
                // Metric query patterns
                val metricPatterns = listOf(
                    // Comparative values
                    "\\b(?:above|over|more\\s+than|greater\\s+than|exceeds?)\\s+(\\d+(?:\\.\\d+)?)\\b",
                    "\\b(?:below|under|less\\s+than|fewer\\s+than|lower\\s+than)\\s+(\\d+(?:\\.\\d+)?)\\b",
                    "\\b(?:between|from)\\s+(\\d+(?:\\.\\d+)?)\\s+(?:to|and|through)\\s+(\\d+(?:\\.\\d+)?)\\b",
                    
                    // Approximate values
                    "\\b(?:around|about|approximately|roughly|~)\\s+(\\d+(?:\\.\\d+)?)\\b",
                    
                    // Number with unit
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|pounds?|lbs?|km|miles?|calories?|kcal|bpm|hours?|minutes?|percent|%)\\b"
                )
                
                for (pattern in metricPatterns) {
                    val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
                    if (match != null && match.groupValues.size > 1) {
                        return match.groupValues[1].toDoubleOrNull()
                    }
                }
                
                // Fallback
                val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
                numbers.firstOrNull()?.groupValues?.get(1)?.toDoubleOrNull()
            }
            
            "Reminder" -> {
                // Reminder time patterns
                val reminderPatterns = listOf(
                    // Time formats
                    "\\b(\\d{1,2}(?::\\d{2})?(?:[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)?)\\b",
                    
                    // "In X time" for reminders
                    "\\b(?:in|after)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|days?)\\b",
                    
                    // "At X" patterns
                    "\\b(?:at|by)\\s+(\\d{1,2}(?::\\d{2})?(?:[\\.\\s]*[ap](?:\\.?\\s*m(?:\\.?)?)?)?)\\b",
                    
                    // "Every X" patterns for recurring
                    "\\b(?:every|each)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|days?)\\b"
                )
                
                for (pattern in reminderPatterns) {
                    val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
                    if (match != null && match.groupValues.size > 1) {
                        return match.groupValues[1]
                    }
                }
                
                // Fallback
                val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
                numbers.firstOrNull()?.groupValues?.get(1)
            }
            
            else -> {
                // Generic number extraction with enhanced patterns
                val genericPatterns = listOf(
                    // Number with any common unit
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|pounds?|lbs?|km|miles?|calories?|kcal|bpm|hours?|minutes?|seconds?|percent|%|grams?|liters?|meters?|feet)\\b",
                    
                    // Standalone numbers
                    "\\b(\\d+(?:\\.\\d+)?)\\b"
                )
                
                for (pattern in genericPatterns) {
                    val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
                    if (match != null && match.groupValues.size > 1) {
                        return match.groupValues[1].toDoubleOrNull()?.toInt()
                    }
                }
                
                null
            }
        }
    }
    
    private fun extractFeature(text: String): String? {
        val features = mapOf(
            "do not disturb" to "\\b(?:do\\s+not\\s+disturb|dnd|d\\.?n\\.?d\\.?|silent\\s+mode|silence|quiet\\s+mode|mute|muted|no\\s+disturb|don'?t\\s+disturb|silence\\s+notifications?|quiet\\s+hours?|sleep\\s+mode|bedtime\\s+mode|focus\\s+mode|zen\\s+mode|peaceful\\s+mode|undisturbed|interruption\\s+free|notification\\s+silence)\\b",
            
            "AOD" to "\\b(?:AOD|aod|a\\.?o\\.?d\\.?|always\\s+on\\s+display|always-on\\s+display|always\\s+on|screen\\s+always\\s+on|display\\s+always\\s+on|persistent\\s+display|constant\\s+display|continuous\\s+display|keep\\s+screen\\s+on|screen\\s+stays\\s+on|display\\s+on|ambient\\s+display|glance\\s+screen|standby\\s+screen)\\b",
            
            "raise to wake" to "\\b(?:raise\\s+to\\s+wake|lift\\s+to\\s+wake|tap\\s+to\\s+wake|double\\s+tap\\s+to\\s+wake|touch\\s+to\\s+wake|wrist\\s+raise|raise\\s+wrist|lift\\s+wrist|wake\\s+on\\s+raise|wake\\s+on\\s+lift|wake\\s+on\\s+tap|wake\\s+on\\s+touch|pick\\s+up\\s+to\\s+wake|gesture\\s+wake|motion\\s+wake|tilt\\s+to\\s+wake|wake\\s+gesture|screen\\s+wake|auto\\s+wake|smart\\s+wake)\\b",
            
            "vibration" to "\\b(?:vibration|vibrate|vibrating|haptic|haptics|buzz|buzzing|rumble|rumbling|tactile|tactile\\s+feedback|vibration\\s+feedback|motor|vibration\\s+motor|shake|shaking|pulse|pulsing|vibe|vibes|vibrate\\s+mode|silent\\s+vibrate|ring\\s+vibrate)\\b",
            
            "volume" to "\\b(?:volume|sound\\s+level|sound\\s+volume|audio\\s+level|audio\\s+volume|loudness|loud|quiet|soft|sound|audio|speaker\\s+volume|media\\s+volume|ringtone\\s+volume|notification\\s+volume|alarm\\s+volume|call\\s+volume|ringer|sound\\s+output|audio\\s+output|volume\\s+level)\\b",

            "torch" to "\\b(?:torch|flashlight|flash\\s+light|led\\s+light|led\\s+torch|camera\\s+flash|light|lamp|lantern|beam|illumination|bright\\s+light|phone\\s+light|mobile\\s+light|emergency\\s+light|torch\\s+light|strobe|strobe\\s+light|spotlight|searchlight|headlight|flash\\s+lamp|portable\\s+light|hand\\s+light|led\\s+flash|camera\\s+light|phone\\s+torch|device\\s+light|built-in\\s+light|integrated\\s+light)\\b"
        )
        
        // Try to match features in order of specificity (more specific patterns first)
        val sortedFeatures = features.entries.sortedByDescending { it.value.length }
        
        for ((feature, pattern) in sortedFeatures) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return feature
            }
        }
        
        return null
    }
    
    private fun extractState(text: String): String? {
        return when {
            // ON state - expanded with 20+ variations
            text.contains("\\b(?:turn\\s+on|enable|enabled|enabling|activate|activated|activating|switch\\s+on|start|started|starting|power\\s+on|boot|boot\\s+up|fire\\s+up|launch|open|unmute|unmuted|resume|allow|permit|engage|engaged|engaging|set\\s+on|put\\s+on|make\\s+it\\s+on|get\\s+it\\s+on|bring\\s+up|wake\\s+up|light\\s+up|flip\\s+on|on)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "on"
            
            // OFF state - expanded with 20+ variations
            text.contains("\\b(?:turn\\s+off|disable|disabled|disabling|deactivate|deactivated|deactivating|switch\\s+off|stop|stopped|stopping|shut\\s+off|shut\\s+down|power\\s+off|kill|close|mute|muted|pause|paused|block|deny|disengage|disengaged|disengaging|set\\s+off|put\\s+off|make\\s+it\\s+off|get\\s+it\\s+off|bring\\s+down|sleep|suspend|flip\\s+off|cut\\s+off|off)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "off"
            
            // INCREASE state - expanded with 20+ variations
            text.contains("\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "increase"
            
            // DECREASE state - expanded with 20+ variations
            text.contains("\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "decrease"
            
            else -> null
        }
    }
    
    private fun extractAction(text: String): String? {
        val actions = mapOf(
            "set" to "\\b(?:set|setup|set\\s+up|configure|configuration|adjust|adjustment|change|modify|edit|customize|establish|define|specify|determine|fix|assign|allocate|program|preset|input|enter|put\\s+in|make\\s+it|arrange|organize|prepare)\\b",
            
            "start" to "\\b(?:start|started|starting|begin|began|beginning|initiate|initiated|initiating|launch|launched|launching|commence|commencing|kick\\s+off|fire\\s+up|boot\\s+up|power\\s+on|turn\\s+on|switch\\s+on|activate|enable|engage|trigger|run|execute|go|let'?s\\s+go|get\\s+going|get\\s+started)\\b",
            
            "stop" to "\\b(?:stop|stopped|stopping|end|ended|ending|finish|finished|finishing|terminate|terminated|terminating|cease|halt|pause|paused|pausing|kill|abort|cancel|cancelled|canceling|quit|exit|close|shut\\s+down|power\\s+off|turn\\s+off|switch\\s+off|deactivate|disable|disengage|cut\\s+off)\\b",
            
            "call" to "\\b(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|reach\\s+out|get\\s+in\\s+touch|give\\s+a\\s+call|make\\s+a\\s+call|place\\s+a\\s+call|telephone|buzz|video\\s+call|voice\\s+call|facetime)\\b",
            
            "message" to "\\b(?:message|messaging|text|texting|sms|send|sending|sent|write|compose|type|drop\\s+a\\s+message|send\\s+a\\s+text|shoot\\s+a\\s+message|ping|dm|direct\\s+message|whatsapp|imessage|chat|msg)\\b",
            
            "open" to "\\b(?:open|opened|opening|launch|launched|launching|start|show|display|view|access|load|bring\\s+up|pull\\s+up|fire\\s+up|boot|go\\s+to|navigate\\s+to|switch\\s+to|take\\s+me\\s+to)\\b",
            
            "check" to "\\b(?:check|checking|verify|verifying|examine|look|looking|see|review|inspect|assess|evaluate|monitor|watch|observe|scan|browse|view|find\\s+out|tell\\s+me|show\\s+me|let\\s+me\\s+see|give\\s+me|what'?s|how'?s|any)\\b",
            
            "measure" to "\\b(?:measure|measuring|measured|test|testing|tested|record|recording|recorded|track|tracking|log|logging|logged|take|capture|monitor|scan|read|reading|sample|collect|gauge|assess|evaluate)\\b",

            "play" to "\\b(?:play|playing|played|resume|resuming|resumed|continue|continuing|continued|unpause|unpausing|unpaused|start\\s+playing|begin\\s+playing|kick\\s+off|fire\\s+up|roll|rolling|spun|spin|spinning)\\b",

            "pause" to "\\b(?:pause|pausing|paused|hold|holding|held|freeze|freezing|frozen|stop\\s+temporarily|suspend|suspended|suspending|halt\\s+temporarily|break|breaking|broke|interrupt|interrupting|interrupted)\\b",

            "increase" to "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b",

            "decrease" to "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b",

            "skip_next" to "\\b(?:skip\\s+(?:forward|next|ahead)|next\\s+(?:track|song|chapter|episode|video|clip)|forward\\s+(?:to\\s+next|one)|advance\\s+(?:to\\s+next|one)|go\\s+(?:to\\s+next|forward\\s+one)|jump\\s+(?:to\\s+next|forward)|fast\\s+forward\\s+(?:to\\s+next|one)|move\\s+(?:to\\s+next|forward)|switch\\s+(?:to\\s+next|forward))\\b",

            "skip_previous" to "\\b(?:skip\\s+(?:back|previous|backward)|previous\\s+(?:track|song|chapter|episode|video|clip)|back\\s+(?:to\\s+previous|one)|go\\s+(?:to\\s+previous|back\\s+one)|jump\\s+(?:to\\s+previous|back)|rewind\\s+(?:to\\s+previous|one)|move\\s+(?:to\\s+previous|back)|switch\\s+(?:to\\s+previous|back)|restart\\s+(?:track|song|current))\\b",

            "fast_forward" to "\\b(?:fast\\s+forward|speed\\s+up|forward\\s+(?:quickly|fast)|accelerate\\s+(?:playback|forward)|rush\\s+forward|zoom\\s+forward|hurry\\s+forward|quick\\s+forward|rapid\\s+forward|expedite\\s+forward|double\\s+speed|triple\\s+speed|increase\\s+speed)\\b",

            "rewind" to "\\b(?:rewind|rewinding|rewound|fast\\s+backward|reverse\\s+(?:quickly|fast)|back\\s+up|go\\s+back|reverse\\s+playback|backward\\s+(?:quickly|fast)|retreat|retreating|retreated|regress|regressing|regressed|slow\\s+reverse|reverse\\s+slowly)\\b",

            "seek" to "\\b(?:seek|seeking|sought|jump\\s+to|go\\s+to|move\\s+to|navigate\\s+to|position\\s+to|scrub\\s+to|advance\\s+to|retreat\\s+to|set\\s+position|change\\s+position|adjust\\s+position|locate\\s+to|find\\s+position|progress\\s+to)\\b",

            "mute" to "\\b(?:mute|muting|muted|silence|silencing|silenced|quiet|quieting|quieted|turn\\s+off\\s+sound|disable\\s+sound|kill\\s+sound|cut\\s+sound|no\\s+sound|sound\\s+off|audio\\s+off|volume\\s+off|silent\\s+mode)\\b",

            "unmute" to "\\b(?:unmute|unmuting|unmuted|unsilence|unsilencing|unsilenced|unquiet|unquieting|unquieted|turn\\s+on\\s+sound|enable\\s+sound|restore\\s+sound|bring\\s+back\\s+sound|sound\\s+on|audio\\s+on|volume\\s+on|exit\\s+silent\\s+mode)\\b",

            "fullscreen" to "\\b(?:full\\s+screen|fullscreen|full-screen|maximize|maximizing|maximized|expand|expanding|expanded|enlarge|enlarging|enlarged|stretch|stretching|stretched|fill\\s+screen|screen\\s+fill|wide\\s+screen|cinema\\s+mode|theater\\s+mode|immersive\\s+mode)\\b",

            "captions" to "\\b(?:caption|captions|subtitle|subtitles|closed\\s+caption|closed\\s+captions|cc|cc'?s|text|texts|transcript|transcripts|sub|subs|overlay|overlays|dialogue|dialogues|speech\\s+text|spoken\\s+text|audio\\s+description)\\b",

            "speed" to "\\b(?:speed|speeding|sped|playback\\s+speed|play\\s+speed|rate|rates|pace|pacing|paced|tempo|tempos|tempoing|tempoed|rhythm|rhythms|rhythmical|velocity|velocities|quickness|quicknesses|rapidity|rapidities)\\b",

            "shuffle" to "\\b(?:shuffle|shuffling|shuffled|random|randomize|randomizing|randomized|mix|mixing|mixed|scramble|scrambling|scrambled|jumble|jumbling|jumbled|disorder|disordering|disordered|rearrange|rearranging|rearranged)\\b",

            "repeat" to "\\b(?:repeat|repeating|repeated|loop|looping|looped|cycle|cycling|cycled|replay|replaying|replayed|encore|encoring|encored|again|repeat\\s+mode|loop\\s+mode|continuous\\s+play|infinite\\s+play)\\b"
        )
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        val sortedActions = actions.entries.sortedByDescending { it.value.length }
        
        for ((action, pattern) in sortedActions) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return action
            }
        }
        
        return null
    }

    private fun extractTimerAction(text: String): String? {
        val timerActions = mapOf(
            "set" to "\\b(?:set|setup|set\\s+up|configure|configuration|adjust|adjustment|change|modify|edit|customize|establish|define|specify|determine|fix|assign|allocate|program|preset|input|enter|put\\s+in|make\\s+it|arrange|organize|prepare|setting)\\b",
            
            "start" to "\\b(?:start|started|starting|begin|began|beginning|initiate|initiated|initiating|launch|launched|launching|commence|commencing|kick\\s+off|fire\\s+up|boot\\s+up|power\\s+on|turn\\s+on|switch\\s+on|activate|enable|engage|trigger|run|execute|go|let'?s\\s+go|get\\s+going|get\\s+started)\\b",
            
            "stop" to "\\b(?:stop|stopped|stopping|end|ended|ending|finish|finished|finishing|terminate|terminated|terminating|cease|halt|pause|paused|pausing|kill|abort|cancel|cancelled|canceling|quit|exit|close|shut\\s+down|power\\s+off|turn\\s+off|switch\\s+off|deactivate|disable|disengage|cut\\s+off)\\b"
        )
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        val sortedActions = timerActions.entries.sortedByDescending { it.value.length }
        
        for ((action, pattern) in sortedActions) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return action
            }
        }
        
        return null
    }

    private fun extractMediaAction(text: String): String? {
        val mediaActions = mapOf(
            "play" to "\\b(?:play|playing|played|resume|resuming|resumed|continue|continuing|continued|unpause|unpausing|unpaused|start\\s+playing|begin\\s+playing|kick\\s+off|fire\\s+up|roll|rolling|spun|spin|spinning)\\b",

            "pause" to "\\b(?:pause|pausing|paused|hold|holding|held|freeze|freezing|frozen|stop\\s+temporarily|suspend|suspended|suspending|halt\\s+temporarily|break|breaking|broke|interrupt|interrupting|interrupted)\\b",

            "stop" to "\\b(?:stop|stopped|stopping|end|ended|ending|finish|finished|finishing|terminate|terminated|terminating|cease|halt|kill|abort|cancel|cancelled|canceling|quit|exit|close|shut\\s+down|power\\s+off|turn\\s+off|switch\\s+off|deactivate|disable|disengage|cut\\s+off)\\b",

            "skip_next" to "\\b(?:skip\\s+(?:forward|next|ahead)|next\\s+(?:track|song|chapter|episode|video|clip)|forward\\s+(?:to\\s+next|one)|advance\\s+(?:to\\s+next|one)|go\\s+(?:to\\s+next|forward\\s+one)|jump\\s+(?:to\\s+next|forward)|fast\\s+forward\\s+(?:to\\s+next|one)|move\\s+(?:to\\s+next|forward)|switch\\s+(?:to\\s+next|forward))\\b",

            "skip_previous" to "\\b(?:skip\\s+(?:back|previous|backward)|previous\\s+(?:track|song|chapter|episode|video|clip)|back\\s+(?:to\\s+previous|one)|go\\s+(?:to\\s+previous|back\\s+one)|jump\\s+(?:to\\s+previous|back)|rewind\\s+(?:to\\s+previous|one)|move\\s+(?:to\\s+previous|back)|switch\\s+(?:to\\s+previous|back)|restart\\s+(?:track|song|current))\\b",

            "fast_forward" to "\\b(?:fast\\s+forward|speed\\s+up|forward\\s+(?:quickly|fast)|accelerate\\s+(?:playback|forward)|rush\\s+forward|zoom\\s+forward|hurry\\s+forward|quick\\s+forward|rapid\\s+forward|expedite\\s+forward|double\\s+speed|triple\\s+speed|increase\\s+speed)\\b",

            "rewind" to "\\b(?:rewind|rewinding|rewound|fast\\s+backward|reverse\\s+(?:quickly|fast)|back\\s+up|go\\s+back|reverse\\s+playback|backward\\s+(?:quickly|fast)|retreat|retreating|retreated|regress|regressing|regressed|slow\\s+reverse|reverse\\s+slowly)\\b",

            "seek" to "\\b(?:seek|seeking|sought|jump\\s+to|go\\s+to|move\\s+to|navigate\\s+to|position\\s+to|scrub\\s+to|advance\\s+to|retreat\\s+to|set\\s+position|change\\s+position|adjust\\s+position|locate\\s+to|find\\s+position|progress\\s+to)\\b",

            "mute" to "\\b(?:mute|muting|muted|silence|silencing|silenced|quiet|quieting|quieted|turn\\s+off\\s+sound|disable\\s+sound|kill\\s+sound|cut\\s+sound|no\\s+sound|sound\\s+off|audio\\s+off|volume\\s+off|silent\\s+mode)\\b",

            "unmute" to "\\b(?:unmute|unmuting|unmuted|unsilence|unsilencing|unsilenced|unquiet|unquieting|unquieted|turn\\s+on\\s+sound|enable\\s+sound|restore\\s+sound|bring\\s+back\\s+sound|sound\\s+on|audio\\s+on|volume\\s+on|exit\\s+silent\\s+mode)\\b",

            "fullscreen" to "\\b(?:full\\s+screen|fullscreen|full-screen|maximize|maximizing|maximized|expand|expanding|expanded|enlarge|enlarging|enlarged|stretch|stretching|stretched|fill\\s+screen|screen\\s+fill|wide\\s+screen|cinema\\s+mode|theater\\s+mode|immersive\\s+mode)\\b",

            "captions" to "\\b(?:caption|captions|subtitle|subtitles|closed\\s+caption|closed\\s+captions|cc|cc'?s|text|texts|transcript|transcripts|sub|subs|overlay|overlays|dialogue|dialogues|speech\\s+text|spoken\\s+text|audio\\s+description)\\b",

            "speed" to "\\b(?:speed|speeding|sped|playback\\s+speed|play\\s+speed|rate|rates|pace|pacing|paced|tempo|tempos|tempoing|tempoed|rhythm|rhythms|rhythmical|velocity|velocities|quickness|quicknesses|rapidity|rapidities)\\b",

            "shuffle" to "\\b(?:shuffle|shuffling|shuffled|random|randomize|randomizing|randomized|mix|mixing|mixed|scramble|scrambling|scrambled|jumble|jumbling|jumbled|disorder|disordering|disordered|rearrange|rearranging|rearranged)\\b",

            "repeat" to "\\b(?:repeat|repeating|repeated|loop|looping|looped|cycle|cycling|cycled|replay|replaying|replayed|encore|encoring|encored|again|repeat\\s+mode|loop\\s+mode|continuous\\s+play|infinite\\s+play)\\b",

            "increase" to "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b",

            "decrease" to "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b"
        )
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        val sortedActions = mediaActions.entries.sortedByDescending { it.value.length }
        
        for ((action, pattern) in sortedActions) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return action
            }
        }
        
        return null
    }

    private fun extractAppAction(text: String): String? {
        val appActions = mapOf(
            "open" to "\\b(?:open|opened|opening|launch|launched|launching|start|show|display|view|access|load|bring\\s+up|pull\\s+up|fire\\s+up|boot|go\\s+to|navigate\\s+to|switch\\s+to|take\\s+me\\s+to|turn\\s+on|on|enable|enabled|activate|activated|power\\s+on|switch\\s+on)\\b",

            "increase" to "\\b(?:increase|increased|increasing|up|higher|raise|raised|raising|boost|boosted|boosting|amplify|amplified|amplifying|enhance|enhanced|enhancing|elevate|elevated|elevating|pump\\s+up|turn\\s+up|crank\\s+up|ramp\\s+up|scale\\s+up|step\\s+up|jack\\s+up|bump\\s+up|push\\s+up|bring\\s+up|make\\s+it\\s+higher|louder|brighter|stronger|more|maximize|max\\s+out|intensify)\\b",

            "decrease" to "\\b(?:decrease|decreased|decreasing|down|lower|lowered|lowering|reduce|reduced|reducing|diminish|diminished|diminishing|lessen|lessened|lessening|drop|dropped|dropping|cut|cutting|turn\\s+down|bring\\s+down|scale\\s+down|step\\s+down|tone\\s+down|dial\\s+down|wind\\s+down|ramp\\s+down|make\\s+it\\s+lower|quieter|dimmer|weaker|less|minimize|min\\s+out|soften)\\b"
        )
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        val sortedActions = appActions.entries.sortedByDescending { it.value.length }
        
        for ((action, pattern) in sortedActions) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return action
            }
        }
        
        return null
    }

    private fun extractPhoneAction(text: String): String? {
        val phoneActions = mapOf(
            "call" to "\\b(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|reach\\s+out|get\\s+in\\s+touch|give\\s+a\\s+call|make\\s+a\\s+call|place\\s+a\\s+call|telephone|buzz|video\\s+call|voice\\s+call|facetime)\\b",
            
            "message" to "\\b(?:message|messaging|text|texting|sms|send|sending|sent|write|compose|type|drop\\s+a\\s+message|send\\s+a\\s+text|shoot\\s+a\\s+message|ping|dm|direct\\s+message|whatsapp|imessage|chat|msg)\\b"
        )
        
        // Sort actions by pattern specificity (longer patterns first for better matching)
        val sortedActions = phoneActions.entries.sortedByDescending { it.value.length }
        
        for ((action, pattern) in sortedActions) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return action
            }
        }
        
        return null
    }
    
    private fun extractTool(text: String): String? {
        return when {
            text.contains("\\b(?:alarm|alarms|wake\\s+up|wake\\s+me|wake\\s+me\\s+up|morning\\s+alarm|set\\s+alarm|alarm\\s+clock|wakeup|wake-up|rouse|rise|get\\s+up|ring|ringer|buzzer|morning\\s+call|wake\\s+call|alert\\s+me|morning\\s+alert|sleep\\s+alarm|snooze|beep)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "alarm"
        
            // Timer - expanded with 20+ variations
            text.contains("\\b(?:timer|timers|countdown|count\\s+down|set\\s+timer|start\\s+timer|kitchen\\s+timer|cooking\\s+timer|egg\\s+timer|time\\s+me|timing|timed|set\\s+a\\s+timer|countdown\\s+timer|interval\\s+timer|remind\\s+me\\s+in|alert\\s+in|notify\\s+in|time\\s+limit|duration|time\\s+out)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "timer"
            
            // Stopwatch - expanded with 20+ variations
            text.contains("\\b(?:stopwatch|stop\\s+watch|chronometer|lap\\s+timer|lap\\s+time|split\\s+time|time\\s+lap|elapsed\\s+time|running\\s+time|measure\\s+time|track\\s+time|timing|chrono|lap|laps|split|splits|time\\s+it|how\\s+long|duration\\s+tracker)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "stopwatch"
            else -> null
        }
    }
    
    private fun extractActivityType(text: String): String? {
        val activities = mapOf(
            "outdoor running" to "\\b(?<!indoor\\s)(?:outdoor\\s+)?(?:run|running|ran|jog|jogging|jogged|sprint|sprinting|sprinted|dash|dashing|race|racing|trail\\s+run|trail\\s+running|distance\\s+run|long\\s+run|short\\s+run|tempo\\s+run|interval\\s+run|fartlek|road\\s+run|cross\\s+country|marathon|half\\s+marathon|5k|10k|runner|runners|pace|pacing|stride|striding|gallop|galloping|bound|bounding|hurdle|hurdling|relay|relays|track\\s+running|road\\s+racing|fun\\s+run|charity\\s+run|morning\\s+run|evening\\s+run|daily\\s+run|weekly\\s+run|cardio\\s+run|fitness\\s+run|outdoor\\s+jog|park\\s+run|street\\s+running|pavement\\s+running|sidewalk\\s+running|neighborhood\\s+run|community\\s+run|group\\s+run|solo\\s+run|recreational\\s+running|competitive\\s+running|distance\\s+running|endurance\\s+running|speed\\s+work)\\b",
            
            "indoor cycling" to "\\b(?<!outdoor\\s)indoor\\s+(?:cycling|cycle|cycled|bike|biking|biked|bicycle|bicycling|spin|spinning|spin\\s+class|stationary\\s+bike|exercise\\s+bike|bike\\s+ride|pedal|pedaling|pedalled|indoor\\s+bike|cycle\\s+class|RPM|cadence|peloton|zwift|virtual\\s+cycling|turbo\\s+trainer|trainer\\s+ride|cycling indoor)\\b",
            
            "yoga" to "\\b(?:yoga|yogi|asana|asanas|meditation|meditate|meditating|meditated|stretch|stretching|stretched|flexibility|vinyasa|hatha|ashtanga|bikram|hot\\s+yoga|power\\s+yoga|yin\\s+yoga|restorative\\s+yoga|pranayama|breathing\\s+exercise|mindfulness|zen|namaste|downward\\s+dog|warrior\\s+pose|sun\\s+salutation|flow|yoga\\s+flow|kundalini|iyengar|anusara|kripalu|sivananda|gentle\\s+yoga|beginner\\s+yoga|advanced\\s+yoga|therapeutic\\s+yoga|chair\\s+yoga|wall\\s+yoga|aerial\\s+yoga|yoga\\s+nidra|relaxation\\s+yoga|spiritual\\s+yoga|classical\\s+yoga|modern\\s+yoga|fusion\\s+yoga|yoga\\s+therapy|yoga\\s+practice|yoga\\s+session|yoga\\s+class|yoga\\s+workout|yoga\\s+routine|morning\\s+yoga|evening\\s+yoga|bedtime\\s+yoga|wake\\s+up\\s+yoga|desk\\s+yoga|office\\s+yoga|travel\\s+yoga|outdoor\\s+yoga|beach\\s+yoga|park\\s+yoga|home\\s+yoga|studio\\s+yoga|group\\s+yoga|private\\s+yoga|online\\s+yoga|virtual\\s+yoga)\\b",

            "indoor running" to "\\b(?:indoor\\s+running|indoor\\s+run|indoor\\s+jog|indoor\\s+jogging|treadmill\\s+running|treadmill\\s+run|stationary\\s+running|stationary\\s+run)\\b",

            "treadmill" to "\\b(?:treadmill|tread\\s+mill|running\\s+machine|jogging\\s+machine|mill|stationary\\s+run|indoor\\s+running\\s+machine|treadmill\\s+walking|treadmill\\s+walk|treadmill\\s+running|treadmill\\s+jogging|treadmill\\s+workout|treadmill\\s+session|treadmill\\s+exercise|motorized\\s+treadmill|manual\\s+treadmill|incline\\s+treadmill|decline\\s+treadmill|folding\\s+treadmill|commercial\\s+treadmill|home\\s+treadmill|gym\\s+treadmill|cardio\\s+machine|running\\s+belt|walking\\s+belt|conveyor\\s+belt\\s+exercise|speed\\s+walking\\s+machine|pace\\s+machine|endurance\\s+machine|fitness\\s+machine|aerobic\\s+machine|indoor\\s+cardio|stationary\\s+running|moving\\s+belt|electric\\s+treadmill|digital\\s+treadmill|programmable\\s+treadmill|interval\\s+treadmill|hill\\s+program\\s+treadmill)\\b",

            "trekking" to "\\b(?:trekking|trek|trekked|backpacking|backpack|mountain\\s+trek|hill\\s+trek|trekking\\s+expedition|trekking\\s+adventure|trekking\\s+journey|hiking\\s+expedition|wilderness\\s+trek|alpine\\s+trek|high\\s+altitude\\s+trek|base\\s+camp\\s+trek|peak\\s+trekking|summit\\s+trekking|long\\s+distance\\s+hiking|multi\\s+day\\s+hiking|overnight\\s+hiking|camping\\s+trek|trail\\s+trekking|forest\\s+trekking|jungle\\s+trekking|desert\\s+trekking|coastal\\s+trekking|ridge\\s+trekking|valley\\s+trekking|glacier\\s+trekking|snow\\s+trekking|technical\\s+trekking|easy\\s+trekking|moderate\\s+trekking|difficult\\s+trekking|extreme\\s+trekking|guided\\s+trekking|solo\\s+trekking|group\\s+trekking|adventure\\s+walking|expedition\\s+walking|mountaineering\\s+approach|base\\s+camp\\s+approach)\\b",

            "trail running" to "\\b(?:trail\\s+running|trail\\s+run|trail\\s+jog|trail\\s+jogging|mountain\\s+running|mountain\\s+run|off\\s+road\\s+running|off\\s+road\\s+run|trail\\s+runner|trail\\s+racing|ultra\\s+trail\\s+running|skyrunning|sky\\s+running|fell\\s+running|hill\\s+running|cross\\s+country\\s+trail|wilderness\\s+running|nature\\s+running|forest\\s+running|single\\s+track\\s+running|technical\\s+trail\\s+running|rocky\\s+trail\\s+running|muddy\\s+trail\\s+running|steep\\s+trail\\s+running|uphill\\s+running|downhill\\s+running|alpine\\s+running|desert\\s+trail\\s+running|coastal\\s+trail\\s+running|ridge\\s+running|valley\\s+running|canyon\\s+running|gorge\\s+running|switchback\\s+running|scrambling\\s+running|adventure\\s+running|endurance\\s+trail\\s+running|ultra\\s+marathon\\s+trail|long\\s+distance\\s+trail|multi\\s+day\\s+trail|stage\\s+race\\s+trail|vertical\\s+running)\\b",

            "outdoor walking" to "\\b(?<!indoor\\s)(?:outdoor\\s+walking|outdoor\\s+walk|outdoor\\s+hike|outdoor\\s+stroll|nature\\s+walk|trail\\s+walk|hill\\s+walk|forest\\s+walk|walk|walking|walked|walker|stroll|strolling|strolled|hike|hiking|hiked|hiker|trek|trekking|trekked|ramble|rambling|wander|wandering|wandered|amble|ambling|march|marching|power\\s+walk|brisk\\s+walk|leisurely\\s+walk|nature\\s+walk|trail\\s+walk|hill\\s+walk|speed\\s+walk|fitness\\s+walk|evening\\s+walk|morning\\s+walk)\\b",

            "indoor walking" to "\\b(?:indoor\\s+walking|indoor\\s+walk|indoor\\s+stroll|treadmill\\s+walking|treadmill\\s+walk|walk indoor|walking indoor)\\b",

            "outdoor cycling" to "\\b(?<!indoor\\s)(?<!quad\\s)(?:outdoor\\s+cycling|outdoor\\s+bike|outdoor\\s+biking|road\\s+cycling|road\\s+bike|mountain\\s+bike|mountain\\s+biking|bike\\s+ride|cycle\\s+ride|bicycle\\s+ride|cycling|cycle|cycled|bike|biking|biked|bicycle|bicycling|pedal|pedaling|pedalled|two\\s+wheeler|pushbike|velocipede|touring\\s+bike|hybrid\\s+bike|commuter\\s+bike|recreational\\s+cycling|leisure\\s+cycling|weekend\\s+ride|group\\s+ride|solo\\s+ride|charity\\s+ride|bike\\s+tour|cycling\\s+tour|bike\\s+trip|cycling\\s+trip|bicycle\\s+tour|road\\s+biking|trail\\s+biking|cross\\s+country\\s+biking|endurance\\s+cycling|speed\\s+cycling|time\\s+trial|criterium|gran\\s+fondo|century\\s+ride|metric\\s+century|bike\\s+workout|cycling\\s+workout|outdoor\\s+pedaling|fresh\\s+air\\s+cycling|nature\\s+cycling|scenic\\s+ride|countryside\\s+cycling)\\b",

            "bmx" to "\\b(?:bmx|bmx\\s+bike|bmx\\s+racing|bmx\\s+riding|bmx\\s+freestyle|bmx\\s+street|bmx\\s+park|bmx\\s+dirt|bmx\\s+vert|bmx\\s+flatland|bicycle\\s+motocross|bike\\s+motocross|twenty\\s+inch\\s+bike|small\\s+wheel\\s+bike|stunt\\s+bike|trick\\s+bike|jump\\s+bike|ramp\\s+bike|skate\\s+park\\s+bike|pump\\s+track|bmx\\s+track|bmx\\s+course|bmx\\s+trail|bmx\\s+session|recreational\\s+bmx|competitive\\s+bmx|professional\\s+bmx|amateur\\s+bmx|youth\\s+bmx|adult\\s+bmx|racing\\s+bmx|freestyle\\s+bmx|old\\s+school\\s+bmx|new\\s+school\\s+bmx|vintage\\s+bmx|modern\\s+bmx|single\\s+speed\\s+bmx|fixed\\s+gear\\s+bmx)\\b",

            "pool swimming" to "\\b(?:pool\\s+swimming|pool\\s+swim|swimming|pool\\s+workout|lap\\s+swimming|lap\\s+swim|indoor\\s+swimming|indoor\\s+swim|chlorinated\\s+pool|swimming\\s+pool|olympic\\s+pool|competition\\s+pool|lane\\s+swimming|freestyle\\s+swimming|backstroke\\s+swimming|breaststroke\\s+swimming|butterfly\\s+swimming|individual\\s+medley|medley\\s+swimming|sprint\\s+swimming|distance\\s+swimming|endurance\\s+swimming|interval\\s+swimming|tempo\\s+swimming|technique\\s+swimming|stroke\\s+work|kick\\s+sets|pull\\s+sets|swim\\s+drills|swimming\\s+laps|length\\s+swimming|recreational\\s+pool\\s+swimming|competitive\\s+pool\\s+swimming|masters\\s+swimming|youth\\s+swimming|adult\\s+swimming|senior\\s+swimming|aquatic\\s+center|natatorium|swimming\\s+facility|heated\\s+pool)\\b",

            "open water" to "\\b(?:open\\s+water|open\\s+water\\s+swimming|open\\s+water\\s+swim|ocean\\s+swimming|ocean\\s+swim|lake\\s+swimming|lake\\s+swim|sea\\s+swimming|sea\\s+swim|river\\s+swimming|river\\s+swim|bay\\s+swimming|bay\\s+swim|pond\\s+swimming|pond\\s+swim|natural\\s+water\\s+swimming|wild\\s+swimming|cold\\s+water\\s+swimming|salt\\s+water\\s+swimming|fresh\\s+water\\s+swimming|tidal\\s+swimming|current\\s+swimming|wave\\s+swimming|surf\\s+swimming|channel\\s+swimming|crossing\\s+swimming|marathon\\s+swimming|ultra\\s+swimming|endurance\\s+open\\s+water|triathlon\\s+swimming|wetsuit\\s+swimming|skin\\s+swimming|outdoor\\s+swimming|natural\\s+swimming|adventure\\s+swimming|expedition\\s+swimming|recreational\\s+open\\s+water|competitive\\s+open\\s+water|mass\\s+start\\s+swimming|beach\\s+start\\s+swimming)\\b",

            "strength training" to "\\b(?:strength\\s+training|strength|weight\\s+training|weightlifting|lift|lifting|resistance\\s+training|muscle\\s+building|power\\s+training|training|pump|pumping|iron|pumping\\s+iron|gym\\s+workout|weight\\s+workout|muscle\\s+workout|bodybuilding|body\\s+building|powerlifting|olympic\\s+lifting|functional\\s+strength|compound\\s+movements|isolation\\s+exercises|progressive\\s+overload|hypertrophy|mass\\s+building|lean\\s+muscle|toning|muscle\\s+toning|resistance\\s+exercises|free\\s+weights|machine\\s+weights|cable\\s+exercises|strength\\s+conditioning|athletic\\s+training|performance\\s+training|sports\\s+training|cross\\s+training|circuit\\s+training|superset|drop\\s+set|pyramid\\s+training|periodization|strength\\s+building|muscle\\s+gain|power\\s+development|maximal\\s+strength|muscular\\s+endurance)\\b",

            "weightlifting" to "\\b(?:weightlifting|weight\\s+lifting|olympic\\s+lifting|powerlifting|clean|snatch|jerk|weightlifting\\s+workout|weightlifting\\s+session|weightlifting\\s+practice|clean\\s+and\\s+jerk|snatch\\s+and\\s+clean|heavy\\s+lifting|competitive\\s+lifting|strength\\s+lifting|power\\s+lifting|olympic\\s+weightlifting|iwf\\s+lifting|platform\\s+lifting|competition\\s+lifting|barbell\\s+lifting|technique\\s+lifting|squat\\s+bench\\s+deadlift|big\\s+three\\s+lifts|maximal\\s+lifting|one\\s+rep\\s+max|personal\\s+record\\s+lifting|pr\\s+attempt|recreational\\s+weightlifting|competitive\\s+weightlifting|professional\\s+weightlifting|amateur\\s+weightlifting|youth\\s+weightlifting|masters\\s+weightlifting|women\\s+weightlifting|men\\s+weightlifting|powerlifting\\s+meet|weightlifting\\s+meet|strength\\s+sport|iron\\s+game|heavy\\s+metal\\s+lifting)\\b",

            "dumbbell training" to "\\b(?:dumbbell\\s+training|dumbbell|dumbells|free\\s+weights|dumbbell\\s+workout|dumbbell\\s+session|dumbbell\\s+exercise|dumbbell\\s+routine|dumbbell\\s+circuit|dumbbell\\s+set|adjustable\\s+dumbbells|fixed\\s+dumbbells|rubber\\s+dumbbells|iron\\s+dumbbells|chrome\\s+dumbbells|neoprene\\s+dumbbells|hex\\s+dumbbells|round\\s+dumbbells|olympic\\s+dumbbells|powerblock\\s+dumbbells|selecttech\\s+dumbbells|unilateral\\s+training|single\\s+arm\\s+training|bilateral\\s+training|functional\\s+dumbbell|compound\\s+dumbbell|isolation\\s+dumbbell|dumbbell\\s+press|dumbbell\\s+curl|dumbbell\\s+row|dumbbell\\s+squat|dumbbell\\s+lunge|dumbbell\\s+swing|dumbbell\\s+clean|dumbbell\\s+snatch|dumbbell\\s+complex|dumbbell\\s+flow|home\\s+dumbbell\\s+workout|gym\\s+dumbbell\\s+workout)\\b",

            "barbell training" to "\\b(?:barbell\\s+training|barbell|barbells|bar\\s+training|barbell\\s+workout|barbell\\s+session|barbell\\s+exercise|barbell\\s+routine|barbell\\s+lifting|olympic\\s+barbell|standard\\s+barbell|powerlifting\\s+barbell|deadlift\\s+bar|squat\\s+bar|bench\\s+press\\s+bar|curl\\s+bar|ez\\s+bar|trap\\s+bar|hex\\s+bar|safety\\s+bar|cambered\\s+bar|swiss\\s+bar|football\\s+bar|multi\\s+grip\\s+bar|straight\\s+bar|curved\\s+bar|knurled\\s+bar|smooth\\s+bar|twenty\\s+kg\\s+bar|forty\\s+five\\s+pound\\s+bar|barbell\\s+complex|barbell\\s+circuit|compound\\s+barbell|barbell\\s+squat|barbell\\s+deadlift|barbell\\s+bench\\s+press|barbell\\s+row|barbell\\s+curl|barbell\\s+press|barbell\\s+clean|barbell\\s+snatch|heavy\\s+barbell|light\\s+barbell|loaded\\s+barbell)\\b",

            "deadlift" to "\\b(?:deadlift|dead\\s+lift|deadlifts|deadlifting|deadlift\\s+workout|deadlift\\s+session|deadlift\\s+practice|conventional\\s+deadlift|sumo\\s+deadlift|romanian\\s+deadlift|stiff\\s+leg\\s+deadlift|trap\\s+bar\\s+deadlift|hex\\s+bar\\s+deadlift|deficit\\s+deadlift|block\\s+deadlift|rack\\s+pull|partial\\s+deadlift|paused\\s+deadlift|touch\\s+and\\s+go\\s+deadlift|single\\s+leg\\s+deadlift|unilateral\\s+deadlift|dumbbell\\s+deadlift|kettlebell\\s+deadlift|barbell\\s+deadlift|heavy\\s+deadlift|light\\s+deadlift|max\\s+deadlift|one\\s+rep\\s+max\\s+deadlift|pr\\s+deadlift|deadlift\\s+max|deadlift\\s+day|deadlift\\s+form|deadlift\\s+technique|deadlift\\s+lockout|deadlift\\s+setup|powerlifting\\s+deadlift|strongman\\s+deadlift)\\b",

            "sit ups" to "\\b(?:sit\\s+ups|situps|crunches|crunch|ab\\s+workout|abs|sit\\s+up\\s+exercise|abdominal\\s+exercise|stomach\\s+exercise|core\\s+crunches|full\\s+sit\\s+ups|partial\\s+sit\\s+ups|decline\\s+sit\\s+ups|incline\\s+sit\\s+ups|weighted\\s+sit\\s+ups|unweighted\\s+sit\\s+ups|bicycle\\s+crunches|reverse\\s+crunches|oblique\\s+crunches|russian\\s+twists|captain\\s+chair|knee\\s+raises|leg\\s+raises|v\\s+ups|flutter\\s+kicks|scissor\\s+kicks|dead\\s+bug|mountain\\s+climbers|ab\\s+ripper|six\\s+pack\\s+workout|rectus\\s+abdominis|transverse\\s+abdominis|internal\\s+obliques|external\\s+obliques|abdominal\\s+training|core\\s+strengthening|midsection\\s+work|trunk\\s+flexion|spinal\\s+flexion)\\b",

            "core training" to "\\b(?:core\\s+training|core|abs|abdominals|planks|plank|core\\s+workout|core\\s+session|core\\s+exercise|core\\s+routine|core\\s+strengthening|core\\s+conditioning|core\\s+stability|core\\s+endurance|core\\s+power|functional\\s+core|dynamic\\s+core|static\\s+core|isometric\\s+core|isotonic\\s+core|rotational\\s+core|anti\\s+rotation|anti\\s+extension|anti\\s+flexion|spinal\\s+stabilization|trunk\\s+training|midsection\\s+training|deep\\s+core|superficial\\s+core|transverse\\s+abdominis|multifidus|pelvic\\s+floor|diaphragm\\s+training|breathing\\s+core|postural\\s+core|athletic\\s+core|rehabilitation\\s+core|therapeutic\\s+core|medicine\\s+ball\\s+core|stability\\s+ball\\s+core|bosu\\s+ball\\s+core|suspension\\s+trainer\\s+core|bodyweight\\s+core|weighted\\s+core)\\b",

            "pilates" to "\\b(?:pilates|pilates\\s+workout|pilates\\s+class|pilates\\s+session|pilates\\s+exercise|pilates\\s+routine|mat\\s+pilates|reformer\\s+pilates|classical\\s+pilates|contemporary\\s+pilates|modern\\s+pilates|clinical\\s+pilates|rehabilitation\\s+pilates|therapeutic\\s+pilates|prenatal\\s+pilates|postnatal\\s+pilates|senior\\s+pilates|beginner\\s+pilates|intermediate\\s+pilates|advanced\\s+pilates|pilates\\s+apparatus|cadillac\\s+pilates|tower\\s+pilates|chair\\s+pilates|barrel\\s+pilates|spine\\s+corrector|magic\\s+circle|pilates\\s+ring|pilates\\s+ball|pilates\\s+band|contrology|joseph\\s+pilates|pilates\\s+method|pilates\\s+principles|concentration\\s+pilates|control\\s+pilates|centering\\s+pilates|flow\\s+pilates|precision\\s+pilates|breathing\\s+pilates)\\b",

            "hiit" to "\\b(?:hiit|hiit\\s+workout|hiit\\s+session|hiit\\s+class|hiit\\s+circuit|high\\s+intensity\\s+workout|intense\\s+interval|sprint\\s+interval|work\\s+rest\\s+interval|tabata|tabata\\s+protocol|emom|every\\s+minute\\s+on\\s+the\\s+minute|amrap|as\\s+many\\s+rounds\\s+as\\s+possible|metabolic\\s+conditioning|metcon|anaerobic\\s+training|lactate\\s+threshold|vo2\\s+max\\s+training|cardio\\s+intervals|strength\\s+intervals|plyometric\\s+intervals|bodyweight\\s+hiit|equipment\\s+hiit|treadmill\\s+hiit|bike\\s+hiit|rowing\\s+hiit|kettlebell\\s+hiit|dumbbell\\s+hiit|functional\\s+hiit|athletic\\s+hiit|fat\\s+burning\\s+hiit|calorie\\s+burning\\s+hiit)\\b",

            "functional training" to "\\b(?:functional\\s+training|functional|functional\\s+fitness|movement\\s+training|functional\\s+exercise|functional\\s+workout|functional\\s+session|functional\\s+routine|functional\\s+movement|natural\\s+movement|primal\\s+movement|everyday\\s+movement|real\\s+world\\s+training|activities\\s+of\\s+daily\\s+living|adl\\s+training|multi\\s+planar\\s+movement|three\\s+dimensional\\s+movement|compound\\s+movement|integrated\\s+movement|kinetic\\s+chain|movement\\s+patterns|fundamental\\s+movement|corrective\\s+exercise|movement\\s+quality|movement\\s+efficiency|functional\\s+strength|functional\\s+mobility|functional\\s+stability|functional\\s+flexibility|functional\\s+power|functional\\s+endurance|sport\\s+specific\\s+training|task\\s+specific\\s+training|occupational\\s+training|rehabilitation\\s+training|injury\\s+prevention|performance\\s+enhancement|movement\\s+screening|movement\\s+assessment)\\b",

            "elliptical machine" to "\\b(?:elliptical|elliptical\\s+machine|elliptical\\s+trainer|cross\\s+trainer|elliptical\\s+workout|elliptical\\s+session|elliptical\\s+exercise|elliptical\\s+cardio|cross\\s+training\\s+machine|dual\\s+action\\s+trainer|total\\s+body\\s+trainer|upper\\s+lower\\s+body\\s+machine|front\\s+drive\\s+elliptical|rear\\s+drive\\s+elliptical|center\\s+drive\\s+elliptical|commercial\\s+elliptical|home\\s+elliptical|gym\\s+elliptical|incline\\s+elliptical|variable\\s+stride\\s+elliptical|adjustable\\s+elliptical|programmable\\s+elliptical|heart\\s+rate\\s+elliptical|interval\\s+elliptical|hill\\s+program\\s+elliptical|fat\\s+burn\\s+elliptical|cardio\\s+elliptical|endurance\\s+elliptical|low\\s+impact\\s+cardio|joint\\s+friendly\\s+cardio|non\\s+impact\\s+exercise|smooth\\s+motion\\s+machine|gliding\\s+motion)\\b",

            "stepper" to "\\b(?:stepper|step\\s+machine|stepper\\s+workout|stepper\\s+exercise|stepper\\s+session|stepper\\s+cardio|stair\\s+stepper|mini\\s+stepper|lateral\\s+stepper|twisting\\s+stepper|adjustable\\s+stepper|resistance\\s+stepper|hydraulic\\s+stepper|magnetic\\s+stepper|pneumatic\\s+stepper|portable\\s+stepper|compact\\s+stepper|home\\s+stepper|gym\\s+stepper|commercial\\s+stepper|step\\s+up\\s+exercise|step\\s+down\\s+exercise|stepping\\s+motion|up\\s+down\\s+motion|alternating\\s+step|continuous\\s+stepping|interval\\s+stepping|steady\\s+state\\s+stepping|low\\s+impact\\s+stepping|cardio\\s+stepping|endurance\\s+stepping|leg\\s+strengthening|calf\\s+workout|lower\\s+body\\s+cardio)\\b",

            "step training" to "\\b(?:step\\s+training|step|step\\s+class|step\\s+aerobics|step\\s+workout|step\\s+session|step\\s+exercise|step\\s+routine|step\\s+choreography|aerobic\\s+step|step\\s+platform|step\\s+bench|adjustable\\s+step|four\\s+inch\\s+step|six\\s+inch\\s+step|eight\\s+inch\\s+step|basic\\s+step|intermediate\\s+step|advanced\\s+step|low\\s+impact\\s+step|high\\s+impact\\s+step|step\\s+and\\s+tone|step\\s+and\\s+sculpt|step\\s+and\\s+strength|cardio\\s+step|power\\s+step|interval\\s+step|circuit\\s+step|dance\\s+step|freestyle\\s+step|choreographed\\s+step|group\\s+step|solo\\s+step|home\\s+step|gym\\s+step|studio\\s+step|step\\s+up|step\\s+down|step\\s+touch|knee\\s+up|kick\\s+back|over\\s+the\\s+top|around\\s+the\\s+world)\\b",

            "climbing machine" to "\\b(?:climbing\\s+machine|stair\\s+climber|stair\\s+master|vertical\\s+climber|climbing\\s+trainer|stair\\s+climbing\\s+machine|step\\s+climber|revolving\\s+stair|endless\\s+stair|stair\\s+mill|escalator\\s+machine|step\\s+mill|cardio\\s+climber|total\\s+body\\s+climber|upper\\s+lower\\s+climber|dual\\s+action\\s+climber|resistance\\s+climber|adjustable\\s+climber|programmable\\s+climber|commercial\\s+climber|home\\s+climber|gym\\s+climber|vertical\\s+cardio|stair\\s+workout|climbing\\s+workout|stair\\s+exercise|climbing\\s+exercise|stair\\s+training|climbing\\s+training|stair\\s+session|climbing\\s+session|stair\\s+cardio|climbing\\s+cardio|step\\s+after\\s+step|continuous\\s+climbing|interval\\s+climbing|endurance\\s+climbing|power\\s+climbing)\\b",

            "rowing machine" to "\\b(?:rowing\\s+machine|rower|indoor\\s+rowing|rowing\\s+workout|rowing\\s+trainer|rowing\\s+ergometer|erg|concept2|water\\s+rower|air\\s+rower|magnetic\\s+rower|hydraulic\\s+rower|resistance\\s+rower|commercial\\s+rower|home\\s+rower|gym\\s+rower|portable\\s+rower|foldable\\s+rower|full\\s+body\\s+rower|total\\s+body\\s+rowing|rowing\\s+exercise|rowing\\s+session|rowing\\s+training|rowing\\s+cardio|rowing\\s+endurance|rowing\\s+power|rowing\\s+intervals|steady\\s+state\\s+rowing|sprint\\s+rowing|long\\s+distance\\s+rowing|2k\\s+rowing|5k\\s+rowing|10k\\s+rowing|rowing\\s+technique|catch\\s+rowing|drive\\s+rowing|finish\\s+rowing|recovery\\s+rowing|stroke\\s+rate|split\\s+time|watts\\s+rowing|meters\\s+rowing|calories\\s+rowing)\\b",

            "rope skipping" to "\\b(?:rope\\s+skipping|jump\\s+rope|skipping|jump\\s+roping|rope\\s+jump|jump\\s+rope\\s+workout|jump\\s+rope\\s+training|jump\\s+rope\\s+session|jump\\s+rope\\s+exercise|skipping\\s+rope|rope\\s+jumping|speed\\s+rope|weighted\\s+rope|beaded\\s+rope|leather\\s+rope|wire\\s+rope|pvc\\s+rope|cotton\\s+rope|adjustable\\s+rope|crossrope|basic\\s+jump|alternate\\s+foot|single\\s+bounce|double\\s+bounce|criss\\s+cross|side\\s+swing|boxer\\s+step|two\\s+foot\\s+hop|single\\s+leg\\s+hop|high\\s+knees\\s+jump|butt\\s+kicks\\s+jump|double\\s+under|triple\\s+under|freestyle\\s+jumping|competitive\\s+jumping|recreational\\s+jumping|cardio\\s+jumping|interval\\s+jumping|endurance\\s+jumping|boxing\\s+jump\\s+rope|mma\\s+jump\\s+rope)\\b",

            "basketball" to "\\b(?:basketball|basketball\\s+game|basketball\\s+practice|b ball|b-ball|hoops|shooting\\s+hoops|bball|roundball|court\\s+game|five\\s+on\\s+five|full\\s+court|half\\s+court|pickup\\s+basketball|streetball|street\\s+basketball|indoor\\s+basketball|outdoor\\s+basketball|recreational\\s+basketball|competitive\\s+basketball|league\\s+basketball|tournament\\s+basketball|scrimmage|basketball\\s+scrimmage|basketball\\s+workout|basketball\\s+drills|free\\s+throws|three\\s+pointers|layups|dunking|dribbling|basketball\\s+skills|basketball\\s+conditioning|basketball\\s+fitness|team\\s+basketball|solo\\s+basketball|basketball\\s+shooting|shooting\\s+practice|basketball\\s+fundamentals)\\b",

            "football" to "\\b(?:football|soccer|football\\s+game|football\\s+practice|soccer\\s+game|soccer\\s+practice|futbol|footy|the\\s+beautiful\\s+game|association\\s+football|pitch\\s+game|eleven\\s+a\\s+side|kickball|ball\\s+game|field\\s+soccer|grass\\s+soccer|turf\\s+soccer|indoor\\s+soccer|outdoor\\s+soccer|recreational\\s+soccer|competitive\\s+soccer|league\\s+soccer|tournament\\s+soccer|pickup\\s+soccer|casual\\s+soccer|friendly\\s+match|soccer\\s+match|football\\s+match|soccer\\s+workout|football\\s+workout|soccer\\s+drills|football\\s+drills|soccer\\s+skills|football\\s+skills|soccer\\s+fitness|football\\s+fitness|team\\s+sport|world\\s+cup\\s+sport|penalty\\s+kicks|corner\\s+kicks|free\\s+kicks|goal\\s+scoring|soccer\\s+conditioning|football\\s+conditioning)\\b",

            "cricket" to "\\b(?:cricket|cricket\\s+game|cricket\\s+practice|cricket\\s+match|cricket\\s+workout|cricket\\s+session|cricket\\s+drill|bat\\s+and\\s+ball|wicket\\s+keeping|bowling\\s+cricket|batting\\s+practice|fielding\\s+practice|test\\s+cricket|one\\s+day\\s+cricket|t20\\s+cricket|twenty\\s+twenty|limited\\s+overs|county\\s+cricket|domestic\\s+cricket|international\\s+cricket|club\\s+cricket|recreational\\s+cricket|competitive\\s+cricket|professional\\s+cricket|amateur\\s+cricket|league\\s+cricket|tournament\\s+cricket|championship\\s+cricket|world\\s+cricket|youth\\s+cricket|junior\\s+cricket|senior\\s+cricket|women\\s+cricket|men\\s+cricket|mixed\\s+cricket|indoor\\s+cricket|outdoor\\s+cricket|turf\\s+cricket|pitch\\s+cricket|nets\\s+practice|cricket\\s+nets|wicket\\s+to\\s+wicket|stumps\\s+and\\s+bails|leather\\s+ball|cork\\s+ball|willow\\s+bat|cricket\\s+bat)\\b",

            "badminton" to "\\b(?:badminton|badminton\\s+game|badminton\\s+practice|badminton\\s+match|badminton\\s+training|badminton\\s+workout|badminton\\s+session|badminton\\s+drill|shuttlecock|birdie|racquet\\s+badminton|racket\\s+badminton|singles\\s+badminton|doubles\\s+badminton|mixed\\s+doubles\\s+badminton|net\\s+play\\s+badminton|court\\s+badminton|smash\\s+practice|drop\\s+shot\\s+practice|clear\\s+practice|drive\\s+practice|serve\\s+practice|recreational\\s+badminton|competitive\\s+badminton|professional\\s+badminton|amateur\\s+badminton|club\\s+badminton|league\\s+badminton|tournament\\s+badminton|championship\\s+badminton|world\\s+badminton|youth\\s+badminton|junior\\s+badminton|senior\\s+badminton|women\\s+badminton|men\\s+badminton|mixed\\s+badminton|indoor\\s+badminton|outdoor\\s+badminton|shuttle\\s+sport|feather\\s+shuttle|synthetic\\s+shuttle|speed\\s+badminton|power\\s+badminton|finesse\\s+badminton)\\b",

            "tennis" to "\\b(?:tennis|tennis\\s+game|tennis\\s+practice|lawn\\s+tennis|court\\s+tennis|racquet\\s+sport|racket\\s+sport|singles\\s+tennis|doubles\\s+tennis|mixed\\s+doubles|tennis\\s+match|tennis\\s+set|tennis\\s+tournament|tennis\\s+lesson|tennis\\s+coaching|tennis\\s+workout|tennis\\s+fitness|tennis\\s+drills|tennis\\s+skills|serve\\s+practice|return\\s+practice|baseline\\s+play|net\\s+play|volley\\s+practice|overhead\\s+practice|forehand\\s+practice|backhand\\s+practice|tennis\\s+conditioning|recreational\\s+tennis|competitive\\s+tennis|club\\s+tennis|league\\s+tennis|hard\\s+court\\s+tennis|clay\\s+court\\s+tennis|grass\\s+court\\s+tennis|indoor\\s+tennis|outdoor\\s+tennis|professional\\s+tennis|amateur\\s+tennis)\\b",

            "volleyball" to "\\b(?:volleyball|volleyball\\s+game|volleyball\\s+practice|volleyball\\s+match|volleyball\\s+workout|volleyball\\s+session|volleyball\\s+drill|volley\\s+ball|net\\s+volleyball|court\\s+volleyball|indoor\\s+volleyball|outdoor\\s+volleyball|beach\\s+volleyball|sand\\s+volleyball|grass\\s+volleyball|sitting\\s+volleyball|standing\\s+volleyball|spike\\s+practice|serve\\s+practice|dig\\s+practice|set\\s+practice|block\\s+practice|attack\\s+practice|defense\\s+practice|rotation\\s+practice|six\\s+a\\s+side|recreational\\s+volleyball|competitive\\s+volleyball|professional\\s+volleyball|amateur\\s+volleyball|club\\s+volleyball|league\\s+volleyball|tournament\\s+volleyball|championship\\s+volleyball|world\\s+volleyball|youth\\s+volleyball|junior\\s+volleyball|senior\\s+volleyball|women\\s+volleyball|men\\s+volleyball|mixed\\s+volleyball|college\\s+volleyball|school\\s+volleyball|team\\s+volleyball|doubles\\s+volleyball|fours\\s+volleyball|power\\s+volleyball|finesse\\s+volleyball)\\b",

            "baseball" to "\\b(?:baseball|baseball\\s+game|baseball\\s+practice|baseball\\s+match|baseball\\s+workout|baseball\\s+session|baseball\\s+drill|america\\s+pastime|diamond\\s+sport|nine\\s+innings|hardball|batting\\s+practice|pitching\\s+practice|fielding\\s+practice|catching\\s+practice|base\\s+running|home\\s+run\\s+practice|major\\s+league|minor\\s+league|little\\s+league|recreational\\s+baseball|competitive\\s+baseball|professional\\s+baseball|amateur\\s+baseball|club\\s+baseball|league\\s+baseball|tournament\\s+baseball|championship\\s+baseball|world\\s+series|youth\\s+baseball|junior\\s+baseball|senior\\s+baseball|college\\s+baseball|school\\s+baseball|tee\\s+ball|coach\\s+pitch|machine\\s+pitch|slow\\s+pitch\\s+baseball|fast\\s+pitch\\s+baseball|wood\\s+bat\\s+baseball|aluminum\\s+bat\\s+baseball|indoor\\s+baseball|outdoor\\s+baseball|diamond\\s+ball|ball\\s+and\\s+bat\\s+sport)\\b",

            "softball" to "\\b(?:softball|softball\\s+game|softball\\s+practice|softball\\s+match|softball\\s+training|softball\\s+workout|softball\\s+session|softball\\s+drill|soft\\s+ball|slow\\s+pitch|fast\\s+pitch|modified\\s+pitch|sixteen\\s+inch|twelve\\s+inch|ten\\s+inch|underhand\\s+softball|windmill\\s+softball|batting\\s+practice\\s+softball|pitching\\s+practice\\s+softball|fielding\\s+practice\\s+softball|seven\\s+innings|recreational\\s+softball|competitive\\s+softball|professional\\s+softball|amateur\\s+softball|club\\s+softball|league\\s+softball|tournament\\s+softball|championship\\s+softball|world\\s+softball|youth\\s+softball|junior\\s+softball|senior\\s+softball|women\\s+softball|men\\s+softball|mixed\\s+softball|co\\s+ed\\s+softball|church\\s+softball|company\\s+softball|beer\\s+league\\s+softball|diamond\\s+softball|mush\\s+ball|chicago\\s+softball|indoor\\s+softball|outdoor\\s+softball)\\b",

            "golf" to "\\b(?:golf|golfing|golf\\s+game|golf\\s+practice|golf\\s+round|golf\\s+workout|golf\\s+session|golf\\s+lesson|eighteen\\s+holes|nine\\s+holes|links\\s+golf|course\\s+golf|driving\\s+range|putting\\s+green|tee\\s+box|fairway\\s+play|approach\\s+shot|chip\\s+shot|putt\\s+practice|swing\\s+practice|iron\\s+play|wood\\s+play|driver\\s+practice|wedge\\s+play|putter\\s+practice|scratch\\s+golf|handicap\\s+golf|recreational\\s+golf|competitive\\s+golf|professional\\s+golf|amateur\\s+golf|club\\s+golf|tournament\\s+golf|championship\\s+golf|stroke\\s+play|match\\s+play|scramble\\s+golf|best\\s+ball|foursome\\s+golf|twosome\\s+golf|solo\\s+golf|par\\s+golf|bogey\\s+golf|birdie\\s+practice|eagle\\s+attempt|hole\\s+in\\s+one|course\\s+management|golf\\s+fitness|golf\\s+conditioning)\\b",

            "hockey" to "\\b(?:hockey|hockey\\s+game|hockey\\s+practice|hockey\\s+match|hockey\\s+workout|hockey\\s+session|hockey\\s+drill|ice\\s+hockey|field\\s+hockey|street\\s+hockey|ball\\s+hockey|roller\\s+hockey|inline\\s+hockey|pond\\s+hockey|rink\\s+hockey|puck\\s+handling|stick\\s+handling|shooting\\s+practice|passing\\s+practice|skating\\s+drills|power\\s+play\\s+practice|penalty\\s+kill|face\\s+off\\s+practice|checking\\s+practice|goalie\\s+practice|recreational\\s+hockey|competitive\\s+hockey|professional\\s+hockey|amateur\\s+hockey|club\\s+hockey|league\\s+hockey|tournament\\s+hockey|championship\\s+hockey|youth\\s+hockey|junior\\s+hockey|senior\\s+hockey|women\\s+hockey|men\\s+hockey|mixed\\s+hockey|college\\s+hockey|school\\s+hockey|shinny|pickup\\s+hockey|beer\\s+league\\s+hockey|outdoor\\s+hockey|indoor\\s+hockey|three\\s+periods|overtime\\s+hockey|shootout\\s+practice)\\b",

            "rugby" to "\\b(?:rugby|rugby\\s+game|rugby\\s+practice|rugby\\s+match|rugby\\s+training|rugby\\s+workout|rugby\\s+session|rugby\\s+drill|rugby\\s+union|rugby\\s+league|rugby\\s+sevens|rugby\\s+tens|rugby\\s+fifteens|rugby\\s+thirteens|oval\\s+ball|scrum\\s+practice|lineout\\s+practice|ruck\\s+practice|maul\\s+practice|tackle\\s+practice|passing\\s+practice|kicking\\s+practice|try\\s+scoring|conversion\\s+practice|penalty\\s+practice|drop\\s+goal\\s+practice|recreational\\s+rugby|competitive\\s+rugby|professional\\s+rugby|amateur\\s+rugby|club\\s+rugby|league\\s+rugby|tournament\\s+rugby|championship\\s+rugby|world\\s+cup\\s+rugby|youth\\s+rugby|junior\\s+rugby|senior\\s+rugby|women\\s+rugby|men\\s+rugby|mixed\\s+rugby|college\\s+rugby|school\\s+rugby|touch\\s+rugby|tag\\s+rugby|beach\\s+rugby|snow\\s+rugby|indoor\\s+rugby|outdoor\\s+rugby|forwards\\s+practice|backs\\s+practice|eighty\\s+minutes)\\b",

            "pingpong" to "\\b(?:pingpong|ping\\s+pong|table\\s+tennis|table\\s+tennis\\s+game|table\\s+tennis\\s+practice|table\\s+tennis\\s+match|table\\s+tennis\\s+training|table\\s+tennis\\s+workout|tt|paddle\\s+sport|bat\\s+and\\s+ball\\s+table|singles\\s+table\\s+tennis|doubles\\s+table\\s+tennis|mixed\\s+doubles\\s+table\\s+tennis|forehand\\s+practice|backhand\\s+practice|serve\\s+practice|spin\\s+practice|topspin|backspin|sidespin|smash\\s+practice|block\\s+practice|push\\s+practice|chop\\s+practice|loop\\s+practice|recreational\\s+table\\s+tennis|competitive\\s+table\\s+tennis|professional\\s+table\\s+tennis|amateur\\s+table\\s+tennis|club\\s+table\\s+tennis|league\\s+table\\s+tennis|tournament\\s+table\\s+tennis|championship\\s+table\\s+tennis|world\\s+table\\s+tennis|youth\\s+table\\s+tennis|junior\\s+table\\s+tennis|senior\\s+table\\s+tennis|women\\s+table\\s+tennis|men\\s+table\\s+tennis|mixed\\s+table\\s+tennis|indoor\\s+table\\s+tennis|basement\\s+table\\s+tennis|garage\\s+table\\s+tennis|eleven\\s+points|twenty\\s+one\\s+points)\\b",

            "squash" to "\\b(?:squash|squash\\s+game|squash\\s+practice|squash\\s+match|squash\\s+training|squash\\s+workout|squash\\s+session|squash\\s+drill|court\\s+squash|wall\\s+squash|racquet\\s+squash|racket\\s+squash|singles\\s+squash|doubles\\s+squash|hardball\\s+squash|softball\\s+squash|international\\s+squash|american\\s+squash|english\\s+squash|drive\\s+practice|drop\\s+shot\\s+practice|boast\\s+practice|kill\\s+shot\\s+practice|lob\\s+practice|serve\\s+practice|return\\s+practice|volley\\s+practice|recreational\\s+squash|competitive\\s+squash|professional\\s+squash|amateur\\s+squash|club\\s+squash|league\\s+squash|tournament\\s+squash|championship\\s+squash|world\\s+squash|youth\\s+squash|junior\\s+squash|senior\\s+squash|women\\s+squash|men\\s+squash|mixed\\s+squash|court\\s+sport|four\\s+wall\\s+sport|racquet\\s+sport\\s+squash|fitness\\s+squash|cardio\\s+squash|interval\\s+squash)\\b",

            "bowling" to "\\b(?:bowling|bowling\\s+game|bowling\\s+practice|tenpin|bowling\\s+alley|ten\\s+pin\\s+bowling|pin\\s+bowling|strike|spare|gutter\\s+ball|split|turkey|perfect\\s+game|300\\s+game|frame|bowling\\s+ball|bowling\\s+pin|bowling\\s+lane|bowling\\s+shoes|candlepin\\s+bowling|duckpin\\s+bowling|nine\\s+pin\\s+bowling|skittles|recreational\\s+bowling|competitive\\s+bowling|professional\\s+bowling|amateur\\s+bowling|league\\s+bowling|tournament\\s+bowling|championship\\s+bowling|cosmic\\s+bowling|glow\\s+bowling|midnight\\s+bowling|rock\\s+and\\s+roll\\s+bowling|family\\s+bowling|team\\s+bowling|couples\\s+bowling|singles\\s+bowling|doubles\\s+bowling|mixed\\s+doubles\\s+bowling|scratch\\s+bowling|handicap\\s+bowling|pba\\s+style|youth\\s+bowling|senior\\s+bowling|bowling\\s+fitness)\\b",

            "billiards" to "\\b(?:billiards|pool|billiards\\s+game|pool\\s+game|snooker|eight\\s+ball|nine\\s+ball|straight\\s+pool|one\\s+pocket|bank\\s+pool|rotation|cutthroat|three\\s+ball|ten\\s+ball|fifteen\\s+ball|cue\\s+sports|pocket\\s+billiards|carom\\s+billiards|english\\s+billiards|american\\s+pool|british\\s+pool|chinese\\s+eight\\s+ball|blackball|kelly\\s+pool|cowboy\\s+pool|bumper\\s+pool|bar\\s+pool|pub\\s+pool|recreational\\s+pool|competitive\\s+pool|professional\\s+pool|amateur\\s+pool|league\\s+pool|tournament\\s+pool|championship\\s+pool|world\\s+pool|masters\\s+pool|trick\\s+shots|artistic\\s+pool|speed\\s+pool|mosconi\\s+cup|pool\\s+hall|billiard\\s+hall|cue\\s+stick|chalk|rack|break|run\\s+out|shark|hustle)\\b",

            "darts" to "\\b(?:darts|darts\\s+game|darts\\s+practice|dartboard|dart\\s+throwing|501|301|cricket\\s+darts|around\\s+the\\s+clock|killer\\s+darts|shanghai|legs|sets|double\\s+out|single\\s+out|master\\s+out|bulls\\s+eye|bullseye|double\\s+bull|outer\\s+bull|treble|triple|double|checkout|finish|180|maximum|ton\\s+eighty|perfect\\s+dart|recreational\\s+darts|competitive\\s+darts|professional\\s+darts|amateur\\s+darts|league\\s+darts|tournament\\s+darts|championship\\s+darts|world\\s+darts|pdc\\s+darts|bdo\\s+darts|soft\\s+tip\\s+darts|steel\\s+tip\\s+darts|electronic\\s+darts|pub\\s+darts|bar\\s+darts|home\\s+darts|office\\s+darts|garage\\s+darts|basement\\s+darts|dart\\s+throwing|arrow\\s+throwing)\\b",

            "archery" to "\\b(?:archery|archery\\s+practice|bow\\s+and\\s+arrow|bow\\s+shooting|target\\s+archery|field\\s+archery|3d\\s+archery|traditional\\s+archery|olympic\\s+archery|recurve\\s+bow|compound\\s+bow|longbow|crossbow|barebow|instinctive\\s+archery|bowhunting|bow\\s+hunting|arrow\\s+shooting|bullseye\\s+archery|clout\\s+archery|flight\\s+archery|roving\\s+archery|popinjay|archer|bowman|toxophilite|recreational\\s+archery|competitive\\s+archery|professional\\s+archery|amateur\\s+archery|club\\s+archery|league\\s+archery|tournament\\s+archery|championship\\s+archery|world\\s+archery|indoor\\s+archery|outdoor\\s+archery|range\\s+archery|backyard\\s+archery|youth\\s+archery|adaptive\\s+archery|para\\s+archery|mounted\\s+archery|horseback\\s+archery|kyudo|japanese\\s+archery)\\b",

            "fencing" to "\\b(?:fencing|fencing\\s+practice|sword\\s+fighting|epee|foil|sabre|saber|rapier|blade\\s+work|swordplay|swordsmanship|touche|en\\s+garde|parry|riposte|lunge|thrust|cut|attack|defense|piste|strip|bout|match|hit|point|right\\s+of\\s+way|priority|electric\\s+fencing|classical\\s+fencing|modern\\s+fencing|sport\\s+fencing|historical\\s+fencing|hema|recreational\\s+fencing|competitive\\s+fencing|professional\\s+fencing|amateur\\s+fencing|club\\s+fencing|league\\s+fencing|tournament\\s+fencing|championship\\s+fencing|world\\s+fencing|olympic\\s+fencing|youth\\s+fencing|veteran\\s+fencing|wheelchair\\s+fencing|para\\s+fencing|team\\s+fencing|individual\\s+fencing|dry\\s+fencing|plastic\\s+fencing|mask\\s+work|footwork\\s+fencing|tactical\\s+fencing)\\b",

            "boxing" to "\\b(?:boxing|boxing\\s+match|fight|fighting|pugilism|prizefighting|sweet\\s+science|fisticuffs|shadow\\s+boxing|heavy\\s+bag|speed\\s+bag|double\\s+end\\s+bag|boxing\\s+conditioning|boxing\\s+drills|boxing\\s+technique|boxing\\s+skills|jab\\s+practice|cross\\s+practice|hook\\s+practice|uppercut\\s+practice|combination\\s+punches|pad\\s+work|mitt\\s+work|boxing\\s+footwork|ring\\s+work|boxing\\s+defense|boxing\\s+offense|amateur\\s+boxing|professional\\s+boxing|recreational\\s+boxing|fitness\\s+boxing|cardio\\s+boxing|boxing\\s+cardio|boxing\\s+class|boxing\\s+gym|white\\s+collar\\s+boxing)\\b",

            "karate" to "\\b(?:karate|karate\\s+class|karate\\s+drill|kata|kumite|kihon|basics|forms|shotokan|kyokushin|goju\\s+ryu|shito\\s+ryu|wado\\s+ryu|kenpo|kempo|karate\\s+do|empty\\s+hand|okinawan\\s+karate|japanese\\s+karate|traditional\\s+karate|sport\\s+karate|full\\s+contact\\s+karate|point\\s+fighting|continuous\\s+fighting|recreational\\s+karate|competitive\\s+karate|professional\\s+karate|amateur\\s+karate|club\\s+karate|dojo\\s+practice|tournament\\s+karate|championship\\s+karata|world\\s+karate|olympic\\s+karate|youth\\s+karate|adult\\s+karate|senior\\s+karate|family\\s+karate|women\\s+karate|men\\s+karate|kids\\s+karate|little\\s+dragons|karate\\s+fitness|karate\\s+conditioning|belt\\s+testing|grading)\\b",

            "judo" to "\\b(?:judo|judo\\s+practice|judo\\s+class|judo\\s+workout|judo\\s+session|judo\\s+drill|randori|kata|nage\\s+waza|katame\\s+waza|throwing|grappling|ground\\s+work|ne\\s+waza|tachi\\s+waza|standing\\s+techniques|jigoro\\s+kano|gentle\\s+way|maximum\\s+efficiency|mutual\\s+welfare|ippon|waza\\s+ari|yuko|koka|hajime|matte|soremade|recreational\\s+judo|competitive\\s+judo|professional\\s+judo|amateur\\s+judo|club\\s+judo|dojo\\s+practice|tournament\\s+judo|championship\\s+judo|world\\s+judo|olympic\\s+judo|ijf\\s+judo|youth\\s+judo|cadet\\s+judo|junior\\s+judo|senior\\s+judo|veteran\\s+judo|masters\\s+judo|adaptive\\s+judo|para\\s+judo|blind\\s+judo|wheelchair\\s+judo|women\\s+judo|men\\s+judo|mixed\\s+judo|team\\s+judo|individual\\s+judo|belt\\s+grading)\\b",

            "wrestling" to "\\b(?:wrestling|wrestling\\s+match|wrestling\\s+practice|wrestling\\s+workout|wrestling\\s+session|wrestling\\s+drill|freestyle\\s+wrestling|greco\\s+roman|folkstyle|collegiate\\s+wrestling|scholastic\\s+wrestling|amateur\\s+wrestling|professional\\s+wrestling|grappling|mat\\s+wrestling|takedown|pin|pinfall|near\\s+fall|reversal|escape|riding\\s+time|technical\\s+fall|mercy\\s+rule|sudden\\s+victory|overtime|wrestle\\s+off|recreational\\s+wrestling|competitive\\s+wrestling|club\\s+wrestling|high\\s+school\\s+wrestling|college\\s+wrestling|university\\s+wrestling|tournament\\s+wrestling|championship\\s+wrestling|world\\s+wrestling|olympic\\s+wrestling|uww\\s+wrestling|youth\\s+wrestling|cadet\\s+wrestling|junior\\s+wrestling|senior\\s+wrestling|veteran\\s+wrestling|masters\\s+wrestling|women\\s+wrestling|men\\s+wrestling|mixed\\s+wrestling|team\\s+wrestling|individual\\s+wrestling|dual\\s+meet|wrestling\\s+conditioning)\\b",

            "taekwondo" to "\\b(?:taekwondo|tae\\s+kwon\\s+do|taekwondo\\s+class|taekwondo\\s+drill|tkd|korean\\s+martial\\s+arts|way\\s+of\\s+foot\\s+and\\s+fist|poomsae|forms|kyorugi|breaking|board\\s+breaking|brick\\s+breaking|high\\s+kicks|jumping\\s+kicks|spinning\\s+kicks|wtf\\s+taekwondo|itf\\s+taekwondo|kukkiwon|chang\\s+hon|traditional\\s+taekwondo|sport\\s+taekwondo|olympic\\s+taekwondo|recreational\\s+taekwondo|competitive\\s+taekwondo|professional\\s+taekwondo|amateur\\s+taekwondo|club\\s+taekwondo|dojang\\s+practice|tournament\\s+taekwondo|championship\\s+taekwondo|world\\s+taekwondo|youth\\s+taekwondo|cadet\\s+taekwondo|junior\\s+taekwondo|senior\\s+taekwondo|veteran\\s+taekwondo|masters\\s+taekwondo|para\\s+taekwondo|adaptive\\s+taekwondo|women\\s+taekwondo|men\\s+taekwondo|mixed\\s+taekwondo|family\\s+taekwondo|kids\\s+taekwondo|little\\s+tigers|belt\\s+testing)\\b",

            "muay thai" to "\\b(?:muay\\s+thai|muay\\s+thai\\s+fighting|thai\\s+boxing|muay\\s+thai\\s+training|muay\\s+thai\\s+workout|muay\\s+thai\\s+practice|muay\\s+thai\\s+session|muay\\s+thai\\s+class|art\\s+of\\s+eight\\s+limbs|science\\s+of\\s+eight\\s+limbs|clinch\\s+work|knee\\s+strikes|elbow\\s+strikes|kicks|punches|thai\\s+pads|heavy\\s+bag\\s+muay\\s+thai|shadowboxing\\s+muay\\s+thai|wai\\s+kru|ram\\s+muay|mongkol|prajioud|shorts\\s+muay\\s+thai|traditional\\s+muay\\s+thai|modern\\s+muay\\s+thai|recreational\\s+muay\\s+thai|competitive\\s+muay\\s+thai|professional\\s+muay\\s+thai|amateur\\s+muay\\s+thai|club\\s+muay\\s+thai|gym\\s+muay\\s+thai|tournament\\s+muay\\s+thai|championship\\s+muay\\s+thai|world\\s+muay\\s+thai|youth\\s+muay\\s+thai|women\\s+muay\\s+thai|men\\s+muay\\s+thai|mixed\\s+muay\\s+thai|fitness\\s+muay\\s+thai|cardio\\s+muay\\s+thai|conditioning\\s+muay\\s+thai|self\\s+defense\\s+muay\\s+thai|street\\s+muay\\s+thai|mma\\s+muay\\s+thai)\\b",

            "martial arts" to "\\b(?:martial\\s+arts|martial\\s+arts\\s+practice|martial\\s+arts\\s+training|martial\\s+arts\\s+workout|martial\\s+arts\\s+session|martial\\s+arts\\s+class|self\\s+defense|self\\s+defence|combat\\s+arts|fighting\\s+arts|warrior\\s+arts|ancient\\s+combat|traditional\\s+fighting|modern\\s+combat|mixed\\s+martial\\s+arts|mma|combat\\s+sports|fighting\\s+sports|defensive\\s+arts|offensive\\s+arts|striking\\s+arts|grappling\\s+arts|throwing\\s+arts|joint\\s+manipulation|pressure\\s+points|chi\\s+kung|qi\\s+gong|internal\\s+arts|external\\s+arts|hard\\s+styles|soft\\s+styles|linear\\s+styles|circular\\s+styles|forms\\s+practice|kata\\s+practice|weapons\\s+training|empty\\s+hand|armed\\s+combat|unarmed\\s+combat|combat\\s+conditioning|martial\\s+conditioning|warrior\\s+training|dojo\\s+training|dojang\\s+training|kwoon\\s+training|academy\\s+training)\\b",

            "kendo" to "\\b(?:kendo|kendo\\s+practice|kendo\\s+class|kendo\\s+training|kendo\\s+workout|kendo\\s+session|kendo\\s+drill|japanese\\s+swordsmanship|way\\s+of\\s+the\\s+sword|shinai\\s+practice|bokken\\s+practice|kata\\s+kendo|jigeiko|keiko|shiai|kamae\\s+practice|men\\s+strike|kote\\s+strike|do\\s+strike|tsuki\\s+strike|iaido\\s+related|kendoka|sensei\\s+instruction|dojo\\s+kendo|bogu\\s+practice|armor\\s+kendo|mask\\s+kendo|glove\\s+kendo|chest\\s+protector|traditional\\s+kendo|modern\\s+kendo|sport\\s+kendo|recreational\\s+kendo|competitive\\s+kendo|tournament\\s+kendo|championship\\s+kendo|world\\s+kendo|youth\\s+kendo|adult\\s+kendo|senior\\s+kendo|women\\s+kendo|men\\s+kendo|mixed\\s+kendo|kendo\\s+federation|dan\\s+grading|kyu\\s+grading|black\\s+belt\\s+kendo|kendo\\s+philosophy|bushido\\s+kendo)\\b",

            "kickboxing" to "\\b(?:kickboxing|kick\\s+boxing|kickboxing\\s+class|kickboxing\\s+drill|american\\s+kickboxing|japanese\\s+kickboxing|dutch\\s+kickboxing|k1\\s+kickboxing|full\\s+contact\\s+kickboxing|semi\\s+contact\\s+kickboxing|light\\s+contact\\s+kickboxing|cardio\\s+kickboxing|fitness\\s+kickboxing|aerobic\\s+kickboxing|punching\\s+and\\s+kicking|kicks\\s+and\\s+punches|pad\\s+work\\s+kickboxing|heavy\\s+bag\\s+kickboxing|shadow\\s+kickboxing|recreational\\s+kickboxing|competitive\\s+kickboxing|professional\\s+kickboxing|amateur\\s+kickboxing|gym\\s+kickboxing|club\\s+kickboxing|tournament\\s+kickboxing|championship\\s+kickboxing|world\\s+kickboxing|youth\\s+kickboxing|women\\s+kickboxing|men\\s+kickboxing|mixed\\s+kickboxing|conditioning\\s+kickboxing|strength\\s+kickboxing|flexibility\\s+kickboxing|self\\s+defense\\s+kickboxing|street\\s+kickboxing|ring\\s+kickboxing)\\b",

            "kayaking" to "\\b(?:kayaking|kayak|kayak\\s+ride|kayaking\\s+trip|rafting|raft|white\\s+water\\s+rafting|whitewater\\s+rafting|river\\s+rafting|rapid\\s+rafting|paddle\\s+boat|paddling|canoe|canoeing|canoe\\s+trip|paddleboard|paddle\\s+boarding|stand\\s+up\\s+paddle|sup|water\\s+paddling|recreational\\s+paddling|touring\\s+kayak|sea\\s+kayaking|ocean\\s+kayaking|lake\\s+kayaking|river\\s+kayaking|creek\\s+kayaking|flatwater\\s+kayaking|whitewater\\s+kayaking|kayak\\s+fishing|fishing\\s+kayak|tandem\\s+kayak|single\\s+kayak|sit\\s+on\\s+top\\s+kayak|sit\\s+inside\\s+kayak|inflatable\\s+kayak|folding\\s+kayak|expedition\\s+kayaking|adventure\\s+kayaking|wilderness\\s+kayaking|backcountry\\s+kayaking|multi\\s+day\\s+kayaking|kayak\\s+camping|paddle\\s+sports|water\\s+sports|aquatic\\s+adventure)\\b",

            "water skiing" to "\\b(?:water\\s+skiing|water\\s+ski|water\\s+skiing\\s+run|water\\s+skiing\\s+session|water\\s+skiing\\s+practice|water\\s+skiing\\s+training|water\\s+skiing\\s+workout|slalom\\s+skiing|slalom\\s+water\\s+ski|trick\\s+skiing|trick\\s+water\\s+ski|jump\\s+skiing|jump\\s+water\\s+ski|barefoot\\s+skiing|barefoot\\s+water\\s+ski|combo\\s+skiing|mono\\s+skiing|single\\s+ski|double\\s+ski|pair\\s+skis|wakeboard\\s+skiing|ski\\s+boat|speed\\s+boat\\s+skiing|tow\\s+rope\\s+skiing|recreational\\s+water\\s+skiing|competitive\\s+water\\s+skiing|professional\\s+water\\s+skiing|tournament\\s+water\\s+skiing|lake\\s+skiing|river\\s+skiing|calm\\s+water\\s+skiing|choppy\\s+water\\s+skiing|beginner\\s+water\\s+skiing|intermediate\\s+water\\s+skiing|advanced\\s+water\\s+skiing|expert\\s+water\\s+skiing)\\b",

            "kite surfing" to "\\b(?:kite\\s+surfing|kite\\s+surf|kiteboarding|kite\\s+boarding|kitesurfing|kitesurf|kite\\s+surfing\\s+session|kite\\s+surfing\\s+practice|kite\\s+surfing\\s+training|kite\\s+surfing\\s+workout|power\\s+kite|traction\\s+kite|inflatable\\s+kite|foil\\s+kite|bow\\s+kite|c\\s+kite|delta\\s+kite|hybrid\\s+kite|kite\\s+harness|control\\s+bar|kite\\s+lines|twin\\s+tip\\s+board|directional\\s+board|surf\\s+style\\s+board|freestyle\\s+kitesurfing|freeride\\s+kitesurfing|wave\\s+riding\\s+kite|big\\s+air\\s+kite|speed\\s+kite|racing\\s+kite|course\\s+racing\\s+kite|upwind\\s+kiting|downwind\\s+kiting|cross\\s+wind\\s+kiting|ocean\\s+kiting|lake\\s+kiting|flat\\s+water\\s+kiting|choppy\\s+water\\s+kiting|wave\\s+kiting|onshore\\s+kiting|offshore\\s+kiting)\\b",

            "snorkeling" to "\\b(?:snorkeling|snorkel|snorkeling\\s+trip|snorkeling\\s+session|snorkeling\\s+practice|snorkeling\\s+training|snorkeling\\s+workout|snorkelling|skin\\s+diving|surface\\s+diving|breath\\s+hold\\s+diving|free\\s+diving\\s+snorkel|mask\\s+and\\s+snorkel|fins\\s+mask\\s+snorkel|snorkel\\s+gear|snorkel\\s+set|snorkel\\s+equipment|ocean\\s+snorkeling|reef\\s+snorkeling|coral\\s+reef\\s+snorkeling|tropical\\s+snorkeling|shallow\\s+water\\s+snorkeling|lagoon\\s+snorkeling|bay\\s+snorkeling|cove\\s+snorkeling|beach\\s+snorkeling|shore\\s+snorkeling|boat\\s+snorkeling|drift\\s+snorkeling|guided\\s+snorkeling|solo\\s+snorkeling|recreational\\s+snorkeling|marine\\s+life\\s+viewing|underwater\\s+exploration|fish\\s+watching|turtle\\s+watching|ray\\s+watching|shark\\s+watching)\\b",

            "diving" to "\\b(?:diving|scuba\\s+diving|diving\\s+trip|diving\\s+session|diving\\s+practice|diving\\s+training|diving\\s+workout|scuba|underwater\\s+diving|deep\\s+sea\\s+diving|wreck\\s+diving|cave\\s+diving|cavern\\s+diving|night\\s+diving|drift\\s+diving|wall\\s+diving|reef\\s+diving|shore\\s+diving|boat\\s+diving|drift\\s+diving|technical\\s+diving|recreational\\s+diving|commercial\\s+diving|scientific\\s+diving|military\\s+diving|rescue\\s+diving|public\\s+safety\\s+diving|underwater\\s+photography|underwater\\s+videography|marine\\s+biology\\s+diving|archaeology\\s+diving|open\\s+water\\s+diving|advanced\\s+open\\s+water|rescue\\s+diver|divemaster|dive\\s+instructor|padi|naui|ssi|trimix\\s+diving|nitrox\\s+diving|gas\\s+blending|decompression\\s+diving)\\b",

            "synchronized swimming" to "\\b(?:synchronized\\s+swimming|synchro|water\\s+ballet|artistic\\s+swimming|water\\s+dance|aquatic\\s+dance|pool\\s+ballet|water\\s+choreography|aquatic\\s+choreography|solo\\s+synchro|duet\\s+synchro|team\\s+synchro|group\\s+synchro|combo\\s+synchro|figures\\s+synchro|routine\\s+synchro|technical\\s+routine|free\\s+routine|highlight\\s+routine|acrobatic\\s+routine|creative\\s+swimming|expressive\\s+swimming|performance\\s+swimming|competitive\\s+synchronized\\s+swimming|recreational\\s+synchronized\\s+swimming|masters\\s+synchronized\\s+swimming|youth\\s+synchronized\\s+swimming|age\\s+group\\s+synchro|synchronized\\s+diving)\\b",

            "fin swimming" to "\\b(?:fin\\s+swimming|fin\\s+swim|monofin|monofin\\s+swimming|mono\\s+fin\\s+swimming|mermaid\\s+swimming|mermaid\\s+fin\\s+swimming|dolphin\\s+kick\\s+swimming|underwater\\s+fin\\s+swimming|surface\\s+fin\\s+swimming|bifin\\s+swimming|bi\\s+fin\\s+swimming|split\\s+fin\\s+swimming|long\\s+fin\\s+swimming|short\\s+fin\\s+swimming|competition\\s+fin\\s+swimming|recreational\\s+fin\\s+swimming|technique\\s+fin\\s+swimming|speed\\s+fin\\s+swimming|endurance\\s+fin\\s+swimming|sprint\\s+fin\\s+swimming|distance\\s+fin\\s+swimming|pool\\s+fin\\s+swimming|open\\s+water\\s+fin\\s+swimming|underwater\\s+swimming|apnea\\s+swimming|breath\\s+hold\\s+swimming|dynamic\\s+apnea|static\\s+apnea|finswimming|cmas\\s+finswimming)\\b",

            "water polo" to "\\b(?:water\\s+polo|water\\s+polo\\s+game|water\\s+polo\\s+practice|water\\s+polo\\s+training|water\\s+polo\\s+workout|water\\s+polo\\s+session|water\\s+polo\\s+match|polo\\s+aquatic|aquatic\\s+polo|pool\\s+polo|swimming\\s+pool\\s+polo|seven\\s+a\\s+side|six\\s+plus\\s+goalie|treading\\s+water\\s+polo|egg\\s+beater\\s+kick|water\\s+polo\\s+shooting|water\\s+polo\\s+passing|water\\s+polo\\s+defense|water\\s+polo\\s+offense|water\\s+polo\\s+goalie|water\\s+polo\\s+goalkeeper|water\\s+polo\\s+field\\s+player|water\\s+polo\\s+center|water\\s+polo\\s+driver|water\\s+polo\\s+wing|water\\s+polo\\s+point|recreational\\s+water\\s+polo|competitive\\s+water\\s+polo|professional\\s+water\\s+polo|amateur\\s+water\\s+polo|masters\\s+water\\s+polo|youth\\s+water\\s+polo|junior\\s+water\\s+polo|senior\\s+water\\s+polo|women\\s+water\\s+polo|men\\s+water\\s+polo|mixed\\s+water\\s+polo)\\b",

            "skiing" to "\\b(?:skiing|ski|ski\\s+run|skiing\\s+trip|skiing\\s+session|skiing\\s+practice|skiing\\s+training|skiing\\s+workout|downhill\\s+skiing|alpine\\s+skiing|slalom\\s+skiing|giant\\s+slalom|super\\s+g|super\\s+giant\\s+slalom|parallel\\s+skiing|mogul\\s+skiing|freestyle\\s+skiing|aerial\\s+skiing|halfpipe\\s+skiing|slopestyle\\s+skiing|ski\\s+cross|skier\\s+cross|powder\\s+skiing|off\\s+piste\\s+skiing|backcountry\\s+skiing|touring\\s+skiing|ski\\s+touring|telemark\\s+skiing|nordic\\s+skiing|recreational\\s+skiing|competitive\\s+skiing|racing\\s+skiing|speed\\s+skiing|carving\\s+skiing|parallel\\s+turn|snow\\s+plow|pizza\\s+skiing|wedge\\s+skiing|groomed\\s+run|fresh\\s+powder|corduroy\\s+snow|spring\\s+skiing|winter\\s+skiing|resort\\s+skiing|mountain\\s+skiing)\\b",

            "snowboarding" to "\\b(?:snowboarding|snowboard|snowboarding\\s+run|snowboarding\\s+session|snowboarding\\s+practice|snowboarding\\s+training|snowboarding\\s+workout|snowboarding\\s+trip|alpine\\s+snowboarding|freestyle\\s+snowboarding|freeride\\s+snowboarding|all\\s+mountain\\s+snowboarding|halfpipe\\s+snowboarding|terrain\\s+park\\s+snowboarding|slopestyle\\s+snowboarding|snowboard\\s+cross|boarder\\s+cross|parallel\\s+snowboarding|slalom\\s+snowboarding|giant\\s+slalom\\s+snowboarding|powder\\s+snowboarding|backcountry\\s+snowboarding|splitboard\\s+snowboarding|touring\\s+snowboarding|carving\\s+snowboarding|jibbing\\s+snowboarding|rail\\s+snowboarding|box\\s+snowboarding|jump\\s+snowboarding|pipe\\s+snowboarding|regular\\s+stance|goofy\\s+stance|switch\\s+snowboarding|fakie\\s+snowboarding|recreational\\s+snowboarding|competitive\\s+snowboarding|racing\\s+snowboarding|resort\\s+snowboarding|mountain\\s+snowboarding)\\b",

            "cross country skiing" to "\\b(?:cross\\s+country\\s+skiing|cross\\s+country\\s+ski|xc\\s+skiing|nordic\\s+skiing|langlauf|ski\\s+de\\s+fond|classic\\s+skiing|classical\\s+skiing|diagonal\\s+stride|kick\\s+and\\s+glide|skate\\s+skiing|freestyle\\s+skiing|v\\s+skate|offset\\s+skate|one\\s+skate|two\\s+skate|marathon\\s+skate|double\\s+pole|no\\s+pole\\s+skiing|touring\\s+skiing|backcountry\\s+cross\\s+country|groomed\\s+trail\\s+skiing|track\\s+skiing|ungroomed\\s+skiing|powder\\s+cross\\s+country|recreational\\s+cross\\s+country|competitive\\s+cross\\s+country|racing\\s+cross\\s+country|distance\\s+skiing|sprint\\s+skiing|pursuit\\s+skiing|relay\\s+skiing|biathlon\\s+skiing|orienteering\\s+skiing|citizen\\s+race|loppet|ski\\s+marathon)\\b",

            "alpine skiing" to "\\b(?:alpine\\s+skiing|alpine\\s+ski|downhill\\s+skiing|downhill\\s+racing|slalom\\s+racing|giant\\s+slalom\\s+racing|super\\s+g\\s+racing|super\\s+giant\\s+slalom\\s+racing|combined\\s+skiing|super\\s+combined|parallel\\s+slalom|parallel\\s+giant\\s+slalom|speed\\s+skiing|technical\\s+skiing|carving\\s+turns|short\\s+turns|long\\s+turns|edge\\s+control|fall\\s+line\\s+skiing|steep\\s+skiing|mogul\\s+skiing|bump\\s+skiing|groomed\\s+run\\s+skiing|piste\\s+skiing|on\\s+piste\\s+skiing|resort\\s+alpine\\s+skiing|recreational\\s+alpine\\s+skiing|competitive\\s+alpine\\s+skiing|racing\\s+alpine\\s+skiing|ski\\s+racing|gates\\s+skiing|course\\s+skiing|timed\\s+skiing|fis\\s+skiing|world\\s+cup\\s+skiing|olympic\\s+skiing)\\b",

            "curling" to "\\b(?:curling|curling\\s+game|curling\\s+practice|curling\\s+session|curling\\s+training|curling\\s+workout|curling\\s+match|ice\\s+curling|sheet\\s+curling|rink\\s+curling|stone\\s+curling|rock\\s+curling|granite\\s+curling|curling\\s+stone|curling\\s+rock|house\\s+curling|button\\s+curling|tee\\s+line|hack\\s+curling|delivery\\s+curling|slide\\s+curling|release\\s+curling|turn\\s+curling|curl\\s+curling|draw\\s+curling|guard\\s+curling|takeout\\s+curling|hit\\s+curling|peel\\s+curling|freeze\\s+curling|raise\\s+curling|double\\s+takeout|triple\\s+takeout|sweep\\s+curling|sweeping\\s+curling|brush\\s+curling|broom\\s+curling|skip\\s+curling|vice\\s+skip|second\\s+curling|lead\\s+curling|team\\s+curling|four\\s+person\\s+curling|mixed\\s+curling|wheelchair\\s+curling|recreational\\s+curling|competitive\\s+curling|league\\s+curling|bonspiel)\\b",

            "zumba" to "\\b(?:zumba|zumba\\s+class|zumba\\s+workout|zumba\\s+session|zumba\\s+training|zumba\\s+fitness|zumba\\s+dance|latin\\s+dance\\s+fitness|dance\\s+fitness|aerobic\\s+dance|cardio\\s+dance|zumba\\s+gold|zumba\\s+toning|zumba\\s+step|zumba\\s+aqua|aqua\\s+zumba|water\\s+zumba|zumba\\s+kids|zumba\\s+sentao|zumba\\s+strong|strong\\s+by\\s+zumba|zumba\\s+in\\s+the\\s+circuit|zumba\\s+circuit|latin\\s+cardio|salsa\\s+fitness|merengue\\s+fitness|reggaeton\\s+fitness|cumbia\\s+fitness|bachata\\s+fitness|cha\\s+cha\\s+fitness|samba\\s+fitness|party\\s+workout|dance\\s+party|fitness\\s+party|latin\\s+party|rhythm\\s+workout|beat\\s+workout|music\\s+fitness|fun\\s+fitness|group\\s+dance\\s+fitness)\\b",

            "ballet" to "\\b(?:ballet|ballet\\s+class|ballet\\s+session|classical\\s+ballet|romantic\\s+ballet|neoclassical\\s+ballet|contemporary\\s+ballet|modern\\s+ballet|russian\\s+ballet|french\\s+ballet|italian\\s+ballet|english\\s+ballet|american\\s+ballet|vaganova\\s+ballet|cecchetti\\s+ballet|royal\\s+ballet|bolshoi\\s+ballet|mariinsky\\s+ballet|barre\\s+work|center\\s+work|adagio\\s+ballet|allegro\\s+ballet|grand\\s+allegro|petit\\s+allegro|port\\s+de\\s+bras|arabesque|attitude|developpe|grand\\s+battement|tendu|degage|releve|plie|saute|echappe|chasse|bourree|pirouette|fouette|grand\\s+jete|sissonne|assemble|entrechat|brise|cabriole|tour\\s+jete|pointe\\s+work|demi\\s+pointe|soft\\s+ballet|adult\\s+ballet|beginner\\s+ballet|intermediate\\s+ballet|advanced\\s+ballet|pre\\s+ballet|baby\\s+ballet|toddler\\s+ballet|youth\\s+ballet|teen\\s+ballet|senior\\s+ballet)\\b",

            "belly dance" to "\\b(?:belly\\s+dance|belly\\s+dancing|oriental\\s+dance|middle\\s+eastern\\s+dance|arabic\\s+dance|egyptian\\s+dance|turkish\\s+dance|lebanese\\s+dance|raqs\\s+sharqi|raks\\s+sharki|cabaret\\s+belly\\s+dance|tribal\\s+belly\\s+dance|fusion\\s+belly\\s+dance|american\\s+tribal\\s+style|ats|improvisational\\s+tribal\\s+style|its|gothic\\s+belly\\s+dance|steampunk\\s+belly\\s+dance|vintage\\s+belly\\s+dance|classic\\s+belly\\s+dance|modern\\s+belly\\s+dance|contemporary\\s+belly\\s+dance|folkloric\\s+belly\\s+dance|saidi|baladi|shaabi|drum\\s+solo|zills|finger\\s+cymbals|veil\\s+dance|fan\\s+dance|sword\\s+dance|cane\\s+dance|isis\\s+wings|belly\\s+dance\\s+fitness|shimmy|undulation|figure\\s+eight|hip\\s+drop|hip\\s+lift|chest\\s+pop|shoulder\\s+shimmy)\\b",

            "street dance" to "\\b(?:street\\s+dance|street\\s+dancing|hip\\s+hop|hip\\s+hop\\s+dance|hip\\s+hop\\s+dancing|breaking|breakdancing|break\\s+dance|b\\s+boying|b\\s+girling|popping|locking|krumping|waacking|voguing|house\\s+dance|new\\s+jack\\s+swing|old\\s+school\\s+hip\\s+hop|new\\s+school\\s+hip\\s+hop|freestyle\\s+dance|battle\\s+dance|cypher\\s+dance|underground\\s+dance|urban\\s+dance|commercial\\s+dance|video\\s+dance|music\\s+video\\s+dance|choreography\\s+hip\\s+hop|lyrical\\s+hip\\s+hop|contemporary\\s+hip\\s+hop|jazz\\s+funk|funk\\s+dance|electric\\s+boogaloo|robot\\s+dance|animation\\s+dance|tutting|finger\\s+tutting|liquid\\s+dance|waving|gliding|moonwalk|windmill|headspin|freeze|toprock|downrock|power\\s+moves|flow\\s+moves|thread|flare|jackhammer|baby\\s+freeze|chair\\s+freeze)\\b",

            "ballroom dancing" to "\\b(?:ballroom\\s+dancing|ballroom|ballroom\\s+dance|partner\\s+dance|social\\s+dance|competitive\\s+ballroom|standard\\s+ballroom|latin\\s+ballroom|waltz|foxtrot|tango|viennese\\s+waltz|quickstep|cha\\s+cha|cha\\s+cha\\s+cha|rumba|samba|paso\\s+doble|jive|swing\\s+dance|lindy\\s+hop|east\\s+coast\\s+swing|west\\s+coast\\s+swing|charleston|balboa|shag|blues\\s+dance|salsa|bachata|merengue|kizomba|zouk|argentine\\s+tango|american\\s+smooth|american\\s+rhythm|international\\s+standard|international\\s+latin|smooth\\s+ballroom|rhythm\\s+ballroom|dancesport|competitive\\s+dance|medal\\s+ballroom|syllabus\\s+ballroom|open\\s+ballroom|showcase\\s+dance|pro\\s+am|formation\\s+ballroom)\\b",

            "square dance" to "\\b(?:square\\s+dance|square\\s+dancing|traditional\\s+square\\s+dance|modern\\s+square\\s+dance|western\\s+square\\s+dance|mainstream\\s+square\\s+dance|plus\\s+square\\s+dance|advanced\\s+square\\s+dance|challenge\\s+square\\s+dance|caller\\s+square\\s+dance|cued\\s+square\\s+dance|live\\s+music\\s+square\\s+dance|recorded\\s+music\\s+square\\s+dance|do\\s+si\\s+do|promenade|allemande|swing\\s+your\\s+partner|circle\\s+left|circle\\s+right|right\\s+and\\s+left\\s+grand|grand\\s+right\\s+and\\s+left|star\\s+promenade|wheel\\s+around|pass\\s+thru|bend\\s+the\\s+line|ladies\\s+chain|rollaway|half\\s+sashay|see\\s+saw|contra\\s+dance|line\\s+dance|folk\\s+dance|barn\\s+dance|hoedown|jamboree|square\\s+dance\\s+club|square\\s+dance\\s+society)\\b",

            "jazz dance" to "\\b(?:jazz\\s+dance|jazz\\s+dancing|traditional\\s+jazz\\s+dance|modern\\s+jazz\\s+dance|contemporary\\s+jazz|lyrical\\s+jazz|theatrical\\s+jazz|broadway\\s+jazz|musical\\s+theatre\\s+jazz|commercial\\s+jazz|street\\s+jazz|funk\\s+jazz|latin\\s+jazz|classical\\s+jazz|swing\\s+jazz|bebop\\s+jazz|smooth\\s+jazz\\s+dance|ragtime\\s+dance|charleston\\s+jazz|lindy\\s+hop\\s+jazz|fosse\\s+style|mattox\\s+style|luigi\\s+style|horton\\s+technique|dunham\\s+technique|jack\\s+cole\\s+style|bob\\s+fosse\\s+style|gus\\s+giordano\\s+style|frank\\s+hatchett\\s+style|jazz\\s+walks|jazz\\s+runs|jazz\\s+squares|chaines|pirouettes\\s+jazz|tour\\s+jetes\\s+jazz|jazz\\s+hands|jazz\\s+split|jazz\\s+leap|stag\\s+leap|switch\\s+leap|calypso\\s+leap)\\b",

            "latin dance" to "\\b(?:latin\\s+dance|latin\\s+dancing|salsa|tango|rumba|salsa\\s+dance|salsa\\s+dancing|bachata|merengue|cha\\s+cha\\s+cha|mambo|samba|bossa\\s+nova|cumbia|reggaeton\\s+dance|kizomba|zouk|lambada|forro|milonga|argentine\\s+tango|ballroom\\s+tango|cuban\\s+salsa|new\\s+york\\s+salsa|la\\s+style\\s+salsa|casino\\s+salsa|rueda\\s+de\\s+casino|dominican\\s+bachata|sensual\\s+bachata|traditional\\s+bachata|modern\\s+bachata|merengue\\s+tipico|merengue\\s+de\\s+calle|brazilian\\s+samba|ballroom\\s+samba|afro\\s+brazilian|capoeira\\s+dance|latin\\s+ballroom|social\\s+latin|competitive\\s+latin|partner\\s+latin|solo\\s+latin|group\\s+latin|latin\\s+fitness|zumba\\s+latin)\\b",

            "national dance" to "\\b(?:national\\s+dance|folk\\s+dance|traditional\\s+dance|cultural\\s+dance|ethnic\\s+dance|heritage\\s+dance|indigenous\\s+dance|regional\\s+dance|country\\s+dance|scottish\\s+dance|irish\\s+dance|russian\\s+dance|polish\\s+dance|german\\s+dance|italian\\s+dance|spanish\\s+dance|greek\\s+dance|turkish\\s+dance|indian\\s+dance|chinese\\s+dance|japanese\\s+dance|korean\\s+dance|african\\s+dance|mexican\\s+dance|brazilian\\s+dance|peruvian\\s+dance|argentine\\s+dance|flamenco|morris\\s+dance|clogging|step\\s+dance|sword\\s+dance|maypole\\s+dance|harvest\\s+dance|wedding\\s+dance|ceremonial\\s+dance|ritual\\s+dance|celebration\\s+dance|festival\\s+dance)\\b",

            "dance" to "\\b(?:dance|dancing|dance\\s+class|dance\\s+session|dance\\s+workout|dance\\s+practice|social\\s+dance|recreational\\s+dance|competitive\\s+dance|performance\\s+dance|creative\\s+dance|expressive\\s+dance|interpretive\\s+dance|modern\\s+dance|contemporary\\s+dance|postmodern\\s+dance|avant\\s+garde\\s+dance|experimental\\s+dance|improvisational\\s+dance|contact\\s+improvisation|movement\\s+improvisation|dance\\s+improvisation|choreography|choreographic|dance\\s+composition|dance\\s+creation|partner\\s+dance|solo\\s+dance|group\\s+dance|ensemble\\s+dance|line\\s+dance|circle\\s+dance|formation\\s+dance|sequence\\s+dance|freestyle\\s+dance|structured\\s+dance|technique\\s+class|movement\\s+class|body\\s+movement|rhythmic\\s+movement|musical\\s+movement|artistic\\s+movement|dance\\s+therapy|movement\\s+therapy|dance\\s+fitness|dance\\s+cardio|dance\\s+aerobics)\\b",

            "hunting" to "\\b(?:hunting|hunt|game\\s+hunting|hunting\\s+session|hunting\\s+practice|hunting\\s+training|hunting\\s+workout|hunting\\s+trip|hunting\\s+expedition|big\\s+game\\s+hunting|small\\s+game\\s+hunting|waterfowl\\s+hunting|upland\\s+hunting|deer\\s+hunting|elk\\s+hunting|moose\\s+hunting|bear\\s+hunting|turkey\\s+hunting|duck\\s+hunting|goose\\s+hunting|rabbit\\s+hunting|squirrel\\s+hunting|bird\\s+hunting|predator\\s+hunting|varmint\\s+hunting|bow\\s+hunting|archery\\s+hunting|rifle\\s+hunting|shotgun\\s+hunting|muzzleloader\\s+hunting|black\\s+powder\\s+hunting|still\\s+hunting|spot\\s+and\\s+stalk|driven\\s+hunt|stand\\s+hunting|blind\\s+hunting|calling\\s+hunting|decoy\\s+hunting|tracking\\s+hunting|stalking\\s+hunting|safari\\s+hunting|guided\\s+hunting|solo\\s+hunting|group\\s+hunting|recreational\\s+hunting|subsistence\\s+hunting|conservation\\s+hunting)\\b",

            "fishing" to "\\b(?:fishing|fish|angling|fishing\\s+session|fishing\\s+practice|fishing\\s+training|fishing\\s+workout|fishing\\s+trip|fishing\\s+expedition|freshwater\\s+fishing|saltwater\\s+fishing|deep\\s+sea\\s+fishing|offshore\\s+fishing|inshore\\s+fishing|surf\\s+fishing|pier\\s+fishing|dock\\s+fishing|bank\\s+fishing|wade\\s+fishing|boat\\s+fishing|kayak\\s+fishing|float\\s+tube\\s+fishing|fly\\s+fishing|spin\\s+fishing|bait\\s+fishing|lure\\s+fishing|trolling|casting|jigging|bottom\\s+fishing|top\\s+water\\s+fishing|ice\\s+fishing|winter\\s+fishing|stream\\s+fishing|river\\s+fishing|lake\\s+fishing|pond\\s+fishing|creek\\s+fishing|ocean\\s+fishing|bay\\s+fishing|estuary\\s+fishing|charter\\s+fishing|guided\\s+fishing|solo\\s+fishing|group\\s+fishing|recreational\\s+fishing|sport\\s+fishing|commercial\\s+fishing|catch\\s+and\\s+release|keep\\s+fishing|tournament\\s+fishing|competitive\\s+fishing)\\b",

            "equestrian" to "\\b(?:equestrian|horse\\s+riding|horseback\\s+riding|equestrian\\s+session|equestrian\\s+practice|equestrian\\s+training|equestrian\\s+workout|horseback\\s+riding\\s+session|horse\\s+riding\\s+lesson|riding\\s+lesson|dressage|show\\s+jumping|cross\\s+country|eventing|three\\s+day\\s+eventing|hunter\\s+jumper|western\\s+riding|english\\s+riding|trail\\s+riding|pleasure\\s+riding|hack\\s+riding|endurance\\s+riding|competitive\\s+trail|mounted\\s+games|polo|polocrosse|vaulting|driving|carriage\\s+driving|combined\\s+driving|reining|cutting|barrel\\s+racing|pole\\s+bending|gymkhana|rodeo\\s+events|ranch\\s+riding|working\\s+equitation|natural\\s+horsemanship|classical\\s+riding|haute\\s+ecole|liberty\\s+work|ground\\s+work|long\\s+lining|lunging|therapeutic\\s+riding|hippotherapy|adaptive\\s+riding|para\\s+equestrian)\\b",

            "sailing" to "\\b(?:sailing|sail|sailing\\s+trip|boat\\s+sailing|boating|sailing\\s+session|sailing\\s+practice|sailing\\s+training|sailing\\s+workout|yacht\\s+sailing|dinghy\\s+sailing|keelboat\\s+sailing|catamaran\\s+sailing|trimaran\\s+sailing|monohull\\s+sailing|multihull\\s+sailing|small\\s+boat\\s+sailing|big\\s+boat\\s+sailing|day\\s+sailing|overnight\\s+sailing|coastal\\s+sailing|offshore\\s+sailing|ocean\\s+sailing|lake\\s+sailing|river\\s+sailing|bay\\s+sailing|harbor\\s+sailing|recreational\\s+sailing|competitive\\s+sailing|racing\\s+sailing|regatta\\s+sailing|cruising\\s+sailing|passage\\s+making|blue\\s+water\\s+sailing|single\\s+handed\\s+sailing|crew\\s+sailing|learn\\s+to\\s+sail|sailing\\s+lessons|sailing\\s+instruction|wind\\s+sailing|light\\s+wind\\s+sailing|heavy\\s+weather\\s+sailing|upwind\\s+sailing|downwind\\s+sailing|reaching|tacking|jibing|spinnaker\\s+sailing)\\b",

            "motorboat" to "\\b(?:motorboat|speedboat|powerboat|motorboat\\s+session|motorboat\\s+practice|motorboat\\s+training|motorboat\\s+workout|motorboat\\s+trip|powerboat\\s+session|speedboat\\s+ride|motor\\s+boating|power\\s+boating|speed\\s+boating|runabout|bowrider|cuddy\\s+cabin|express\\s+cruiser|sport\\s+boat|ski\\s+boat|wakeboard\\s+boat|fishing\\s+boat|bass\\s+boat|center\\s+console|pontoon\\s+boat|deck\\s+boat|jet\\s+boat|inflatable\\s+boat|rib|rigid\\s+inflatable|go\\s+fast\\s+boat|cigarette\\s+boat|offshore\\s+powerboat|high\\s+performance\\s+boat|racing\\s+boat|pleasure\\s+boat|recreational\\s+powerboat|family\\s+boat|day\\s+boat|cabin\\s+cruiser|motor\\s+yacht|sport\\s+fishing\\s+boat|water\\s+sports\\s+boat|towing\\s+boat|wakesurfing\\s+boat|tubing\\s+boat)\\b",

            "atv" to "\\b(?:atv|all\\s+terrain\\s+vehicle|quad|quad\\s+bike|atv\\s+session|atv\\s+practice|atv\\s+training|atv\\s+workout|atv\\s+ride|atv\\s+riding|quad\\s+riding|four\\s+wheeler|four\\s+wheel\\s+drive\\s+atv|2wd\\s+atv|4wd\\s+atv|sport\\s+atv|utility\\s+atv|side\\s+by\\s+side|utv|utility\\s+task\\s+vehicle|rzr|can\\s+am|polaris|yamaha\\s+atv|honda\\s+atv|kawasaki\\s+atv|suzuki\\s+atv|trail\\s+riding\\s+atv|mud\\s+riding|sand\\s+riding|desert\\s+riding|mountain\\s+atv\\s+riding|forest\\s+atv\\s+riding|off\\s+road\\s+vehicle|orv|recreational\\s+off\\s+highway\\s+vehicle|ohv|dirt\\s+bike\\s+atv|motocross\\s+atv|enduro\\s+atv|cross\\s+country\\s+atv|racing\\s+atv|recreational\\s+atv|work\\s+atv|farm\\s+atv|ranch\\s+atv)\\b",

            "paraglider" to "\\b(?:paraglider|paragliding|paraglide|paragliding\\s+session|paragliding\\s+practice|paragliding\\s+training|paragliding\\s+workout|paragliding\\s+flight|paraglider\\s+flight|free\\s+flight|soaring|thermal\\s+flying|ridge\\s+soaring|cross\\s+country\\s+paragliding|xc\\s+paragliding|competition\\s+paragliding|acro\\s+paragliding|aerobatic\\s+paragliding|speed\\s+flying|mini\\s+wing|paramotor|powered\\s+paragliding|ppg|tandem\\s+paragliding|solo\\s+paragliding|student\\s+paragliding|instructor\\s+paragliding|thermal\\s+hunting|lift\\s+finding|ground\\s+handling|kiting|forward\\s+launch|reverse\\s+launch|alpine\\s+launch|winch\\s+launch|tow\\s+launch|hill\\s+training|mountain\\s+flying|coastal\\s+flying|desert\\s+flying|recreational\\s+paragliding|adventure\\s+paragliding|touring\\s+paragliding|hike\\s+and\\s+fly)\\b",

            "rock climbing" to "\\b(?:rock\\s+climbing|climbing|rock\\s+climb|bouldering|rock\\s+climbing\\s+session|rock\\s+climbing\\s+practice|rock\\s+climbing\\s+training|rock\\s+climbing\\s+workout|sport\\s+climbing|trad\\s+climbing|traditional\\s+climbing|top\\s+rope\\s+climbing|lead\\s+climbing|multi\\s+pitch\\s+climbing|single\\s+pitch\\s+climbing|free\\s+climbing|aid\\s+climbing|big\\s+wall\\s+climbing|crack\\s+climbing|face\\s+climbing|slab\\s+climbing|overhang\\s+climbing|roof\\s+climbing|indoor\\s+climbing|outdoor\\s+climbing|gym\\s+climbing|wall\\s+climbing|artificial\\s+climbing|crag\\s+climbing|cliff\\s+climbing|mountain\\s+climbing|alpine\\s+climbing|ice\\s+climbing|mixed\\s+climbing|dry\\s+tooling|bouldering\\s+session|boulder\\s+problems|highball\\s+bouldering|competition\\s+climbing|recreational\\s+climbing|adventure\\s+climbing|technical\\s+climbing|scrambling|via\\s+ferrata)\\b",

            "parkour" to "\\b(?:parkour|parkour\\s+training|free\\s+running|parkour\\s+session|parkour\\s+practice|parkour\\s+workout|parkour\\s+conditioning|freerunning|free\\s+run|pk|traceur|traceuse|art\\s+of\\s+movement|movement\\s+discipline|urban\\s+movement|natural\\s+movement|efficient\\s+movement|obstacle\\s+navigation|environmental\\s+training|functional\\s+movement\\s+parkour|precision\\s+jumping|safety\\s+vault|lazy\\s+vault|speed\\s+vault|dash\\s+vault|turn\\s+vault|reverse\\s+vault|wall\\s+run|wall\\s+climb|cat\\s+balance|rail\\s+balance|precision\\s+jump|standing\\s+jump|running\\s+jump|stride\\s+to\\s+stride|tic\\s+tac|wall\\s+spin|under\\s+bar|flow\\s+parkour|power\\s+parkour|creative\\s+parkour|military\\s+parkour|tactical\\s+parkour|training\\s+parkour|conditioning\\s+parkour|jam\\s+parkour|recreational\\s+parkour|competitive\\s+parkour)\\b",

            "frisbee" to "\\b(?:frisbee|disc|frisbee\\s+game|frisbee\\s+session|frisbee\\s+practice|frisbee\\s+workout|disc\\s+game|disc\\s+sport|flying\\s+disc|ultimate\\s+frisbee|ultimate\\s+disc|disc\\s+golf|frisbee\\s+golf|disc\\s+golf\\s+course|freestyle\\s+frisbee|freestyle\\s+disc|double\\s+disc\\s+court|ddc|guts\\s+frisbee|accuracy\\s+frisbee|distance\\s+frisbee|mta|maximum\\s+time\\s+aloft|discathlon|overall\\s+frisbee|beach\\s+ultimate|grass\\s+ultimate|indoor\\s+ultimate|pickup\\s+ultimate|league\\s+ultimate|tournament\\s+ultimate|recreational\\s+frisbee|competitive\\s+frisbee|throwing\\s+disc|catching\\s+disc|backhand\\s+throw|forehand\\s+throw|hammer\\s+throw|scoober\\s+throw|thumber\\s+throw|disc\\s+sports|flying\\s+saucer)\\b",

            "kite flying" to "\\b(?:kite\\s+flying|kite|kite\\s+fly|kite\\s+flying\\s+session|kite\\s+flying\\s+practice|kite\\s+flying\\s+training|kite\\s+flying\\s+workout|single\\s+line\\s+kite|dual\\s+line\\s+kite|quad\\s+line\\s+kite|stunt\\s+kite|sport\\s+kite|power\\s+kite|delta\\s+kite|box\\s+kite|diamond\\s+kite|cellular\\s+kite|traditional\\s+kite|modern\\s+kite|fighter\\s+kite|artistic\\s+kite|show\\s+kite|display\\s+kite|festival\\s+kite|competition\\s+kite|precision\\s+kite|freestyle\\s+kite|team\\s+flying|synchronized\\s+flying|kite\\s+ballet|wind\\s+surfing\\s+kite|aerial\\s+photography\\s+kite|kap|meteorological\\s+kite|scientific\\s+kite|recreational\\s+kite\\s+flying|competitive\\s+kite\\s+flying|beach\\s+kite\\s+flying|field\\s+kite\\s+flying|park\\s+kite\\s+flying|hill\\s+kite\\s+flying|indoor\\s+kite\\s+flying|static\\s+display)\\b",

            "tug of war" to "\\b(?:tug\\s+of\\s+war|tug\\s+war|rope\\s+pull|tug\\s+of\\s+war\\s+session|tug\\s+of\\s+war\\s+practice|tug\\s+of\\s+war\\s+training|tug\\s+of\\s+war\\s+workout|tug\\s+of\\s+war\\s+competition|rope\\s+pulling|team\\s+pulling|strength\\s+pulling|power\\s+pulling|indoor\\s+tug\\s+of\\s+war|outdoor\\s+tug\\s+of\\s+war|grass\\s+tug\\s+of\\s+war|mat\\s+tug\\s+of\\s+war|international\\s+tug\\s+of\\s+war|competitive\\s+tug\\s+of\\s+war|recreational\\s+tug\\s+of\\s+war|youth\\s+tug\\s+of\\s+war|adult\\s+tug\\s+of\\s+war|mixed\\s+tug\\s+of\\s+war|men\\s+tug\\s+of\\s+war|women\\s+tug\\s+of\\s+war|lightweight\\s+tug\\s+of\\s+war|heavyweight\\s+tug\\s+of\\s+war|open\\s+weight\\s+tug\\s+of\\s+war|club\\s+tug\\s+of\\s+war|school\\s+tug\\s+of\\s+war|corporate\\s+tug\\s+of\\s+war|family\\s+tug\\s+of\\s+war|beach\\s+tug\\s+of\\s+war|mud\\s+tug\\s+of\\s+war)\\b",

            "hula hoop" to "\\b(?:hula\\s+hoop|hula\\s+hooping|hula\\s+hoop\\s+session|hula\\s+hoop\\s+practice|hula\\s+hoop\\s+training|hula\\s+hoop\\s+workout|hoop\\s+dance|hooping|flow\\s+hooping|led\\s+hoop|fire\\s+hoop|weighted\\s+hoop|fitness\\s+hoop|dance\\s+hoop|performance\\s+hoop|trick\\s+hoop|beginner\\s+hoop|intermediate\\s+hoop|advanced\\s+hoop|off\\s+body\\s+hooping|on\\s+body\\s+hooping|waist\\s+hooping|chest\\s+hooping|shoulder\\s+hooping|hand\\s+hooping|leg\\s+hooping|foot\\s+hooping|multiple\\s+hoop|multi\\s+hoop|twin\\s+hoop|double\\s+hoop|mini\\s+hoop|large\\s+hoop|small\\s+hoop|heavy\\s+hoop|light\\s+hoop|traditional\\s+hoop|modern\\s+hoop|artistic\\s+hooping|creative\\s+hooping|meditative\\s+hooping|therapeutic\\s+hooping|cardio\\s+hooping|strength\\s+hooping|flexibility\\s+hooping)\\b",

            "track and field" to "\\b(?:track\\s+and\\s+field|track|field\\s+events|athletics|track\\s+and\\s+field\\s+session|track\\s+and\\s+field\\s+practice|track\\s+and\\s+field\\s+training|track\\s+and\\s+field\\s+workout|running\\s+events|jumping\\s+events|throwing\\s+events|sprint\\s+events|distance\\s+events|middle\\s+distance|long\\s+distance|hurdles|steeplechase|relay\\s+races|100\\s+meters|200\\s+meters|400\\s+meters|800\\s+meters|1500\\s+meters|5000\\s+meters|10000\\s+meters|marathon\\s+track|half\\s+marathon\\s+track|110\\s+meter\\s+hurdles|400\\s+meter\\s+hurdles|3000\\s+meter\\s+steeplechase|4x100\\s+relay|4x400\\s+relay|high\\s+jump|long\\s+jump|triple\\s+jump|pole\\s+vault|shot\\s+put|discus\\s+throw|hammer\\s+throw|javelin\\s+throw|decathlon|heptathlon|pentathlon|combined\\s+events|youth\\s+athletics|masters\\s+athletics|para\\s+athletics|paralympic\\s+athletics|olympic\\s+athletics|collegiate\\s+track|high\\s+school\\s+track|club\\s+track|professional\\s+track)\\b",

            "racing car" to "\\b(?:racing\\s+car|car\\s+racing|auto\\s+racing|racing\\s+car\\s+session|racing\\s+car\\s+practice|racing\\s+car\\s+training|racing\\s+car\\s+workout|motor\\s+racing|motorsport|formula\\s+racing|open\\s+wheel\\s+racing|stock\\s+car\\s+racing|touring\\s+car\\s+racing|sports\\s+car\\s+racing|gt\\s+racing|endurance\\s+racing|sprint\\s+racing|oval\\s+racing|road\\s+racing|circuit\\s+racing|drag\\s+racing|hill\\s+climb\\s+racing|autocross|rallycross|rally\\s+racing|karting|go\\s+kart\\s+racing|formula\\s+one|f1|indy\\s+car|nascar|btcc|wtcc|le\\s+mans|daytona|indianapolis\\s+500|monaco\\s+grand\\s+prix|silverstone|spa|nurburgring|laguna\\s+seca|sebring|petit\\s+le\\s+mans|professional\\s+racing|amateur\\s+racing|club\\s+racing|grassroots\\s+racing|sim\\s+racing|virtual\\s+racing)\\b",

            "triathlon" to "\\b(?:triathlon|triathlon\\s+training|ironman|triathlon\\s+session|triathlon\\s+practice|triathlon\\s+workout|tri\\s+training|swim\\s+bike\\s+run|multisport|three\\s+sport|endurance\\s+racing|sprint\\s+triathlon|olympic\\s+triathlon|half\\s+ironman|full\\s+ironman|ironman\\s+70\\.3|ironman\\s+140\\.6|super\\s+sprint\\s+triathlon|long\\s+course\\s+triathlon|ultra\\s+triathlon|aquathlon|duathlon|run\\s+bike\\s+run|aquabike|swim\\s+bike|brick\\s+workout|transition\\s+practice|t1\\s+practice|t2\\s+practice|open\\s+water\\s+swim\\s+tri|pool\\s+swim\\s+tri|road\\s+bike\\s+tri|time\\s+trial\\s+bike|tri\\s+bike|aero\\s+bike|wetsuit\\s+swimming|non\\s+wetsuit|age\\s+group\\s+triathlon|elite\\s+triathlon|professional\\s+triathlon|recreational\\s+triathlon|beginner\\s+triathlon|youth\\s+triathlon|junior\\s+triathlon|masters\\s+triathlon|para\\s+triathlon|team\\s+triathlon|relay\\s+triathlon)\\b",

            "cross training crossfit" to "\\b(?:crossfit|cross\\s+fit|crossfit\\s+workout|cross\\s+training)\\b",

            "free exercise" to "\\b(?:free\\s+exercise|free\\s+workout|unspecified\\s+exercise)\\b",

            "kabaddi" to "\\b(?:kabaddi|kabaddi\\s+game|kabaddi\\s+match|kabaddi\\s+sport|kabaddi\\s+practice|kabaddi\\s+session|kabaddi\\s+workout|kabaddi\\s+competition|kabaddi\\s+tournament|kabaddi\\s+league|traditional\\s+kabaddi|circle\\s+kabaddi|standard\\s+kabaddi|beach\\s+kabaddi|indoor\\s+kabaddi|outdoor\\s+kabaddi|pro\\s+kabaddi|team\\s+kabaddi|contact\\s+sport|raid\\s+sport|wrestling\\s+sport|grappling\\s+sport|tackle\\s+sport|defend\\s+sport|chase\\s+game|tag\\s+sport|indian\\s+sport|south\\s+asian\\s+sport|traditional\\s+game|folk\\s+sport|rural\\s+sport|village\\s+game|cultural\\s+sport|heritage\\s+sport|ancient\\s+sport|native\\s+sport|ethnic\\s+sport|regional\\s+sport|local\\s+sport|community\\s+sport)\\b",

            "table football" to "\\b(?:table\\s+football|foosball|table\\s+soccer|table\\s+football\\s+game|foosball\\s+game|table\\s+soccer\\s+game|table\\s+football\\s+match|foosball\\s+match|table\\s+soccer\\s+match|table\\s+football\\s+practice|foosball\\s+practice|table\\s+soccer\\s+practice|table\\s+football\\s+training|foosball\\s+training|table\\s+soccer\\s+training|table\\s+football\\s+session|foosball\\s+session|table\\s+soccer\\s+session|table\\s+football\\s+workout|foosball\\s+workout|table\\s+soccer\\s+workout|baby\\s+foot|kicker|fusball|calciobalilla|football\\s+table|soccer\\s+table|bar\\s+football|pub\\s+football|arcade\\s+football|miniature\\s+football|indoor\\s+football\\s+table|recreational\\s+football|casual\\s+football|competitive\\s+foosball|tournament\\s+foosball|professional\\s+foosball|tabletop\\s+football|miniature\\s+soccer|small\\s+football|desktop\\s+football)\\b",

            "seven stones" to "\\b(?:seven\\s+stones|seven\\s+stones\\s+game|seven\\s+stones\\s+sport|seven\\s+stones\\s+match|seven\\s+stones\\s+practice|seven\\s+stones\\s+training|seven\\s+stones\\s+session|seven\\s+stones\\s+workout|seven\\s+stones\\s+competition|seven\\s+stones\\s+tournament|lagori|pittu|satodiyu|ezhu\\s+kallu|sat\\s+pathar|stone\\s+pile\\s+game|stone\\s+stack\\s+game|stone\\s+tower\\s+game|flat\\s+stone\\s+game|piled\\s+stone\\s+game|stacked\\s+stone\\s+game|traditional\\s+stone\\s+game|indian\\s+stone\\s+game|south\\s+asian\\s+stone\\s+game|village\\s+stone\\s+game|rural\\s+stone\\s+game|outdoor\\s+stone\\s+game|folk\\s+stone\\s+game|cultural\\s+stone\\s+game|heritage\\s+stone\\s+game|ancient\\s+stone\\s+game|traditional\\s+game|native\\s+game|ethnic\\s+game|regional\\s+game|local\\s+game|community\\s+game|childhood\\s+game|playground\\s+game|street\\s+game)\\b",

            "kho kho" to "\\b(?:kho\\s+kho|kho\\s+kho\\s+game|kho\\s+kho\\s+sport|kho\\s+kho\\s+match|kho\\s+kho\\s+practice|kho\\s+kho\\s+training|kho\\s+kho\\s+session|kho\\s+kho\\s+workout|kho\\s+kho\\s+competition|kho\\s+kho\\s+tournament|kho\\s+kho\\s+league|traditional\\s+kho\\s+kho|competitive\\s+kho\\s+kho|professional\\s+kho\\s+kho|amateur\\s+kho\\s+kho|school\\s+kho\\s+kho|college\\s+kho\\s+kho|university\\s+kho\\s+kho|national\\s+kho\\s+kho|international\\s+kho\\s+kho|tag\\s+sport|chase\\s+sport|pursuit\\s+sport|running\\s+sport|speed\\s+sport|agility\\s+sport|team\\s+sport|contact\\s+sport|indian\\s+sport|south\\s+asian\\s+sport|traditional\\s+game|folk\\s+sport|cultural\\s+sport|heritage\\s+sport|ancient\\s+sport|native\\s+sport|ethnic\\s+sport|regional\\s+sport|local\\s+sport|community\\s+sport|outdoor\\s+sport|field\\s+sport|ground\\s+sport)\\b",

            "sepak takraw" to "\\b(?:sepak\\s+takraw|sepak\\s+takraw\\s+game|sepak\\s+takraw\\s+sport|sepak\\s+takraw\\s+match|sepak\\s+takraw\\s+practice|sepak\\s+takraw\\s+training|sepak\\s+takraw\\s+session|sepak\\s+takraw\\s+workout|sepak\\s+takraw\\s+competition|sepak\\s+takraw\\s+tournament|takraw|sepaktakraw|kick\\s+volleyball|foot\\s+volleyball|rattan\\s+ball|wicker\\s+ball|cane\\s+ball|bamboo\\s+ball|asian\\s+volleyball|southeast\\s+asian\\s+sport|thai\\s+sport|malaysian\\s+sport|indonesian\\s+sport|filipino\\s+sport|vietnamese\\s+sport|cambodian\\s+sport|laotian\\s+sport|myanmar\\s+sport|brunei\\s+sport|singapore\\s+sport|traditional\\s+sepak\\s+takraw|competitive\\s+sepak\\s+takraw|professional\\s+sepak\\s+takraw|amateur\\s+sepak\\s+takraw|recreational\\s+sepak\\s+takraw|team\\s+sepak\\s+takraw|doubles\\s+sepak\\s+takraw|singles\\s+sepak\\s+takraw|net\\s+sport|acrobatic\\s+sport|aerial\\s+sport|flexibility\\s+sport|martial\\s+arts\\s+sport)\\b",

            "snow sports" to "\\b(?:snow\\s+sports|winter\\s+sports|snow\\s+activities|winter\\s+activities|snow\\s+recreation|winter\\s+recreation|alpine\\s+sports|nordic\\s+sports|cold\\s+weather\\s+sports|frozen\\s+sports|ice\\s+sports|snow\\s+games|winter\\s+games|snow\\s+competitions|winter\\s+competitions|snow\\s+olympics|winter\\s+olympics|snow\\s+season\\s+sports|winter\\s+season\\s+sports|outdoor\\s+winter\\s+sports|mountain\\s+winter\\s+sports|backcountry\\s+winter\\s+sports|resort\\s+winter\\s+sports|recreational\\s+snow\\s+sports|competitive\\s+snow\\s+sports|professional\\s+snow\\s+sports|amateur\\s+snow\\s+sports|extreme\\s+snow\\s+sports|adventure\\s+snow\\s+sports|traditional\\s+snow\\s+sports|modern\\s+snow\\s+sports|snow\\s+based\\s+activities|ice\\s+based\\s+activities|winter\\s+weather\\s+activities|cold\\s+climate\\s+sports|freezing\\s+weather\\s+sports|sub\\s+zero\\s+sports|arctic\\s+sports|polar\\s+sports|glacial\\s+sports)\\b",

            "snowmobile" to "\\b(?:snowmobile|snow\\s+mobile|snowmobile\\s+riding|snowmobiling|snow\\s+machine|sled|sledding|snow\\s+sled|motor\\s+sled|power\\s+sled|snowmobile\\s+practice|snowmobile\\s+training|snowmobile\\s+session|snowmobile\\s+workout|snowmobile\\s+tour|snowmobile\\s+trip|snowmobile\\s+expedition|snowmobile\\s+adventure|snowmobile\\s+excursion|snowmobile\\s+safari|snowmobile\\s+race|snowmobile\\s+racing|snowmobile\\s+competition|snowmobile\\s+rally|snowmobile\\s+cross\\s+country|snowmobile\\s+trail\\s+riding|snowmobile\\s+backcountry|snowmobile\\s+mountain|snowmobile\\s+deep\\s+snow|snowmobile\\s+powder|snowmobile\\s+groomed\\s+trails|snowmobile\\s+touring|snowmobile\\s+recreational|snowmobile\\s+sport|snowmobile\\s+freestyle|snowmobile\\s+jumping|snowmobile\\s+stunts|snowmobile\\s+tricks|winter\\s+vehicle|snow\\s+vehicle|arctic\\s+vehicle|all\\s+terrain\\s+vehicle\\s+snow|atv\\s+snow|snow\\s+cat|snow\\s+scooter)\\b",

            "puck" to "\\b(?:puck|puck\\s+game|ice\\s+puck|puck\\s+sport|puck\\s+practice|puck\\s+training|puck\\s+session|puck\\s+workout|puck\\s+match|puck\\s+competition|puck\\s+tournament|hockey\\s+puck|ice\\s+hockey\\s+puck|field\\s+hockey\\s+puck|roller\\s+hockey\\s+puck|street\\s+hockey\\s+puck|ball\\s+hockey\\s+puck|floor\\s+hockey\\s+puck|inline\\s+hockey\\s+puck|deck\\s+hockey\\s+puck|pond\\s+hockey\\s+puck|shinny\\s+puck|pickup\\s+hockey\\s+puck|recreational\\s+hockey\\s+puck|competitive\\s+hockey\\s+puck|professional\\s+hockey\\s+puck|amateur\\s+hockey\\s+puck|youth\\s+hockey\\s+puck|junior\\s+hockey\\s+puck|senior\\s+hockey\\s+puck|adult\\s+hockey\\s+puck|women\\s+hockey\\s+puck|men\\s+hockey\\s+puck|mixed\\s+hockey\\s+puck|league\\s+hockey\\s+puck|team\\s+hockey\\s+puck|individual\\s+hockey\\s+puck|stick\\s+and\\s+puck|puck\\s+handling|puck\\s+skills)\\b",

            "snow car" to "\\b(?:snow\\s+car|snow\\s+vehicle|snow\\s+car\\s+driving|snow\\s+vehicle\\s+driving|winter\\s+car|winter\\s+vehicle|winter\\s+driving|snow\\s+driving|ice\\s+driving|winter\\s+car\\s+racing|snow\\s+car\\s+racing|ice\\s+car\\s+racing|rally\\s+car\\s+snow|rally\\s+car\\s+winter|snow\\s+rally|winter\\s+rally|ice\\s+rally|snow\\s+autocross|winter\\s+autocross|ice\\s+autocross|snow\\s+drift|winter\\s+drift|ice\\s+drift|snow\\s+circuit|winter\\s+circuit|ice\\s+circuit|snow\\s+track|winter\\s+track|ice\\s+track|all\\s+wheel\\s+drive\\s+snow|four\\s+wheel\\s+drive\\s+snow|awd\\s+snow|4wd\\s+snow|snow\\s+chains|winter\\s+tires|studded\\s+tires|snow\\s+tires|winter\\s+car\\s+practice|snow\\s+car\\s+training|winter\\s+car\\s+session|snow\\s+car\\s+workout|winter\\s+motorsport|snow\\s+motorsport|ice\\s+motorsport)\\b",

            "sled" to "\\b(?:sled|sledding|sleigh|sled\\s+riding|sledding\\s+activity|sleigh\\s+riding|sled\\s+practice|sledding\\s+practice|sleigh\\s+practice|sled\\s+training|sledding\\s+training|sleigh\\s+training|sled\\s+session|sledding\\s+session|sleigh\\s+session|sled\\s+workout|sledding\\s+workout|sleigh\\s+workout|toboggan|tobogganing|bobsled|bobsledding|luge|skeleton|snow\\s+tube|snow\\s+tubing|inner\\s+tube\\s+sledding|plastic\\s+sled|wooden\\s+sled|metal\\s+sled|inflatable\\s+sled|racing\\s+sled|recreational\\s+sled|family\\s+sledding|kids\\s+sledding|adult\\s+sledding|downhill\\s+sledding|hill\\s+sledding|mountain\\s+sledding|snow\\s+sledding|ice\\s+sledding|winter\\s+sledding|cold\\s+weather\\s+sledding|outdoor\\s+sledding|backyard\\s+sledding|park\\s+sledding|slope\\s+sledding|gravity\\s+sledding|speed\\s+sledding|thrill\\s+sledding|adventure\\s+sledding|extreme\\s+sledding)\\b",

            "paddleboard surfing" to "\\b(?:paddleboard\\s+surfing|stand\\s+up\\s+paddleboarding|sup|surfing|paddleboard\\s+surfing\\s+session|paddleboard\\s+surfing\\s+practice|paddleboard\\s+surfing\\s+training|paddleboard\\s+surfing\\s+workout|stand\\s+up\\s+paddle\\s+surfing|sup\\s+surfing|paddle\\s+surf|sup\\s+surf|paddleboard\\s+wave\\s+riding|stand\\s+up\\s+paddle\\s+wave\\s+riding|sup\\s+wave\\s+riding|paddle\\s+wave\\s+riding|paddleboard\\s+ocean\\s+surfing|paddleboard\\s+sea\\s+surfing|paddleboard\\s+beach\\s+surfing|paddleboard\\s+coastal\\s+surfing|paddleboard\\s+reef\\s+surfing|paddleboard\\s+break\\s+surfing|longboard\\s+sup|shortboard\\s+sup|inflatable\\s+sup|rigid\\s+sup|touring\\s+sup|racing\\s+sup|yoga\\s+sup|fitness\\s+sup|recreational\\s+sup|competitive\\s+sup|professional\\s+sup|amateur\\s+sup|beginner\\s+sup|intermediate\\s+sup|advanced\\s+sup|expert\\s+sup|flatwater\\s+sup|whitewater\\s+sup|river\\s+sup|lake\\s+sup)\\b",

            "double board skiing" to "\\b(?:double\\s+board\\s+skiing|double\\s+ski|alpine\\s+skiing|downhill\\s+skiing|ski\\s+alpine|ski\\s+downhill|two\\s+ski\\s+skiing|parallel\\s+skiing|carving\\s+skiing|slalom\\s+skiing|giant\\s+slalom|super\\s+g|super\\s+giant\\s+slalom|mogul\\s+skiing|freestyle\\s+skiing|freeride\\s+skiing|all\\s+mountain\\s+skiing|powder\\s+skiing|groomed\\s+skiing|piste\\s+skiing|off\\s+piste\\s+skiing|backcountry\\s+skiing|resort\\s+skiing|recreational\\s+skiing|competitive\\s+skiing|professional\\s+skiing|amateur\\s+skiing|beginner\\s+skiing|intermediate\\s+skiing|advanced\\s+skiing|expert\\s+skiing|racing\\s+skiing|speed\\s+skiing|technical\\s+skiing|traditional\\s+skiing|modern\\s+skiing|winter\\s+skiing|snow\\s+skiing|mountain\\s+skiing|hill\\s+skiing|slope\\s+skiing|trail\\s+skiing|run\\s+skiing|lift\\s+skiing|chairlift\\s+skiing|gondola\\s+skiing)\\b",

            "paddle board" to "\\b(?:paddle\\s+board|stand\\s+up\\s+paddle|sup|paddle\\s+board\\s+practice|paddle\\s+board\\s+training|paddle\\s+board\\s+session|paddle\\s+board\\s+workout|stand\\s+up\\s+paddleboarding|paddleboarding|sup\\s+boarding|paddle\\s+boarding|sup\\s+practice|sup\\s+training|sup\\s+session|sup\\s+workout|flatwater\\s+paddleboarding|ocean\\s+paddleboarding|lake\\s+paddleboarding|river\\s+paddleboarding|sea\\s+paddleboarding|coastal\\s+paddleboarding|touring\\s+paddleboarding|racing\\s+paddleboarding|fitness\\s+paddleboarding|yoga\\s+paddleboarding|recreational\\s+paddleboarding|competitive\\s+paddleboarding|professional\\s+paddleboarding|amateur\\s+paddleboarding|beginner\\s+paddleboarding|intermediate\\s+paddleboarding|advanced\\s+paddleboarding|expert\\s+paddleboarding|longboard\\s+sup|allround\\s+sup|touring\\s+sup|racing\\s+sup|inflatable\\s+sup|rigid\\s+sup|hard\\s+sup|soft\\s+sup|foam\\s+sup|fiberglass\\s+sup|carbon\\s+sup|epoxy\\s+sup)\\b",

            "water sports" to "\\b(?:water\\s+sports|aquatic\\s+sports|water\\s+activities|aquatic\\s+activities|water\\s+recreation|aquatic\\s+recreation|marine\\s+sports|maritime\\s+sports|nautical\\s+sports|ocean\\s+sports|sea\\s+sports|lake\\s+sports|river\\s+sports|pool\\s+sports|swimming\\s+sports|diving\\s+sports|surfing\\s+sports|sailing\\s+sports|rowing\\s+sports|paddling\\s+sports|canoeing\\s+sports|kayaking\\s+sports|windsurfing\\s+sports|kitesurfing\\s+sports|wakeboarding\\s+sports|waterskiing\\s+sports|jet\\s+skiing\\s+sports|fishing\\s+sports|boating\\s+sports|yachting\\s+sports|competitive\\s+water\\s+sports|recreational\\s+water\\s+sports|professional\\s+water\\s+sports|amateur\\s+water\\s+sports|extreme\\s+water\\s+sports|adventure\\s+water\\s+sports|traditional\\s+water\\s+sports|modern\\s+water\\s+sports|olympic\\s+water\\s+sports|international\\s+water\\s+sports|freshwater\\s+sports|saltwater\\s+sports|open\\s+water\\s+sports|calm\\s+water\\s+sports|rough\\s+water\\s+sports)\\b",

            "kayak rafting" to "\\b(?:kayak\\s+rafting|whitewater\\s+kayaking|rafting|kayak\\s+rafting\\s+practice|kayak\\s+rafting\\s+training|kayak\\s+rafting\\s+session|kayak\\s+rafting\\s+workout|whitewater\\s+rafting|river\\s+rafting|rapids\\s+rafting|white\\s+water\\s+rafting|wild\\s+water\\s+rafting|adventure\\s+rafting|extreme\\s+rafting|class\\s+i\\s+rafting|class\\s+ii\\s+rafting|class\\s+iii\\s+rafting|class\\s+iv\\s+rafting|class\\s+v\\s+rafting|class\\s+vi\\s+rafting|recreational\\s+rafting|commercial\\s+rafting|guided\\s+rafting|self\\s+guided\\s+rafting|multi\\s+day\\s+rafting|day\\s+trip\\s+rafting|overnight\\s+rafting|expedition\\s+rafting|family\\s+rafting|team\\s+building\\s+rafting|corporate\\s+rafting|group\\s+rafting|solo\\s+rafting|tandem\\s+rafting|inflatable\\s+kayak|rigid\\s+kayak|sit\\s+on\\s+top\\s+kayak|sit\\s+in\\s+kayak|creek\\s+boat|playboat|river\\s+runner|touring\\s+kayak|sea\\s+kayak|ocean\\s+kayak)\\b",

            "climb the stairs" to "\\b(?:climb\\s+the\\s+stairs|stair\\s+climbing|stair\\s+climb|climb\\s+stairs|climbing\\s+stairs|stair\\s+climbing\\s+practice|stair\\s+climbing\\s+training|stair\\s+climbing\\s+session|stair\\s+climbing\\s+workout|stair\\s+climbing\\s+exercise|step\\s+climbing|step\\s+ups|stair\\s+stepper|stair\\s+master|stair\\s+running|stair\\s+sprints|stair\\s+intervals|stair\\s+cardio|stair\\s+fitness|stair\\s+workout|step\\s+workout|vertical\\s+climbing|vertical\\s+training|tower\\s+climbing|building\\s+climbing|stadium\\s+stairs|escalator\\s+climbing|fire\\s+escape\\s+climbing|emergency\\s+stairs|spiral\\s+stairs|straight\\s+stairs|outdoor\\s+stairs|indoor\\s+stairs|public\\s+stairs|home\\s+stairs|office\\s+stairs|apartment\\s+stairs|mall\\s+stairs|hotel\\s+stairs|parking\\s+garage\\s+stairs|subway\\s+stairs|train\\s+station\\s+stairs|airport\\s+stairs|stadium\\s+climbing|arena\\s+climbing|monument\\s+climbing|climbing stairs)\\b",

            "aerobics" to "\\b(?:aerobics|aerobics\\s+class|aerobics\\s+workout|aerobics\\s+session|aerobics\\s+practice|aerobics\\s+exercise|aerobic\\s+exercise|aerobic\\s+workout|aerobic\\s+fitness|cardio\\s+aerobics|dance\\s+aerobics|step\\s+aerobics|water\\s+aerobics|aqua\\s+aerobics|low\\s+impact\\s+aerobics|high\\s+impact\\s+aerobics|chair\\s+aerobics|senior\\s+aerobics|prenatal\\s+aerobics|postnatal\\s+aerobics|kickboxing\\s+aerobics|martial\\s+arts\\s+aerobics|latin\\s+aerobics|zumba\\s+aerobics|funk\\s+aerobics|jazz\\s+aerobics|hip\\s+hop\\s+aerobics|retro\\s+aerobics|vintage\\s+aerobics|classic\\s+aerobics|traditional\\s+aerobics|modern\\s+aerobics|freestyle\\s+aerobics|circuit\\s+aerobics|interval\\s+aerobics|bootcamp\\s+aerobics|military\\s+aerobics|sports\\s+aerobics|functional\\s+aerobics|core\\s+aerobics|toning\\s+aerobics|sculpting\\s+aerobics|fat\\s+burning\\s+aerobics|calorie\\s+burning\\s+aerobics)\\b",

            "physical training" to "\\b(?:physical\\s+training|pt|physical\\s+fitness|physical\\s+training\\s+session|physical\\s+training\\s+practice|physical\\s+training\\s+workout|physical\\s+training\\s+exercise|physical\\s+conditioning|fitness\\s+training|fitness\\s+conditioning|body\\s+conditioning|strength\\s+and\\s+conditioning|athletic\\s+training|sports\\s+training|military\\s+training|army\\s+training|navy\\s+training|air\\s+force\\s+training|marine\\s+training|boot\\s+camp|bootcamp|military\\s+fitness|tactical\\s+fitness|functional\\s+fitness|combat\\s+fitness|warrior\\s+fitness|soldier\\s+fitness|recruit\\s+training|basic\\s+training|advanced\\s+training|specialized\\s+training|unit\\s+training|squad\\s+training|team\\s+training|group\\s+training|individual\\s+training|personal\\s+training|one\\s+on\\s+one\\s+training|private\\s+training|supervised\\s+training|self\\s+directed\\s+training|unsupervised\\s+training|structured\\s+training|unstructured\\s+training|formal\\s+training|informal\\s+training)\\b",

            "wall ball" to "\\b(?:wall\\s+ball|wall\\s+ball\\s+workout|wall\\s+ball\\s+exercise|wall\\s+ball\\s+training|wall\\s+ball\\s+session|wall\\s+ball\\s+practice|wall\\s+ball\\s+throws|wall\\s+ball\\s+shots|medicine\\s+ball\\s+wall|med\\s+ball\\s+wall|slam\\s+ball\\s+wall|weighted\\s+ball\\s+wall|fitness\\s+ball\\s+wall|exercise\\s+ball\\s+wall|wall\\s+ball\\s+squats|wall\\s+ball\\s+lunges|wall\\s+ball\\s+burpees|wall\\s+ball\\s+thrusters|wall\\s+ball\\s+slams|wall\\s+ball\\s+tosses|wall\\s+ball\\s+catches|wall\\s+ball\\s+rebounds|crossfit\\s+wall\\s+ball|functional\\s+wall\\s+ball|athletic\\s+wall\\s+ball|sports\\s+wall\\s+ball|conditioning\\s+wall\\s+ball|strength\\s+wall\\s+ball|power\\s+wall\\s+ball|explosive\\s+wall\\s+ball|plyometric\\s+wall\\s+ball|cardio\\s+wall\\s+ball|hiit\\s+wall\\s+ball|circuit\\s+wall\\s+ball|interval\\s+wall\\s+ball|endurance\\s+wall\\s+ball|full\\s+body\\s+wall\\s+ball|compound\\s+wall\\s+ball|functional\\s+movement\\s+wall\\s+ball)\\b",

            "bobby jump" to "\\b(?:bobby\\s+jump|bobby\\s+jump\\s+workout)\\b",

            "upper limb training" to "\\b(?:upper\\s+limb\\s+training|upper\\s+body\\s+training|arm\\s+training)\\b",

            "lower limb training" to "\\b(?:lower\\s+limb\\s+training|lower\\s+body\\s+training|leg\\s+training)\\b",

            "waist and abdomen training" to "\\b(?:waist\\s+and\\s+abdomen\\s+training|waist\\s+training|abdomen\\s+training|core\\s+training)\\b",

            "back training" to "\\b(?:back\\s+training|back\\s+workout|back\\s+exercises)\\b",

            "gymnastics" to "\\b(?:gymnastics|gymnastics\\s+practice|gymnastics\\s+class|gymnastics\\s+session|gymnastics\\s+training|gymnastics\\s+workout|artistic\\s+gymnastics|rhythmic\\s+gymnastics|trampoline\\s+gymnastics|acrobatic\\s+gymnastics|aerobic\\s+gymnastics|men\\s+artistic\\s+gymnastics|women\\s+artistic\\s+gymnastics|floor\\s+exercise|vault|uneven\\s+bars|balance\\s+beam|pommel\\s+horse|still\\s+rings|parallel\\s+bars|horizontal\\s+bar|high\\s+bar|ribbon\\s+gymnastics|hoop\\s+gymnastics|ball\\s+gymnastics|clubs\\s+gymnastics|rope\\s+gymnastics|individual\\s+rhythmic|group\\s+rhythmic|double\\s+mini\\s+trampoline|tumbling\\s+gymnastics|power\\s+tumbling|recreational\\s+gymnastics|competitive\\s+gymnastics|elite\\s+gymnastics|developmental\\s+gymnastics|preschool\\s+gymnastics|toddler\\s+gymnastics|youth\\s+gymnastics|adult\\s+gymnastics|masters\\s+gymnastics|special\\s+olympics\\s+gymnastics|adaptive\\s+gymnastics|therapeutic\\s+gymnastics)\\b",

            "freestyle" to "\\b(?:freestyle|freestyle\\s+workout|free\\s+style|freestyle\\s+session|freestyle\\s+practice|freestyle\\s+training|freestyle\\s+exercise|freestyle\\s+movement|free\\s+form\\s+exercise|unstructured\\s+exercise|improvisational\\s+exercise|creative\\s+exercise|open\\s+format\\s+exercise|flexible\\s+workout|adaptable\\s+workout|personalized\\s+workout|individualized\\s+exercise|self\\s+directed\\s+exercise|spontaneous\\s+exercise|intuitive\\s+exercise|flowing\\s+exercise|dynamic\\s+exercise|variable\\s+exercise|mixed\\s+movement|eclectic\\s+workout|fusion\\s+exercise|cross\\s+training\\s+freestyle|functional\\s+freestyle|bodyweight\\s+freestyle|calisthenics\\s+freestyle|dance\\s+freestyle|martial\\s+arts\\s+freestyle|swimming\\s+freestyle|running\\s+freestyle|cycling\\s+freestyle|general\\s+fitness|open\\s+gym|free\\s+play\\s+exercise|recreational\\s+freestyle|competitive\\s+freestyle)\\b",

            "indoor fitness" to "\\b(?:indoor\\s+fitness|indoor\\s+gym|indoor\\s+workout|indoor\\s+fitness\\s+session|indoor\\s+fitness\\s+practice|indoor\\s+fitness\\s+training|indoor\\s+exercise|gym\\s+workout|fitness\\s+center|health\\s+club|fitness\\s+facility|recreation\\s+center|wellness\\s+center|exercise\\s+facility|indoor\\s+training|climate\\s+controlled\\s+fitness|all\\s+weather\\s+fitness|year\\s+round\\s+fitness|home\\s+gym|basement\\s+gym|garage\\s+gym|spare\\s+room\\s+gym|apartment\\s+gym|condo\\s+gym|office\\s+gym|hotel\\s+gym|corporate\\s+fitness|workplace\\s+wellness|group\\s+fitness\\s+class|personal\\s+training|circuit\\s+training\\s+indoor|bootcamp\\s+indoor|crossfit\\s+indoor|spinning\\s+class|indoor\\s+cycling\\s+class|weight\\s+room|cardio\\s+room|fitness\\s+studio|exercise\\s+studio|multi\\s+purpose\\s+room)\\b",

            "flexibility training" to "\\b(?:flexibility\\s+training|flexibility|stretching|stretch|flexibility\\s+session|flexibility\\s+practice|flexibility\\s+workout|flexibility\\s+exercise|stretch\\s+session|stretch\\s+practice|stretch\\s+workout|stretch\\s+routine|static\\s+stretching|dynamic\\s+stretching|active\\s+stretching|passive\\s+stretching|pnf\\s+stretching|proprioceptive\\s+neuromuscular\\s+facilitation|ballistic\\s+stretching|myofascial\\s+release|foam\\s+rolling|mobility\\s+work|range\\s+of\\s+motion|rom\\s+training|joint\\s+mobility|muscle\\s+lengthening|tendon\\s+stretching|fascia\\s+release|trigger\\s+point\\s+release|soft\\s+tissue\\s+work|flexibility\\s+conditioning|suppleness\\s+training|limberness\\s+training|elasticity\\s+training|pliability\\s+training|therapeutic\\s+stretching|corrective\\s+stretching|injury\\s+prevention\\s+stretching|recovery\\s+stretching|maintenance\\s+stretching|performance\\s+stretching)\\b",

            "stretching" to "\\b(?:stretching|stretch|stretch\\s+workout|stretching\\s+session|stretching\\s+practice|stretching\\s+routine|stretching\\s+exercise|morning\\s+stretch|evening\\s+stretch|bedtime\\s+stretch|wake\\s+up\\s+stretch|pre\\s+workout\\s+stretch|post\\s+workout\\s+stretch|cool\\s+down\\s+stretch|warm\\s+up\\s+stretch|maintenance\\s+stretch|recovery\\s+stretch|gentle\\s+stretch|deep\\s+stretch|full\\s+body\\s+stretch|upper\\s+body\\s+stretch|lower\\s+body\\s+stretch|back\\s+stretch|neck\\s+stretch|shoulder\\s+stretch|arm\\s+stretch|leg\\s+stretch|hip\\s+stretch|hamstring\\s+stretch|calf\\s+stretch|quad\\s+stretch|glute\\s+stretch|chest\\s+stretch|spinal\\s+stretch|seated\\s+stretch|standing\\s+stretch|lying\\s+stretch|floor\\s+stretch|wall\\s+stretch|partner\\s+stretch|assisted\\s+stretch|self\\s+stretch|active\\s+stretch|passive\\s+stretch)\\b",

            "mixed aerobics" to "\\b(?:mixed\\s+aerobics|mixed\\s+aerobics\\s+class|mixed\\s+aerobics\\s+session|mixed\\s+aerobics\\s+workout|combination\\s+aerobics|fusion\\s+aerobics|hybrid\\s+aerobics|variety\\s+aerobics|multi\\s+style\\s+aerobics|cross\\s+training\\s+aerobics|circuit\\s+aerobics|interval\\s+aerobics|dance\\s+aerobics\\s+mix|step\\s+aerobics\\s+mix|kickboxing\\s+aerobics|martial\\s+arts\\s+aerobics|aqua\\s+aerobics\\s+mix|low\\s+impact\\s+high\\s+impact|cardio\\s+strength\\s+mix|toning\\s+aerobics|sculpting\\s+aerobics|bootcamp\\s+aerobics|military\\s+style\\s+aerobics|athletic\\s+aerobics|sports\\s+conditioning\\s+aerobics|functional\\s+aerobics|core\\s+aerobics|flexibility\\s+aerobics|balance\\s+aerobics|coordination\\s+aerobics|agility\\s+aerobics|power\\s+aerobics|endurance\\s+aerobics|fat\\s+burning\\s+aerobics|calorie\\s+burning\\s+aerobics)\\b",

            "outdoor hiking" to "\\b(?:outdoor\\s+hiking|outdoor\\s+hike|hiking|hike|outdoor\\s+hiking\\s+session|outdoor\\s+hiking\\s+practice|outdoor\\s+hiking\\s+training|outdoor\\s+hiking\\s+workout|trail\\s+hiking|mountain\\s+hiking|hill\\s+hiking|forest\\s+hiking|woodland\\s+hiking|desert\\s+hiking|coastal\\s+hiking|ridge\\s+hiking|valley\\s+hiking|canyon\\s+hiking|gorge\\s+hiking|alpine\\s+hiking|subalpine\\s+hiking|lowland\\s+hiking|upland\\s+hiking|backcountry\\s+hiking|wilderness\\s+hiking|day\\s+hiking|overnight\\s+hiking|multi\\s+day\\s+hiking|long\\s+distance\\s+hiking|section\\s+hiking|thru\\s+hiking|peak\\s+bagging|summit\\s+hiking|nature\\s+hiking|scenic\\s+hiking|photography\\s+hiking|bird\\s+watching\\s+hiking|wildlife\\s+hiking|botanical\\s+hiking|geological\\s+hiking|historical\\s+hiking|cultural\\s+hiking|pilgrimage\\s+hiking|spiritual\\s+hiking|meditation\\s+hiking|fitness\\s+hiking|conditioning\\s+hiking|training\\s+hiking)\\b",

            "indoor skating" to "\\b(?:indoor\\s+skating|indoor\\s+skate|ice\\s+skating|ice\\s+skate|indoor\\s+skating\\s+session|indoor\\s+skating\\s+practice|indoor\\s+skating\\s+workout|figure\\s+skating|speed\\s+skating|hockey\\s+skating|recreational\\s+skating|public\\s+skating|freestyle\\s+skating|synchronized\\s+skating|ice\\s+dancing|pairs\\s+skating|singles\\s+skating|competitive\\s+skating|artistic\\s+skating|precision\\s+skating|learn\\s+to\\s+skate|basic\\s+skating|intermediate\\s+skating|advanced\\s+skating|power\\s+skating|edge\\s+work|stroking|crossovers|turns|jumps|spins|spirals|rink\\s+skating|arena\\s+skating|ice\\s+rink|skating\\s+rink|artificial\\s+ice|natural\\s+ice|indoor\\s+ice|climate\\s+controlled\\s+skating)\\b",

            "outdoor skating" to "\\b(?:outdoor\\s+skating|outdoor\\s+skate|roller\\s+skating|roller\\s+skate|skating|outdoor\\s+skating\\s+session|outdoor\\s+skating\\s+practice|outdoor\\s+skating\\s+workout|inline\\s+skating|rollerblading|quad\\s+skating|artistic\\s+roller\\s+skating|speed\\s+roller\\s+skating|roller\\s+hockey|roller\\s+derby|jam\\s+skating|dance\\s+skating|freestyle\\s+roller\\s+skating|aggressive\\s+skating|street\\s+skating\\s+roller|park\\s+skating\\s+roller|vert\\s+skating\\s+roller|recreational\\s+roller\\s+skating|fitness\\s+skating|endurance\\s+skating|distance\\s+skating|trail\\s+skating|path\\s+skating|sidewalk\\s+skating|pavement\\s+skating|asphalt\\s+skating|concrete\\s+skating|smooth\\s+surface\\s+skating|rink\\s+skating\\s+outdoor|outdoor\\s+rink|skate\\s+park|roller\\s+rink\\s+outdoor)\\b",

            "roller skating" to "\\b(?:roller\\s+skating|roller\\s+skate|rollerblading|inline\\s+skating|roller\\s+skating\\s+session|roller\\s+skating\\s+practice|roller\\s+skating\\s+workout|quad\\s+roller\\s+skating|inline\\s+roller\\s+skating|recreational\\s+roller\\s+skating|artistic\\s+roller\\s+skating|rhythm\\s+roller\\s+skating|dance\\s+roller\\s+skating|freestyle\\s+roller\\s+skating|speed\\s+roller\\s+skating|fitness\\s+roller\\s+skating|aggressive\\s+inline\\s+skating|vert\\s+roller\\s+skating|street\\s+roller\\s+skating|park\\s+roller\\s+skating|jam\\s+roller\\s+skating|roller\\s+derby|flat\\s+track\\s+derby|banked\\s+track\\s+derby|roller\\s+hockey|inline\\s+hockey|rink\\s+roller\\s+skating|outdoor\\s+roller\\s+skating|trail\\s+roller\\s+skating|distance\\s+roller\\s+skating|endurance\\s+roller\\s+skating|competitive\\s+roller\\s+skating|recreational\\s+roller\\s+skating|social\\s+roller\\s+skating|family\\s+roller\\s+skating)\\b",

            "skateboarding" to "\\b(?:skateboarding|skateboard|skate\\s+boarding|skateboarding\\s+session|skateboarding\\s+practice|skateboarding\\s+training|skateboarding\\s+workout|street\\s+skating|vert\\s+skating|park\\s+skating|bowl\\s+skating|mini\\s+ramp\\s+skating|half\\s+pipe\\s+skating|quarter\\s+pipe\\s+skating|transition\\s+skating|freestyle\\s+skateboarding|longboarding|longboard\\s+skating|cruiser\\s+skateboarding|downhill\\s+skateboarding|slalom\\s+skateboarding|freeride\\s+skateboarding|dancing\\s+skateboarding|carving\\s+skateboarding|pumping\\s+skateboarding|pushing\\s+skateboarding|technical\\s+skateboarding|flow\\s+skateboarding|creative\\s+skateboarding|artistic\\s+skateboarding|old\\s+school\\s+skateboarding|new\\s+school\\s+skateboarding|traditional\\s+skateboarding|modern\\s+skateboarding|recreational\\s+skateboarding|competitive\\s+skateboarding|amateur\\s+skateboarding|professional\\s+skateboarding|youth\\s+skateboarding|adult\\s+skateboarding|beginner\\s+skateboarding|intermediate\\s+skateboarding|advanced\\s+skateboarding|expert\\s+skateboarding)\\b",

            "croquet" to "\\b(?:croquet|croquet\\s+game|croquet\\s+session|croquet\\s+practice|croquet\\s+training|croquet\\s+workout|association\\s+croquet|golf\\s+croquet|american\\s+croquet|six\\s+wicket\\s+croquet|nine\\s+wicket\\s+croquet|english\\s+croquet|french\\s+croquet|garden\\s+croquet|lawn\\s+croquet|tournament\\s+croquet|competitive\\s+croquet|recreational\\s+croquet|social\\s+croquet|family\\s+croquet|club\\s+croquet|professional\\s+croquet|amateur\\s+croquet|singles\\s+croquet|doubles\\s+croquet|team\\s+croquet|handicap\\s+croquet|level\\s+play\\s+croquet|advanced\\s+play\\s+croquet|short\\s+croquet|full\\s+bisque\\s+croquet|half\\s+bisque\\s+croquet|mallet\\s+sport|wicket\\s+sport|hoop\\s+sport|precision\\s+sport\\s+croquet|strategy\\s+game\\s+croquet|tactical\\s+game\\s+croquet|skill\\s+game\\s+croquet)\\b",

            "handball" to "\\b(?:handball|handball\\s+game|handball\\s+session|handball\\s+practice|handball\\s+training|handball\\s+workout|team\\s+handball|olympic\\s+handball|european\\s+handball|international\\s+handball|indoor\\s+handball|outdoor\\s+handball|court\\s+handball|field\\s+handball|seven\\s+a\\s+side\\s+handball|beach\\s+handball|sand\\s+handball|four\\s+a\\s+side\\s+handball|american\\s+handball|one\\s+wall\\s+handball|three\\s+wall\\s+handball|four\\s+wall\\s+handball|wallball|racquetball\\s+handball|squash\\s+handball|recreational\\s+handball|competitive\\s+handball|professional\\s+handball|amateur\\s+handball|youth\\s+handball|junior\\s+handball|senior\\s+handball|women\\s+handball|men\\s+handball|mixed\\s+handball|club\\s+handball|league\\s+handball|tournament\\s+handball|championship\\s+handball|world\\s+handball|ball\\s+sport\\s+handball|goal\\s+sport\\s+handball)\\b",

            "free sparring" to "\\b(?:free\\s+sparring|sparring|sparring\\s+practice|free\\s+sparring\\s+session|free\\s+sparring\\s+workout|light\\s+sparring|medium\\s+sparring|heavy\\s+sparring|full\\s+contact\\s+sparring|semi\\s+contact\\s+sparring|no\\s+contact\\s+sparring|point\\s+sparring|continuous\\s+sparring|controlled\\s+sparring|technical\\s+sparring|flow\\s+sparring|positional\\s+sparring|situational\\s+sparring|conditional\\s+sparring|theme\\s+sparring|drill\\s+sparring|partner\\s+practice|combat\\s+practice|fighting\\s+practice|martial\\s+arts\\s+sparring|boxing\\s+sparring|kickboxing\\s+sparring|mma\\s+sparring|karate\\s+sparring|taekwondo\\s+sparring|judo\\s+sparring|wrestling\\s+sparring|bjj\\s+sparring|grappling\\s+sparring|stand\\s+up\\s+sparring|ground\\s+sparring|clinch\\s+sparring|defensive\\s+sparring|offensive\\s+sparring|reaction\\s+practice|timing\\s+practice|distance\\s+practice)\\b",

            "tai chi" to "\\b(?:tai\\s+chi|tai\\s+chi\\s+practice|tai\\s+chi\\s+session|tai\\s+chi\\s+training|tai\\s+chi\\s+workout|tai\\s+chi\\s+class|taiji|tai\\s+chi\\s+chuan|taijiquan|chen\\s+style\\s+tai\\s+chi|yang\\s+style\\s+tai\\s+chi|wu\\s+style\\s+tai\\s+chi|sun\\s+style\\s+tai\\s+chi|hao\\s+style\\s+tai\\s+chi|traditional\\s+tai\\s+chi|modern\\s+tai\\s+chi|simplified\\s+tai\\s+chi|24\\s+form\\s+tai\\s+chi|42\\s+form\\s+tai\\s+chi|108\\s+form\\s+tai\\s+chi|long\\s+form\\s+tai\\s+chi|short\\s+form\\s+tai\\s+chi|slow\\s+form\\s+tai\\s+chi|fast\\s+form\\s+tai\\s+chi|weapon\\s+forms\\s+tai\\s+chi|sword\\s+tai\\s+chi|fan\\s+tai\\s+chi|pushing\\s+hands|tui\\s+shou|meditation\\s+in\\s+motion|moving\\s+meditation|qigong\\s+tai\\s+chi|chi\\s+kung\\s+tai\\s+chi|internal\\s+martial\\s+arts|soft\\s+martial\\s+arts|health\\s+tai\\s+chi|therapeutic\\s+tai\\s+chi|senior\\s+tai\\s+chi|chair\\s+tai\\s+chi)\\b",

            "mountain cycling" to "\\b(?:mountain\\s+cycling|mountain\\s+bike|mountain\\s+biking|mtb|mountain\\s+terrain\\s+bike|off\\s+road\\s+cycling|trail\\s+cycling|single\\s+track\\s+cycling|technical\\s+cycling|cross\\s+country\\s+mountain\\s+biking|xc\\s+mountain\\s+biking|downhill\\s+mountain\\s+biking|enduro\\s+mountain\\s+biking|all\\s+mountain\\s+biking|freeride\\s+mountain\\s+biking|dirt\\s+jumping|trials\\s+mountain\\s+biking|fat\\s+bike|plus\\s+bike|hardtail\\s+mountain\\s+bike|full\\s+suspension|dual\\s+suspension|29er\\s+mountain\\s+bike|27\\.5\\s+mountain\\s+bike|26\\s+inch\\s+mountain\\s+bike|aggressive\\s+mountain\\s+biking|technical\\s+mountain\\s+biking|flow\\s+trail\\s+biking|jump\\s+trail\\s+biking|pump\\s+track\\s+biking|skills\\s+park\\s+biking|bike\\s+park|mountain\\s+bike\\s+park|lift\\s+access\\s+biking|shuttle\\s+biking|epic\\s+mountain\\s+biking|backcountry\\s+mountain\\s+biking)\\b"
        )
        
        // Sort by pattern length for more specific matching
        val sortedActivities = activities.entries.sortedByDescending { it.value.length }
        
        for ((activity, pattern) in sortedActivities) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return activity
            }
        }
        
        return null
    }
    
    private fun extractActivityNumber(text: String): Int? {
        // First extract the activity type
        val activityType = extractActivityType(text)
        Log.d(LOG_TAG, "🔢 extractActivityNumber - activity_type: $activityType")
        
        if (activityType == null) {
            return null
        }
        
        // Then look up the activity number from the mapping
        val activityNumber = activityNumberMap[activityType]
        Log.d(LOG_TAG, "🔢 extractActivityNumber - activity_number: $activityNumber")
        
        return activityNumber
    }
    
    private fun extractApp(text: String): String? {
        val apps = mapOf(
            "heart rate" to "\\b(?:heart\\s+rate|heartrate|heart\\s+beat|heartbeat|pulse|pulse\\s+rate|bpm|beats\\s+per\\s+minute|cardiac|cardiac\\s+rate|heart\\s+rhythm|resting\\s+heart\\s+rate|rhr|max\\s+heart\\s+rate|maximum\\s+heart\\s+rate|heart\\s+health|cardiovascular|cardio|ticker|heart\\s+monitor|heart\\s+sensor|hr|beat|beats|beating|palpitation|palpitations|tachycardia|bradycardia|heart\\s+zone|target\\s+heart\\s+rate|recovery\\s+heart\\s+rate)\\b",
            
            "blood oxygen" to "\\b(?:blood\\s+oxygen|oxygen|o2|spo2|sp\\s+o2|oxygen\\s+saturation|oxygen\\s+level|oxygen\\s+levels|blood\\s+o2|oxygen\\s+sat|o2\\s+sat|o2\\s+level|o2\\s+saturation|pulse\\s+ox|pulse\\s+oximetry|oximeter|oxygen\\s+reading|oxygen\\s+sensor|saturation|sat|blood\\s+oxygen\\s+level|arterial\\s+oxygen|respiratory|respiration|breathing|breath|lung\\s+function|oxygenation|hypoxia|oxygen\\s+content|sp2|SP2)\\b",
            
            "stress" to "\\b(?:stress|stressed|stressful|stress\\s+level|stress\\s+score|stress\\s+index|anxiety|anxious|worried|worry|worrying|tension|tense|pressure|pressured|strain|strained|overwhelm|overwhelmed|nervous|nervousness|burnout|burnt\\s+out|mental\\s+stress|emotional\\s+stress|psychological\\s+stress|chronic\\s+stress|acute\\s+stress|relaxation|relax|calm|calmness|peace|peaceful|tranquil|serene|zen|mindfulness)\\b",

            "brightness" to "\\b(?:brightness|bright|brighter|brighten|brightening|screen\\s+brightness|display\\s+brightness|luminosity|luminance|backlight|screen\\s+light|light\\s+level|dim|dimmer|dimming|dimness|darken|darker|darkening|auto\\s+brightness|adaptive\\s+brightness|brightness\\s+level|screen\\s+intensity|display\\s+intensity|illumination|illuminate|glow|glowing|radiance|light\\s+output|ambient\\s+light|screen\\s+glow|visibility|contrast|gamma|exposure|luminous)\\b"
        )
        
        // Sort by pattern length for more specific matching
        val sortedApps = apps.entries.sortedByDescending { it.value.length }
        
        for ((app, pattern) in sortedApps) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return app
            }
        }
        
        return null
    }
    
    private fun extractContact(text: String): String? {
        // First check for hardcoded contacts
        val contacts = mapOf(
            "mom" to "\\b(?:mom|mother|mama|mum)\\b",
            "dad" to "\\b(?:dad|father|papa|pop)\\b",
            "sister" to "\\b(?:sister|sis)\\b",
            "brother" to "\\b(?:brother|bro)\\b"
        )

        for ((contact, pattern) in contacts) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return contact
            }
        }

        // Special case: check for emergency services first
        val emergencyServices = mapOf(
            "police" to "\\b(?:police|cops?|law\\s+enforcement|911|emergency|authorities)\\b",
            "ambulance" to "\\b(?:ambulance|paramedics?|medical\\s+emergency|hospital\\s+emergency)\\b",
            "fire department" to "\\b(?:fire\\s+department|fire\\s+brigade|firefighters?)\\b"
        )
        
        for ((service, pattern) in emergencyServices) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return service
            }
        }

        // If no hardcoded contact found, try to extract any name after phone action keywords
        val phoneActionPatterns = listOf(
            // Pattern 1: Standard action + optional modifiers + contact (handles "call back rahul india")
            "(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|reach\\s+out|give\\s+a\\s+call|make\\s+a\\s+call|place\\s+a\\s+call|telephone|buzz|video\\s+call|voice\\s+call|facetime)\\s+(?:back\\s+)?(.+?)(?:\\s*$|\\s+(?:now|please|right\\s+now|immediately|asap|urgently)\\s*$)",
            
            // Pattern 2: Messaging actions + contact
            "(?:message|messaging|text|texting|sms|send|sending|write|compose|type|drop\\s+a\\s+message|send\\s+a\\s+text|shoot\\s+a\\s+message|ping|dm|direct\\s+message|whatsapp|imessage|chat|msg)\\s+(.+?)(?:\\s*$|\\s+(?:now|please|right\\s+now|immediately|asap|urgently)\\s*$)",
            
            // Pattern 3: "get in touch with X" format
            "(?:get\\s+in\\s+touch\\s+with|reach\\s+out\\s+to|contact)\\s+(.+?)(?:\\s*$|\\s+(?:now|please|right\\s+now|immediately|asap|urgently)\\s*$)",
            
            // Pattern 4: "call back" specific pattern
            "(?:call\\s+back|ring\\s+back|phone\\s+back)\\s+(.+?)(?:\\s*$|\\s+(?:now|please|right\\s+now|immediately|asap|urgently)\\s*$)"
        )

        for (pattern in phoneActionPatterns) {
            val regex = pattern.toRegex(RegexOption.IGNORE_CASE)
            val match = regex.find(text)
            if (match != null && match.groupValues.size > 1) {
                val extractedName = match.groupValues[1].trim()
                
                // Check if the extracted text is a phone number
                val phoneNumberPattern = "\\b\\d{10,15}\\b".toRegex()
                if (phoneNumberPattern.matches(extractedName)) {
                    return extractedName // Return the phone number directly
                }
                
                // Clean up the extracted name (remove common stop words and punctuation at the beginning)
                val cleanedName = extractedName
                    .replace(Regex("^(?:to|my|the|a|an)\\s+", RegexOption.IGNORE_CASE), "")
                    .replace(Regex("\\s+(?:please|now|right\\s+now|immediately|asap|urgently)$", RegexOption.IGNORE_CASE), "")
                    .replace(Regex("[^a-zA-Z\\s]"), "")
                    .trim()

                // Additional validation: make sure it's not just common words
                val commonWords = setOf(
                    "the", "a", "an", "to", "my", "please", "now", "today", "tomorrow", "here", "there",
                    "this", "that", "these", "those", "is", "are", "was", "were", "be", "been", "being",
                    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can",
                    "may", "might", "must", "shall", "and", "or"
                )
                if (cleanedName.isNotEmpty() && 
                    cleanedName.length > 1 && 
                    !commonWords.contains(cleanedName.toLowerCase()) &&
                    !cleanedName.matches(Regex("^\\d+$"))) { // Not just numbers
                    return cleanedName
                }
            }
        }

        // Additional pattern specifically for phone numbers with common formats
        val phoneNumberPatterns = listOf(
            // Pattern for standalone phone numbers (10-15 digits) - updated to handle "back" modifier
            "(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|message|messaging|text|texting|sms)\\s+(?:back\\s+)?(\\d{10,15})\\b",
            
            // Pattern for phone numbers with separators (spaces, hyphens, dots)
            "(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|message|messaging|text|texting|sms)\\s+(?:back\\s+)?(\\d{3,4}[\\s\\-\\.]{0,1}\\d{3,4}[\\s\\-\\.]{0,1}\\d{4,6})\\b",
            
            // Pattern for phone numbers with country codes (+91, +1, etc.)
            "(?:call|calling|phone|dial|dialing|ring|ringing|contact|reach|message|messaging|text|texting|sms)\\s+(?:back\\s+)?(?:\\+\\d{1,3}[\\s\\-\\.]{0,1})?([\\d\\s\\-\\.]{10,20})\\b"
        )

        for (pattern in phoneNumberPatterns) {
            val regex = pattern.toRegex(RegexOption.IGNORE_CASE)
            val match = regex.find(text)
            if (match != null && match.groupValues.size > 1) {
                val extractedNumber = match.groupValues[1].trim()
                // Clean the number by removing separators and keeping only digits
                val cleanedNumber = extractedNumber.replace(Regex("[^\\d]"), "")
                // Validate that it's a reasonable phone number length (10-15 digits)
                if (cleanedNumber.length in 10..15) {
                    return cleanedNumber
                }
            }
        }

        return null
    }
    
    private fun extractLocation(text: String): String? {
        // First check for hardcoded locations
        val locations = mapOf(
            "london" to "\\b(?:london|uk|england)\\b",
            "bangalore" to "\\b(?:bangalore|bengaluru|blr)\\b",
            "mumbai" to "\\b(?:mumbai|bombay)\\b",
            "delhi" to "\\b(?:delhi|new\\s+delhi)\\b"
        )
        
        for ((location, pattern) in locations) {
            if (text.contains(pattern.toRegex())) {
                return location
            }
        }

        // Try to extract location after weather-related keywords
        val weatherPattern = "(?:weather|forecast|temperature|climate|conditions)\\s+(.+?)(?:\\s|$|\\.|\\?|!)"
        val regex = weatherPattern.toRegex(RegexOption.IGNORE_CASE)
        val match = regex.find(text)
        if (match != null && match.groupValues.size > 1) {
            val potentialLocation = match.groupValues[1].trim()
            // Clean up the extracted location (remove common stop words and punctuation)
            val cleanedLocation = potentialLocation
                .replace(Regex("^(in|at|for|of|the|a|an)\\s+", RegexOption.IGNORE_CASE), "")
                .replace(Regex("[^a-zA-Z\\s]"), "")
                .trim()
            
            if (cleanedLocation.isNotEmpty() && cleanedLocation != "now" && cleanedLocation != "today" && cleanedLocation != "tomorrow") {
                return cleanedLocation
            }
        }
        
        return "current location"  // Default
    }
    
    private fun extractAttribute(text: String): String? {
        val attributes = mapOf(
            "forecast" to "\\b(?:forecast|forecasts|forecasting|prediction|predictions|predict|predicted|outlook|outlooks|projection|projections|future\\s+weather|upcoming\\s+weather|expected\\s+weather|weather\\s+ahead|what'?s\\s+coming|coming\\s+weather|next\\s+days|next\\s+week|tomorrow|later|ahead|anticipate|anticipated|expect|expected|probable|likely|chance\\s+of)\\b",
        
            "temperature" to "\\b(?:temperature|temperatures|temp|temps|hot|cold|warm|cool|heat|heated|heating|chill|chilly|freezing|frozen|frost|frosty|degrees|degree|celsius|fahrenheit|thermometer|thermal|feels\\s+like|wind\\s+chill|heat\\s+index|mild|moderate|extreme|scorching|boiling|icy|frigid|lukewarm|toasty)\\b",
            
            "rain" to "\\b(?:rain|raining|rainy|rained|rainfall|rainwater|shower|showers|showery|drizzle|drizzling|drizzly|sprinkle|sprinkling|downpour|pouring|pour|umbrella|wet|wetness|damp|dampness|moisture|precipitation|precipitating|storm|stormy|thunderstorm|cloudburst|deluge|mist|misty|monsoon)\\b",
            
            "humidity" to "\\b(?:humidity|humid|moisture|moistness|damp|dampness|dank|muggy|sticky|clammy|steamy|sultry|wet|wetness|dew|dew\\s+point|relative\\s+humidity|water\\s+vapor|vapour|atmospheric\\s+moisture|air\\s+moisture|condensation|saturated|saturation|dry|dryness|arid)\\b",
            
            "air quality" to "\\b(?:air\\s+quality|aqi|air\\s+quality\\s+index|pollution|polluted|pollutants|smog|smoggy|haze|hazy|particulate|particles|pm2\\.?5|pm10|pm\\s+2\\.?5|pm\\s+10|ozone|allergens|pollen|dust|emissions|exhaust|fumes|toxic|toxins|clean\\s+air|dirty\\s+air|unhealthy\\s+air|breathable|air\\s+pollution)\\b"
        )
        
        for ((attr, pattern) in attributes) {
            if (text.contains(pattern.toRegex())) {
                return attr
            }
        }
        
        return null
    }
    
    private fun extractType(text: String): String? {
        return when {
            // HIGH type - expanded with 20+ variations
            text.contains("\\b(?:above|over|exceed|exceeded|exceeding|exceeds|higher|high|more\\s+than|greater\\s+than|greater|beyond|past|upwards?\\s+of|in\\s+excess\\s+of|surpass|surpassed|surpassing|top|topped|topping|beat|beaten|beating|outperform|outperformed|maximum|max|peak|peaked|peaking|spike|spiked|spiking|rise|rose|risen|rising|increase|increased|increasing|elevate|elevated|elevating|climb|climbed|climbing|jump|jumped|jumping|soar|soared|soaring)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "high"
            
            // LOW type - expanded with 20+ variations
            text.contains("\\b(?:below|under|less\\s+than|lower|low|fewer\\s+than|drops?|dropped|dropping|fall|fell|fallen|falling|decrease|decreased|decreasing|decline|declined|declining|reduce|reduced|reducing|dip|dipped|dipping|plunge|plunged|plunging|sink|sank|sunk|sinking|minimum|min|bottom|bottomed|bottoming|down|downward|descend|descended|descending|tumble|tumbled|tumbling|slump|slumped|slumping)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "low"
            
            else -> null
        }
    }
    
    private fun extractPeriod(text: String): String? {
        val periods = mapOf(
            "daily" to "\\b(?:daily|day|days|every\\s+day|each\\s+day|per\\s+day|a\\s+day|one\\s+day|single\\s+day|today|everyday|day\\s+by\\s+day|day-by-day|day\\s+to\\s+day|day-to-day|24\\s+hours|24\\s+hour|twenty\\s+four\\s+hours|round\\s+the\\s+clock|all\\s+day|whole\\s+day|throughout\\s+the\\s+day|during\\s+the\\s+day|daytime|diurnal)\\b",
            
            "weekly" to "\\b(?:weekly|week|weeks|every\\s+week|each\\s+week|per\\s+week|a\\s+week|one\\s+week|single\\s+week|this\\s+week|week\\s+by\\s+week|week-by-week|week\\s+to\\s+week|week-to-week|7\\s+days|seven\\s+days|hebdomadal|once\\s+a\\s+week|twice\\s+a\\s+week|throughout\\s+the\\s+week|during\\s+the\\s+week|all\\s+week|whole\\s+week|weeklong)\\b",
            
            "monthly" to "\\b(?:monthly|month|months|every\\s+month|each\\s+month|per\\s+month|a\\s+month|one\\s+month|single\\s+month|this\\s+month|month\\s+by\\s+month|month-by-month|month\\s+to\\s+month|month-to-month|30\\s+days|thirty\\s+days|once\\s+a\\s+month|throughout\\s+the\\s+month|during\\s+the\\s+month|all\\s+month|whole\\s+month|monthlong)\\b"
        )
        
        // Sort by pattern length for more specific matching
        val sortedPeriods = periods.entries.sortedByDescending { it.value.length }
        
        for ((period, pattern) in sortedPeriods) {
            if (text.contains(pattern.toRegex(RegexOption.IGNORE_CASE))) {
                return period
            }
        }
        
        return null
    }
    
    private fun extractEventType(text: String): String? {
        return when {
            // Weight - expanded with 20+ variations
            text.contains("\\b(?:weight|weigh|weighing|weighed|kg|kgs|kilogram|kilograms|kilogramme|kilogrammes|kilo|kilos|pounds?|lbs?|lb|body\\s+weight|body\\s+mass|mass|scale|scales|weighing\\s+scale|weight\\s+scale|heavy|heaviness|light|lightness|bmi|body\\s+mass\\s+index|stone|st|grams?|grammes?|g|ounces?|oz|#)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "weight"
            
            // Menstrual cycle - expanded with 20+ variations
            text.contains("\\b(?:menstrual|menstruation|menstruating|menstruate|period|periods|cycle|cycles|monthly\\s+cycle|time\\s+of\\s+month|that\\s+time|aunt\\s+flo|flow|bleeding|spotting|pms|premenstrual|ovulation|ovulating|ovulate|fertile|fertility|fertility\\s+window|luteal\\s+phase|follicular\\s+phase|cramping|cramps|menses|feminine\\s+hygiene|menstrual\\s+health|reproductive\\s+cycle)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "menstrual cycle"
            
            else -> null
        }
    }
    
    private fun addContextualSlots(text: String, intent: String, slots: MutableMap<String, Any>) {
        when (intent) {
            "QueryPoint" -> {
                if (!slots.containsKey("metric")) {
                    val inferredMetric = inferMetricFromContext(text)
                    if (inferredMetric != null) {
                        slots["metric"] = inferredMetric
                    }
                }
                if (!slots.containsKey("time_ref")) {
                    val timeRef = extractTimeRef(text) ?: "today"
                    slots["time_ref"] = timeRef
                }
                if (!slots.containsKey("unit")) {
                    val unit = extractUnit(text)
                    if (unit != null) {
                        slots["unit"] = unit
                    }
                }
                if (!slots.containsKey("qualifier")) {
                    // Qualifier not required. if needed, use extractQualifier function
                }
                if (!slots.containsKey("identifier")) {
                    val identifier = extractIdentifier(text) ?: "average"
                    slots["identifier"] = identifier
                }
            }
            "SetGoal" -> {
                if (!slots.containsKey("metric")) {
                    val inferredMetric = inferMetricFromContext(text)
                    if (inferredMetric != null) {
                        slots["metric"] = inferredMetric
                    }
                }
                if (!slots.containsKey("target")) {
                    val target = extractTarget(text)
                    if (target != null) {
                        slots["target"] = target
                    }
                }
                if (!slots.containsKey("unit")) {
                    val unit = extractUnit(text)
                    if (unit != null) {
                        slots["unit"] = unit
                    }
                }
                if (!slots.containsKey("period")) {
                    val period = extractPeriod(text) ?: "daily"
                    slots["period"] = period
                }
            }
            "SetThreshold" -> {
                if (!slots.containsKey("metric")) {
                    val inferredMetric = inferMetricFromContext(text)
                    if (inferredMetric != null) {
                        slots["metric"] = inferredMetric
                    }
                }
                if (!slots.containsKey("threshold")) {
                    val threshold = extractThreshold(text)
                    if (threshold != null) {
                        slots["threshold"] = threshold
                    }
                }
                if (!slots.containsKey("type")) {
                    val type = extractType(text)
                    if (type != null) {
                        slots["type"] = type
                    }
                }
                if (!slots.containsKey("unit")) {
                    val unit = extractUnit(text)
                    if (unit != null) {
                        slots["unit"] = unit
                    }
                }
            }
            "TimerStopwatch" -> {
                if (!slots.containsKey("tool")) {
                    val tool = extractTool(text)
                    if (tool != null) {
                        slots["tool"] = tool
                    }
                }
                if (!slots.containsKey("action")) {
                    val action = extractTimerAction(text)
                    if (action != null) {
                        slots["action"] = action
                    }
                }
                if (!slots.containsKey("value")) {
                    val value = extractValue(text, intent)
                    if (value != null) {
                        slots["value"] = value
                    }
                }
            }
            "ToggleFeature" -> {
                if (!slots.containsKey("feature")) {
                    val feature = extractFeature(text)
                    if (feature != null) {
                        slots["feature"] = feature
                    }
                }
                if (!slots.containsKey("state")) {
                    val state = extractState(text)
                    if (state != null) {
                        slots["state"] = state
                    }
                }
            }
            "LogEvent" -> {
                if (!slots.containsKey("event_type")) {
                    val eventType = extractEventType(text)
                    if (eventType != null) {
                        slots["event_type"] = eventType
                    }
                }
                if (!slots.containsKey("value")) {
                    val value = extractValue(text, intent)
                    if (value != null) {
                        slots["value"] = value
                    }
                }
                if (!slots.containsKey("unit")) {
                    val unit = extractUnit(text)
                    if (unit != null) {
                        slots["unit"] = unit
                    }
                }
                if (!slots.containsKey("time_ref")) {
                    val timeRef = extractTimeRef(text) ?: "today"
                    slots["time_ref"] = timeRef
                }
            }
            "StartActivity", "StopActivity" -> {
                if (!slots.containsKey("activity_type")) {
                    val activityType = extractActivityType(text)
                    if (activityType != null) {
                        slots["activity_type"] = activityType
                        Log.d(LOG_TAG, "  ✓ Added activity_type contextually: $activityType")
                    }
                }
                if (!slots.containsKey("activity_number")) {
                    val activityNumber = extractActivityNumber(text)
                    if (activityNumber != null) {
                        slots["activity_number"] = activityNumber
                        Log.d(LOG_TAG, "  ✓ Added activity_number contextually: $activityNumber")
                    }
                }
            }
            "OpenApp" -> {
                if (!slots.containsKey("app")) {
                    val app = extractApp(text)
                    if (app != null) {
                        slots["app"] = app
                    }
                }
                if (!slots.containsKey("action")) {
                    val action = extractAppAction(text)
                    if (action != null) {
                        slots["action"] = action
                    }
                }
                if (!slots.containsKey("target")) {
                    // Target might be the same as app or something, but for now leave it
                }
            }
            "PhoneAction" -> {
                if (!slots.containsKey("action")) {
                    val action = extractPhoneAction(text)
                    if (action != null) {
                        slots["action"] = action
                    }
                }
                if (!slots.containsKey("contact")) {
                    val contact = extractContact(text)
                    if (contact != null) {
                        slots["contact"] = contact
                    }
                }
            }
            "MediaAction" -> {
                if (!slots.containsKey("action")) {
                    val action = extractMediaAction(text)
                    if (action != null) {
                        slots["action"] = action
                    }
                }
                if (!slots.containsKey("target")) {
                    // Target could be media type, but for now leave it
                }
                if (!slots.containsKey("state")) {
                    val state = extractState(text)
                    if (state != null) {
                        slots["state"] = state
                    }
                }
            }
            "WeatherQuery" -> {
                if (!slots.containsKey("location")) {
                    val location = extractLocation(text)
                    if (location != null) {
                        slots["location"] = location
                    }
                }
                if (!slots.containsKey("attribute")) {
                    val attribute = extractAttribute(text)
                    if (attribute != null) {
                        slots["attribute"] = attribute
                    }
                }
                if (!slots.containsKey("time_ref")) {
                    val timeRef = extractTimeRef(text) ?: "today"
                    slots["time_ref"] = timeRef
                }
            }
            "QueryTrend" -> {
                if (!slots.containsKey("metric")) {
                    val inferredMetric = inferMetricFromContext(text)
                    if (inferredMetric != null) {
                        slots["metric"] = inferredMetric
                    }
                }
                if (!slots.containsKey("period")) {
                    val period = extractPeriod(text)
                    if (period != null) {
                        slots["period"] = period
                    }
                }
                if (!slots.containsKey("unit")) {
                    val unit = extractUnit(text)
                    if (unit != null) {
                        slots["unit"] = unit
                    }
                }
            }
        }
    }
    
    private fun inferMetricFromContext(text: String): String? {
        // Pre-compiled regex patterns for better performance
        val inferencePatterns = mapOf(
            "distance" to listOf(
                Regex("\\bhow\\s+(?:much|many)\\s+distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+far\\b|\\bhow\\s+long\\s+(?:of\\s+)?(?:a\\s+)?distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\bdistance.*(?:walk|walked|run|ran|travel|travelled|traveled|cover|covered)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many).*distance.*(?:walk|walked|run|ran|travel|travelled|traveled|cover|covered)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:walk|walked|walking|run|ran|running|jog|jogged|jogging|hike|hiked|hiking)\\s+(?:distance|far|length)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bkilometers?\\b|\\bkilometres?\\b|\\bmiles?\\b|\\bkm\\b|\\bmi\\b|\\bmeters?\\b|\\bmetres?\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many)\\s+(?:km|miles?|meters?)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:total|overall|entire)\\s+distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:covered|travelled|traveled)\\s+(?:distance|how\\s+far)\\b", RegexOption.IGNORE_CASE),
                Regex("\\brange\\b|\\bspan\\b|\\blength\\b|\\bmileage\\b", RegexOption.IGNORE_CASE),
                Regex("\\bjourney\\s+length\\b|\\btrip\\s+distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+long\\s+(?:of\\s+)?(?:a\\s+)?(?:walk|run|hike)\\b", RegexOption.IGNORE_CASE)
            ),
            "steps" to listOf(
                Regex("\\b(?:walk|walked|walking|stroll|strolling|strolled|hike|hiking|hiked|trek|trekking|trekked|march|marching|marched|wander|wandering|wandered|amble|ambling|ambled|pace|pacing|paced)\\b(?!\\s+(?:distance|far|km|miles?|kilometers?))", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many)(?!.*distance).*(?:walk|walked|stroll|hike|move|moved|step)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bsteps?\\b|\\bfootsteps?\\b|\\bfoot\\s+steps?\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:count|counting|total|number).*(?:steps?|walk)(?!.*distance)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:daily|today'?s|my)\\s+(?:steps?|walk|walking)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bstep\\s+(?:count|counter|goal|target|total)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:active|much\\s+activity)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bmovement\\b|\\bactivity\\s+level\\b", RegexOption.IGNORE_CASE),
                Regex("\\bgait\\b|\\btread\\b|\\bstride\\b", RegexOption.IGNORE_CASE)
            ),
            "heart rate" to listOf(
                Regex("\\bheart\\s+rate\\b|\\bheartrate\\b|\\bheart\\s+beat\\b|\\bheartbeat\\b", RegexOption.IGNORE_CASE),
                Regex("\\bpulse\\b|\\bpulse\\s+rate\\b|\\bheart\\s+rhythm\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhr\\b|\\bbpm\\b|\\bbeats\\s+per\\s+minute\\b|\\bbeats?\\b", RegexOption.IGNORE_CASE),
                Regex("\\bcardiac\\b|\\bcardiac\\s+rate\\b|\\bcardiovascular\\b|\\bcardio\\b", RegexOption.IGNORE_CASE),
                Regex("\\bheart\\s+(?:health|monitor|sensor|zone|performance)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:resting|max|maximum|average|current)\\s+(?:heart\\s+rate|hr|pulse)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:fast|slow)\\s+(?:is\\s+)?(?:my\\s+)?heart\\b", RegexOption.IGNORE_CASE),
                Regex("\\bmy\\s+(?:heart|pulse|bpm|hr)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bheart\\s+(?:beating|pounding|racing|palpitating)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bticker\\b|\\bheart\\s+pump\\b", RegexOption.IGNORE_CASE)
            ),
            "calories" to listOf(
                Regex("\\bcalories?\\b|\\bcalorie\\s+(?:count|intake|burn|burned|burnt)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bkcal\\b|\\bcals?\\b|\\bkilocalories?\\b", RegexOption.IGNORE_CASE),
                Regex("\\benergy\\b|\\benergy\\s+(?:burn|burned|burnt|expenditure|spent)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bburn\\b|\\bburned\\b|\\bburnt\\b|\\bburning\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many)\\s+(?:calories?|energy|kcal)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:total|daily|today'?s)\\s+(?:calories?|energy|burn)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bcalorie\\s+(?:goal|target|deficit|surplus)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bfat\\s+burn\\b|\\bfat\\s+burning\\b|\\bmetabolism\\b", RegexOption.IGNORE_CASE),
                Regex("\\benergy\\s+(?:consumption|used|expended)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+much\\s+(?:did\\s+I\\s+)?burn\\b", RegexOption.IGNORE_CASE)
            ),
            "sleep" to listOf(
                Regex("\\bsleep\\b|\\bslept\\b|\\bsleeping\\b|\\basleep\\b", RegexOption.IGNORE_CASE),
                Regex("\\brest\\b|\\brested\\b|\\bresting\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|long|well).*sleep\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:last\\s+)?night'?s\\s+sleep\\b|\\btonight'?s\\s+sleep\\b", RegexOption.IGNORE_CASE),
                Regex("\\bsleep\\s+(?:time|duration|hours|quality|data|tracking|pattern)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:total|nightly)\\s+sleep\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhours?\\s+(?:of\\s+)?sleep\\b|\\bslept\\s+(?:for\\s+)?\\d+\\s+hours?\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:deep|light|rem)\\s+sleep\\b", RegexOption.IGNORE_CASE),
                Regex("\\bnap\\b|\\bnapped\\b|\\bnapping\\b|\\bsnooze\\b", RegexOption.IGNORE_CASE),
                Regex("\\bsleep\\s+(?:score|rating|efficiency|quality)\\b", RegexOption.IGNORE_CASE)
            ),
            "weight" to listOf(
                Regex("\\bweight\\b|\\bweigh\\b|\\bweighing\\b|\\bweighed\\b", RegexOption.IGNORE_CASE),
                Regex("\\bkg\\b|\\bkgs\\b|\\bkilograms?\\b|\\bpounds?\\b|\\blbs?\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+much\\s+(?:do\\s+I\\s+)?weigh\\b|\\bwhat'?s\\s+my\\s+weight\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:body\\s+)?(?:mass|weight)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bscale\\b|\\bweighing\\s+scale\\b|\\bweight\\s+scale\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:current|today'?s|my)\\s+weight\\b", RegexOption.IGNORE_CASE),
                Regex("\\bweight\\s+(?:loss|gain|change|goal|target|tracking)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bbmi\\b|\\bbody\\s+mass\\s+index\\b", RegexOption.IGNORE_CASE),
                Regex("\\bheavy\\b|\\blight\\b|\\bhow\\s+heavy\\b", RegexOption.IGNORE_CASE)
            ),
            "stress" to listOf(
                Regex("\\bstress\\b|\\bstressed\\b|\\bstressful\\b|\\bstress\\s+level\\b", RegexOption.IGNORE_CASE),
                Regex("\\banxiety\\b|\\banxious\\b|\\bworried\\b|\\bworry\\b", RegexOption.IGNORE_CASE),
                Regex("\\btension\\b|\\btense\\b|\\bpressure\\b|\\bstrain\\b", RegexOption.IGNORE_CASE),
                Regex("\\boverwhelm\\b|\\boverwhelmed\\b|\\bburnout\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:stressed|anxious|tense|overwhelmed)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:mental|emotional)\\s+(?:stress|health|state)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bstress\\s+(?:score|index|management|reduction)\\b", RegexOption.IGNORE_CASE),
                Regex("\\brelaxation\\b|\\brelaxed\\b|\\bcalm\\b|\\bpeaceful\\b", RegexOption.IGNORE_CASE),
                Regex("\\bmindfulness\\b|\\bzen\\b", RegexOption.IGNORE_CASE)
            ),
            "spo2" to listOf(
                Regex("\\bspo2\\b|\\bsp\\s+o2\\b|\\bsp2\\b|\\bsp\\s+2\\b|\\bo2\\b|\\boxygen\\b", RegexOption.IGNORE_CASE),
                Regex("\\bblood\\s+oxygen\\b|\\boxygen\\s+(?:level|saturation|sat)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:pulse\\s+)?ox(?:imeter)?\\b|\\boximetry\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|high|low)\\s+(?:is\\s+)?(?:my\\s+)?oxygen\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:blood\\s+)?o2\\s+(?:level|saturation|sat)\\b", RegexOption.IGNORE_CASE),
                Regex("\\boxygen\\s+(?:reading|measurement|sensor)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bbreath\\b|\\bbreathing\\b|\\brespiratory\\b", RegexOption.IGNORE_CASE)
            ),
            "active hours" to listOf(
                Regex("\\bactive\\s+hours?\\b|\\bactivity\\s+hours?\\b|\\bhours?\\s+active\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many|long).*(?:active|activity).*hours?\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhours?\\s+(?:of\\s+)?(?:activity|active\\s+time|movement)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:daily|today'?s|total)\\s+active\\s+(?:time|hours?|duration)\\b", RegexOption.IGNORE_CASE),
                Regex("\\btime\\s+(?:spent\\s+)?active\\b|\\bactive\\s+time\\b|\\bactivity\\s+time\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+active.*(?:today|been)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bmoving\\s+(?:time|hours?|duration)\\b|\\btime\\s+moving\\b", RegexOption.IGNORE_CASE),
                Regex("\\bactive\\s+(?:duration|period|minutes?)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bphysical\\s+activity\\s+(?:time|hours?|duration)\\b", RegexOption.IGNORE_CASE)
            )
        )
        
        // Sort by specificity (more specific patterns first)
        val sortedMetrics = inferencePatterns.entries.sortedByDescending { 
            it.value.sumOf { pattern -> pattern.pattern.length }
        }
        
        for ((metric, patterns) in sortedMetrics) {
            for (pattern in patterns) {
                if (pattern.containsMatchIn(text)) {
                    return metric
                }
            }
        }
        
        return null
    }
}