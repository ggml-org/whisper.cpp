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
    
    private val intentSlotTemplates = mapOf(
        "QueryPoint" to listOf("metric"),  // time_ref and unit can have defaults
        "SetGoal" to listOf("metric", "target"),  // unit can be inferred
        "SetThreshold" to listOf("metric", "threshold", "type"),  // unit can be inferred
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
            "threshold" -> extractThreshold(originalText)
            "target" -> extractTarget(originalText)
            "value" -> extractValue(originalText, intent)
            "feature" -> extractFeature(originalText)
            "state" -> extractState(originalText)
            "action" -> extractAction(originalText)
            "tool" -> extractTool(originalText)
            "activity_type" -> extractActivityType(originalText)
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
    
    private fun extractUnit(text: String): String? {
        // Try context-based unit inference first
        when {
            heartRateUnitRegex.containsMatchIn(text) -> return "bpm"
            stressRegex.containsMatchIn(text) -> return "score"
            oxygenRegex.containsMatchIn(text) -> return "percent"
            sleepRegex.containsMatchIn(text) -> return "hours"
            sleepQualityRegex.containsMatchIn(text) -> return "score"
            distanceRegex.containsMatchIn(text) -> return "distance"
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
            // Goal/Target/Aim patterns - expanded
            "\\b(?:goal|target|aim|objective|plan|intention|aspiration|ambition|desire|want|wish|hope)\\s*(?:is|of|to|for|at)?\\s*(?:be|reach|hit|achieve|get|make|do)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Set/Change/Update patterns - expanded
            "\\b(?:set|change|update|modify|adjust|edit|configure|make|establish|create|define|specify)\\s*(?:my|the)?\\s*(?:goal|target|aim|objective)?\\s*(?:to|at|as|for)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Action + number + unit pattern - expanded
            "\\b(?:reach|hit|achieve|attain|get|get\\s+to|make|do|complete|finish|accomplish|meet)\\s*(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|kilogram|kilograms|pounds?|lbs?|km|kms|kilometer|kilometers|miles?|hours?|hrs?|minutes?|mins?|calories?|kcal|bpm)\\b",
            
            // Number + unit pattern - expanded
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|kgs|kilogram|kilograms|pounds?|lbs?|lb|km|kms|kilometer|kilometers|kilometre|kilometres|miles?|mi|hours?|hrs?|hr|h|minutes?|mins?|min|m|calories?|kcal|cal|cals|bpm|beats?|meter|meters|metre|metres|feet|foot|ft)\\b",
            
            // "I want to" patterns - NEW
            "\\b(?:I|i)\\s+(?:want|wanna|need|must|should|have\\s+to|got\\s+to|gotta)\\s+(?:to\\s+)?(?:reach|hit|get|do|achieve|make|walk|run|burn|lose|gain|sleep)\\s*(?:to|at)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // "Trying to" patterns - NEW
            "\\b(?:trying|attempting|aiming|working|striving|shooting|going)\\s+(?:to|for)\\s+(?:reach|hit|get|do|achieve|make)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Daily/Weekly goal patterns - NEW
            "\\b(?:daily|weekly|monthly|per\\s+day|each\\s+day|every\\s+day)\\s+(?:goal|target|aim|objective)?\\s*(?:is|of|to)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Increase/Decrease to patterns - NEW
            "\\b(?:increase|raise|boost|bump|up|improve|decrease|reduce|lower|drop|cut|bring\\s+down)\\s+(?:to|by)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Minimum/Maximum patterns - NEW
            "\\b(?:at\\s+least|minimum\\s+of|no\\s+less\\s+than|minimum|min)\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // "Need to be" patterns - NEW
            "\\b(?:need|needs)\\s+to\\s+(?:be|reach|hit|get)\\s*(?:at|to)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Suggestion patterns - NEW
            "\\b(?:suggest|recommend|advise|tell\\s+me|remind\\s+me|notify\\s+me).*?(?:when|if|at)\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Limit patterns - NEW
            "\\b(?:limit|cap|max|maximum|ceiling|upper\\s+limit)\\s*(?:of|to|at|is)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Challenge patterns - NEW
            "\\b(?:challenge|dare|bet|see\\s+if).*?(?:to\\s+)?(?:reach|hit|do|get|make)\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // "X or more/less" patterns - NEW
            "\\b(\\d+(?:\\.\\d+)?)\\s+(?:or\\s+(?:more|above|over|higher|greater|less|below|under|lower|fewer))\\b",
            
            // Alert/Notify patterns - NEW
            "\\b(?:alert|notify|tell|remind|ping|warn|let\\s+me\\s+know).*?(?:when|if|at|after|once).*?(\\d+(?:\\.\\d+)?)\\b",
            
            // Threshold patterns - NEW
            "\\b(?:threshold|cutoff|mark|milestone|benchmark)\\s*(?:of|at|is)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // "Should be" patterns - NEW
            "\\b(?:should|must|ought\\s+to|supposed\\s+to)\\s+(?:be|reach|hit|get)\\s*(?:at|to)?\\s*(\\d+(?:\\.\\d+)?)\\b",
            
            // Track patterns - NEW
            "\\b(?:track|monitor|watch|follow|check).*?(?:until|till|to|up\\s+to)\\s*(\\d+(?:\\.\\d+)?)\\b"
        )
        
        // Try each pattern
        for (pattern in goalPatterns) {
            val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
            if (match != null && match.groupValues.size > 1) {
                val value = match.groupValues[1].toDoubleOrNull()?.toInt()
                if (value != null && value > 0) {
                    return value
                }
            }
        }
    
        // Fallback: extract any number from the text
        val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
        return numbers.firstOrNull()?.groupValues?.get(1)?.toDoubleOrNull()?.toInt()
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
                val timePatterns = listOf(
                    // Time with AM/PM - expanded
                    "\\b(\\d{1,2}(?::\\d{2})?(?:\\s*[ap]\\.?m\\.?))\\b",
                    
                    // Hours patterns - expanded
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:hours?|hrs?|hr|h|hour)\\b",
                    
                    // Minutes patterns - expanded
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|m)\\b",
                    
                    // Seconds patterns - NEW
                    "\\b(\\d+(?:\\.\\d+)?)\\s*(?:sec|secs|second|seconds|s)\\b",
                    
                    // Combined time patterns - NEW
                    "\\b(\\d+)\\s*(?:h|hr|hours?)\\s*(?:and\\s+)?(\\d+)\\s*(?:m|min|minutes?)\\b",
                    "\\b(\\d+)\\s*(?:m|min|minutes?)\\s*(?:and\\s+)?(\\d+)\\s*(?:s|sec|seconds?)\\b",
                    "\\b(\\d+):(\\d+):(\\d+)\\b", // HH:MM:SS format
                    "\\b(\\d+):(\\d+)\\b", // MM:SS or HH:MM format
                    
                    // Duration keywords - NEW
                    "\\b(?:for|during|lasting|takes?)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Timer-specific patterns - NEW
                    "\\b(?:set|start|begin|run|timer|stopwatch)\\s*(?:for|to|at)?\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Alarm-specific patterns - NEW
                    "\\b(?:alarm|wake|remind|alert)\\s*(?:at|for|in)?\\s*(\\d{1,2}(?::\\d{2})?(?:\\s*[ap]\\.?m\\.?)?)\\b",
                    
                    // "In X time" patterns - NEW
                    "\\b(?:in|after|within)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b",
                    
                    // Countdown patterns - NEW
                    "\\b(?:countdown|count\\s+down)\\s*(?:from|for)?\\s*(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|seconds?|secs?)\\b"
                )
                
                for (pattern in timePatterns) {
                    val match = pattern.toRegex(RegexOption.IGNORE_CASE).find(text)
                    if (match != null && match.groupValues.size > 1) {
                        // Handle combined time formats (e.g., "1h 30m" or "1:30")
                        if (match.groupValues.size > 2 && match.groupValues[2].isNotEmpty()) {
                            val hours = match.groupValues[1].toIntOrNull() ?: 0
                            val minutes = match.groupValues[2].toIntOrNull() ?: 0
                            return (hours * 60 + minutes).toString() // Return total minutes
                        }
                        return match.groupValues[1]
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
                    "\\b(\\d{1,2}(?::\\d{2})?(?:\\s*[ap]\\.?m\\.?)?)\\b",
                    
                    // "In X time" for reminders
                    "\\b(?:in|after)\\s+(\\d+(?:\\.\\d+)?)\\s*(?:min|mins|minute|minutes|hours?|hrs?|days?)\\b",
                    
                    // "At X" patterns
                    "\\b(?:at|by)\\s+(\\d{1,2}(?::\\d{2})?(?:\\s*[ap]\\.?m\\.?)?)\\b",
                    
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
            
            "brightness" to "\\b(?:brightness|screen\\s+brightness|display\\s+brightness|luminosity|backlight|screen\\s+light|light\\s+level|dim|dimness|brighten|darken|auto\\s+brightness|adaptive\\s+brightness|brightness\\s+level|screen\\s+intensity|display\\s+intensity|illumination|glow|radiance)\\b",
            
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
            text.contains("\\b(?:turn\\s+on|enable|enabled|enabling|activate|activated|activating|switch\\s+on|start|started|starting|power\\s+on|boot|boot\\s+up|fire\\s+up|launch|open|unmute|unmuted|resume|allow|permit|engage|engaged|engaging|set\\s+on|put\\s+on|make\\s+it\\s+on|get\\s+it\\s+on|bring\\s+up|wake\\s+up|light\\s+up|flip\\s+on)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "on"
            
            // OFF state - expanded with 20+ variations
            text.contains("\\b(?:turn\\s+off|disable|disabled|disabling|deactivate|deactivated|deactivating|switch\\s+off|stop|stopped|stopping|shut\\s+off|shut\\s+down|power\\s+off|kill|close|mute|muted|pause|paused|block|deny|disengage|disengaged|disengaging|set\\s+off|put\\s+off|make\\s+it\\s+off|get\\s+it\\s+off|bring\\s+down|sleep|suspend|flip\\s+off|cut\\s+off)\\b".toRegex(RegexOption.IGNORE_CASE)) -> "off"
            
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
            "set" to "\\b(?:set|setup|set\\s+up|configure|configuration|adjust|adjustment|change|modify|edit|customize|establish|define|specify|determine|fix|assign|allocate|program|preset|input|enter|put\\s+in|make\\s+it|arrange|organize|prepare)\\b",
            
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
            "outdoor run" to "\\b(?:outdoor\\s+)?(?:run|running|ran|jog|jogging|jogged|sprint|sprinting|sprinted|dash|dashing|race|racing|trail\\s+run|trail\\s+running|distance\\s+run|long\\s+run|short\\s+run|tempo\\s+run|interval\\s+run|fartlek|road\\s+run|cross\\s+country|marathon|half\\s+marathon|5k|10k|runner|runners)\\b",
            
            "indoor cycling" to "\\b(?:indoor\\s+)?(?:cycling|cycle|cycled|bike|biking|biked|bicycle|bicycling|spin|spinning|spin\\s+class|stationary\\s+bike|exercise\\s+bike|bike\\s+ride|pedal|pedaling|pedalled|indoor\\s+bike|cycle\\s+class|RPM|cadence\\s+training|peloton|zwift|virtual\\s+cycling|turbo\\s+trainer|trainer\\s+ride)\\b",
            
            "swimming" to "\\b(?:swim|swimming|swam|swum|swimmer|pool|lap|laps|freestyle|backstroke|breaststroke|butterfly|stroke|strokes|aquatic|water\\s+exercise|lap\\s+swimming|pool\\s+workout|open\\s+water|triathlon\\s+swim|swim\\s+training|water\\s+aerobics|aqua|diving|float|floating)\\b",
            
            "yoga" to "\\b(?:yoga|yogi|asana|asanas|meditation|meditate|meditating|meditated|stretch|stretching|stretched|flexibility|vinyasa|hatha|ashtanga|bikram|hot\\s+yoga|power\\s+yoga|yin\\s+yoga|restorative\\s+yoga|pranayama|breathing\\s+exercise|mindfulness|zen|namaste|downward\\s+dog|warrior\\s+pose|sun\\s+salutation|flow|yoga\\s+flow)\\b",
            
            "walking" to "\\b(?:walk|walking|walked|walker|stroll|strolling|strolled|hike|hiking|hiked|hiker|trek|trekking|trekked|ramble|rambling|wander|wandering|wandered|amble|ambling|march|marching|power\\s+walk|brisk\\s+walk|leisurely\\s+walk|nature\\s+walk|trail\\s+walk|hill\\s+walk|speed\\s+walk|fitness\\s+walk|evening\\s+walk|morning\\s+walk)\\b",
            
            "workout" to "\\b(?:workout|work\\s+out|worked\\s+out|exercise|exercising|exercised|training|train|trained|gym|gymnasium|fitness|fit|strength\\s+training|weight\\s+training|weightlifting|lift|lifting|cardio|HIIT|circuit\\s+training|crossfit|bootcamp|boot\\s+camp|calisthenics|bodyweight|resistance\\s+training|functional\\s+training|core\\s+workout|abs\\s+workout|upper\\s+body|lower\\s+body|full\\s+body)\\b"
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
    
    private fun extractApp(text: String): String? {
        val apps = mapOf(
            "weather" to "\\b(?:weather|forecast|forecasts|temperature|temp|temps|climate|conditions|rain|raining|rainy|rainfall|precipitation|snow|snowing|snowy|snowfall|sunny|sun|sunshine|cloud|clouds|cloudy|overcast|storm|stormy|thunder|thunderstorm|lightning|wind|windy|humidity|humid|fog|foggy|mist|misty|hail|sleet|tornado|hurricane|cyclone|meteorology|barometer|pressure)\\b",
            
            "settings" to "\\b(?:settings?|setting|preferences|preference|prefs|config|configuration|configure|options|option|control|controls|control\\s+panel|setup|set\\s+up|adjust|adjustment|customize|customization|personalize|personalization|system\\s+settings|device\\s+settings|app\\s+settings|change\\s+settings|modify\\s+settings|tweak|parameters)\\b",
            
            "health" to "\\b(?:health|healthy|healthcare|fitness|fit|medical|medicine|wellbeing|well-being|wellness|vitals|vital\\s+signs|workout|exercise|activity|activities|steps|weight|bmi|body\\s+mass\\s+index|calories|sleep|hydration|nutrition|diet|mental\\s+health|physical\\s+health|body|metrics)\\b",
            
            "calendar" to "\\b(?:calendar|calendars|schedule|schedules|scheduling|scheduled|appointment|appointments|meeting|meetings|event|events|agenda|agendas|planner|plan|plans|planning|date|dates|day\\s+planner|organizer|diary|diaries|booking|bookings|reservation|reservations|engagement|engagements|commitment|commitments|reminder|reminders|time\\s+slot|availability)\\b",
            
            "heart rate" to "\\b(?:heart\\s+rate|heartrate|heart\\s+beat|heartbeat|pulse|pulse\\s+rate|bpm|beats\\s+per\\s+minute|cardiac|cardiac\\s+rate|heart\\s+rhythm|resting\\s+heart\\s+rate|rhr|max\\s+heart\\s+rate|maximum\\s+heart\\s+rate|heart\\s+health|cardiovascular|cardio|ticker|heart\\s+monitor|heart\\s+sensor|hr|beat|beats|beating|palpitation|palpitations|tachycardia|bradycardia|heart\\s+zone|target\\s+heart\\s+rate|recovery\\s+heart\\s+rate)\\b",
            
            "blood oxygen" to "\\b(?:blood\\s+oxygen|oxygen|o2|spo2|sp\\s+o2|oxygen\\s+saturation|oxygen\\s+level|oxygen\\s+levels|blood\\s+o2|oxygen\\s+sat|o2\\s+sat|o2\\s+level|o2\\s+saturation|pulse\\s+ox|pulse\\s+oximetry|oximeter|oxygen\\s+reading|oxygen\\s+sensor|saturation|sat|blood\\s+oxygen\\s+level|arterial\\s+oxygen|respiratory|respiration|breathing|breath|lung\\s+function|oxygenation|hypoxia|oxygen\\s+content)\\b",
            
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
                    }
                }
                if (!slots.containsKey("time_ref")) {
                    slots["time_ref"] = "today"
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
            "steps" to listOf(
                Regex("\\b(?:walk|walked|walking|stroll|strolling|strolled|hike|hiking|hiked|trek|trekking|trekked|march|marching|marched|wander|wandering|wandered|amble|ambling|ambled|pace|pacing|paced)\\b(?!\\s+(?:distance|far|km|miles?|kilometers?))", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many).*(?:walk|walked|stroll|hike|move|moved|step)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bsteps?\\b|\\bfootsteps?\\b|\\bfoot\\s+steps?\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:count|counting|total|number).*(?:steps?|walk)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:daily|today'?s|my)\\s+(?:steps?|walk|walking)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bstep\\s+(?:count|counter|goal|target|total)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:active|much\\s+activity)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bmovement\\b|\\bactivity\\s+level\\b", RegexOption.IGNORE_CASE),
                Regex("\\bgait\\b|\\btread\\b|\\bstride\\b", RegexOption.IGNORE_CASE)
            ),
            "distance" to listOf(
                Regex("\\bhow\\s+far\\b|\\bhow\\s+much\\s+distance\\b|\\bhow\\s+long\\s+(?:of\\s+)?(?:a\\s+)?distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:walk|walked|walking|run|ran|running|jog|jogged|jogging|hike|hiked|hiking)\\s+(?:distance|far|length)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bdistance.*(?:walk|walked|run|ran|travel|travelled|traveled|cover|covered)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bkilometers?\\b|\\bkilometres?\\b|\\bmiles?\\b|\\bkm\\b|\\bmi\\b|\\bmeters?\\b|\\bmetres?\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|many)\\s+(?:km|miles?|meters?)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:total|overall|entire)\\s+distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:covered|travelled|traveled)\\s+(?:distance|how\\s+far)\\b", RegexOption.IGNORE_CASE),
                Regex("\\brange\\b|\\bspan\\b|\\blength\\b|\\bmileage\\b", RegexOption.IGNORE_CASE),
                Regex("\\bjourney\\s+length\\b|\\btrip\\s+distance\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+long\\s+(?:of\\s+)?(?:a\\s+)?(?:walk|run|hike)\\b", RegexOption.IGNORE_CASE)
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
                Regex("\\bspo2\\b|\\bsp\\s+o2\\b|\\bo2\\b|\\boxygen\\b", RegexOption.IGNORE_CASE),
                Regex("\\bblood\\s+oxygen\\b|\\boxygen\\s+(?:level|saturation|sat)\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:pulse\\s+)?ox(?:imeter)?\\b|\\boximetry\\b", RegexOption.IGNORE_CASE),
                Regex("\\bhow\\s+(?:much|high|low)\\s+(?:is\\s+)?(?:my\\s+)?oxygen\\b", RegexOption.IGNORE_CASE),
                Regex("\\b(?:blood\\s+)?o2\\s+(?:level|saturation|sat)\\b", RegexOption.IGNORE_CASE),
                Regex("\\boxygen\\s+(?:reading|measurement|sensor)\\b", RegexOption.IGNORE_CASE),
                Regex("\\bbreath\\b|\\bbreathing\\b|\\brespiratory\\b", RegexOption.IGNORE_CASE)
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