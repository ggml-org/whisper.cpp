package com.whispercppdemo.intent

import android.util.Log
import org.json.JSONObject
import java.util.*
import java.util.regex.Pattern

private const val LOG_TAG = "SlotExtractor"

data class SlotExtractionResult(
    val slots: Map<String, Any>,
    val confidence: Float
)

class SlotExtractor {
    
    private val intentSlotTemplates = mapOf(
        "QueryPoint" to listOf("metric", "time_ref", "unit", "qualifier"),
        "SetGoal" to listOf("metric", "target", "unit"),
        "SetThreshold" to listOf("metric", "threshold", "type", "unit"),
        "TimerStopwatch" to listOf("tool", "action", "value"),
        "ToggleFeature" to listOf("feature", "state"),
        "LogEvent" to listOf("event_type", "value", "unit"),
        "StartActivity" to listOf("activity_type"),
        "StopActivity" to listOf("activity_type"),
        "OpenApp" to listOf("app", "action", "target"),
        "PhoneAction" to listOf("action", "contact"),
        "MediaAction" to listOf("action", "target"),
        "WeatherQuery" to listOf("location", "attribute"),
        "QueryTrend" to listOf("metric", "period", "unit")
    )
    
    // Synonym mappings based on Python implementation
    private val metricSynonyms = mapOf(
        "steps" to listOf("steps", "step", "walk", "walked", "walking", "footsteps", "pace"),
        "distance" to listOf("distance", "walked", "walk", "miles", "kilometers", "km", "far"),
        "calories" to listOf("calories", "calorie", "kcal", "energy", "burned", "burn"),
        "heart rate" to listOf("heart rate", "heartrate", "hr", "pulse", "bpm", "heart beat", "heartbeat"),
        "sleep" to listOf("sleep", "slept", "sleeping", "rest", "rested"),
        "sleep score" to listOf("sleep score", "sleep quality", "sleep rating"),
        "spo2" to listOf("spo2", "oxygen", "blood oxygen", "o2", "saturation"),
        "weight" to listOf("weight", "weigh", "kg", "pounds", "lbs"),
        "stress" to listOf("stress", "stressed", "anxiety", "tension")
    )
    
    private val timeSynonyms = mapOf(
        "today" to listOf("today", "now", "currently", "this day", "present"),
        "yesterday" to listOf("yesterday", "last day"),
        "last night" to listOf("last night", "night", "overnight", "during sleep"),
        "this morning" to listOf("this morning", "morning", "am"),
        "this week" to listOf("this week", "current week", "weekly"),
        "last week" to listOf("last week", "past week", "previous week"),
        "this month" to listOf("this month", "current month", "monthly")
    )
    
    private val qualifierSynonyms = mapOf(
        "minimum" to listOf("minimum", "min", "lowest", "least", "bottom"),
        "maximum" to listOf("maximum", "max", "highest", "most", "peak", "top"),
        "average" to listOf("average", "avg", "mean", "typical", "normal"),
        "total" to listOf("total", "sum", "overall", "complete", "entire")
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
            else -> null
        }
    }
    
    private fun extractMetric(processedText: String, originalText: String): String? {
        // Direct synonym matching on processed text first
        for ((metric, synonyms) in metricSynonyms) {
            for (synonym in synonyms) {
                if (processedText.contains("\\b${Regex.escape(synonym)}\\b".toRegex())) {
                    return metric
                }
            }
        }
        
        // Fallback to original text
        for ((metric, synonyms) in metricSynonyms) {
            for (synonym in synonyms) {
                if (originalText.contains("\\b${Regex.escape(synonym)}\\b".toRegex())) {
                    return metric
                }
            }
        }
        
        // Context-based inference
        if (originalText.contains("\\b(?:walk|walked|walking)\\b".toRegex())) {
            return if (originalText.contains("\\b(?:far|distance|km|miles?)\\b".toRegex())) {
                "distance"
            } else {
                "steps"
            }
        }
        
        return null
    }
    
    private fun extractTimeRef(text: String): String? {
        val timePatterns = mapOf(
            "last night" to "\\blast\\s+night\\b|\\bduring\\s+(?:the\\s+)?night\\b|\\bovernight\\b",
            "yesterday" to "\\byesterday\\b(?!\\s+night)",
            "yesterday morning" to "\\byesterday\\s+morning\\b",
            "yesterday afternoon" to "\\byesterday\\s+afternoon\\b",
            "yesterday evening" to "\\byesterday\\s+evening\\b",
            "today" to "\\btoday\\b|\\bnow\\b|\\bcurrently\\b|\\bthis\\s+day\\b",
            "this morning" to "\\bthis\\s+morning\\b|\\bmorning\\b",
            "this afternoon" to "\\bthis\\s+afternoon\\b|\\bafternoon\\b",
            "this evening" to "\\bthis\\s+evening\\b|\\bevening\\b",
            "this week" to "\\bthis\\s+week\\b|\\bcurrent\\s+week\\b|\\bweekly\\b",
            "last week" to "\\blast\\s+week\\b|\\bpast\\s+week\\b|\\bprevious\\s+week\\b",
            "this month" to "\\bthis\\s+month\\b|\\bcurrent\\s+month\\b|\\bmonthly\\b"
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
            "minimum" to "\\b(?:minimum|min|lowest|least|bottom|smallest)\\b",
            "maximum" to "\\b(?:maximum|max|highest|most|peak|top|largest)\\b",
            "average" to "\\b(?:average|avg|mean|typical|normal)\\b",
            "total" to "\\b(?:total|sum|overall|complete|entire|all)\\b"
        )
        
        for ((qualifier, pattern) in qualifierPatterns) {
            if (text.contains(pattern.toRegex())) {
                return qualifier
            }
        }
        
        return null
    }
    
    private fun extractUnit(text: String): String? {
        val unitPatterns = mapOf(
            "bpm" to "\\b(?:bpm|beats?\\s+per\\s+minute)\\b",
            "kg" to "\\b(?:kg|kilogram|kgs)\\b",
            "pounds" to "\\b(?:pounds?|lbs?|lb)\\b",
            "km" to "\\b(?:km|kilometer|kilometres?)\\b",
            "miles" to "\\b(?:miles?|mi)\\b",
            "kcal" to "\\b(?:kcal|calories?)\\b",
            "hours" to "\\b(?:hours?|hrs?|h)\\b",
            "minutes" to "\\b(?:min|minutes?|mins)\\b",
            "percent" to "\\b(?:percent|%)\\b",
            "count" to "\\bsteps?\\b"
        )
        
        for ((unit, pattern) in unitPatterns) {
            if (text.contains(pattern.toRegex())) {
                return unit
            }
        }
        
        // Context-based unit inference
        return when {
            text.contains("\\b(?:heart\\s+rate|pulse|hr)\\b".toRegex()) -> "bpm"
            text.contains("\\b(?:weight|weigh)\\b".toRegex()) -> "kg"
            text.contains("\\bsteps?\\b".toRegex()) -> "count"
            else -> null
        }
    }
    
    private fun extractThreshold(text: String): Int? {
        // Look for numbers in context
        val numberPatterns = listOf(
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:bpm|kg|km|miles?|percent|%|hours?|minutes?)\\b",
            "\\b(?:above|over|exceeds?|higher\\s+than)\\s+(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:below|under|less\\s+than|lower\\s+than)\\s+(\\d+(?:\\.\\d+)?)\\b"
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
        // Look for goal-setting patterns
        val goalPatterns = listOf(
            "\\b(?:goal|target|aim).*?(\\d+(?:\\.\\d+)?)\\b",
            "\\b(?:set|change|update).*?(?:to|at)\\s*(\\d+(?:\\.\\d+)?)\\b",
            "\\b(\\d+(?:\\.\\d+)?)\\s*(?:steps?|kg|km|hours?|minutes?|calories?)\\b"
        )
        
        for (pattern in goalPatterns) {
            val match = pattern.toRegex().find(text)
            if (match != null) {
                return match.groupValues[1].toDoubleOrNull()?.toInt()
            }
        }
        
        val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
        return numbers.firstOrNull()?.groupValues?.get(1)?.toDoubleOrNull()?.toInt()
    }
    
    private fun extractValue(text: String, intent: String): Any? {
        return when (intent) {
            "LogEvent" -> {
                val weightPattern = "\\b(\\d+(?:\\.\\d+)?)\\s*(?:kg|pounds?|lbs?)\\b".toRegex()
                val match = weightPattern.find(text)
                match?.groupValues?.get(1)?.toDoubleOrNull()
            }
            "TimerStopwatch" -> {
                val timePatterns = listOf(
                    "\\b(\\d{1,2}(?::\\d{2})?(?:\\s*[ap]m)?)\\b",
                    "\\b(\\d+)\\s*(?:min|minutes?)\\b",
                    "\\b(\\d+)\\s*(?:hours?|hrs?)\\b"
                )
                
                for (pattern in timePatterns) {
                    val match = pattern.toRegex().find(text)
                    if (match != null) {
                        return match.groupValues[1]
                    }
                }
                
                val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
                numbers.firstOrNull()?.groupValues?.get(1)?.toIntOrNull()
            }
            else -> {
                val numbers = "\\b(\\d+(?:\\.\\d+)?)\\b".toRegex().findAll(text)
                numbers.firstOrNull()?.groupValues?.get(1)?.toIntOrNull()
            }
        }
    }
    
    private fun extractFeature(text: String): String? {
        val features = mapOf(
            "do not disturb" to "\\b(?:do\\s+not\\s+disturb|dnd|silent\\s+mode)\\b",
            "AOD" to "\\b(?:AOD|always\\s+on\\s+display|always-on)\\b",
            "raise to wake" to "\\b(?:raise\\s+to\\s+wake|lift\\s+to\\s+wake|tap\\s+to\\s+wake)\\b",
            "vibration" to "\\b(?:vibration|vibrate|haptic)\\b",
            "brightness" to "\\b(?:brightness|screen\\s+brightness)\\b",
            "volume" to "\\b(?:volume|sound\\s+level)\\b"
        )
        
        for ((feature, pattern) in features) {
            if (text.contains(pattern.toRegex())) {
                return feature
            }
        }
        
        return null
    }
    
    private fun extractState(text: String): String? {
        return when {
            text.contains("\\b(?:turn\\s+on|enable|activate|switch\\s+on|start)\\b".toRegex()) -> "on"
            text.contains("\\b(?:turn\\s+off|disable|deactivate|switch\\s+off|stop)\\b".toRegex()) -> "off"
            text.contains("\\b(?:increase|up|higher|raise)\\b".toRegex()) -> "increase"
            text.contains("\\b(?:decrease|down|lower|reduce)\\b".toRegex()) -> "decrease"
            else -> null
        }
    }
    
    private fun extractAction(text: String): String? {
        val actions = mapOf(
            "set" to "\\b(?:set|setup|configure)\\b",
            "start" to "\\b(?:start|begin|initiate|launch)\\b",
            "stop" to "\\b(?:stop|end|finish|terminate)\\b",
            "call" to "\\b(?:call|phone|dial)\\b",
            "message" to "\\b(?:message|text|sms|send)\\b",
            "open" to "\\b(?:open|launch|start|show)\\b",
            "check" to "\\b(?:check|verify|examine|look)\\b",
            "measure" to "\\b(?:measure|test|record)\\b"
        )
        
        for ((action, pattern) in actions) {
            if (text.contains(pattern.toRegex())) {
                return action
            }
        }
        
        return null
    }
    
    private fun extractTool(text: String): String? {
        return when {
            text.contains("\\b(?:alarm|wake\\s+up|wake\\s+me)\\b".toRegex()) -> "alarm"
            text.contains("\\b(?:timer|countdown)\\b".toRegex()) -> "timer"
            text.contains("\\b(?:stopwatch|chronometer)\\b".toRegex()) -> "stopwatch"
            else -> null
        }
    }
    
    private fun extractActivityType(text: String): String? {
        val activities = mapOf(
            "outdoor run" to "\\b(?:outdoor\\s+)?(?:run|running|jog|jogging)\\b",
            "indoor cycling" to "\\b(?:indoor\\s+)?(?:cycling|bike|biking)\\b",
            "swimming" to "\\b(?:swim|swimming|pool)\\b",
            "yoga" to "\\b(?:yoga|meditation|stretch)\\b",
            "walking" to "\\b(?:walk|walking|hike|hiking)\\b",
            "workout" to "\\b(?:workout|exercise|training|gym)\\b"
        )
        
        for ((activity, pattern) in activities) {
            if (text.contains(pattern.toRegex())) {
                return activity
            }
        }
        
        return null
    }
    
    private fun extractApp(text: String): String? {
        val apps = mapOf(
            "weather" to "\\b(?:weather|forecast|temperature|rain|snow)\\b",
            "settings" to "\\b(?:settings?|preferences|config)\\b",
            "health" to "\\b(?:health|fitness|medical)\\b",
            "calendar" to "\\b(?:calendar|schedule|appointment)\\b"
        )
        
        for ((app, pattern) in apps) {
            if (text.contains(pattern.toRegex())) {
                return app
            }
        }
        
        return null
    }
    
    private fun extractContact(text: String): String? {
        val contacts = mapOf(
            "mom" to "\\b(?:mom|mother|mama|mum)\\b",
            "dad" to "\\b(?:dad|father|papa|pop)\\b",
            "sister" to "\\b(?:sister|sis)\\b",
            "brother" to "\\b(?:brother|bro)\\b"
        )
        
        for ((contact, pattern) in contacts) {
            if (text.contains(pattern.toRegex())) {
                return contact
            }
        }
        
        return null
    }
    
    private fun extractLocation(text: String): String? {
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
        
        return "current location"  // Default
    }
    
    private fun extractAttribute(text: String): String? {
        val attributes = mapOf(
            "forecast" to "\\b(?:forecast|prediction|outlook)\\b",
            "temperature" to "\\b(?:temperature|temp|hot|cold|warm|cool)\\b",
            "rain" to "\\b(?:rain|raining|shower|umbrella|wet)\\b",
            "humidity" to "\\b(?:humidity|humid|moisture)\\b",
            "air quality" to "\\b(?:air\\s+quality|aqi|pollution|smog)\\b"
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
            text.contains("\\b(?:above|over|exceed|higher|more\\s+than|greater)\\b".toRegex()) -> "high"
            text.contains("\\b(?:below|under|less\\s+than|lower|drops?)\\b".toRegex()) -> "low"
            else -> null
        }
    }
    
    private fun extractPeriod(text: String): String? {
        val periods = mapOf(
            "daily" to "\\b(?:daily|every\\s+day|each\\s+day)\\b",
            "weekly" to "\\b(?:weekly|every\\s+week|each\\s+week)\\b",
            "monthly" to "\\b(?:monthly|every\\s+month|each\\s+month)\\b"
        )
        
        for ((period, pattern) in periods) {
            if (text.contains(pattern.toRegex())) {
                return period
            }
        }
        
        return null
    }
    
    private fun extractEventType(text: String): String? {
        return when {
            text.contains("\\b(?:weight|weigh|kg|pounds)\\b".toRegex()) -> "weight"
            text.contains("\\b(?:menstrual|period|cycle)\\b".toRegex()) -> "menstrual cycle"
            else -> null
        }
    }
    
    private fun addContextualSlots(text: String, intent: String, slots: MutableMap<String, Any>) {
        // For QueryPoint intent, try to infer missing metric
        if (intent == "QueryPoint" && !slots.containsKey("metric")) {
            val inferredMetric = inferMetricFromContext(text)
            if (inferredMetric != null) {
                slots["metric"] = inferredMetric
            }
        }
        
        // Add qualifier if detected but not extracted
        if (intent == "QueryPoint" && !slots.containsKey("qualifier")) {
            val qualifier = extractQualifier(text)
            if (qualifier != null) {
                slots["qualifier"] = qualifier
            }
        }
    }
    
    private fun inferMetricFromContext(text: String): String? {
        val inferencePatterns = mapOf(
            "steps" to listOf(
                "\\b(?:walk|walked|walking)\\b(?!\\s+distance)",
                "\\bhow\\s+much.*(?:walk|walked)\\b",
                "\\bsteps?\\b"
            ),
            "distance" to listOf(
                "\\bhow\\s+far\\b",
                "\\b(?:walk|walked|walking)\\s+(?:distance|far)\\b",
                "\\bdistance.*(?:walk|walked)\\b",
                "\\bkilometers?\\b|\\bmiles?\\b|\\bkm\\b"
            ),
            "heart rate" to listOf(
                "\\bheart\\s+rate\\b|\\bheartrate\\b|\\bpulse\\b|\\bhr\\b|\\bbpm\\b"
            ),
            "calories" to listOf(
                "\\bcalories?\\b|\\bkcal\\b|\\benergy\\b|\\bburn\\b"
            )
        )
        
        for ((metric, patterns) in inferencePatterns) {
            for (pattern in patterns) {
                if (text.contains(pattern.toRegex())) {
                    return metric
                }
            }
        }
        
        return null
    }
}