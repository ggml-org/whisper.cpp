import Foundation
import os.log

class SlotExtractor {
    
    // MARK: - Constants
    
    private static let logger = Logger(subsystem: "com.whispercpp.demo", category: "SlotExtractor")
    
    // MARK: - Slot Templates
    
    private let intentSlotTemplates: [String: [String]] = [
        "QueryPoint": ["metric", "time_ref", "unit", "qualifier"],
        "SetGoal": ["metric", "target", "unit"],
        "SetThreshold": ["metric", "threshold", "type", "unit"],
        "TimerStopwatch": ["tool", "action", "value"],
        "ToggleFeature": ["feature", "state"],
        "LogEvent": ["event_type", "value", "unit"],
        "StartActivity": ["activity_type"],
        "StopActivity": ["activity_type"],
        "OpenApp": ["app", "action", "target"],
        "PhoneAction": ["action", "contact"],
        "MediaAction": ["action", "target"],
        "WeatherQuery": ["location", "attribute"],
        "QueryTrend": ["metric", "period", "unit"]
    ]
    
    // MARK: - Synonym Mappings
    
    private let metricSynonyms: [String: [String]] = [
        "steps": ["steps", "step", "walk", "walked", "walking", "footsteps", "pace"],
        "distance": ["distance", "walked", "walk", "miles", "kilometers", "km", "far"],
        "calories": ["calories", "calorie", "kcal", "energy", "burned", "burn"],
        "heart rate": ["heart rate", "heartrate", "hr", "pulse", "bpm", "heart beat", "heartbeat"],
        "sleep": ["sleep", "slept", "sleeping", "rest", "rested"],
        "sleep score": ["sleep score", "sleep quality", "sleep rating"],
        "spo2": ["spo2", "oxygen", "blood oxygen", "o2", "saturation"],
        "weight": ["weight", "weigh", "kg", "pounds", "lbs"],
        "stress": ["stress", "stressed", "anxiety", "tension"]
    ]
    
    private let timeSynonyms: [String: [String]] = [
        "today": ["today", "now", "currently", "this day", "present"],
        "yesterday": ["yesterday", "last day"],
        "last night": ["last night", "night", "overnight", "during sleep"],
        "this morning": ["this morning", "morning", "am"],
        "this week": ["this week", "current week", "weekly"],
        "last week": ["last week", "past week", "previous week"],
        "this month": ["this month", "current month", "monthly"]
    ]
    
    private let qualifierSynonyms: [String: [String]] = [
        "minimum": ["minimum", "min", "lowest", "least", "bottom"],
        "maximum": ["maximum", "max", "highest", "most", "peak", "top"],
        "average": ["average", "avg", "mean", "typical", "normal"],
        "total": ["total", "sum", "overall", "complete", "entire"]
    ]
    
    // MARK: - Main Extraction Method
    
    func extractSlots(text: String, intent: String) async -> SlotExtractionResult {
        Self.logger.info("🏷️ Extracting slots for intent: \(intent), text: '\(text)'")
        
        var slots: [String: Any] = [:]
        let textLower = text.lowercased()
        
        // Get required slots for this intent
        let requiredSlots = intentSlotTemplates[intent] ?? []
        
        // Pre-process text
        let processedText = preprocessText(textLower)
        
        // Extract each required slot
        for slotName in requiredSlots {
            if let value = extractSingleSlot(processedText: processedText, originalText: textLower, slotName: slotName, intent: intent) {
                slots[slotName] = value
                Self.logger.info("  ✓ Extracted \(slotName): \(value)")
            }
        }
        
        // Add contextual slots
        addContextualSlots(text: textLower, intent: intent, slots: &slots)
        
        // Calculate confidence based on how many slots were extracted
        let confidence: Float = requiredSlots.isEmpty ? 1.0 : Float(slots.count) / Float(requiredSlots.count)
        
        Self.logger.info("🎯 Final slots: \(slots) (confidence: \(String(format: "%.2f", confidence)))")
        
        return SlotExtractionResult(slots: slots, confidence: confidence)
    }
    
    // MARK: - Text Preprocessing
    
    private func preprocessText(_ text: String) -> String {
        var processed = text
        
        // Normalize common variations
        processed = processed.replacingOccurrences(of: #"\bhow\s+much\s+did\s+i\s+walk"#, with: "walking distance", options: .regularExpression)
        processed = processed.replacingOccurrences(of: #"\bhow\s+many\s+steps"#, with: "steps", options: .regularExpression)
        processed = processed.replacingOccurrences(of: #"\bhow\s+far\s+did\s+i\s+walk"#, with: "walking distance", options: .regularExpression)
        processed = processed.replacingOccurrences(of: #"\bwhat\s+is\s+my"#, with: "my", options: .regularExpression)
        processed = processed.replacingOccurrences(of: #"\bshow\s+me\s+my"#, with: "my", options: .regularExpression)
        
        return processed
    }
    
    // MARK: - Single Slot Extraction
    
    private func extractSingleSlot(processedText: String, originalText: String, slotName: String, intent: String) -> Any? {
        switch slotName {
        case "metric":
            return extractMetric(processedText: processedText, originalText: originalText)
        case "time_ref":
            return extractTimeRef(text: originalText)
        case "unit":
            return extractUnit(text: originalText)
        case "qualifier":
            return extractQualifier(text: originalText)
        case "threshold":
            return extractThreshold(text: originalText)
        case "target":
            return extractTarget(text: originalText)
        case "value":
            return extractValue(text: originalText, intent: intent)
        case "feature":
            return extractFeature(text: originalText)
        case "state":
            return extractState(text: originalText)
        case "action":
            return extractAction(text: originalText)
        case "tool":
            return extractTool(text: originalText)
        case "activity_type":
            return extractActivityType(text: originalText)
        case "app":
            return extractApp(text: originalText)
        case "contact":
            return extractContact(text: originalText)
        case "location":
            return extractLocation(text: originalText)
        case "attribute":
            return extractAttribute(text: originalText)
        case "type":
            return extractType(text: originalText)
        case "period":
            return extractPeriod(text: originalText)
        case "event_type":
            return extractEventType(text: originalText)
        default:
            return nil
        }
    }
    
    // MARK: - Specific Extraction Methods
    
    private func extractMetric(processedText: String, originalText: String) -> String? {
        // Direct synonym matching on processed text first
        for (metric, synonyms) in metricSynonyms {
            for synonym in synonyms {
                let pattern = #"\b"# + NSRegularExpression.escapedPattern(for: synonym) + #"\b"#
                if processedText.range(of: pattern, options: .regularExpression) != nil {
                    return metric
                }
            }
        }
        
        // Fallback to original text
        for (metric, synonyms) in metricSynonyms {
            for synonym in synonyms {
                let pattern = #"\b"# + NSRegularExpression.escapedPattern(for: synonym) + #"\b"#
                if originalText.range(of: pattern, options: .regularExpression) != nil {
                    return metric
                }
            }
        }
        
        // Context-based inference
        if originalText.range(of: #"\b(?:walk|walked|walking)\b"#, options: .regularExpression) != nil {
            if originalText.range(of: #"\b(?:far|distance|km|miles?)\b"#, options: .regularExpression) != nil {
                return "distance"
            } else {
                return "steps"
            }
        }
        
        return nil
    }
    
    private func extractTimeRef(text: String) -> String? {
        let timePatterns: [String: String] = [
            "last night": #"\blast\s+night\b|\bduring\s+(?:the\s+)?night\b|\bovernight\b"#,
            "yesterday": #"\byesterday\b(?!\s+night)"#,
            "yesterday morning": #"\byesterday\s+morning\b"#,
            "yesterday afternoon": #"\byesterday\s+afternoon\b"#,
            "yesterday evening": #"\byesterday\s+evening\b"#,
            "today": #"\btoday\b|\bnow\b|\bcurrently\b|\bthis\s+day\b"#,
            "this morning": #"\bthis\s+morning\b|\bmorning\b"#,
            "this afternoon": #"\bthis\s+afternoon\b|\bafternoon\b"#,
            "this evening": #"\bthis\s+evening\b|\bevening\b"#,
            "this week": #"\bthis\s+week\b|\bcurrent\s+week\b|\bweekly\b"#,
            "last week": #"\blast\s+week\b|\bpast\s+week\b|\bprevious\s+week\b"#,
            "this month": #"\bthis\s+month\b|\bcurrent\s+month\b|\bmonthly\b"#
        ]
        
        for (timeRef, pattern) in timePatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return timeRef
            }
        }
        
        return nil
    }
    
    private func extractQualifier(text: String) -> String? {
        let qualifierPatterns: [String: String] = [
            "minimum": #"\b(?:minimum|min|lowest|least|bottom|smallest)\b"#,
            "maximum": #"\b(?:maximum|max|highest|most|peak|top|largest)\b"#,
            "average": #"\b(?:average|avg|mean|typical|normal)\b"#,
            "total": #"\b(?:total|sum|overall|complete|entire|all)\b"#
        ]
        
        for (qualifier, pattern) in qualifierPatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return qualifier
            }
        }
        
        return nil
    }
    
    private func extractUnit(text: String) -> String? {
        let unitPatterns: [String: String] = [
            "bpm": #"\b(?:bpm|beats?\s+per\s+minute)\b"#,
            "kg": #"\b(?:kg|kilogram|kgs)\b"#,
            "pounds": #"\b(?:pounds?|lbs?|lb)\b"#,
            "km": #"\b(?:km|kilometer|kilometres?)\b"#,
            "miles": #"\b(?:miles?|mi)\b"#,
            "kcal": #"\b(?:kcal|calories?)\b"#,
            "hours": #"\b(?:hours?|hrs?|h)\b"#,
            "minutes": #"\b(?:min|minutes?|mins)\b"#,
            "percent": #"\b(?:percent|%)\b"#,
            "count": #"\bsteps?\b"#
        ]
        
        for (unit, pattern) in unitPatterns {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return unit
            }
        }
        
        // Context-based unit inference
        if text.range(of: #"\b(?:heart\s+rate|pulse|hr)\b"#, options: .regularExpression) != nil {
            return "bpm"
        }
        if text.range(of: #"\b(?:weight|weigh)\b"#, options: .regularExpression) != nil {
            return "kg"
        }
        if text.range(of: #"\bsteps?\b"#, options: .regularExpression) != nil {
            return "count"
        }
        
        return nil
    }
    
    private func extractThreshold(text: String) -> Int? {
        // Look for numbers in context
        let numberPatterns = [
            #"\b(\d+(?:\.\d+)?)\s*(?:bpm|kg|km|miles?|percent|%|hours?|minutes?)\b"#,
            #"\b(?:above|over|exceeds?|higher\s+than)\s+(\d+(?:\.\d+)?)\b"#,
            #"\b(?:below|under|less\s+than|lower\s+than)\s+(\d+(?:\.\d+)?)\b"#
        ]
        
        for pattern in numberPatterns {
            if let match = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[match])
                if let number = extractNumber(from: matchedText) {
                    return Int(number)
                }
            }
        }
        
        // Fallback to any number
        if let match = text.range(of: #"\b(\d+(?:\.\d+)?)\b"#, options: .regularExpression) {
            let matchedText = String(text[match])
            if let number = extractNumber(from: matchedText) {
                return Int(number)
            }
        }
        
        return nil
    }
    
    private func extractTarget(text: String) -> Int? {
        // Look for goal-setting patterns
        let goalPatterns = [
            #"\b(?:goal|target|aim).*?(\d+(?:\.\d+)?)\b"#,
            #"\b(?:set|change|update).*?(?:to|at)\s*(\d+(?:\.\d+)?)\b"#,
            #"\b(\d+(?:\.\d+)?)\s*(?:steps?|kg|km|hours?|minutes?|calories?)\b"#
        ]
        
        for pattern in goalPatterns {
            if let match = text.range(of: pattern, options: .regularExpression) {
                let matchedText = String(text[match])
                if let number = extractNumber(from: matchedText) {
                    return Int(number)
                }
            }
        }
        
        if let match = text.range(of: #"\b(\d+(?:\.\d+)?)\b"#, options: .regularExpression) {
            let matchedText = String(text[match])
            if let number = extractNumber(from: matchedText) {
                return Int(number)
            }
        }
        
        return nil
    }
    
    private func extractValue(text: String, intent: String) -> Any? {
        switch intent {
        case "LogEvent":
            let weightPattern = #"\b(\d+(?:\.\d+)?)\s*(?:kg|pounds?|lbs?)\b"#
            if let match = text.range(of: weightPattern, options: .regularExpression) {
                let matchedText = String(text[match])
                return extractNumber(from: matchedText)
            }
            
        case "TimerStopwatch":
            let timePatterns = [
                #"\b(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\b"#,
                #"\b(\d+)\s*(?:min|minutes?)\b"#,
                #"\b(\d+)\s*(?:hours?|hrs?)\b"#
            ]
            
            for pattern in timePatterns {
                if let match = text.range(of: pattern, options: .regularExpression) {
                    return String(text[match])
                }
            }
            
            if let match = text.range(of: #"\b(\d+(?:\.\d+)?)\b"#, options: .regularExpression) {
                let matchedText = String(text[match])
                if let number = extractNumber(from: matchedText) {
                    return Int(number)
                }
            }
            
        default:
            if let match = text.range(of: #"\b(\d+(?:\.\d+)?)\b"#, options: .regularExpression) {
                let matchedText = String(text[match])
                if let number = extractNumber(from: matchedText) {
                    return Int(number)
                }
            }
        }
        
        return nil
    }
    
    private func extractFeature(text: String) -> String? {
        let features: [String: String] = [
            "do not disturb": #"\b(?:do\s+not\s+disturb|dnd|silent\s+mode)\b"#,
            "AOD": #"\b(?:AOD|always\s+on\s+display|always-on)\b"#,
            "raise to wake": #"\b(?:raise\s+to\s+wake|lift\s+to\s+wake|tap\s+to\s+wake)\b"#,
            "vibration": #"\b(?:vibration|vibrate|haptic)\b"#,
            "brightness": #"\b(?:brightness|screen\s+brightness)\b"#,
            "volume": #"\b(?:volume|sound\s+level)\b"#
        ]
        
        for (feature, pattern) in features {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return feature
            }
        }
        
        return nil
    }
    
    private func extractState(text: String) -> String? {
        if text.range(of: #"\b(?:turn\s+on|enable|activate|switch\s+on|start)\b"#, options: .regularExpression) != nil {
            return "on"
        }
        if text.range(of: #"\b(?:turn\s+off|disable|deactivate|switch\s+off|stop)\b"#, options: .regularExpression) != nil {
            return "off"
        }
        if text.range(of: #"\b(?:increase|up|higher|raise)\b"#, options: .regularExpression) != nil {
            return "increase"
        }
        if text.range(of: #"\b(?:decrease|down|lower|reduce)\b"#, options: .regularExpression) != nil {
            return "decrease"
        }
        
        return nil
    }
    
    private func extractAction(text: String) -> String? {
        let actions: [String: String] = [
            "set": #"\b(?:set|setup|configure)\b"#,
            "start": #"\b(?:start|begin|initiate|launch)\b"#,
            "stop": #"\b(?:stop|end|finish|terminate)\b"#,
            "call": #"\b(?:call|phone|dial)\b"#,
            "message": #"\b(?:message|text|sms|send)\b"#,
            "open": #"\b(?:open|launch|start|show)\b"#,
            "check": #"\b(?:check|verify|examine|look)\b"#,
            "measure": #"\b(?:measure|test|record)\b"#
        ]
        
        for (action, pattern) in actions {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return action
            }
        }
        
        return nil
    }
    
    private func extractTool(text: String) -> String? {
        if text.range(of: #"\b(?:alarm|wake\s+up|wake\s+me)\b"#, options: .regularExpression) != nil {
            return "alarm"
        }
        if text.range(of: #"\b(?:timer|countdown)\b"#, options: .regularExpression) != nil {
            return "timer"
        }
        if text.range(of: #"\b(?:stopwatch|chronometer)\b"#, options: .regularExpression) != nil {
            return "stopwatch"
        }
        
        return nil
    }
    
    private func extractActivityType(text: String) -> String? {
        let activities: [String: String] = [
            "outdoor run": #"\b(?:outdoor\s+)?(?:run|running|jog|jogging)\b"#,
            "indoor cycling": #"\b(?:indoor\s+)?(?:cycling|bike|biking)\b"#,
            "swimming": #"\b(?:swim|swimming|pool)\b"#,
            "yoga": #"\b(?:yoga|meditation|stretch)\b"#,
            "walking": #"\b(?:walk|walking|hike|hiking)\b"#,
            "workout": #"\b(?:workout|exercise|training|gym)\b"#
        ]
        
        for (activity, pattern) in activities {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return activity
            }
        }
        
        return nil
    }
    
    private func extractApp(text: String) -> String? {
        let apps: [String: String] = [
            "weather": #"\b(?:weather|forecast|temperature|rain|snow)\b"#,
            "settings": #"\b(?:settings?|preferences|config)\b"#,
            "health": #"\b(?:health|fitness|medical)\b"#,
            "calendar": #"\b(?:calendar|schedule|appointment)\b"#
        ]
        
        for (app, pattern) in apps {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return app
            }
        }
        
        return nil
    }
    
    private func extractContact(text: String) -> String? {
        let contacts: [String: String] = [
            "mom": #"\b(?:mom|mother|mama|mum)\b"#,
            "dad": #"\b(?:dad|father|papa|pop)\b"#,
            "sister": #"\b(?:sister|sis)\b"#,
            "brother": #"\b(?:brother|bro)\b"#
        ]
        
        for (contact, pattern) in contacts {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return contact
            }
        }
        
        return nil
    }
    
    private func extractLocation(text: String) -> String? {
        let locations: [String: String] = [
            "london": #"\b(?:london|uk|england)\b"#,
            "bangalore": #"\b(?:bangalore|bengaluru|blr)\b"#,
            "mumbai": #"\b(?:mumbai|bombay)\b"#,
            "delhi": #"\b(?:delhi|new\s+delhi)\b"#
        ]
        
        for (location, pattern) in locations {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return location
            }
        }
        
        return "current location"  // Default
    }
    
    private func extractAttribute(text: String) -> String? {
        let attributes: [String: String] = [
            "forecast": #"\b(?:forecast|prediction|outlook)\b"#,
            "temperature": #"\b(?:temperature|temp|hot|cold|warm|cool)\b"#,
            "rain": #"\b(?:rain|raining|shower|umbrella|wet)\b"#,
            "humidity": #"\b(?:humidity|humid|moisture)\b"#,
            "air quality": #"\b(?:air\s+quality|aqi|pollution|smog)\b"#
        ]
        
        for (attr, pattern) in attributes {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return attr
            }
        }
        
        return nil
    }
    
    private func extractType(text: String) -> String? {
        if text.range(of: #"\b(?:above|over|exceed|higher|more\s+than|greater)\b"#, options: .regularExpression) != nil {
            return "high"
        }
        if text.range(of: #"\b(?:below|under|less\s+than|lower|drops?)\b"#, options: .regularExpression) != nil {
            return "low"
        }
        
        return nil
    }
    
    private func extractPeriod(text: String) -> String? {
        let periods: [String: String] = [
            "daily": #"\b(?:daily|every\s+day|each\s+day)\b"#,
            "weekly": #"\b(?:weekly|every\s+week|each\s+week)\b"#,
            "monthly": #"\b(?:monthly|every\s+month|each\s+month)\b"#
        ]
        
        for (period, pattern) in periods {
            if text.range(of: pattern, options: .regularExpression) != nil {
                return period
            }
        }
        
        return nil
    }
    
    private func extractEventType(text: String) -> String? {
        if text.range(of: #"\b(?:weight|weigh|kg|pounds)\b"#, options: .regularExpression) != nil {
            return "weight"
        }
        if text.range(of: #"\b(?:menstrual|period|cycle)\b"#, options: .regularExpression) != nil {
            return "menstrual cycle"
        }
        
        return nil
    }
    
    // MARK: - Helper Methods
    
    private func addContextualSlots(text: String, intent: String, slots: inout [String: Any]) {
        // For QueryPoint intent, try to infer missing metric
        if intent == "QueryPoint" && slots["metric"] == nil {
            if let inferredMetric = inferMetricFromContext(text: text) {
                slots["metric"] = inferredMetric
            }
        }
        
        // Add qualifier if detected but not extracted
        if intent == "QueryPoint" && slots["qualifier"] == nil {
            if let qualifier = extractQualifier(text: text) {
                slots["qualifier"] = qualifier
            }
        }
    }
    
    private func inferMetricFromContext(text: String) -> String? {
        let inferencePatterns: [String: [String]] = [
            "steps": [
                #"\b(?:walk|walked|walking)\b(?!\s+distance)"#,
                #"\bhow\s+much.*(?:walk|walked)\b"#,
                #"\bsteps?\b"#
            ],
            "distance": [
                #"\bhow\s+far\b"#,
                #"\b(?:walk|walked|walking)\s+(?:distance|far)\b"#,
                #"\bdistance.*(?:walk|walked)\b"#,
                #"\bkilometers?\b|\bmiles?\b|\bkm\b"#
            ],
            "heart rate": [
                #"\bheart\s+rate\b|\bheartrate\b|\bpulse\b|\bhr\b|\bbpm\b"#
            ],
            "calories": [
                #"\bcalories?\b|\bkcal\b|\benergy\b|\bburn\b"#
            ]
        ]
        
        for (metric, patterns) in inferencePatterns {
            for pattern in patterns {
                if text.range(of: pattern, options: .regularExpression) != nil {
                    return metric
                }
            }
        }
        
        return nil
    }
    
    private func extractNumber(from text: String) -> Double? {
        let pattern = #"\d+(?:\.\d+)?"#
        if let match = text.range(of: pattern, options: .regularExpression) {
            let numberString = String(text[match])
            return Double(numberString)
        }
        return nil
    }
}